"""Per-op PCC trace: run full pipeline, compare every intermediate to HF."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7
NEXPERTS = 128
TOPK = 8
N_CHIPS = 4
EPC = NEXPERTS // N_CHIPS
MOE_INTER = 768

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd,
                                dev_norm, dev_attn, dev_add, dev_moe)
    import dflash_device
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    # ---- HF reference with hooks to capture every intermediate ----
    print("Loading HF model (eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    hf_intermediates = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                hf_intermediates[name] = out[0].detach()
            else:
                hf_intermediates[name] = out.detach()
        return hook

    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(make_hook("l0.norm1"))
    layer0.self_attn.q_proj.register_forward_hook(make_hook("l0.q_proj"))
    layer0.self_attn.k_proj.register_forward_hook(make_hook("l0.k_proj"))
    layer0.self_attn.v_proj.register_forward_hook(make_hook("l0.v_proj"))
    layer0.self_attn.q_norm.register_forward_hook(make_hook("l0.q_norm"))
    layer0.self_attn.k_norm.register_forward_hook(make_hook("l0.k_norm"))
    layer0.self_attn.register_forward_hook(make_hook("l0.attn_out"))
    layer0.self_attn.o_proj.register_forward_hook(make_hook("l0.o_proj"))
    layer0.post_attention_layernorm.register_forward_hook(make_hook("l0.norm2"))
    layer0.mlp.gate.register_forward_hook(make_hook("l0.router"))
    layer0.register_forward_hook(make_hook("l0.final"))

    print("Running HF forward...")
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0_out = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)

    # Extract HF intermediates
    hf_norm1 = hf_intermediates["l0.norm1"][0, :sl].to(torch.bfloat16)
    hf_q = hf_intermediates["l0.q_proj"][0, :sl].to(torch.bfloat16)
    hf_k = hf_intermediates["l0.k_proj"][0, :sl].to(torch.bfloat16)
    hf_v = hf_intermediates["l0.v_proj"][0, :sl].to(torch.bfloat16)
    hf_qn = hf_intermediates["l0.q_norm"][0, :sl].to(torch.bfloat16)
    hf_kn = hf_intermediates["l0.k_norm"][0, :sl].to(torch.bfloat16)
    hf_o_proj = hf_intermediates["l0.o_proj"][0, :sl].to(torch.bfloat16)
    hf_norm2 = hf_intermediates["l0.norm2"][0, :sl].to(torch.bfloat16)
    hf_router = hf_intermediates["l0.router"][0, :sl].to(torch.bfloat16)

    # Get the residual after attention (emb + attn_output)
    # HF: hidden_states = hidden_states + self_attn output (which is o_proj output)
    hf_resid1 = (hf_emb.float() + hf_o_proj.float()).to(torch.bfloat16)

    del model
    import gc; gc.collect()

    # ---- Device pipeline ----
    print("\nOpening device...")
    d = open_dev()
    try:
        w = load_weights(d)
        lp = "t.0"

        # Start from HF's exact embedding
        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        print("\n" + "=" * 70)
        print("PER-OP PCC TRACE: Layer 0")
        print("=" * 70)

        # Step 1: RMS Norm
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        dev_norm1 = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"1. RMSNorm1:   PCC={pcc(dev_norm1, hf_norm1):.6f}  MaxDiff={maxdiff(dev_norm1, hf_norm1):.6f}")

        # Step 2: QKV projections
        q = ttnn.matmul(n, w[f"{lp}.qw"])
        k = ttnn.matmul(n, w[f"{lp}.kw"])
        v = ttnn.matmul(n, w[f"{lp}.vw"])

        dev_q = rb_dim1(q)[:sl].to(torch.bfloat16)
        dev_k = rb_dim1(k)[:sl].to(torch.bfloat16)
        dev_v = rb_dim1(v)[:sl].to(torch.bfloat16)
        print(f"2a. Q proj:    PCC={pcc(dev_q, hf_q):.6f}  MaxDiff={maxdiff(dev_q, hf_q):.6f}")
        print(f"2b. K proj:    PCC={pcc(dev_k, hf_k):.6f}  MaxDiff={maxdiff(dev_k, hf_k):.6f}")
        print(f"2c. V proj:    PCC={pcc(dev_v, hf_v):.6f}  MaxDiff={maxdiff(dev_v, hf_v):.6f}")

        # Step 3: QK-norm (on host, same as dev_attn)
        qh = rb_dim1(q)[:sl].float()
        kh = rb_dim1(k)[:sl].float()
        vh = rb_dim1(v)[:sl].float()

        qnw = w[f"{lp}.qnw"].float()
        knw = w[f"{lp}.knw"].float()

        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw).view(x.shape)

        qh = head_rms_norm(qh, qnw, NQH)
        kh = head_rms_norm(kh, knw, NKVH)

        # HF q_norm output is (batch, seq, heads, head_dim), we need to reshape for comparison
        dev_qn = qh.to(torch.bfloat16)
        dev_kn = kh.to(torch.bfloat16)
        # HF stores these as (batch, seq, heads, head_dim) -> flatten to (seq, heads*head_dim)
        hf_qn_flat = hf_qn.view(sl, -1)
        hf_kn_flat = hf_kn.view(sl, -1)
        print(f"3a. QK-norm Q: PCC={pcc(dev_qn, hf_qn_flat):.6f}  MaxDiff={maxdiff(dev_qn, hf_qn_flat):.6f}")
        print(f"3b. QK-norm K: PCC={pcc(dev_kn, hf_kn_flat):.6f}  MaxDiff={maxdiff(dev_kn, hf_kn_flat):.6f}")

        # Step 4: RoPE
        def rotate_half_flat(x, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
            return torch.cat((-x2, x1), dim=-1).view(x.shape)

        cos_h = rb(w["rope_cos"])[:sl].float()
        sin_h = rb(w["rope_sin"])[:sl].float()
        cos_full = cos_h.repeat(1, 2)[:, :HDIM]
        sin_full = sin_h.repeat(1, 2)[:, :HDIM]

        q_roped = qh * cos_full.repeat(1, NQH) + rotate_half_flat(qh, NQH) * sin_full.repeat(1, NQH)
        k_roped = kh * cos_full.repeat(1, NKVH) + rotate_half_flat(kh, NKVH) * sin_full.repeat(1, NKVH)
        print(f"4a. RoPE Q:    PCC={pcc(q_roped.to(torch.bfloat16), hf_qn_flat):.6f}  (vs pre-RoPE HF qnorm, expect low)")

        # Step 5: SDPA
        q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k4.repeat(1, GQA, 1, 1)
        v4 = v4.repeat(1, GQA, 1, 1)

        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        q4p[:, :, :sl] = q4
        k4p[:, :, :sl] = k4
        v4p[:, :, :sl] = v4

        attn = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        ah = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :]
        ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)

        # Step 6: O projection + all_reduce
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)
        o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)

        dev_o = rb(o)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"6. O proj+AR:  PCC={pcc(dev_o, hf_o_proj):.6f}  MaxDiff={maxdiff(dev_o, hf_o_proj):.6f}")

        # Step 7: Residual add
        h = dev_add(rep(h_pt, d), o, sp, d)
        dev_resid1 = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"7. Resid1:     PCC={pcc(dev_resid1, hf_resid1):.6f}  MaxDiff={maxdiff(dev_resid1, hf_resid1):.6f}")

        # BUT WAIT: dev_attn already does the full attention pipeline internally.
        # Let's also run the ACTUAL dev_attn call and compare:
        print("\n--- Now running actual dev_attn() call for comparison ---")
        h2 = rep(h_pt, d)
        n2 = dev_norm(h2, w[f"{lp}.in_w"], sp, w, d)
        attn_out2 = dev_attn(n2, w, lp, sl, sp, d)
        dev_attn_out = rb(attn_out2)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"   dev_attn:   PCC={pcc(dev_attn_out, hf_o_proj):.6f}  MaxDiff={maxdiff(dev_attn_out, hf_o_proj):.6f}")

        h2 = dev_add(h2, attn_out2, sp, d)
        dev_resid1b = rb(h2)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"   Resid1(fn): PCC={pcc(dev_resid1b, hf_resid1):.6f}  MaxDiff={maxdiff(dev_resid1b, hf_resid1):.6f}")

        # Step 8: Norm2
        nm = dev_norm(h2, w[f"{lp}.pa_w"], sp, w, d)
        dev_norm2 = rb(nm)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"8. RMSNorm2:   PCC={pcc(dev_norm2, hf_norm2):.6f}  MaxDiff={maxdiff(dev_norm2, hf_norm2):.6f}")

        # Step 9: MoE
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        dev_moe_out = rb(moe_out)[:sl, :HIDDEN].to(torch.bfloat16)

        # Step 10: Final residual
        h2 = dev_add(h2, moe_out, sp, d)
        dev_l0 = rb(h2)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"10. Layer0 out: PCC={pcc(dev_l0, hf_l0_out):.6f}  MaxDiff={maxdiff(dev_l0, hf_l0_out):.6f}")

        # Sanity: what does HF compute for residual1?
        print(f"\n--- Sanity checks ---")
        print(f"HF emb[0,:5]:    {hf_emb[0,:5].float()}")
        print(f"HF o_proj[0,:5]: {hf_o_proj[0,:5].float()}")
        print(f"HF resid1[0,:5]: {hf_resid1[0,:5].float()}")
        print(f"Dev resid1[0,:5]:{dev_resid1b[0,:5].float()}")
        print(f"HF norm2[0,:5]:  {hf_norm2[0,:5].float()}")
        print(f"Dev norm2[0,:5]: {dev_norm2[0,:5].float()}")
        print(f"HF L0 out[0,:5]: {hf_l0_out[0,:5].float()}")
        print(f"Dev L0 out[0,:5]:{dev_l0[0,:5].float()}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
