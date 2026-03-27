"""Narrow down: compare RoPE, SDPA, O-proj between device and HF."""
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
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd)
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

    # HF reference: capture every intermediate via manual forward
    print("Loading HF model (eager)...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    layer0 = model.model.layers[0]
    sa = layer0.self_attn

    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)  # (1, sl, HIDDEN)
        ln_out = layer0.input_layernorm(hidden)

        bsz, q_len = 1, sl
        hidden_shape = (bsz, q_len, NKVH, -1)

        # QKV + QK-norm (matches HF forward exactly)
        q_states = sa.q_norm(sa.q_proj(ln_out).view(bsz, q_len, NQH, HDIM)).transpose(1, 2)
        k_states = sa.k_norm(sa.k_proj(ln_out).view(bsz, q_len, NKVH, HDIM)).transpose(1, 2)
        v_states = sa.v_proj(ln_out).view(bsz, q_len, NKVH, HDIM).transpose(1, 2)

        # RoPE (get position embeddings the same way HF model does)
        position_ids = torch.arange(q_len).unsqueeze(0)
        # The model passes position_embeddings from rotary_emb in the model forward
        cos, sin = model.model.rotary_emb(v_states, position_ids)

        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        q_roped_hf, k_roped_hf = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # SDPA (eager)
        k_exp = k_roped_hf.repeat(1, GQA, 1, 1)
        v_exp = v_states.repeat(1, GQA, 1, 1)
        scale = sa.scaling  # 1/sqrt(head_dim)
        attn_w = torch.matmul(q_roped_hf, k_exp.transpose(2, 3)) * scale
        causal = torch.triu(torch.full((q_len, q_len), float('-inf')), diagonal=1)
        attn_w = attn_w + causal.unsqueeze(0).unsqueeze(0)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        hf_sdpa_out = torch.matmul(attn_w, v_exp)  # (1, NQH, sl, HDIM)

        # O-proj
        hf_sdpa_flat = hf_sdpa_out.transpose(1, 2).reshape(bsz, q_len, NQH * HDIM)
        hf_oproj = sa.o_proj(hf_sdpa_flat)  # (1, sl, HIDDEN)

        # Full forward for comparison
        hf_out = model(ids, output_hidden_states=True)
        hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
        hf_l0_out = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)

    print(f"HF manual o_proj vs full forward: sanity check")
    print(f"  scaling = {sa.scaling}")

    del model
    import gc; gc.collect()

    # ---- Device pipeline ----
    print("\nOpening device...")
    d = open_dev()
    try:
        w = load_weights(d)
        lp = "t.0"

        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)
        n = ttnn.rms_norm(h, weight=rep(w[f"{lp}.in_w"].unsqueeze(0).contiguous(), d), epsilon=EPS)

        # QKV on device
        q = ttnn.matmul(n, w[f"{lp}.qw"])
        k = ttnn.matmul(n, w[f"{lp}.kw"])
        v = ttnn.matmul(n, w[f"{lp}.vw"])

        qh = rb_dim1(q)[:sl].float()
        kh = rb_dim1(k)[:sl].float()
        vh = rb_dim1(v)[:sl].float()

        # QK-norm on host
        qnw = w[f"{lp}.qnw"].float()
        knw = w[f"{lp}.knw"].float()

        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw).view(x.shape)

        qh = head_rms_norm(qh, qnw, NQH)
        kh = head_rms_norm(kh, knw, NKVH)

        # Compare QK-norm output (4D)
        dev_q_4d = qh.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        dev_k_4d = kh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        print(f"\n=== Pre-RoPE (after QK-norm) ===")
        print(f"Q pre-RoPE:  PCC={pcc(dev_q_4d, q_states.to(torch.bfloat16)):.6f}")
        print(f"K pre-RoPE:  PCC={pcc(dev_k_4d, k_states.to(torch.bfloat16)):.6f}")

        # RoPE on host (our implementation)
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

        dev_qr_4d = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        dev_kr_4d = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)

        print(f"\n=== Post-RoPE ===")
        print(f"Q post-RoPE: PCC={pcc(dev_qr_4d, q_roped_hf.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(dev_qr_4d, q_roped_hf):.6f}")
        print(f"K post-RoPE: PCC={pcc(dev_kr_4d, k_roped_hf.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(dev_kr_4d, k_roped_hf):.6f}")

        # Also print HF cos/sin vs ours
        print(f"\nHF cos shape: {cos.shape}, our cos shape: {cos_full.shape}")
        print(f"HF cos[0,:5]: {cos[0,0,:5].float()}")
        print(f"Our cos[0,:5]: {cos_full[0,:5].float()}")
        print(f"HF sin[0,:5]: {sin[0,0,:5].float()}")
        print(f"Our sin[0,:5]: {sin_full[0,:5].float()}")

        # SDPA comparison
        q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k4.repeat(1, GQA, 1, 1)
        v4 = v4.repeat(1, GQA, 1, 1)

        # Host SDPA
        host_sdpa = F.scaled_dot_product_attention(
            q4.float(), k4.float(), v4.float(), is_causal=True).to(torch.bfloat16)
        print(f"\n=== SDPA ===")
        print(f"Host SDPA vs HF: PCC={pcc(host_sdpa, hf_sdpa_out.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(host_sdpa, hf_sdpa_out):.6f}")

        # Device SDPA
        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        q4p[:, :, :sl] = q4
        k4p[:, :, :sl] = k4
        v4p[:, :, :sl] = v4

        attn = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        dev_sdpa = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :].to(torch.bfloat16)
        print(f"Dev SDPA vs HF:  PCC={pcc(dev_sdpa, hf_sdpa_out.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(dev_sdpa, hf_sdpa_out):.6f}")
        print(f"Dev SDPA vs Host:PCC={pcc(dev_sdpa, host_sdpa):.6f}")

        # O-proj
        ah = dev_sdpa.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)
        o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)
        dev_o = rb(o)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"\n=== O-proj ===")
        print(f"Dev O vs HF:   PCC={pcc(dev_o, hf_oproj[0,:sl].to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(dev_o, hf_oproj[0,:sl]):.6f}")

        # What if we feed HF's exact SDPA output into our O-proj?
        ah_hf = hf_sdpa_out.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp_hf = _p(ah_hf)
        if ahp_hf.shape[0] < sp:
            ahp_hf = F.pad(ahp_hf, (0, 0, 0, sp - ahp_hf.shape[0]))
        attn_tt_hf = shd(ahp_hf, d, dim=1)
        o_hf = ttnn.matmul(attn_tt_hf, w[f"{lp}.ow"])
        o_hf = ttnn.all_reduce(o_hf)
        dev_o_hf = rb(o_hf)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"HF_SDPA->DevO: PCC={pcc(dev_o_hf, hf_oproj[0,:sl].to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(dev_o_hf, hf_oproj[0,:sl]):.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
