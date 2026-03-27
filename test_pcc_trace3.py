"""Compare dev_attn() output vs inline-identical code, find the divergence."""
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
NQH_TP = NQH // N_CHIPS
NKVH_TP = NKVH // N_CHIPS

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
                                dev_norm, dev_attn, dev_add)
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

    # HF reference
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0 = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)
    del model
    import gc; gc.collect()

    print(f"\nsl={sl}, sp={sp}")

    d = open_dev()
    try:
        w = load_weights(d)
        lp = "t.0"

        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))

        # === Run A: dev_attn() as a function call ===
        print("\n=== Run A: dev_attn() function call ===")
        h_a = rep(h_pt, d)
        n_a = dev_norm(h_a, w[f"{lp}.in_w"], sp, w, d)

        # Check norm output
        n_a_host = rb(n_a)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Norm A: {n_a_host[0,:3].float()}")

        attn_a = dev_attn(n_a, w, lp, sl, sp, d)
        attn_a_host = rb(attn_a)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Attn A output: {attn_a_host[0,:5].float()}")

        # === Run B: inline identical code (copied from dev_attn) ===
        print("\n=== Run B: inline code ===")
        h_b = rep(h_pt, d)
        n_b = dev_norm(h_b, w[f"{lp}.in_w"], sp, w, d)

        n_b_host = rb(n_b)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Norm B: {n_b_host[0,:3].float()}")
        print(f"Norm A vs B: PCC={pcc(n_a_host, n_b_host):.6f}")

        # QKV
        q = ttnn.matmul(n_b, w[f"{lp}.qw"])
        k = ttnn.matmul(n_b, w[f"{lp}.kw"])
        v = ttnn.matmul(n_b, w[f"{lp}.vw"])

        qh = rb_dim1(q)[:sl].float()
        kh = rb_dim1(k)[:sl].float()
        vh = rb_dim1(v)[:sl].float()

        # QK-norm
        qnw = w[f"{lp}.qnw"].float()
        knw = w[f"{lp}.knw"].float()

        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw).view(x.shape)

        qh = head_rms_norm(qh, qnw, NQH)
        kh = head_rms_norm(kh, knw, NKVH)

        # RoPE
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

        # SDPA
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
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)
        o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)

        attn_b_host = rb(o)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Attn B output: {attn_b_host[0,:5].float()}")

        # === Compare ===
        print(f"\n=== Comparison ===")
        print(f"A vs B:  PCC={pcc(attn_a_host, attn_b_host):.6f}  MaxDiff={maxdiff(attn_a_host, attn_b_host):.6f}")
        print(f"A vs HF: PCC (expect ~0.627 as before)")

        # Full layer with A
        h_a = dev_add(rep(h_pt, d), rep(_p(attn_a_host), d), sp, d)
        # Full layer with B
        h_b = dev_add(rep(h_pt, d), rep(_p(attn_b_host), d), sp, d)

        # Now do the rest: norm2 + MoE
        from dflash_device import dev_moe

        nm_a = dev_norm(h_a, w[f"{lp}.pa_w"], sp, w, d)
        moe_a = dev_moe(nm_a, w, lp, sl, sp, d)
        h_a = dev_add(h_a, moe_a, sp, d)
        l0_a = rb(h_a)[:sl, :HIDDEN].to(torch.bfloat16)

        nm_b = dev_norm(h_b, w[f"{lp}.pa_w"], sp, w, d)
        moe_b = dev_moe(nm_b, w, lp, sl, sp, d)
        h_b = dev_add(h_b, moe_b, sp, d)
        l0_b = rb(h_b)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"\nFull L0 A vs HF: PCC={pcc(l0_a, hf_l0):.6f}")
        print(f"Full L0 B vs HF: PCC={pcc(l0_b, hf_l0):.6f}")
        print(f"Full L0 A vs B:  PCC={pcc(l0_a, l0_b):.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
