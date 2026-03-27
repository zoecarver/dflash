"""A/B test: call dev_attn vs inline the same steps. See where they diverge."""
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
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd, ztt,
                                dev_norm, dev_attn, _MESH)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        n = dev_norm(h, w["t.0.in_w"], sp, w, d)
        lp = "t.0"

        # Path A: call dev_attn
        o_a = dev_attn(n, w, lp, sl, sp, d)
        result_a = rb(o_a)[:sl, :HIDDEN].float()

        # Path B: inline the exact same code
        normed = n
        q = ttnn.matmul(normed, w[f"{lp}.qw"])
        k = ttnn.matmul(normed, w[f"{lp}.kw"])
        v = ttnn.matmul(normed, w[f"{lp}.vw"])

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

        q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k4.repeat(1, GQA, 1, 1)
        v4 = v4.repeat(1, GQA, 1, 1)

        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); q4p[:, :, :sl] = q4
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); k4p[:, :, :sl] = k4
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); v4p[:, :, :sl] = v4

        attn = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        ah = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :]
        ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)

        o_b = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o_b = ttnn.all_reduce(o_b)
        result_b = rb(o_b)[:sl, :HIDDEN].float()

        print(f"Path A (dev_attn) vs Path B (inline): PCC = {pcc(result_a, result_b):.6f}")
        print(f"MaxDiff: {(result_a - result_b).abs().max().item():.4f}")
        print(f"Path A mean: {result_a.abs().mean().item():.6f}")
        print(f"Path B mean: {result_b.abs().mean().item():.6f}")
        print(f"Path A[0,:5]: {result_a[0,:5]}")
        print(f"Path B[0,:5]: {result_b[0,:5]}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
