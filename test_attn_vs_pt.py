"""Compare dev_attn against pure PyTorch on same input, same weights."""
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
import json

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


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd,
                                dev_norm, dev_attn)
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

        # Use a known input: just the embedding
        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))

        # Device: norm + attn
        h = rep(h_pt, d)
        n = dev_norm(h, w["t.0.in_w"], sp, w, d)
        o_dev = dev_attn(n, w, "t.0", sl, sp, d)
        result_dev = rb(o_dev)[:sl, :HIDDEN].float()

        # Read back the norm output to use as PyTorch input
        norm_out = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)

        # Load raw weights for PyTorch reference
        with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
            idx = json.load(f)
        kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
        def gt(k):
            with safe_open(kf[k], framework="pt") as f:
                return f.get_tensor(k)

        p = "model.layers.0"
        qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
        kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
        vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
        ow = gt(f"{p}.self_attn.o_proj.weight").to(torch.bfloat16)  # (hidden, heads*hdim)
        qnw = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
        knw = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

        # PyTorch: same ops as dev_attn but all on host
        x = norm_out.float()
        q_pt = (x @ qw.float())
        k_pt = (x @ kw.float())
        v_pt = (x @ vw.float())

        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw.float()).view(x.shape)

        q_pt = head_rms_norm(q_pt, qnw, NQH)
        k_pt = head_rms_norm(k_pt, knw, NKVH)

        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
        pos = torch.arange(sl, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        cos_full = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
        sin_full = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

        def rotate_half_flat(x, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
            return torch.cat((-x2, x1), dim=-1).view(x.shape)

        q_pt = q_pt * cos_full.repeat(1, NQH) + rotate_half_flat(q_pt, NQH) * sin_full.repeat(1, NQH)
        k_pt = k_pt * cos_full.repeat(1, NKVH) + rotate_half_flat(k_pt, NKVH) * sin_full.repeat(1, NKVH)

        q4 = q_pt.view(1, sl, NQH, HDIM).transpose(1, 2)
        k4 = k_pt.view(1, sl, NKVH, HDIM).transpose(1, 2)
        v4 = v_pt.view(1, sl, NKVH, HDIM).transpose(1, 2)
        k4 = k4.repeat(1, GQA, 1, 1)
        v4 = v4.repeat(1, GQA, 1, 1)

        attn_out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
        ah = attn_out.transpose(1, 2).contiguous().view(sl, NQH * HDIM)
        result_pt = (ah @ ow.float().T).to(torch.bfloat16).float()

        print(f"dev_attn vs PyTorch (same norm input):")
        print(f"  PCC: {pcc(result_dev, result_pt):.6f}")
        print(f"  MaxDiff: {(result_dev - result_pt).abs().max().item():.4f}")
        print(f"  Dev mean: {result_dev.abs().mean().item():.6f}")
        print(f"  PT  mean: {result_pt.abs().mean().item():.6f}")
        print(f"  Dev[0,:5]: {result_dev[0,:5]}")
        print(f"  PT [0,:5]: {result_pt[0,:5]}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
