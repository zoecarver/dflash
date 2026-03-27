"""Isolate O-proj bug: run SDPA on device, read back, do O-proj on host vs device."""
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open

TILE = 32
HIDDEN = 2048
HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7
N_CHIPS = 4
VOCAB = 151936

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd, ztt,
                                dev_norm, _MESH)
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

        # QKV
        q = ttnn.matmul(n, w[f"{lp}.qw"])
        k = ttnn.matmul(n, w[f"{lp}.kw"])
        v = ttnn.matmul(n, w[f"{lp}.vw"])

        qh = rb_dim1(q)[:sl].float()
        kh = rb_dim1(k)[:sl].float()
        vh = rb_dim1(v)[:sl].float()

        # QK-norm + RoPE (same as dev_attn)
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

        # SDPA on host (reference)
        pt_sdpa = F.scaled_dot_product_attention(
            q4.float(), k4.float(), v4.float(), is_causal=True)

        # SDPA on device
        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); q4p[:, :, :sl] = q4
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); k4p[:, :, :sl] = k4
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); v4p[:, :, :sl] = v4
        dev_sdpa_tt = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        dev_sdpa = rb(dev_sdpa_tt)
        print(f"SDPA rb shape: {dev_sdpa.shape}")
        dev_sdpa_4d = dev_sdpa.view(1, NQH, -1, HDIM)[:, :, :sl, :]
        print(f"SDPA PCC (dev vs pt): {pcc(dev_sdpa_4d, pt_sdpa):.6f}")

        # O-proj: host reference
        # Load raw O weight
        import json
        with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
            idx = json.load(f)
        kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
        with safe_open(kf["model.layers.0.self_attn.o_proj.weight"], framework="pt") as f:
            ow_raw = f.get_tensor("model.layers.0.self_attn.o_proj.weight").to(torch.bfloat16)

        # HF does: attn_output.reshape(batch, seq, -1) @ o_proj.weight.T
        # o_proj.weight is (hidden, heads*hdim) = (2048, 4096)
        # attn_output is (batch, seq, heads*hdim)
        ah_host = pt_sdpa.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        o_host = (ah_host.float() @ ow_raw.float().T).to(torch.bfloat16)
        print(f"O-proj host shape: {o_host.shape}")

        # O-proj: device path (same as dev_attn)
        ah = dev_sdpa_4d.squeeze(0).transpose(0, 1).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)
        o_dev = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o_dev = ttnn.all_reduce(o_dev)
        o_dev_pt = rb(o_dev)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"\n{'Step':<30} {'PCC':>10} {'MaxDiff':>10}")
        print("-" * 55)
        print(f"{'O-proj (host ref)':<30} -- reference --")
        print(f"{'O-proj (device)':<30} {pcc(o_dev_pt, o_host):>10.6f} {(o_dev_pt.float()-o_host.float()).abs().max().item():>10.4f}")

        # Also: what if we do O-proj on host using device SDPA output?
        ah_from_dev = dev_sdpa_4d.squeeze(0).transpose(0, 1).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        o_hybrid = (ah_from_dev.float() @ ow_raw.float().T).to(torch.bfloat16)
        print(f"{'O-proj (host, dev SDPA)':<30} {pcc(o_hybrid, o_host):>10.6f} {(o_hybrid.float()-o_host.float()).abs().max().item():>10.4f}")

        # Key check: does sharding+matmul+all_reduce give same result as host matmul?
        print(f"{'O-proj (dev vs hybrid)':<30} {pcc(o_dev_pt, o_hybrid):>10.6f} {(o_dev_pt.float()-o_hybrid.float()).abs().max().item():>10.4f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
