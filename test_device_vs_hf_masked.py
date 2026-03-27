"""Compare device attention output vs HF with proper causal mask."""
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
                                rep, rb, rb_dim1, _p, shd,
                                dev_norm, dev_attn)
    import dflash_device
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, repeat_kv

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    print(f"sl={sl}, sp={sp}")

    # HF with proper causal mask
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    sa = model.model.layers[0].self_attn

    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)
        ln1 = model.model.layers[0].input_layernorm(hidden)

        position_ids = torch.arange(sl).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)

        bsz = 1
        q = sa.q_norm(sa.q_proj(ln1).view(bsz, sl, NQH, HDIM)).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(ln1).view(bsz, sl, NKVH, HDIM)).transpose(1, 2)
        v = sa.v_proj(ln1).view(bsz, sl, NKVH, HDIM).transpose(1, 2)

        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)

        # HF reference: with proper causal mask
        k_exp = repeat_kv(k_r, GQA)
        v_exp = repeat_kv(v, GQA)
        scale = sa.scaling
        print(f"scale = {scale}")

        attn_w = torch.matmul(q_r, k_exp.transpose(2, 3)) * scale
        min_val = torch.finfo(torch.bfloat16).min
        causal = torch.triu(torch.full((sl, sl), min_val, dtype=torch.bfloat16), diagonal=1)
        attn_w = attn_w + causal.unsqueeze(0).unsqueeze(0)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        hf_sdpa = torch.matmul(attn_w, v_exp)  # (1, NQH, sl, HDIM)
        hf_oproj = sa.o_proj(hf_sdpa.transpose(1, 2).reshape(bsz, sl, NQH * HDIM))

        # Also compute with PyTorch SDPA (which uses is_causal=True)
        pt_sdpa = F.scaled_dot_product_attention(
            q_r.float(), k_exp.float(), v_exp.float(), is_causal=True).to(torch.bfloat16)

        print(f"\nHF eager SDPA vs PT SDPA (is_causal): PCC={pcc(hf_sdpa, pt_sdpa):.6f}  MaxDiff={maxdiff(hf_sdpa, pt_sdpa):.6f}")

        # Save Q/K/V for device comparison
        hf_q_r = q_r
        hf_k_r = k_r
        hf_v = v

    hf_emb = hidden[0, :sl].to(torch.bfloat16)

    del model
    import gc; gc.collect()

    # Device
    print("\nOpening device...")
    d = open_dev()
    try:
        w = load_weights(d)
        lp = "t.0"

        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)

        # QKV on device
        q_dev = ttnn.matmul(n, w[f"{lp}.qw"])
        k_dev = ttnn.matmul(n, w[f"{lp}.kw"])
        v_dev = ttnn.matmul(n, w[f"{lp}.vw"])

        qh = rb_dim1(q_dev)[:sl].float()
        kh = rb_dim1(k_dev)[:sl].float()
        vh = rb_dim1(v_dev)[:sl].float()

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

        # Compare post-RoPE Q/K with HF
        dev_qr = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2)
        dev_kr = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2)
        print(f"\nQ post-RoPE: PCC={pcc(dev_qr, hf_q_r):.6f}")
        print(f"K post-RoPE: PCC={pcc(dev_kr, hf_k_r):.6f}")

        # Run SDPA on HOST with is_causal=True using our Q/K/V
        q4 = dev_qr.to(torch.bfloat16)
        k4 = dev_kr.repeat(1, GQA, 1, 1).to(torch.bfloat16)
        v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).repeat(1, GQA, 1, 1).to(torch.bfloat16)

        host_sdpa = F.scaled_dot_product_attention(
            q4.float(), k4.float(), v4.float(), is_causal=True).to(torch.bfloat16)
        print(f"\nHost SDPA (our Q/K/V, is_causal) vs HF eager masked: PCC={pcc(host_sdpa, hf_sdpa):.6f}  MaxDiff={maxdiff(host_sdpa, hf_sdpa):.6f}")

        # Run SDPA on DEVICE with is_causal=True, padded to sp
        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        q4p[:, :, :sl] = q4
        k4p[:, :, :sl] = k4
        v4p[:, :, :sl] = v4

        attn = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        dev_sdpa = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :].to(torch.bfloat16)
        print(f"Dev SDPA vs HF eager masked: PCC={pcc(dev_sdpa, hf_sdpa):.6f}  MaxDiff={maxdiff(dev_sdpa, hf_sdpa):.6f}")
        print(f"Dev SDPA vs Host SDPA:       PCC={pcc(dev_sdpa, host_sdpa):.6f}")

        # O-proj
        ah = dev_sdpa.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_tt = shd(ahp, d, dim=1)
        o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)
        dev_o = rb(o)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"\nDev O-proj vs HF O-proj (masked): PCC={pcc(dev_o, hf_oproj[0,:sl].to(torch.bfloat16)):.6f}")

        # What about using HF's exact Q/K/V in device SDPA?
        hf_q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        hf_k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        hf_v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
        hf_k_exp = hf_k_r.repeat(1, GQA, 1, 1).to(torch.bfloat16)
        hf_v_exp = hf_v.repeat(1, GQA, 1, 1).to(torch.bfloat16)
        hf_q4p[:, :, :sl] = hf_q_r.to(torch.bfloat16)
        hf_k4p[:, :, :sl] = hf_k_exp
        hf_v4p[:, :, :sl] = hf_v_exp

        hf_attn_dev = ttnn.transformer.scaled_dot_product_attention(
            rep(hf_q4p, d), rep(hf_k4p, d), rep(hf_v4p, d), is_causal=True)
        hf_sdpa_dev = rb(hf_attn_dev).view(1, NQH, -1, HDIM)[:, :, :sl, :].to(torch.bfloat16)
        print(f"Dev SDPA (HF Q/K/V) vs HF eager: PCC={pcc(hf_sdpa_dev, hf_sdpa):.6f}  MaxDiff={maxdiff(hf_sdpa_dev, hf_sdpa):.6f}")

        # Key: host SDPA with HF's Q/K/V
        host_hf_sdpa = F.scaled_dot_product_attention(
            hf_q_r.float(), hf_k_exp.float(), hf_v_exp.float(), is_causal=True).to(torch.bfloat16)
        print(f"Host SDPA (HF Q/K/V) vs HF eager: PCC={pcc(host_hf_sdpa, hf_sdpa):.6f}  MaxDiff={maxdiff(host_hf_sdpa, hf_sdpa):.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
