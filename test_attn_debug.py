"""Debug attention step by step: compare QKV, QK-norm, RoPE, SDPA against HF."""
import time
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


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd, ztt,
                                dev_norm, _MESH)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE
    print(f"Prompt: {sl} tokens, padded to {sp}")

    # HF reference
    print("\nLoading HF model...")
    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()

    # Get HF attention intermediates
    layer0_attn = model.model.layers[0].self_attn
    hf_intermediates = {}

    orig_forward = layer0_attn.forward

    def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer0_attn.head_dim)

        # QKV
        q_raw = layer0_attn.q_proj(hidden_states)
        k_raw = layer0_attn.k_proj(hidden_states)
        v_raw = layer0_attn.v_proj(hidden_states)
        hf_intermediates["q_raw"] = q_raw.detach().clone()
        hf_intermediates["k_raw"] = k_raw.detach().clone()
        hf_intermediates["v_raw"] = v_raw.detach().clone()

        # QK-norm
        q_normed = layer0_attn.q_norm(q_raw.view(hidden_shape))
        k_normed = layer0_attn.k_norm(k_raw.view(hidden_shape))
        hf_intermediates["q_normed"] = q_normed.detach().clone()  # (1, sl, heads, hdim)
        hf_intermediates["k_normed"] = k_normed.detach().clone()

        query_states = q_normed.transpose(1, 2)  # (1, heads, sl, hdim)
        key_states = k_normed.transpose(1, 2)
        value_states = v_raw.view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        hf_intermediates["rope_cos"] = cos.detach().clone()
        hf_intermediates["rope_sin"] = sin.detach().clone()

        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        hf_intermediates["q_roped"] = query_states.detach().clone()  # (1, heads, sl, hdim)
        hf_intermediates["k_roped"] = key_states.detach().clone()

        # Now call the rest normally
        return orig_forward(hidden_states, position_embeddings, attention_mask,
                           past_key_values, **kwargs)

    layer0_attn.forward = patched_forward

    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)

    layer0_attn.forward = orig_forward

    print(f"HF intermediates captured: {list(hf_intermediates.keys())}")
    for k, v in hf_intermediates.items():
        print(f"  {k}: {v.shape}")

    del model
    import gc; gc.collect()

    # Device forward
    print("\nLoading device weights...")
    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))

        h = rep(h_pt, d)
        lp = "t.0"

        # RMSNorm
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)

        # QKV matmul
        q = ttnn.matmul(n, w[f"{lp}.qw"])
        k = ttnn.matmul(n, w[f"{lp}.kw"])
        v = ttnn.matmul(n, w[f"{lp}.vw"])

        qh = rb_dim1(q)[:sl].float()
        kh = rb_dim1(k)[:sl].float()
        vh = rb_dim1(v)[:sl].float()

        print(f"\n{'Step':<30} {'PCC':>10} {'MaxDiff':>10}")
        print("-" * 55)

        # Compare Q raw
        hf_q = hf_intermediates["q_raw"][0, :sl].float()
        print(f"{'Q projection':<30} {pcc(qh, hf_q):>10.6f} {(qh-hf_q).abs().max().item():>10.4f}")

        # Compare K raw
        hf_k = hf_intermediates["k_raw"][0, :sl].float()
        print(f"{'K projection':<30} {pcc(kh, hf_k):>10.6f} {(kh-hf_k).abs().max().item():>10.4f}")

        # Compare V raw
        hf_v = hf_intermediates["v_raw"][0, :sl].float()
        print(f"{'V projection':<30} {pcc(vh, hf_v):>10.6f} {(vh-hf_v).abs().max().item():>10.4f}")

        # QK-norm (our implementation)
        qnw = w[f"{lp}.qnw"].float()
        knw = w[f"{lp}.knw"].float()

        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw).view(x.shape)

        qh_normed = head_rms_norm(qh, qnw, NQH)
        kh_normed = head_rms_norm(kh, knw, NKVH)

        # HF q_normed is (1, sl, heads, hdim) -> flatten to (sl, heads*hdim)
        hf_qn = hf_intermediates["q_normed"][0].view(sl, -1).float()
        hf_kn = hf_intermediates["k_normed"][0].view(sl, -1).float()
        print(f"{'Q after QK-norm':<30} {pcc(qh_normed, hf_qn):>10.6f} {(qh_normed-hf_qn).abs().max().item():>10.4f}")
        print(f"{'K after QK-norm':<30} {pcc(kh_normed, hf_kn):>10.6f} {(kh_normed-hf_kn).abs().max().item():>10.4f}")

        # RoPE cos/sin comparison
        cos_h = rb(w["rope_cos"])[:sl].float()
        sin_h = rb(w["rope_sin"])[:sl].float()
        cos_full = cos_h.repeat(1, 2)[:, :HDIM]
        sin_full = sin_h.repeat(1, 2)[:, :HDIM]

        hf_cos = hf_intermediates["rope_cos"][0, :sl].float()  # (sl, hdim)
        hf_sin = hf_intermediates["rope_sin"][0, :sl].float()
        print(f"{'RoPE cos':<30} {pcc(cos_full, hf_cos):>10.6f} {(cos_full-hf_cos).abs().max().item():>10.4f}")
        print(f"{'RoPE sin':<30} {pcc(sin_full, hf_sin):>10.6f} {(sin_full-hf_sin).abs().max().item():>10.4f}")

        # Apply RoPE (our way)
        def rotate_half_flat(x, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
            return torch.cat((-x2, x1), dim=-1).view(x.shape)

        q_roped = qh_normed * cos_full.repeat(1, NQH) + rotate_half_flat(qh_normed, NQH) * sin_full.repeat(1, NQH)
        k_roped = kh_normed * cos_full.repeat(1, NKVH) + rotate_half_flat(kh_normed, NKVH) * sin_full.repeat(1, NKVH)

        # HF q_roped is (1, heads, sl, hdim) -> (sl, heads*hdim)
        hf_qr = hf_intermediates["q_roped"][0].transpose(0, 1).contiguous().view(sl, -1).float()
        hf_kr = hf_intermediates["k_roped"][0].transpose(0, 1).contiguous().view(sl, -1).float()
        print(f"{'Q after RoPE':<30} {pcc(q_roped, hf_qr):>10.6f} {(q_roped-hf_qr).abs().max().item():>10.4f}")
        print(f"{'K after RoPE':<30} {pcc(k_roped, hf_kr):>10.6f} {(k_roped-hf_kr).abs().max().item():>10.4f}")

        # ---- SDPA comparison ----
        # Our SDPA path
        q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
        k4 = k4.repeat(1, GQA, 1, 1)
        v4 = v4.repeat(1, GQA, 1, 1)

        # PyTorch SDPA reference
        pt_sdpa = F.scaled_dot_product_attention(
            q4.float(), k4.float(), v4.float(), is_causal=True)
        print(f"{'PyTorch SDPA shape':<30} {str(pt_sdpa.shape):>20}")

        # Pad and run on device
        q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); q4p[:, :, :sl] = q4
        k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); k4p[:, :, :sl] = k4
        v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); v4p[:, :, :sl] = v4

        attn_tt = ttnn.transformer.scaled_dot_product_attention(
            rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

        # Read back SDPA result
        attn_raw = rb(attn_tt)
        print(f"{'SDPA rb() shape':<30} {str(attn_raw.shape):>20}")

        attn_4d = attn_raw.view(1, NQH, -1, HDIM)[:, :, :sl, :]
        dev_sdpa = attn_4d.float()
        print(f"{'SDPA (device vs PyTorch)':<30} {pcc(dev_sdpa, pt_sdpa):>10.6f} {(dev_sdpa-pt_sdpa).abs().max().item():>10.4f}")

        # O-proj
        ah = attn_4d.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ahp = _p(ah)
        if ahp.shape[0] < sp:
            ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
        attn_shd = shd(ahp, d, dim=1)
        o = ttnn.matmul(attn_shd, w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)
        dev_attn_full = rb(o)[:sl, :HIDDEN].float()

        # HF full attention output (from hidden_states comparison)
        hf_hs = hf_out.hidden_states
        hf_emb = hf_hs[0][0, :sl].float()
        hf_after_l0 = hf_hs[1][0, :sl].float()
        hf_attn_out = (hf_after_l0 - hf_emb)  # residual: attn_out = layer_out - input
        # Actually this includes MoE too. Let me just compare the o_proj output
        # by computing it from pt_sdpa
        ow = w[f"{lp}.ow"]  # on device, sharded
        # Instead, just report what we have
        print(f"{'O-proj output shape':<30} {str(dev_attn_full.shape):>20}")

        # Compare SDPA output more carefully: check individual heads
        print(f"\nPer-head SDPA PCC (first 4 heads):")
        for hi in range(min(4, NQH)):
            dev_h = dev_sdpa[0, hi, :sl].float()
            pt_h = pt_sdpa[0, hi, :sl].float()
            print(f"  head {hi}: {pcc(dev_h, pt_h):.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
