"""Compare host-only Q/K/V at every step vs HF to find the divergence."""
import torch
import torch.nn.functional as F

TILE = 32
HIDDEN = 2048
HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, repeat_kv
    from safetensors import safe_open
    import json

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]

    # HF reference
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    sa = model.model.layers[0].self_attn
    layer0 = model.model.layers[0]

    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)
        hf_emb = hidden[0, :sl].to(torch.bfloat16)

        # HF layer 0 internals
        ln1 = layer0.input_layernorm(hidden)
        hf_ln1 = ln1[0, :sl].to(torch.bfloat16)

        bsz = 1
        hf_q_proj = sa.q_proj(ln1)
        hf_k_proj = sa.k_proj(ln1)
        hf_v_proj = sa.v_proj(ln1)

        hf_q_normed = sa.q_norm(hf_q_proj.view(bsz, sl, NQH, HDIM)).transpose(1, 2)
        hf_k_normed = sa.k_norm(hf_k_proj.view(bsz, sl, NKVH, HDIM)).transpose(1, 2)
        hf_v = hf_v_proj.view(bsz, sl, NKVH, HDIM).transpose(1, 2)

        position_ids = torch.arange(sl).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)
        hf_q_r, hf_k_r = apply_rotary_pos_emb(hf_q_normed, hf_k_normed, cos, sin)

    # Load raw weights
    with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
    def gt(k):
        with safe_open(kf[k], framework="pt") as f:
            return f.get_tensor(k)

    p = "model.layers.0"
    in_w = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
    qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
    kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
    vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
    qnw = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
    knw = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

    # Our host-only computation
    h = hf_emb.clone()

    # RMSNorm
    def rms_norm(x, w):
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((xf / rms) * w.float()).to(torch.bfloat16)

    our_ln1 = rms_norm(h, in_w)
    print(f"Norm1: PCC={pcc(our_ln1, hf_ln1):.6f}")

    # Q proj
    our_q = (our_ln1.float() @ qw.float()).to(torch.bfloat16)
    our_k = (our_ln1.float() @ kw.float()).to(torch.bfloat16)
    our_v = (our_ln1.float() @ vw.float()).to(torch.bfloat16)
    print(f"Q proj: PCC={pcc(our_q, hf_q_proj[0,:sl].to(torch.bfloat16)):.6f}")
    print(f"K proj: PCC={pcc(our_k, hf_k_proj[0,:sl].to(torch.bfloat16)):.6f}")
    print(f"V proj: PCC={pcc(our_v, hf_v_proj[0,:sl].to(torch.bfloat16)):.6f}")

    # QK-norm (our implementation)
    def head_rms_norm(x, nw, n_heads):
        x4 = x.float().view(-1, n_heads, HDIM)
        rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((x4 / rms) * nw.float()).view(x.shape)

    our_qn = head_rms_norm(our_q, qnw, NQH)
    our_kn = head_rms_norm(our_k, knw, NKVH)

    # Compare to HF's QK-norm output
    hf_qn_flat = hf_q_normed.transpose(1, 2).reshape(sl, NQH * HDIM).to(torch.bfloat16)
    hf_kn_flat = hf_k_normed.transpose(1, 2).reshape(sl, NKVH * HDIM).to(torch.bfloat16)
    print(f"QK-norm Q: PCC={pcc(our_qn.to(torch.bfloat16), hf_qn_flat):.6f}")
    print(f"QK-norm K: PCC={pcc(our_kn.to(torch.bfloat16), hf_kn_flat):.6f}")

    # RoPE comparison
    # HF cos/sin
    print(f"\nHF cos shape: {cos.shape}")
    print(f"HF cos[0,0,:5]: {cos[0,0,:5].float()}")

    # Our RoPE
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(sl, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    cos_full = cos_t.repeat(1, 2)[:, :HDIM]
    sin_full = sin_t.repeat(1, 2)[:, :HDIM]

    print(f"Our cos[0,:5]: {cos_full[0,:5]}")
    print(f"Cos match: PCC={pcc(cos_full, cos[0].float()):.6f}")

    def rotate_half_flat(x, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
        return torch.cat((-x2, x1), dim=-1).view(x.shape)

    q_roped = our_qn * cos_full.repeat(1, NQH) + rotate_half_flat(our_qn, NQH) * sin_full.repeat(1, NQH)
    k_roped = our_kn * cos_full.repeat(1, NKVH) + rotate_half_flat(our_kn, NKVH) * sin_full.repeat(1, NKVH)

    hf_qr_flat = hf_q_r.transpose(1, 2).reshape(sl, NQH * HDIM)
    hf_kr_flat = hf_k_r.transpose(1, 2).reshape(sl, NKVH * HDIM)
    print(f"\nRoPE Q: PCC={pcc(q_roped.to(torch.bfloat16), hf_qr_flat.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(q_roped, hf_qr_flat):.6f}")
    print(f"RoPE K: PCC={pcc(k_roped.to(torch.bfloat16), hf_kr_flat.to(torch.bfloat16)):.6f}  MaxDiff={maxdiff(k_roped, hf_kr_flat):.6f}")

    # SDPA comparison
    q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    v4 = our_v.float().view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k4.repeat(1, GQA, 1, 1)
    v4 = v4.repeat(1, GQA, 1, 1)

    our_sdpa = F.scaled_dot_product_attention(
        q4.float(), k4.float(), v4.float(), is_causal=True).to(torch.bfloat16)

    hf_k_exp = repeat_kv(hf_k_r, GQA)
    hf_v_exp = repeat_kv(hf_v, GQA)
    hf_sdpa = F.scaled_dot_product_attention(
        hf_q_r.float(), hf_k_exp.float(), hf_v_exp.float(), is_causal=True).to(torch.bfloat16)

    print(f"\nSDPA: PCC={pcc(our_sdpa, hf_sdpa):.6f}  MaxDiff={maxdiff(our_sdpa, hf_sdpa):.6f}")

    # Check repeat_kv vs repeat
    our_k_exp = k4  # already repeated
    hf_k_exp_bf = hf_k_exp.to(torch.bfloat16)
    print(f"\nK expanded: PCC={pcc(our_k_exp, hf_k_exp_bf):.6f}")
    # Check: what does repeat_kv actually do?
    import inspect
    print(f"\nrepeat_kv source:\n{inspect.getsource(repeat_kv)}")

    # Check if repeat_kv interleaves differently than repeat
    test_k = torch.arange(4*HDIM).float().view(1, 4, 1, HDIM)
    rep_kv = repeat_kv(test_k, 8)
    rep_repeat = test_k.repeat(1, 8, 1, 1)
    print(f"\nrepeat_kv head order: {[rep_kv[0, i, 0, 0].item() for i in range(32)]}")
    print(f"repeat head order:    {[rep_repeat[0, i, 0, 0].item() for i in range(32)]}")


if __name__ == "__main__":
    main()
