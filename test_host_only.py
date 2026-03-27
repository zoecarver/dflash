"""Run layer 0 entirely on HOST mimicking dev_attn logic, compare to HF."""
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
NEXPERTS = 128
TOPK = 8
MOE_INTER = 768

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors import safe_open
    import json

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE
    print(f"sl={sl}, sp={sp}")

    # HF reference
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True, use_cache=False)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0 = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)

    # Load raw weights
    with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
    def gt(k):
        with safe_open(kf[k], framework="pt") as f:
            return f.get_tensor(k)

    p = "model.layers.0"
    in_w = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
    pa_w = gt(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)
    qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
    kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
    vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
    ow = gt(f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
    qnw = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
    knw = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

    del model
    import gc; gc.collect()

    # Host-only layer 0, mimicking device code exactly
    h = hf_emb.clone()  # (sl, HIDDEN)

    # RMSNorm
    def rms_norm(x, w):
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((xf / rms) * w.float()).to(torch.bfloat16)

    n = rms_norm(h, in_w)

    # QKV matmul
    q = (n.float() @ qw.float()).to(torch.bfloat16)
    k = (n.float() @ kw.float()).to(torch.bfloat16)
    v = (n.float() @ vw.float()).to(torch.bfloat16)

    # QK-norm
    def head_rms_norm(x, nw, n_heads):
        x4 = x.float().view(-1, n_heads, HDIM)
        rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((x4 / rms) * nw.float()).view(x.shape)

    qh = head_rms_norm(q, qnw, NQH)
    kh = head_rms_norm(k, knw, NKVH)

    # RoPE
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(sl, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    cos_full = cos_t.repeat(1, 2)[:, :HDIM]
    sin_full = sin_t.repeat(1, 2)[:, :HDIM]

    def rotate_half_flat(x, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
        return torch.cat((-x2, x1), dim=-1).view(x.shape)

    q_roped = qh * cos_full.repeat(1, NQH) + rotate_half_flat(qh, NQH) * sin_full.repeat(1, NQH)
    k_roped = kh * cos_full.repeat(1, NKVH) + rotate_half_flat(kh, NKVH) * sin_full.repeat(1, NKVH)

    # SDPA
    q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    v4 = v.float().view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k4.repeat(1, GQA, 1, 1)
    v4 = v4.repeat(1, GQA, 1, 1)

    sdpa = F.scaled_dot_product_attention(
        q4.float(), k4.float(), v4.float(), is_causal=True).to(torch.bfloat16)

    # O-proj
    ah = sdpa.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
    o = (ah.float() @ ow.float()).to(torch.bfloat16)

    # Residual
    h = (h.float() + o.float()).to(torch.bfloat16)

    # Norm2
    nm = rms_norm(h, pa_w)

    # MoE
    rw = gt(f"{p}.mlp.gate.weight").T.contiguous().to(torch.bfloat16)
    router = (nm.float() @ rw.float()).to(torch.bfloat16)
    scores = F.softmax(router.float(), dim=-1)
    topk_vals, topk_idx = torch.topk(scores, TOPK, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)

    # Compute all experts
    moe_sum = torch.zeros(sl, HIDDEN, dtype=torch.float32)
    for ei in range(NEXPERTS):
        ep = f"{p}.mlp.experts.{ei}"
        gw = gt(f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16)
        uw = gt(f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16)
        dw = gt(f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16)

        gate = torch.nn.functional.silu(nm.float() @ gw.float())
        up = nm.float() @ uw.float()
        act = gate * up
        down = (act @ dw.float())

        # Get routing weight for this expert
        mask = (topk_idx == ei).any(dim=-1)  # (sl,)
        weights_for_expert = torch.zeros(sl, dtype=torch.float32)
        for t in range(sl):
            for k_i in range(TOPK):
                if topk_idx[t, k_i] == ei:
                    weights_for_expert[t] = topk_vals[t, k_i]

        moe_sum += down * weights_for_expert.unsqueeze(-1)

    h_final = (h.float() + moe_sum).to(torch.bfloat16)

    print(f"\n=== Host-only layer 0 vs HF ===")
    print(f"Host L0 vs HF: PCC={pcc(h_final[:sl], hf_l0):.6f}")
    print(f"Host[0,:5]: {h_final[0,:5].float()}")
    print(f"HF  [0,:5]: {hf_l0[0,:5].float()}")

    # Intermediate checks
    print(f"\nHost attn out[0,:5]: {o[0,:5].float()}")


if __name__ == "__main__":
    main()
