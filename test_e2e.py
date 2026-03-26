"""End-to-end test: 2 transformer layers with random weights on single chip.

Validates the full model pipeline: Embedding -> N layers -> Final norm -> LM head.
Uses small dimensions to run quickly.
"""
import math
import torch
import ttnn
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel

TILE = 32
HIDDEN_SIZE = 2048
HIDDEN_TILES = HIDDEN_SIZE // TILE
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
MOE_INTER = 768
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
RMS_NORM_EPS = 1e-6
VOCAB_SIZE = 1024
NUM_LAYERS = 2

rmsnorm_kernel = make_rmsnorm_kernel(dim_tiles=HIDDEN_TILES, eps=RMS_NORM_EPS)


def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pad_to_tile(t, dim=-1):
    size = t.shape[dim]
    pad_size = (TILE - size % TILE) % TILE
    if pad_size == 0:
        return t
    pad_spec = [0] * (2 * len(t.shape))
    pad_spec[-(2 * (dim % len(t.shape)) + 1)] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


def rms_norm_host(x, w, eps=RMS_NORM_EPS):
    xf = x.float()
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((xf / rms) * w.float()).to(torch.bfloat16)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def test_e2e(device, seq_len=32):
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE
    torch.manual_seed(123)

    # Create random weights
    embed_w = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.02
    lm_head_w = torch.randn(HIDDEN_SIZE, VOCAB_SIZE, dtype=torch.bfloat16) * 0.02
    final_norm_w = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16)

    layers = []
    for _ in range(NUM_LAYERS):
        layer = {
            "input_norm_w": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16),
            "post_attn_norm_w": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16),
            "q_w": torch.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01,
            "k_w": torch.randn(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01,
            "v_w": torch.randn(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01,
            "o_w": torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01,
            "q_norm_w": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
            "k_norm_w": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
            "router_w": torch.randn(HIDDEN_SIZE, NUM_EXPERTS, dtype=torch.bfloat16) * 0.01,
            "expert_gate": [torch.randn(HIDDEN_SIZE, MOE_INTER, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)],
            "expert_up": [torch.randn(HIDDEN_SIZE, MOE_INTER, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)],
            "expert_down": [torch.randn(MOE_INTER, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)],
        }
        layers.append(layer)

    # Random input tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (seq_len,))

    # ==== PyTorch reference ====
    hidden_ref = embed_w[input_ids].float()  # (seq_len, HIDDEN_SIZE)

    for li, layer in enumerate(layers):
        # RMSNorm -> Attention -> Residual
        normed = rms_norm_host(hidden_ref.to(torch.bfloat16), layer["input_norm_w"]).float()
        q = (normed @ layer["q_w"].float()).view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = (normed @ layer["k_w"].float()).view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = (normed @ layer["v_w"].float()).view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = rms_norm_host(q.to(torch.bfloat16), layer["q_norm_w"]).float()
        k = rms_norm_host(k.to(torch.bfloat16), layer["k_norm_w"]).float()
        k = k.repeat(1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1)
        v = v.repeat(1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(seq_len, -1)
        o_out = attn_out @ layer["o_w"].float()
        hidden_ref = hidden_ref + o_out

        # RMSNorm -> MoE -> Residual
        normed_moe = rms_norm_host(hidden_ref.to(torch.bfloat16), layer["post_attn_norm_w"]).float()
        router = normed_moe @ layer["router_w"].float()
        scores = torch.softmax(router, dim=-1)
        tw, ti = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)
        moe_out = torch.zeros_like(hidden_ref)
        for tok in range(seq_len):
            for kidx in range(NUM_EXPERTS_PER_TOK):
                eidx = ti[tok, kidx].item()
                w = tw[tok, kidx].item()
                inp = normed_moe[tok:tok+1]
                g = inp @ layer["expert_gate"][eidx].float()
                u = inp @ layer["expert_up"][eidx].float()
                out = (torch.nn.functional.silu(g) * u) @ layer["expert_down"][eidx].float()
                moe_out[tok] += out.squeeze(0) * w
        hidden_ref = hidden_ref + moe_out

    # Final norm + LM head
    hidden_ref = rms_norm_host(hidden_ref.to(torch.bfloat16), final_norm_w).float()
    logits_ref = (hidden_ref @ lm_head_w.float()).to(torch.bfloat16)

    # ==== TT-Lang / TTNN implementation ====
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16)
    scaler_tt = to_dev(scaler, device)
    mean_scale_tt = to_dev(mean_scale, device)

    # Embedding
    hidden_host = embed_w[input_ids].to(torch.bfloat16)
    hidden_padded = pad_to_tile(hidden_host, dim=0)
    if hidden_padded.shape[0] < seq_pad:
        hidden_padded = torch.nn.functional.pad(hidden_padded, (0, 0, 0, seq_pad - hidden_padded.shape[0]))
    hidden_tt = to_dev(hidden_padded, device)

    for li, layer in enumerate(layers):
        # RMSNorm 1
        w_exp = layer["input_norm_w"].unsqueeze(0).expand(seq_pad, -1).contiguous()
        normed_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
        rmsnorm_kernel(hidden_tt, to_dev(w_exp, device), scaler_tt, mean_scale_tt, normed_tt)

        # Attention (QKV matmul on device, rest on host)
        q_tt = ttnn.matmul(normed_tt, to_dev(layer["q_w"], device))
        k_tt = ttnn.matmul(normed_tt, to_dev(layer["k_w"], device))
        v_tt = ttnn.matmul(normed_tt, to_dev(layer["v_w"], device))

        q_h = ttnn.to_torch(q_tt)[:seq_len].float().view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k_h = ttnn.to_torch(k_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v_h = ttnn.to_torch(v_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q_h = rms_norm_host(q_h.to(torch.bfloat16), layer["q_norm_w"]).float()
        k_h = rms_norm_host(k_h.to(torch.bfloat16), layer["k_norm_w"]).float()
        k_h = k_h.repeat(1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1)
        v_h = v_h.repeat(1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        attn = torch.softmax(torch.matmul(q_h, k_h.transpose(-2, -1)) * scale, dim=-1)
        attn_out = torch.matmul(attn, v_h).transpose(1, 2).contiguous().view(seq_len, -1).to(torch.bfloat16)

        attn_padded = pad_to_tile(attn_out, dim=0)
        if attn_padded.shape[0] < seq_pad:
            attn_padded = torch.nn.functional.pad(attn_padded, (0, 0, 0, seq_pad - attn_padded.shape[0]))
        attn_dev = to_dev(attn_padded, device)
        o_proj = ttnn.matmul(attn_dev, to_dev(layer["o_w"], device))

        # Residual 1
        res1 = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
        residual_add_kernel(hidden_tt, o_proj, res1)

        # RMSNorm 2
        w2_exp = layer["post_attn_norm_w"].unsqueeze(0).expand(seq_pad, -1).contiguous()
        normed_moe_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
        rmsnorm_kernel(res1, to_dev(w2_exp, device), scaler_tt, mean_scale_tt, normed_moe_tt)

        # MoE: router on device, expert FFN on device
        router_tt = ttnn.matmul(normed_moe_tt, to_dev(layer["router_w"], device))
        router_h = ttnn.to_torch(router_tt)[:seq_len].float()
        scores = torch.softmax(router_h, dim=-1)
        tw, ti = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)

        normed_moe_h = ttnn.to_torch(normed_moe_tt)[:seq_len].to(torch.bfloat16)

        expert_gate_tt = [to_dev(layer["expert_gate"][i], device) for i in range(NUM_EXPERTS)]
        expert_up_tt = [to_dev(layer["expert_up"][i], device) for i in range(NUM_EXPERTS)]
        expert_down_tt = [to_dev(layer["expert_down"][i], device) for i in range(NUM_EXPERTS)]

        expert_to_tokens = {}
        for tok in range(seq_len):
            for kidx in range(NUM_EXPERTS_PER_TOK):
                eidx = ti[tok, kidx].item()
                w = tw[tok, kidx].item()
                if eidx not in expert_to_tokens:
                    expert_to_tokens[eidx] = []
                expert_to_tokens[eidx].append((tok, w))

        moe_out_h = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
        for eidx, tinfo in expert_to_tokens.items():
            toks = [t[0] for t in tinfo]
            n = len(toks)
            np_ = ((n + TILE - 1) // TILE) * TILE
            inp = torch.nn.functional.pad(normed_moe_h[toks], (0, 0, 0, np_ - n))
            inp_tt = to_dev(inp, device)

            g = ttnn.matmul(inp_tt, expert_gate_tt[eidx])
            u = ttnn.matmul(inp_tt, expert_up_tt[eidx])
            act = to_dev(torch.zeros(np_, MOE_INTER, dtype=torch.bfloat16), device)
            silu_mul_kernel(g, u, act)
            eout = ttnn.matmul(act, expert_down_tt[eidx])
            eout_h = ttnn.to_torch(eout)[:n, :HIDDEN_SIZE].to(torch.bfloat16)

            for i, (tok, w) in enumerate(tinfo):
                moe_out_h[tok] += eout_h[i] * w

        moe_padded = pad_to_tile(moe_out_h, dim=0)
        if moe_padded.shape[0] < seq_pad:
            moe_padded = torch.nn.functional.pad(moe_padded, (0, 0, 0, seq_pad - moe_padded.shape[0]))
        moe_dev = to_dev(moe_padded, device)

        # Residual 2
        hidden_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
        residual_add_kernel(res1, moe_dev, hidden_tt)

        print(f"  Layer {li} done")

    # Final norm
    fn_exp = final_norm_w.unsqueeze(0).expand(seq_pad, -1).contiguous()
    final_normed = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(hidden_tt, to_dev(fn_exp, device), scaler_tt, mean_scale_tt, final_normed)

    # LM head
    lm_head_padded = pad_to_tile(lm_head_w, dim=1)
    logits_tt = ttnn.matmul(final_normed, to_dev(lm_head_padded, device))
    logits_result = ttnn.to_torch(logits_tt)[:seq_len, :VOCAB_SIZE]

    # Compare
    r = logits_result.float().flatten()
    e = logits_ref.float().flatten()
    p = torch.corrcoef(torch.stack([r, e]))[0, 1].item()
    max_err = (logits_result.float() - logits_ref.float()).abs().max().item()

    # Also check argmax agreement
    pred_tt = logits_result.argmax(dim=-1)
    pred_ref = logits_ref.argmax(dim=-1)
    argmax_match = (pred_tt == pred_ref).float().mean().item()

    print(f"\nE2E test ({NUM_LAYERS} layers, seq_len={seq_len}, {NUM_EXPERTS} experts):")
    print(f"  PCC: {p:.6f}")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Argmax agreement: {argmax_match * 100:.1f}%")
    print(f"  {'PASS' if p > 0.95 else 'FAIL'}")
    return p > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    ok = test_e2e(device, seq_len=32)
    ttnn.close_device(device)
    if not ok:
        exit(1)
