"""Test a single Qwen3 MoE transformer block with random weights.

Validates the full pipeline: RMSNorm -> GQA Attention -> Residual -> RMSNorm -> MoE -> Residual
on a single chip with small dimensions.
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
MOE_INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 8        # reduced for test
NUM_EXPERTS_PER_TOK = 2  # reduced for test
RMS_NORM_EPS = 1e-6

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


def rms_norm_ref(x, w, eps=RMS_NORM_EPS):
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x.float() / rms) * w.float()).to(torch.bfloat16)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def test_single_block(device, seq_len=32):
    """Test a full transformer block with random weights."""
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE
    torch.manual_seed(42)

    # Random input
    hidden = torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)

    # Random weights
    input_norm_w = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16).abs()
    post_attn_norm_w = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16).abs()

    q_w = torch.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01
    k_w = torch.randn(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01
    v_w = torch.randn(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.01
    o_w = torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01

    q_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    k_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16)

    router_w = torch.randn(HIDDEN_SIZE, NUM_EXPERTS, dtype=torch.bfloat16) * 0.01
    expert_gate_w = [torch.randn(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]
    expert_up_w = [torch.randn(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]
    expert_down_w = [torch.randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]

    # -------------------------------------------------------
    # PyTorch reference
    # -------------------------------------------------------
    h_ref = hidden.float()

    # Pre-attention RMSNorm
    normed_ref = rms_norm_ref(hidden, input_norm_w).float()

    # Attention (simplified: no RoPE, no causal mask for this test)
    q = (normed_ref @ q_w.float()).view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = (normed_ref @ k_w.float()).view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = (normed_ref @ v_w.float()).view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # QK norm
    q = rms_norm_ref(q.to(torch.bfloat16), q_norm_w).float()
    k = rms_norm_ref(k.to(torch.bfloat16), k_norm_w).float()

    # GQA expand
    num_groups = NUM_Q_HEADS // NUM_KV_HEADS
    k = k.repeat(1, num_groups, 1, 1)
    v = v.repeat(1, num_groups, 1, 1)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(1, 2).contiguous().view(seq_len, -1)

    o_out = attn_out @ o_w.float()
    h_after_attn = h_ref + o_out

    # Pre-MoE RMSNorm
    normed_moe_ref = rms_norm_ref(h_after_attn.to(torch.bfloat16), post_attn_norm_w).float()

    # MoE
    router_logits = normed_moe_ref @ router_w.float()
    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    moe_out = torch.zeros(seq_len, HIDDEN_SIZE)
    for tok in range(seq_len):
        for k_idx in range(NUM_EXPERTS_PER_TOK):
            eidx = topk_indices[tok, k_idx].item()
            w = topk_weights[tok, k_idx].item()
            inp = normed_moe_ref[tok:tok+1]
            gate = inp @ expert_gate_w[eidx].float()
            up = inp @ expert_up_w[eidx].float()
            activated = torch.nn.functional.silu(gate) * up
            out = activated @ expert_down_w[eidx].float()
            moe_out[tok] += (out.squeeze(0) * w)

    h_final_ref = (h_after_attn + moe_out).to(torch.bfloat16)

    # -------------------------------------------------------
    # TT-Lang / TTNN implementation
    # -------------------------------------------------------
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16)

    scaler_tt = to_dev(scaler, device)
    mean_scale_tt = to_dev(mean_scale, device)

    # Pad and load input
    hidden_padded = pad_to_tile(hidden, dim=0)
    if hidden_padded.shape[0] < seq_pad:
        hidden_padded = torch.nn.functional.pad(hidden_padded, (0, 0, 0, seq_pad - hidden_padded.shape[0]))
    hidden_tt = to_dev(hidden_padded, device)

    # RMSNorm 1
    w_expanded = input_norm_w.unsqueeze(0).expand(seq_pad, -1).contiguous()
    w_tt = to_dev(w_expanded, device)
    normed_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(hidden_tt, w_tt, scaler_tt, mean_scale_tt, normed_tt)

    # Attention QKV (TTNN matmul)
    q_w_tt = to_dev(q_w, device)
    k_w_tt = to_dev(k_w, device)
    v_w_tt = to_dev(v_w, device)
    o_w_tt = to_dev(o_w, device)

    q_tt = ttnn.matmul(normed_tt, q_w_tt)
    k_tt = ttnn.matmul(normed_tt, k_w_tt)
    v_tt = ttnn.matmul(normed_tt, v_w_tt)

    # Readback for attention (host-side for now)
    q_host = ttnn.to_torch(q_tt)[:seq_len].float().view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k_host = ttnn.to_torch(k_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_host = ttnn.to_torch(v_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    q_host = rms_norm_ref(q_host.to(torch.bfloat16), q_norm_w).float()
    k_host = rms_norm_ref(k_host.to(torch.bfloat16), k_norm_w).float()

    k_host = k_host.repeat(1, num_groups, 1, 1)
    v_host = v_host.repeat(1, num_groups, 1, 1)

    attn_w = torch.matmul(q_host, k_host.transpose(-2, -1)) * scale
    attn_w = torch.softmax(attn_w, dim=-1)
    attn_o = torch.matmul(attn_w, v_host)
    attn_o = attn_o.transpose(1, 2).contiguous().view(seq_len, -1).to(torch.bfloat16)

    attn_o_padded = pad_to_tile(attn_o, dim=0)
    if attn_o_padded.shape[0] < seq_pad:
        attn_o_padded = torch.nn.functional.pad(attn_o_padded, (0, 0, 0, seq_pad - attn_o_padded.shape[0]))
    attn_o_tt = to_dev(attn_o_padded, device)

    o_proj_tt = ttnn.matmul(attn_o_tt, o_w_tt)

    # Residual 1
    h_after_attn_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    residual_add_kernel(hidden_tt, o_proj_tt, h_after_attn_tt)

    # RMSNorm 2
    w2_expanded = post_attn_norm_w.unsqueeze(0).expand(seq_pad, -1).contiguous()
    w2_tt = to_dev(w2_expanded, device)
    normed_moe_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(h_after_attn_tt, w2_tt, scaler_tt, mean_scale_tt, normed_moe_tt)

    # MoE (router on device, expert computation on host for now)
    router_w_tt = to_dev(router_w, device)
    router_logits_tt = ttnn.matmul(normed_moe_tt, router_w_tt)
    router_host = ttnn.to_torch(router_logits_tt)[:seq_len].float()

    scores_tt = torch.softmax(router_host, dim=-1)
    topk_w_tt, topk_i_tt = torch.topk(scores_tt, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w_tt = topk_w_tt / topk_w_tt.sum(dim=-1, keepdim=True)

    normed_moe_host = ttnn.to_torch(normed_moe_tt)[:seq_len].to(torch.bfloat16)

    moe_out_tt = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    for tok in range(seq_len):
        for k_idx in range(NUM_EXPERTS_PER_TOK):
            eidx = topk_i_tt[tok, k_idx].item()
            w = topk_w_tt[tok, k_idx].item()
            inp = normed_moe_host[tok:tok+1].float()
            gate = inp @ expert_gate_w[eidx].float()
            up = inp @ expert_up_w[eidx].float()
            activated = torch.nn.functional.silu(gate) * up
            out = activated @ expert_down_w[eidx].float()
            moe_out_tt[tok] += (out.squeeze(0) * w).to(torch.bfloat16)

    moe_out_padded = pad_to_tile(moe_out_tt, dim=0)
    if moe_out_padded.shape[0] < seq_pad:
        moe_out_padded = torch.nn.functional.pad(moe_out_padded, (0, 0, 0, seq_pad - moe_out_padded.shape[0]))
    moe_out_dev = to_dev(moe_out_padded, device)

    # Residual 2
    h_final_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    residual_add_kernel(h_after_attn_tt, moe_out_dev, h_final_tt)

    result = ttnn.to_torch(h_final_tt)[:seq_len, :HIDDEN_SIZE]

    # Compare
    r_flat = result.float().flatten()
    e_flat = h_final_ref.float().flatten()
    p = torch.corrcoef(torch.stack([r_flat, e_flat]))[0, 1].item()
    max_err = (result.float() - h_final_ref.float()).abs().max().item()

    print(f"Single block test (seq_len={seq_len}):")
    print(f"  PCC: {p:.6f}")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  {'PASS' if p > 0.98 else 'FAIL'}")
    return p > 0.98


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    ok = test_single_block(device, seq_len=32)
    ttnn.close_device(device)
    if not ok:
        exit(1)
