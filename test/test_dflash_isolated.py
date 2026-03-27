"""Isolated test for DFlash draft model: random weights, compare device vs PyTorch.

Single device (no TP). Uses stacked Q heads + TTNN matmul + TT-Lang softmax
for cross-attention with implicit GQA. Tests 1 layer first, then full 8 layers.
"""
import torch
import torch.nn.functional as F
import ttnn
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel
from rope import make_rope_kernel
from softmax import make_softmax_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE
HDIM = 128
HDIM_TILES = HDIM // TILE
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7

DLAYERS = 8
DINTER = 6144
BSIZE = 16
SP = ((BSIZE + TILE - 1) // TILE) * TILE  # 32


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def torch_rmsnorm(x, weight, eps=EPS):
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight


def torch_rope(x, cos, sin):
    x1, x2 = x[..., :HDIM//2], x[..., HDIM//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def torch_layer_fwd(h, ctx, weights, lp, cos_q, sin_q, cos_kv, sin_kv):
    """Single layer forward in PyTorch."""
    kv_len = ctx.shape[0] + h.shape[0]

    normed = torch_rmsnorm(h, weights[f"{lp}.in_w"].float())
    q = normed @ weights[f"{lp}.qw"].float()
    kv_in = torch.cat([ctx, normed], dim=0)
    k = kv_in @ weights[f"{lp}.kw"].float()
    v = kv_in @ weights[f"{lp}.vw"].float()

    bsize = h.shape[0]
    q = q.view(bsize, NQH, HDIM)
    k = k.view(kv_len, NKVH, HDIM)
    v = v.view(kv_len, NKVH, HDIM)

    for head in range(NQH):
        q[:, head] = torch_rmsnorm(q[:, head], weights[f"{lp}.qnw"].float())
    for head in range(NKVH):
        k[:, head] = torch_rmsnorm(k[:, head], weights[f"{lp}.knw"].float())
    for head in range(NQH):
        q[:, head] = torch_rope(q[:, head], cos_q, sin_q)
    for head in range(NKVH):
        k[:, head] = torch_rope(k[:, head], cos_kv, sin_kv)

    # GQA attention with stacked Q (matches device path)
    scale = 1.0 / (HDIM ** 0.5)
    attn_out = torch.zeros(bsize, NQH, HDIM)
    for qh in range(NQH):
        kvh = qh // GQA
        scores = (q[:, qh] @ k[:, kvh].T) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out[:, qh] = probs @ v[:, kvh]
    attn_out = attn_out.view(bsize, NQH * HDIM)

    o = attn_out @ weights[f"{lp}.ow"].float()
    h = h + o

    normed2 = torch_rmsnorm(h, weights[f"{lp}.pa_w"].float())
    gate = normed2 @ weights[f"{lp}.gw"].float()
    up = normed2 @ weights[f"{lp}.uw"].float()
    down = (F.silu(gate) * up) @ weights[f"{lp}.fc2"].float()
    h = h + down
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _p(t):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def to_dev(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def dev_layer_fwd(h, ctx_dev, w, lp, norm_k, q_rope_k, k_rope_k, softmax_k,
                  sc, ms, kv_sp, d):
    """Single layer forward on device using stacked Q + softmax."""
    scale = 1.0 / (HDIM ** 0.5)

    # Input LayerNorm
    normed = to_dev(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"{lp}.in_w_tt"], sc, ms, normed)

    # Q projection
    q = ttnn.matmul(normed, w[f"{lp}.qw"])

    # KV concat + projection
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)
    k = ttnn.matmul(kv_in, w[f"{lp}.kw"])
    v = ttnn.matmul(kv_in, w[f"{lp}.vw"])

    # Per-head QK-norm: reshape to (n_tokens * n_heads, HDIM), rms_norm, reshape back
    q_flat = ttnn.reshape(q, (SP * NQH, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=w[f"{lp}.qnw_dev"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=w[f"{lp}.knw_dev"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))

    # RoPE
    q_roped = to_dev(torch.zeros(SP, NQH * HDIM), d)
    k_roped = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    q_rope_k(q_normed, w["rope_cos_q"], w["rope_sin_q"], q_roped)
    k_rope_k(k_normed, w["rope_cos_kv"], w["rope_sin_kv"], k_roped)

    # Cross-attention: stacked Q heads + batched matmul + TT-Lang softmax
    # Separate into per-head 4D tensors
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH, HDIM)), 1, 2)

    # Group Q heads by KV group for GQA: (1, NQH, SP, HDIM) -> (1, NKVH, GQA*SP, HDIM)
    q_grouped = ttnn.reshape(q4, (1, NKVH, GQA * SP, HDIM))

    # Batched Q @ K^T: (1, NKVH, GQA*SP, HDIM) @ (1, NKVH, HDIM, kv_sp)
    k_t = ttnn.transpose(k4, -2, -1)
    scores = ttnn.matmul(q_grouped, k_t)
    scores = ttnn.multiply(scores, scale)

    # Flatten for TT-Lang softmax: (NKVH*GQA*SP, kv_sp) = (1024, kv_sp)
    total_q_rows = NKVH * GQA * SP
    scores_flat = ttnn.reshape(scores, (total_q_rows, kv_sp))
    probs_flat = to_dev(torch.zeros(total_q_rows, kv_sp), d)
    softmax_k(scores_flat, sc, probs_flat)

    # Reshape back for batched probs @ V: (1, NKVH, GQA*SP, kv_sp)
    probs_4d = ttnn.reshape(probs_flat, (1, NKVH, GQA * SP, kv_sp))
    attn_4d = ttnn.matmul(probs_4d, v4)

    # Un-group: (1, NKVH, GQA*SP, HDIM) -> (1, NQH, SP, HDIM) -> (SP, NQH*HDIM)
    attn_heads = ttnn.reshape(attn_4d, (1, NQH, SP, HDIM))
    attn_flat = ttnn.reshape(ttnn.transpose(attn_heads, 1, 2), (SP, NQH * HDIM))

    # O projection
    o = ttnn.matmul(attn_flat, w[f"{lp}.ow"])

    # Residual
    add_out = to_dev(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, o, add_out)
    h = add_out

    # Post-attention LayerNorm
    normed2 = to_dev(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"{lp}.pa_w_tt"], sc, ms, normed2)

    # Dense MLP
    gate = ttnn.matmul(normed2, w[f"{lp}.gw"])
    up = ttnn.matmul(normed2, w[f"{lp}.uw"])
    act = ttnn.zeros_like(gate)
    silu_mul_kernel(gate, up, act)
    down = ttnn.matmul(act, w[f"{lp}.fc2"])

    # Residual
    add_out2 = to_dev(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, down, add_out2)
    return add_out2


def main():
    torch.manual_seed(42)

    ctx_len = 64
    kv_len = ctx_len + BSIZE
    kv_sp = _tile_pad(kv_len)
    total_q_rows = NKVH * GQA * SP  # 1024

    print(f"Config: BSIZE={BSIZE}, ctx={ctx_len}, kv={kv_len}, kv_sp={kv_sp}")
    print(f"Attention: {NKVH} KV groups, {GQA} Q heads/group, "
          f"scores ({total_q_rows}, {kv_sp})")
    print(f"Softmax: {total_q_rows // TILE} row tiles, {kv_sp // TILE} col tiles\n")

    # Random weights
    weights = {}
    weights["d.fn_w"] = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
    for li in range(DLAYERS):
        lp = f"d.{li}"
        weights[f"{lp}.in_w"] = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
        weights[f"{lp}.pa_w"] = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
        weights[f"{lp}.qw"] = torch.randn(HIDDEN, NQH * HDIM).to(torch.bfloat16) * 0.02
        weights[f"{lp}.kw"] = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
        weights[f"{lp}.vw"] = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
        weights[f"{lp}.ow"] = torch.randn(NQH * HDIM, HIDDEN).to(torch.bfloat16) * 0.02
        weights[f"{lp}.qnw"] = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0
        weights[f"{lp}.knw"] = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0
        weights[f"{lp}.gw"] = torch.randn(HIDDEN, DINTER).to(torch.bfloat16) * 0.02
        weights[f"{lp}.uw"] = torch.randn(HIDDEN, DINTER).to(torch.bfloat16) * 0.02
        weights[f"{lp}.fc2"] = torch.randn(DINTER, HIDDEN).to(torch.bfloat16) * 0.02

    # Random inputs
    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    ctx_bf = torch.randn(ctx_len, HIDDEN).to(torch.bfloat16) * 0.1

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    cos_q = torch.cos(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    sin_q = torch.sin(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
    sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))

    # --- PyTorch reference: layer 0 ---
    print("PyTorch reference layer 0...")
    ref_h = torch_layer_fwd(noise_bf.float(), ctx_bf.float(), weights, "d.0",
                            cos_q, sin_q, cos_kv, sin_kv)
    print(f"  ref range: [{ref_h.min():.4f}, {ref_h.max():.4f}]")

    # --- Device forward ---
    print("\nDevice forward...")
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
        q_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
        k_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)
        softmax_k = make_softmax_kernel(total_q_rows // TILE, kv_sp // TILE)

        # Upload weights
        w = {}
        w["d.fn_w_tt"] = to_dev(weights["d.fn_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
        for li in range(DLAYERS):
            lp = f"d.{li}"
            w[f"{lp}.in_w_tt"] = to_dev(weights[f"{lp}.in_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.pa_w_tt"] = to_dev(weights[f"{lp}.pa_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.qw"] = to_dev(weights[f"{lp}.qw"], d)
            w[f"{lp}.kw"] = to_dev(weights[f"{lp}.kw"], d)
            w[f"{lp}.vw"] = to_dev(weights[f"{lp}.vw"], d)
            w[f"{lp}.ow"] = to_dev(weights[f"{lp}.ow"], d)
            w[f"{lp}.qnw_dev"] = to_dev(weights[f"{lp}.qnw"].unsqueeze(0).contiguous(), d)
            w[f"{lp}.knw_dev"] = to_dev(weights[f"{lp}.knw"].unsqueeze(0).contiguous(), d)
            w[f"{lp}.gw"] = to_dev(weights[f"{lp}.gw"], d)
            w[f"{lp}.uw"] = to_dev(weights[f"{lp}.uw"], d)
            w[f"{lp}.fc2"] = to_dev(weights[f"{lp}.fc2"], d)

        # RoPE tables (full-width for TT-Lang kernel)
        max_seq = max(SP, kv_sp)
        pos = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
        w["rope_cos_q"] = to_dev(cos_full[:SP], d)
        w["rope_sin_q"] = to_dev(sin_adj[:SP], d)
        w["rope_cos_kv"] = to_dev(cos_full[:kv_sp], d)
        w["rope_sin_kv"] = to_dev(sin_adj[:kv_sp], d)

        sc = to_dev(torch.ones(TILE, TILE), d)
        ms = to_dev(torch.full((TILE, TILE), 1.0 / HIDDEN), d)

        # Layer 0
        print("  Layer 0...", end=" ", flush=True)
        h = to_dev(noise_bf, d)
        ctx_dev = to_dev(ctx_bf, d)
        h = dev_layer_fwd(h, ctx_dev, w, "d.0", norm_k, q_rope, k_rope,
                          softmax_k, sc, ms, kv_sp, d)
        tt_h = ttnn.to_torch(h).float()[:BSIZE, :HIDDEN]
        p = pcc(ref_h, tt_h)
        print(f"PCC={p:.6f} {'PASS' if p > 0.95 else 'FAIL'}")

        # Full 8 layers (correctness)
        print("\n  Full 8-layer forward...")
        h = to_dev(noise_bf, d)
        ref_h_full = noise_bf.float()
        for li in range(DLAYERS):
            lp = f"d.{li}"
            print(f"    Layer {li}...", end=" ", flush=True)
            h = dev_layer_fwd(h, ctx_dev, w, lp, norm_k, q_rope, k_rope,
                              softmax_k, sc, ms, kv_sp, d)
            ref_h_full = torch_layer_fwd(ref_h_full, ctx_bf.float(), weights, lp,
                                         cos_q, sin_q, cos_kv, sin_kv)
            tt_h = ttnn.to_torch(h).float()[:BSIZE, :HIDDEN]
            p = pcc(ref_h_full, tt_h)
            print(f"PCC={p:.6f} {'OK' if p > 0.70 else 'BAD'}")

        # Final norm
        final = to_dev(torch.zeros(SP, HIDDEN), d)
        norm_k(h, w["d.fn_w_tt"], sc, ms, final)
        tt_out = ttnn.to_torch(final).float()[:BSIZE, :HIDDEN]
        ref_final = torch_rmsnorm(ref_h_full, weights["d.fn_w"].float())
        p = pcc(ref_final, tt_out)
        print(f"\n  Final PCC: {p:.6f} {'PASS' if p > 0.70 else 'FAIL'}")

        # Performance: warmup + timed runs
        import time
        n_warmup = 3
        n_timed = 10
        print(f"\n  Performance ({n_warmup} warmup, {n_timed} timed)...")

        def run_8_layers():
            h = to_dev(noise_bf, d)
            for li in range(DLAYERS):
                h = dev_layer_fwd(h, ctx_dev, w, f"d.{li}", norm_k, q_rope,
                                  k_rope, softmax_k, sc, ms, kv_sp, d)
            norm_k(h, w["d.fn_w_tt"], sc, ms, final)
            ttnn.synchronize_device(d)

        for _ in range(n_warmup):
            run_8_layers()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            run_8_layers()
        elapsed = time.perf_counter() - t0
        per_fwd = elapsed / n_timed * 1000
        print(f"    {per_fwd:.1f} ms/forward ({n_timed} runs)")
        print(f"    {per_fwd / DLAYERS:.1f} ms/layer")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
