"""DFlash draft model test on 4-chip mesh with TP sharding.

Uses TT-Lang softmax kernel for attention (stacked Q heads, implicit GQA).
Column-parallel QKV/gate/up, row-parallel O/down with all_reduce.
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

N_CHIPS = 4
NQH_TP = NQH // N_CHIPS    # 8
NKVH_TP = NKVH // N_CHIPS   # 1
Q_TP = NQH_TP * HDIM        # 1024
KV_TP = NKVH_TP * HDIM      # 128

DLAYERS = 8
DINTER = 6144
DINTER_TP = DINTER // N_CHIPS  # 1536
BSIZE = 16
SP = ((BSIZE + TILE - 1) // TILE) * TILE


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


# ---------------------------------------------------------------------------
# PyTorch reference (same as single-device test)
# ---------------------------------------------------------------------------
def torch_rmsnorm(x, weight, eps=EPS):
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight


def torch_rope(x, cos, sin):
    x1, x2 = x[..., :HDIM//2], x[..., HDIM//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def torch_layer_fwd(h, ctx, weights, lp, cos_q, sin_q, cos_kv, sin_kv):
    kv_len = ctx.shape[0] + h.shape[0]
    bsize = h.shape[0]

    normed = torch_rmsnorm(h, weights[f"{lp}.in_w"].float())
    q = normed @ weights[f"{lp}.qw"].float()
    kv_in = torch.cat([ctx, normed], dim=0)
    k = kv_in @ weights[f"{lp}.kw"].float()
    v = kv_in @ weights[f"{lp}.vw"].float()

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


def rep(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(d))


def shd(t, d, dim):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(d, dim=dim))


def rb(t, d):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(d, dim=0))[:t.shape[0]]


def rb_dim1(t, d):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(d, dim=1))


# ---------------------------------------------------------------------------
# Device layer forward (4-chip mesh, TP sharded)
# ---------------------------------------------------------------------------
def dev_layer_fwd(h, ctx_dev, w, lp, norm_k, q_rope_k, k_rope_k, softmax_k,
                  sc, ms, kv_sp, d):
    """Single layer forward on 4-chip mesh with TP sharding."""
    # Input LayerNorm (replicated -> replicated)
    normed = rep(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"{lp}.in_w_tt"], sc, ms, normed)

    # QKV projections (replicated @ col-sharded = col-sharded)
    q = ttnn.matmul(normed, w[f"{lp}.qw"])   # (SP, Q_TP) per chip
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)  # (kv_sp, HIDDEN) replicated
    k = ttnn.matmul(kv_in, w[f"{lp}.kw"])    # (kv_sp, KV_TP) per chip
    v = ttnn.matmul(kv_in, w[f"{lp}.vw"])    # (kv_sp, KV_TP) per chip

    # Per-head QK-norm via reshape + ttnn.rms_norm
    # Q: (SP, Q_TP=NQH_TP*HDIM) -> (SP*NQH_TP, HDIM) per chip
    q_flat = ttnn.reshape(q, (SP * NQH_TP, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH_TP, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=w[f"{lp}.qnw_dev"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=w[f"{lp}.knw_dev"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH_TP * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH_TP * HDIM))

    # RoPE (per chip, NQH_TP Q heads, NKVH_TP KV heads)
    # Output must be sharded to preserve per-chip data for subsequent TTNN ops
    q_roped = shd(torch.zeros(SP, Q_TP * N_CHIPS), d, dim=1)
    k_roped = shd(torch.zeros(kv_sp, KV_TP * N_CHIPS), d, dim=1)
    q_rope_k(q_normed, w["rope_cos_q"], w["rope_sin_q"], q_roped)
    k_rope_k(k_normed, w["rope_cos_kv"], w["rope_sin_kv"], k_roped)

    # Cross-attention via SDPA (handles GQA natively)
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH_TP, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH_TP, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH_TP, HDIM)), 1, 2)
    attn = ttnn.transformer.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    attn_flat = ttnn.reshape(ttnn.transpose(attn, 1, 2), (SP, NQH_TP * HDIM))

    # O projection (col-sharded @ row-sharded = partial sum, then all_reduce)
    o = ttnn.matmul(attn_flat, w[f"{lp}.ow"])
    o = ttnn.all_reduce(o)

    # Residual
    add_out = rep(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, o, add_out)
    h = add_out

    # Post-attention LayerNorm
    normed2 = rep(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"{lp}.pa_w_tt"], sc, ms, normed2)

    # Dense MLP (column-parallel gate/up, row-parallel down)
    gate = ttnn.matmul(normed2, w[f"{lp}.gw"])
    up = ttnn.matmul(normed2, w[f"{lp}.uw"])
    act = shd(torch.zeros(SP, DINTER), d, dim=1)
    silu_mul_kernel(gate, up, act)
    down = ttnn.matmul(act, w[f"{lp}.fc2"])
    down = ttnn.all_reduce(down)

    # Residual
    add_out2 = rep(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, down, add_out2)
    return add_out2


def main():
    torch.manual_seed(42)

    ctx_len = 64
    kv_len = ctx_len + BSIZE
    kv_sp = _tile_pad(kv_len)
    kv_tiles = kv_sp // TILE
    q_stacked_rows = NQH_TP * SP

    print(f"Config: BSIZE={BSIZE}, ctx={ctx_len}, kv={kv_len}, kv_sp={kv_sp}")
    print(f"Per chip: NQH_TP={NQH_TP}, NKVH_TP={NKVH_TP}, Q_TP={Q_TP}, KV_TP={KV_TP}")
    print(f"Softmax: {q_stacked_rows//TILE} row tiles, {kv_tiles} col tiles\n")

    # Random weights (full, unsharded)
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

    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    ctx_bf = torch.randn(ctx_len, HIDDEN).to(torch.bfloat16) * 0.1

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    cos_q = torch.cos(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    sin_q = torch.sin(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
    sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))

    # Open 4-chip mesh
    print("Opening 4-chip mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    try:
        # Kernel instances
        norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
        q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP)
        k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP)
        softmax_k = make_softmax_kernel(q_stacked_rows // TILE, kv_tiles)

        # Upload weights (sharded for TP)
        w = {}
        w["d.fn_w_tt"] = rep(weights["d.fn_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
        for li in range(DLAYERS):
            lp = f"d.{li}"
            w[f"{lp}.in_w_tt"] = rep(weights[f"{lp}.in_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.pa_w_tt"] = rep(weights[f"{lp}.pa_w"].unsqueeze(0).expand(TILE, -1).contiguous(), d)
            # Column-parallel: shard along dim=1
            w[f"{lp}.qw"] = shd(weights[f"{lp}.qw"], d, dim=1)
            w[f"{lp}.kw"] = shd(weights[f"{lp}.kw"], d, dim=1)
            w[f"{lp}.vw"] = shd(weights[f"{lp}.vw"], d, dim=1)
            # Row-parallel: shard along dim=0
            w[f"{lp}.ow"] = shd(weights[f"{lp}.ow"], d, dim=0)
            # QK-norm weights (replicated, (1, HDIM))
            w[f"{lp}.qnw_dev"] = rep(weights[f"{lp}.qnw"].unsqueeze(0).contiguous(), d)
            w[f"{lp}.knw_dev"] = rep(weights[f"{lp}.knw"].unsqueeze(0).contiguous(), d)
            # MLP: column-parallel gate/up, row-parallel down
            w[f"{lp}.gw"] = shd(weights[f"{lp}.gw"], d, dim=1)
            w[f"{lp}.uw"] = shd(weights[f"{lp}.uw"], d, dim=1)
            w[f"{lp}.fc2"] = shd(weights[f"{lp}.fc2"], d, dim=0)

        # RoPE tables (replicated, full-width for TT-Lang kernel)
        max_seq = max(SP, kv_sp)
        pos = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
        w["rope_cos_q"] = rep(cos_full[:SP], d)
        w["rope_sin_q"] = rep(sin_adj[:SP], d)
        w["rope_cos_kv"] = rep(cos_full[:kv_sp], d)
        w["rope_sin_kv"] = rep(sin_adj[:kv_sp], d)

        # Support tensors
        sc = rep(torch.ones(TILE, TILE), d)
        ms = rep(torch.full((TILE, TILE), 1.0 / HIDDEN), d)

        # Run all 8 layers
        print("\nRunning 8-layer forward on mesh...")
        h = rep(noise_bf, d)
        ctx_dev = rep(ctx_bf, d)
        ref_h = noise_bf.float()

        for li in range(DLAYERS):
            lp = f"d.{li}"
            print(f"  Layer {li}...", end=" ", flush=True)

            h = dev_layer_fwd(h, ctx_dev, w, lp, norm_k, q_rope_k, k_rope_k,
                              softmax_k, sc, ms, kv_sp, d)
            ref_h = torch_layer_fwd(ref_h, ctx_bf.float(), weights, lp,
                                    cos_q, sin_q, cos_kv, sin_kv)

            tt_h = rb(h, d)[:BSIZE, :HIDDEN].float()
            p = pcc(ref_h, tt_h)
            print(f"PCC={p:.6f} {'OK' if p > 0.70 else 'BAD'}")

        # Final norm
        final = rep(torch.zeros(SP, HIDDEN), d)
        norm_k(h, w["d.fn_w_tt"], sc, ms, final)
        tt_out = rb(final, d)[:BSIZE, :HIDDEN].float()
        ref_final = torch_rmsnorm(ref_h, weights["d.fn_w"].float())
        p = pcc(ref_final, tt_out)
        print(f"\nFinal PCC: {p:.6f} {'PASS' if p > 0.70 else 'FAIL'}")

    finally:
        ttnn.close_mesh_device(d)


if __name__ == "__main__":
    main()
