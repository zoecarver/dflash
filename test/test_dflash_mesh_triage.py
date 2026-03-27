"""Triage layer 0 on 4-chip mesh: test each sub-op."""
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
NQH_TP = NQH // N_CHIPS
NKVH_TP = NKVH // N_CHIPS
Q_TP = NQH_TP * HDIM
KV_TP = NKVH_TP * HDIM
DINTER = 6144
BSIZE = 16
SP = ((BSIZE + TILE - 1) // TILE) * TILE


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


def torch_rmsnorm(x, weight, eps=EPS):
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight


def torch_rope(x, cos, sin):
    x1, x2 = x[..., :HDIM//2], x[..., HDIM//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


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


def rb1(t, d):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(d, dim=1))


def main():
    torch.manual_seed(42)
    ctx_len = 64
    kv_len = ctx_len + BSIZE
    kv_sp = _tile_pad(kv_len)
    kv_tiles = kv_sp // TILE

    in_w = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
    qw = torch.randn(HIDDEN, NQH * HDIM).to(torch.bfloat16) * 0.02
    kw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    vw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    qnw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0
    knw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0

    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    ctx_bf = torch.randn(ctx_len, HIDDEN).to(torch.bfloat16) * 0.1

    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))

    print("Opening 4-chip mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    try:
        norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
        q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP)
        k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP)
        softmax_k = make_softmax_kernel(NQH_TP * SP // TILE, kv_tiles)
        sc = rep(torch.ones(TILE, TILE), d)
        ms = rep(torch.full((TILE, TILE), 1.0 / HIDDEN), d)

        # Step 1: Input LayerNorm (replicated -> replicated)
        print("=== Step 1: Input LayerNorm ===")
        ref_normed = torch_rmsnorm(noise_bf.float(), in_w.float())
        h_dev = rep(noise_bf, d)
        normed_dev = rep(torch.zeros(SP, HIDDEN), d)
        norm_k(h_dev, rep(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d), sc, ms, normed_dev)
        tt_normed = rb(normed_dev, d)[:BSIZE, :HIDDEN].float()
        print(f"  PCC={pcc(ref_normed, tt_normed):.6f}")

        # Step 2: Q projection (replicated @ col-sharded = col-sharded)
        print("\n=== Step 2: Q projection ===")
        ref_q = ref_normed @ qw.float()  # (BSIZE, NQH*HDIM)
        qw_dev = shd(qw, d, dim=1)
        q_dev = ttnn.matmul(normed_dev, qw_dev)  # col-sharded
        tt_q = rb1(q_dev, d)[:BSIZE, :NQH*HDIM].float()
        print(f"  PCC={pcc(ref_q, tt_q):.6f}")

        # Step 3: KV concat + K projection
        print("\n=== Step 3: KV concat + K proj ===")
        ref_kv_in = torch.cat([ctx_bf.float(), ref_normed], dim=0)
        ref_k = ref_kv_in @ kw.float()
        ctx_dev = rep(ctx_bf, d)
        kv_in_dev = ttnn.concat([ctx_dev, normed_dev], dim=0)
        kw_dev = shd(kw, d, dim=1)
        k_dev = ttnn.matmul(kv_in_dev, kw_dev)
        tt_k = rb1(k_dev, d)[:kv_len, :NKVH*HDIM].float()
        print(f"  PCC={pcc(ref_k, tt_k):.6f}")

        # Step 4: Per-head QK-norm
        print("\n=== Step 4: QK-norm ===")
        ref_q_heads = ref_q.view(BSIZE, NQH, HDIM).clone()
        for head in range(NQH):
            ref_q_heads[:, head] = torch_rmsnorm(ref_q_heads[:, head], qnw.float())
        ref_q_normed = ref_q_heads.view(BSIZE, NQH * HDIM)

        qnw_dev = rep(qnw.unsqueeze(0).contiguous(), d)
        q_flat = ttnn.reshape(q_dev, (SP * NQH_TP, HDIM))
        q_normed_flat = ttnn.rms_norm(q_flat, weight=qnw_dev, epsilon=EPS)
        q_normed_dev = ttnn.reshape(q_normed_flat, (SP, NQH_TP * HDIM))
        # Readback per-chip, concat along dim1 to get full Q
        tt_q_normed = rb1(q_normed_dev, d)[:BSIZE, :NQH*HDIM].float()
        print(f"  PCC={pcc(ref_q_normed, tt_q_normed):.6f}")

        # Step 5: RoPE on Q
        print("\n=== Step 5: RoPE on Q ===")
        cos_q = torch.cos(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
        sin_q = torch.sin(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
        ref_q_roped = ref_q_normed.view(BSIZE, NQH, HDIM).clone()
        for head in range(NQH):
            ref_q_roped[:, head] = torch_rope(ref_q_roped[:, head], cos_q, sin_q)
        ref_q_roped = ref_q_roped.view(BSIZE, NQH * HDIM)

        cos_full = torch.cos(torch.outer(torch.arange(max(SP, kv_sp), dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(torch.outer(torch.arange(max(SP, kv_sp), dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]

        q_roped_dev = shd(torch.zeros(SP, Q_TP * N_CHIPS), d, dim=1)
        q_rope_k(q_normed_dev, rep(cos_full[:SP], d), rep(sin_adj[:SP], d), q_roped_dev)
        tt_q_roped = rb1(q_roped_dev, d)[:BSIZE, :Q_TP].float()
        # Compare per-chip (chip 0 has heads 0-7)
        ref_chip0 = ref_q_roped[:BSIZE, :Q_TP]
        print(f"  PCC (chip0 heads 0-7)={pcc(ref_chip0, tt_q_roped):.6f}")

        # Step 6: Cross-attention (stacked Q + softmax)
        print("\n=== Step 6: Cross-attention ===")
        ref_k_heads = ref_k.view(kv_len, NKVH, HDIM).clone()
        for head in range(NKVH):
            ref_k_heads[:, head] = torch_rmsnorm(ref_k_heads[:, head], knw.float())
        cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
        sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
        for head in range(NKVH):
            ref_k_heads[:, head] = torch_rope(ref_k_heads[:, head], cos_kv, sin_kv)
        ref_v = ref_kv_in @ vw.float()
        ref_v_heads = ref_v.view(kv_len, NKVH, HDIM)

        # Per-chip reference: chip 0 has Q heads 0-7, KV head 0
        scale = 1.0 / (HDIM ** 0.5)
        ref_attn_chip0 = torch.zeros(BSIZE, NQH_TP, HDIM)
        for qh in range(NQH_TP):
            scores = (ref_q_roped.view(BSIZE, NQH, HDIM)[:, qh] @ ref_k_heads[:, 0].T) * scale
            probs = torch.softmax(scores, dim=-1)
            ref_attn_chip0[:, qh] = probs @ ref_v_heads[:, 0]

        # Device attention
        knw_dev = rep(knw.unsqueeze(0).contiguous(), d)
        k_flat = ttnn.reshape(k_dev, (kv_sp * NKVH_TP, HDIM))
        k_normed_flat = ttnn.rms_norm(k_flat, weight=knw_dev, epsilon=EPS)
        k_normed_dev = ttnn.reshape(k_normed_flat, (kv_sp, NKVH_TP * HDIM))
        k_roped_dev = shd(torch.zeros(kv_sp, KV_TP * N_CHIPS), d, dim=1)
        k_rope_k(k_normed_dev, rep(cos_full[:kv_sp], d), rep(sin_adj[:kv_sp], d), k_roped_dev)

        vw_dev = shd(vw, d, dim=1)
        v_dev = ttnn.matmul(kv_in_dev, vw_dev)

        q_stacked = ttnn.reshape(q_roped_dev, (NQH_TP * SP, HDIM))
        k_hdim = ttnn.reshape(k_roped_dev, (kv_sp, HDIM))
        v_hdim = ttnn.reshape(v_dev, (kv_sp, HDIM))
        k_t = ttnn.transpose(k_hdim, -2, -1)
        scores_dev = ttnn.matmul(q_stacked, k_t)
        scores_dev = ttnn.multiply(scores_dev, scale)

        # Check intermediate scores
        tt_scores = rb(scores_dev, d)[:NQH_TP * BSIZE, :kv_len].float()
        ref_scores = torch.zeros(NQH_TP, BSIZE, kv_len)
        for qh in range(NQH_TP):
            ref_scores[qh] = (ref_q_roped.view(BSIZE, NQH, HDIM)[:, qh] @ ref_k_heads[:, 0].T) * scale
        ref_scores_flat = ref_scores.view(NQH_TP * BSIZE, kv_len)
        # tt_scores is (N_CHIPS * NQH_TP * SP, kv_sp) - take chip 0
        tt_scores_chip0 = rb(scores_dev, d)[:NQH_TP * SP, :kv_sp].float()
        # Reshape to (NQH_TP, SP, kv_sp) and trim
        tt_scores_r = tt_scores_chip0.view(NQH_TP, SP, kv_sp)[:, :BSIZE, :kv_len]
        print(f"  Scores PCC (chip0)={pcc(ref_scores, tt_scores_r):.6f}")

        # Check softmax output
        probs_dev = rep(torch.zeros(NQH_TP * SP, kv_sp), d)
        softmax_k(scores_dev, sc, probs_dev)
        tt_probs_chip0 = rb(probs_dev, d)[:NQH_TP * SP, :kv_sp].float()
        ref_probs = torch.zeros(NQH_TP, BSIZE, kv_len)
        for qh in range(NQH_TP):
            ref_probs[qh] = torch.softmax(ref_scores[qh], dim=-1)
        tt_probs_r = tt_probs_chip0.view(NQH_TP, SP, kv_sp)[:, :BSIZE, :kv_len]
        print(f"  Probs PCC (chip0)={pcc(ref_probs, tt_probs_r):.6f}")

        # Check final attention output
        attn_out_dev = ttnn.matmul(probs_dev, v_hdim)
        tt_attn_raw = rb(attn_out_dev, d)
        chip_rows = NQH_TP * SP
        tt_attn = tt_attn_raw[:chip_rows, :HDIM].float()
        tt_attn_reshaped = tt_attn.view(NQH_TP, SP, HDIM)[:, :BSIZE, :]
        ref_attn_reshaped = ref_attn_chip0.transpose(0, 1)
        print(f"  Attn PCC (chip0)={pcc(ref_attn_reshaped, tt_attn_reshaped):.6f}")

    finally:
        ttnn.close_mesh_device(d)


if __name__ == "__main__":
    main()
