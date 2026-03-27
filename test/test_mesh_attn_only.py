"""Test just the attention path on 4-chip mesh, following qwen3.py patterns exactly.

Skip MLP, just test: norm -> QKV proj -> QK-norm -> RoPE -> SDPA -> O proj -> residual
"""
import torch
import torch.nn.functional as F
import ttnn
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from rope import make_rope_kernel

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

    # Weights
    in_w = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
    qw = torch.randn(HIDDEN, NQH * HDIM).to(torch.bfloat16) * 0.02
    kw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    vw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    ow = torch.randn(NQH * HDIM, HIDDEN).to(torch.bfloat16) * 0.02
    qnw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0
    knw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0

    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    ctx_bf = torch.randn(ctx_len, HIDDEN).to(torch.bfloat16) * 0.1

    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    cos_q = torch.cos(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    sin_q = torch.sin(torch.outer(torch.arange(BSIZE, dtype=torch.float32), freqs))
    cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
    sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))

    # PyTorch reference
    ref_normed = torch_rmsnorm(noise_bf.float(), in_w.float())
    ref_q = ref_normed @ qw.float()
    ref_kv_in = torch.cat([ctx_bf.float(), ref_normed], dim=0)
    ref_k = ref_kv_in @ kw.float()
    ref_v = ref_kv_in @ vw.float()

    ref_q_h = ref_q.view(BSIZE, NQH, HDIM)
    ref_k_h = ref_k.view(kv_len, NKVH, HDIM)
    ref_v_h = ref_v.view(kv_len, NKVH, HDIM)
    for h in range(NQH):
        ref_q_h[:, h] = torch_rmsnorm(ref_q_h[:, h], qnw.float())
    for h in range(NKVH):
        ref_k_h[:, h] = torch_rmsnorm(ref_k_h[:, h], knw.float())
    for h in range(NQH):
        ref_q_h[:, h] = torch_rope(ref_q_h[:, h], cos_q, sin_q)
    for h in range(NKVH):
        ref_k_h[:, h] = torch_rope(ref_k_h[:, h], cos_kv, sin_kv)

    scale = 1.0 / (HDIM ** 0.5)
    ref_attn = torch.zeros(BSIZE, NQH, HDIM)
    for qh in range(NQH):
        kvh = qh // GQA
        scores = (ref_q_h[:, qh] @ ref_k_h[:, kvh].T) * scale
        probs = torch.softmax(scores, dim=-1)
        ref_attn[:, qh] = probs @ ref_v_h[:, kvh]
    ref_attn_flat = ref_attn.view(BSIZE, NQH * HDIM)
    ref_o = ref_attn_flat @ ow.float()
    ref_out = noise_bf.float() + ref_o

    # Device
    print("Opening 4-chip mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    try:
        norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
        q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP)
        k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP)

        cos_full = torch.cos(torch.outer(torch.arange(max(SP, kv_sp), dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(torch.outer(torch.arange(max(SP, kv_sp), dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]

        sc = rep(torch.ones(TILE, TILE), d)
        ms = rep(torch.full((TILE, TILE), 1.0 / HIDDEN), d)

        # Pre-allocate scratch following qwen3.py patterns exactly
        s_norm = rep(torch.zeros(SP, HIDDEN), d)
        s_q = shd(torch.zeros(SP, Q_TP * N_CHIPS), d, dim=1)
        s_k = shd(torch.zeros(kv_sp, KV_TP * N_CHIPS), d, dim=1)
        s_v = shd(torch.zeros(kv_sp, KV_TP * N_CHIPS), d, dim=1)
        s_q_rope = shd(torch.zeros(SP, Q_TP * N_CHIPS), d, dim=1)
        s_k_rope = shd(torch.zeros(kv_sp, KV_TP * N_CHIPS), d, dim=1)
        s_o = rep(torch.zeros(SP, HIDDEN), d)
        s_add = rep(torch.zeros(SP, HIDDEN), d)

        # Upload weights
        in_w_tt = rep(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        qw_tt = shd(qw, d, dim=1)
        kw_tt = shd(kw, d, dim=1)
        vw_tt = shd(vw, d, dim=1)
        ow_tt = shd(ow, d, dim=0)
        qnw_tt = rep(qnw.unsqueeze(0).contiguous(), d)
        knw_tt = rep(knw.unsqueeze(0).contiguous(), d)
        cos_q_tt = rep(cos_full[:SP], d)
        sin_q_tt = rep(sin_adj[:SP], d)
        cos_kv_tt = rep(cos_full[:kv_sp], d)
        sin_kv_tt = rep(sin_adj[:kv_sp], d)

        h_dev = rep(noise_bf, d)
        ctx_dev = rep(ctx_bf, d)

        # Step 1: Norm
        print("Step 1: Norm...", end=" ")
        norm_k(h_dev, in_w_tt, sc, ms, s_norm)
        p = pcc(ref_normed, rb(s_norm, d)[:BSIZE, :HIDDEN].float())
        print(f"PCC={p:.6f}")

        # Step 2: Q proj (replicated @ col-sharded -> col-sharded)
        print("Step 2: Q proj...", end=" ")
        ttnn.matmul(s_norm, qw_tt, optional_output_tensor=s_q)
        # Check with rb1 (concat dim=1)
        q_rb = rb1(s_q, d)
        print(f"rb1 shape={q_rb.shape}", end=" ")
        p = pcc(ref_q, q_rb[:BSIZE, :NQH*HDIM].float())
        print(f"PCC={p:.6f}")
        # Also check without optional_output_tensor
        s_q2 = ttnn.matmul(s_norm, qw_tt)
        q_rb2 = rb1(s_q2, d)
        print(f"  (no opt_out) rb1 shape={q_rb2.shape}", end=" ")
        p2 = pcc(ref_q, q_rb2[:BSIZE, :NQH*HDIM].float())
        print(f"PCC={p2:.6f}")

        # Step 3: KV concat + K/V proj
        print("Step 3: KV proj...", end=" ")
        kv_in = ttnn.concat([ctx_dev, s_norm], dim=0)
        ttnn.matmul(kv_in, kw_tt, optional_output_tensor=s_k)
        ttnn.matmul(kv_in, vw_tt, optional_output_tensor=s_v)
        p = pcc(ref_k, rb1(s_k, d)[:kv_len, :NKVH*HDIM].float())
        print(f"K PCC={p:.6f}")

        # Step 4: QK-norm
        print("Step 4: QK-norm...", end=" ")
        q_flat = ttnn.reshape(s_q, (SP * NQH_TP, HDIM))
        k_flat = ttnn.reshape(s_k, (kv_sp * NKVH_TP, HDIM))
        q_normed = ttnn.rms_norm(q_flat, weight=qnw_tt, epsilon=EPS)
        k_normed = ttnn.rms_norm(k_flat, weight=knw_tt, epsilon=EPS)
        q_normed_2d = ttnn.reshape(q_normed, (SP, NQH_TP * HDIM))
        k_normed_2d = ttnn.reshape(k_normed, (kv_sp, NKVH_TP * HDIM))
        p = pcc(ref_q_h.reshape(BSIZE, NQH*HDIM), rb1(q_normed_2d, d)[:BSIZE, :NQH*HDIM].float())
        print(f"PCC={p:.6f}")

        # Step 5: RoPE (output to sharded scratch, matching qwen3.py)
        print("Step 5: RoPE...", end=" ")
        q_rope_k(q_normed_2d, cos_q_tt, sin_q_tt, s_q_rope)
        k_rope_k(k_normed_2d, cos_kv_tt, sin_kv_tt, s_k_rope)
        p = pcc(ref_q_h.reshape(BSIZE, NQH*HDIM), rb1(s_q_rope, d)[:BSIZE, :NQH*HDIM].float())
        print(f"PCC={p:.6f}")

        # Step 6: SDPA (matching qwen3.py exactly)
        print("Step 6: SDPA...", end=" ")
        q4 = ttnn.transpose(ttnn.reshape(s_q_rope, (1, SP, NQH_TP, HDIM)), 1, 2)
        k4 = ttnn.transpose(ttnn.reshape(s_k_rope, (1, kv_sp, NKVH_TP, HDIM)), 1, 2)
        v4 = ttnn.transpose(ttnn.reshape(s_v, (1, kv_sp, NKVH_TP, HDIM)), 1, 2)
        attn = ttnn.transformer.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        attn_tt = ttnn.reshape(ttnn.transpose(attn, 1, 2), (SP, NQH_TP * HDIM))
        p = pcc(ref_attn_flat, rb1(attn_tt, d)[:BSIZE, :NQH*HDIM].float())
        print(f"PCC={p:.6f}")

        # Step 7: O proj + all_reduce + residual
        print("Step 7: O proj + residual...", end=" ")
        ttnn.matmul(attn_tt, ow_tt, optional_output_tensor=s_o)
        o_reduced = ttnn.all_reduce(s_o)
        residual_add_kernel(h_dev, o_reduced, s_add)
        p = pcc(ref_out, rb(s_add, d)[:BSIZE, :HIDDEN].float())
        print(f"PCC={p:.6f}")

    finally:
        ttnn.close_mesh_device(d)


if __name__ == "__main__":
    main()
