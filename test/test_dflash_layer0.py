"""Triage layer 0: test each sub-op and find where PCC drops."""
import torch
import torch.nn.functional as F
import ttnn
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel
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


def to_dev(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(42)
    ctx_len = 64
    kv_len = ctx_len + BSIZE
    kv_sp = _tile_pad(kv_len)

    # Random weights for layer 0
    in_w = torch.randn(HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
    qw = torch.randn(HIDDEN, NQH * HDIM).to(torch.bfloat16) * 0.02
    kw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    vw = torch.randn(HIDDEN, NKVH * HDIM).to(torch.bfloat16) * 0.02
    ow = torch.randn(NQH * HDIM, HIDDEN).to(torch.bfloat16) * 0.02
    qnw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0
    knw = torch.randn(HDIM).to(torch.bfloat16) * 0.1 + 1.0

    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    ctx_bf = torch.randn(ctx_len, HIDDEN).to(torch.bfloat16) * 0.1

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))

    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
        q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
        k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)
        sc = to_dev(torch.ones(TILE, TILE), d)
        ms = to_dev(torch.full((TILE, TILE), 1.0 / HIDDEN), d)

        # Step 1: Input LayerNorm
        print("=== Step 1: Input LayerNorm ===")
        ref_normed = torch_rmsnorm(noise_bf.float(), in_w.float())
        h_dev = to_dev(noise_bf, d)
        normed_dev = to_dev(torch.zeros(SP, HIDDEN), d)
        norm_k(h_dev, to_dev(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d), sc, ms, normed_dev)
        tt_normed = ttnn.to_torch(normed_dev).float()[:BSIZE, :HIDDEN]
        print(f"  PCC={pcc(ref_normed, tt_normed):.6f}")

        # Step 2: Q projection
        print("\n=== Step 2: Q projection ===")
        ref_q = ref_normed @ qw.float()  # (BSIZE, NQH*HDIM)
        qw_dev = to_dev(qw, d)
        q_dev = ttnn.matmul(normed_dev, qw_dev)
        tt_q = ttnn.to_torch(q_dev).float()[:BSIZE, :NQH*HDIM]
        print(f"  PCC={pcc(ref_q, tt_q):.6f}")

        # Step 3: KV concat + K projection
        print("\n=== Step 3: KV concat + K projection ===")
        ref_kv_in = torch.cat([ctx_bf.float(), ref_normed], dim=0)
        ref_k = ref_kv_in @ kw.float()
        ctx_dev = to_dev(ctx_bf, d)
        kv_in_dev = ttnn.concat([ctx_dev, normed_dev], dim=0)
        kw_dev = to_dev(kw, d)
        k_dev = ttnn.matmul(kv_in_dev, kw_dev)
        tt_k = ttnn.to_torch(k_dev).float()[:kv_len, :NKVH*HDIM]
        print(f"  PCC={pcc(ref_k, tt_k):.6f}")

        # Step 4: Per-head QK-norm
        print("\n=== Step 4: Per-head QK-norm ===")
        ref_q_heads = ref_q.view(BSIZE, NQH, HDIM).clone()
        for head in range(NQH):
            ref_q_heads[:, head] = torch_rmsnorm(ref_q_heads[:, head], qnw.float())
        ref_q_normed = ref_q_heads.view(BSIZE, NQH * HDIM)

        qnw_dev = to_dev(qnw.unsqueeze(0).contiguous(), d)
        q_flat = ttnn.reshape(q_dev, (SP * NQH, HDIM))
        q_normed_flat = ttnn.rms_norm(q_flat, weight=qnw_dev, epsilon=EPS)
        q_normed_dev = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
        tt_q_normed = ttnn.to_torch(q_normed_dev).float()[:BSIZE, :NQH*HDIM]
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

        q_roped_dev = to_dev(torch.zeros(SP, NQH * HDIM), d)
        q_rope_k(q_normed_dev, to_dev(cos_full[:SP], d), to_dev(sin_adj[:SP], d), q_roped_dev)
        tt_q_roped = ttnn.to_torch(q_roped_dev).float()[:BSIZE, :NQH*HDIM]
        print(f"  PCC={pcc(ref_q_roped, tt_q_roped):.6f}")

        # Step 6: SDPA
        print("\n=== Step 6: Cross-attention (SDPA) ===")
        ref_k_heads = ref_k.view(kv_len, NKVH, HDIM).clone()
        for head in range(NKVH):
            ref_k_heads[:, head] = torch_rmsnorm(ref_k_heads[:, head], knw.float())
        cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
        sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
        for head in range(NKVH):
            ref_k_heads[:, head] = torch_rope(ref_k_heads[:, head], cos_kv, sin_kv)

        ref_v = ref_kv_in @ vw.float()
        ref_v_heads = ref_v.view(kv_len, NKVH, HDIM)

        # GQA attention (with 1/sqrt(head_dim) scaling, matching SDPA)
        scale = 1.0 / (HDIM ** 0.5)
        attn_out = torch.zeros(BSIZE, NQH, HDIM)
        for qh in range(NQH):
            kvh = qh // GQA
            scores = (ref_q_roped.view(BSIZE, NQH, HDIM)[:, qh] @ ref_k_heads[:, kvh].T) * scale
            probs = torch.softmax(scores, dim=-1)
            attn_out[:, qh] = probs @ ref_v_heads[:, kvh]
        ref_attn = attn_out.view(BSIZE, NQH * HDIM)

        # Device SDPA
        knw_dev = to_dev(knw.unsqueeze(0).contiguous(), d)
        k_flat = ttnn.reshape(k_dev, (kv_sp * NKVH, HDIM))
        k_normed_flat = ttnn.rms_norm(k_flat, weight=knw_dev, epsilon=EPS)
        k_normed_dev = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))
        k_roped_dev = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
        k_rope_k(k_normed_dev, to_dev(cos_full[:kv_sp], d), to_dev(sin_adj[:kv_sp], d), k_roped_dev)

        v_dev = ttnn.matmul(kv_in_dev, to_dev(vw, d))

        q4 = ttnn.transpose(ttnn.reshape(q_roped_dev, (1, SP, NQH, HDIM)), 1, 2)
        k4 = ttnn.transpose(ttnn.reshape(k_roped_dev, (1, kv_sp, NKVH, HDIM)), 1, 2)
        v4 = ttnn.transpose(ttnn.reshape(v_dev, (1, kv_sp, NKVH, HDIM)), 1, 2)
        attn_dev = ttnn.transformer.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        attn_flat_dev = ttnn.reshape(ttnn.transpose(attn_dev, 1, 2), (SP, NQH * HDIM))
        tt_attn = ttnn.to_torch(attn_flat_dev).float()[:BSIZE, :NQH*HDIM]
        print(f"  PCC={pcc(ref_attn, tt_attn):.6f}")

        # Step 7: O projection + residual
        print("\n=== Step 7: O proj + residual ===")
        ref_o = ref_attn @ ow.float()
        ref_h1 = noise_bf.float() + ref_o

        o_dev = ttnn.matmul(attn_flat_dev, to_dev(ow, d))
        add_out = to_dev(torch.zeros(SP, HIDDEN), d)
        residual_add_kernel(h_dev, o_dev, add_out)
        tt_h1 = ttnn.to_torch(add_out).float()[:BSIZE, :HIDDEN]
        print(f"  PCC={pcc(ref_h1, tt_h1):.6f}")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
