"""Test per-head RMSNorm and RoPE kernels against PyTorch reference."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE  # 4
NQH = 8  # per chip
NKVH = 1  # per chip
EPS = 1e-6

from per_head_rmsnorm import make_per_head_rmsnorm_kernel
from rope import make_rope_kernel

q_norm_k = make_per_head_rmsnorm_kernel(head_tiles=HDIM_TILES, n_heads=NQH, eps=EPS)
k_norm_k = make_per_head_rmsnorm_kernel(head_tiles=HDIM_TILES, n_heads=NKVH, eps=EPS)
q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)


def _p(t):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph, pw = (TILE - h % TILE) % TILE, (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def rep(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    d = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)
        sl = 4  # small sequence for testing
        sp = ((sl + TILE - 1) // TILE) * TILE

        # Generate Q and K tensors
        q = torch.randn(sl, NQH * HDIM, dtype=torch.float32)
        k = torch.randn(sl, NKVH * HDIM, dtype=torch.float32)
        qnw = torch.randn(HDIM, dtype=torch.float32)
        knw = torch.randn(HDIM, dtype=torch.float32)

        # ===== Test 1: Per-head RMSNorm =====
        print("=== Per-head RMSNorm ===")

        # PyTorch reference
        def head_rms_norm(x, nw, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return ((x4 / rms) * nw).view(x.shape)

        q_ref = head_rms_norm(q.float(), qnw.float(), NQH).to(torch.bfloat16)
        k_ref = head_rms_norm(k.float(), knw.float(), NKVH).to(torch.bfloat16)

        # Device
        q_tt = rep(q.to(torch.bfloat16), d)
        k_tt = rep(k.to(torch.bfloat16), d)
        qnw_tt = rep(qnw.unsqueeze(0).expand(TILE, -1).to(torch.bfloat16), d)
        knw_tt = rep(knw.unsqueeze(0).expand(TILE, -1).to(torch.bfloat16), d)
        sc = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d)
        ms_q = rep(torch.full((TILE, TILE), 1.0 / HDIM, dtype=torch.bfloat16), d)

        q_out = rep(torch.zeros(sp, NQH * HDIM, dtype=torch.bfloat16), d)
        k_out = rep(torch.zeros(sp, NKVH * HDIM, dtype=torch.bfloat16), d)

        q_norm_k(q_tt, qnw_tt, sc, ms_q, q_out)
        k_norm_k(k_tt, knw_tt, sc, ms_q, k_out)

        q_dev = ttnn.to_torch(q_out)[:sl, :NQH * HDIM].to(torch.bfloat16)
        k_dev = ttnn.to_torch(k_out)[:sl, :NKVH * HDIM].to(torch.bfloat16)

        print(f"  Q norm PCC: {pcc(q_ref, q_dev):.6f}")
        print(f"  K norm PCC: {pcc(k_ref, k_dev):.6f}")

        # Per-head check
        for h in range(min(4, NQH)):
            r = q_ref[:, h*HDIM:(h+1)*HDIM]
            d_out = q_dev[:, h*HDIM:(h+1)*HDIM]
            print(f"  Q head {h} PCC: {pcc(r, d_out):.6f}")

        # ===== Test 2: RoPE =====
        print("\n=== RoPE ===")

        ROPE_THETA = 1e7
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
        pos = torch.arange(sl, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        cos_t = torch.cos(angles).to(torch.bfloat16)  # (sl, HDIM/2)
        sin_t = torch.sin(angles).to(torch.bfloat16)

        # Full cos/sin for HDIM
        cos_full = cos_t.repeat(1, 2)[:, :HDIM]
        sin_full = sin_t.repeat(1, 2)[:, :HDIM]

        # PyTorch reference for RoPE
        def rotate_half_flat(x, n_heads):
            x4 = x.view(-1, n_heads, HDIM)
            x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
            return torch.cat((-x2, x1), dim=-1).view(x.shape)

        q_norm_ref = q_ref.float()
        q_rope_ref = (q_norm_ref * cos_full.float().repeat(1, NQH) +
                      rotate_half_flat(q_norm_ref, NQH) * sin_full.float().repeat(1, NQH)).to(torch.bfloat16)

        k_norm_ref = k_ref.float()
        k_rope_ref = (k_norm_ref * cos_full.float().repeat(1, NKVH) +
                      rotate_half_flat(k_norm_ref, NKVH) * sin_full.float().repeat(1, NKVH)).to(torch.bfloat16)

        # Pre-adjusted sin: negate first half
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]

        cos_tt = rep(cos_full, d)
        sin_adj_tt = rep(sin_adj, d)

        # Use q_dev (already normalized on device) as input to RoPE
        q_roped_out = rep(torch.zeros(sp, NQH * HDIM, dtype=torch.bfloat16), d)
        k_roped_out = rep(torch.zeros(sp, NKVH * HDIM, dtype=torch.bfloat16), d)

        q_rope_k(q_out, cos_tt, sin_adj_tt, q_roped_out)
        k_rope_k(k_out, cos_tt, sin_adj_tt, k_roped_out)

        q_roped_dev = ttnn.to_torch(q_roped_out)[:sl, :NQH * HDIM].to(torch.bfloat16)
        k_roped_dev = ttnn.to_torch(k_roped_out)[:sl, :NKVH * HDIM].to(torch.bfloat16)

        print(f"  Q RoPE PCC: {pcc(q_rope_ref, q_roped_dev):.6f}")
        print(f"  K RoPE PCC: {pcc(k_rope_ref, k_roped_dev):.6f}")

        # Per-head check
        for h in range(min(4, NQH)):
            r = q_rope_ref[:, h*HDIM:(h+1)*HDIM]
            d_out = q_roped_dev[:, h*HDIM:(h+1)*HDIM]
            print(f"  Q head {h} RoPE PCC: {pcc(r, d_out):.6f}")

        print(f"\n  Q ref first 8: {q_rope_ref[0, :8].tolist()}")
        print(f"  Q dev first 8: {q_roped_dev[0, :8].tolist()}")

    finally:
        ttnn.CloseDevice(d)


if __name__ == "__main__":
    main()
