"""Test fused gate/up matmul + silu_mul kernel."""
import sys
sys.path.insert(0, "/tmp")

import torch
import torch.nn.functional as F
import ttnn

TILE = 32

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def to_dev(t, d):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=d,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_small(d):
    """Small: (32, 128) @ (128, 256) for both gate and up."""
    from matmul_silu_mul import make_matmul_silu_mul_kernel
    M, K, N = 32, 128, 256
    k_tiles = K // TILE

    a = torch.randn(M, K, dtype=torch.bfloat16)
    gw = torch.randn(K, N, dtype=torch.bfloat16)
    uw = torch.randn(K, N, dtype=torch.bfloat16)
    expected = F.silu(a.float() @ gw.float()) * (a.float() @ uw.float())

    kernel = make_matmul_silu_mul_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(gw, d), to_dev(uw, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"small ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


def test_mlp_size(d):
    """MLP size: (32, 2048) @ (2048, 6144)."""
    from matmul_silu_mul import make_matmul_silu_mul_kernel
    M, K, N = 32, 2048, 6144
    k_tiles = K // TILE

    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    gw = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    uw = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    expected = F.silu(a.float() @ gw.float()) * (a.float() @ uw.float())

    kernel = make_matmul_silu_mul_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(gw, d), to_dev(uw, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"mlp ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        test_small(d)
        test_mlp_size(d)
        print("\nAll tests PASSED!")
    finally:
        ttnn.close_device(d)
