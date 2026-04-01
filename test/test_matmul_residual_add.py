"""Test fused matmul + residual_add kernel."""
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
    """Small test: (1tile, 4tiles) @ (4tiles, 8tiles) + (1tile, 8tiles)."""
    from matmul_residual_add import make_matmul_residual_add_kernel

    M, K, N = 32, 128, 256
    k_tiles = K // TILE  # 4

    a = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    res = torch.randn(M, N, dtype=torch.bfloat16)
    expected = (a.float() @ w.float()) + res.float()

    kernel = make_matmul_residual_add_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), to_dev(res, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"small ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


def test_o_proj_size(d):
    """O-projection size: (32, 4096) @ (4096, 2048) + (32, 2048)."""
    from matmul_residual_add import make_matmul_residual_add_kernel

    M, K, N = 32, 4096, 2048
    k_tiles = K // TILE  # 128

    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    res = torch.randn(M, N, dtype=torch.bfloat16) * 0.1
    expected = (a.float() @ w.float()) + res.float()

    kernel = make_matmul_residual_add_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), to_dev(res, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"o_proj ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


def test_down_proj_size(d):
    """Down-projection size: (32, 6144) @ (6144, 2048) + (32, 2048)."""
    from matmul_residual_add import make_matmul_residual_add_kernel

    M, K, N = 32, 6144, 2048
    k_tiles = K // TILE  # 192

    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    res = torch.randn(M, N, dtype=torch.bfloat16) * 0.1
    expected = (a.float() @ w.float()) + res.float()

    kernel = make_matmul_residual_add_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), to_dev(res, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"down_proj ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        test_small(d)
        test_o_proj_size(d)
        test_down_proj_size(d)
        print("\nAll tests PASSED!")
    finally:
        ttnn.close_device(d)
