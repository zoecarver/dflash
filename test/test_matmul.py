"""Test streaming matmul kernel."""
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
    """Small: (32, 128) @ (128, 256)."""
    from streaming_matmul import make_matmul_kernel
    M, K, N = 32, 128, 256
    k_tiles = K // TILE

    a = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    expected = a.float() @ w.float()

    kernel = make_matmul_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"small ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


def test_q_proj(d):
    """Q projection size: (32, 2048) @ (2048, 4096)."""
    from streaming_matmul import make_matmul_kernel
    M, K, N = 32, 2048, 4096
    k_tiles = K // TILE

    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    expected = a.float() @ w.float()

    kernel = make_matmul_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"q_proj ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


def test_kv_proj(d):
    """KV projection size: (32, 2048) @ (2048, 512)."""
    from streaming_matmul import make_matmul_kernel
    M, K, N = 32, 2048, 512
    k_tiles = K // TILE

    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    expected = a.float() @ w.float()

    kernel = make_matmul_kernel(k_tiles=k_tiles)
    out_dev = to_dev(torch.zeros(M, N), d)
    kernel(to_dev(a, d), to_dev(w, d), out_dev)
    result = ttnn.to_torch(out_dev)[:M, :N]
    p = pcc(expected, result)
    print(f"kv_proj ({M}x{K} @ {K}x{N}): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"PCC too low: {p}"


if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        test_small(d)
        test_q_proj(d)
        test_kv_proj(d)
        print("\nAll tests PASSED!")
    finally:
        ttnn.close_device(d)
