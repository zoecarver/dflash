"""Minimal smoke test to verify ttl.kernel -> ttl.operation rename works."""
import sys
sys.path.insert(0, "/tmp")

import torch
import torch.nn.functional as F
import ttnn
import ttl

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
    t = t.contiguous().to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def test_residual_add(d):
    from residual_add import residual_add_kernel
    a = torch.randn(32, 256, dtype=torch.bfloat16)
    b = torch.randn(32, 256, dtype=torch.bfloat16)
    expected = a + b
    a_dev = to_dev(a, d)
    b_dev = to_dev(b, d)
    out_dev = to_dev(torch.zeros_like(a), d)
    residual_add_kernel(a_dev, b_dev, out_dev)
    result = ttnn.to_torch(out_dev)[:32, :256]
    p = pcc(expected, result)
    print(f"residual_add: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"residual_add PCC too low: {p}"

def test_silu_mul(d):
    from silu_mul import silu_mul_kernel
    gate = torch.randn(32, 256, dtype=torch.bfloat16)
    up = torch.randn(32, 256, dtype=torch.bfloat16)
    expected = F.silu(gate.float()) * up.float()
    gate_dev = to_dev(gate, d)
    up_dev = to_dev(up, d)
    out_dev = to_dev(torch.zeros_like(gate), d)
    silu_mul_kernel(gate_dev, up_dev, out_dev)
    result = ttnn.to_torch(out_dev)[:32, :256]
    p = pcc(expected, result)
    print(f"silu_mul: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"silu_mul PCC too low: {p}"

def test_rmsnorm(d):
    from rmsnorm import make_rmsnorm_kernel
    dim = 256
    dim_tiles = dim // TILE
    x = torch.randn(32, dim, dtype=torch.bfloat16)
    w = (torch.randn(dim, dtype=torch.bfloat16) * 0.1 + 1.0)
    eps = 1e-6
    expected = (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) * w.float()

    kernel = make_rmsnorm_kernel(dim_tiles=dim_tiles, eps=eps)
    x_dev = to_dev(x, d)
    w_dev = to_dev(w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
    sc = to_dev(torch.ones(TILE, TILE), d)
    ms = to_dev(torch.full((TILE, TILE), 1.0 / dim), d)
    out_dev = to_dev(torch.zeros_like(x), d)
    kernel(x_dev, w_dev, sc, ms, out_dev)
    result = ttnn.to_torch(out_dev)[:32, :dim]
    p = pcc(expected, result)
    print(f"rmsnorm: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"rmsnorm PCC too low: {p}"

def test_softmax(d):
    from softmax import make_softmax_kernel
    rows = 2
    cols = 4
    x = torch.randn(rows * TILE, cols * TILE, dtype=torch.bfloat16)
    expected = torch.softmax(x.float(), dim=-1)

    kernel = make_softmax_kernel(n_rows=rows, col_tiles=cols)
    x_dev = to_dev(x, d)
    sc = to_dev(torch.ones(TILE, TILE), d)
    out_dev = to_dev(torch.zeros_like(x), d)
    kernel(x_dev, sc, out_dev)
    result = ttnn.to_torch(out_dev)[:rows*TILE, :cols*TILE]
    p = pcc(expected, result)
    print(f"softmax: PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}")
    assert p > 0.98, f"softmax PCC too low: {p}"

if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        test_residual_add(d)
        test_silu_mul(d)
        test_rmsnorm(d)
        test_softmax(d)
        print("\nAll smoke tests PASSED!")
    finally:
        ttnn.close_device(d)
