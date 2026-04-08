"""Single-invocation matmul+residual_add for profiling.

Tests the down_proj size (worst case): (32, 6144) @ (6144, 2048) + residual.
k_tiles=192, which is the largest K dimension in dflash.
"""
import sys
sys.path.insert(0, "/tmp")

import torch
import ttnn

TILE = 32
SP = 32
DINTER = 6144
HIDDEN = 2048
K_TILES = DINTER // TILE  # 192


def to_dev(t, d):
    return ttnn.from_torch(
        t.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(42)
    d = ttnn.open_device(device_id=0)
    try:
        from matmul_residual_add import make_matmul_residual_add_kernel
        kernel = make_matmul_residual_add_kernel(k_tiles=K_TILES)

        a = torch.randn(SP, DINTER, dtype=torch.bfloat16) * 0.02
        w = torch.randn(DINTER, HIDDEN, dtype=torch.bfloat16) * 0.02
        res = torch.randn(SP, HIDDEN, dtype=torch.bfloat16) * 0.1
        out = torch.zeros(SP, HIDDEN, dtype=torch.bfloat16)

        a_dev = to_dev(a, d)
        w_dev = to_dev(w, d)
        res_dev = to_dev(res, d)
        out_dev = to_dev(out, d)

        # Single invocation for profiling
        kernel(a_dev, w_dev, res_dev, out_dev)
        ttnn.synchronize_device(d)

        result = ttnn.to_torch(out_dev)[:SP, :HIDDEN]
        expected = (a.float() @ w.float()) + res.float()
        err = (expected - result.float()).abs().max().item()
        print(f"down+resadd ({SP}x{DINTER} @ {DINTER}x{HIDDEN}): max_err={err:.4f} {'PASS' if err < 5.0 else 'FAIL'}")
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
