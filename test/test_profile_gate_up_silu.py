"""Single-invocation gate_up+silu for profiling.

Full model size: (32, 2048) @ (2048, 6144) with k_tiles=64.
No warmup, single call for auto-profiler / perf summary.
"""
import sys
sys.path.insert(0, "/tmp")

import torch
import ttnn

TILE = 32
# DFlash model dimensions
SP = 32
HIDDEN = 2048
DINTER = 6144
K_TILES = HIDDEN // TILE  # 64


def to_dev(t, d):
    return ttnn.from_torch(
        t.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(42)
    d = ttnn.open_device(device_id=0)
    try:
        from matmul_silu_mul import make_matmul_silu_mul_kernel
        kernel = make_matmul_silu_mul_kernel(k_tiles=K_TILES)

        a = torch.randn(SP, HIDDEN, dtype=torch.bfloat16) * 0.02
        gw = torch.randn(HIDDEN, DINTER, dtype=torch.bfloat16) * 0.02
        uw = torch.randn(HIDDEN, DINTER, dtype=torch.bfloat16) * 0.02
        out = torch.zeros(SP, DINTER, dtype=torch.bfloat16)

        a_dev = to_dev(a, d)
        gw_dev = to_dev(gw, d)
        uw_dev = to_dev(uw, d)
        out_dev = to_dev(out, d)

        # Single invocation for profiling
        kernel(a_dev, gw_dev, uw_dev, out_dev)
        ttnn.synchronize_device(d)

        # Verify correctness
        result = ttnn.to_torch(out_dev)[:SP, :DINTER]
        expected = torch.nn.functional.silu(a.float() @ gw.float()) * (a.float() @ uw.float())
        err = (expected - result.float()).abs().max().item()
        print(f"gate_up+silu ({SP}x{HIDDEN} @ {HIDDEN}x{DINTER}): max_err={err:.4f} {'PASS' if err < 5.0 else 'FAIL'}")
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
