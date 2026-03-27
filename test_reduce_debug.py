"""Debug TT-Lang reduce_sum: verify it computes what we think it computes.

Test with known input values to check if reduce_sum gives sum or mean,
and whether the scaler tile affects the result.
"""
import torch
import ttnn
import ttl

TILE = 32


@ttl.kernel(grid=(1, 1))
def reduce_sum_kernel(x, scaler, out):
    """Reduce a single tile along dim=1 (columns)."""
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            with x_dfb.wait() as xv, out_dfb.reserve() as o:
                o.store(ttl.math.reduce_sum(xv, sc, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_sum_2tile_kernel(x, scaler, out):
    """Reduce two column tiles (1x2) -> (1x1), accumulating."""
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    red_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            # First tile
            with x_dfb.wait() as xv, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(xv, sc, dims=[1]))
            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                acc.store(rv)
            # Second tile
            with x_dfb.wait() as xv, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(xv, sc, dims=[1]))
            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                acc.store(av + rv)
            # Output
            with acc_dfb.wait() as final, out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        # Two tiles from row 0
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk); tx.wait()
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, 1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t)


def test():
    device = ttnn.open_device(device_id=0)
    try:
        # Test 1: reduce_sum on a single tile of all 1.0s
        print("=== Test 1: reduce_sum(ones_32x32, dims=[1]) ===")
        x1 = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        sc = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        out1 = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

        reduce_sum_kernel(to_tt(x1, device), to_tt(sc, device), to_tt(out1, device))
        r1 = from_tt(to_tt(out1, device))  # need to read from device
        # Actually kernel writes to out, let me re-do
        x1_tt = to_tt(x1, device)
        sc_tt = to_tt(sc, device)
        out1_tt = to_tt(out1, device)
        reduce_sum_kernel(x1_tt, sc_tt, out1_tt)
        r1 = from_tt(out1_tt)
        print(f"  Expected: each row sums 32 ones = 32.0")
        print(f"  Got [0,0]: {r1[0,0].item():.4f}")
        print(f"  Got [1,0]: {r1[1,0].item():.4f}")
        print(f"  Got [0,1]: {r1[0,1].item():.4f} (should be 0 if reduced to col 0)")

        # Test 2: reduce_sum on a tile with known values
        print("\n=== Test 2: reduce_sum(arange_tile, dims=[1]) ===")
        x2 = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        x2[0, :] = torch.arange(TILE, dtype=torch.bfloat16)  # row 0: 0,1,2,...,31
        x2[1, :] = 1.0  # row 1: all 1s
        out2_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_sum_kernel(to_tt(x2, device), to_tt(sc, device), out2_tt)
        r2 = from_tt(out2_tt)
        print(f"  Row 0 expected: sum(0..31) = {sum(range(32))}")
        print(f"  Row 0 got [0,0]: {r2[0,0].item():.4f}")
        print(f"  Row 1 expected: 32.0")
        print(f"  Row 1 got [1,0]: {r2[1,0].item():.4f}")

        # Test 3: two-tile accumulation (like rmsnorm does)
        print("\n=== Test 3: reduce_sum two tiles accumulated ===")
        x3 = torch.ones(TILE, 2 * TILE, dtype=torch.bfloat16)  # 32x64, all 1s
        out3_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_sum_2tile_kernel(to_tt(x3, device), to_tt(sc, device), out3_tt)
        r3 = from_tt(out3_tt)
        print(f"  Expected: each row sums 64 ones = 64.0")
        print(f"  Got [0,0]: {r3[0,0].item():.4f}")
        print(f"  Got [1,0]: {r3[1,0].item():.4f}")

        # Test 4: reduce with scaler = 0.5
        print("\n=== Test 4: reduce_sum with scaler=0.5 ===")
        sc_half = torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16)
        out4_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        reduce_sum_kernel(to_tt(x1, device), to_tt(sc_half, device), out4_tt)
        r4 = from_tt(out4_tt)
        print(f"  With scaler=0.5, expected: 32*0.5=16 or 32*1=32?")
        print(f"  Got [0,0]: {r4[0,0].item():.4f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test()
