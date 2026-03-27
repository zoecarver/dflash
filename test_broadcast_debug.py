"""Debug TT-Lang broadcast: verify dims=[1] broadcasts column across row."""
import torch
import ttnn
import ttl

TILE = 32


@ttl.kernel(grid=(1, 1))
def broadcast_kernel(x, scaler, out):
    """reduce_sum(dims=[1]) then broadcast(dims=[1])."""
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            # Reduce to get column sums
            with x_dfb.wait() as xv, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(xv, sc, dims=[1]))
            # Broadcast back
            with red_dfb.wait() as rv, bc_dfb.reserve() as bc:
                bc.store(ttl.math.broadcast(rv, dims=[1]))
            # Output the broadcast result
            with bc_dfb.wait() as bv, out_dfb.reserve() as o:
                o.store(bv)

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
def full_rmsnorm_pattern(x, scaler, ms, out):
    """Reproduce the exact rmsnorm pattern: reduce -> broadcast -> multiply ms -> rsqrt."""
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(ms, shape=(1, 1), buffer_factor=1)
    sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc, ms_dfb.wait() as msv:
            # x^2
            with x_dfb.wait() as xv, sq_dfb.reserve() as sq:
                sq.store(xv * xv)
            # reduce_sum
            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
            # broadcast
            with red_dfb.wait() as rv, bc_dfb.reserve() as bc:
                bc.store(ttl.math.broadcast(rv, dims=[1]))
            # rsqrt(bc * ms + eps)
            with bc_dfb.wait() as bv, out_dfb.reserve() as o:
                o.store(ttl.math.rsqrt(bv * msv + ttl.math.fill(bv, 0.000001)))

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(ms[0, 0], blk); tx.wait()
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk); tx.wait()

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
        sc = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        # Test 1: broadcast after reduce
        print("=== Test 1: reduce_sum(dims=[1]) -> broadcast(dims=[1]) ===")
        x1 = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        out1_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        broadcast_kernel(to_tt(x1, device), to_tt(sc, device), out1_tt)
        r1 = from_tt(out1_tt)
        print(f"  Input: all 1s (32x32)")
        print(f"  After reduce_sum(dims=[1]): each row should be 32")
        print(f"  After broadcast(dims=[1]): all elements should be 32")
        print(f"  Got [0,0]: {r1[0,0].item():.4f}")
        print(f"  Got [0,15]: {r1[0,15].item():.4f}")
        print(f"  Got [0,31]: {r1[0,31].item():.4f}")

        # Test 2: broadcast with varying rows
        print("\n=== Test 2: reduce + broadcast with row 0 = arange ===")
        x2 = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        x2[0, :] = torch.arange(TILE, dtype=torch.bfloat16)  # sum = 496
        x2[1, :] = 2.0  # sum = 64
        out2_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        broadcast_kernel(to_tt(x2, device), to_tt(sc, device), out2_tt)
        r2 = from_tt(out2_tt)
        print(f"  Row 0: sum(0..31)=496, broadcast should fill row with 496")
        print(f"  Got [0,0]: {r2[0,0].item():.4f}  [0,15]: {r2[0,15].item():.4f}  [0,31]: {r2[0,31].item():.4f}")
        print(f"  Row 1: sum(32 * 2)=64, broadcast should fill row with 64")
        print(f"  Got [1,0]: {r2[1,0].item():.4f}  [1,15]: {r2[1,15].item():.4f}  [1,31]: {r2[1,31].item():.4f}")

        # Test 3: full rmsnorm pattern on a single tile
        print("\n=== Test 3: full rmsnorm pattern (single tile) ===")
        HIDDEN_1TILE = 32  # pretend hidden dim is 32 for 1 tile
        ms_val = 1.0 / HIDDEN_1TILE
        ms = torch.full((TILE, TILE), ms_val, dtype=torch.bfloat16)
        x3 = torch.randn(TILE, TILE).to(torch.bfloat16)
        out3_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        full_rmsnorm_pattern(to_tt(x3, device), to_tt(sc, device), to_tt(ms, device), out3_tt)
        r3 = from_tt(out3_tt)

        # PyTorch reference: rsqrt(mean(x^2) + eps) for each row
        x3f = x3.float()
        variance = x3f.pow(2).sum(dim=-1, keepdim=True) * ms_val
        ref = torch.rsqrt(variance + 1e-6).expand_as(x3f).to(torch.bfloat16)

        diff = (r3.float() - ref.float()).abs()
        print(f"  rsqrt(sum(x^2)*ms + eps) per row")
        print(f"  Ref [0,0]: {ref[0,0].item():.6f}  Dev [0,0]: {r3[0,0].item():.6f}")
        print(f"  Ref [0,15]: {ref[0,15].item():.6f}  Dev [0,15]: {r3[0,15].item():.6f}")
        print(f"  Ref [1,0]: {ref[1,0].item():.6f}  Dev [1,0]: {r3[1,0].item():.6f}")
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Ratio dev/ref [0,0]: {r3[0,0].item() / ref[0,0].item():.4f}")
        print(f"  Ratio dev/ref [1,0]: {r3[1,0].item() / ref[1,0].item():.4f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test()
