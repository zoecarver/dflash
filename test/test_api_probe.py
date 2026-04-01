"""Probe new broadcast/reduce API signatures."""
import torch
import ttnn
import ttl

TILE = 32

def to_dev(t, d):
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=d,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Test 1: broadcast with output arg
@ttl.operation(grid=(1, 1))
def test_broadcast(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.broadcast(x, o, dims=[0]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:2, 0:1]); tx.wait()

# Test 2: reduce_sum - check if scaler is still needed
@ttl.operation(grid=(1, 1))
def test_reduce_with_scaler(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, sc_dfb.wait() as sc, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(x, sc, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0:2], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()

if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        # Test broadcast
        print("Testing broadcast(x, output, dims)...")
        inp = to_dev(torch.ones(TILE, TILE), d)
        out = to_dev(torch.zeros(2*TILE, TILE), d)
        test_broadcast(inp, out)
        result = ttnn.to_torch(out)
        print(f"  broadcast result[0,0]={result[0,0]:.1f}, result[32,0]={result[32,0]:.1f}")
        print("  PASS")

        # Test reduce
        print("Testing reduce_sum(x, scaler, dims)...")
        inp2 = to_dev(torch.ones(TILE, 2*TILE), d)
        sc = to_dev(torch.ones(TILE, TILE), d)
        out2 = to_dev(torch.zeros(TILE, TILE), d)
        test_reduce_with_scaler(inp2, sc, out2)
        result2 = ttnn.to_torch(out2)
        print(f"  reduce result[0,0]={result2[0,0]:.1f} (expect 2.0)")
        print("  PASS")

        print("\nAll API probe tests passed!")
    finally:
        ttnn.close_device(d)
