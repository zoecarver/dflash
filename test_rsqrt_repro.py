"""Reproducer: tile_mul between outer-scope DFB value and inner-scope DFB value fails.

Pattern from rmsnorm: ms is held in outer with, bc_val comes from inner with.
bc_val * ms fails to legalize tile_mul.
"""
import torch
import ttnn
import ttl

TILE = 32


@ttl.kernel(grid=(1, 1))
def repro_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # b is held in outer scope (like ms in rmsnorm)
        with b_dfb.wait() as bv:
            # a comes from inner scope (like bc_val in rmsnorm)
            with a_dfb.wait() as av, out_dfb.reserve() as o:
                o.store(av * bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test():
    device = ttnn.open_device(device_id=0)
    try:
        a_pt = torch.rand(TILE, TILE, dtype=torch.bfloat16)
        b_pt = torch.rand(TILE, TILE, dtype=torch.bfloat16)
        out_pt = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

        a = to_tt(a_pt, device)
        b = to_tt(b_pt, device)
        out = to_tt(out_pt, device)

        repro_kernel(a, b, out)

        result = ttnn.to_torch(out)
        ref = (a_pt.float() * b_pt.float()).bfloat16()
        diff = (result.float() - ref.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        print("PASS" if diff < 0.05 else "FAIL")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test()
