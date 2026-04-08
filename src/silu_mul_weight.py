"""Fused SiLU-gated MLP with routing weight: out = silu(gate) * up * weight.

Used in MoE to fuse the activation function with the per-expert routing
weight multiplication, eliminating a device-to-host round-trip for the
intermediate activations."""
import ttl

TILE = 32
BLOCK_R = 8
BLOCK_C = 8
BUF = 3


def make_silu_mul_weight_kernel(block_r=BLOCK_R, block_c=BLOCK_C, buf=BUF):

    @ttl.operation(grid="auto")
    def silu_mul_weight_kernel(gate, up, weight, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        row_blocks = gate.shape[0] // TILE // block_r
        col_blocks = gate.shape[1] // TILE // block_c
        total = row_blocks * col_blocks
        units_per_core = -(-total // grid_cols)

        gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(block_r, block_c), buffer_factor=buf)
        up_dfb = ttl.make_dataflow_buffer_like(up, shape=(block_r, block_c), buffer_factor=buf)
        weight_dfb = ttl.make_dataflow_buffer_like(weight, shape=(block_r, block_c), buffer_factor=buf)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(block_r, block_c), buffer_factor=buf)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    with gate_dfb.wait() as gv, up_dfb.wait() as uv, weight_dfb.wait() as wv, out_dfb.reserve() as o:
                        o.store(gv * ttl.math.sigmoid(gv) * uv * wv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    rb = t // col_blocks
                    cb = t % col_blocks
                    sr = rb * block_r
                    sc = cb * block_c
                    with gate_dfb.reserve() as b1, up_dfb.reserve() as b2, weight_dfb.reserve() as b3:
                        tx1 = ttl.copy(gate[sr:sr + block_r, sc:sc + block_c], b1)
                        tx2 = ttl.copy(up[sr:sr + block_r, sc:sc + block_c], b2)
                        tx3 = ttl.copy(weight[sr:sr + block_r, sc:sc + block_c], b3)
                        tx1.wait(); tx2.wait(); tx3.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    rb = t // col_blocks
                    cb = t % col_blocks
                    sr = rb * block_r
                    sc = cb * block_c
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[sr:sr + block_r, sc:sc + block_c]); tx.wait()

    return silu_mul_weight_kernel


# Default: 8x8 block, buffer_factor=3
silu_mul_weight_kernel = make_silu_mul_weight_kernel()
