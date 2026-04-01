"""Fused SiLU-gated MLP with routing weight: out = silu(gate) * up * weight.

Used in MoE to fuse the activation function with the per-expert routing
weight multiplication, eliminating a device-to-host round-trip for the
intermediate activations."""
import ttl

TILE = 32
ELEM_GRAN = 8


@ttl.operation(grid="auto")
def silu_mul_weight_kernel(gate, up, weight, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = gate.shape[0] // TILE
    col_blocks = gate.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)

    gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, ELEM_GRAN), buffer_factor=2)
    up_dfb = ttl.make_dataflow_buffer_like(up, shape=(1, ELEM_GRAN), buffer_factor=2)
    weight_dfb = ttl.make_dataflow_buffer_like(weight, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with gate_dfb.wait() as gv, up_dfb.wait() as uv, weight_dfb.wait() as wv, out_dfb.reserve() as o:
                    o.store(gv * ttl.math.sigmoid(gv) * uv * wv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with gate_dfb.reserve() as blk:
                    tx = ttl.copy(gate[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with up_dfb.reserve() as blk:
                    tx = ttl.copy(up[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with weight_dfb.reserve() as blk:
                    tx = ttl.copy(weight[row, sc:sc + ELEM_GRAN], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()
