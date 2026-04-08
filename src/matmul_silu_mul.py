"""Fused gate/up matmul + SiLU * mul: out = silu(a @ gw) * (a @ uw).

Loads full K dimension at once so hardware handles accumulation. A is
read once and reused for both gate and up matmuls. Reads are split
across both datamovement threads: dm_read loads a + gw, dm_write
loads uw and writes the output.
"""
import ttl

TILE = 32
TILE_BYTES = TILE * TILE * 2
L1_BUDGET = 1400000


def _pick_ncols(k_tiles):
    """Pick largest ncols that fits: a=(1,k), gw=(k,n), uw=(k,n), + 3 small."""
    for ncols in (8, 4, 2, 1):
        total = (k_tiles + 2 * k_tiles * ncols + 3 * ncols) * TILE_BYTES
        if total <= L1_BUDGET:
            return ncols
    return 1


def make_matmul_silu_mul_kernel(k_tiles):
    ncols = _pick_ncols(k_tiles)

    @ttl.operation(grid="auto")
    def matmul_silu_mul(a, gw, uw, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = a.shape[0] // TILE
        n_tiles = gw.shape[1] // TILE
        col_groups = n_tiles // ncols
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, k_tiles), buffer_factor=1)
        gw_dfb = ttl.make_dataflow_buffer_like(gw, shape=(k_tiles, ncols), buffer_factor=1)
        uw_dfb = ttl.make_dataflow_buffer_like(uw, shape=(k_tiles, ncols), buffer_factor=1)
        gate_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ncols), buffer_factor=1)
        up_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ncols), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ncols), buffer_factor=1)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    a_blk = a_dfb.wait()
                    gw_blk = gw_dfb.wait()
                    with gate_dfb.reserve() as g:
                        g.store(a_blk @ gw_blk)
                    gw_blk.pop()

                    uw_blk = uw_dfb.wait()
                    with up_dfb.reserve() as u:
                        u.store(a_blk @ uw_blk)
                    uw_blk.pop()
                    a_blk.pop()

                    with gate_dfb.wait() as gate, up_dfb.wait() as up, out_dfb.reserve() as o:
                        o.store(gate * ttl.math.sigmoid(gate) * up)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    m = t // col_groups
                    g = t % col_groups
                    sc = g * ncols
                    with a_dfb.reserve() as b1, gw_dfb.reserve() as b2:
                        tx1 = ttl.copy(a[m, 0:k_tiles], b1)
                        tx2 = ttl.copy(gw[0:k_tiles, sc:sc + ncols], b2)
                        tx1.wait(); tx2.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    m = t // col_groups
                    g = t % col_groups
                    sc = g * ncols
                    with uw_dfb.reserve() as blk:
                        tx = ttl.copy(uw[0:k_tiles, sc:sc + ncols], blk); tx.wait()
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[m, sc:sc + ncols]); tx.wait()

    return matmul_silu_mul
