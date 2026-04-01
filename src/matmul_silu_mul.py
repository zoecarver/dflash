"""Fused gate/up matmul + SiLU * mul: out = silu(a @ gw) * (a @ uw).

Two matmuls share the same input `a`, streaming through K. The gate and
up intermediate tensors never touch DRAM. Per output column group, we
accumulate both matmuls in parallel, then apply silu * mul.
"""
import ttl

TILE = 32
NCOLS = 8


def make_matmul_silu_mul_kernel(k_tiles):

    @ttl.operation(grid="auto")
    def matmul_silu_mul(a, gw, uw, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = a.shape[0] // TILE
        n_tiles = gw.shape[1] // TILE
        col_groups = n_tiles // NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        gw_dfb = ttl.make_dataflow_buffer_like(gw, shape=(1, NCOLS), buffer_factor=2)
        uw_dfb = ttl.make_dataflow_buffer_like(uw, shape=(1, NCOLS), buffer_factor=2)
        gmm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        umm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        gacc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        uacc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    # First K step: seed both accumulators
                    with a_dfb.wait() as av, gw_dfb.wait() as gv, gmm_dfb.reserve() as gm:
                        gm.store(av @ gv)
                    with gmm_dfb.wait() as gmv, gacc_dfb.reserve() as ga:
                        ga.store(gmv)

                    with a_dfb.wait() as av, uw_dfb.wait() as uv, umm_dfb.reserve() as um:
                        um.store(av @ uv)
                    with umm_dfb.wait() as umv, uacc_dfb.reserve() as ua:
                        ua.store(umv)

                    # Remaining K steps: accumulate both
                    for k in range(1, k_tiles):
                        with a_dfb.wait() as av, gw_dfb.wait() as gv, gmm_dfb.reserve() as gm:
                            gm.store(av @ gv)
                        with gmm_dfb.wait() as gmv, gacc_dfb.wait() as old, gacc_dfb.reserve() as new:
                            new.store(old + gmv)

                        with a_dfb.wait() as av, uw_dfb.wait() as uv, umm_dfb.reserve() as um:
                            um.store(av @ uv)
                        with umm_dfb.wait() as umv, uacc_dfb.wait() as old, uacc_dfb.reserve() as new:
                            new.store(old + umv)

                    # Fuse silu(gate) * up
                    with gacc_dfb.wait() as gate, uacc_dfb.wait() as up, out_dfb.reserve() as o:
                        o.store(gate * ttl.math.sigmoid(gate) * up)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    m = t // col_groups
                    g = t % col_groups
                    sc = g * NCOLS
                    for k in range(k_tiles):
                        # A tile read once, used for both gate and up matmuls
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[m, k], blk); tx.wait()
                        with gw_dfb.reserve() as blk:
                            tx = ttl.copy(gw[k, sc:sc + NCOLS], blk); tx.wait()
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[m, k], blk); tx.wait()
                        with uw_dfb.reserve() as blk:
                            tx = ttl.copy(uw[k, sc:sc + NCOLS], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    m = t // col_groups
                    g = t % col_groups
                    sc = g * NCOLS
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[m, sc:sc + NCOLS]); tx.wait()

    return matmul_silu_mul
