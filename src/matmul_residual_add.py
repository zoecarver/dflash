"""Fused matmul + residual add: out = (a @ w) + residual.

Streams through the K (inner) dimension in tiles, accumulating partial
matmul results in L1. The intermediate matmul output never touches DRAM.

Processes NCOLS output columns per work unit for A-tile reuse within
each unit. Output tiles distributed across cores via grid="auto".
"""
import ttl

TILE = 32
NCOLS = 8


def make_matmul_residual_add_kernel(k_tiles):

    @ttl.operation(grid="auto")
    def matmul_residual_add(a, w, residual, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = a.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        col_groups = n_tiles // NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, NCOLS), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, NCOLS), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    # First K step: seed the accumulator
                    with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                        mm.store(av @ wv)
                    with mm_dfb.wait() as mv, acc_dfb.reserve() as acc:
                        acc.store(mv)

                    # Remaining K steps: accumulate
                    for k in range(1, k_tiles):
                        with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                            mm.store(av @ wv)
                        with mm_dfb.wait() as mv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
                            new.store(old + mv)

                    # Fuse residual add
                    with acc_dfb.wait() as final, res_dfb.wait() as rv, out_dfb.reserve() as o:
                        o.store(final + rv)

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
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[m, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(w[k, sc:sc + NCOLS], blk); tx.wait()
                    with res_dfb.reserve() as blk:
                        tx = ttl.copy(residual[m, sc:sc + NCOLS], blk); tx.wait()

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

    return matmul_residual_add
