"""RMSNorm kernel: out = x * rsqrt(mean(x^2) + eps) * weight.

Two-pass streaming: accumulate sum-of-squares, then normalize + scale.
Factory function parameterized by dim_tiles (tiles in reduction dimension)."""
import ttl

TILE = 32


def make_rmsnorm_kernel(dim_tiles, eps=1e-6):
    c_eps = eps

    @ttl.kernel(grid="auto")
    def rmsnorm_kernel(x, weight, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)

        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: accumulate sum of squares
                        with x_dfb.wait() as x0:
                            with sq_dfb.reserve() as sq:
                                sq.store(x0 * x0)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)

                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj:
                                with sq_dfb.reserve() as sq:
                                    sq.store(xj * xj)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                acc.store(av + rv)

                        # Compute rsqrt(mean(x^2) + eps)
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as bc_val, istd_dfb.reserve() as istd:
                            istd.store(ttl.math.rsqrt(bc_val * ms + ttl.math.fill(bc_val, c_eps)))

                        # Pass 2: normalize and scale
                        with istd_dfb.wait() as inv_std:
                            for j in range(dim_tiles):
                                with x_dfb.wait() as xj, w_dfb.wait() as wj, out_dfb.reserve() as o:
                                    o.store(xj * inv_std * wj)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: x tiles for sum-of-squares
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 2: x tiles + weight tiles for normalize
                    # Weight is (1, dim_tiles) -- same for all sequence positions
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(weight[0, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()

    return rmsnorm_kernel
