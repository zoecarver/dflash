"""Per-head RMSNorm: normalizes each head independently within a packed QKV tensor.

Input Q is (sp, n_heads * HDIM) with heads packed contiguously along columns.
Each head's HDIM=128 columns (head_tiles=4 tiles) are normalized independently.
Weight is (TILE, HDIM), broadcast across all heads and sequence positions."""
import ttl

TILE = 32


def make_per_head_rmsnorm_kernel(head_tiles, n_heads, eps=1e-6):
    c_eps = eps

    @ttl.kernel(grid="auto")
    def per_head_rmsnorm(x, weight, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        total = seq_tiles * n_heads
        units_per_core = -(-total // grid_cols)

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
                for local_t in range(units_per_core):
                    unit = core_x * units_per_core + local_t
                    if unit < total:
                        # Pass 1: accumulate sum of squares across head_tiles
                        with x_dfb.wait() as x0:
                            with sq_dfb.reserve() as sq:
                                sq.store(x0 * x0)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)

                        for j in range(head_tiles - 1):
                            with x_dfb.wait() as xj:
                                with sq_dfb.reserve() as sq:
                                    sq.store(xj * xj)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                acc.store(av + rv)

                        # Compute rsqrt(mean(x^2) + eps)
                        with acc_dfb.wait() as total_v, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total_v, dims=[1]))
                        with bcast_dfb.wait() as bc_val, istd_dfb.reserve() as istd:
                            istd.store(ttl.math.rsqrt(bc_val * ms + ttl.math.fill(bc_val, c_eps)))

                        # Pass 2: normalize and scale
                        with istd_dfb.wait() as inv_std:
                            for j in range(head_tiles):
                                with x_dfb.wait() as xj, w_dfb.wait() as wj, out_dfb.reserve() as o:
                                    o.store(xj * inv_std * wj)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()

            for local_t in range(units_per_core):
                unit = core_x * units_per_core + local_t
                if unit < total:
                    row = unit // n_heads
                    head = unit % n_heads
                    hc = head * head_tiles

                    # Pass 1: x tiles for sum-of-squares
                    for j in range(head_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, hc + j], blk); tx.wait()
                    # Pass 2: x tiles + weight tiles
                    for j in range(head_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, hc + j], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(weight[0, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(units_per_core):
                unit = core_x * units_per_core + local_t
                if unit < total:
                    row = unit // n_heads
                    head = unit % n_heads
                    hc = head * head_tiles
                    for j in range(head_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[row, hc + j]); tx.wait()

    return per_head_rmsnorm
