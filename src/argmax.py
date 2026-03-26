"""TT-Lang argmax kernel: column index of maximum value per row.

Two-phase streaming:
  Phase 1 - reduce_max across column tiles to find per-row maximum value.
  Phase 2 - stream again, use ttl.where to create exact one-hot mask at max
             positions, multiply by column indices, reduce_sum to get argmax.

bf16 limitation: column indices > 256 lose precision due to bf16 rounding.
For vocab=152K, returned indices may be off by up to ~512.
"""
import ttl

TILE = 32


def make_argmax_kernel(col_tiles):
    """Create argmax kernel for given column tile count.

    Args:
        col_tiles: columns // TILE
    Returns:
        kernel(x, indices, scaler, out)
          x:       (R, C) input
          indices: (1, C) column indices (value at position j = j)
          scaler:  (TILE, TILE) all 1.0s
          out:     (R, TILE) argmax index per row in column 0
    """

    @ttl.kernel(grid="auto")
    def argmax_kernel(x, indices, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        row_tiles = x.shape[0] // TILE
        rows_per_core = -(-row_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        idx_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        # Accumulator for running max (phase 1) and running index sum (phase 2)
        acc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        # Temp for per-tile reduction results
        red_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        # Stores broadcast row-max for phase 2 (held across all phase 2 iterations)
        maxb_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc:
                for local_r in range(rows_per_core):
                    row = core_x * rows_per_core + local_r
                    if row < row_tiles:
                        # === Phase 1: find row-wise max ===
                        with x_dfb.wait() as tile, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_max(tile, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)

                        for c in range(1, col_tiles):
                            with x_dfb.wait() as tile, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_max(tile, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as old, acc_dfb.reserve() as acc:
                                acc.store(ttl.math.max(old, rv))

                        # Broadcast max to full tile shape for phase 2
                        with acc_dfb.wait() as max_scalar, maxb_dfb.reserve() as mb:
                            mb.store(ttl.math.broadcast(max_scalar, dims=[1]))

                        # === Phase 2: find index via mask * indices ===
                        with maxb_dfb.wait() as row_max:
                            # First tile
                            with x_dfb.wait() as tile, idx_dfb.wait() as idx:
                                diff = tile - row_max
                                zeros = ttl.math.fill(diff, 0.0)
                                ones = ttl.math.fill(diff, 1.0)
                                mask = ttl.where(diff, zeros, ones)
                                with red_dfb.reserve() as partial:
                                    partial.store(ttl.math.reduce_sum(mask * idx, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                                acc.store(rv)

                            for c in range(1, col_tiles):
                                with x_dfb.wait() as tile, idx_dfb.wait() as idx:
                                    diff = tile - row_max
                                    zeros = ttl.math.fill(diff, 0.0)
                                    ones = ttl.math.fill(diff, 1.0)
                                    mask = ttl.where(diff, zeros, ones)
                                    with red_dfb.reserve() as partial:
                                        partial.store(ttl.math.reduce_sum(mask * idx, sc, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as old, acc_dfb.reserve() as acc:
                                    acc.store(old + rv)

                        with acc_dfb.wait() as final, out_dfb.reserve() as o:
                            o.store(final)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()

            for local_r in range(rows_per_core):
                row = core_x * rows_per_core + local_r
                if row < row_tiles:
                    # Phase 1: stream x tiles
                    for c in range(col_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, c], blk); tx.wait()
                    # Phase 2: stream x tiles + index tiles
                    for c in range(col_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, c], blk); tx.wait()
                        with idx_dfb.reserve() as blk:
                            tx = ttl.copy(indices[0, c], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_r in range(rows_per_core):
                row = core_x * rows_per_core + local_r
                if row < row_tiles:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, 0]); tx.wait()

    return argmax_kernel
