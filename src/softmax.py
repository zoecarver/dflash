"""Fused row-wise softmax kernel (no matmul, pure elementwise/reduce/broadcast).

3-pass streaming softmax over column tiles:
  Pass 1: reduce_max per tile, accumulate global row max
  Pass 2: exp(s - broadcast(max)), reduce_sum per tile, accumulate
  Pass 3: exp(s - broadcast(max)) * broadcast(1/sum), write output

Input/output: (n_row_blocks, col_tiles) in tile units.
Softmax is along dim=1 (columns).
"""
import ttl

TILE = 32


def make_softmax_kernel(n_rows, col_tiles):
    bf = 2

    @ttl.operation(grid=(1, 1))
    def softmax(inp, scaler, out):
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        row_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        max_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                for row in range(n_rows):
                    # Pass 1: streaming row max
                    with inp_dfb.wait() as tile, max_acc_dfb.reserve() as mx:
                        mx.store(ttl.math.reduce_max(tile, sc, dims=[1]))
                    for c in range(1, col_tiles):
                        with inp_dfb.wait() as tile, row_max_dfb.reserve() as rm:
                            rm.store(ttl.math.reduce_max(tile, sc, dims=[1]))
                        with row_max_dfb.wait() as rmv, max_acc_dfb.wait() as old, max_acc_dfb.reserve() as new:
                            new.store(ttl.math.max(old, rmv))

                    # Broadcast max for passes 2 and 3
                    with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                        mxbc.store(ttl.math.broadcast(mx, mxbc, dims=[1]))
                        mx_keep.store(mx)

                    # Pass 2: streaming sum of exp(s - max)
                    with inp_dfb.wait() as tile, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                        e.store(ttl.math.exp(tile - mxbc))
                    with exp_dfb.wait() as ev, sum_acc_dfb.reserve() as sm:
                        sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                    for c in range(1, col_tiles):
                        with inp_dfb.wait() as tile:
                            with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                                mxbc.store(ttl.math.broadcast(mx, mxbc, dims=[1]))
                                mx_keep.store(mx)
                            with max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                                e.store(ttl.math.exp(tile - mxbc))
                        with exp_dfb.wait() as ev, exp_sum_dfb.reserve() as esum:
                            esum.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                        with exp_sum_dfb.wait() as esv, sum_acc_dfb.wait() as old, sum_acc_dfb.reserve() as new:
                            new.store(old + esv)

                    # 1/sum + broadcast
                    with sum_acc_dfb.wait() as sm, sinv_dfb.reserve() as si:
                        si.store(ttl.math.recip(sm))
                    with sinv_dfb.wait() as si, sinv_bc_dfb.reserve() as sibc:
                        sibc.store(ttl.math.broadcast(si, sibc, dims=[1]))

                    # Pass 3: normalize and write
                    for c in range(col_tiles):
                        with inp_dfb.wait() as tile:
                            with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                                mxbc.store(ttl.math.broadcast(mx, mxbc, dims=[1]))
                                mx_keep.store(mx)
                            with max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                                e.store(ttl.math.exp(tile - mxbc))
                        with exp_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, out_dfb.reserve() as o, sinv_bc_dfb.reserve() as sibc_keep:
                            o.store(ev * sibc)
                            sibc_keep.store(sibc)

                    # Drain accumulators for next row
                    with max_acc_dfb.wait() as _:
                        pass
                    with sinv_bc_dfb.wait() as _:
                        pass

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for row in range(n_rows):
                # 3 passes, each reads all col_tiles
                for _pass in range(3):
                    for c in range(col_tiles):
                        with inp_dfb.reserve() as blk:
                            tx = ttl.copy(inp[row:row+1, c:c+1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            for row in range(n_rows):
                for c in range(col_tiles):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row:row+1, c:c+1]); tx.wait()

    return softmax
