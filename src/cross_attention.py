"""Fused cross-attention kernel for DiT/DFlash models.

GQA handled implicitly: Q heads stacked along dim 0, single KV head.
Processes one head at a time with online softmax, streaming one KV tile
at a time. All matmuls produce single-tile outputs (N=1).

Per head, three passes through KV tiles:
  Pass 1: Q_h @ k_col -> (1,1) scores, accumulate row max
  Pass 2: Q_h @ k_col -> exp(s - broadcast(max)), accumulate row sum
  Pass 3: Q_h @ k_col -> exp(s - broadcast(max)) * broadcast(1/sum), weight V, accumulate

Layout:
    Q:      (n_heads * TILE, HDIM) -- heads stacked, pre-scaled by 1/sqrt(head_dim)
    K^T:    (HDIM, kv_len_padded) -- pre-transposed
    V:      (kv_len_padded, HDIM)
    scaler: (TILE, TILE) -- tile of 1.0s for reductions
    out:    (n_heads * TILE, HDIM)
"""
import ttl

TILE = 32


def make_cross_attention_kernel(n_heads, hdim_tiles, kv_tiles):
    # Matmul (1,K)@(K,1) compiler bug: writes K tiles at write_ptr offset.
    # scores_dfb needs extra space so overflow doesn't corrupt adjacent CBs.
    scores_bf = hdim_tiles + 8
    # All other (1,1) DFBs need hdim_tiles to survive as neighbors
    bf = hdim_tiles

    @ttl.kernel(grid=(1, 1))
    def cross_attention(q, k_t, v, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_col_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        v_row_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, hdim_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)

        scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=scores_bf)
        row_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        max_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        norm_bc_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, hdim_tiles), buffer_factor=2)
        pv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)
        out_acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            with scaler_dfb.wait() as sc:
                for head in range(n_heads):
                    # ====== Pass 1: streaming row max ======
                    with q_dfb.wait() as qb:
                        with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                            s.store(qb @ kc)
                        with scores_dfb.wait() as sv, max_acc_dfb.reserve() as mx:
                            mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                        for kv_idx in range(1, kv_tiles):
                            with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                                s.store(qb @ kc)
                            with scores_dfb.wait() as sv, row_max_dfb.reserve() as rm:
                                rm.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                            with row_max_dfb.wait() as rmv, max_acc_dfb.wait() as old, max_acc_dfb.reserve() as new:
                                new.store(ttl.math.max(old, rmv))

                    # Broadcast max for use in Pass 2 and 3
                    with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                        mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                        mx_keep.store(mx)

                    # ====== Pass 2: streaming sum of exp(s - broadcast(max)) ======
                    with q_dfb.wait() as qb:
                        with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                            s.store(qb @ kc)
                        with scores_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_scores_dfb.reserve() as es:
                            es.store(ttl.math.exp(sv - mxbc))
                        with exp_scores_dfb.wait() as esv, sum_acc_dfb.reserve() as sm:
                            sm.store(ttl.math.reduce_sum(esv, sc, dims=[1]))
                        for kv_idx in range(1, kv_tiles):
                            with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                                s.store(qb @ kc)
                            # Re-broadcast max for each tile
                            with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                                mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                                mx_keep.store(mx)
                            with scores_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_scores_dfb.reserve() as es:
                                es.store(ttl.math.exp(sv - mxbc))
                            with exp_scores_dfb.wait() as esv, exp_sum_dfb.reserve() as esum:
                                esum.store(ttl.math.reduce_sum(esv, sc, dims=[1]))
                            with exp_sum_dfb.wait() as ev, sum_acc_dfb.wait() as old, sum_acc_dfb.reserve() as new:
                                new.store(old + ev)

                    # ====== 1/sum + broadcast ======
                    with sum_acc_dfb.wait() as sm, exp_sum_dfb.reserve() as sinv:
                        sinv.store(ttl.math.recip(sm))
                    with exp_sum_dfb.wait() as sinv, sinv_bc_dfb.reserve() as sibc:
                        sibc.store(ttl.math.broadcast(sinv, dims=[1]))

                    # ====== Pass 3: weighted V accumulation ======
                    with q_dfb.wait() as qb:
                        # First KV tile: init output acc
                        with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                            s.store(qb @ kc)
                        with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                            mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                            mx_keep.store(mx)
                        with scores_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_scores_dfb.reserve() as e:
                            e.store(ttl.math.exp(sv - mxbc))
                        with exp_scores_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, exp_sum_dfb.reserve() as n, sinv_bc_dfb.reserve() as sibc_keep:
                            n.store(ev * sibc)
                            sibc_keep.store(sibc)
                        with exp_sum_dfb.wait() as nv, norm_bc_dfb.reserve() as nbc:
                            nbc.store(ttl.math.broadcast(nv, dims=[1]))
                        with norm_bc_dfb.wait() as nbc, v_row_dfb.wait() as vr, out_acc_dfb.reserve() as oa:
                            oa.store(nbc * vr)

                        # Remaining KV tiles: accumulate
                        for kv_idx in range(1, kv_tiles):
                            with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                                s.store(qb @ kc)
                            with max_acc_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_acc_dfb.reserve() as mx_keep:
                                mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                                mx_keep.store(mx)
                            with scores_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_scores_dfb.reserve() as e:
                                e.store(ttl.math.exp(sv - mxbc))
                            with exp_scores_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, exp_sum_dfb.reserve() as n, sinv_bc_dfb.reserve() as sibc_keep:
                                n.store(ev * sibc)
                                sibc_keep.store(sibc)
                            with exp_sum_dfb.wait() as nv, norm_bc_dfb.reserve() as nbc:
                                nbc.store(ttl.math.broadcast(nv, dims=[1]))
                            with norm_bc_dfb.wait() as nbc, v_row_dfb.wait() as vr, pv_dfb.reserve() as pv:
                                pv.store(nbc * vr)
                            with pv_dfb.wait() as pvv, out_acc_dfb.wait() as old, out_acc_dfb.reserve() as new:
                                new.store(old + pvv)

                    with out_acc_dfb.wait() as final, out_dfb.reserve() as o:
                        o.store(final)
                    # Drain leftover accumulators for next head
                    with max_acc_dfb.wait() as _mx:
                        pass
                    with sinv_bc_dfb.wait() as _si:
                        pass

        @ttl.datamovement()
        def dm_read():
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for head in range(n_heads):
                # Pass 1: Q + K^T cols
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[head:head+1, 0:hdim_tiles], blk); tx.wait()
                for kv_idx in range(kv_tiles):
                    with kt_col_dfb.reserve() as blk:
                        tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
                # Pass 2: Q + K^T cols
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[head:head+1, 0:hdim_tiles], blk); tx.wait()
                for kv_idx in range(kv_tiles):
                    with kt_col_dfb.reserve() as blk:
                        tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
                # Pass 3: Q + K^T cols interleaved with V rows
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[head:head+1, 0:hdim_tiles], blk); tx.wait()
                for kv_idx in range(kv_tiles):
                    with kt_col_dfb.reserve() as blk:
                        tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
                    with v_row_dfb.reserve() as blk:
                        tx = ttl.copy(v[kv_idx:kv_idx+1, 0:hdim_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            for head in range(n_heads):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[head:head+1, 0:hdim_tiles]); tx.wait()

    return cross_attention
