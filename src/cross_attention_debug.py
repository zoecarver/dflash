"""Debug: just Q@K^T matmul, output the scores directly."""
import ttl

TILE = 32


def make_scores_only_kernel(n_heads, hdim_tiles, kv_tiles):
    """Just compute Q @ K^T -> scores. No softmax."""
    @ttl.operation(grid=(1, 1))
    def scores_kernel(q, k_t, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(n_heads, hdim_tiles), buffer_factor=1)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, kv_tiles), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(n_heads, kv_tiles), buffer_factor=1)

        @ttl.compute()
        def compute():
            with q_dfb.wait() as q_blk, kt_dfb.wait() as kt_blk, out_dfb.reserve() as o:
                o.store(q_blk @ kt_blk)

        @ttl.datamovement()
        def dm_read():
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:n_heads, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:kv_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:n_heads, 0:kv_tiles]); tx.wait()

    return scores_kernel


def make_softmax_kernel(n_heads, kv_tiles):
    """Softmax on (n_heads, kv_tiles) score matrix. dims=[1] reduction."""
    @ttl.operation(grid=(1, 1))
    def softmax_kernel(scores, scaler, out):
        s_dfb = ttl.make_dataflow_buffer_like(scores, shape=(n_heads, kv_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(n_heads, kv_tiles), buffer_factor=1)

        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(n_heads, 1), buffer_factor=1)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scores, shape=(n_heads, kv_tiles), buffer_factor=1)
        exp_dfb = ttl.make_dataflow_buffer_like(scores, shape=(n_heads, kv_tiles), buffer_factor=1)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(n_heads, 1), buffer_factor=1)
        sum_recip_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(n_heads, 1), buffer_factor=1)
        sum_bc_dfb = ttl.make_dataflow_buffer_like(scores, shape=(n_heads, kv_tiles), buffer_factor=1)

        @ttl.compute()
        def compute():
            # Max
            with s_dfb.wait() as s, scaler_dfb.wait() as sc, max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(s, sc, dims=[1]))
            with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mx, mxb, dims=[1]))
            # Exp(s - max)
            with s_dfb.wait() as s2, max_bc_dfb.wait() as mxb, exp_dfb.reserve() as e:
                e.store(ttl.math.exp(s2 - mxb))
            # Sum
            with exp_dfb.wait() as e, scaler_dfb.wait() as sc2, sum_dfb.reserve() as sm:
                sm.store(ttl.math.reduce_sum(e, sc2, dims=[1]))
            with sum_dfb.wait() as sm, sum_recip_dfb.reserve() as sr:
                sr.store(ttl.math.recip(sm))
            with sum_recip_dfb.wait() as sr, sum_bc_dfb.reserve() as sbc:
                sbc.store(ttl.math.broadcast(sr, sbc, dims=[1]))
            # But exp was consumed! Need to recompute or we just output sum_bc for debugging.
            # Let's just output sum_bc (1/sum broadcast) to verify the reduction pipeline.
            with sum_bc_dfb.wait() as sbc, out_dfb.reserve() as o:
                o.store(sbc)

        @ttl.datamovement()
        def dm_read():
            for _ in range(2):
                with s_dfb.reserve() as blk:
                    tx = ttl.copy(scores[0:n_heads, 0:kv_tiles], blk); tx.wait()
            for _ in range(2):
                with scaler_dfb.reserve() as blk:
                    tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:n_heads, 0:kv_tiles]); tx.wait()

    return softmax_kernel
