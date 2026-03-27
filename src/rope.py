"""RoPE kernel: applies rotary position embeddings to pre-normalized Q/K.

For each head, computes:
  out[j] = q[j] * cos[j] + q[rot_j] * sin_adj[j]

where rot_j = (j + head_tiles//2) % head_tiles maps the rotate_half pattern,
and sin_adj has the first half negated to encode the sign change.

Pre-compute sin_adj during weight loading:
  sin_adj = sin.clone()
  sin_adj[:, :HDIM//2] = -sin_adj[:, :HDIM//2]
"""
import ttl

TILE = 32


def make_rope_kernel(head_tiles, n_heads):
    half = head_tiles // 2

    @ttl.kernel(grid="auto")
    def rope_kernel(q, cos_tab, sin_adj_tab, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = q.shape[0] // TILE
        total = seq_tiles * n_heads
        units_per_core = -(-total // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, 1), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_adj_tab, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(units_per_core):
                unit = core_x * units_per_core + local_t
                if unit < total:
                    for j in range(head_tiles):
                        with q_dfb.wait() as qv, qr_dfb.wait() as qrv, cos_dfb.wait() as cv, sin_dfb.wait() as sv, out_dfb.reserve() as o:
                            o.store(qv * cv + qrv * sv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(units_per_core):
                unit = core_x * units_per_core + local_t
                if unit < total:
                    row = unit // n_heads
                    head = unit % n_heads
                    hc = head * head_tiles

                    for j in range(head_tiles):
                        rot_j = (j + half) % head_tiles
                        with q_dfb.reserve() as blk:
                            tx = ttl.copy(q[row, hc + j], blk); tx.wait()
                        with qr_dfb.reserve() as blk:
                            tx = ttl.copy(q[row, hc + rot_j], blk); tx.wait()
                        with cos_dfb.reserve() as blk:
                            tx = ttl.copy(cos_tab[row, j], blk); tx.wait()
                        with sin_dfb.reserve() as blk:
                            tx = ttl.copy(sin_adj_tab[row, j], blk); tx.wait()

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

    return rope_kernel
