"""Test each pass of cross-attention in isolation for 1 head, 1 KV tile.

Pass 1: Q @ K^T -> reduce_max -> row_max
Pass 2: Q @ K^T -> exp(s - broadcast(max)) -> reduce_sum -> recip -> broadcast(1/sum)
Pass 3: Q @ K^T -> exp(s - broadcast(max)) * broadcast(1/sum) -> weight V

Each pass is a separate kernel to isolate which pass produces wrong results.
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE


def _p(t):
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def to_dev(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# === Pass 1 kernel: Q @ K^T -> scores ===
@ttl.kernel(grid=(1, 1))
def pass1_matmul(q, k_t, out):
    """Just do Q @ K^T and write the scores tile."""
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
    kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with q_dfb.wait() as qb, kt_dfb.wait() as kc, out_dfb.reserve() as o:
            o.store(qb @ kc)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
        with kt_dfb.reserve() as blk:
            tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# === Pass 1b: scores -> reduce_max ===
@ttl.kernel(grid=(1, 1))
def pass1b_reduce_max(scores, scaler, out):
    s_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with s_dfb.wait() as sv, sc_dfb.wait() as sc, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_max(sv, sc, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scores[0:1, 0:1], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# === Full pass 1: Q @ K^T -> reduce_max (fused) ===
@ttl.kernel(grid=(1, 1))
def pass1_fused(q, k_t, scaler, out):
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
    kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with q_dfb.wait() as qb, kt_dfb.wait() as kc, sc_dfb.wait() as sc:
            with s_dfb.reserve() as s:
                s.store(qb @ kc)
            with s_dfb.wait() as sv, out_dfb.reserve() as o:
                o.store(ttl.math.reduce_max(sv, sc, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
        with kt_dfb.reserve() as blk:
            tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# === Full 3-pass single-tile attention (1 head, 1 kv tile) ===
@ttl.kernel(grid=(1, 1))
def full_single_tile(q, k_t, v, scaler, out):
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
    kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, HDIM_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HDIM_TILES), buffer_factor=2)

    s1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    s2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    s3_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    maxbc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    sinvbc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    norm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=8)
    normbc_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, HDIM_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            # Pass 1: Q @ K^T -> reduce_max
            with q_dfb.wait() as qb:
                with kt_dfb.wait() as kc, s1_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s1_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))

            # Broadcast max
            with max_dfb.wait() as mx, maxbc_dfb.reserve() as mxbc:
                mxbc.store(ttl.math.broadcast(mx, dims=[1]))

            # Pass 2: Q @ K^T -> exp(s - max) -> reduce_sum
            with q_dfb.wait() as qb:
                with kt_dfb.wait() as kc, s2_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s2_dfb.wait() as sv, maxbc_dfb.wait() as mxbc, exp_dfb.reserve() as es:
                    es.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as esv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(esv, sc, dims=[1]))

            # 1/sum -> broadcast
            with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                si.store(ttl.math.recip(sm))
            with sinv_dfb.wait() as si, sinvbc_dfb.reserve() as sibc:
                sibc.store(ttl.math.broadcast(si, dims=[1]))

            # Pass 3: Q @ K^T -> exp(s - max) * (1/sum) -> weight V
            with q_dfb.wait() as qb:
                with kt_dfb.wait() as kc, s3_dfb.reserve() as s:
                    s.store(qb @ kc)
                # Need max again - re-broadcast since we consumed it
                # Actually we need max_dfb again. Let me keep it.

            # Hmm, we consumed max in pass 2 broadcast. Need to re-derive or keep.
            # Let me restructure: keep max across passes.

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
        # Pass 1
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
        with kt_dfb.reserve() as blk:
            tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()
        # Pass 2
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
        with kt_dfb.reserve() as blk:
            tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()
        # Pass 3
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
        with kt_dfb.reserve() as blk:
            tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(v[0:1, 0:HDIM_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:HDIM_TILES]); tx.wait()


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        torch.manual_seed(42)
        scale = 1.0 / (HDIM ** 0.5)

        q_raw = torch.randn(1, TILE, HDIM) * 0.1
        k_raw = torch.randn(1, TILE, HDIM) * 0.1
        v_raw = torch.randn(1, TILE, HDIM) * 0.1

        q_scaled = (q_raw.squeeze(0) * scale).to(torch.bfloat16)
        k_t = k_raw.squeeze(0).T.contiguous().to(torch.bfloat16)
        v_bf = v_raw.squeeze(0).to(torch.bfloat16)

        # Reference
        scores_ref = q_scaled.float() @ k_t.float()
        row_max_ref = scores_ref.max(dim=1, keepdim=True).values
        exp_ref = torch.exp(scores_ref - row_max_ref)
        sum_ref = exp_ref.sum(dim=1, keepdim=True)
        probs_ref = exp_ref / sum_ref
        out_ref = probs_ref @ v_bf.float()

        # TT tensors
        q_tt = to_dev(q_scaled.float(), d)
        kt_tt = to_dev(k_t.float(), d)
        v_tt = to_dev(v_bf.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # Test A: matmul only
        print("=== Test A: Q @ K^T (matmul only) ===")
        scores_tt = to_dev(torch.zeros(TILE, TILE), d)
        pass1_matmul(q_tt, kt_tt, scores_tt)
        scores_result = ttnn.to_torch(scores_tt).float()[:TILE, :TILE]
        p = pcc(scores_ref, scores_result)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}")

        # Test B: reduce_max on reference scores (no matmul)
        print("\n=== Test B: reduce_max(scores) ===")
        scores_input = to_dev(scores_ref, d)
        max_tt = to_dev(torch.zeros(TILE, TILE), d)
        pass1b_reduce_max(scores_input, sc_tt, max_tt)
        max_result = ttnn.to_torch(max_tt).float()[:TILE, :TILE]
        # reduce_max puts result in col 0
        max_col0 = max_result[:, 0:1]
        p = pcc(row_max_ref, max_col0)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}")

        # Test C: fused matmul -> reduce_max
        print("\n=== Test C: Q @ K^T -> reduce_max (fused) ===")
        max_tt2 = to_dev(torch.zeros(TILE, TILE), d)
        pass1_fused(q_tt, kt_tt, sc_tt, max_tt2)
        max_result2 = ttnn.to_torch(max_tt2).float()[:TILE, :TILE]
        max_col0_2 = max_result2[:, 0:1]
        p = pcc(row_max_ref, max_col0_2)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
