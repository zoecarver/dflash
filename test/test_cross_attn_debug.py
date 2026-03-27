"""Incremental cross-attention debugging: build up piece by piece."""
import torch
import torch.nn.functional as F
import ttl
import ttnn

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE
NQH_TP = 8
KV_LEN = 96
KV_LEN_PAD = ((KV_LEN + TILE - 1) // TILE) * TILE
KV_TILES = KV_LEN_PAD // TILE


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


# ==========================================================================
# Level 1: Single matmul Q@K^T -> scores (1 head, 1 kv tile)
# ==========================================================================
def make_matmul_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def matmul_k(q, k_t, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with q_dfb.wait() as qb, kt_dfb.wait() as kc, out_dfb.reserve() as o:
                o.store(qb @ kc)

        @ttl.datamovement()
        def dm_read():
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return matmul_k


# ==========================================================================
# Level 2: Matmul + reduce_max (internal DFB)
# ==========================================================================
def make_matmul_reduce_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def matmul_reduce_k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=hdim_tiles)

        @ttl.compute()
        def compute():
            with q_dfb.wait() as qb, kt_dfb.wait() as kc, scores_dfb.reserve() as s:
                s.store(qb @ kc)
            with scores_dfb.wait() as sv, scaler_dfb.wait() as sc, out_dfb.reserve() as o:
                o.store(ttl.math.reduce_max(sv, sc, dims=[1]))

        @ttl.datamovement()
        def dm_read():
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return matmul_reduce_k


# ==========================================================================
# Level 3: Streaming max over multiple KV tiles (1 head)
# ==========================================================================
def make_streaming_max_kernel(hdim_tiles, kv_tiles):
    @ttl.kernel(grid=(1, 1))
    def streaming_max_k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_col_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=hdim_tiles)
        row_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        max_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with scaler_dfb.wait() as sc:
                # First KV tile: fresh Q
                with q_dfb.wait() as qb, kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                    s.store(qb @ kc)
                with scores_dfb.wait() as sv, max_acc_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                # Remaining KV tiles: fresh Q each time
                for kv_idx in range(1, kv_tiles):
                    with q_dfb.wait() as qb, kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with scores_dfb.wait() as sv, row_max_dfb.reserve() as rm:
                        rm.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                    with row_max_dfb.wait() as rmv, max_acc_dfb.wait() as old, max_acc_dfb.reserve() as new:
                        new.store(ttl.math.max(old, rmv))
                # Output the max
                with max_acc_dfb.wait() as mx, out_dfb.reserve() as o:
                    o.store(mx)

        @ttl.datamovement()
        def dm_read():
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            # Push Q fresh for each KV tile
            for kv_idx in range(kv_tiles):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
                with kt_col_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return streaming_max_k


# ==========================================================================
# Level 4: Full softmax-weighted V for 1 head (3 passes)
# ==========================================================================
def make_single_head_kernel(hdim_tiles, kv_tiles):
    @ttl.kernel(grid=(1, 1))
    def single_head_k(q, k_t, v, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_col_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        v_row_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, hdim_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)

        # All (1,1) DFBs need buffer_factor=hdim_tiles to absorb matmul overflow
        bf = hdim_tiles
        scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        row_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        max_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_scores_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        exp_sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        norm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=bf)
        norm_bc_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, hdim_tiles), buffer_factor=2)
        pv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)
        out_acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, hdim_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            with scaler_dfb.wait() as sc:
                # Pass 1: streaming row max
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

                # Pass 2: streaming sum of exp(s - max)
                with q_dfb.wait() as qb:
                    with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with scores_dfb.wait() as sv, max_acc_dfb.wait() as mx, exp_scores_dfb.reserve() as es, max_acc_dfb.reserve() as mx_keep:
                        es.store(ttl.math.exp(sv - mx))
                        mx_keep.store(mx)
                    with exp_scores_dfb.wait() as esv, sum_acc_dfb.reserve() as sm:
                        sm.store(ttl.math.reduce_sum(esv, sc, dims=[1]))
                    for kv_idx in range(1, kv_tiles):
                        with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                            s.store(qb @ kc)
                        with scores_dfb.wait() as sv, max_acc_dfb.wait() as mx, exp_scores_dfb.reserve() as es, max_acc_dfb.reserve() as mx_keep:
                            es.store(ttl.math.exp(sv - mx))
                            mx_keep.store(mx)
                        with exp_scores_dfb.wait() as esv, exp_sum_dfb.reserve() as esum:
                            esum.store(ttl.math.reduce_sum(esv, sc, dims=[1]))
                        with exp_sum_dfb.wait() as ev, sum_acc_dfb.wait() as old, sum_acc_dfb.reserve() as new:
                            new.store(old + ev)

                # 1/sum
                with sum_acc_dfb.wait() as sm, exp_sum_dfb.reserve() as sinv:
                    sinv.store(ttl.math.recip(sm))

                # Pass 3: weighted V accumulation
                with q_dfb.wait() as qb:
                    with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with scores_dfb.wait() as sv, max_acc_dfb.wait() as mx, exp_sum_dfb.wait() as sinv, norm_dfb.reserve() as n, max_acc_dfb.reserve() as mx_keep, exp_sum_dfb.reserve() as sinv_keep:
                        n.store(ttl.math.exp(sv - mx) * sinv)
                        mx_keep.store(mx)
                        sinv_keep.store(sinv)
                    with norm_dfb.wait() as nv, norm_bc_dfb.reserve() as nbc:
                        nbc.store(ttl.math.broadcast(nv, dims=[1]))
                    with norm_bc_dfb.wait() as nbc, v_row_dfb.wait() as vr, out_acc_dfb.reserve() as oa:
                        oa.store(nbc * vr)
                    for kv_idx in range(1, kv_tiles):
                        with kt_col_dfb.wait() as kc, scores_dfb.reserve() as s:
                            s.store(qb @ kc)
                        with scores_dfb.wait() as sv, max_acc_dfb.wait() as mx, exp_sum_dfb.wait() as sinv, norm_dfb.reserve() as n, max_acc_dfb.reserve() as mx_keep, exp_sum_dfb.reserve() as sinv_keep:
                            n.store(ttl.math.exp(sv - mx) * sinv)
                            mx_keep.store(mx)
                            sinv_keep.store(sinv)
                        with norm_dfb.wait() as nv, norm_bc_dfb.reserve() as nbc:
                            nbc.store(ttl.math.broadcast(nv, dims=[1]))
                        with norm_bc_dfb.wait() as nbc, v_row_dfb.wait() as vr, pv_dfb.reserve() as pv:
                            pv.store(nbc * vr)
                        with pv_dfb.wait() as pvv, out_acc_dfb.wait() as old, out_acc_dfb.reserve() as new:
                            new.store(old + pvv)

                with out_acc_dfb.wait() as final, out_dfb.reserve() as o:
                    o.store(final)

        @ttl.datamovement()
        def dm_read():
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            # Pass 1
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            for kv_idx in range(kv_tiles):
                with kt_col_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
            # Pass 2
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            for kv_idx in range(kv_tiles):
                with kt_col_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
            # Pass 3
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            for kv_idx in range(kv_tiles):
                with kt_col_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, kv_idx:kv_idx+1], blk); tx.wait()
                with v_row_dfb.reserve() as blk:
                    tx = ttl.copy(v[kv_idx:kv_idx+1, 0:hdim_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:hdim_tiles]); tx.wait()

    return single_head_k


# ==========================================================================
# Test runner
# ==========================================================================
def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        torch.manual_seed(42)
        scale = 1.0 / (HDIM ** 0.5)

        # Build test data: 1 head of Q (1 tile row), full K^T and V
        q_raw = torch.randn(1, TILE, HDIM) * 0.1
        k_raw = torch.randn(1, KV_LEN, HDIM) * 0.1
        v_raw = torch.randn(1, KV_LEN, HDIM) * 0.1

        q_padded = (q_raw.squeeze(0) * scale).to(torch.bfloat16)  # (32, 128)
        k_t_padded = F.pad(k_raw.squeeze(0).T.contiguous(), (0, KV_LEN_PAD - KV_LEN)).to(torch.bfloat16)  # (128, 96)
        v_padded = F.pad(v_raw.squeeze(0), (0, 0, 0, KV_LEN_PAD - KV_LEN)).to(torch.bfloat16)  # (96, 128)

        q_tt = to_dev(q_padded.float(), d)
        kt_tt = to_dev(k_t_padded.float(), d)
        v_tt = to_dev(v_padded.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # ---- Level 1: Single matmul ----
        print("=== Level 1: Single matmul Q@K^T (1 tile) ===")
        out1 = to_dev(torch.zeros(TILE, TILE), d)
        k1 = make_matmul_kernel(HDIM_TILES)
        k1(q_tt, kt_tt, out1)
        r1 = ttnn.to_torch(out1).float()
        ref1 = (q_padded.float() @ k_t_padded[:, :TILE].float())
        p1 = pcc(ref1, r1[:TILE, :TILE])
        print(f"  PCC={p1:.4f}")
        assert p1 > 0.98, f"Level 1 FAIL: PCC={p1}"
        print("  PASS")

        # ---- Level 2: Matmul + reduce_max ----
        print("\n=== Level 2: Matmul + reduce_max ===")
        out2 = to_dev(torch.zeros(TILE, TILE), d)
        k2 = make_matmul_reduce_kernel(HDIM_TILES)
        k2(q_tt, kt_tt, sc_tt, out2)
        r2 = ttnn.to_torch(out2).float()
        ref2_scores = q_padded.float() @ k_t_padded[:, :TILE].float()
        ref2 = ref2_scores.max(dim=1, keepdim=True)[0]  # (32, 1)
        # reduce dims=[1] collapses cols -> (1,1) tile, result per-row in col 0
        tt2 = r2[:TILE, :1]  # (32, 1)
        p2 = pcc(ref2, tt2)
        print(f"  PCC={p2:.4f}")
        print(f"  ref max range: [{ref2.min():.4f}, {ref2.max():.4f}]")
        print(f"  tt  max range: [{tt2.min():.4f}, {tt2.max():.4f}]")

        # ---- Level 3: Streaming max over KV tiles ----
        print(f"\n=== Level 3: Streaming max ({KV_TILES} KV tiles) ===")
        out3 = to_dev(torch.zeros(TILE, TILE), d)
        k3 = make_streaming_max_kernel(HDIM_TILES, KV_TILES)
        k3(q_tt, kt_tt, sc_tt, out3)
        r3 = ttnn.to_torch(out3).float()
        # Reference: max across all KV tiles
        all_scores = []
        for kvi in range(KV_TILES):
            s = q_padded.float() @ k_t_padded[:, kvi*TILE:(kvi+1)*TILE].float()
            all_scores.append(s)
        ref3_scores = torch.cat(all_scores, dim=1)
        ref3 = ref3_scores.max(dim=1, keepdim=True)[0]  # (32, 1)
        tt3 = r3[:TILE, :1]
        p3 = pcc(ref3, tt3)
        print(f"  PCC={p3:.4f}")
        print(f"  ref max range: [{ref3.min():.4f}, {ref3.max():.4f}]")
        print(f"  tt  max range: [{tt3.min():.4f}, {tt3.max():.4f}]")

        # ---- Level 4: Full single-head cross-attention ----
        # ---- Level 4a: Full attention with 1 KV tile (no accumulation) ----
        print(f"\n=== Level 4a: Full single-head attention (1 KV tile) ===")
        out4a = to_dev(torch.zeros(TILE, HDIM), d)
        k4a = make_single_head_kernel(HDIM_TILES, 1)
        # Use only first KV tile
        kt_1tile = to_dev(k_t_padded[:, :TILE].float(), d)
        v_1tile = to_dev(v_padded[:TILE, :].float(), d)
        k4a(q_tt, kt_1tile, v_1tile, sc_tt, out4a)
        r4a = ttnn.to_torch(out4a).float()
        # Reference for 1 KV tile
        s4a = q_padded.float() @ k_t_padded[:, :TILE].float()
        probs4a = torch.softmax(s4a, dim=-1)
        ref4a = probs4a @ v_padded[:TILE, :].float()
        p4a = pcc(ref4a[:TILE, :HDIM], r4a[:TILE, :HDIM])
        md4a = (ref4a[:TILE, :HDIM] - r4a[:TILE, :HDIM]).abs().max().item()
        print(f"  PCC={p4a:.4f} max_diff={md4a:.6f}")
        print("  PASS" if p4a > 0.95 else "  FAIL")

        # ---- Level 4: Full single-head attention (3 KV tiles) ----
        print(f"\n=== Level 4: Full single-head attention ({KV_TILES} KV tiles) ===")
        out4 = to_dev(torch.zeros(TILE, HDIM), d)
        k4 = make_single_head_kernel(HDIM_TILES, KV_TILES)
        k4(q_tt, kt_tt, v_tt, sc_tt, out4)
        r4 = ttnn.to_torch(out4).float()
        # Reference: full softmax attention
        probs = torch.softmax(ref3_scores, dim=-1)
        ref4 = probs @ v_padded.float()
        p4 = pcc(ref4[:TILE, :HDIM], r4[:TILE, :HDIM])
        md4 = (ref4[:TILE, :HDIM] - r4[:TILE, :HDIM]).abs().max().item()
        print(f"  PCC={p4:.4f} max_diff={md4:.6f}")
        print("  PASS" if p4 > 0.98 else "  FAIL")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
