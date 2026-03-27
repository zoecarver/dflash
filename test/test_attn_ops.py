"""Isolate each attention op: build up from matmul+reduce_max (Level 3 passes)."""
import torch
import torch.nn.functional as F
import ttl
import ttnn

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE
BF = HDIM_TILES  # buffer_factor for matmul overflow workaround


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
# Test A: matmul -> exp(scores) -> output
# ==========================================================================
def make_matmul_exp_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                s.store(qb @ kc)
            with s_dfb.wait() as sv, exp_dfb.reserve() as e:
                e.store(ttl.math.exp(sv))
            with exp_dfb.wait() as ev, out_dfb.reserve() as o:
                o.store(ev)

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

    return k


# ==========================================================================
# Test B: matmul -> reduce_max -> exp(scores - max) -> output
# (scores recomputed via 2nd matmul)
# ==========================================================================
def make_matmul_exp_sub_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1: matmul -> reduce_max
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                # Broadcast max to full tile
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2: matmul -> exp(score - broadcast_max)
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
            with exp_dfb.wait() as ev, out_dfb.reserve() as o:
                o.store(ev)
            with max_dfb.wait() as _:
                pass

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            # Pass 1
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()
            # Pass 2
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


# ==========================================================================
# Test C: matmul -> reduce_max -> exp(score-max) -> reduce_sum -> output
# ==========================================================================
def make_matmul_softmax_sum_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1: reduce_max + broadcast
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2: exp(score - broadcast_max) -> reduce_sum
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
            with sum_dfb.wait() as sv, out_dfb.reserve() as o:
                o.store(sv)
            with max_dfb.wait() as _:
                pass

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt_dfb.reserve() as blk:
                tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


# ==========================================================================
# Test D: full softmax probs -> output (no V multiply)
# Pass1: max, Pass2: exp-sum, compute 1/sum, Pass3: exp(s-max)*1/sum -> out
# ==========================================================================
def make_softmax_probs_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1: max + broadcast
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2: exp(s - broadcast_max) -> sum
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                # 1/sum + broadcast
                with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                    si.store(ttl.math.recip(sm))
                with sinv_dfb.wait() as si, sinv_bc_dfb.reserve() as sibc:
                    sibc.store(ttl.math.broadcast(si, dims=[1]))
                # Pass 3: exp(s - broadcast_max) * broadcast(1/sum)
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, out_dfb.reserve() as o:
                    o.store(ev * sibc)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(3):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
                with kt_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        torch.manual_seed(42)
        scale = 1.0 / (HDIM ** 0.5)

        q = torch.randn(TILE, HDIM) * 0.1
        kt = torch.randn(HDIM, TILE) * 0.1
        q_bf = (q * scale).to(torch.bfloat16)
        kt_bf = kt.to(torch.bfloat16)
        scores_ref = q_bf.float() @ kt_bf.float()

        q_tt = to_dev((q * scale).float(), d)
        kt_tt = to_dev(kt.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # ---- Test A: matmul -> exp ----
        print("=== Test A: matmul -> exp ===")
        out_a = to_dev(torch.zeros(TILE, TILE), d)
        make_matmul_exp_kernel(HDIM_TILES)(q_tt, kt_tt, out_a)
        r_a = ttnn.to_torch(out_a).float()[:TILE, :TILE]
        ref_a = torch.exp(scores_ref)
        pa = pcc(ref_a, r_a)
        print(f"  PCC={pa:.4f}")

        # ---- Test B: matmul -> reduce_max -> exp(s-max) ----
        print("\n=== Test B: matmul -> reduce_max -> exp(score - max) ===")
        out_b = to_dev(torch.zeros(TILE, TILE), d)
        make_matmul_exp_sub_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out_b)
        r_b = ttnn.to_torch(out_b).float()[:TILE, :TILE]
        row_max = scores_ref.max(dim=1, keepdim=True)[0]
        ref_b = torch.exp(scores_ref - row_max)
        pb = pcc(ref_b, r_b)
        print(f"  PCC={pb:.4f}")
        print(f"  ref range: [{ref_b.min():.4f}, {ref_b.max():.4f}]")
        print(f"  tt  range: [{r_b.min():.4f}, {r_b.max():.4f}]")

        # ---- Test C: matmul -> reduce_max -> exp(s-max) -> reduce_sum ----
        print("\n=== Test C: exp(s-max) -> reduce_sum ===")
        out_c = to_dev(torch.zeros(TILE, TILE), d)
        make_matmul_softmax_sum_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out_c)
        r_c = ttnn.to_torch(out_c).float()[:TILE, :1]
        ref_c = ref_b.sum(dim=1, keepdim=True)
        pc = pcc(ref_c, r_c)
        print(f"  PCC={pc:.4f}")
        print(f"  ref range: [{ref_c.min():.4f}, {ref_c.max():.4f}]")
        print(f"  tt  range: [{r_c.min():.4f}, {r_c.max():.4f}]")

        # ---- Test D: full softmax probs ----
        print("\n=== Test D: full softmax probs (3 passes) ===")
        out_d = to_dev(torch.zeros(TILE, TILE), d)
        make_softmax_probs_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out_d)
        r_d = ttnn.to_torch(out_d).float()[:TILE, :TILE]
        ref_d = torch.softmax(scores_ref, dim=-1)
        pd = pcc(ref_d, r_d)
        print(f"  PCC={pd:.4f}")
        print(f"  ref range: [{ref_d.min():.4f}, {ref_d.max():.4f}]")
        print(f"  tt  range: [{r_d.min():.4f}, {r_d.max():.4f}]")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
