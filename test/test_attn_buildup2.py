"""Continue buildup from C5 (PCC=0.9941): test the V-weighting pipeline.

C5 outputs: exp(s - max) from 3rd matmul pass (1,1 tile)

C6: C5 + multiply by broadcast(1/sum) -> normalized weight (1,1)
C7: C6 + broadcast norm to (1, hdim_tiles)
C8: C7 + multiply by V row -> final output (1, hdim_tiles)
"""
import torch
import torch.nn.functional as F
import ttl
import ttnn

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE
BF = HDIM_TILES
S_BF = HDIM_TILES + 8


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


# C6: normalized weight = exp(s-max) * broadcast(1/sum)
def make_c6_kernel():
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        norm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1: max
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2: sum
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                    si.store(ttl.math.recip(sm))
                with sinv_dfb.wait() as si, sinv_bc_dfb.reserve() as sibc:
                    sibc.store(ttl.math.broadcast(si, dims=[1]))
                # Pass 3: exp(s-max) * (1/sum)
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, norm_dfb.reserve() as n:
                    n.store(ev * sibc)
            with norm_dfb.wait() as nv, out_dfb.reserve() as o:
                o.store(nv)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(3):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
                with kt_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


# C7: broadcast(norm) to (1, hdim_tiles)
def make_c7_kernel():
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        norm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        normbc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HDIM_TILES), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HDIM_TILES), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                    si.store(ttl.math.recip(sm))
                with sinv_dfb.wait() as si, sinv_bc_dfb.reserve() as sibc:
                    sibc.store(ttl.math.broadcast(si, dims=[1]))
                # Pass 3
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, norm_dfb.reserve() as n:
                    n.store(ev * sibc)
                with norm_dfb.wait() as nv, normbc_dfb.reserve() as nbc:
                    nbc.store(ttl.math.broadcast(nv, dims=[1]))
            with normbc_dfb.wait() as nbc, out_dfb.reserve() as o:
                o.store(nbc)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(3):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:HDIM_TILES], blk); tx.wait()
                with kt_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:HDIM_TILES, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:HDIM_TILES]); tx.wait()

    return k


# C8: normbc * V -> final output (1, hdim_tiles)
def make_c8_kernel():
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, v, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, HDIM_TILES), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(HDIM_TILES, 1), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, HDIM_TILES), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        norm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        normbc_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, HDIM_TILES), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HDIM_TILES), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                # Pass 1
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc, max_dfb.reserve() as mx_keep:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    mx_keep.store(mx)
                # Pass 2
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                    si.store(ttl.math.recip(sm))
                with sinv_dfb.wait() as si, sinv_bc_dfb.reserve() as sibc:
                    sibc.store(ttl.math.broadcast(si, dims=[1]))
                # Pass 3
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sinv_bc_dfb.wait() as sibc, norm_dfb.reserve() as n:
                    n.store(ev * sibc)
                with norm_dfb.wait() as nv, normbc_dfb.reserve() as nbc:
                    nbc.store(ttl.math.broadcast(nv, dims=[1]))
                with normbc_dfb.wait() as nbc, v_dfb.wait() as vr, out_dfb.reserve() as o:
                    o.store(nbc * vr)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(3):
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

    return k


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        torch.manual_seed(42)
        scale = 1.0 / (HDIM ** 0.5)

        q = torch.randn(TILE, HDIM) * 0.1
        kt = torch.randn(HDIM, TILE) * 0.1
        v = torch.randn(TILE, HDIM) * 0.1

        q_bf = (q * scale).to(torch.bfloat16)
        kt_bf = kt.to(torch.bfloat16)
        v_bf = v.to(torch.bfloat16)

        scores_ref = q_bf.float() @ kt_bf.float()
        row_max = scores_ref.max(dim=1, keepdim=True)[0]
        exp_ref = torch.exp(scores_ref - row_max)
        sum_ref = exp_ref.sum(dim=1, keepdim=True)
        probs_ref = exp_ref / sum_ref
        out_ref = probs_ref @ v_bf.float()

        q_tt = to_dev(q_bf.float(), d)
        kt_tt = to_dev(kt_bf.float(), d)
        v_tt = to_dev(v_bf.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # C6: normalized weight (1,1)
        print("=== C6: exp(s-max) * broadcast(1/sum) ===")
        out_tt = to_dev(torch.zeros(TILE, TILE), d)
        make_c6_kernel()(q_tt, kt_tt, sc_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :TILE]
        ref = probs_ref
        p = pcc(ref, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # C7: broadcast norm to (1, hdim_tiles)
        print("\n=== C7: broadcast(norm) -> (1, hdim_tiles) ===")
        out_tt = to_dev(torch.zeros(TILE, HDIM), d)
        make_c7_kernel()(q_tt, kt_tt, sc_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :HDIM]
        # Each col-tile should be a copy of probs col 0 broadcast
        # Actually: norm is probs for a single KV tile, broadcast(norm, dims=[1])
        # expands col-0 of the (1,1) prob tile across hdim_tiles tiles
        # So all hdim_tiles tiles should have same content = probs[:, 0:1] broadcast
        # Wait - norm_dfb is (1,1) = one tile. broadcast(nv, dims=[1]) into normbc_dfb
        # which is (1, hdim_tiles). This broadcasts the (32,32) tile's column values
        # across hdim_tiles tiles... actually broadcast(x, dims=[1]) on a (1,1)->
        # (1, hdim_tiles) should replicate the single tile hdim_tiles times.
        # Each output tile = broadcast of column 0 of input tile across all columns.
        # So each of the 4 output tiles = probs (since probs is a full 32x32 tile
        # and broadcast(dims=[1]) copies col 0 to all cols within each tile... wait.
        # Actually for a (1,1)->(1,N) broadcast, it just replicates the tile N times.
        # No, broadcast(dims=[1]) means expand dim 1. Input (1,1) = 1 tile with
        # per-row values in col 0. Output (1, hdim_tiles) = hdim_tiles tiles, each
        # being broadcast of col 0 across all columns.
        # So ref for each tile = probs[:, 0:1].expand(-1, 32)
        ref_col0 = probs_ref[:, 0:1].expand(-1, HDIM)
        p = pcc(ref_col0, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref_col0.min():.4f}, {ref_col0.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # C8: full output = normbc * V
        print("\n=== C8: normbc * V -> final output (1, hdim_tiles) ===")
        out_tt = to_dev(torch.zeros(TILE, HDIM), d)
        make_c8_kernel()(q_tt, kt_tt, v_tt, sc_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :HDIM]
        # This should be probs[:, 0:1].expand(-1, 128) * v_bf
        # But that's NOT the same as probs @ V !
        # probs @ V = sum over kv_dim of probs[i,j] * V[j,:]
        # With 1 KV tile: probs is (32, 32), V is (32, 128)
        # The kernel does: for each kv tile, compute norm_weight (scalar per row),
        # broadcast it, multiply by V row. With 1 kv tile, this is:
        # norm = probs (32x32 tile), broadcast col0 -> (32, 128), multiply V
        # = probs[:, 0:1] * V  ... which is WRONG!
        #
        # The correct result is probs @ V (matrix multiply, not elementwise!)
        # But the kernel does elementwise multiply of broadcast(probs_col0) * V.
        #
        # THIS IS THE BUG. The kernel treats each KV tile's attention weight as
        # a single scalar per row, but with a (32,32) score tile, each row has
        # 32 attention weights (one per KV position in that tile).
        # We need a (1,1)@(1,hdim_tiles) matmul to properly weight V, not
        # an elementwise multiply.
        ref_wrong = probs_ref[:, 0:1].expand(-1, HDIM) * v_bf.float()
        p_wrong = pcc(ref_wrong, r)
        p_correct = pcc(out_ref, r)
        print(f"  PCC vs correct (probs @ V):     {p_correct:.4f}")
        print(f"  PCC vs wrong (bc(prob)*V):       {p_wrong:.4f}")
        print(f"  ref range: [{out_ref.min():.4f}, {out_ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
