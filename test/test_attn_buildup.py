"""Build up from Test C (passes PCC=0.95): add one op at a time."""
import torch
import torch.nn.functional as F
import ttl
import ttnn

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE
BF = HDIM_TILES
# s_dfb needs extra padding: matmul writes hdim_tiles tiles at write_ptr offset.
# After N push/pop cycles, wptr=N. Overflow extends to wptr+hdim_tiles-1.
# With 3 matmuls max, wptr can reach 2, so need at least 2+4=6 tiles.
S_BF = HDIM_TILES + 8  # generous padding for scores DFB


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


# C2: Pass1+Pass2 + recip(sum) -> output 1/sum
def make_c2_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        max_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        exp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        sinv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=BF)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                    e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(ev, sc, dims=[1]))
                with sum_dfb.wait() as sm, sinv_dfb.reserve() as si:
                    si.store(ttl.math.recip(sm))
            with sinv_dfb.wait() as sv, out_dfb.reserve() as o:
                o.store(sv)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(2):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
                with kt_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


# C3: C2 + broadcast(1/sum) -> output broadcast(1/sum)
def make_c3_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
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
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
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
            with sinv_bc_dfb.wait() as sv, out_dfb.reserve() as o:
                o.store(sv)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            for _ in range(2):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
                with kt_dfb.reserve() as blk:
                    tx = ttl.copy(k_t[0:hdim_tiles, 0:1], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()

    return k


# C4: C3 + 3rd matmul -> output raw scores from 3rd matmul
def make_c4_kernel(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, k_t, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(k_t, shape=(hdim_tiles, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=S_BF)
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
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
                with s_dfb.wait() as sv, max_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                    mxbc.store(ttl.math.broadcast(mx, dims=[1]))
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
                # 3rd matmul, output raw scores (drain sinv_bc)
                with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                    s.store(qb @ kc)
            with s_dfb.wait() as sv, out_dfb.reserve() as o:
                o.store(sv)
            with sinv_bc_dfb.wait() as _:
                pass

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
        row_max = scores_ref.max(dim=1, keepdim=True)[0]
        exp_ref = torch.exp(scores_ref - row_max)
        sum_ref = exp_ref.sum(dim=1, keepdim=True)

        q_tt = to_dev((q * scale).float(), d)
        kt_tt = to_dev(kt.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # C2: recip(sum)
        print("=== C2: Pass1+2 + recip(sum) ===")
        out = to_dev(torch.zeros(TILE, TILE), d)
        make_c2_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out)
        r = ttnn.to_torch(out).float()[:TILE, :1]
        ref = 1.0 / sum_ref
        p = pcc(ref, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # C3: broadcast(1/sum)
        print("\n=== C3: + broadcast(1/sum) ===")
        out = to_dev(torch.zeros(TILE, TILE), d)
        make_c3_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out)
        r = ttnn.to_torch(out).float()[:TILE, :TILE]
        ref = (1.0 / sum_ref).expand(-1, TILE)
        p = pcc(ref, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # C4: 3rd matmul raw scores
        print("\n=== C4: + 3rd matmul (raw scores) ===")
        out = to_dev(torch.zeros(TILE, TILE), d)
        make_c4_kernel(HDIM_TILES)(q_tt, kt_tt, sc_tt, out)
        r = ttnn.to_torch(out).float()[:TILE, :TILE]
        ref = scores_ref  # Should be same as 1st/2nd matmul
        p = pcc(ref, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # C5: 3rd matmul -> exp(s-max) (reuse broadcast max from pass 1)
        print("\n=== C5: + 3rd matmul -> exp(s - broadcast_max) ===")

        @ttl.kernel(grid=(1, 1))
        def c5_kernel(qin, ktin, scin, outv):
            q_dfb = ttl.make_dataflow_buffer_like(qin, shape=(1, HDIM_TILES), buffer_factor=2)
            kt_dfb = ttl.make_dataflow_buffer_like(ktin, shape=(HDIM_TILES, 1), buffer_factor=2)
            sc_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            s_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=S_BF)
            max_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            max_bc_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            exp_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            sum_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            sinv_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            sinv_bc_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=BF)
            out_dfb = ttl.make_dataflow_buffer_like(outv, shape=(1, 1), buffer_factor=2)

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
                    # Pass 2: exp + sum
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
                    # Pass 3: 3rd matmul -> exp(s-max)
                    with q_dfb.wait() as qb, kt_dfb.wait() as kc, s_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with max_dfb.wait() as mx, max_bc_dfb.reserve() as mxbc:
                        mxbc.store(ttl.math.broadcast(mx, dims=[1]))
                    with s_dfb.wait() as sv, max_bc_dfb.wait() as mxbc, exp_dfb.reserve() as e:
                        e.store(ttl.math.exp(sv - mxbc))
                with exp_dfb.wait() as ev, out_dfb.reserve() as o:
                    o.store(ev)
                with sinv_bc_dfb.wait() as _:
                    pass

            @ttl.datamovement()
            def dm_read():
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scin[0:1, 0:1], blk); tx.wait()
                for _ in range(3):
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                    with kt_dfb.reserve() as blk:
                        tx = ttl.copy(ktin[0:HDIM_TILES, 0:1], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, outv[0:1, 0:1]); tx.wait()

        out = to_dev(torch.zeros(TILE, TILE), d)
        c5_kernel(q_tt, kt_tt, sc_tt, out)
        r = ttnn.to_torch(out).float()[:TILE, :TILE]
        ref = exp_ref  # exp(scores - max)
        p = pcc(ref, r)
        print(f"  PCC={p:.4f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
