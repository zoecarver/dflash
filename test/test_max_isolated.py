"""Test ttl.math.max in isolation."""
import torch
import torch.nn.functional as F
import ttl
import ttnn

TILE = 32


def to_dev(t, d):
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=d,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ==========================================================================
# Test 1: Simple element-wise max of two tiles
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def max_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            o.store(ttl.math.max(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:1], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# ==========================================================================
# Test 2: max with accumulator pattern (wait+reserve same DFB)
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def max_acc_kernel(a, b, c, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # acc = a
        with a_dfb.wait() as av, acc_dfb.reserve() as acc:
            acc.store(av)
        # acc = max(acc, b)
        with b_dfb.wait() as bv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
            new.store(ttl.math.max(old, bv))
        # acc = max(acc, c)
        with c_dfb.wait() as cv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
            new.store(ttl.math.max(old, cv))
        # output
        with acc_dfb.wait() as final, out_dfb.reserve() as o:
            o.store(final)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:1], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:1], blk); tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0:1, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# ==========================================================================
# Test 3: reduce_max then max accumulation (mimics Level 3)
# ==========================================================================
@ttl.kernel(grid=(1, 1))
def reduce_max_acc_kernel(a, b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            # acc = reduce_max(a)
            with a_dfb.wait() as av, acc_dfb.reserve() as acc:
                acc.store(ttl.math.reduce_max(av, sc, dims=[1]))
            # tmp = reduce_max(b)
            with b_dfb.wait() as bv, tmp_dfb.reserve() as t:
                t.store(ttl.math.reduce_max(bv, sc, dims=[1]))
            # acc = max(acc, tmp)
            with tmp_dfb.wait() as tv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
                new.store(ttl.math.max(old, tv))
        # output
        with acc_dfb.wait() as final, out_dfb.reserve() as o:
            o.store(final)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:1], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:1, 0:1], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:1, 0:1]); tx.wait()


# ==========================================================================
# Test 4: matmul -> reduce_max -> max acc (exact Level 3 pattern, 2 iterations)
# ==========================================================================
def make_matmul_reduce_max_acc(hdim_tiles):
    @ttl.kernel(grid=(1, 1))
    def k(q, kt0, kt1, scaler, out):
        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, hdim_tiles), buffer_factor=2)
        kt0_dfb = ttl.make_dataflow_buffer_like(kt0, shape=(hdim_tiles, 1), buffer_factor=2)
        kt1_dfb = ttl.make_dataflow_buffer_like(kt1, shape=(hdim_tiles, 1), buffer_factor=2)
        # Separate scalers for each reduce_max
        sc0_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        sc1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        # buffer_factor=hdim_tiles to absorb matmul codegen overflow (compiler bug)
        s0_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=4)
        s1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=4)
        r0_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        r1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        result_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            # matmul 1 -> reduce_max -> r0 (fresh scaler)
            with q_dfb.wait() as qb, kt0_dfb.wait() as kc, s0_dfb.reserve() as s:
                s.store(qb @ kc)
            with s0_dfb.wait() as sv, sc0_dfb.wait() as sc, r0_dfb.reserve() as r:
                r.store(ttl.math.reduce_max(sv, sc, dims=[1]))
            # matmul 2 -> reduce_max -> r1 (fresh scaler)
            with q_dfb.wait() as qb, kt1_dfb.wait() as kc, s1_dfb.reserve() as s:
                s.store(qb @ kc)
            with s1_dfb.wait() as sv, sc1_dfb.wait() as sc, r1_dfb.reserve() as r:
                r.store(ttl.math.reduce_max(sv, sc, dims=[1]))
            # max(r0, r1) -> result
            with r0_dfb.wait() as a, r1_dfb.wait() as b, result_dfb.reserve() as res:
                res.store(ttl.math.max(a, b))
            with result_dfb.wait() as final, out_dfb.reserve() as o:
                o.store(final)

        @ttl.datamovement()
        def dm_read():
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt0_dfb.reserve() as blk:
                tx = ttl.copy(kt0[0:hdim_tiles, 0:1], blk); tx.wait()
            with sc0_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()
            with q_dfb.reserve() as blk:
                tx = ttl.copy(q[0:1, 0:hdim_tiles], blk); tx.wait()
            with kt1_dfb.reserve() as blk:
                tx = ttl.copy(kt1[0:hdim_tiles, 0:1], blk); tx.wait()
            with sc1_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0:1, 0:1], blk); tx.wait()

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

        a = torch.randn(TILE, TILE) * 0.1
        b = torch.randn(TILE, TILE) * 0.1
        c = torch.randn(TILE, TILE) * 0.1

        a_tt = to_dev(a, d)
        b_tt = to_dev(b, d)
        c_tt = to_dev(c, d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)

        # ---- Test 1: Simple max ----
        print("=== Test 1: Simple element-wise max ===")
        out1 = to_dev(torch.zeros(TILE, TILE), d)
        max_kernel(a_tt, b_tt, out1)
        r1 = ttnn.to_torch(out1).float()[:TILE, :TILE]
        ref1 = torch.max(a.to(torch.bfloat16).float(), b.to(torch.bfloat16).float())
        p1 = pcc(ref1, r1)
        md1 = (ref1 - r1).abs().max().item()
        print(f"  PCC={p1:.4f} max_diff={md1:.6f}")
        print(f"  ref range: [{ref1.min():.4f}, {ref1.max():.4f}]")
        print(f"  tt  range: [{r1.min():.4f}, {r1.max():.4f}]")

        # ---- Test 2: max with accumulator ----
        print("\n=== Test 2: max with accumulator (3 inputs) ===")
        out2 = to_dev(torch.zeros(TILE, TILE), d)
        max_acc_kernel(a_tt, b_tt, c_tt, out2)
        r2 = ttnn.to_torch(out2).float()[:TILE, :TILE]
        ref2 = torch.max(torch.max(a.to(torch.bfloat16).float(), b.to(torch.bfloat16).float()), c.to(torch.bfloat16).float())
        p2 = pcc(ref2, r2)
        md2 = (ref2 - r2).abs().max().item()
        print(f"  PCC={p2:.4f} max_diff={md2:.6f}")
        print(f"  ref range: [{ref2.min():.4f}, {ref2.max():.4f}]")
        print(f"  tt  range: [{r2.min():.4f}, {r2.max():.4f}]")

        # ---- Test 3: reduce_max then max accumulation ----
        print("\n=== Test 3: reduce_max + max accumulation ===")
        out3 = to_dev(torch.zeros(TILE, TILE), d)
        reduce_max_acc_kernel(a_tt, b_tt, sc_tt, out3)
        r3 = ttnn.to_torch(out3).float()[:TILE, :1]
        a_bf = a.to(torch.bfloat16).float()
        b_bf = b.to(torch.bfloat16).float()
        ref3 = torch.max(a_bf.max(dim=1, keepdim=True)[0], b_bf.max(dim=1, keepdim=True)[0])
        p3 = pcc(ref3, r3)
        md3 = (ref3 - r3).abs().max().item()
        print(f"  PCC={p3:.4f} max_diff={md3:.6f}")
        print(f"  ref range: [{ref3.min():.4f}, {ref3.max():.4f}]")
        print(f"  tt  range: [{r3.min():.4f}, {r3.max():.4f}]")

        # ---- Test 4a: two matmul->reduce_max, output 2nd only ----
        print("\n=== Test 4a: two matmul+reduce_max, output 2nd ===")
        HDIM = 128
        HDIM_TILES = HDIM // TILE
        q = torch.randn(TILE, HDIM) * 0.1
        kt0 = torch.randn(HDIM, TILE) * 0.1
        kt1 = torch.randn(HDIM, TILE) * 0.1
        q_bf = q.to(torch.bfloat16).float()
        kt0_bf = kt0.to(torch.bfloat16).float()
        kt1_bf = kt1.to(torch.bfloat16).float()

        @ttl.kernel(grid=(1, 1))
        def two_matmul_reduce(qin, k0in, k1in, scin, outv):
            q_dfb = ttl.make_dataflow_buffer_like(qin, shape=(1, HDIM_TILES), buffer_factor=2)
            k0_dfb = ttl.make_dataflow_buffer_like(k0in, shape=(HDIM_TILES, 1), buffer_factor=2)
            k1_dfb = ttl.make_dataflow_buffer_like(k1in, shape=(HDIM_TILES, 1), buffer_factor=2)
            sc_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            s_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            discard_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            r_dfb = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            out_dfb = ttl.make_dataflow_buffer_like(outv, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                with sc_dfb.wait() as sc:
                    # 1st matmul + reduce (discard result)
                    with q_dfb.wait() as qb, k0_dfb.wait() as kc, s_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with s_dfb.wait() as sv, discard_dfb.reserve() as d:
                        d.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                    with discard_dfb.wait() as dv:
                        pass
                    # 2nd matmul + reduce (output this)
                    with q_dfb.wait() as qb, k1_dfb.wait() as kc, s_dfb.reserve() as s:
                        s.store(qb @ kc)
                    with s_dfb.wait() as sv, r_dfb.reserve() as r:
                        r.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                with r_dfb.wait() as rv, out_dfb.reserve() as o:
                    o.store(rv)

            @ttl.datamovement()
            def dm_read():
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scin[0:1, 0:1], blk); tx.wait()
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                with k0_dfb.reserve() as blk:
                    tx = ttl.copy(k0in[0:HDIM_TILES, 0:1], blk); tx.wait()
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                with k1_dfb.reserve() as blk:
                    tx = ttl.copy(k1in[0:HDIM_TILES, 0:1], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, outv[0:1, 0:1]); tx.wait()

        q_tt = to_dev(q, d)
        kt0_tt = to_dev(kt0, d)
        kt1_tt = to_dev(kt1, d)
        out4a = to_dev(torch.zeros(TILE, TILE), d)
        two_matmul_reduce(q_tt, kt0_tt, kt1_tt, sc_tt, out4a)
        r4a = ttnn.to_torch(out4a).float()[:TILE, :1]
        ref4a = (q_bf @ kt1_bf).max(dim=1, keepdim=True)[0]
        p4a = pcc(ref4a, r4a)
        md4a = (ref4a - r4a).abs().max().item()
        print(f"  PCC={p4a:.4f} max_diff={md4a:.6f}")
        print(f"  ref range: [{ref4a.min():.4f}, {ref4a.max():.4f}]")
        print(f"  tt  range: [{r4a.min():.4f}, {r4a.max():.4f}]")

        # ---- Test 4: matmul -> reduce_max -> max acc ----
        print("\n=== Test 4: matmul + reduce_max + max accumulation ===")
        q_tt = to_dev(q, d)
        kt0_tt = to_dev(kt0, d)
        kt1_tt = to_dev(kt1, d)
        out4 = to_dev(torch.zeros(TILE, TILE), d)

        k4 = make_matmul_reduce_max_acc(HDIM_TILES)
        k4(q_tt, kt0_tt, kt1_tt, sc_tt, out4)
        r4 = ttnn.to_torch(out4).float()[:TILE, :1]
        s0 = (q_bf @ kt0_bf).max(dim=1, keepdim=True)[0]
        s1 = (q_bf @ kt1_bf).max(dim=1, keepdim=True)[0]
        ref4 = torch.max(s0, s1)
        p4 = pcc(ref4, r4)
        md4 = (ref4 - r4).abs().max().item()
        print(f"  PCC={p4:.4f} max_diff={md4:.6f}")
        print(f"  ref range: [{ref4.min():.4f}, {ref4.max():.4f}]")
        print(f"  tt  range: [{r4.min():.4f}, {r4.max():.4f}]")

        # ---- Test 5: matmul -> reduce_max -> ADD acc (is it max-specific?) ----
        print("\n=== Test 5: matmul + reduce_max + ADD accumulation ===")

        @ttl.kernel(grid=(1, 1))
        def matmul_reduce_add_acc(qin, k0in, k1in, scin, outv):
            q_dfb2 = ttl.make_dataflow_buffer_like(qin, shape=(1, HDIM_TILES), buffer_factor=2)
            k0_dfb2 = ttl.make_dataflow_buffer_like(k0in, shape=(HDIM_TILES, 1), buffer_factor=2)
            k1_dfb2 = ttl.make_dataflow_buffer_like(k1in, shape=(HDIM_TILES, 1), buffer_factor=2)
            sc_dfb2 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            s_dfb2 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            tmp_dfb2 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            acc_dfb2 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=2)
            out_dfb2 = ttl.make_dataflow_buffer_like(outv, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                with sc_dfb2.wait() as sc:
                    with q_dfb2.wait() as qb, k0_dfb2.wait() as kc, s_dfb2.reserve() as s:
                        s.store(qb @ kc)
                    with s_dfb2.wait() as sv, acc_dfb2.reserve() as acc:
                        acc.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                    with q_dfb2.wait() as qb, k1_dfb2.wait() as kc, s_dfb2.reserve() as s:
                        s.store(qb @ kc)
                    with s_dfb2.wait() as sv, tmp_dfb2.reserve() as t:
                        t.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                    with tmp_dfb2.wait() as tv, acc_dfb2.wait() as old, acc_dfb2.reserve() as new:
                        new.store(old + tv)
                with acc_dfb2.wait() as final, out_dfb2.reserve() as o:
                    o.store(final)

            @ttl.datamovement()
            def dm_read():
                with sc_dfb2.reserve() as blk:
                    tx = ttl.copy(scin[0:1, 0:1], blk); tx.wait()
                with q_dfb2.reserve() as blk:
                    tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                with k0_dfb2.reserve() as blk:
                    tx = ttl.copy(k0in[0:HDIM_TILES, 0:1], blk); tx.wait()
                with q_dfb2.reserve() as blk:
                    tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                with k1_dfb2.reserve() as blk:
                    tx = ttl.copy(k1in[0:HDIM_TILES, 0:1], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                with out_dfb2.wait() as blk:
                    tx = ttl.copy(blk, outv[0:1, 0:1]); tx.wait()

        # ---- Test 6: matmul -> reduce_max -> acc -> read acc back -> output ----
        print("\n=== Test 6: matmul->reduce_max->acc, then readback acc ===")

        @ttl.kernel(grid=(1, 1))
        def matmul_reduce_readback(qin, k0in, scin, outv):
            q_dfb3 = ttl.make_dataflow_buffer_like(qin, shape=(1, HDIM_TILES), buffer_factor=2)
            k0_dfb3 = ttl.make_dataflow_buffer_like(k0in, shape=(HDIM_TILES, 1), buffer_factor=2)
            sc_dfb3 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            s_dfb3 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=1)
            acc_dfb3 = ttl.make_dataflow_buffer_like(scin, shape=(1, 1), buffer_factor=2)
            out_dfb3 = ttl.make_dataflow_buffer_like(outv, shape=(1, 1), buffer_factor=2)

            @ttl.compute()
            def compute():
                with sc_dfb3.wait() as sc:
                    with q_dfb3.wait() as qb, k0_dfb3.wait() as kc, s_dfb3.reserve() as s:
                        s.store(qb @ kc)
                    with s_dfb3.wait() as sv, acc_dfb3.reserve() as acc:
                        acc.store(ttl.math.reduce_max(sv, sc, dims=[1]))
                # Read acc back and output
                with acc_dfb3.wait() as val, out_dfb3.reserve() as o:
                    o.store(val)

            @ttl.datamovement()
            def dm_read():
                with sc_dfb3.reserve() as blk:
                    tx = ttl.copy(scin[0:1, 0:1], blk); tx.wait()
                with q_dfb3.reserve() as blk:
                    tx = ttl.copy(qin[0:1, 0:HDIM_TILES], blk); tx.wait()
                with k0_dfb3.reserve() as blk:
                    tx = ttl.copy(k0in[0:HDIM_TILES, 0:1], blk); tx.wait()

            @ttl.datamovement()
            def dm_write():
                with out_dfb3.wait() as blk:
                    tx = ttl.copy(blk, outv[0:1, 0:1]); tx.wait()

        q_tt6 = to_dev(q, d)
        kt0_tt6 = to_dev(kt0, d)
        out6 = to_dev(torch.zeros(TILE, TILE), d)
        matmul_reduce_readback(q_tt6, kt0_tt6, sc_tt, out6)
        r6 = ttnn.to_torch(out6).float()[:TILE, :1]
        ref6 = (q_bf @ kt0_bf).max(dim=1, keepdim=True)[0]
        p6 = pcc(ref6, r6)
        md6 = (ref6 - r6).abs().max().item()
        print(f"  PCC={p6:.4f} max_diff={md6:.6f}")
        print(f"  ref range: [{ref6.min():.4f}, {ref6.max():.4f}]")
        print(f"  tt  range: [{r6.min():.4f}, {r6.max():.4f}]")

        q_tt2 = to_dev(q, d)
        kt0_tt2 = to_dev(kt0, d)
        kt1_tt2 = to_dev(kt1, d)
        out5 = to_dev(torch.zeros(TILE, TILE), d)
        matmul_reduce_add_acc(q_tt2, kt0_tt2, kt1_tt2, sc_tt, out5)
        r5 = ttnn.to_torch(out5).float()[:TILE, :1]
        ref5 = s0 + s1
        p5 = pcc(ref5, r5)
        md5 = (ref5 - r5).abs().max().item()
        print(f"  PCC={p5:.4f} max_diff={md5:.6f}")
        print(f"  ref range: [{ref5.min():.4f}, {ref5.max():.4f}]")
        print(f"  tt  range: [{r5.min():.4f}, {r5.max():.4f}]")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
