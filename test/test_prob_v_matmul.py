"""Test (1,1) @ (1, hdim_tiles) matmul: prob_weights @ V_row."""
import torch
import torch.nn.functional as F
import ttl
import ttnn

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


# Simple (1,1) @ (1, HDIM_TILES) matmul
@ttl.kernel(grid=(1, 1))
def prob_v_matmul(prob, v, out):
    p_dfb = ttl.make_dataflow_buffer_like(prob, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, HDIM_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HDIM_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with p_dfb.wait() as pw, v_dfb.wait() as vr, out_dfb.reserve() as o:
            o.store(pw @ vr)

    @ttl.datamovement()
    def dm_read():
        with p_dfb.reserve() as blk:
            tx = ttl.copy(prob[0:1, 0:1], blk); tx.wait()
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

        # Test A: simple matmul with known data
        print("=== Test A: identity-like prob @ V ===")
        prob = torch.eye(TILE).to(torch.bfloat16)  # identity should pass V through
        v = torch.randn(TILE, HDIM).to(torch.bfloat16)
        ref = prob.float() @ v.float()

        p_tt = to_dev(prob.float(), d)
        v_tt = to_dev(v.float(), d)
        out_tt = to_dev(torch.zeros(TILE, HDIM), d)
        prob_v_matmul(p_tt, v_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :HDIM]
        p = pcc(ref, r)
        print(f"  PCC={p:.6f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # Test B: random softmax-like probs @ V
        print("\n=== Test B: random probs @ V ===")
        scores = torch.randn(TILE, TILE)
        prob = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v = torch.randn(TILE, HDIM).to(torch.bfloat16)
        ref = prob.float() @ v.float()

        p_tt = to_dev(prob.float(), d)
        v_tt = to_dev(v.float(), d)
        out_tt = to_dev(torch.zeros(TILE, HDIM), d)
        prob_v_matmul(p_tt, v_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :HDIM]
        p = pcc(ref, r)
        print(f"  PCC={p:.6f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

        # Test C: scale that matches our attention (small values)
        print("\n=== Test C: small-valued probs @ V ===")
        scale = 1.0 / (HDIM ** 0.5)
        q = torch.randn(TILE, HDIM) * 0.1 * scale
        k = torch.randn(TILE, HDIM) * 0.1
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        scores = q_bf.float() @ k_bf.float().T
        mx = scores.max(dim=1, keepdim=True)[0]
        prob = torch.exp(scores - mx)
        prob = prob / prob.sum(dim=1, keepdim=True)
        prob_bf = prob.to(torch.bfloat16)
        v = torch.randn(TILE, HDIM).to(torch.bfloat16) * 0.1
        ref = prob_bf.float() @ v.float()

        p_tt = to_dev(prob_bf.float(), d)
        v_tt = to_dev(v.float(), d)
        out_tt = to_dev(torch.zeros(TILE, HDIM), d)
        prob_v_matmul(p_tt, v_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()[:TILE, :HDIM]
        p = pcc(ref, r)
        print(f"  PCC={p:.6f}")
        print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"  tt  range: [{r.min():.4f}, {r.max():.4f}]")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
