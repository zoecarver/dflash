"""Test cross-attention via TTNN matmul + TT-Lang softmax + TTNN matmul.

Step 1: TTNN matmul Q @ K^T -> scores
Step 2: TT-Lang softmax kernel -> probs
Step 3: TTNN matmul probs @ V -> output
"""
import torch
import torch.nn.functional as F
import ttnn
from softmax import make_softmax_kernel

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


def run_test(n_heads, kv_tiles, d, test_name):
    print(f"=== {test_name}: {n_heads} heads, {kv_tiles} KV tiles ===")
    torch.manual_seed(42)
    kv_len = kv_tiles * TILE

    scale = 1.0 / (HDIM ** 0.5)
    q_raw = torch.randn(n_heads, TILE, HDIM) * 0.1
    k_raw = torch.randn(1, kv_len, HDIM) * 0.1
    v_raw = torch.randn(1, kv_len, HDIM) * 0.1

    # Stack Q heads, apply scale
    q_stacked = torch.zeros(n_heads * TILE, HDIM)
    for h in range(n_heads):
        q_stacked[h * TILE:(h + 1) * TILE] = q_raw[h] * scale
    q_bf = q_stacked.to(torch.bfloat16)
    k_bf = k_raw.squeeze(0).to(torch.bfloat16)
    v_bf = v_raw.squeeze(0).to(torch.bfloat16)

    # === Reference ===
    scores_ref = q_bf.float() @ k_bf.float().T  # (n_heads*TILE, kv_len)
    # Pad scores to tile boundary
    scores_padded = F.pad(scores_ref, (0, kv_len - scores_ref.shape[1])).to(torch.bfloat16) if scores_ref.shape[1] % TILE else scores_ref.to(torch.bfloat16)
    # Actually kv_len = kv_tiles * TILE, so already aligned
    probs_ref = torch.softmax(scores_ref, dim=-1)
    # Pad V
    v_padded = v_bf
    out_ref = probs_ref @ v_padded.float()

    # === Step 1: TTNN matmul Q @ K^T ===
    q_tt = to_dev(q_bf.float(), d)
    k_tt = to_dev(k_bf.float(), d)  # (kv_len, HDIM)
    v_tt = to_dev(v_bf.float(), d)  # (kv_len, HDIM)

    # TTNN matmul: Q @ K^T
    k_t_tt = ttnn.transpose(k_tt, -2, -1)  # (HDIM, kv_len)
    scores_tt = ttnn.matmul(q_tt, k_t_tt)  # (n_heads*TILE, kv_len)

    # Check scores
    scores_result = ttnn.to_torch(scores_tt).float()
    rows = n_heads * TILE
    cols = kv_len
    p_scores = pcc(scores_ref[:rows, :cols], scores_result[:rows, :cols])
    print(f"  Scores PCC: {p_scores:.6f}")

    # === Step 2: TT-Lang softmax ===
    sc_tt = to_dev(torch.ones(TILE, TILE), d)
    probs_tt = to_dev(torch.zeros(n_heads * TILE, kv_len), d)

    softmax_k = make_softmax_kernel(n_heads, kv_tiles)
    softmax_k(scores_tt, sc_tt, probs_tt)

    probs_result = ttnn.to_torch(probs_tt).float()
    # Reference: softmax on the bf16 scores (matching what kernel sees)
    scores_bf16 = scores_result[:rows, :cols].to(torch.bfloat16).float()
    probs_ref_bf16 = torch.softmax(scores_bf16, dim=-1)
    p_probs = pcc(probs_ref_bf16, probs_result[:rows, :cols])
    print(f"  Probs PCC:  {p_probs:.6f}")

    # === Step 3: TTNN matmul probs @ V ===
    out_tt = ttnn.matmul(probs_tt, v_tt)  # (n_heads*TILE, HDIM)

    out_result = ttnn.to_torch(out_tt).float()
    # Reference using bf16 probs
    out_ref_bf16 = probs_ref_bf16.to(torch.bfloat16).float() @ v_bf.float()
    p_out = pcc(out_ref_bf16[:rows, :HDIM], out_result[:rows, :HDIM])
    print(f"  Output PCC: {p_out:.6f}")

    # Per-head output PCC
    for h in range(n_heads):
        s = h * TILE
        hp = pcc(out_ref[:rows, :HDIM][s:s+TILE], out_result[:rows, :HDIM][s:s+TILE])
        print(f"    head {h}: PCC={hp:.4f}")

    overall = pcc(out_ref[:rows, :HDIM], out_result[:rows, :HDIM])
    print(f"  Overall PCC: {overall:.6f} {'PASS' if overall > 0.98 else 'FAIL'}")
    return overall


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        # Test 1: softmax kernel only (feed known scores)
        print("=== Test 0: Softmax kernel only ===")
        torch.manual_seed(42)
        scores = torch.randn(TILE, 3 * TILE).to(torch.bfloat16)
        ref = torch.softmax(scores.float(), dim=-1)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)
        s_tt = to_dev(scores.float(), d)
        p_tt = to_dev(torch.zeros(TILE, 3 * TILE), d)
        softmax_k = make_softmax_kernel(1, 3)
        softmax_k(s_tt, sc_tt, p_tt)
        r = ttnn.to_torch(p_tt).float()[:TILE, :3*TILE]
        p = pcc(ref, r)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}")
        print(f"  ref row0 sum: {ref[0].sum():.4f}, tt row0 sum: {r[0].sum():.4f}")
        print()

        # Test 1: 1 head, 1 KV tile
        run_test(1, 1, d, "Test 1")
        print()

        # Test 2: 1 head, 3 KV tiles
        run_test(1, 3, d, "Test 2")
        print()

        # Test 3: 8 heads, 3 KV tiles
        run_test(8, 3, d, "Test 3")
        print()

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
