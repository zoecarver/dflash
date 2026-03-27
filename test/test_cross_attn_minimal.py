"""Minimal cross-attention test: 1 head, 1 KV tile."""
import torch
import torch.nn.functional as F
import ttnn
from cross_attention import make_cross_attention_kernel

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


def run_test(n_heads, kv_tiles, d):
    torch.manual_seed(42)
    kv_len = kv_tiles * TILE

    scale = 1.0 / (HDIM ** 0.5)
    q_raw = torch.randn(n_heads, TILE, HDIM) * 0.1
    k_raw = torch.randn(1, kv_len, HDIM) * 0.1
    v_raw = torch.randn(1, kv_len, HDIM) * 0.1

    q_padded = torch.zeros(n_heads * TILE, HDIM)
    for h in range(n_heads):
        q_padded[h * TILE:(h + 1) * TILE] = q_raw[h] * scale
    q_bf16 = q_padded.to(torch.bfloat16)
    k_t_padded = k_raw.squeeze(0).T.contiguous().to(torch.bfloat16)
    v_padded = v_raw.squeeze(0).to(torch.bfloat16)

    # Reference
    ref_out = torch.zeros(n_heads * TILE, HDIM)
    for h in range(n_heads):
        q_h = q_bf16[h * TILE:(h + 1) * TILE].float()
        all_scores = []
        for kv_idx in range(kv_tiles):
            k_col = k_t_padded[:, kv_idx * TILE:(kv_idx + 1) * TILE].float()
            all_scores.append(q_h @ k_col)
        scores = torch.cat(all_scores, dim=1)
        probs = torch.softmax(scores, dim=-1)
        ref_out[h * TILE:(h + 1) * TILE] = probs @ v_padded.float()

    # TT
    q_tt = to_dev(q_bf16.float(), d)
    kt_tt = to_dev(k_t_padded.float(), d)
    v_tt = to_dev(v_padded.float(), d)
    sc_tt = to_dev(torch.ones(TILE, TILE), d)
    out_tt = to_dev(torch.zeros(n_heads * TILE, HDIM), d)

    kernel = make_cross_attention_kernel(n_heads, HDIM_TILES, kv_tiles)
    kernel(q_tt, kt_tt, v_tt, sc_tt, out_tt)

    tt_result = ttnn.to_torch(out_tt).float()

    for h in range(n_heads):
        s = h * TILE
        hp = pcc(ref_out[s:s + TILE, :HDIM], tt_result[s:s + TILE, :HDIM])
        md = (ref_out[s:s + TILE, :HDIM] - tt_result[s:s + TILE, :HDIM]).abs().max().item()
        print(f"  head {h}: PCC={hp:.4f} max_diff={md:.6f}")

    p = pcc(ref_out, tt_result[:n_heads * TILE, :HDIM])
    return p


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        # Test 1: 1 head, 1 KV tile (absolute minimum)
        print("=== Test 1: 1 head, 1 KV tile ===")
        p = run_test(1, 1, d)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}\n")

        # Test 2: 1 head, 3 KV tiles (streaming)
        print("=== Test 2: 1 head, 3 KV tiles ===")
        p = run_test(1, 3, d)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}\n")

        # Test 3: 2 heads, 1 KV tile (multi-head, no streaming)
        print("=== Test 3: 2 heads, 1 KV tile ===")
        p = run_test(2, 1, d)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}\n")

        # Test 4: 2 heads, 3 KV tiles
        print("=== Test 4: 2 heads, 3 KV tiles ===")
        p = run_test(2, 3, d)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}\n")

        # Test 5: 8 heads, 3 KV tiles (full config)
        print("=== Test 5: 8 heads, 3 KV tiles ===")
        p = run_test(8, 3, d)
        print(f"  PCC={p:.6f} {'PASS' if p > 0.98 else 'FAIL'}\n")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
