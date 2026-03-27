"""Test fused cross-attention kernel against PyTorch reference."""
import torch
import torch.nn.functional as F
import ttnn
from cross_attention import make_cross_attention_kernel

TILE = 32
HDIM = 128
HDIM_TILES = HDIM // TILE
NQH_TP = 8
BSIZE = 16


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


def main():
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        torch.manual_seed(42)

        ctx_len = 64
        kv_len = ctx_len + BSIZE
        kv_len_padded = ((kv_len + TILE - 1) // TILE) * TILE
        kv_tiles = kv_len_padded // TILE

        print(f"NQH_TP={NQH_TP}, BSIZE={BSIZE}, ctx={ctx_len}, "
              f"kv={kv_len}, kv_pad={kv_len_padded}, kv_tiles={kv_tiles}")

        # Build padded bf16 tensors (exact kernel input)
        scale = 1.0 / (HDIM ** 0.5)
        q_raw = torch.randn(NQH_TP, BSIZE, HDIM) * 0.1
        k_raw = torch.randn(1, kv_len, HDIM) * 0.1
        v_raw = torch.randn(1, kv_len, HDIM) * 0.1

        q_padded = torch.zeros(NQH_TP * TILE, HDIM)
        for h in range(NQH_TP):
            q_padded[h * TILE:h * TILE + BSIZE] = q_raw[h] * scale
        q_bf16 = q_padded.to(torch.bfloat16)
        k_t_padded = F.pad(k_raw.squeeze(0).T.contiguous(), (0, kv_len_padded - kv_len)).to(torch.bfloat16)
        v_padded = F.pad(v_raw.squeeze(0), (0, 0, 0, kv_len_padded - kv_len)).to(torch.bfloat16)

        # Per-head reference on padded bf16, streaming per KV tile (matches kernel exactly)
        ref_out = torch.zeros(NQH_TP * TILE, HDIM)
        for h in range(NQH_TP):
            q_h = q_bf16[h * TILE:(h + 1) * TILE].float()  # (32, 128)
            # Compute scores per KV tile, softmax, weight V
            all_scores = []
            for kv_idx in range(kv_tiles):
                k_col = k_t_padded[:, kv_idx * TILE:(kv_idx + 1) * TILE].float()  # (128, 32)
                s = q_h @ k_col  # (32, 32)
                all_scores.append(s)
            scores = torch.cat(all_scores, dim=1)  # (32, kv_padded)
            probs = torch.softmax(scores, dim=-1)
            ref_out[h * TILE:(h + 1) * TILE] = probs @ v_padded.float()

        # TT tensors
        q_tt = to_dev(q_bf16.float(), d)
        kt_tt = to_dev(k_t_padded.float(), d)
        v_tt = to_dev(v_padded.float(), d)
        sc_tt = to_dev(torch.ones(TILE, TILE), d)
        out_tt = to_dev(torch.zeros(NQH_TP * TILE, HDIM), d)

        kernel = make_cross_attention_kernel(NQH_TP, HDIM_TILES, kv_tiles)
        print("Running cross_attention kernel...")
        kernel(q_tt, kt_tt, v_tt, sc_tt, out_tt)

        tt_result = ttnn.to_torch(out_tt).float()

        for h in range(NQH_TP):
            s = h * TILE
            hp = pcc(ref_out[s:s + TILE, :HDIM], tt_result[s:s + TILE, :HDIM])
            md = (ref_out[s:s + TILE, :HDIM] - tt_result[s:s + TILE, :HDIM]).abs().max().item()
            print(f"  head {h}: PCC={hp:.4f} max_diff={md:.6f}")

        p = pcc(ref_out, tt_result[:NQH_TP * TILE, :HDIM])
        print(f"\nOverall PCC: {p:.6f}")
        print("PASS" if p > 0.98 else "FAIL")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
