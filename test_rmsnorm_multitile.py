"""Test rmsnorm kernel with 64 tiles (the actual model config) vs PyTorch."""
import torch
import ttnn
from rmsnorm import make_rmsnorm_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE  # 64
EPS = 1e-6


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def from_tt(t):
    return ttnn.to_torch(t)

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def test():
    device = ttnn.open_device(device_id=0)
    try:
        rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)

        # Random input: 1 row of 2048 (padded to 32x64 tiles = 32x2048)
        sp = TILE  # 1 tile row
        x = torch.randn(sp, HIDDEN, dtype=torch.bfloat16)
        w = torch.ones(sp, HIDDEN, dtype=torch.bfloat16)  # weight = 1 for simplicity
        sc = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        ms = torch.full((TILE, TILE), 1.0 / HIDDEN, dtype=torch.bfloat16)
        out = torch.zeros(sp, HIDDEN, dtype=torch.bfloat16)

        x_tt = to_tt(x, device)
        w_tt = to_tt(w, device)
        sc_tt = to_tt(sc, device)
        ms_tt = to_tt(ms, device)
        out_tt = to_tt(out, device)

        rmsnorm_k(x_tt, w_tt, sc_tt, ms_tt, out_tt)
        dev_out = from_tt(out_tt)

        # PyTorch reference
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        ref = (xf * torch.rsqrt(variance + EPS)).to(torch.bfloat16)

        print(f"=== RMSNorm 64-tile test (sp={sp}, hidden={HIDDEN}) ===")
        print(f"PCC: {pcc(dev_out, ref):.6f}")
        print(f"Max diff: {(dev_out.float() - ref.float()).abs().max().item():.6f}")

        # Check ratio
        mask = ref.abs() > 0.01
        if mask.any():
            ratios = dev_out[mask].float() / ref[mask].float()
            print(f"Mean ratio: {ratios.mean().item():.6f}")
            print(f"Ratio std: {ratios.std().item():.6f}")

        print(f"\nRef [0,:5]:  {ref[0,:5].float()}")
        print(f"Dev [0,:5]:  {dev_out[0,:5].float()}")
        print(f"Ratio [0,:5]: {(dev_out[0,:5].float() / ref[0,:5].float())}")

        # Also check: what's the sum of squares?
        sum_sq = xf[0].pow(2).sum().item()
        mean_sq = sum_sq / HIDDEN
        print(f"\nRow 0: sum(x^2) = {sum_sq:.4f}")
        print(f"Row 0: mean(x^2) = {mean_sq:.6f}")
        print(f"Row 0: rsqrt(mean+eps) = {1.0 / (mean_sq + EPS)**0.5:.6f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test()
