"""Test RMSNorm TT-Lang kernel against PyTorch reference."""
import torch
import ttnn
from rmsnorm import make_rmsnorm_kernel

TILE = 32
HIDDEN_SIZE = 2048
HIDDEN_TILES = HIDDEN_SIZE // TILE
EPS = 1e-6

rmsnorm_kernel = make_rmsnorm_kernel(dim_tiles=HIDDEN_TILES, eps=EPS)


def test_rmsnorm(seq_len=32):
    device = ttnn.open_device(device_id=0)

    # Create test data
    x_torch = torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    w_torch = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    out_torch = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)

    # PyTorch reference
    x_f32 = x_torch.float()
    w_f32 = w_torch.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    expected = ((x_f32 / rms) * w_f32).to(torch.bfloat16)

    # Expand weight to (seq_len, HIDDEN_SIZE) for the kernel
    w_expanded = w_torch.unsqueeze(0).expand(seq_len, -1).contiguous()

    # Scaler: tile of 1.0s for reduction
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    # Mean scale: 1/HIDDEN_SIZE for computing mean of squares
    mean_scale = torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16)

    # Convert to device tensors
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_tt = ttnn.from_torch(w_expanded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt = ttnn.from_torch(out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scaler_tt = ttnn.from_torch(scaler, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mean_scale_tt = ttnn.from_torch(mean_scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run kernel
    rmsnorm_kernel(x_tt, w_tt, scaler_tt, mean_scale_tt, out_tt)

    # Readback and compare
    result = ttnn.to_torch(out_tt)[:seq_len, :HIDDEN_SIZE]

    # Compute PCC
    r_flat = result.float().flatten()
    e_flat = expected.float().flatten()
    pcc = torch.corrcoef(torch.stack([r_flat, e_flat]))[0, 1].item()

    # Max absolute error
    max_err = (result.float() - expected.float()).abs().max().item()

    print(f"RMSNorm test (seq_len={seq_len}, hidden={HIDDEN_SIZE}):")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Result sample: {result[0, :4]}")
    print(f"  Expected sample: {expected[0, :4]}")

    if pcc > 0.99:
        print("  PASS")
    else:
        print("  FAIL - PCC too low")

    ttnn.close_device(device)
    return pcc > 0.99


if __name__ == "__main__":
    test_rmsnorm(seq_len=32)
    test_rmsnorm(seq_len=64)
