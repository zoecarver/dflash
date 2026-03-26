"""Test argmax TT-Lang kernel against PyTorch reference."""
import torch
import ttnn
from argmax import make_argmax_kernel

TILE = 32
ROWS = 32
COLS = 1024
COL_TILES = COLS // TILE

argmax_kernel = make_argmax_kernel(col_tiles=COL_TILES)


def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_argmax(device):
    # Create input with known max positions per row
    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    known_positions = torch.randint(0, COLS, (ROWS,))
    for i in range(ROWS):
        x[i, known_positions[i]] = 100.0

    expected = torch.argmax(x.float(), dim=-1)

    # Column indices: value at position j = j
    col_indices = torch.arange(COLS, dtype=torch.bfloat16).unsqueeze(0)

    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    out = torch.zeros(ROWS, TILE, dtype=torch.bfloat16)

    x_tt = to_dev(x, device)
    idx_tt = to_dev(col_indices, device)
    sc_tt = to_dev(scaler, device)
    out_tt = to_dev(out, device)

    argmax_kernel(x_tt, idx_tt, sc_tt, out_tt)

    result = ttnn.to_torch(out_tt)[:ROWS, :1].flatten()

    matches = 0
    for i in range(ROWS):
        got = int(result[i].item())
        exp = int(expected[i].item())
        if got == exp:
            matches += 1
        else:
            print(f"  row {i}: got={got} expected={exp}")

    accuracy = matches / ROWS
    print(f"Argmax test ({ROWS}x{COLS}): {matches}/{ROWS} rows correct ({accuracy:.1%})")
    if accuracy == 1.0:
        print("  PASS")
    else:
        print(f"  FAIL - {ROWS - matches} mismatches")
    return accuracy == 1.0


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    test_argmax(device)
    ttnn.close_device(device)
