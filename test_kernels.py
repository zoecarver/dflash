"""Test all TT-Lang kernels against PyTorch references."""
import torch
import ttnn
from rmsnorm import make_rmsnorm_kernel
from silu import silu_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel

TILE = 32


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_silu(device):
    x = torch.randn(32, 256, dtype=torch.bfloat16)
    out = torch.zeros_like(x)
    expected = (x.float() * torch.sigmoid(x.float())).to(torch.bfloat16)

    x_tt = to_dev(x, device)
    out_tt = to_dev(out, device)
    silu_kernel(x_tt, out_tt)
    result = ttnn.to_torch(out_tt)[:32, :256]

    p = pcc(result, expected)
    print(f"SiLU: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    return p > 0.99


def test_residual_add(device):
    a = torch.randn(32, 256, dtype=torch.bfloat16)
    b = torch.randn(32, 256, dtype=torch.bfloat16)
    out = torch.zeros_like(a)
    expected = (a.float() + b.float()).to(torch.bfloat16)

    a_tt = to_dev(a, device)
    b_tt = to_dev(b, device)
    out_tt = to_dev(out, device)
    residual_add_kernel(a_tt, b_tt, out_tt)
    result = ttnn.to_torch(out_tt)[:32, :256]

    p = pcc(result, expected)
    print(f"Residual Add: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    return p > 0.99


def test_silu_mul(device):
    gate = torch.randn(32, 256, dtype=torch.bfloat16)
    up = torch.randn(32, 256, dtype=torch.bfloat16)
    out = torch.zeros_like(gate)
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)

    gate_tt = to_dev(gate, device)
    up_tt = to_dev(up, device)
    out_tt = to_dev(out, device)
    silu_mul_kernel(gate_tt, up_tt, out_tt)
    result = ttnn.to_torch(out_tt)[:32, :256]

    p = pcc(result, expected)
    print(f"SiLU*Mul: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    return p > 0.99


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    results = []
    results.append(test_silu(device))
    results.append(test_residual_add(device))
    results.append(test_silu_mul(device))
    ttnn.close_device(device)

    print(f"\n{sum(results)}/{len(results)} tests passed")
