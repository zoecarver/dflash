"""Smoke test for remaining kernels: rope, per_head_rmsnorm, silu_mul_weight, argmax, cross_attention."""
import sys
sys.path.insert(0, "/tmp")

import torch
import torch.nn.functional as F
import ttnn

TILE = 32

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def to_dev(t, d):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    t = t.contiguous().to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def test_silu_mul_weight(d):
    from silu_mul_weight import make_silu_mul_weight_kernel
    # 8x8 block = 256x256 elements, use tensor that divides evenly
    R, C = 256, 256
    gate = torch.randn(R, C, dtype=torch.bfloat16)
    up = torch.randn(R, C, dtype=torch.bfloat16)
    w = torch.randn(R, C, dtype=torch.bfloat16) * 0.1 + 1.0
    expected = F.silu(gate.float()) * up.float() * w.float()
    out_dev = to_dev(torch.zeros_like(gate), d)
    kernel = make_silu_mul_weight_kernel(block_r=8, block_c=8, buf=3)
    kernel(to_dev(gate, d), to_dev(up, d), to_dev(w, d), out_dev)
    result = ttnn.to_torch(out_dev)[:R, :C]
    p = pcc(expected, result)
    print(f"silu_mul_weight (8x8 block): PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"silu_mul_weight PCC too low: {p}"

def test_rope(d):
    from rope import make_rope_kernel
    HDIM = 128
    head_tiles = HDIM // TILE
    n_heads = 4
    seq = 32
    half = HDIM // 2

    q = torch.randn(seq, n_heads * HDIM, dtype=torch.bfloat16)
    freqs = 1.0 / (1e7 ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    cos_full = torch.cos(torch.outer(torch.arange(seq, dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_full = torch.sin(torch.outer(torch.arange(seq, dtype=torch.float32), freqs)).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_adj = sin_full.clone()
    sin_adj[:, :half] = -sin_adj[:, :half]

    # PyTorch reference
    q_f = q.float().view(seq, n_heads, HDIM)
    cos_f = cos_full[:seq, :half].float()
    sin_f = sin_full[:seq, :half].float()
    expected = torch.zeros_like(q_f)
    for h in range(n_heads):
        x1, x2 = q_f[:, h, :half], q_f[:, h, half:]
        expected[:, h] = torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)
    expected = expected.view(seq, n_heads * HDIM)

    kernel = make_rope_kernel(head_tiles=head_tiles, n_heads=n_heads)
    out_dev = to_dev(torch.zeros(seq, n_heads * HDIM), d)
    kernel(to_dev(q, d), to_dev(cos_full[:seq], d), to_dev(sin_adj[:seq], d), out_dev)
    result = ttnn.to_torch(out_dev)[:seq, :n_heads * HDIM]
    p = pcc(expected, result)
    print(f"rope: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"rope PCC too low: {p}"

def test_per_head_rmsnorm(d):
    from per_head_rmsnorm import make_per_head_rmsnorm_kernel
    HDIM = 128
    head_tiles = HDIM // TILE
    n_heads = 4
    seq = 32
    eps = 1e-6

    x = torch.randn(seq, n_heads * HDIM, dtype=torch.bfloat16)
    w = (torch.randn(HDIM, dtype=torch.bfloat16) * 0.1 + 1.0)

    # PyTorch reference
    x_f = x.float().view(seq, n_heads, HDIM)
    w_f = w.float()
    expected = torch.zeros_like(x_f)
    for h in range(n_heads):
        xh = x_f[:, h]
        expected[:, h] = (xh / torch.sqrt(xh.pow(2).mean(-1, keepdim=True) + eps)) * w_f
    expected = expected.view(seq, n_heads * HDIM)

    kernel = make_per_head_rmsnorm_kernel(head_tiles=head_tiles, n_heads=n_heads, eps=eps)
    sc = to_dev(torch.ones(TILE, TILE), d)
    ms = to_dev(torch.full((TILE, TILE), 1.0 / HDIM), d)
    w_dev = to_dev(w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
    out_dev = to_dev(torch.zeros(seq, n_heads * HDIM), d)
    kernel(to_dev(x, d), w_dev, sc, ms, out_dev)
    result = ttnn.to_torch(out_dev)[:seq, :n_heads * HDIM]
    p = pcc(expected, result)
    print(f"per_head_rmsnorm: PCC={p:.6f} {'PASS' if p > 0.99 else 'FAIL'}")
    assert p > 0.99, f"per_head_rmsnorm PCC too low: {p}"

def test_cross_attention(d):
    from cross_attention import make_cross_attention_kernel
    HDIM = 128
    hdim_tiles = HDIM // TILE
    n_heads = 2
    kv_tiles = 2
    kv_len = kv_tiles * TILE

    q = torch.randn(n_heads * TILE, HDIM, dtype=torch.bfloat16)
    k = torch.randn(kv_len, HDIM, dtype=torch.bfloat16)
    v = torch.randn(kv_len, HDIM, dtype=torch.bfloat16)
    scale = 1.0 / (HDIM ** 0.5)

    # PyTorch reference
    expected = torch.zeros(n_heads, TILE, HDIM)
    for h in range(n_heads):
        qh = q[h*TILE:(h+1)*TILE].float()
        scores = (qh @ k.float().T) * scale
        probs = torch.softmax(scores, dim=-1)
        expected[h] = probs @ v.float()
    expected = expected.view(n_heads * TILE, HDIM)

    kernel = make_cross_attention_kernel(n_heads=n_heads, hdim_tiles=hdim_tiles, kv_tiles=kv_tiles)
    # Pre-scale Q
    q_scaled = (q.float() * scale).to(torch.bfloat16)
    k_t = k.T.contiguous()
    sc = to_dev(torch.ones(TILE, TILE), d)
    out_dev = to_dev(torch.zeros(n_heads * TILE, HDIM), d)
    kernel(to_dev(q_scaled, d), to_dev(k_t, d), to_dev(v, d), sc, out_dev)
    result = ttnn.to_torch(out_dev)[:n_heads * TILE, :HDIM]
    p = pcc(expected, result)
    print(f"cross_attention: PCC={p:.6f} {'PASS' if p > 0.95 else 'FAIL'}")
    assert p > 0.95, f"cross_attention PCC too low: {p}"

if __name__ == "__main__":
    d = ttnn.open_device(device_id=0)
    try:
        test_silu_mul_weight(d)
        test_rope(d)
        test_per_head_rmsnorm(d)
        test_cross_attention(d)
        print("\nAll smoke tests PASSED!")
    finally:
        ttnn.close_device(d)
