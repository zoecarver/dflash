"""DFlash draft model: swap TT-Lang softmax/rmsnorm/silu_mul/add with TTNN ops.

Keeps TT-Lang rope kernel (precision impact is small).
Tests whether TTNN ops improve PCC vs PyTorch reference.
"""
import time
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from rope import make_rope_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE
HDIM = 128
HDIM_TILES = HDIM // TILE
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7

DLAYERS = 8
DINTER = 6144
BSIZE = 16
SP = ((BSIZE + TILE - 1) // TILE) * TILE
N_CTX_LAYERS = 5
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


# ---------------------------------------------------------------------------
# PyTorch reference (same as test_dflash_real_weights.py)
# ---------------------------------------------------------------------------
def torch_rmsnorm(x, weight, eps=EPS):
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight

def torch_rope(x, cos, sin):
    x1, x2 = x[..., :HDIM//2], x[..., HDIM//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

def torch_prepare_context(target_hidden, fc_w, hn_w):
    projected = target_hidden.float() @ fc_w.float()
    return torch_rmsnorm(projected, hn_w.float())

def torch_layer_fwd(h, ctx, w, li, cos_q, sin_q, cos_kv, sin_kv):
    kv_len = ctx.shape[0] + h.shape[0]
    bsize = h.shape[0]
    normed = torch_rmsnorm(h, w[f"in_w.{li}"].float())
    q = normed @ w[f"qw.{li}"].float()
    kv_in = torch.cat([ctx, normed], dim=0)
    k = kv_in @ w[f"kw.{li}"].float()
    v = kv_in @ w[f"vw.{li}"].float()
    q = q.view(bsize, NQH, HDIM)
    k = k.view(kv_len, NKVH, HDIM)
    v = v.view(kv_len, NKVH, HDIM)
    for head in range(NQH):
        q[:, head] = torch_rmsnorm(q[:, head], w[f"qnw.{li}"].float())
    for head in range(NKVH):
        k[:, head] = torch_rmsnorm(k[:, head], w[f"knw.{li}"].float())
    for head in range(NQH):
        q[:, head] = torch_rope(q[:, head], cos_q, sin_q)
    for head in range(NKVH):
        k[:, head] = torch_rope(k[:, head], cos_kv, sin_kv)
    scale = 1.0 / (HDIM ** 0.5)
    attn_out = torch.zeros(bsize, NQH, HDIM)
    for qh in range(NQH):
        kvh = qh // GQA
        scores = (q[:, qh] @ k[:, kvh].T) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out[:, qh] = probs @ v[:, kvh]
    attn_out = attn_out.view(bsize, NQH * HDIM)
    o = attn_out @ w[f"ow.{li}"].float()
    h = h + o
    normed2 = torch_rmsnorm(h, w[f"pa_w.{li}"].float())
    gate = normed2 @ w[f"gw.{li}"].float()
    up = normed2 @ w[f"uw.{li}"].float()
    down = (F.silu(gate) * up) @ w[f"fc2.{li}"].float()
    h = h + down
    return h


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def _p(t):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph = (TILE - h % TILE) % TILE
    pw = (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)

def to_dev(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------
# TTNN-based layer forward (TT-Lang rope only, everything else TTNN)
# ---------------------------------------------------------------------------
def dev_layer_fwd_ttnn(h, ctx_dev, dw, li, q_rope_k, k_rope_k, kv_sp, d):
    scale = 1.0 / (HDIM ** 0.5)

    # ttnn.rms_norm for input layernorm
    normed = ttnn.rms_norm(h, weight=dw[f"in_w.{li}"], epsilon=EPS)

    q = ttnn.matmul(normed, dw[f"qw.{li}"])
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)
    k = ttnn.matmul(kv_in, dw[f"kw.{li}"])
    v = ttnn.matmul(kv_in, dw[f"vw.{li}"])

    # QK-norm (already using ttnn.rms_norm)
    q_flat = ttnn.reshape(q, (SP * NQH, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=dw[f"qnw.{li}"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=dw[f"knw.{li}"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))

    # TT-Lang rope (keep for now)
    q_roped = to_dev(torch.zeros(SP, NQH * HDIM), d)
    k_roped = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    q_rope_k(q_normed, dw["rope_cos_q"], dw["rope_sin_q"], q_roped)
    k_rope_k(k_normed, dw["rope_cos_kv"], dw["rope_sin_kv"], k_roped)

    # Stacked Q + batched matmul
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH, HDIM)), 1, 2)
    q_grouped = ttnn.reshape(q4, (1, NKVH, GQA * SP, HDIM))
    k_t = ttnn.transpose(k4, -2, -1)
    scores = ttnn.matmul(q_grouped, k_t)
    scores = ttnn.multiply(scores, scale)

    # ttnn.softmax instead of TT-Lang
    probs = ttnn.softmax(scores, dim=-1)

    attn_4d = ttnn.matmul(probs, v4)
    attn_heads = ttnn.reshape(attn_4d, (1, NQH, SP, HDIM))
    attn_flat = ttnn.reshape(ttnn.transpose(attn_heads, 1, 2), (SP, NQH * HDIM))

    o = ttnn.matmul(attn_flat, dw[f"ow.{li}"])

    # ttnn.add for residual
    h = ttnn.add(h, o)

    # ttnn.rms_norm for post-attention
    normed2 = ttnn.rms_norm(h, weight=dw[f"pa_w.{li}"], epsilon=EPS)

    gate = ttnn.matmul(normed2, dw[f"gw.{li}"])
    up = ttnn.matmul(normed2, dw[f"uw.{li}"])

    # ttnn.silu + ttnn.mul
    act = ttnn.mul(ttnn.silu(gate), up)
    down = ttnn.matmul(act, dw[f"fc2.{li}"])

    h = ttnn.add(h, down)
    return h


def main():
    torch.manual_seed(42)
    ctx_len = 64
    ctx_sp = _tile_pad(ctx_len)
    kv_len = ctx_len + BSIZE
    kv_sp = ctx_sp + SP

    print(f"Config: BSIZE={BSIZE}, ctx={ctx_len}, ctx_sp={ctx_sp}, kv_sp={kv_sp}")

    # Load weights
    print("Loading weights from safetensors...")
    t0 = time.time()
    w = {}
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        w["fc_w"] = f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16)
        w["hn_w"] = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        w["fn_w"] = f.get_tensor("norm.weight").to(torch.bfloat16)
        for li in range(DLAYERS):
            dp = f"layers.{li}"
            w[f"in_w.{li}"] = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            w[f"pa_w.{li}"] = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)
            w[f"qw.{li}"] = f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"kw.{li}"] = f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"vw.{li}"] = f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"ow.{li}"] = f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"qnw.{li}"] = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            w[f"knw.{li}"] = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)
            w[f"gw.{li}"] = f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"uw.{li}"] = f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"fc2.{li}"] = f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Random inputs
    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    target_hidden = torch.randn(ctx_len, N_CTX_LAYERS * HIDDEN).to(torch.bfloat16) * 0.1

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    cos_q = torch.cos(torch.outer(torch.arange(ctx_len, ctx_len + BSIZE, dtype=torch.float32), freqs))
    sin_q = torch.sin(torch.outer(torch.arange(ctx_len, ctx_len + BSIZE, dtype=torch.float32), freqs))
    cos_kv = torch.cos(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))
    sin_kv = torch.sin(torch.outer(torch.arange(kv_len, dtype=torch.float32), freqs))

    # PyTorch reference
    print("\nPyTorch reference...")
    ctx_bf = torch_prepare_context(target_hidden, w["fc_w"], w["hn_w"])
    ref_h = noise_bf.float()
    for li in range(DLAYERS):
        ref_h = torch_layer_fwd(ref_h, ctx_bf.float(), w, li, cos_q, sin_q, cos_kv, sin_kv)
    ref_final = torch_rmsnorm(ref_h, w["fn_w"].float())

    # Device forward
    print("\nDevice forward (TTNN ops)...")
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        q_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
        k_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)

        # Upload weights (ttnn.rms_norm uses (1, dim) weight)
        print("  Uploading weights...")
        t0 = time.time()
        dw = {}
        dw["fc_w"] = to_dev(w["fc_w"], d)
        dw["hn_w"] = to_dev(w["hn_w"].unsqueeze(0), d)
        dw["fn_w"] = to_dev(w["fn_w"].unsqueeze(0), d)
        for li in range(DLAYERS):
            dw[f"in_w.{li}"] = to_dev(w[f"in_w.{li}"].unsqueeze(0), d)
            dw[f"pa_w.{li}"] = to_dev(w[f"pa_w.{li}"].unsqueeze(0), d)
            dw[f"qw.{li}"] = to_dev(w[f"qw.{li}"], d)
            dw[f"kw.{li}"] = to_dev(w[f"kw.{li}"], d)
            dw[f"vw.{li}"] = to_dev(w[f"vw.{li}"], d)
            dw[f"ow.{li}"] = to_dev(w[f"ow.{li}"], d)
            dw[f"qnw.{li}"] = to_dev(w[f"qnw.{li}"].unsqueeze(0), d)
            dw[f"knw.{li}"] = to_dev(w[f"knw.{li}"].unsqueeze(0), d)
            dw[f"gw.{li}"] = to_dev(w[f"gw.{li}"], d)
            dw[f"uw.{li}"] = to_dev(w[f"uw.{li}"], d)
            dw[f"fc2.{li}"] = to_dev(w[f"fc2.{li}"], d)

        # RoPE tables (for TT-Lang rope kernel)
        max_seq = 256
        pos_t = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(pos_t, freqs)
        cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
        dw["rope_cos_q"] = to_dev(cos_full[ctx_sp:ctx_sp + SP], d)
        dw["rope_sin_q"] = to_dev(sin_adj[ctx_sp:ctx_sp + SP], d)
        dw["rope_cos_kv"] = to_dev(cos_full[:kv_sp], d)
        dw["rope_sin_kv"] = to_dev(sin_adj[:kv_sp], d)
        print(f"  Uploaded in {time.time()-t0:.1f}s")

        # Prepare context
        print("  Preparing context...")
        target_hidden_dev = to_dev(target_hidden, d)
        ctx_proj = ttnn.matmul(target_hidden_dev, dw["fc_w"])
        ctx_norm = ttnn.rms_norm(ctx_proj, weight=dw["hn_w"], epsilon=EPS)
        tt_ctx = ttnn.to_torch(ctx_norm).float()[:ctx_len, :HIDDEN]
        p = pcc(ctx_bf, tt_ctx)
        print(f"  Context PCC={p:.6f}")

        # Run 8 layers
        print("\n  8-layer forward (TTNN ops)...")
        h = to_dev(noise_bf, d)
        ref_h_per_layer = noise_bf.float()
        for li in range(DLAYERS):
            print(f"    Layer {li}...", end=" ", flush=True)
            h = dev_layer_fwd_ttnn(h, ctx_norm, dw, li, q_rope, k_rope, kv_sp, d)
            ref_h_per_layer = torch_layer_fwd(ref_h_per_layer, ctx_bf.float(), w, li,
                                              cos_q, sin_q, cos_kv, sin_kv)
            tt_h = ttnn.to_torch(h).float()[:BSIZE, :HIDDEN]
            p = pcc(ref_h_per_layer, tt_h)
            print(f"PCC={p:.6f} {'OK' if p > 0.95 else 'BAD'}")

        # Final norm
        final = ttnn.rms_norm(h, weight=dw["fn_w"], epsilon=EPS)
        tt_out = ttnn.to_torch(final).float()[:BSIZE, :HIDDEN]
        p = pcc(ref_final, tt_out)
        print(f"\n  Final PCC: {p:.6f} {'PASS' if p > 0.95 else 'FAIL'}")

        # Performance
        n_warmup = 3
        n_timed = 10
        print(f"\n  Performance ({n_warmup} warmup, {n_timed} timed)...")

        def run_full():
            hh = to_dev(noise_bf, d)
            for li in range(DLAYERS):
                hh = dev_layer_fwd_ttnn(hh, ctx_norm, dw, li, q_rope, k_rope, kv_sp, d)
            ttnn.rms_norm(hh, weight=dw["fn_w"], epsilon=EPS)
            ttnn.synchronize_device(d)

        for _ in range(n_warmup):
            run_full()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            run_full()
        elapsed = time.perf_counter() - t0
        per_fwd = elapsed / n_timed * 1000
        print(f"    {per_fwd:.1f} ms/forward ({n_timed} runs)")
        print(f"    {per_fwd / DLAYERS:.1f} ms/layer")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
