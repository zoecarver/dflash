"""DFlash draft model: TTNN forward + KV cache perf test at 250k context.

Tests both non-cached and cached forward at long context lengths.
"""
import sys
sys.path.insert(0, "/tmp")
import time
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from rope import make_rope_kernel
from dflash_draft import (
    draft_fwd_cached, setup_rope_tables_cached, prepare_context_ttnn,
    load_draft_weights, setup_rope_tables, draft_fwd_ttnn,
    prealloc_cached_scratch, crop_cache,
    to_dev as dd_to_dev, _tile_pad,
)

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
    normed = ttnn.rms_norm(h, weight=dw[f"in_w.{li}"], epsilon=EPS)

    q = ttnn.matmul(normed, dw[f"qw.{li}"])
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)
    k = ttnn.matmul(kv_in, dw[f"kw.{li}"])
    v = ttnn.matmul(kv_in, dw[f"vw.{li}"])

    q_flat = ttnn.reshape(q, (SP * NQH, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=dw[f"qnw.{li}"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=dw[f"knw.{li}"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))

    q_roped = to_dev(torch.zeros(SP, NQH * HDIM), d)
    k_roped = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    q_rope_k(q_normed, dw["rope_cos_q"], dw["rope_sin_q"], q_roped)
    k_rope_k(k_normed, dw["rope_cos_kv"], dw["rope_sin_kv"], k_roped)

    # SDPA: fused attention
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH, HDIM)), 1, 2)
    attn = ttnn.transformer.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    attn_flat = ttnn.reshape(ttnn.transpose(attn, 1, 2), (SP, NQH * HDIM))

    o = ttnn.matmul(attn_flat, dw[f"ow.{li}"])
    h = ttnn.add(h, o)

    normed2 = ttnn.rms_norm(h, weight=dw[f"pa_w.{li}"], epsilon=EPS)
    gate = ttnn.matmul(normed2, dw[f"gw.{li}"])
    up = ttnn.matmul(normed2, dw[f"uw.{li}"])
    act = ttnn.mul(ttnn.silu(gate), up)
    down = ttnn.matmul(act, dw[f"fc2.{li}"])

    h = ttnn.add(h, down)
    return h


def main():
    torch.manual_seed(42)
    ctx_len = 120_000
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
    print(f"Generating random inputs (ctx_len={ctx_len})...")
    noise_bf = torch.randn(BSIZE, HIDDEN).to(torch.bfloat16) * 0.1
    # Generate random context (skip PyTorch reference -- too slow at 250k)
    target_hidden = torch.randn(ctx_len, N_CTX_LAYERS * HIDDEN).to(torch.bfloat16) * 0.1

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))

    # Device forward
    print(f"\nDevice forward (TTNN ops, ctx={ctx_len})...")
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    d = ttnn.open_device(device_id=0, worker_l1_size=default_size - 131072)
    try:
        q_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
        k_rope = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)

        # Upload weights
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

        # RoPE tables
        max_seq = ctx_sp + SP + TILE
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

        # Prepare context on device
        print("  Preparing context...")
        t0 = time.time()
        target_hidden_dev = to_dev(target_hidden, d)
        ctx_proj = ttnn.matmul(target_hidden_dev, dw["fc_w"])
        ctx_norm = ttnn.rms_norm(ctx_proj, weight=dw["hn_w"], epsilon=EPS)
        ttnn.synchronize_device(d)
        print(f"  Context prepared in {time.time()-t0:.1f}s")

        # Run 8 layers (single pass, no PCC -- reference too slow at 250k)
        print("\n  8-layer forward...")
        t0 = time.time()
        h = to_dev(noise_bf, d)
        for li in range(DLAYERS):
            print(f"    Layer {li}...", flush=True)
            h = dev_layer_fwd_ttnn(h, ctx_norm, dw, li, q_rope, k_rope, kv_sp, d)
        final = ttnn.rms_norm(h, weight=dw["fn_w"], epsilon=EPS)
        ttnn.synchronize_device(d)
        first_pass = time.time() - t0
        tt_out = ttnn.to_torch(final).float()[:BSIZE, :HIDDEN]
        print(f"\n  First pass: {first_pass:.1f}s")
        print(f"  Output range: [{tt_out.min():.4f}, {tt_out.max():.4f}]")

        # Performance
        n_warmup = 2
        n_timed = 5
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
        print(f"    Non-cached: {per_fwd:.1f} ms/forward ({n_timed} runs)")
        print(f"    Non-cached: {per_fwd / DLAYERS:.1f} ms/layer")

        # ----- KV-cached forward -----
        # Simulate: full context already cached, small incremental update
        print(f"\n  === KV-cached forward ===")
        new_accepted = 3  # typical acceptance
        new_ctx_sp = _tile_pad(new_accepted)
        new_ctx_data = torch.randn(new_accepted, N_CTX_LAYERS * HIDDEN).to(torch.bfloat16) * 0.1
        if new_ctx_sp > new_accepted:
            pad = new_ctx_data[-1:].expand(new_ctx_sp - new_accepted, -1)
            new_ctx_data = torch.cat([new_ctx_data, pad], dim=0)

        # Load draft weights into a dict compatible with cached forward
        # (reuse the already-loaded weights on device)
        cw = {}
        cw["fc_w"] = dw["fc_w"]
        cw["hn_w"] = dw["hn_w"]
        cw["fn_w"] = dw["fn_w"]
        cw["rope_cos_full"] = cos_full
        cw["rope_sin_full"] = sin_adj
        cw["sc"] = to_dev(torch.ones(TILE, TILE), d)
        cw["ms"] = to_dev(torch.full((TILE, TILE), 1.0 / HIDDEN), d)
        for li in range(DLAYERS):
            for key in ["in_w", "pa_w", "qw", "kw", "vw", "ow", "qnw", "knw", "gw", "uw", "fc2"]:
                cw[f"{key}.{li}"] = dw[f"{key}.{li}"]
        for ci in range(N_CTX_LAYERS):
            cw[f"fc_w.{ci}"] = to_dev(
                w["fc_w"][ci * HIDDEN:(ci + 1) * HIDDEN, :], d)

        # First call: populate cache with full context (like prefill)
        print(f"  Populating cache (ctx={ctx_len})...")
        setup_rope_tables_cached(cw, 0, ctx_sp, d, q_start=ctx_len)
        noise_dev_c = to_dev(noise_bf, d)
        ctx_proj = prepare_context_ttnn(to_dev(target_hidden, d), cw, d)
        t0 = time.time()
        _, cache = draft_fwd_cached(noise_dev_c, ctx_proj, cw, d, None)
        ttnn.synchronize_device(d)
        print(f"  Cache populated in {time.time()-t0:.1f}s")
        print(f"  Cache K/V rows: {cache[0]['k'].shape[2]} (ctx+noise)")

        # Crop: simulate accepting 3 of 16 noise tokens
        # Use exact (non-tile-aligned) row count to avoid stale rejected-noise
        # K/V rows corrupting subsequent attention
        real_pos = ctx_len + new_accepted + 1  # accepted + target correction
        cache_rows = real_pos  # exact, not tile-padded
        cache = crop_cache(cache, cache_rows)
        ttnn.synchronize_device(d)
        print(f"  After crop: {cache_rows} rows (exact)")

        # Subsequent calls: small incremental update with crop
        new_ctx_dev = prepare_context_ttnn(
            to_dev(new_ctx_data, d), cw, d)
        setup_rope_tables_cached(cw, cache_rows, new_ctx_sp, d,
                                 q_start=real_pos + new_accepted)
        sc = prealloc_cached_scratch(new_ctx_sp + SP, d)

        print(f"  Cached forward (new_ctx={new_ctx_sp}, cache={cache_rows})...")
        n_warmup_c = 2
        n_timed_c = 5

        for _ in range(n_warmup_c):
            noise_dev_c = to_dev(noise_bf, d)
            _, tmp_cache = draft_fwd_cached(noise_dev_c, new_ctx_dev, cw, d, cache, sc)
            ttnn.synchronize_device(d)

        t0 = time.perf_counter()
        for _ in range(n_timed_c):
            noise_dev_c = to_dev(noise_bf, d)
            _, tmp_cache = draft_fwd_cached(noise_dev_c, new_ctx_dev, cw, d, cache, sc)
            ttnn.synchronize_device(d)
        elapsed_c = time.perf_counter() - t0
        per_fwd_c = elapsed_c / n_timed_c * 1000
        print(f"    Cached: {per_fwd_c:.1f} ms/forward ({n_timed_c} runs)")
        print(f"    Cached: {per_fwd_c / DLAYERS:.1f} ms/layer")
        print(f"    Speedup: {per_fwd / per_fwd_c:.1f}x vs non-cached")

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
