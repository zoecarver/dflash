"""DFlash draft model: 8-layer cross-attention on Tenstorrent device.

All compute runs via TT-Lang fused kernels. TTNN used only for SDPA,
reshape/transpose (SDPA interface), and KV cache management.
"""
import time
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel
from rope import make_rope_kernel
from softmax import make_softmax_kernel
from matmul_residual_add import make_matmul_residual_add_kernel
from matmul_silu_mul import make_matmul_silu_mul_kernel
from streaming_matmul import make_matmul_kernel

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
VOCAB = 151936

DLAYERS = 8
DINTER = 6144
BSIZE = 16
SP = ((BSIZE + TILE - 1) // TILE) * TILE
N_CTX_LAYERS = 5
TLAYER_IDS = [1, 12, 23, 34, 45]
MASK_ID = 151669
DRAFT_DIR = "/home/zcarver/dflash"

# When True, cache noise K/V (matching reference DynamicCache behavior).
# When False, only cache context K/V (may improve acceptance at bf16).
CACHE_NOISE = True

# Per-operation timing accumulator
_op_times = {}
_op_counts = {}
_timing_enabled = False


def enable_op_timing():
    global _timing_enabled
    _timing_enabled = True
    _op_times.clear()
    _op_counts.clear()


def disable_op_timing():
    global _timing_enabled
    _timing_enabled = False


def print_op_timing():
    if not _op_times:
        print("  (no op timings recorded)")
        return
    print(f"  {'Operation':<25} {'Total (ms)':>10} {'Count':>6} {'Avg (ms)':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*10}")
    for name in sorted(_op_times, key=lambda k: -_op_times[k]):
        total = _op_times[name] * 1000
        count = _op_counts[name]
        avg = total / count
        print(f"  {name:<25} {total:>10.2f} {count:>6} {avg:>10.3f}")
    grand = sum(_op_times.values()) * 1000
    print(f"  {'TOTAL':<25} {grand:>10.2f}")


def _timed_call(name, device, fn, *args):
    if not _timing_enabled:
        return fn(*args)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    result = fn(*args)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    _op_times[name] = _op_times.get(name, 0) + elapsed
    _op_counts[name] = _op_counts.get(name, 0) + 1
    return result


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


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
    kw = {}
    if isinstance(d, ttnn.MeshDevice):
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(d)
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)


# ---------------------------------------------------------------------------
# Kernel instances
# ---------------------------------------------------------------------------
norm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
head_norm_k = make_rmsnorm_kernel(dim_tiles=HDIM_TILES, eps=EPS)
q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)
o_proj_resadd_k = make_matmul_residual_add_kernel(k_tiles=NQH * HDIM_TILES)
down_proj_resadd_k = make_matmul_residual_add_kernel(k_tiles=DINTER // TILE)
gate_up_silu_k = make_matmul_silu_mul_kernel(k_tiles=HTILES)
q_proj_k = make_matmul_kernel(k_tiles=HTILES)
kv_proj_k = make_matmul_kernel(k_tiles=HTILES)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_draft_weights(d):
    """Load draft model weights from safetensors onto device. Returns weight dict."""
    t0 = time.time()
    print("Loading draft weights...")
    w = {}
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        fc_full = f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16)  # (10240, 2048)
        w["fc_w"] = to_dev(fc_full, d)
        # Split FC into per-layer chunks for higher precision accumulation
        for ci in range(N_CTX_LAYERS):
            chunk = fc_full[ci * HIDDEN:(ci + 1) * HIDDEN, :]  # (2048, 2048)
            w[f"fc_w.{ci}"] = to_dev(chunk, d)
        hn_w = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        w["hn_w_tt"] = to_dev(hn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        w["hn_w"] = to_dev(hn_w.unsqueeze(0), d)
        fn_w = f.get_tensor("norm.weight").to(torch.bfloat16)
        w["fn_w_tt"] = to_dev(fn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        w["fn_w"] = to_dev(fn_w.unsqueeze(0), d)

        for li in range(DLAYERS):
            dp = f"layers.{li}"
            in_w = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            pa_w = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)
            w[f"in_w_tt.{li}"] = to_dev(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"pa_w_tt.{li}"] = to_dev(pa_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"in_w.{li}"] = to_dev(in_w.unsqueeze(0), d)
            w[f"pa_w.{li}"] = to_dev(pa_w.unsqueeze(0), d)
            w[f"qw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"kw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"vw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"ow.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16), d)
            qnw = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            knw = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)
            w[f"qnw.{li}"] = to_dev(qnw.unsqueeze(0).contiguous(), d)
            w[f"knw.{li}"] = to_dev(knw.unsqueeze(0).contiguous(), d)
            w[f"qnw_tt.{li}"] = to_dev(qnw.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"knw_tt.{li}"] = to_dev(knw.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"gw.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"uw.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"fc2.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16), d)

    # RoPE tables and support tensors
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    # Sized for max expected position; cached forward grows with each step
    max_seq = 512
    pos = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_adj = sin_full.clone()
    sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
    w["rope_cos_full"] = cos_full
    w["rope_sin_full"] = sin_adj

    w["sc"] = to_dev(torch.ones(TILE, TILE), d)
    w["ms"] = to_dev(torch.full((TILE, TILE), 1.0 / HIDDEN), d)
    w["ms_head"] = to_dev(torch.full((TILE, TILE), 1.0 / HDIM), d)

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return w


def _build_attn_mask(kv_sp, real_positions, d):
    """Build additive attention mask for SDPA: 0 = attend, -inf = ignore.

    kv_sp: total (tile-padded) K/V sequence length
    real_positions: list of (start, end) ranges that are real (non-padding)
    d: device
    Returns: ttnn tensor of shape (1, 1, SP, kv_sp) in bfloat16
    """
    # Start with large negative (masked), set real positions to 0
    NEG_INF = -1e9  # bf16-safe large negative; true -inf can cause NaN
    mask = torch.full((1, 1, SP, kv_sp), NEG_INF, dtype=torch.bfloat16)
    for s, e in real_positions:
        mask[:, :, :, s:e] = 0.0  # all Q rows (including padding) attend to real K/V
    kw = {}
    if isinstance(d, ttnn.MeshDevice):
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(d)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)


def setup_rope_tables(w, ctx_sp, d, q_start=None, ctx_real=None):
    """Upload RoPE tables sized for a given tile-padded context length.

    ctx_sp: tile-padded context rows. kv_sp = ctx_sp + SP (context + draft).
    q_start: actual start position for Q/noise tokens (default: ctx_sp).
             When ctx_sp > q_start due to tile padding, this corrects the RoPE
             positions so noise tokens get their true positions.
    ctx_real: real (unpadded) context length. When provided, builds an attention
              mask to ignore tile-padding positions in SDPA.
    """
    if q_start is None:
        q_start = ctx_sp
    kv_sp = ctx_sp + SP
    # Q: actual positions q_start..q_start+SP-1
    w["rope_cos_q"] = to_dev(w["rope_cos_full"][q_start:q_start + SP], d)
    w["rope_sin_q"] = to_dev(w["rope_sin_full"][q_start:q_start + SP], d)
    # KV: context positions 0..ctx_sp-1, noise positions q_start..q_start+SP-1
    kv_cos = torch.cat([w["rope_cos_full"][:ctx_sp],
                        w["rope_cos_full"][q_start:q_start + SP]], dim=0)
    kv_sin = torch.cat([w["rope_sin_full"][:ctx_sp],
                        w["rope_sin_full"][q_start:q_start + SP]], dim=0)
    w["rope_cos_kv"] = to_dev(kv_cos, d)
    w["rope_sin_kv"] = to_dev(kv_sin, d)
    w["kv_sp"] = kv_sp
    # Attention mask: mark real context + real noise positions
    if ctx_real is not None:
        real_positions = [(0, ctx_real), (ctx_sp, ctx_sp + BSIZE)]
        w["attn_mask"] = _build_attn_mask(kv_sp, real_positions, d)
    else:
        w["attn_mask"] = None


# ---------------------------------------------------------------------------
# Non-cached forward
# ---------------------------------------------------------------------------
def draft_layer_fwd_ttnn(h, ctx_dev, w, li, d):
    """Single layer forward using TT-Lang kernels."""
    kv_sp = w["kv_sp"]

    # Input RMSNorm
    normed = to_dev(torch.zeros(SP, HIDDEN), d)
    _timed_call("in_rmsnorm", d, norm_k, h, w[f"in_w_tt.{li}"], w["sc"], w["ms"], normed)

    # Q/K/V projections
    q = to_dev(torch.zeros(SP, NQH * HDIM), d)
    _timed_call("q_proj", d, q_proj_k, normed, w[f"qw.{li}"], q)
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)
    k = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    _timed_call("k_proj", d, kv_proj_k, kv_in, w[f"kw.{li}"], k)
    v = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    _timed_call("v_proj", d, kv_proj_k, kv_in, w[f"vw.{li}"], v)

    # QK-norm RMSNorm
    q_flat = ttnn.reshape(q, (SP * NQH, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH, HDIM))
    q_normed_flat = to_dev(torch.zeros(SP * NQH, HDIM), d)
    _timed_call("qk_rmsnorm", d, head_norm_k, q_flat, w[f"qnw_tt.{li}"], w["sc"], w["ms_head"], q_normed_flat)
    k_normed_flat = to_dev(torch.zeros(kv_sp * NKVH, HDIM), d)
    _timed_call("qk_rmsnorm", d, head_norm_k, k_flat, w[f"knw_tt.{li}"], w["sc"], w["ms_head"], k_normed_flat)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))

    # RoPE
    q_roped = to_dev(torch.zeros(SP, NQH * HDIM), d)
    k_roped = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    _timed_call("q_rope", d, q_rope_k, q_normed, w["rope_cos_q"], w["rope_sin_q"], q_roped)
    _timed_call("k_rope", d, k_rope_k, k_normed, w["rope_cos_kv"], w["rope_sin_kv"], k_roped)

    # SDPA
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH, HDIM)), 1, 2)
    attn_mask = w.get("attn_mask")
    attn = ttnn.transformer.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, attn_mask=attn_mask)
    attn_flat = ttnn.reshape(ttnn.transpose(attn, 1, 2), (SP, NQH * HDIM))

    # Fused o_proj matmul + residual add
    h_new = to_dev(torch.zeros(SP, HIDDEN), d)
    _timed_call("o_proj+resadd", d, o_proj_resadd_k, attn_flat, w[f"ow.{li}"], h, h_new)
    h = h_new

    # Post-attention RMSNorm
    normed2 = to_dev(torch.zeros(SP, HIDDEN), d)
    _timed_call("pa_rmsnorm", d, norm_k, h, w[f"pa_w_tt.{li}"], w["sc"], w["ms"], normed2)

    # Fused gate/up matmul + silu_mul
    act = to_dev(torch.zeros(SP, DINTER), d)
    _timed_call("gate_up+silu", d, gate_up_silu_k, normed2, w[f"gw.{li}"], w[f"uw.{li}"], act)

    # Fused down_proj matmul + residual add
    h_new2 = to_dev(torch.zeros(SP, HIDDEN), d)
    _timed_call("down+resadd", d, down_proj_resadd_k, act, w[f"fc2.{li}"], h, h_new2)
    return h_new2


def draft_fwd_ttnn(noise, ctx, w, d):
    """Full 8-layer draft forward."""
    h = noise
    for li in range(DLAYERS):
        h = draft_layer_fwd_ttnn(h, ctx, w, li, d)
    out = to_dev(torch.zeros(SP, HIDDEN), d)
    _timed_call("fn_rmsnorm", d, norm_k, h, w["fn_w_tt"], w["sc"], w["ms"], out)
    return out


def prepare_context_ttnn(target_hidden, w, d):
    """FC projection + hidden norm.

    Splits the 10240->2048 FC into 5 x 2048->2048 matmuls to reduce
    bf16 accumulation error (inner dim 2048 vs 10240).
    """
    ctx_sp = target_hidden.shape[0]
    # Split target_hidden along feature dim, project each chunk separately
    parts = []
    for ci in range(N_CTX_LAYERS):
        sl = ttnn.slice(target_hidden, [0, ci * HIDDEN], [ctx_sp, (ci + 1) * HIDDEN])
        parts.append(ttnn.matmul(sl, w[f"fc_w.{ci}"]))
    ctx_proj = parts[0]
    for p in parts[1:]:
        ctx_proj = ttnn.add(ctx_proj, p)
    ctx_norm = to_dev(torch.zeros(ctx_sp, HIDDEN), d)
    norm_k(ctx_proj, w["hn_w_tt"], w["sc"], w["ms"], ctx_norm)
    return ctx_norm


# ---------------------------------------------------------------------------
# KV-cached forward
# ---------------------------------------------------------------------------
def prealloc_cached_scratch(new_kv_sp, d):
    """Pre-allocate reusable scratch tensors for cached forward.

    new_kv_sp: tile-padded (new_ctx_sp + SP), fixed during decode.
    """
    return {
        "q_roped": to_dev(torch.zeros(SP, NQH * HDIM), d),
        "new_k_roped": to_dev(torch.zeros(new_kv_sp, NKVH * HDIM), d),
    }


def setup_rope_tables_cached(w, cache_rows, new_ctx_sp, d, q_start=None,
                             new_ctx_real=None):
    """RoPE tables for cached forward.

    cache_rows: exact rows in cache (non-tile-aligned OK)
    new_ctx_sp: tile-padded new context rows
    q_start: actual position of noise tokens (default: cache_rows + new_ctx_sp).
    new_ctx_real: real (unpadded) new context length. When provided, builds an
                  attention mask and stores the real count for cache slicing.
    """
    new_kv_sp = new_ctx_sp + SP
    if q_start is None:
        q_start = cache_rows + new_ctx_sp
    # Q positions: noise tokens at their real positions
    w["rope_cos_q"] = to_dev(w["rope_cos_full"][q_start:q_start + SP], d)
    w["rope_sin_q"] = to_dev(w["rope_sin_full"][q_start:q_start + SP], d)
    # New KV: context starts after cache, noise matches Q positions
    kv_cos = torch.cat([w["rope_cos_full"][cache_rows:cache_rows + new_ctx_sp],
                        w["rope_cos_full"][q_start:q_start + SP]], dim=0)
    kv_sin = torch.cat([w["rope_sin_full"][cache_rows:cache_rows + new_ctx_sp],
                        w["rope_sin_full"][q_start:q_start + SP]], dim=0)
    w["rope_cos_new_kv"] = to_dev(kv_cos, d)
    w["rope_sin_new_kv"] = to_dev(kv_sin, d)
    w["new_kv_sp"] = new_kv_sp
    w["total_kv"] = cache_rows + new_kv_sp
    # Store real context count for cache slicing in CACHE_NOISE=False path
    w["new_ctx_real"] = new_ctx_real if new_ctx_real is not None else new_ctx_sp
    # Attention mask: cache rows are all real (exact), new has padding
    if new_ctx_real is not None:
        total_kv = cache_rows + new_kv_sp
        new_kv_offset = cache_rows
        real_positions = [
            (0, cache_rows),                                              # cached K/V
            (new_kv_offset, new_kv_offset + new_ctx_real),                # new context
            (new_kv_offset + new_ctx_sp, new_kv_offset + new_ctx_sp + BSIZE),  # noise
        ]
        w["attn_mask"] = _build_attn_mask(total_kv, real_positions, d)
    else:
        w["attn_mask"] = None


def _layer_fwd_cached(h, new_ctx, w, li, d, layer_cache, s):
    """Single layer forward with KV cache.

    h: noise hidden states (SP, HIDDEN)
    new_ctx: new context rows (new_ctx_sp, HIDDEN), already projected+normed
    layer_cache: {"k": (1, NKVH, cache_len, HDIM), "v": same} in 4D or None
    s: scratch buffers dict
    Returns: (h_out, updated_layer_cache)
    """
    new_kv_sp = w["new_kv_sp"]
    cache_len = w["total_kv"] - new_kv_sp

    # Input RMSNorm
    _timed_call("in_rmsnorm", d, norm_k, h, w[f"in_w_tt.{li}"], w["sc"], w["ms"], s["normed"])

    # Q from noise only, K/V from new context + noise
    _timed_call("q_proj", d, q_proj_k, s["normed"], w[f"qw.{li}"], s["q"])
    new_kv_in = ttnn.concat([new_ctx, s["normed"]], dim=0)
    _timed_call("k_proj", d, kv_proj_k, new_kv_in, w[f"kw.{li}"], s["new_k"])
    _timed_call("v_proj", d, kv_proj_k, new_kv_in, w[f"vw.{li}"], s["new_v"])

    # QK-norm
    q_flat = ttnn.reshape(s["q"], (SP * NQH, HDIM))
    new_k_flat = ttnn.reshape(s["new_k"], (new_kv_sp * NKVH, HDIM))
    _timed_call("qk_rmsnorm", d, head_norm_k, q_flat, w[f"qnw_tt.{li}"], w["sc"], w["ms_head"], s["q_normed_flat"])
    _timed_call("qk_rmsnorm", d, head_norm_k, new_k_flat, w[f"knw_tt.{li}"], w["sc"], w["ms_head"], s["new_k_normed_flat"])
    q_normed = ttnn.reshape(s["q_normed_flat"], (SP, NQH * HDIM))
    new_k_normed = ttnn.reshape(s["new_k_normed_flat"], (new_kv_sp, NKVH * HDIM))

    # RoPE
    _timed_call("q_rope", d, q_rope_k, q_normed, w["rope_cos_q"], w["rope_sin_q"], s["q_roped"])
    _timed_call("k_rope", d, k_rope_k, new_k_normed, w["rope_cos_new_kv"], w["rope_sin_new_kv"], s["new_k_roped"])

    # Reshape new K/V to 4D for cache concat and SDPA
    new_k_4d = ttnn.transpose(ttnn.reshape(s["new_k_roped"], (1, new_kv_sp, NKVH, HDIM)), 1, 2)
    new_v_4d = ttnn.transpose(ttnn.reshape(s["new_v"], (1, new_kv_sp, NKVH, HDIM)), 1, 2)

    # Build full K/V in 4D: cached + new (concat along seq dim)
    if layer_cache is not None:
        k4 = ttnn.concat([layer_cache["k"], new_k_4d], dim=2)
        v4 = ttnn.concat([layer_cache["v"], new_v_4d], dim=2)
    else:
        k4 = new_k_4d
        v4 = new_v_4d

    # SDPA
    q4 = ttnn.transpose(ttnn.reshape(s["q_roped"], (1, SP, NQH, HDIM)), 1, 2)
    attn_mask = w.get("attn_mask")
    attn = ttnn.transformer.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, attn_mask=attn_mask)
    attn_flat = ttnn.reshape(ttnn.transpose(attn, 1, 2), (SP, NQH * HDIM))

    # Fused o_proj matmul + residual add
    _timed_call("o_proj+resadd", d, o_proj_resadd_k, attn_flat, w[f"ow.{li}"], h, s["h_new"])

    # Post-attention RMSNorm
    _timed_call("pa_rmsnorm", d, norm_k, s["h_new"], w[f"pa_w_tt.{li}"], w["sc"], w["ms"], s["normed2"])

    # Fused gate/up matmul + silu_mul
    _timed_call("gate_up+silu", d, gate_up_silu_k, s["normed2"], w[f"gw.{li}"], w[f"uw.{li}"], s["act"])

    # Fused down_proj matmul + residual add
    _timed_call("down+resadd", d, down_proj_resadd_k, s["act"], w[f"fc2.{li}"], s["h_new"], s["h_new2"])

    # Cache K/V
    if CACHE_NOISE:
        updated = {"k": k4, "v": v4}
    else:
        new_ctx_real_count = w.get("new_ctx_real", new_ctx.shape[0])
        ctx_end = cache_len + new_ctx_real_count
        updated = {
            "k": ttnn.slice(k4, [0, 0, 0, 0], [1, NKVH, ctx_end, HDIM]),
            "v": ttnn.slice(v4, [0, 0, 0, 0], [1, NKVH, ctx_end, HDIM]),
        }

    return s["h_new2"], updated


# ---------------------------------------------------------------------------
# DFlash class: init + step with pre-allocated scratch
# ---------------------------------------------------------------------------
class DFlashDraft:
    """DFlash draft model with pre-allocated scratch buffers.

    Usage:
        draft = DFlashDraft(device)
        draft.alloc_scratch(new_kv_sp)
        ctx = draft.prepare_context(target_hidden)
        draft.setup_rope_cached(cache_rows, new_ctx_sp, ...)
        out, cache = draft.step(noise, new_ctx, cache)
    """

    def __init__(self, d):
        self.d = d
        self.w = load_draft_weights(d)
        self._scratch = None

    def alloc_scratch(self, new_kv_sp):
        """Pre-allocate output buffers for cached forward. Call once before decode."""
        d = self.d
        z = lambda *shape: to_dev(torch.zeros(*shape), d)
        self._scratch = {
            # Fixed size (SP-based), reused every layer
            "normed": z(SP, HIDDEN),
            "q": z(SP, NQH * HDIM),
            "q_normed_flat": z(SP * NQH, HDIM),
            "q_roped": z(SP, NQH * HDIM),
            "h_new": z(SP, HIDDEN),
            "normed2": z(SP, HIDDEN),
            "act": z(SP, DINTER),
            "h_new2": z(SP, HIDDEN),
            "fn_out": z(SP, HIDDEN),
            # Variable size (new_kv_sp-based)
            "new_k": z(new_kv_sp, NKVH * HDIM),
            "new_v": z(new_kv_sp, NKVH * HDIM),
            "new_k_normed_flat": z(new_kv_sp * NKVH, HDIM),
            "new_k_roped": z(new_kv_sp, NKVH * HDIM),
        }

    def prepare_context(self, target_hidden):
        """FC projection + hidden norm on target hidden states."""
        return prepare_context_ttnn(target_hidden, self.w, self.d)

    def setup_rope(self, cache_rows, new_ctx_sp, q_start=None, new_ctx_real=None):
        """Upload RoPE tables for cached forward."""
        setup_rope_tables_cached(self.w, cache_rows, new_ctx_sp, self.d,
                                 q_start=q_start, new_ctx_real=new_ctx_real)

    def step(self, noise, new_ctx, cache):
        """One cached forward pass through 8 layers + final norm.

        Returns: (output, updated_cache)
        """
        s = self._scratch
        w, d = self.w, self.d
        h = noise
        new_cache = []
        for li in range(DLAYERS):
            lc = cache[li] if cache is not None else None
            h, updated_lc = _layer_fwd_cached(h, new_ctx, w, li, d, lc, s)
            new_cache.append(updated_lc)
        _timed_call("fn_rmsnorm", d, norm_k, h, w["fn_w_tt"], w["sc"], w["ms"], s["fn_out"])
        return s["fn_out"], new_cache


def draft_fwd_cached(noise, new_ctx, w, d, cache, scratch=None):
    """Backward-compat wrapper: 8-layer cached forward without class.

    Allocates output buffers inline. For production use DFlashDraft.step()
    which uses pre-allocated scratch.
    """
    new_kv_sp = w["new_kv_sp"]
    if scratch is None:
        scratch = {}
    z = lambda *shape: to_dev(torch.zeros(*shape), d)
    s = {
        "normed": z(SP, HIDDEN),
        "q": z(SP, NQH * HDIM),
        "q_normed_flat": z(SP * NQH, HDIM),
        "q_roped": scratch.get("q_roped", z(SP, NQH * HDIM)),
        "h_new": z(SP, HIDDEN),
        "normed2": z(SP, HIDDEN),
        "act": z(SP, DINTER),
        "h_new2": z(SP, HIDDEN),
        "new_k": z(new_kv_sp, NKVH * HDIM),
        "new_v": z(new_kv_sp, NKVH * HDIM),
        "new_k_normed_flat": z(new_kv_sp * NKVH, HDIM),
        "new_k_roped": scratch.get("new_k_roped", z(new_kv_sp, NKVH * HDIM)),
        "fn_out": z(SP, HIDDEN),
    }
    h = noise
    new_cache = []
    for li in range(DLAYERS):
        lc = cache[li] if cache is not None else None
        h, updated_lc = _layer_fwd_cached(h, new_ctx, w, li, d, lc, s)
        new_cache.append(updated_lc)
    _timed_call("fn_rmsnorm", d, norm_k, h, w["fn_w_tt"], w["sc"], w["ms"], s["fn_out"])
    return s["fn_out"], new_cache


def crop_cache(cache, keep_rows):
    """Crop KV cache to keep_rows.

    Called after acceptance to remove rejected noise K/V.
    Keeps context + accepted noise, discards rejected noise positions.
    keep_rows need not be tile-aligned; TTNN handles internal tile padding.
    """
    return [
        {
            "k": ttnn.slice(lc["k"], [0, 0, 0, 0], [1, NKVH, keep_rows, HDIM]),
            "v": ttnn.slice(lc["v"], [0, 0, 0, 0], [1, NKVH, keep_rows, HDIM]),
        }
        for lc in cache
    ]
