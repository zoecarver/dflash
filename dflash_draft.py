"""DFlash draft model: 8-layer cross-attention on single Tenstorrent device.

Stacked Q heads + TTNN matmul + TT-Lang softmax for GQA attention.
All ops on device, zero host readbacks in the forward pass.
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
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"


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
q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH)
k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_draft_weights(d):
    """Load draft model weights from safetensors onto device. Returns weight dict."""
    t0 = time.time()
    print("Loading draft weights...")
    w = {}
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        w["fc_w"] = to_dev(f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16), d)
        hn_w = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        w["hn_w_tt"] = to_dev(hn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        fn_w = f.get_tensor("norm.weight").to(torch.bfloat16)
        w["fn_w_tt"] = to_dev(fn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

        for li in range(DLAYERS):
            dp = f"layers.{li}"
            in_w = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            pa_w = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)
            w[f"in_w_tt.{li}"] = to_dev(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"pa_w_tt.{li}"] = to_dev(pa_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"qw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"kw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"vw.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"ow.{li}"] = to_dev(f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16), d)
            qnw = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            knw = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)
            w[f"qnw.{li}"] = to_dev(qnw.unsqueeze(0).contiguous(), d)
            w[f"knw.{li}"] = to_dev(knw.unsqueeze(0).contiguous(), d)
            w[f"gw.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"uw.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16), d)
            w[f"fc2.{li}"] = to_dev(f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16), d)

    # RoPE tables and support tensors
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    # Sized for max expected kv_sp; will be sliced if ctx changes
    max_seq = 256
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

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return w


def setup_rope_tables(w, ctx_sp, d):
    """Upload RoPE tables sized for a given tile-padded context length.

    ctx_sp: tile-padded context rows. kv_sp = ctx_sp + SP (context + draft).
    """
    kv_sp = ctx_sp + SP
    w["rope_cos_q"] = to_dev(w["rope_cos_full"][ctx_sp:ctx_sp + SP], d)
    w["rope_sin_q"] = to_dev(w["rope_sin_full"][ctx_sp:ctx_sp + SP], d)
    w["rope_cos_kv"] = to_dev(w["rope_cos_full"][:kv_sp], d)
    w["rope_sin_kv"] = to_dev(w["rope_sin_full"][:kv_sp], d)
    w["kv_sp"] = kv_sp
    total_q_rows = NKVH * GQA * SP
    w["softmax_k"] = make_softmax_kernel(total_q_rows // TILE, kv_sp // TILE)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
def draft_layer_fwd(h, ctx_dev, w, li, d):
    """Single layer forward on device."""
    scale = 1.0 / (HDIM ** 0.5)
    kv_sp = w["kv_sp"]

    normed = to_dev(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"in_w_tt.{li}"], w["sc"], w["ms"], normed)

    q = ttnn.matmul(normed, w[f"qw.{li}"])
    kv_in = ttnn.concat([ctx_dev, normed], dim=0)
    k = ttnn.matmul(kv_in, w[f"kw.{li}"])
    v = ttnn.matmul(kv_in, w[f"vw.{li}"])

    q_flat = ttnn.reshape(q, (SP * NQH, HDIM))
    k_flat = ttnn.reshape(k, (kv_sp * NKVH, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=w[f"qnw.{li}"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=w[f"knw.{li}"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (SP, NQH * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH * HDIM))

    q_roped = to_dev(torch.zeros(SP, NQH * HDIM), d)
    k_roped = to_dev(torch.zeros(kv_sp, NKVH * HDIM), d)
    q_rope_k(q_normed, w["rope_cos_q"], w["rope_sin_q"], q_roped)
    k_rope_k(k_normed, w["rope_cos_kv"], w["rope_sin_kv"], k_roped)

    # Stacked Q + batched matmul + TT-Lang softmax for GQA
    q4 = ttnn.transpose(ttnn.reshape(q_roped, (1, SP, NQH, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_roped, (1, kv_sp, NKVH, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v, (1, kv_sp, NKVH, HDIM)), 1, 2)
    q_grouped = ttnn.reshape(q4, (1, NKVH, GQA * SP, HDIM))
    k_t = ttnn.transpose(k4, -2, -1)
    scores = ttnn.matmul(q_grouped, k_t)
    scores = ttnn.multiply(scores, scale)

    total_q_rows = NKVH * GQA * SP
    scores_flat = ttnn.reshape(scores, (total_q_rows, kv_sp))
    probs_flat = to_dev(torch.zeros(total_q_rows, kv_sp), d)
    w["softmax_k"](scores_flat, w["sc"], probs_flat)

    probs_4d = ttnn.reshape(probs_flat, (1, NKVH, GQA * SP, kv_sp))
    attn_4d = ttnn.matmul(probs_4d, v4)
    attn_heads = ttnn.reshape(attn_4d, (1, NQH, SP, HDIM))
    attn_flat = ttnn.reshape(ttnn.transpose(attn_heads, 1, 2), (SP, NQH * HDIM))

    o = ttnn.matmul(attn_flat, w[f"ow.{li}"])

    add_out = to_dev(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, o, add_out)
    h = add_out

    normed2 = to_dev(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w[f"pa_w_tt.{li}"], w["sc"], w["ms"], normed2)

    gate = ttnn.matmul(normed2, w[f"gw.{li}"])
    up = ttnn.matmul(normed2, w[f"uw.{li}"])
    act = ttnn.zeros_like(gate)
    silu_mul_kernel(gate, up, act)
    down = ttnn.matmul(act, w[f"fc2.{li}"])

    add_out2 = to_dev(torch.zeros(SP, HIDDEN), d)
    residual_add_kernel(h, down, add_out2)
    return add_out2


def draft_fwd(noise, ctx, w, d):
    """Full 8-layer draft forward. Returns (SP, HIDDEN) after final norm."""
    h = noise
    for li in range(DLAYERS):
        h = draft_layer_fwd(h, ctx, w, li, d)
    out = to_dev(torch.zeros(SP, HIDDEN), d)
    norm_k(h, w["fn_w_tt"], w["sc"], w["ms"], out)
    return out


def prepare_context(target_hidden, w, d):
    """FC projection + hidden norm on target hidden states.

    Args:
        target_hidden: (ctx_sp, N_CTX_LAYERS * HIDDEN) on device
        w: weight dict
        d: device

    Returns:
        (ctx_sp, HIDDEN) on device
    """
    ctx_proj = ttnn.matmul(target_hidden, w["fc_w"])
    ctx_sp = ctx_proj.shape[0]
    ctx_norm = to_dev(torch.zeros(ctx_sp, HIDDEN), d)
    norm_k(ctx_proj, w["hn_w_tt"], w["sc"], w["ms"], ctx_norm)
    return ctx_norm
