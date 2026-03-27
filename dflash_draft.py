"""DFlash draft model (8-layer cross-attention) on Tenstorrent 4-chip TP.

Fully on-device forward pass: zero host readbacks in the hot loop.
Uses TTNN matmul for linear ops, TT-Lang kernels for everything else.

Cross-attention uses stacked Q heads for implicit GQA:
  Q_stacked: (NQH_TP * BSIZE, HDIM) -- all 8 Q heads share 1 KV head
  K: (kv_len, HDIM) per chip
  Attention: Q @ K^T -> softmax -> probs @ V, all via TTNN matmul + TT-Lang softmax
"""
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from silu_mul import silu_mul_kernel
from softmax import make_softmax_kernel
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from rope import make_rope_kernel

from device import (
    TILE, HIDDEN, HTILES, HDIM, HDIM_TILES,
    NQH, NKVH, GQA, EPS, ROPE_THETA,
    N_CHIPS, NQH_TP, NKVH_TP, Q_TP, KV_TP,
    _p, rep, shd, ztt, rb, rb_dim1,
)

DLAYERS = 8
DINTER = 6144
DINTER_TP = DINTER // N_CHIPS  # 1536 per chip
BSIZE = 16
TLAYER_IDS = [1, 12, 23, 34, 45]
N_CTX_LAYERS = len(TLAYER_IDS)
MASK_ID = 151669
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"

# Tile-padded sequence length for BSIZE=16
SP = ((BSIZE + TILE - 1) // TILE) * TILE  # 32

# Kernel instances
rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
rmsnorm_hdim_k = make_rmsnorm_kernel(dim_tiles=HDIM_TILES, eps=EPS)
q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP)
k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP)


def _tile_pad(n):
    return ((n + TILE - 1) // TILE) * TILE


# ---------------------------------------------------------------------------
# Scratch preallocation
# ---------------------------------------------------------------------------
def prealloc_draft_scratch(ctx_len, d):
    """Pre-allocate all reusable scratch tensors for draft_fwd."""
    L1 = ttnn.L1_MEMORY_CONFIG
    kv_len = ctx_len + BSIZE
    kv_sp = _tile_pad(kv_len)
    # Q heads stacked for GQA attention
    q_stacked_rows = NQH_TP * SP  # 8 * 32 = 256

    def _rep(shape, mem=L1):
        return rep(torch.zeros(*shape, dtype=torch.bfloat16), d, mem=mem)
    def _shd(shape, dim, mem=L1):
        return shd(torch.zeros(*shape, dtype=torch.bfloat16), d, dim=dim, mem=mem)

    s = {}
    # Hidden state
    s["norm"] = _rep((SP, HIDDEN))
    s["add"] = _rep((SP, HIDDEN))

    # Context FC projection
    s["ctx_proj"] = _rep((_tile_pad(ctx_len), HIDDEN))
    s["ctx_norm"] = _rep((_tile_pad(ctx_len), HIDDEN))

    # KV concat input (ctx + draft after layernorm, for K/V projection)
    s["kv_in"] = _rep((kv_sp, HIDDEN))

    # QKV projections (column-parallel)
    s["q"] = _shd((SP, Q_TP * N_CHIPS), dim=1)
    s["k"] = _shd((kv_sp, KV_TP * N_CHIPS), dim=1)
    s["v"] = _shd((kv_sp, KV_TP * N_CHIPS), dim=1)

    # QK-norm intermediate (reshaped for ttnn.rms_norm)
    s["q_normed"] = _shd((SP, Q_TP * N_CHIPS), dim=1)
    s["k_normed"] = _shd((kv_sp, KV_TP * N_CHIPS), dim=1)

    # RoPE output
    s["q_rope"] = _shd((SP, Q_TP * N_CHIPS), dim=1)
    s["k_rope"] = _shd((kv_sp, KV_TP * N_CHIPS), dim=1)

    # Attention: stacked Q heads
    s["q_stacked"] = _rep((q_stacked_rows, HDIM))
    s["k_hdim"] = _rep((kv_sp, HDIM))
    s["v_hdim"] = _rep((kv_sp, HDIM))
    s["scores"] = _rep((q_stacked_rows, kv_sp))
    s["probs"] = _rep((q_stacked_rows, kv_sp))
    s["attn_out_stacked"] = _rep((q_stacked_rows, HDIM))

    # Attention output reshaped back
    s["attn_flat"] = _shd((SP, Q_TP * N_CHIPS), dim=1)
    s["o"] = _rep((SP, HIDDEN))

    # MLP scratch
    s["gate"] = _shd((SP, DINTER), dim=1)
    s["up"] = _shd((SP, DINTER), dim=1)
    s["act"] = _shd((SP, DINTER), dim=1)
    s["mlp_out"] = _rep((SP, HIDDEN))

    # Softmax kernel (attention rows = NQH_TP * SP/TILE tile rows, kv_tiles cols)
    kv_tiles = kv_sp // TILE
    q_tile_rows = q_stacked_rows // TILE  # NQH_TP * (SP // TILE) = 8
    s["softmax_k"] = make_softmax_kernel(q_tile_rows, kv_tiles)

    # Scaler for softmax
    s["sc"] = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d)

    # RMSNorm support tensors
    s["ms"] = rep(torch.full((TILE, TILE), 1.0 / HIDDEN, dtype=torch.bfloat16), d, mem=L1)
    s["ms_hdim"] = rep(torch.full((TILE, TILE), 1.0 / HDIM, dtype=torch.bfloat16), d, mem=L1)

    return s


# ---------------------------------------------------------------------------
# Load draft weights to device
# ---------------------------------------------------------------------------
def load_draft_weights(d, w):
    """Load draft model weights into w dict."""
    import time
    t0 = time.time()
    print("  draft weights...")
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        # FC projection: (5*HIDDEN, HIDDEN) for context feature fusion
        w["d.fc"] = rep(f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16), d)

        # Hidden norm (applied after FC)
        hn_w = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        w["d.hn_w_tt"] = rep(hn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

        # Final norm
        fn_w = f.get_tensor("norm.weight").to(torch.bfloat16)
        w["d.fn_w_tt"] = rep(fn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

        for li in range(DLAYERS):
            dp = f"layers.{li}"
            lp = f"d.{li}"

            # Layer norms
            din_w = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            dpa_w = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)
            w[f"{lp}.in_w_tt"] = rep(din_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.pa_w_tt"] = rep(dpa_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

            # QKV projections (column-parallel)
            w[f"{lp}.qw"] = shd(f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
            w[f"{lp}.kw"] = shd(f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
            w[f"{lp}.vw"] = shd(f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=1)

            # O projection (row-parallel)
            w[f"{lp}.ow"] = shd(f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=0)

            # QK-norm weights for ttnn.rms_norm: (1, HDIM)
            qnw = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            knw = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)
            w[f"{lp}.qnw_dev"] = rep(qnw.unsqueeze(0).contiguous(), d)
            w[f"{lp}.knw_dev"] = rep(knw.unsqueeze(0).contiguous(), d)
            # TILE-expanded for TT-Lang rmsnorm kernel
            w[f"{lp}.qnw_tt"] = rep(qnw.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.knw_tt"] = rep(knw.unsqueeze(0).expand(TILE, -1).contiguous(), d)

            # MLP (column-parallel gate/up, row-parallel down)
            w[f"{lp}.gw"] = shd(f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
            w[f"{lp}.uw"] = shd(f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
            w[f"{lp}.fc2"] = shd(f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16), d, dim=0)

    print(f"  draft weights loaded in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Draft forward -- fully on device
# ---------------------------------------------------------------------------
def draft_fwd(noise, ctx_features, w, s, d):
    """DFlash draft forward. All ops on device, zero host readbacks.

    Args:
        noise: (SP, HIDDEN) -- embeddings of draft tokens, replicated
        ctx_features: (ctx_sp, HIDDEN) -- FC-projected + normed context, replicated
        w: weight dict
        s: scratch dict from prealloc_draft_scratch
        d: device/mesh

    Returns:
        (SP, HIDDEN) -- final hidden states after 8 layers + final norm
    """
    h = noise
    ctx = ctx_features
    ctx_sp = ctx.shape[0]  # tile-padded context length
    kv_sp = s["k"].shape[0]  # tile-padded kv length

    for li in range(DLAYERS):
        lp = f"d.{li}"

        # --- Input LayerNorm ---
        rmsnorm_k(h, w[f"{lp}.in_w_tt"], s["sc"], s["ms"], s["norm"])
        normed = s["norm"]

        # --- Q projection (from draft tokens only) ---
        ttnn.matmul(normed, w[f"{lp}.qw"], optional_output_tensor=s["q"])

        # --- KV input: concat context + draft on device ---
        # Copy ctx into top rows, draft into bottom rows of kv_in
        # ctx: (ctx_sp, HIDDEN), normed: (SP, HIDDEN) -> kv_in: (kv_sp, HIDDEN)
        kv_in = ttnn.concat([ctx, normed], dim=0)

        # --- K, V projections (from concatenated input) ---
        ttnn.matmul(kv_in, w[f"{lp}.kw"], optional_output_tensor=s["k"])
        ttnn.matmul(kv_in, w[f"{lp}.vw"], optional_output_tensor=s["v"])

        # --- Per-head QK-norm via reshape + ttnn.rms_norm ---
        q_flat = ttnn.reshape(s["q"], (SP * NQH_TP, HDIM))
        k_flat = ttnn.reshape(s["k"], (kv_sp * NKVH_TP, HDIM))
        q_normed_flat = ttnn.rms_norm(q_flat, weight=w[f"{lp}.qnw_dev"], epsilon=EPS)
        k_normed_flat = ttnn.rms_norm(k_flat, weight=w[f"{lp}.knw_dev"], epsilon=EPS)
        q_normed = ttnn.reshape(q_normed_flat, (SP, NQH_TP * HDIM))
        k_normed = ttnn.reshape(k_normed_flat, (kv_sp, NKVH_TP * HDIM))

        # --- RoPE ---
        # Q RoPE: positions 0..SP-1 (draft token positions)
        # K RoPE: positions 0..kv_sp-1 (context + draft positions)
        q_rope_k(q_normed, w["rope_cos_q"], w["rope_sin_q"], s["q_rope"])
        k_rope_k(k_normed, w["rope_cos_kv"], w["rope_sin_kv"], s["k_rope"])

        # --- Cross-attention via TTNN matmul + TT-Lang softmax ---
        # Stack Q heads: (SP, NQH_TP*HDIM) -> (NQH_TP*SP, HDIM)
        # Since NQH_TP=8 and NKVH_TP=1 (GQA 8:1), all Q heads attend to same KV
        q_stacked = ttnn.reshape(s["q_rope"], (NQH_TP * SP, HDIM))

        # Extract single KV head: (kv_sp, NKVH_TP*HDIM) -> (kv_sp, HDIM)
        # NKVH_TP=1 so this is just a reshape
        k_hdim = ttnn.reshape(s["k_rope"], (kv_sp, HDIM))
        v_hdim = ttnn.reshape(s["v"], (kv_sp, HDIM))

        # Step 1: Q @ K^T = (NQH_TP*SP, HDIM) @ (HDIM, kv_sp) = (NQH_TP*SP, kv_sp)
        k_t = ttnn.transpose(k_hdim, -2, -1)
        scores = ttnn.matmul(q_stacked, k_t)

        # Step 2: TT-Lang softmax (row-wise)
        s["softmax_k"](scores, s["sc"], s["probs"])

        # Step 3: probs @ V = (NQH_TP*SP, kv_sp) @ (kv_sp, HDIM) = (NQH_TP*SP, HDIM)
        attn_out = ttnn.matmul(s["probs"], v_hdim)

        # Un-stack: (NQH_TP*SP, HDIM) -> (SP, NQH_TP*HDIM)
        attn_flat = ttnn.reshape(attn_out, (SP, NQH_TP * HDIM))

        # --- O projection (row-parallel) + all_reduce ---
        ttnn.matmul(attn_flat, w[f"{lp}.ow"], optional_output_tensor=s["o"])
        o = ttnn.all_reduce(s["o"])

        # --- Residual add ---
        residual_add_kernel(h, o, s["add"])
        h = s["add"]

        # --- Post-attention LayerNorm ---
        rmsnorm_k(h, w[f"{lp}.pa_w_tt"], s["sc"], s["ms"], s["norm"])

        # --- Dense MLP ---
        ttnn.matmul(s["norm"], w[f"{lp}.gw"], optional_output_tensor=s["gate"])
        ttnn.matmul(s["norm"], w[f"{lp}.uw"], optional_output_tensor=s["up"])
        silu_mul_kernel(s["gate"], s["up"], s["act"])
        ttnn.matmul(s["act"], w[f"{lp}.fc2"], optional_output_tensor=s["mlp_out"])
        mlp_out = ttnn.all_reduce(s["mlp_out"])

        # --- Residual add ---
        residual_add_kernel(h, mlp_out, s["add"])
        h = s["add"]

    # --- Final norm ---
    rmsnorm_k(h, w["d.fn_w_tt"], s["sc"], s["ms"], s["norm"])
    return s["norm"]


def prepare_context(target_hidden, w, s, d):
    """Project and normalize target context features.

    Args:
        target_hidden: (ctx_sp, N_CTX_LAYERS * HIDDEN) -- concatenated target hidden states
        w: weight dict
        s: scratch dict

    Returns:
        (ctx_sp, HIDDEN) -- projected + normed context
    """
    # FC: (ctx_sp, 5*HIDDEN) @ (5*HIDDEN, HIDDEN) -> (ctx_sp, HIDDEN)
    ttnn.matmul(target_hidden, w["d.fc"], optional_output_tensor=s["ctx_proj"])
    # Hidden norm
    rmsnorm_k(s["ctx_proj"], w["d.hn_w_tt"], s["sc"], s["ms"], s["ctx_norm"])
    return s["ctx_norm"]
