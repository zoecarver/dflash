"""Qwen3-Coder-30B-A3B (MoE) inference on Tenstorrent Blackhole.

4-chip tensor parallelism:
  - Attention: head-parallel (8 Q heads + 1 KV head per chip), all-reduce after O proj
  - MoE: expert-parallel (32 experts per chip), all-to-all routing via host
"""
import os
import sys
import time
import math
import torch
import numpy as np

import ttnn

# TT-Lang kernel imports (flat layout when running on remote)
try:
    from src.rmsnorm import make_rmsnorm_kernel
    from src.silu import silu_kernel
    from src.residual_add import residual_add_kernel
    from src.silu_mul import silu_mul_kernel
except ImportError:
    from rmsnorm import make_rmsnorm_kernel
    from silu import silu_kernel
    from residual_add import residual_add_kernel
    from silu_mul import silu_mul_kernel

# ---------------------------------------------------------------------------
# Model constants (Qwen3-Coder-30B-A3B)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2048
NUM_LAYERS = 48
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_SIZE = 6144       # dense MLP reference size (unused, MoE replaces)
MOE_INTERMEDIATE_SIZE = 768    # per-expert intermediate size
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
MAX_SEQ_LEN = 4096             # practical limit for on-chip KV cache

TILE = 32
N_CHIPS = 4

# Derived
NUM_Q_HEADS_TP = NUM_Q_HEADS // N_CHIPS       # 8 per chip
NUM_KV_HEADS_TP = NUM_KV_HEADS // N_CHIPS     # 1 per chip
HIDDEN_TILES = HIDDEN_SIZE // TILE             # 64
EXPERTS_PER_CHIP = NUM_EXPERTS // N_CHIPS      # 32

# Tile-padded sizes
VOCAB_PAD = ((VOCAB_SIZE + TILE - 1) // TILE) * TILE  # 151968
MOE_INTER_PAD = ((MOE_INTERMEDIATE_SIZE + TILE - 1) // TILE) * TILE  # 768 (already aligned)
Q_SIZE = NUM_Q_HEADS * HEAD_DIM    # 4096
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 512
Q_SIZE_TP = Q_SIZE // N_CHIPS      # 1024
KV_SIZE_TP = KV_SIZE // N_CHIPS    # 128

# TT-Lang kernel instances
rmsnorm_d2048 = make_rmsnorm_kernel(dim_tiles=HIDDEN_TILES, eps=RMS_NORM_EPS)

# ---------------------------------------------------------------------------
# Mesh device management
# ---------------------------------------------------------------------------
_MESH_DEVICE = None


def open_device():
    global _MESH_DEVICE
    if N_CHIPS > 1:
        _MESH_DEVICE = ttnn.open_mesh_device(
            ttnn.MeshShape(1, N_CHIPS),
            fabric_config=ttnn.FabricConfig.FABRIC_1D,
        )
        return _MESH_DEVICE
    else:
        return ttnn.open_device(device_id=0)


def close_device(device):
    global _MESH_DEVICE
    if _MESH_DEVICE is not None:
        ttnn.close_mesh_device(_MESH_DEVICE)
        _MESH_DEVICE = None
    else:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Tensor helpers (replicate / shard across mesh)
# ---------------------------------------------------------------------------
def _mesh_kwargs(device):
    if isinstance(device, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
    return {}


def to_tt(t, device, mem=None):
    if mem is None:
        mem = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=mem, **_mesh_kwargs(device),
    )


def to_tt_l1(t, device):
    return to_tt(t, device, mem=ttnn.L1_MEMORY_CONFIG)


def shard_tt(t, device, dim, mem=None):
    if mem is None:
        mem = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
    )


def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)


def readback_torch(t, device):
    if _MESH_DEVICE is not None:
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH_DEVICE, dim=0),
        )[:t.shape[0]]
    return ttnn.to_torch(t)


def pad_to_tile(t, dim=-1):
    """Pad tensor to tile alignment (32) along given dimension."""
    size = t.shape[dim]
    pad_size = (TILE - size % TILE) % TILE
    if pad_size == 0:
        return t
    pad_spec = [0] * (2 * len(t.shape))
    pad_spec[-(2 * (dim % len(t.shape)) + 1)] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


# ---------------------------------------------------------------------------
# RoPE precomputation (host-side, real-valued cos/sin)
# ---------------------------------------------------------------------------
def precompute_rope(seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """Precompute RoPE cos/sin tables on host."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    cos_table = torch.cos(angles).to(torch.bfloat16)  # (seq_len, head_dim/2)
    sin_table = torch.sin(angles).to(torch.bfloat16)
    return cos_table, sin_table


def apply_rope_host(q, k, cos, sin):
    """Apply RoPE on host tensors. q: (B, H, S, D), k: (B, H, S, D)."""

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # cos/sin: (S, D/2) -> (1, 1, S, D) by repeating
    cos_full = cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)  # (1, 1, S, D)
    sin_full = sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)

    q_len = q.shape[2]
    q_rope = q * cos_full[:, :, :q_len, :] + rotate_half(q) * sin_full[:, :, :q_len, :]
    k_rope = k * cos_full + rotate_half(k) * sin_full
    return q_rope, k_rope


# ---------------------------------------------------------------------------
# Weight loading from HuggingFace safetensors
# ---------------------------------------------------------------------------
def find_weight_files(model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct"):
    """Find safetensors files in HuggingFace cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_id.replace("/", "--")
    model_path = os.path.join(cache_dir, model_dir_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Download with: huggingface-cli download {model_id}"
        )
    # Find the snapshots directory
    snapshots = os.path.join(model_path, "snapshots")
    if os.path.exists(snapshots):
        # Use latest snapshot
        snapshot_dirs = sorted(os.listdir(snapshots))
        if snapshot_dirs:
            snapshot_path = os.path.join(snapshots, snapshot_dirs[-1])
            safetensor_files = sorted(
                f for f in os.listdir(snapshot_path) if f.endswith(".safetensors")
            )
            return [os.path.join(snapshot_path, f) for f in safetensor_files]
    raise FileNotFoundError(f"No safetensors found in {model_path}")


def load_weights(tt_device, model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct"):
    """Load all model weights onto device(s).

    Returns dict of device tensors keyed by component name.
    """
    from safetensors import safe_open

    weight_files = find_weight_files(model_id)
    print(f"Loading weights from {len(weight_files)} files...")
    t0 = time.time()

    dev = {}

    # We need a unified view of all safetensors
    # Build a key -> file mapping first
    key_to_file = {}
    for fpath in weight_files:
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                key_to_file[key] = fpath

    def get_tensor(key):
        fpath = key_to_file[key]
        with safe_open(fpath, framework="pt") as f:
            return f.get_tensor(key)

    # Embedding + LM head
    embed_w = get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
    embed_w_padded = pad_to_tile(embed_w, dim=0)  # pad vocab to tile alignment
    dev["embed_weight"] = to_tt(embed_w_padded, tt_device)

    lm_head_w = get_tensor("lm_head.weight").T.contiguous().to(torch.bfloat16)
    lm_head_w_padded = pad_to_tile(lm_head_w, dim=1)
    dev["lm_head_weight"] = to_tt(lm_head_w_padded, tt_device)

    # Final norm
    dev["final_norm_weight"] = get_tensor("model.norm.weight").to(torch.bfloat16)

    # RoPE cos/sin (precomputed on host, loaded per forward call)

    # Per-layer weights
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}"
        lp = f"layer.{layer_idx}"  # device key prefix

        # Input layernorm (pre-attention)
        dev[f"{lp}.input_norm_w"] = get_tensor(f"{prefix}.input_layernorm.weight").to(torch.bfloat16)

        # Post-attention layernorm (pre-MoE)
        dev[f"{lp}.post_attn_norm_w"] = get_tensor(f"{prefix}.post_attention_layernorm.weight").to(torch.bfloat16)

        # Attention Q/K/V/O projections
        q_w = get_tensor(f"{prefix}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
        k_w = get_tensor(f"{prefix}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
        v_w = get_tensor(f"{prefix}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
        o_w = get_tensor(f"{prefix}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)

        # QK norms (per-head, applied after projection)
        dev[f"{lp}.q_norm_w"] = get_tensor(f"{prefix}.self_attn.q_norm.weight").to(torch.bfloat16)
        dev[f"{lp}.k_norm_w"] = get_tensor(f"{prefix}.self_attn.k_norm.weight").to(torch.bfloat16)

        # Column-parallel Q: shard along output dim (heads)
        if N_CHIPS > 1:
            dev[f"{lp}.q_w"] = shard_tt(q_w, tt_device, dim=1)
            dev[f"{lp}.k_w"] = shard_tt(k_w, tt_device, dim=1)
            dev[f"{lp}.v_w"] = shard_tt(v_w, tt_device, dim=1)
            dev[f"{lp}.o_w"] = shard_tt(o_w, tt_device, dim=0)
        else:
            dev[f"{lp}.q_w"] = to_tt(q_w, tt_device)
            dev[f"{lp}.k_w"] = to_tt(k_w, tt_device)
            dev[f"{lp}.v_w"] = to_tt(v_w, tt_device)
            dev[f"{lp}.o_w"] = to_tt(o_w, tt_device)

        # MoE router
        router_w = get_tensor(f"{prefix}.mlp.gate.weight").T.contiguous().to(torch.bfloat16)
        dev[f"{lp}.router_w"] = to_tt(router_w, tt_device)

        # Expert weights: shard 32 experts per chip
        # Each expert has: gate_proj, up_proj, down_proj
        # We pack experts into batched weight matrices for efficient computation
        #
        # Strategy: store per-expert weights, group by chip assignment
        # Expert i lives on chip (i // EXPERTS_PER_CHIP)
        for expert_idx in range(NUM_EXPERTS):
            ep = f"{prefix}.mlp.experts.{expert_idx}"
            gate_w = get_tensor(f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16)
            up_w = get_tensor(f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16)
            down_w = get_tensor(f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16)

            chip_id = expert_idx // EXPERTS_PER_CHIP
            local_expert_idx = expert_idx % EXPERTS_PER_CHIP

            # Store as host tensors first, load to specific chip
            dev[f"{lp}.expert.{expert_idx}.gate_w"] = gate_w
            dev[f"{lp}.expert.{expert_idx}.up_w"] = up_w
            dev[f"{lp}.expert.{expert_idx}.down_w"] = down_w

        if (layer_idx + 1) % 8 == 0:
            print(f"  Loaded layer {layer_idx + 1}/{NUM_LAYERS}")

    elapsed = time.time() - t0
    print(f"Weight loading complete: {len(dev)} tensors in {elapsed:.1f}s")
    return dev


def load_expert_weights_to_device(dev, tt_device):
    """Load expert weights to device as per-expert tensors."""
    t0 = time.time()

    for layer_idx in range(NUM_LAYERS):
        lp = f"layer.{layer_idx}"

        for expert_idx in range(NUM_EXPERTS):
            gate_w = dev.pop(f"{lp}.expert.{expert_idx}.gate_w")
            up_w = dev.pop(f"{lp}.expert.{expert_idx}.up_w")
            down_w = dev.pop(f"{lp}.expert.{expert_idx}.down_w")

            dev[f"{lp}.expert_dev.{expert_idx}.gate_w"] = to_tt(gate_w, tt_device)
            dev[f"{lp}.expert_dev.{expert_idx}.up_w"] = to_tt(up_w, tt_device)
            dev[f"{lp}.expert_dev.{expert_idx}.down_w"] = to_tt(down_w, tt_device)

        if (layer_idx + 1) % 8 == 0:
            print(f"  Expert weights to device: layer {layer_idx + 1}/{NUM_LAYERS}")

    elapsed = time.time() - t0
    print(f"Expert weight loading complete in {elapsed:.1f}s")
    return dev


# ---------------------------------------------------------------------------
# RMSNorm helper (host-side weight expansion + device kernel call)
# ---------------------------------------------------------------------------
def make_norm_tensors(weight_1d, seq_len, device):
    """Expand 1D norm weight to (seq_len, hidden_size) for the TT-Lang kernel."""
    w_expanded = weight_1d.unsqueeze(0).expand(seq_len, -1).contiguous().to(torch.bfloat16)
    return to_tt(w_expanded, device)


def run_rmsnorm(x_tt, weight_expanded, scaler, mean_scale, out_tt, device):
    """Run RMSNorm TT-Lang kernel."""
    rmsnorm_d2048(x_tt, weight_expanded, scaler, mean_scale, out_tt)
    return out_tt


# ---------------------------------------------------------------------------
# GQA Attention forward (per-layer, using TTNN ops)
# ---------------------------------------------------------------------------
def attention_forward(
    hidden_tt, dev, layer_prefix, seq_len, cos_host, sin_host,
    kv_cache_k, kv_cache_v, cache_pos, tt_device, scratch,
):
    """Run GQA attention for one layer.

    Args:
        hidden_tt: (seq_len, HIDDEN_SIZE) device tensor (post-RMSNorm)
        cos_host, sin_host: RoPE tables on host (seq_len, HEAD_DIM/2)
        kv_cache_k/v: (MAX_SEQ_LEN, KV_SIZE) device tensors (or None for no cache)
        cache_pos: int, current position in KV cache
    Returns:
        attn_output: (seq_len, HIDDEN_SIZE) device tensor
    """
    lp = layer_prefix
    q_len = seq_len

    # QKV projections (column-parallel)
    q_tt = ttnn.matmul(hidden_tt, dev[f"{lp}.q_w"])   # (S, Q_SIZE_TP)
    k_tt = ttnn.matmul(hidden_tt, dev[f"{lp}.k_w"])   # (S, KV_SIZE_TP)
    v_tt = ttnn.matmul(hidden_tt, dev[f"{lp}.v_w"])   # (S, KV_SIZE_TP)

    # Readback for RoPE + QK norm (host-side for now)
    q_host = readback_torch(q_tt, tt_device).float()
    k_host = readback_torch(k_tt, tt_device).float()
    v_host = readback_torch(v_tt, tt_device).float()

    # Reshape to (B, H, S, D)
    q_host = q_host.view(1, q_len, NUM_Q_HEADS_TP, HEAD_DIM).transpose(1, 2)
    k_host = k_host.view(1, q_len, NUM_KV_HEADS_TP, HEAD_DIM).transpose(1, 2)
    v_host = v_host.view(1, q_len, NUM_KV_HEADS_TP, HEAD_DIM).transpose(1, 2)

    # QK RMSNorm (per-head, on host)
    q_norm_w = dev[f"{lp}.q_norm_w"].float()  # (HEAD_DIM,)
    k_norm_w = dev[f"{lp}.k_norm_w"].float()

    def rms_norm_host(x, w):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
        return (x / rms) * w

    q_host = rms_norm_host(q_host, q_norm_w)
    k_host = rms_norm_host(k_host, k_norm_w)

    # RoPE (host-side)
    cos_slice = cos_host[:q_len].to(torch.float32)
    sin_slice = sin_host[:q_len].to(torch.float32)
    q_host, k_host = apply_rope_host(q_host, k_host, cos_slice, sin_slice)

    # KV cache update
    if kv_cache_k is not None:
        # Update cache on host
        kv_cache_k[:, :, cache_pos:cache_pos + q_len, :] = k_host.to(torch.bfloat16)
        kv_cache_v[:, :, cache_pos:cache_pos + q_len, :] = v_host.to(torch.bfloat16)
        k_full = kv_cache_k[:, :, :cache_pos + q_len, :].float()
        v_full = kv_cache_v[:, :, :cache_pos + q_len, :].float()
    else:
        k_full = k_host
        v_full = v_host

    # GQA: repeat KV heads for grouped attention
    num_groups = NUM_Q_HEADS_TP // NUM_KV_HEADS_TP  # 8
    k_full = k_full.repeat(1, num_groups, 1, 1)
    v_full = v_full.repeat(1, num_groups, 1, 1)

    # Scaled dot-product attention (host for correctness, move to device later)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_weights = torch.matmul(q_host, k_full.transpose(-2, -1)) * scale

    # Causal mask
    if q_len > 1:
        causal_mask = torch.triu(
            torch.full((q_len, k_full.shape[2]), float("-inf")),
            diagonal=cache_pos + 1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = torch.softmax(attn_weights, dim=-1).to(torch.bfloat16)
    attn_out = torch.matmul(attn_weights.float(), v_full)

    # Reshape back to (S, Q_SIZE_TP)
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, q_len, -1).squeeze(0)
    attn_out = attn_out.to(torch.bfloat16)

    # Pad seq to tile alignment if needed
    attn_out_padded = pad_to_tile(attn_out, dim=0)

    # O projection (row-parallel) + all-reduce
    attn_tt = to_tt(attn_out_padded, tt_device)
    o_tt = ttnn.matmul(attn_tt, dev[f"{lp}.o_w"])
    if N_CHIPS > 1:
        o_tt = ttnn.all_reduce(o_tt)

    return o_tt


# ---------------------------------------------------------------------------
# MoE forward (per-layer)
# ---------------------------------------------------------------------------
def moe_forward(hidden_tt, dev, layer_prefix, seq_len, tt_device):
    """Run MoE layer with on-device expert computation.

    1. Router matmul on device -> readback for top-k
    2. Group tokens by expert
    3. Per-expert: gather tokens, gate/up matmul on device, silu_mul kernel, down matmul
    4. Scatter weighted outputs back

    Args:
        hidden_tt: (seq_len_pad, HIDDEN_SIZE) device tensor (post-RMSNorm)
    Returns:
        moe_output: (seq_len_pad, HIDDEN_SIZE) device tensor
    """
    lp = layer_prefix
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    # Router logits on device: (seq_pad, NUM_EXPERTS)
    router_logits = ttnn.matmul(hidden_tt, dev[f"{lp}.router_w"])

    # Readback for top-k selection on host
    router_host = readback_torch(router_logits, tt_device)[:seq_len].float()

    # Top-k expert selection per token
    scores = torch.softmax(router_host, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Readback hidden for token gathering
    hidden_host = readback_torch(hidden_tt, tt_device)[:seq_len].to(torch.bfloat16)

    # Group tokens by expert
    expert_to_tokens = {}
    for tok_idx in range(seq_len):
        for k in range(NUM_EXPERTS_PER_TOK):
            expert_idx = topk_indices[tok_idx, k].item()
            weight = topk_weights[tok_idx, k].item()
            if expert_idx not in expert_to_tokens:
                expert_to_tokens[expert_idx] = []
            expert_to_tokens[expert_idx].append((tok_idx, weight))

    # Run each active expert on device
    output_host = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)

    for expert_idx, token_info in expert_to_tokens.items():
        tok_indices = [t[0] for t in token_info]
        n_tok = len(tok_indices)
        n_tok_pad = ((n_tok + TILE - 1) // TILE) * TILE

        # Gather and pad tokens for this expert
        expert_input = hidden_host[tok_indices]
        inp_padded = torch.nn.functional.pad(expert_input, (0, 0, 0, n_tok_pad - n_tok))
        inp_tt = to_tt(inp_padded, tt_device)

        # Get expert weights (already on device)
        gate_w_tt = dev[f"{lp}.expert_dev.{expert_idx}.gate_w"]
        up_w_tt = dev[f"{lp}.expert_dev.{expert_idx}.up_w"]
        down_w_tt = dev[f"{lp}.expert_dev.{expert_idx}.down_w"]

        # Expert FFN on device: SiLU(gate_proj(x)) * up_proj(x) -> down_proj(...)
        gate_out = ttnn.matmul(inp_tt, gate_w_tt)
        up_out = ttnn.matmul(inp_tt, up_w_tt)

        activated = zeros_tt((n_tok_pad, MOE_INTERMEDIATE_SIZE), tt_device)
        silu_mul_kernel(gate_out, up_out, activated)

        expert_out_tt = ttnn.matmul(activated, down_w_tt)
        expert_out = readback_torch(expert_out_tt, tt_device)[:n_tok, :HIDDEN_SIZE].to(torch.bfloat16)

        # Scatter weighted outputs
        for i, (tok_idx, weight) in enumerate(token_info):
            output_host[tok_idx] += expert_out[i] * weight

    # Pad and send back to device
    output_padded = pad_to_tile(output_host.unsqueeze(0), dim=1).squeeze(0)
    if output_padded.shape[0] < seq_pad:
        output_padded = torch.nn.functional.pad(
            output_padded, (0, 0, 0, seq_pad - output_padded.shape[0])
        )
    return to_tt(output_padded, tt_device)


# ---------------------------------------------------------------------------
# Transformer block forward
# ---------------------------------------------------------------------------
def transformer_block_forward(
    hidden_tt, dev, layer_idx, seq_len, cos_host, sin_host,
    kv_cache_k, kv_cache_v, cache_pos, tt_device, scaler, mean_scale, scratch,
):
    """Run one transformer block: RMSNorm -> Attn -> Residual -> RMSNorm -> MoE -> Residual."""
    lp = f"layer.{layer_idx}"
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    # Pre-attention RMSNorm
    input_norm_w = make_norm_tensors(dev[f"{lp}.input_norm_w"], seq_pad, tt_device)
    normed_tt = zeros_tt((seq_pad, HIDDEN_SIZE), tt_device)
    rmsnorm_d2048(hidden_tt, input_norm_w, scaler, mean_scale, normed_tt)

    # Attention
    attn_out = attention_forward(
        normed_tt, dev, lp, seq_len, cos_host, sin_host,
        kv_cache_k[layer_idx] if kv_cache_k else None,
        kv_cache_v[layer_idx] if kv_cache_v else None,
        cache_pos, tt_device, scratch,
    )

    # Residual connection: hidden = hidden + attn_out
    residual_out = zeros_tt((seq_pad, HIDDEN_SIZE), tt_device)
    residual_add_kernel(hidden_tt, attn_out, residual_out)

    # Pre-MoE RMSNorm
    post_attn_norm_w = make_norm_tensors(dev[f"{lp}.post_attn_norm_w"], seq_pad, tt_device)
    normed_moe = zeros_tt((seq_pad, HIDDEN_SIZE), tt_device)
    rmsnorm_d2048(residual_out, post_attn_norm_w, scaler, mean_scale, normed_moe)

    # MoE
    moe_out = moe_forward(normed_moe, dev, lp, seq_len, tt_device)

    # Residual connection: hidden = residual + moe_out
    final_out = zeros_tt((seq_pad, HIDDEN_SIZE), tt_device)
    residual_add_kernel(residual_out, moe_out, final_out)

    return final_out


# ---------------------------------------------------------------------------
# Full model forward
# ---------------------------------------------------------------------------
def model_forward(
    input_ids, dev, tt_device, cos_host, sin_host,
    kv_cache_k=None, kv_cache_v=None, cache_pos=0,
):
    """Full forward pass through the Qwen3-Coder-30B-A3B model.

    Args:
        input_ids: (batch=1, seq_len) token IDs on host
        kv_cache_k/v: list of per-layer KV cache tensors (host), or None
        cache_pos: current position in KV cache
    Returns:
        logits: (1, seq_len, VOCAB_SIZE) on host
    """
    seq_len = input_ids.shape[1]
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    # Embedding lookup (host-side, then to device)
    # For efficiency, we do embedding on host and send to device
    # Device-side embedding gather is possible but complex for large vocab
    embed_host = torch.nn.functional.embedding(
        input_ids, readback_torch(dev["embed_weight"], tt_device)[:VOCAB_SIZE],
    ).squeeze(0).to(torch.bfloat16)  # (seq_len, HIDDEN_SIZE)

    embed_padded = pad_to_tile(embed_host, dim=0)
    if embed_padded.shape[0] < seq_pad:
        embed_padded = torch.nn.functional.pad(
            embed_padded, (0, 0, 0, seq_pad - embed_padded.shape[0])
        )
    hidden_tt = to_tt(embed_padded, tt_device)

    # Scaler and mean_scale for RMSNorm kernel
    scaler_host = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale_host = torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16)
    scaler = to_tt(scaler_host, tt_device)
    mean_scale = to_tt(mean_scale_host, tt_device)

    scratch = {}

    # Run all transformer blocks
    for layer_idx in range(NUM_LAYERS):
        t_layer = time.time()
        hidden_tt = transformer_block_forward(
            hidden_tt, dev, layer_idx, seq_len, cos_host, sin_host,
            kv_cache_k, kv_cache_v, cache_pos, tt_device,
            scaler, mean_scale, scratch,
        )
        elapsed = time.time() - t_layer
        if (layer_idx + 1) % 8 == 0:
            print(f"  Layer {layer_idx + 1}/{NUM_LAYERS}: {elapsed * 1000:.1f}ms")

    # Final RMSNorm
    final_norm_w = make_norm_tensors(dev["final_norm_weight"], seq_pad, tt_device)
    normed_final = zeros_tt((seq_pad, HIDDEN_SIZE), tt_device)
    rmsnorm_d2048(hidden_tt, final_norm_w, scaler, mean_scale, normed_final)

    # LM head: (seq_pad, HIDDEN_SIZE) @ (HIDDEN_SIZE, VOCAB_PAD) -> (seq_pad, VOCAB_PAD)
    logits_tt = ttnn.matmul(normed_final, dev["lm_head_weight"])

    # Readback logits
    logits_host = readback_torch(logits_tt, tt_device)[:seq_len, :VOCAB_SIZE]
    return logits_host.unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------
def generate(
    input_ids,
    dev,
    tt_device,
    max_new_tokens=128,
    temperature=0.6,
    top_p=0.9,
):
    """Autoregressive token generation.

    Args:
        input_ids: (1, prompt_len) token IDs
        max_new_tokens: max tokens to generate
    Returns:
        output_ids: (1, prompt_len + generated) token IDs
    """
    prompt_len = input_ids.shape[1]

    # Precompute RoPE for max sequence length
    total_len = min(prompt_len + max_new_tokens, MAX_SEQ_LEN)
    cos_host, sin_host = precompute_rope(total_len)

    # Initialize KV cache on host
    kv_cache_k = [
        torch.zeros(1, NUM_KV_HEADS_TP, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]
    kv_cache_v = [
        torch.zeros(1, NUM_KV_HEADS_TP, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]

    # Prefill: process entire prompt
    print(f"Prefill: {prompt_len} tokens...")
    t0 = time.time()
    logits = model_forward(
        input_ids, dev, tt_device, cos_host, sin_host,
        kv_cache_k, kv_cache_v, cache_pos=0,
    )
    t_prefill = time.time() - t0
    print(f"Prefill done in {t_prefill:.2f}s ({prompt_len / t_prefill:.1f} tok/s)")

    # Sample first token
    next_token = sample_token(logits[:, -1:, :], temperature, top_p)
    generated = [next_token.item()]

    # Decode: generate tokens one at a time
    cache_pos = prompt_len
    for step in range(max_new_tokens - 1):
        t_step = time.time()
        token_input = next_token.unsqueeze(0)  # (1, 1)

        logits = model_forward(
            token_input, dev, tt_device, cos_host, sin_host,
            kv_cache_k, kv_cache_v, cache_pos=cache_pos,
        )
        cache_pos += 1

        next_token = sample_token(logits[:, -1:, :], temperature, top_p)
        generated.append(next_token.item())

        elapsed = time.time() - t_step
        if (step + 1) % 10 == 0:
            print(f"  Decode step {step + 1}: {elapsed * 1000:.1f}ms/tok")

        # Check for EOS
        if next_token.item() in (151643, 151645):  # bos/eos token IDs
            break

    output_ids = torch.cat([
        input_ids,
        torch.tensor([generated], dtype=torch.long),
    ], dim=1)
    return output_ids


def sample_token(logits, temperature=0.6, top_p=0.9):
    """Sample a token from logits with temperature and top-p."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1).squeeze(-1)

    logits = logits.squeeze(0) / temperature
    probs = torch.softmax(logits, dim=-1)

    # Top-p (nucleus) sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, token).squeeze(-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Qwen3-Coder-30B-A3B on Tenstorrent Blackhole")
    print(f"  Chips: {N_CHIPS}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Experts: {NUM_EXPERTS} total, {NUM_EXPERTS_PER_TOK} active/token")
    print(f"  Expert parallelism: {EXPERTS_PER_CHIP} experts/chip")
    print(f"  Attention TP: {NUM_Q_HEADS_TP} Q heads + {NUM_KV_HEADS_TP} KV heads per chip")
    print("=" * 60)

    # Open device
    tt_device = open_device()
    print("Device opened.")

    # Load weights
    try:
        dev = load_weights(tt_device)
        dev = load_expert_weights_to_device(dev, tt_device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please download the model first:")
        print("  huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct")
        close_device(tt_device)
        return

    # Tokenize a test prompt
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        print("Using dummy input for testing.")
        input_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long)
        output_ids = generate(input_ids, dev, tt_device, max_new_tokens=16, temperature=0.0)
        print(f"Output shape: {output_ids.shape}")
        close_device(tt_device)
        return

    prompt = "Write a Python function that computes the Fibonacci sequence."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    print(f"Prompt: {prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    t0 = time.time()
    output_ids = generate(input_ids, dev, tt_device, max_new_tokens=256, temperature=0.6)
    elapsed = time.time() - t0

    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    num_generated = output_ids.shape[1] - input_ids.shape[1]
    print(f"\nGenerated {num_generated} tokens in {elapsed:.2f}s")
    print(f"Throughput: {num_generated / elapsed:.1f} tok/s")
    print(f"\n--- Output ---\n{output_text}")

    close_device(tt_device)


if __name__ == "__main__":
    main()
