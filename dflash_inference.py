"""DFlash speculative decoding on Tenstorrent Blackhole.

Target model: Qwen3-Coder-30B-A3B (48-layer MoE, 128 experts, top-8)
Draft model: DFlash (8-layer cross-attention + dense MLP)

The draft model generates blocks of 16 tokens at a time using target hidden
states as context, then the target model verifies them.
"""
import math
import time
import json
import torch
import ttnn
from safetensors import safe_open
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
TILE = 32
HIDDEN_SIZE = 2048
HIDDEN_TILES = HIDDEN_SIZE // TILE
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
VOCAB_SIZE = 151936
VOCAB_PAD = ((VOCAB_SIZE + TILE - 1) // TILE) * TILE

# Target model (MoE)
TARGET_LAYERS = 48
MOE_INTER = 768
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8

# Draft model (DFlash)
DRAFT_LAYERS = 8
DRAFT_INTER = 6144  # dense MLP intermediate
BLOCK_SIZE = 16
TARGET_LAYER_IDS = [1, 12, 23, 34, 45]  # 5 layers
NUM_TARGET_FEATURES = len(TARGET_LAYER_IDS)
MASK_TOKEN_ID = 151669

# Paths
TARGET_WEIGHTS_DIR = "/workspace/qwen-coder-30b-a3b/weights"
DRAFT_WEIGHTS_DIR = "/workspace/qwen-coder-30b-a3b/dflash"

# TT-Lang kernels
rmsnorm_kernel = make_rmsnorm_kernel(dim_tiles=HIDDEN_TILES, eps=RMS_NORM_EPS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pad_to_tile(t, dim=-1):
    size = t.shape[dim]
    pad_size = (TILE - size % TILE) % TILE
    if pad_size == 0:
        return t
    pad_spec = [0] * (2 * len(t.shape))
    pad_spec[-(2 * (dim % len(t.shape)) + 1)] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


def rms_norm_host(x, w, eps=RMS_NORM_EPS):
    xf = x.float()
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((xf / rms) * w.float()).to(torch.bfloat16)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope(seq_len):
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles).to(torch.bfloat16), torch.sin(angles).to(torch.bfloat16)


def apply_rope(q, k, cos, sin):
    cos_full = cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2).float()
    sin_full = sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2).float()
    q_len = q.shape[2]
    q_r = q * cos_full[:, :, :q_len, :] + rotate_half(q) * sin_full[:, :, :q_len, :]
    k_r = k * cos_full + rotate_half(k) * sin_full
    return q_r, k_r


def run_rmsnorm_tt(x_tt, weight_1d, seq_pad, scaler_tt, mean_scale_tt, device):
    """Run RMSNorm on device, returns output device tensor."""
    w_exp = weight_1d.unsqueeze(0).expand(seq_pad, -1).contiguous()
    out_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(x_tt, to_dev(w_exp, device), scaler_tt, mean_scale_tt, out_tt)
    return out_tt


def gqa_attention_host(normed_host, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
                       cos, sin, seq_len, kv_cache_k=None, kv_cache_v=None,
                       cache_pos=0, is_causal=True, ctx_hidden=None):
    """Run GQA attention on host. Returns attn_output (seq_len, HIDDEN_SIZE).

    If ctx_hidden is provided (DFlash cross-attention), K/V are computed from
    the concatenation of [ctx_hidden, hidden_states].
    """
    h = normed_host.float()

    q = (h @ q_w.float()).view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    q = rms_norm_host(q.to(torch.bfloat16), q_norm_w).float()

    if ctx_hidden is not None:
        # Cross-attention: K/V from [ctx, noise]
        ctx_len = ctx_hidden.shape[0]
        kv_input = torch.cat([ctx_hidden.float(), h], dim=0)
        kv_len = kv_input.shape[0]
    else:
        kv_input = h
        kv_len = seq_len
        ctx_len = 0

    k = (kv_input @ k_w.float()).view(1, kv_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = (kv_input @ v_w.float()).view(1, kv_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = rms_norm_host(k.to(torch.bfloat16), k_norm_w).float()

    # RoPE
    cos_slice = cos[:kv_len].float()
    sin_slice = sin[:kv_len].float()
    q_cos = cos[:seq_len] if ctx_hidden is not None else cos[:seq_len]
    q_sin = sin[:seq_len] if ctx_hidden is not None else sin[:seq_len]
    # For cross-attn, Q gets positions starting from ctx_len
    if ctx_hidden is not None:
        q_cos = cos[ctx_len:ctx_len+seq_len].float()
        q_sin = sin[ctx_len:ctx_len+seq_len].float()
        q_cos_full = q_cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        q_sin_full = q_sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        q = q * q_cos_full + rotate_half(q) * q_sin_full

        k_cos_full = cos_slice.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        k_sin_full = sin_slice.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
        k = k * k_cos_full + rotate_half(k) * k_sin_full
    else:
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])

    # KV cache
    if kv_cache_k is not None:
        kv_cache_k[:, :, cache_pos:cache_pos + kv_len, :] = k.to(torch.bfloat16)
        kv_cache_v[:, :, cache_pos:cache_pos + kv_len, :] = v.to(torch.bfloat16)
        k_full = kv_cache_k[:, :, :cache_pos + kv_len, :].float()
        v_full = kv_cache_v[:, :, :cache_pos + kv_len, :].float()
    else:
        k_full = k
        v_full = v

    # GQA expand
    num_groups = NUM_Q_HEADS // NUM_KV_HEADS
    k_full = k_full.repeat(1, num_groups, 1, 1)
    v_full = v_full.repeat(1, num_groups, 1, 1)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_w = torch.matmul(q, k_full.transpose(-2, -1)) * scale

    if is_causal and seq_len > 1:
        total_kv = k_full.shape[2]
        causal_mask = torch.triu(
            torch.full((seq_len, total_kv), float("-inf")), diagonal=cache_pos + 1,
        )
        attn_w = attn_w + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_w = torch.softmax(attn_w, dim=-1)
    attn_out = torch.matmul(attn_w, v_full)
    attn_out = attn_out.transpose(1, 2).contiguous().view(seq_len, -1)

    o_out = attn_out @ o_w.float()
    return o_out.to(torch.bfloat16)


def moe_forward_host(hidden, router_w, expert_gate, expert_up, expert_down, device):
    """Run MoE on device (router + per-expert FFN)."""
    seq_len = hidden.shape[0]
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE
    hidden_bf = hidden.to(torch.bfloat16)

    h_padded = pad_to_tile(hidden_bf, dim=0)
    if h_padded.shape[0] < seq_pad:
        h_padded = torch.nn.functional.pad(h_padded, (0, 0, 0, seq_pad - h_padded.shape[0]))
    h_tt = to_dev(h_padded, device)

    router_tt = ttnn.matmul(h_tt, to_dev(router_w, device))
    router_h = ttnn.to_torch(router_tt)[:seq_len].float()

    scores = torch.softmax(router_h, dim=-1)
    tw, ti = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)

    expert_to_tokens = {}
    for tok in range(seq_len):
        for k in range(NUM_EXPERTS_PER_TOK):
            eidx = ti[tok, k].item()
            w = tw[tok, k].item()
            if eidx not in expert_to_tokens:
                expert_to_tokens[eidx] = []
            expert_to_tokens[eidx].append((tok, w))

    output = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    active = set(ti.flatten().tolist())

    for eidx in active:
        gate_tt = to_dev(expert_gate[eidx], device)
        up_tt = to_dev(expert_up[eidx], device)
        down_tt = to_dev(expert_down[eidx], device)

        tinfo = expert_to_tokens[eidx]
        toks = [t[0] for t in tinfo]
        n = len(toks)
        np_ = ((n + TILE - 1) // TILE) * TILE
        inp = torch.nn.functional.pad(hidden_bf[toks], (0, 0, 0, np_ - n))
        inp_tt = to_dev(inp, device)

        g = ttnn.matmul(inp_tt, gate_tt)
        u = ttnn.matmul(inp_tt, up_tt)
        act = to_dev(torch.zeros(np_, MOE_INTER, dtype=torch.bfloat16), device)
        silu_mul_kernel(g, u, act)
        eout = ttnn.matmul(act, down_tt)
        eout_h = ttnn.to_torch(eout)[:n, :HIDDEN_SIZE].to(torch.bfloat16)

        for i, (tok, w) in enumerate(tinfo):
            output[tok] += eout_h[i] * w

    return output


def dense_mlp_host(hidden, fc1_w, fc1_b, fc2_w, fc2_b, device):
    """Run dense SiLU-gated MLP. fc1 projects to 2*intermediate, split into gate+up."""
    seq_len = hidden.shape[0]
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    h_padded = pad_to_tile(hidden.to(torch.bfloat16), dim=0)
    if h_padded.shape[0] < seq_pad:
        h_padded = torch.nn.functional.pad(h_padded, (0, 0, 0, seq_pad - h_padded.shape[0]))
    h_tt = to_dev(h_padded, device)

    # fc1: (seq, hidden) -> (seq, 2*intermediate)
    fc1_out = ttnn.matmul(h_tt, to_dev(fc1_w, device))
    fc1_h = ttnn.to_torch(fc1_out)[:seq_len].float()

    if fc1_b is not None:
        fc1_h = fc1_h + fc1_b.float()

    # SiLU gate: split into gate and up halves
    gate = fc1_h[:, :DRAFT_INTER]
    up = fc1_h[:, DRAFT_INTER:]
    activated = torch.nn.functional.silu(gate) * up

    # fc2: (seq, intermediate) -> (seq, hidden)
    act_padded = pad_to_tile(activated.to(torch.bfloat16), dim=0)
    if act_padded.shape[0] < seq_pad:
        act_padded = torch.nn.functional.pad(act_padded, (0, 0, 0, seq_pad - act_padded.shape[0]))
    act_tt = to_dev(act_padded, device)
    fc2_out = ttnn.matmul(act_tt, to_dev(fc2_w, device))
    fc2_h = ttnn.to_torch(fc2_out)[:seq_len].to(torch.bfloat16)

    if fc2_b is not None:
        fc2_h = fc2_h + fc2_b.to(torch.bfloat16)

    return fc2_h


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def build_key_map(weights_dir):
    index_path = f"{weights_dir}/model.safetensors.index.json"
    if not __import__('os').path.exists(index_path):
        # Single file
        return None
    with open(index_path) as f:
        index = json.load(f)
    return {k: f"{weights_dir}/{v}" for k, v in index["weight_map"].items()}


def get_tensor_from_map(key_map, key):
    fpath = key_map[key]
    with safe_open(fpath, framework="pt") as f:
        return f.get_tensor(key)


def get_tensor_single(path, key):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)


def load_target_layer(key_map, layer_idx):
    """Load a single target model layer's weights (host tensors)."""
    p = f"model.layers.{layer_idx}"
    layer = {}
    layer["input_norm_w"] = get_tensor_from_map(key_map, f"{p}.input_layernorm.weight").to(torch.bfloat16)
    layer["post_attn_norm_w"] = get_tensor_from_map(key_map, f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)
    layer["q_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
    layer["k_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
    layer["v_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
    layer["o_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
    layer["q_norm_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
    layer["k_norm_w"] = get_tensor_from_map(key_map, f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)
    layer["router_w"] = get_tensor_from_map(key_map, f"{p}.mlp.gate.weight").T.contiguous().to(torch.bfloat16)

    expert_gate, expert_up, expert_down = [], [], []
    for eidx in range(NUM_EXPERTS):
        ep = f"{p}.mlp.experts.{eidx}"
        expert_gate.append(get_tensor_from_map(key_map, f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16))
        expert_up.append(get_tensor_from_map(key_map, f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16))
        expert_down.append(get_tensor_from_map(key_map, f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16))
    layer["expert_gate"] = expert_gate
    layer["expert_up"] = expert_up
    layer["expert_down"] = expert_down
    return layer


def load_draft_weights(weights_path):
    """Load all DFlash draft model weights."""
    draft = {}
    with safe_open(weights_path, framework="pt") as f:
        # fc: projects concatenated target hidden states
        draft["fc_w"] = f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16)  # (5*2048, 2048)
        draft["hidden_norm_w"] = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        draft["final_norm_w"] = f.get_tensor("norm.weight").to(torch.bfloat16)

        for li in range(DRAFT_LAYERS):
            p = f"layers.{li}"
            draft[f"{p}.input_norm_w"] = f.get_tensor(f"{p}.input_layernorm.weight").to(torch.bfloat16)
            draft[f"{p}.post_attn_norm_w"] = f.get_tensor(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)
            draft[f"{p}.q_w"] = f.get_tensor(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
            draft[f"{p}.k_w"] = f.get_tensor(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
            draft[f"{p}.v_w"] = f.get_tensor(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
            draft[f"{p}.o_w"] = f.get_tensor(f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
            draft[f"{p}.q_norm_w"] = f.get_tensor(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
            draft[f"{p}.k_norm_w"] = f.get_tensor(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

            # Dense MLP (SiLU-gated): gate_proj + up_proj fused into one weight
            gate_w = f.get_tensor(f"{p}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16)
            up_w = f.get_tensor(f"{p}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16)
            draft[f"{p}.fc1_w"] = torch.cat([gate_w, up_w], dim=1)  # (2048, 2*6144)
            draft[f"{p}.fc2_w"] = f.get_tensor(f"{p}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16)

    return draft


# ---------------------------------------------------------------------------
# Target model forward (layer-by-layer to save memory)
# ---------------------------------------------------------------------------
def target_forward(input_ids, key_map, embed_w, lm_head_w, final_norm_w,
                   cos, sin, device, scaler_tt, mean_scale_tt,
                   output_hidden_states=False):
    """Run target model forward, loading one layer at a time.

    Returns: (logits, hidden_states_dict) where hidden_states_dict maps
    layer_idx -> hidden_states tensor for layers in TARGET_LAYER_IDS.
    """
    seq_len = input_ids.shape[0]
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    hidden = embed_w[input_ids].to(torch.bfloat16)  # (seq_len, HIDDEN_SIZE)
    hidden_states_out = {}

    for layer_idx in range(TARGET_LAYERS):
        t0 = time.time()
        layer = load_target_layer(key_map, layer_idx)
        t_load = time.time() - t0

        # Save hidden states for DFlash if needed (before this layer's transform)
        if output_hidden_states and layer_idx in TARGET_LAYER_IDS:
            hidden_states_out[layer_idx] = hidden.clone()

        # RMSNorm 1 (on device)
        h_padded = pad_to_tile(hidden, dim=0)
        if h_padded.shape[0] < seq_pad:
            h_padded = torch.nn.functional.pad(h_padded, (0, 0, 0, seq_pad - h_padded.shape[0]))
        h_tt = to_dev(h_padded, device)
        normed_tt = run_rmsnorm_tt(h_tt, layer["input_norm_w"], seq_pad, scaler_tt, mean_scale_tt, device)
        normed_host = ttnn.to_torch(normed_tt)[:seq_len].to(torch.bfloat16)

        # Attention (host)
        attn_out = gqa_attention_host(
            normed_host, layer["q_w"], layer["k_w"], layer["v_w"], layer["o_w"],
            layer["q_norm_w"], layer["k_norm_w"], cos, sin, seq_len,
        )

        # Residual 1
        hidden = hidden + attn_out

        # RMSNorm 2 (on device)
        h_padded = pad_to_tile(hidden, dim=0)
        if h_padded.shape[0] < seq_pad:
            h_padded = torch.nn.functional.pad(h_padded, (0, 0, 0, seq_pad - h_padded.shape[0]))
        h_tt = to_dev(h_padded, device)
        normed_moe_tt = run_rmsnorm_tt(h_tt, layer["post_attn_norm_w"], seq_pad, scaler_tt, mean_scale_tt, device)
        normed_moe_host = ttnn.to_torch(normed_moe_tt)[:seq_len].to(torch.bfloat16)

        # MoE (on device)
        moe_out = moe_forward_host(
            normed_moe_host, layer["router_w"],
            layer["expert_gate"], layer["expert_up"], layer["expert_down"], device,
        )

        # Residual 2
        hidden = hidden + moe_out

        elapsed = time.time() - t0
        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            print(f"  Target layer {layer_idx}: {elapsed:.2f}s (load: {t_load:.1f}s)")

        del layer

    # Final norm
    h_padded = pad_to_tile(hidden, dim=0)
    if h_padded.shape[0] < seq_pad:
        h_padded = torch.nn.functional.pad(h_padded, (0, 0, 0, seq_pad - h_padded.shape[0]))
    h_tt = to_dev(h_padded, device)
    final_tt = run_rmsnorm_tt(h_tt, final_norm_w, seq_pad, scaler_tt, mean_scale_tt, device)
    final_host = ttnn.to_torch(final_tt)[:seq_len].to(torch.bfloat16)

    # LM head
    lm_padded = pad_to_tile(final_host, dim=0)
    if lm_padded.shape[0] < seq_pad:
        lm_padded = torch.nn.functional.pad(lm_padded, (0, 0, 0, seq_pad - lm_padded.shape[0]))
    lm_tt = to_dev(lm_padded, device)
    logits_tt = ttnn.matmul(lm_tt, to_dev(lm_head_w, device))
    logits = ttnn.to_torch(logits_tt)[:seq_len, :VOCAB_SIZE].float()

    return logits, hidden_states_out


# ---------------------------------------------------------------------------
# Draft model forward
# ---------------------------------------------------------------------------
def draft_forward(noise_embedding, target_hidden, draft_weights, cos, sin,
                  device, scaler_tt, mean_scale_tt):
    """Run DFlash draft model forward.

    Args:
        noise_embedding: (block_size, HIDDEN_SIZE) - embedded draft tokens
        target_hidden: (ctx_len, HIDDEN_SIZE) - projected target features
        cos, sin: RoPE tables covering ctx_len + block_size positions
    Returns:
        output: (block_size, HIDDEN_SIZE)
    """
    seq_len = noise_embedding.shape[0]
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE
    dw = draft_weights

    hidden = noise_embedding.to(torch.bfloat16)

    for li in range(DRAFT_LAYERS):
        p = f"layers.{li}"

        # RMSNorm 1
        normed = rms_norm_host(hidden, dw[f"{p}.input_norm_w"])

        # Cross-attention: Q from draft, K/V from [target_hidden, draft_hidden]
        attn_out = gqa_attention_host(
            normed, dw[f"{p}.q_w"], dw[f"{p}.k_w"], dw[f"{p}.v_w"], dw[f"{p}.o_w"],
            dw[f"{p}.q_norm_w"], dw[f"{p}.k_norm_w"],
            cos, sin, seq_len, is_causal=False, ctx_hidden=target_hidden,
        )

        # Residual 1
        hidden = hidden + attn_out

        # RMSNorm 2
        normed2 = rms_norm_host(hidden, dw[f"{p}.post_attn_norm_w"])

        # Dense MLP (on device)
        mlp_out = dense_mlp_host(normed2, dw[f"{p}.fc1_w"], None, dw[f"{p}.fc2_w"], None, device)

        # Residual 2
        hidden = hidden + mlp_out

    # Final norm
    hidden = rms_norm_host(hidden, dw["final_norm_w"])
    return hidden


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_token(logits, temperature=0.0):
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# DFlash speculative decoding
# ---------------------------------------------------------------------------
def spec_generate(input_ids, key_map, embed_w, lm_head_w, final_norm_w,
                  draft_weights, device, max_new_tokens=128, temperature=0.0):
    """DFlash speculative decoding.

    1. Target prefill -> extract hidden states from target_layer_ids
    2. Draft generates block of tokens using target hidden context
    3. Target verifies the block
    4. Accept prefix of matching tokens
    """
    prompt_len = input_ids.shape[0]
    total_len = prompt_len + max_new_tokens

    cos, sin = precompute_rope(total_len + BLOCK_SIZE)
    scaler_tt = to_dev(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale_tt = to_dev(torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16), device)

    output_ids = torch.full((total_len + BLOCK_SIZE,), MASK_TOKEN_ID, dtype=torch.long)
    output_ids[:prompt_len] = input_ids

    # Prefill
    print("=" * 60)
    print("Prefill...")
    t0 = time.time()
    logits, target_hidden_states = target_forward(
        input_ids, key_map, embed_w, lm_head_w, final_norm_w,
        cos, sin, device, scaler_tt, mean_scale_tt,
        output_hidden_states=True,
    )
    t_prefill = time.time() - t0
    print(f"Prefill: {t_prefill:.1f}s ({prompt_len} tokens, {prompt_len/t_prefill:.1f} tok/s)")

    # Sample first token
    first_token = sample_token(logits[-1:], temperature)
    output_ids[prompt_len] = first_token

    # Extract and project target hidden context
    target_features = torch.cat(
        [target_hidden_states[lid] for lid in TARGET_LAYER_IDS], dim=-1
    )  # (prompt_len, 5 * HIDDEN_SIZE)

    # Project: fc(target_features) -> (prompt_len, HIDDEN_SIZE)
    seq_pad = ((prompt_len + TILE - 1) // TILE) * TILE
    tf_padded = pad_to_tile(target_features, dim=0)
    if tf_padded.shape[0] < seq_pad:
        tf_padded = torch.nn.functional.pad(tf_padded, (0, 0, 0, seq_pad - tf_padded.shape[0]))
    tf_tt = to_dev(tf_padded, device)
    proj_tt = ttnn.matmul(tf_tt, to_dev(draft_weights["fc_w"], device))
    projected = ttnn.to_torch(proj_tt)[:prompt_len, :HIDDEN_SIZE].to(torch.bfloat16)
    target_hidden_ctx = rms_norm_host(projected, draft_weights["hidden_norm_w"])

    # Decode loop
    acceptance_lengths = []
    start = prompt_len
    generated = 0

    print("=" * 60)
    print("Decoding...")

    while start < total_len:
        t_step = time.time()

        # Draft: generate block of tokens
        block_ids = output_ids[start:start + BLOCK_SIZE].clone()
        noise_embedding = embed_w[block_ids].to(torch.bfloat16)

        # Use last target_hidden_ctx for context
        draft_out = draft_forward(
            noise_embedding, target_hidden_ctx,
            draft_weights, cos, sin, device, scaler_tt, mean_scale_tt,
        )

        # Get draft logits using target's LM head
        do_padded = pad_to_tile(draft_out, dim=0)
        seq_pad_d = ((BLOCK_SIZE + TILE - 1) // TILE) * TILE
        if do_padded.shape[0] < seq_pad_d:
            do_padded = torch.nn.functional.pad(do_padded, (0, 0, 0, seq_pad_d - do_padded.shape[0]))
        do_tt = to_dev(do_padded, device)
        draft_logits_tt = ttnn.matmul(do_tt, to_dev(lm_head_w, device))
        draft_logits = ttnn.to_torch(draft_logits_tt)[:BLOCK_SIZE, :VOCAB_SIZE].float()

        # Sample draft tokens (skip first position which is already known)
        draft_tokens = sample_token(draft_logits[:-1], temperature)
        block_ids[1:] = draft_tokens

        # Target: verify the block
        t_verify = time.time()
        verify_logits, verify_hidden = target_forward(
            block_ids, key_map, embed_w, lm_head_w, final_norm_w,
            cos, sin, device, scaler_tt, mean_scale_tt,
            output_hidden_states=True,
        )
        t_verify = time.time() - t_verify

        # Sample from target
        posterior = sample_token(verify_logits, temperature)

        # Accept matching prefix
        matches = (block_ids[1:] == posterior[:-1])
        acceptance_length = matches.to(torch.int64).cumprod(0).sum().item()

        output_ids[start:start + acceptance_length + 1] = block_ids[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = posterior[acceptance_length]
        start += acceptance_length + 1
        generated += acceptance_length + 1
        acceptance_lengths.append(acceptance_length + 1)

        # Update target hidden context from verification
        verify_features = torch.cat(
            [verify_hidden[lid] for lid in TARGET_LAYER_IDS], dim=-1
        )
        vf_len = verify_features.shape[0]
        vf_pad = ((vf_len + TILE - 1) // TILE) * TILE
        vf_padded = pad_to_tile(verify_features, dim=0)
        if vf_padded.shape[0] < vf_pad:
            vf_padded = torch.nn.functional.pad(vf_padded, (0, 0, 0, vf_pad - vf_padded.shape[0]))
        vf_tt = to_dev(vf_padded, device)
        vp_tt = ttnn.matmul(vf_tt, to_dev(draft_weights["fc_w"], device))
        vp_host = ttnn.to_torch(vp_tt)[:vf_len, :HIDDEN_SIZE].to(torch.bfloat16)
        target_hidden_ctx = rms_norm_host(vp_host, draft_weights["hidden_norm_w"])

        elapsed = time.time() - t_step
        avg_accept = sum(acceptance_lengths) / len(acceptance_lengths)
        print(f"  Step {len(acceptance_lengths)}: accepted {acceptance_length+1}/{BLOCK_SIZE} "
              f"(avg={avg_accept:.1f}), verify={t_verify:.1f}s, total={elapsed:.1f}s, "
              f"generated={generated}")

        # Check EOS
        if output_ids[start - 1].item() in (151643, 151645):
            break

    output_ids = output_ids[:start]
    output_ids = output_ids[output_ids != MASK_TOKEN_ID]

    avg_accept = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0
    print(f"\nDone: {generated} tokens, avg acceptance={avg_accept:.1f}, "
          f"steps={len(acceptance_lengths)}")

    return output_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("DFlash Speculative Decoding on Tenstorrent Blackhole")
    print(f"  Target: Qwen3-Coder-30B-A3B ({TARGET_LAYERS} layers, {NUM_EXPERTS} experts)")
    print(f"  Draft: DFlash ({DRAFT_LAYERS} layers, block_size={BLOCK_SIZE})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)

    try:
        # Load shared weights
        print("Loading target weight index...")
        key_map = build_key_map(TARGET_WEIGHTS_DIR)

        print("Loading embedding + LM head...")
        embed_w = get_tensor_from_map(key_map, "model.embed_tokens.weight").to(torch.bfloat16)
        lm_head_w = get_tensor_from_map(key_map, "lm_head.weight").T.contiguous().to(torch.bfloat16)
        lm_head_w = pad_to_tile(lm_head_w, dim=1)
        final_norm_w = get_tensor_from_map(key_map, "model.norm.weight").to(torch.bfloat16)

        print("Loading DFlash draft model...")
        draft_weights = load_draft_weights(f"{DRAFT_WEIGHTS_DIR}/model.safetensors")
        print(f"  Draft weights: {len(draft_weights)} tensors")

        # Tokenize prompt
        prompt = "Write a Python function that computes fibonacci numbers."
        # Simple tokenization: just use the prompt text as-is with a known token sequence
        # For real usage, load the tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TARGET_WEIGHTS_DIR)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        except Exception as e:
            print(f"Tokenizer not available ({e}), using dummy tokens")
            input_ids = torch.tensor([151643, 872, 13, 5765, 264, 13325, 734, 430,
                                       58303, 87971, 5109, 13, 151645, 198])

        print(f"Prompt: {prompt}")
        print(f"Input tokens: {input_ids.shape[0]}")

        # Run speculative decoding
        output_ids = spec_generate(
            input_ids, key_map, embed_w, lm_head_w, final_norm_w,
            draft_weights, device, max_new_tokens=64, temperature=0.0,
        )

        # Decode
        try:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"\n--- Output ---\n{output_text}")
        except:
            print(f"\nOutput token IDs: {output_ids.tolist()}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
