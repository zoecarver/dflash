"""Test inference with real Qwen3-Coder-30B-A3B weights.

Loads one layer from safetensors, runs a forward pass, checks output is reasonable.
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

TILE = 32
HIDDEN_SIZE = 2048
HIDDEN_TILES = HIDDEN_SIZE // TILE
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
MOE_INTER = 768
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
VOCAB_SIZE = 151936
VOCAB_PAD = ((VOCAB_SIZE + TILE - 1) // TILE) * TILE

WEIGHTS_DIR = "/workspace/qwen-coder-30b-a3b/weights"

rmsnorm_kernel = make_rmsnorm_kernel(dim_tiles=HIDDEN_TILES, eps=RMS_NORM_EPS)


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
    cos_t = torch.cos(angles).to(torch.bfloat16)
    sin_t = torch.sin(angles).to(torch.bfloat16)
    return cos_t, sin_t


def apply_rope(q, k, cos, sin):
    cos_full = cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2).float()
    sin_full = sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2).float()
    q_len = q.shape[2]
    q_r = q * cos_full[:, :, :q_len, :] + rotate_half(q) * sin_full[:, :, :q_len, :]
    k_r = k * cos_full + rotate_half(k) * sin_full
    return q_r, k_r


def build_key_map():
    """Build mapping from weight key to safetensors file."""
    index_path = f"{WEIGHTS_DIR}/model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    key_to_file = {}
    for key, filename in index["weight_map"].items():
        key_to_file[key] = f"{WEIGHTS_DIR}/{filename}"
    return key_to_file


def get_tensor(key_map, key):
    fpath = key_map[key]
    with safe_open(fpath, framework="pt") as f:
        return f.get_tensor(key)


def test_single_layer(device):
    """Load layer 0 weights and run a single forward pass."""
    print("Building weight key map...")
    key_map = build_key_map()

    seq_len = 32
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    # Load embedding
    print("Loading embedding...")
    embed_w = get_tensor(key_map, "model.embed_tokens.weight").to(torch.bfloat16)

    # Load layer 0 weights
    print("Loading layer 0 weights...")
    t0 = time.time()
    li = 0
    prefix = f"model.layers.{li}"

    input_norm_w = get_tensor(key_map, f"{prefix}.input_layernorm.weight").to(torch.bfloat16)
    post_attn_norm_w = get_tensor(key_map, f"{prefix}.post_attention_layernorm.weight").to(torch.bfloat16)

    q_w = get_tensor(key_map, f"{prefix}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
    k_w = get_tensor(key_map, f"{prefix}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
    v_w = get_tensor(key_map, f"{prefix}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
    o_w = get_tensor(key_map, f"{prefix}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
    q_norm_w = get_tensor(key_map, f"{prefix}.self_attn.q_norm.weight").to(torch.bfloat16)
    k_norm_w = get_tensor(key_map, f"{prefix}.self_attn.k_norm.weight").to(torch.bfloat16)

    router_w = get_tensor(key_map, f"{prefix}.mlp.gate.weight").T.contiguous().to(torch.bfloat16)

    # Load only the experts that will be selected (load all for now, small per-expert)
    expert_gate = []
    expert_up = []
    expert_down = []
    for eidx in range(NUM_EXPERTS):
        ep = f"{prefix}.mlp.experts.{eidx}"
        expert_gate.append(get_tensor(key_map, f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16))
        expert_up.append(get_tensor(key_map, f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16))
        expert_down.append(get_tensor(key_map, f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16))
    print(f"Layer 0 weights loaded in {time.time() - t0:.1f}s")

    # Create input: simple token sequence
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] * 4)  # 32 tokens

    # Embedding
    hidden = embed_w[input_ids].to(torch.bfloat16)  # (32, 2048)

    # Setup device tensors
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / HIDDEN_SIZE, dtype=torch.bfloat16)
    scaler_tt = to_dev(scaler, device)
    mean_scale_tt = to_dev(mean_scale, device)

    hidden_padded = pad_to_tile(hidden, dim=0)
    if hidden_padded.shape[0] < seq_pad:
        hidden_padded = torch.nn.functional.pad(hidden_padded, (0, 0, 0, seq_pad - hidden_padded.shape[0]))
    hidden_tt = to_dev(hidden_padded, device)

    # RMSNorm 1
    print("Running RMSNorm 1...")
    w_exp = input_norm_w.unsqueeze(0).expand(seq_pad, -1).contiguous()
    normed_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(hidden_tt, to_dev(w_exp, device), scaler_tt, mean_scale_tt, normed_tt)

    # Attention
    print("Running Attention...")
    q_tt = ttnn.matmul(normed_tt, to_dev(q_w, device))
    k_tt = ttnn.matmul(normed_tt, to_dev(k_w, device))
    v_tt = ttnn.matmul(normed_tt, to_dev(v_w, device))

    q_h = ttnn.to_torch(q_tt)[:seq_len].float().view(1, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k_h = ttnn.to_torch(k_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_h = ttnn.to_torch(v_tt)[:seq_len].float().view(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    q_h = rms_norm_host(q_h.to(torch.bfloat16), q_norm_w).float()
    k_h = rms_norm_host(k_h.to(torch.bfloat16), k_norm_w).float()

    cos, sin = precompute_rope(seq_len)
    q_h, k_h = apply_rope(q_h, k_h, cos, sin)

    # GQA expand
    num_groups = NUM_Q_HEADS // NUM_KV_HEADS
    k_h = k_h.repeat(1, num_groups, 1, 1)
    v_h = v_h.repeat(1, num_groups, 1, 1)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_w = torch.softmax(torch.matmul(q_h, k_h.transpose(-2, -1)) * scale, dim=-1)
    attn_out = torch.matmul(attn_w, v_h).transpose(1, 2).contiguous().view(seq_len, -1).to(torch.bfloat16)

    attn_padded = pad_to_tile(attn_out, dim=0)
    if attn_padded.shape[0] < seq_pad:
        attn_padded = torch.nn.functional.pad(attn_padded, (0, 0, 0, seq_pad - attn_padded.shape[0]))
    o_proj = ttnn.matmul(to_dev(attn_padded, device), to_dev(o_w, device))

    # Residual 1
    res1 = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    residual_add_kernel(hidden_tt, o_proj, res1)

    # RMSNorm 2
    print("Running RMSNorm 2...")
    w2_exp = post_attn_norm_w.unsqueeze(0).expand(seq_pad, -1).contiguous()
    normed_moe = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    rmsnorm_kernel(res1, to_dev(w2_exp, device), scaler_tt, mean_scale_tt, normed_moe)

    # MoE
    print("Running MoE (128 experts, top-8)...")
    t_moe = time.time()
    router_tt = ttnn.matmul(normed_moe, to_dev(router_w, device))
    router_h = ttnn.to_torch(router_tt)[:seq_len].float()
    scores = torch.softmax(router_h, dim=-1)
    tw, ti = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)

    normed_moe_h = ttnn.to_torch(normed_moe)[:seq_len].to(torch.bfloat16)

    # Load active expert weights to device
    active_experts = set(ti.flatten().tolist())
    print(f"  Active experts: {len(active_experts)} unique out of {NUM_EXPERTS}")

    expert_gate_tt = {}
    expert_up_tt = {}
    expert_down_tt = {}
    for eidx in active_experts:
        expert_gate_tt[eidx] = to_dev(expert_gate[eidx], device)
        expert_up_tt[eidx] = to_dev(expert_up[eidx], device)
        expert_down_tt[eidx] = to_dev(expert_down[eidx], device)

    expert_to_tokens = {}
    for tok in range(seq_len):
        for kidx in range(NUM_EXPERTS_PER_TOK):
            eidx = ti[tok, kidx].item()
            w = tw[tok, kidx].item()
            if eidx not in expert_to_tokens:
                expert_to_tokens[eidx] = []
            expert_to_tokens[eidx].append((tok, w))

    moe_out_h = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    for eidx, tinfo in expert_to_tokens.items():
        toks = [t[0] for t in tinfo]
        n = len(toks)
        np_ = ((n + TILE - 1) // TILE) * TILE
        inp = torch.nn.functional.pad(normed_moe_h[toks], (0, 0, 0, np_ - n))
        inp_tt = to_dev(inp, device)

        g = ttnn.matmul(inp_tt, expert_gate_tt[eidx])
        u = ttnn.matmul(inp_tt, expert_up_tt[eidx])
        act = to_dev(torch.zeros(np_, MOE_INTER, dtype=torch.bfloat16), device)
        silu_mul_kernel(g, u, act)
        eout = ttnn.matmul(act, expert_down_tt[eidx])
        eout_h = ttnn.to_torch(eout)[:n, :HIDDEN_SIZE].to(torch.bfloat16)

        for i, (tok, w) in enumerate(tinfo):
            moe_out_h[tok] += eout_h[i] * w

    moe_padded = pad_to_tile(moe_out_h, dim=0)
    if moe_padded.shape[0] < seq_pad:
        moe_padded = torch.nn.functional.pad(moe_padded, (0, 0, 0, seq_pad - moe_padded.shape[0]))

    # Residual 2
    final_tt = to_dev(torch.zeros(seq_pad, HIDDEN_SIZE, dtype=torch.bfloat16), device)
    residual_add_kernel(res1, to_dev(moe_padded, device), final_tt)

    moe_time = time.time() - t_moe
    print(f"  MoE done in {moe_time:.2f}s")

    # Readback and check
    result = ttnn.to_torch(final_tt)[:seq_len, :HIDDEN_SIZE]

    print(f"\nLayer 0 output stats:")
    print(f"  Mean: {result.float().mean().item():.6f}")
    print(f"  Std: {result.float().std().item():.6f}")
    print(f"  Min: {result.float().min().item():.4f}")
    print(f"  Max: {result.float().max().item():.4f}")
    print(f"  Any NaN: {result.isnan().any().item()}")
    print(f"  Any Inf: {result.isinf().any().item()}")

    ok = not result.isnan().any().item() and not result.isinf().any().item()
    ok = ok and result.float().std().item() > 0.01  # not all zeros
    print(f"\n{'PASS' if ok else 'FAIL'}: Layer 0 produces valid output")
    return ok


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = test_single_layer(device)
    finally:
        ttnn.close_device(device)
    if not ok:
        exit(1)
