"""Qwen3-Coder-30B-A3B target model on Tenstorrent 4-chip TP.

48-layer MoE model with traced execution, zero host transfers in hot loop.
"""
import time
import json
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from collections import defaultdict

from device import (
    TILE, HIDDEN, HTILES, HDIM, HDIM_TILES,
    NQH, NKVH, GQA, EPS, ROPE_THETA, VOCAB,
    N_CHIPS, NQH_TP, NKVH_TP, Q_TP, KV_TP,
    TARGET_DIR,
    rmsnorm_k, q_rope_k, k_rope_k,
    _p, rep, shd, rb, rb_dim1,
    dev_norm, dev_add,
)

TLAYERS = 48
MOE_INTER = 768
NEXPERTS = 128
TOPK = 8
EPC = NEXPERTS // N_CHIPS  # 32 experts per chip


# ---------------------------------------------------------------------------
# Load target weights to device
# ---------------------------------------------------------------------------
def load_target_weights(d):
    w = {}
    t0 = time.time()

    with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}

    def gt(k):
        with safe_open(kf[k], framework="pt") as f:
            return f.get_tensor(k)

    # Embedding: keep on host for token gather (tiny transfer: seq_len*HIDDEN per call)
    print("  embed/lm_head...")
    w["embed_h"] = gt("model.embed_tokens.weight").to(torch.bfloat16)
    w["lm_head"] = shd(gt("lm_head.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
    fn_w = gt("model.norm.weight").to(torch.bfloat16)
    w["final_norm"] = fn_w  # host tensor for draft model
    w["final_norm_tt"] = rep(fn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

    w["sc"] = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)
    w["ms"] = rep(torch.full((TILE, TILE), 1.0 / HIDDEN, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)

    # Per-head norm support for TT-Lang q_head_norm_k/k_head_norm_k (unused, see PCC note above)
    w["ms_head"] = rep(torch.full((TILE, TILE), 1.0 / HDIM, dtype=torch.bfloat16), d)

    # MoE routing expansion matrix: maps (sp, 128) routing mask to (sp, 24576) per chip
    # Each expert's scalar weight is broadcast across MOE_INTER=768 columns via matmul
    # replicated @ sharded_dim1 = sharded_dim1 (same pattern as QKV projections)
    expand = torch.zeros(NEXPERTS, NEXPERTS * MOE_INTER, dtype=torch.bfloat16)
    for e in range(NEXPERTS):
        expand[e, e * MOE_INTER:(e + 1) * MOE_INTER] = 1.0
    w["moe_expand"] = shd(expand, d, dim=1)  # (128, 24576) per chip

    # RoPE tables (replicated)
    max_seq = 4096
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_t = torch.cos(angles).to(torch.bfloat16)  # (max_seq, HDIM/2)
    sin_t = torch.sin(angles).to(torch.bfloat16)
    # Half-width tables for host fallback (draft model)
    w["rope_cos"] = rep(cos_t, d)
    w["rope_sin"] = rep(sin_t, d)
    # Full-width tables for device RoPE kernel
    cos_full = cos_t.repeat(1, 2)[:, :HDIM]  # (max_seq, HDIM)
    sin_full = sin_t.repeat(1, 2)[:, :HDIM]
    sin_adj = sin_full.clone()
    sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
    w["rope_cos_full"] = rep(cos_full, d)
    w["rope_sin_adj"] = rep(sin_adj, d)

    for li in range(TLAYERS):
        p = f"model.layers.{li}"
        lp = f"t.{li}"

        in_w = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
        pa_w = gt(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)
        w[f"{lp}.in_w"] = in_w  # host tensor for draft model
        w[f"{lp}.pa_w"] = pa_w
        # Precomputed TILE-expanded norm weights for device (eliminates rep() in hot loop)
        w[f"{lp}.in_w_tt"] = rep(in_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        w[f"{lp}.pa_w_tt"] = rep(pa_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

        # Combined QKV + swap: [Q|K|V|Q_swap|K_swap] column-parallel
        qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
        kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
        vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)

        # For TP, interleave so shard(dim=1) gives each chip its heads
        # Q: (HIDDEN, NQH*HDIM) -> shard dim=1 -> each chip gets (HIDDEN, NQH_TP*HDIM)
        # K: (HIDDEN, NKVH*HDIM) -> shard dim=1 -> each chip gets (HIDDEN, NKVH_TP*HDIM)
        w[f"{lp}.qw"] = shd(qw, d, dim=1)
        w[f"{lp}.kw"] = shd(kw, d, dim=1)
        w[f"{lp}.vw"] = shd(vw, d, dim=1)

        # O proj: row-parallel (dim=0) + all_reduce
        ow = gt(f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
        w[f"{lp}.ow"] = shd(ow, d, dim=0)

        qnw_raw = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
        knw_raw = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)
        w[f"{lp}.qnw"] = qnw_raw  # host tensor for draft model
        w[f"{lp}.knw"] = knw_raw
        # Device tensors for TT-Lang per-head rmsnorm kernel (unused, see PCC note above)
        w[f"{lp}.qnw_tt"] = rep(qnw_raw.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        w[f"{lp}.knw_tt"] = rep(knw_raw.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        # Device tensors for ttnn.rms_norm (1, HDIM) -- used in hot loop
        w[f"{lp}.qnw_dev"] = rep(qnw_raw.unsqueeze(0).contiguous(), d)
        w[f"{lp}.knw_dev"] = rep(knw_raw.unsqueeze(0).contiguous(), d)

        # Router: replicated (softmax needs all 128 experts)
        w[f"{lp}.rw"] = rep(gt(f"{p}.mlp.gate.weight").T.contiguous().to(torch.bfloat16), d)

        # Expert weights: concat all then shard across chips
        gate_parts, up_parts, down_parts = [], [], []
        for ei in range(NEXPERTS):
            ep = f"{p}.mlp.experts.{ei}"
            gate_parts.append(gt(f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16))
            up_parts.append(gt(f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16))
            down_parts.append(gt(f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16))

        gate_all = torch.cat(gate_parts, dim=1)  # (2048, 128*768=98304)
        up_all = torch.cat(up_parts, dim=1)
        down_all = torch.cat(down_parts, dim=0)  # (128*768=98304, 2048)

        w[f"{lp}.gate_all"] = shd(gate_all, d, dim=1)  # each chip: (2048, 32*768)
        w[f"{lp}.up_all"] = shd(up_all, d, dim=1)
        w[f"{lp}.down_all"] = shd(down_all, d, dim=0)  # each chip: (32*768, 2048)

        del gate_parts, up_parts, down_parts, gate_all, up_all, down_all

        if (li + 1) % 4 == 0:
            print(f"  target layer {li+1}/{TLAYERS} ({time.time()-t0:.0f}s)")

    print(f"Target weights loaded in {time.time()-t0:.0f}s")
    return w


# ---------------------------------------------------------------------------
# Scratch preallocation for traced hot loop
# ---------------------------------------------------------------------------
def prealloc_scratch(sp, d):
    """Pre-allocate all reusable scratch tensors for target_fwd.

    Tensors are reused across layers via output_tensor params to avoid
    allocation during trace replay.

    Hot, small tensors go in L1 for lower latency between ops.
    Large tensors (MoE activations, logits) stay in DRAM.
    """
    L1 = ttnn.L1_MEMORY_CONFIG
    s = {}
    # All scratch in L1 interleaved (~800MB available across all cores)
    def _rep(shape): return rep(torch.zeros(*shape, dtype=torch.bfloat16), d, mem=L1)
    def _shd(shape, dim): return shd(torch.zeros(*shape, dtype=torch.bfloat16), d, dim=dim, mem=L1)

    # Norm + add outputs
    s["norm"] = _rep((sp, HIDDEN))
    s["add"] = _rep((sp, HIDDEN))
    # Attention scratch
    s["q"] = _shd((sp, Q_TP * N_CHIPS), dim=1)
    s["k"] = _shd((sp, KV_TP * N_CHIPS), dim=1)
    s["v"] = _shd((sp, KV_TP * N_CHIPS), dim=1)
    s["q_rope"] = _shd((sp, Q_TP * N_CHIPS), dim=1)
    s["k_rope"] = _shd((sp, KV_TP * N_CHIPS), dim=1)
    s["o"] = _rep((sp, HIDDEN))
    # MoE scratch
    moe_cols = EPC * MOE_INTER * N_CHIPS
    s["router"] = _rep((sp, NEXPERTS))
    s["gate"] = _shd((sp, moe_cols), dim=1)
    s["up"] = _shd((sp, moe_cols), dim=1)
    s["silu"] = _shd((sp, moe_cols), dim=1)
    s["act"] = _shd((sp, moe_cols), dim=1)
    s["weighted"] = _shd((sp, moe_cols), dim=1)
    s["moe_out"] = _rep((sp, HIDDEN))
    s["routing_tt"] = _shd((sp, moe_cols), dim=1)
    s["down"] = _rep((sp, HIDDEN))
    # Final
    s["logits"] = _shd((sp, VOCAB), dim=1)
    return s


# ---------------------------------------------------------------------------
# On-device building blocks
# ---------------------------------------------------------------------------
def dev_attn(normed, w, lp, sp, s):
    """GQA attention. QKV on device, QK-norm + RoPE on device via TT-Lang."""
    # Linear: QKV projections (column-parallel, each chip has its heads)
    ttnn.matmul(normed, w[f"{lp}.qw"], optional_output_tensor=s["q"])
    ttnn.matmul(normed, w[f"{lp}.kw"], optional_output_tensor=s["k"])
    ttnn.matmul(normed, w[f"{lp}.vw"], optional_output_tensor=s["v"])

    # Device: per-head RMSNorm via reshape + ttnn.rms_norm
    q_flat = ttnn.reshape(s["q"], (sp * NQH_TP, HDIM))
    k_flat = ttnn.reshape(s["k"], (sp * NKVH_TP, HDIM))
    q_normed_flat = ttnn.rms_norm(q_flat, weight=w[f"{lp}.qnw_dev"], epsilon=EPS)
    k_normed_flat = ttnn.rms_norm(k_flat, weight=w[f"{lp}.knw_dev"], epsilon=EPS)
    q_normed = ttnn.reshape(q_normed_flat, (sp, NQH_TP * HDIM))
    k_normed = ttnn.reshape(k_normed_flat, (sp, NKVH_TP * HDIM))

    # Device: RoPE on normed Q and K
    q_rope_k(q_normed, w["rope_cos_sp"], w["rope_sin_sp"], s["q_rope"])
    k_rope_k(k_normed, w["rope_cos_sp"], w["rope_sin_sp"], s["k_rope"])

    # Device: reshape for per-chip SDPA
    q4 = ttnn.transpose(ttnn.reshape(s["q_rope"], (1, sp, NQH_TP, HDIM)), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(s["k_rope"], (1, sp, NKVH_TP, HDIM)), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(s["v"], (1, sp, NKVH_TP, HDIM)), 1, 2)

    # Per-chip SDPA with GQA
    attn = ttnn.transformer.scaled_dot_product_attention(q4, k4, v4, is_causal=True)

    # Reshape back: (1, NQH_TP, sp, HDIM) -> (sp, NQH_TP*HDIM)
    attn_tt = ttnn.reshape(ttnn.transpose(attn, 1, 2), (sp, NQH_TP * HDIM))

    # Linear: O projection (row-parallel) + collective all_reduce
    ttnn.matmul(attn_tt, w[f"{lp}.ow"], optional_output_tensor=s["o"])
    return ttnn.all_reduce(s["o"])


def dev_moe(h, w, lp, s):
    """MoE: all experts compute, weighted by top-8 softmax scores, all on device.

    Device-only routing: softmax + topk + scatter on replicated scores,
    then matmul-based expansion (replicated @ sharded = sharded).
    """
    # Router + softmax (replicated: all 128 experts needed for correct topk)
    ttnn.matmul(h, w[f"{lp}.rw"], optional_output_tensor=s["router"])
    scores = ttnn.softmax(s["router"], dim=-1)

    # Top-8 routing on device
    topk_vals, topk_idx = ttnn.topk(scores, TOPK)
    topk_sum = ttnn.sum(topk_vals, dim=-1, keepdim=True)
    topk_norm = topk_vals * ttnn.reciprocal(topk_sum)

    # Scatter into full routing mask
    routing_mask = ttnn.scatter(ttnn.zeros_like(scores), 1, topk_idx, topk_norm)

    # Expand via matmul: (sp, 128) @ (128, 24576) -> (sp, 24576) per chip
    ttnn.matmul(routing_mask, w["moe_expand"], optional_output_tensor=s["routing_tt"])

    # Linear: batched gate/up across all chip-local experts
    ttnn.matmul(h, w[f"{lp}.gate_all"], optional_output_tensor=s["gate"])
    ttnn.matmul(h, w[f"{lp}.up_all"], optional_output_tensor=s["up"])
    ttnn.silu(s["gate"], output_tensor=s["silu"])
    ttnn.multiply(s["silu"], s["up"], output_tensor=s["act"])

    # Apply routing weights
    ttnn.multiply(s["act"], s["routing_tt"], output_tensor=s["weighted"])

    # Linear: down projection (row-parallel) + collective all_reduce
    ttnn.matmul(s["weighted"], w[f"{lp}.down_all"], optional_output_tensor=s["moe_out"])
    return ttnn.all_reduce(s["moe_out"])


# ---------------------------------------------------------------------------
# Target forward
# ---------------------------------------------------------------------------
def target_fwd(h, w, sp, d, s):
    for li in range(TLAYERS):
        lp = f"t.{li}"

        n = dev_norm(h, f"{lp}.in_w_tt", w, s["norm"])
        attn_out = dev_attn(n, w, lp, sp, s)
        h = dev_add(h, attn_out, s["add"])

        nm = dev_norm(h, f"{lp}.pa_w_tt", w, s["norm"])
        moe_out = dev_moe(nm, w, lp, s)
        h = dev_add(h, moe_out, s["add"])

    fn = dev_norm(h, "final_norm_tt", w, s["norm"])
    ttnn.matmul(fn, w["lm_head"], optional_output_tensor=s["logits"])
    return s["logits"]


def target_fwd_profiled(h, w, sp, d, s):
    """Profiled version of target_fwd -- syncs after each op to measure time."""
    timers = defaultdict(float)
    def _t(name, fn):
        ttnn.synchronize_device(d)
        t0 = time.time()
        result = fn()
        ttnn.synchronize_device(d)
        timers[name] += time.time() - t0
        return result

    for li in range(TLAYERS):
        lp = f"t.{li}"
        n = _t("norm", lambda: dev_norm(h, f"{lp}.in_w_tt", w, s["norm"]))
        attn_out = _t("attn", lambda: dev_attn(n, w, lp, sp, s))
        h = _t("add", lambda: dev_add(h, attn_out, s["add"]))
        nm = _t("norm", lambda: dev_norm(h, f"{lp}.pa_w_tt", w, s["norm"]))
        moe_out = _t("moe", lambda: dev_moe(nm, w, lp, s))
        h = _t("add", lambda: dev_add(h, moe_out, s["add"]))

    fn = _t("norm", lambda: dev_norm(h, "final_norm_tt", w, s["norm"]))
    _t("lm_head", lambda: ttnn.matmul(fn, w["lm_head"], optional_output_tensor=s["logits"]))

    print("--- Per-op time (48 layers total) ---")
    for name, elapsed in sorted(timers.items(), key=lambda x: -x[1]):
        print(f"  {name:10s}: {elapsed:.3f}s")
    print(f"  {'TOTAL':10s}: {sum(timers.values()):.3f}s")
    return s["logits"]


# ---------------------------------------------------------------------------
# Tracing: preallocate -> compile -> capture -> replay
# ---------------------------------------------------------------------------
def prealloc_target_scratch(sp, w, d):
    """Pre-allocate all scratch tensors and compile target_fwd for tracing.

    Returns scratch dict with preallocated device tensors and trace state.
    The compile pass triggers JIT compilation of all TT-Lang kernels
    and ttnn op programs so the trace capture has no first-run overhead.
    """
    scr = prealloc_scratch(sp, d)
    scr["sp"] = sp
    # Input tensor -- updated via copy_host_to_device_tensor before each replay
    scr["h"] = rep(torch.zeros(sp, HIDDEN, dtype=torch.bfloat16), d)
    # RoPE tables trimmed to sp rows and placed in L1 (8KB each vs 1MB in DRAM)
    cos_full_host = rb(w["rope_cos_full"])[:sp].to(torch.bfloat16)
    sin_adj_host = rb(w["rope_sin_adj"])[:sp].to(torch.bfloat16)
    w["rope_cos_sp"] = rep(cos_full_host, d, mem=ttnn.L1_MEMORY_CONFIG)
    w["rope_sin_sp"] = rep(sin_adj_host, d, mem=ttnn.L1_MEMORY_CONFIG)
    # Compile pass (warmup) with per-op profiling
    print("Compiling target_fwd...")
    target_fwd(scr["h"], w, sp, d, scr)
    ttnn.synchronize_device(d)
    print("Compilation done. Profiling second pass...")
    target_fwd_profiled(scr["h"], w, sp, d, scr)
    return scr


def capture_target_trace(scr, w, d):
    """Capture trace of target_fwd. Must call prealloc_target_scratch first."""
    sp = scr["sp"]
    print("Capturing trace...")
    tid = ttnn.begin_trace_capture(d, cq_id=0)
    target_fwd(scr["h"], w, sp, d, scr)
    ttnn.end_trace_capture(d, tid, cq_id=0)
    scr["trace_id"] = tid
    print("Trace captured.")


def execute_target_trace(scr, h_host, d):
    """Execute traced target_fwd with new input.

    h_host: ttnn host tensor (from ttnn.from_torch with no device)
    """
    ttnn.copy_host_to_device_tensor(h_host, scr["h"])
    ttnn.execute_trace(d, scr["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(d)
    return scr["logits"]
