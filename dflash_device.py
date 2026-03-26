"""DFlash speculative decoding -- all weights and ops on device, 4-chip TP.

Target: Qwen3-Coder-30B-A3B (48-layer MoE, 128 experts, top-8)
Draft: DFlash (8-layer cross-attention + dense MLP)

Memory budget per chip (~32GB DRAM):
  Attention (sharded): ~432MB
  Experts (32/chip via concat+shard): ~14.4GB
  Router (replicated): ~24MB
  Embedding (sharded dim=0): ~150MB
  LM head (sharded dim=1): ~150MB
  Draft model: ~750MB
  Total: ~16GB/chip -- fits

Sharding:
  Column-parallel (dim=1): Q, K, V, expert gate_all, expert up_all, draft fc1
  Row-parallel (dim=0): O proj, expert down (per-expert), draft fc2 + all_reduce
  Shard dim=0: embedding
  Shard dim=1: LM head
  Replicated: norms, router, constants
"""
import os
import math
import time
import json
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE
HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
ROPE_THETA = 1e7
VOCAB = 151936

TLAYERS = 48
MOE_INTER = 768
NEXPERTS = 128
TOPK = 8

DLAYERS = 8
DINTER = 6144
BSIZE = 16
TLAYER_IDS = [1, 12, 23, 34, 45]
MASK_ID = 151669

N_CHIPS = 4
EXPERTS_PER_CHIP = NEXPERTS // N_CHIPS  # 32
NQH_TP = NQH // N_CHIPS  # 8
NKVH_TP = NKVH // N_CHIPS  # 1

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"

rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
_MESH = None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def open_dev():
    global _MESH
    if N_CHIPS > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        _MESH = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
        return _MESH
    return ttnn.open_device(device_id=0)


def close_dev(d):
    global _MESH
    if _MESH:
        ttnn.close_mesh_device(_MESH)
        _MESH = None
    else:
        ttnn.close_device(d)


def _p(t):
    """Pad to tile alignment."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph, pw = (TILE - h % TILE) % TILE, (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def _mk(d):
    if isinstance(d, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(d)}
    return {}


def rep(t, d, mem=None):
    """Replicate tensor to all chips."""
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
                           **_mk(d))


def shd(t, d, dim, mem=None):
    """Shard tensor across chips along dim."""
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(d, dim=dim))


def ztt(shape, d):
    return rep(torch.zeros(shape, dtype=torch.bfloat16), d)


def rb(t):
    if _MESH:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=0))[:t.shape[0]]
    return ttnn.to_torch(t)


# ---------------------------------------------------------------------------
# Load all weights to device
# ---------------------------------------------------------------------------
def load_weights(d):
    w = {}
    t0 = time.time()

    with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}

    def gt(k):
        with safe_open(kf[k], framework="pt") as f:
            return f.get_tensor(k)

    # Embedding: shard along vocab (dim=0) -- each chip gets VOCAB/4 rows
    print("  embedding (shard dim=0)...")
    w["embed"] = shd(gt("model.embed_tokens.weight").to(torch.bfloat16), d, dim=0)
    print("  lm_head (shard dim=1)...")
    w["lm_head"] = shd(gt("lm_head.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
    w["final_norm"] = gt("model.norm.weight").to(torch.bfloat16)
    print("  embedding + lm_head done")

    # Embedding on host too for token lookup (small transfer per forward call)
    w["embed_h"] = gt("model.embed_tokens.weight").to(torch.bfloat16)

    # RMSNorm constants in L1
    w["sc"] = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)
    w["ms"] = rep(torch.full((TILE, TILE), 1.0 / HIDDEN, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)

    for li in range(TLAYERS):
        p = f"model.layers.{li}"
        lp = f"t.{li}"

        w[f"{lp}.in_w"] = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
        w[f"{lp}.pa_w"] = gt(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)

        # Attention: Q/K/V column-parallel (dim=1), O row-parallel (dim=0)
        qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
        kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
        vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
        ow = gt(f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
        if N_CHIPS > 1:
            w[f"{lp}.qw"] = shd(qw, d, dim=1)
            w[f"{lp}.kw"] = shd(kw, d, dim=1)
            w[f"{lp}.vw"] = shd(vw, d, dim=1)
            w[f"{lp}.ow"] = shd(ow, d, dim=0)
        else:
            w[f"{lp}.qw"] = rep(qw, d)
            w[f"{lp}.kw"] = rep(kw, d)
            w[f"{lp}.vw"] = rep(vw, d)
            w[f"{lp}.ow"] = rep(ow, d)
        w[f"{lp}.qnw"] = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
        w[f"{lp}.knw"] = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

        # Router: replicated (small: 2048x128)
        w[f"{lp}.rw"] = rep(gt(f"{p}.mlp.gate.weight").T.contiguous().to(torch.bfloat16), d)

        # Expert weights: concatenate all 128 experts then shard across 4 chips
        # gate_all: (HIDDEN, NEXPERTS*MOE_INTER) sharded dim=1 -> each chip gets 32*768 cols
        # up_all: same
        # down: per-expert (MOE_INTER, HIDDEN), store individually on owning chip
        gate_parts = []
        up_parts = []
        down_parts = []
        for ei in range(NEXPERTS):
            ep = f"{p}.mlp.experts.{ei}"
            gate_parts.append(gt(f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16))
            up_parts.append(gt(f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16))
            down_parts.append(gt(f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16))

        # Concat gate/up across all experts: (2048, 128*768)
        gate_all = torch.cat(gate_parts, dim=1)  # (2048, 98304)
        up_all = torch.cat(up_parts, dim=1)       # (2048, 98304)
        w[f"{lp}.gate_all"] = shd(gate_all, d, dim=1)  # each chip: (2048, 24576)
        w[f"{lp}.up_all"] = shd(up_all, d, dim=1)

        # Down: concat per-chip group then shard
        # Each chip's down: (32*768, 2048) = (24576, 2048), sharded dim=0
        down_all = torch.cat(down_parts, dim=0)  # (128*768, 2048) = (98304, 2048)
        w[f"{lp}.down_all"] = shd(down_all, d, dim=0)  # each chip: (24576, 2048)

        del gate_parts, up_parts, down_parts, gate_all, up_all, down_all

        if (li + 1) % 4 == 0:
            print(f"  target layer {li+1}/{TLAYERS} ({time.time()-t0:.0f}s)")

    # Draft model
    print("  draft weights...")
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        w["d.fc"] = rep(f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16), d)
        w["d.hn_w"] = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        w["d.fn_w"] = f.get_tensor("norm.weight").to(torch.bfloat16)

        for li in range(DLAYERS):
            dp = f"layers.{li}"
            lp = f"d.{li}"
            w[f"{lp}.in_w"] = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            w[f"{lp}.pa_w"] = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)

            dq = f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
            dk = f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
            dv = f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
            do_ = f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
            if N_CHIPS > 1:
                w[f"{lp}.qw"] = shd(dq, d, dim=1)
                w[f"{lp}.kw"] = shd(dk, d, dim=1)
                w[f"{lp}.vw"] = shd(dv, d, dim=1)
                w[f"{lp}.ow"] = shd(do_, d, dim=0)
            else:
                w[f"{lp}.qw"] = rep(dq, d)
                w[f"{lp}.kw"] = rep(dk, d)
                w[f"{lp}.vw"] = rep(dv, d)
                w[f"{lp}.ow"] = rep(do_, d)
            w[f"{lp}.qnw"] = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            w[f"{lp}.knw"] = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)

            gw = f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16)
            uw = f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16)
            fc1 = torch.cat([gw, uw], dim=1)
            fc2 = f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16)
            if N_CHIPS > 1:
                w[f"{lp}.fc1"] = shd(fc1, d, dim=1)
                w[f"{lp}.fc2"] = shd(fc2, d, dim=0)
            else:
                w[f"{lp}.fc1"] = rep(fc1, d)
                w[f"{lp}.fc2"] = rep(fc2, d)

    print(f"All weights loaded in {time.time()-t0:.0f}s")
    return w


# ---------------------------------------------------------------------------
# On-device ops
# ---------------------------------------------------------------------------
def dev_norm(x, nw, sp, w, d):
    """RMSNorm via TT-Lang kernel."""
    we = nw.unsqueeze(0).expand(sp, -1).contiguous()
    out = ztt((sp, HIDDEN), d)
    rmsnorm_k(x, rep(we, d), w["sc"], w["ms"], out)
    return out


def dev_add(a, b, sp, d):
    """Residual add via TT-Lang kernel."""
    out = ztt((sp, HIDDEN), d)
    residual_add_kernel(a, b, out)
    return out


def dev_qknorm_rope_sdpa(q, k, v, qnw, knw, sl, causal=True):
    """QK norm + RoPE + SDPA. Host reshape (will be TT-Lang kernel later)."""
    qh = rb(q)[:sl].float()
    kh = rb(k)[:sl].float()
    vh = rb(v)[:sl].float()
    nq = qh.shape[1] // HDIM
    nk = kh.shape[1] // HDIM

    qh = qh.view(1, sl, nq, HDIM).transpose(1, 2)
    kh = kh.view(1, sl, nk, HDIM).transpose(1, 2)
    vh = vh.view(1, sl, nk, HDIM).transpose(1, 2)

    def hrms(x, wt):
        return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * wt.float()
    qh = hrms(qh, qnw)
    kh = hrms(kh, knw)

    fr = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    a = torch.outer(torch.arange(sl, dtype=torch.float32), fr).unsqueeze(0).unsqueeze(0)
    c, s = torch.cos(a), torch.sin(a)
    q1, q2 = qh[..., :HDIM//2], qh[..., HDIM//2:]
    qh = torch.cat([q1*c - q2*s, q2*c + q1*s], -1)
    k1, k2 = kh[..., :HDIM//2], kh[..., HDIM//2:]
    kh = torch.cat([k1*c - k2*s, k2*c + k1*s], -1)

    g = nq // nk
    kh = kh.repeat(1, g, 1, 1)
    vh = vh.repeat(1, g, 1, 1)

    sp = ((sl + TILE - 1) // TILE) * TILE
    def pk(x, nh, s):
        o = torch.zeros(1, nh, sp, HDIM, dtype=torch.bfloat16)
        o[:, :, :s] = x.to(torch.bfloat16)
        return o

    dev = q.device()
    at = ttnn.transformer.scaled_dot_product_attention(
        rep(pk(qh, nq, sl), dev), rep(pk(kh, nq, sl), dev),
        rep(pk(vh, nq, sl), dev), is_causal=causal)

    ah = rb(at).view(1, nq, -1, HDIM)[:, :, :sl, :]
    ah = ah.transpose(1, 2).contiguous().view(sl, nq * HDIM).to(torch.bfloat16)
    ap = _p(ah)
    if ap.shape[0] < sp:
        ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))
    return rep(ap, dev)


def dev_moe(h, w, lp, sl, sp, d):
    """MoE with batched expert matmuls on device.

    gate_all/up_all are (HIDDEN, NEXPERTS*MOE_INTER) sharded across chips.
    Each chip computes its 32 experts in one matmul, then we select+weight.
    """
    # Router on device
    router = ttnn.matmul(h, w[f"{lp}.rw"])  # (sp, NEXPERTS) replicated

    # Batched gate/up: one big matmul per chip
    # h: (sp, HIDDEN) replicated, gate_all: (HIDDEN, 32*768) per chip
    gate_out = ttnn.matmul(h, w[f"{lp}.gate_all"])  # (sp, 32*768) per chip
    up_out = ttnn.matmul(h, w[f"{lp}.up_all"])       # (sp, 32*768) per chip

    # SiLU(gate) * up via TT-Lang kernel
    local_expert_dim = EXPERTS_PER_CHIP * MOE_INTER  # 32*768 = 24576
    act = ztt((sp, local_expert_dim), d)
    silu_mul_kernel(gate_out, up_out, act)

    # Down projection: (sp, 32*768) @ (32*768, HIDDEN) per chip -> (sp, HIDDEN)
    down_out = ttnn.matmul(act, w[f"{lp}.down_all"])  # (sp, HIDDEN) per chip

    # All-reduce to sum across chips (each chip computed 32 experts)
    if N_CHIPS > 1:
        down_out = ttnn.all_reduce(down_out)

    # Now down_out has the sum of ALL 128 experts applied to ALL tokens
    # But we need top-k selection! The batched approach computes all experts.
    # For correctness, we need to weight by routing scores.

    # Read back router logits for weighting (tiny: sl*128 values)
    rh = rb(router)[:sl, :NEXPERTS].float()
    scores = torch.softmax(rh, dim=-1)

    # For the batched approach, all experts ran on all tokens.
    # The output is sum over all experts (via down_all matmul).
    # This is WRONG for MoE - we need weighted sum of only top-k experts.
    #
    # The batched matmul gives: out[tok] = sum_{all experts on chip} silu(tok@gate_e) * (tok@up_e) @ down_e
    # But MoE wants: out[tok] = sum_{top-k experts} weight_e * silu(tok@gate_e) * (tok@up_e) @ down_e
    #
    # To fix this properly we need per-expert intermediate results.
    # For now: readback the activated tensor, apply routing weights, push back.

    act_h = rb(act)[:sl, :local_expert_dim].to(torch.bfloat16)  # (sl, 32*768) per chip
    # But with sharding, rb gives us chip 0's data. We need all chips.
    # Actually with ConcatMeshToTensor dim=0, we get (N_CHIPS*sl, local_expert_dim)
    # That's not right either for the routing approach.

    # SIMPLER APPROACH: do per-expert matmuls for the active experts only.
    # Read hidden, run 8 experts on device, weight and sum.
    hh = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
    tw, ti = torch.topk(scores, TOPK, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)

    e2t = {}
    for tok in range(sl):
        for k in range(TOPK):
            ei = ti[tok, k].item()
            wt = tw[tok, k].item()
            if ei not in e2t:
                e2t[ei] = []
            e2t[ei].append((tok, wt))

    out_h = torch.zeros(sl, HIDDEN, dtype=torch.bfloat16)
    for ei, tinfo in e2t.items():
        toks = [t[0] for t in tinfo]
        n = len(toks)
        np_ = ((n + TILE - 1) // TILE) * TILE

        inp = F.pad(hh[toks], (0, 0, 0, np_ - n))
        inp_tt = rep(inp, d)

        # Extract this expert's weight columns from the concatenated gate_all/up_all
        # Expert ei's gate cols: [ei*MOE_INTER : (ei+1)*MOE_INTER] in the full concat
        # But gate_all is sharded across chips. Expert ei lives on chip ei//32.
        # We need to slice within that chip's portion.
        #
        # This is hard to do with sharded tensors. Use host expert weights instead.
        gw = w[f"_eh.{lp}.e{ei}.gw"]
        uw = w[f"_eh.{lp}.e{ei}.uw"]
        dw_h = w[f"_eh.{lp}.e{ei}.dw"]

        g = ttnn.matmul(inp_tt, rep(gw, d))
        u = ttnn.matmul(inp_tt, rep(uw, d))
        ac = ztt((np_, MOE_INTER), d)
        silu_mul_kernel(g, u, ac)
        eo = ttnn.matmul(ac, rep(dw_h, d))
        eoh = rb(eo)[:n, :HIDDEN].to(torch.bfloat16)

        for i, (tok, wt) in enumerate(tinfo):
            out_h[tok] += eoh[i] * wt

    op = _p(out_h)
    if op.shape[0] < sp:
        op = F.pad(op, (0, 0, 0, sp - op.shape[0]))
    return rep(op, d)


# Wait - the approach above stores expert weights on host AND device (duplicated).
# The gate_all/up_all/down_all on device are wasted if we do per-expert.
# Let me just do per-expert with the sharded weights on device.
# I need to rethink this.

# The core problem: MoE routing is dynamic. We can't easily index into sharded
# tensors across chips. Two clean approaches:
#
# A) All experts always compute (wasteful but simple, all on device)
#    - Needs per-expert weighting BEFORE down_proj, not after
#    - Requires reshaping intermediate to (sl, NEXPERTS_PER_CHIP, MOE_INTER)
#
# B) Per-expert matmuls with replicated expert weights (current approach)
#    - Expert weights replicated = 4x memory = too much
#
# C) Per-expert matmuls with expert weights on only the owning chip
#    - Need to dispatch tokens to correct chip = complex
#
# Let's go with A: run all experts, weight before down_proj.
# This means: for each token, all 128 experts run, but only top-8 weights are nonzero.
# The cost is 128/8 = 16x more compute in gate/up, but it's all on device with no routing.


def dev_moe_all_experts(h, w, lp, sl, sp, d):
    """MoE: run ALL experts, weight by routing scores, all on device.

    gate_all: (HIDDEN, NEXPERTS*MOE_INTER) sharded dim=1 -> each chip: (HIDDEN, 32*768)
    For each token: compute all experts, multiply by routing weight, then down_proj + sum.

    The trick: insert routing weights between activation and down_proj.
    activated: (sp, 32*768) per chip = (sp, 32, 768) logically
    Multiply each expert's 768-dim output by its routing weight.
    Then down_all: (32*768, HIDDEN) per chip sums all expert contributions.
    """
    # Router
    router = ttnn.matmul(h, w[f"{lp}.rw"])  # (sp, NEXPERTS) replicated
    router_scores = ttnn.softmax(router, dim=-1)  # (sp, NEXPERTS) on device

    # Batched gate/up on device
    gate_out = ttnn.matmul(h, w[f"{lp}.gate_all"])  # (sp, 32*768) per chip
    up_out = ttnn.matmul(h, w[f"{lp}.up_all"])

    # SiLU * mul
    local_dim = EXPERTS_PER_CHIP * MOE_INTER
    act = ztt((sp, local_dim), d)
    silu_mul_kernel(gate_out, up_out, act)

    # Apply routing weights: need to scale each expert's MOE_INTER columns
    # router_scores: (sp, 128) replicated
    # Each chip owns experts [chip_id*32 : (chip_id+1)*32]
    # We need to slice the router scores for this chip's experts and
    # broadcast-multiply into the activated tensor.
    #
    # This requires knowing chip_id at runtime, which is tricky with mesh tensors.
    # Alternative: readback scores, build weight matrix, push back.
    rh = rb(router_scores)[:sl, :NEXPERTS].float()
    # For top-k: zero out non-top-k entries
    tw, ti = torch.topk(rh, TOPK, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)
    routing_mask = torch.zeros_like(rh)
    routing_mask.scatter_(1, ti, tw)

    # Build per-chip weight vectors: (sl, 32) per chip, then expand to (sl, 32*768)
    # Chip c gets routing_mask[:, c*32:(c+1)*32]
    chip_weights = []
    for c in range(N_CHIPS):
        cw = routing_mask[:, c * EXPERTS_PER_CHIP:(c + 1) * EXPERTS_PER_CHIP]  # (sl, 32)
        # Expand each weight to cover MOE_INTER columns: (sl, 32) -> (sl, 32*768)
        cw_expanded = cw.unsqueeze(-1).expand(-1, -1, MOE_INTER).reshape(sl, local_dim)
        chip_weights.append(cw_expanded.to(torch.bfloat16))

    # Stack and shard: (sl, local_dim) per chip
    weight_tensor = torch.cat(chip_weights, dim=1)  # (sl, 128*768)
    weight_padded = _p(weight_tensor)
    if weight_padded.shape[0] < sp:
        weight_padded = F.pad(weight_padded, (0, 0, 0, sp - weight_padded.shape[0]))
    weight_tt = shd(weight_padded, d, dim=1)

    # Elementwise multiply: act * weights (both (sp, 32*768) per chip)
    weighted_act = ttnn.multiply(act, weight_tt)

    # Down projection: (sp, 32*768) @ (32*768, HIDDEN) per chip
    down_out = ttnn.matmul(weighted_act, w[f"{lp}.down_all"])

    # All-reduce sums contributions from all 4 chips
    if N_CHIPS > 1:
        down_out = ttnn.all_reduce(down_out)

    return down_out


# ---------------------------------------------------------------------------
# Target forward
# ---------------------------------------------------------------------------
def target_fwd(h, w, sl, sp, d, save_hs=False):
    hs = {}
    for li in range(TLAYERS):
        lp = f"t.{li}"
        if save_hs and li in TLAYER_IDS:
            hs[li] = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)

        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        q = ttnn.matmul(n, w[f"{lp}.qw"])
        k = ttnn.matmul(n, w[f"{lp}.kw"])
        v = ttnn.matmul(n, w[f"{lp}.vw"])

        a = dev_qknorm_rope_sdpa(q, k, v, w[f"{lp}.qnw"], w[f"{lp}.knw"], sl)
        o = ttnn.matmul(a, w[f"{lp}.ow"])
        if N_CHIPS > 1:
            o = ttnn.all_reduce(o)
        h = dev_add(h, o, sp, d)

        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        moe = dev_moe_all_experts(nm, w, lp, sl, sp, d)
        h = dev_add(h, moe, sp, d)

        if (li + 1) % 8 == 0:
            print(f"    layer {li+1}/{TLAYERS}")

    fn = dev_norm(h, w["final_norm"], sp, w, d)
    logits = ttnn.matmul(fn, w["lm_head"])
    return logits, hs


# ---------------------------------------------------------------------------
# Draft forward
# ---------------------------------------------------------------------------
def draft_fwd(noise, ctx, w, sl, ctx_len, sp, d):
    h = noise
    for li in range(DLAYERS):
        lp = f"d.{li}"
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)

        q = ttnn.matmul(n, w[f"{lp}.qw"])
        # Cross-attn K/V from [ctx, draft]
        nh = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        ch = rb(ctx)[:ctx_len, :HIDDEN].to(torch.bfloat16)
        kv_in = torch.cat([ch, nh], dim=0)
        kv_len = kv_in.shape[0]
        kv_sp = ((kv_len + TILE - 1) // TILE) * TILE
        kv_p = _p(kv_in)
        if kv_p.shape[0] < kv_sp:
            kv_p = F.pad(kv_p, (0, 0, 0, kv_sp - kv_p.shape[0]))
        kv = rep(kv_p, d)

        k = ttnn.matmul(kv, w[f"{lp}.kw"])
        v = ttnn.matmul(kv, w[f"{lp}.vw"])

        # QK norm + RoPE + SDPA for cross-attention
        nq = NQH_TP if N_CHIPS > 1 else NQH
        nk = NKVH_TP if N_CHIPS > 1 else NKVH
        qh = rb(q)[:sl].float().view(1, sl, nq, HDIM).transpose(1, 2)
        kh = rb(k)[:kv_len].float().view(1, kv_len, nk, HDIM).transpose(1, 2)
        vh = rb(v)[:kv_len].float().view(1, kv_len, nk, HDIM).transpose(1, 2)

        def hrms(x, wt):
            return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * wt.float()
        qh = hrms(qh, w[f"{lp}.qnw"])
        kh = hrms(kh, w[f"{lp}.knw"])

        fr = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
        qa = torch.outer(torch.arange(sl, dtype=torch.float32), fr).unsqueeze(0).unsqueeze(0)
        ka = torch.outer(torch.arange(kv_len, dtype=torch.float32), fr).unsqueeze(0).unsqueeze(0)
        q1, q2 = qh[..., :HDIM//2], qh[..., HDIM//2:]
        qh = torch.cat([q1*torch.cos(qa) - q2*torch.sin(qa), q2*torch.cos(qa) + q1*torch.sin(qa)], -1)
        k1, k2 = kh[..., :HDIM//2], kh[..., HDIM//2:]
        kh = torch.cat([k1*torch.cos(ka) - k2*torch.sin(ka), k2*torch.cos(ka) + k1*torch.sin(ka)], -1)

        g = nq // nk
        kh = kh.repeat(1, g, 1, 1)
        vh = vh.repeat(1, g, 1, 1)

        def pk4(x, nh, s, total):
            o = torch.zeros(1, nh, total, HDIM, dtype=torch.bfloat16)
            o[:, :, :s] = x.to(torch.bfloat16)
            return o

        at = ttnn.transformer.scaled_dot_product_attention(
            rep(pk4(qh, nq, sl, sp), d),
            rep(pk4(kh, nq, kv_len, kv_sp), d),
            rep(pk4(vh, nq, kv_len, kv_sp), d),
            is_causal=False)

        ah = rb(at).view(1, nq, -1, HDIM)[:, :, :sl, :]
        ah = ah.transpose(1, 2).contiguous().view(sl, nq * HDIM).to(torch.bfloat16)
        ap = _p(ah)
        if ap.shape[0] < sp:
            ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))
        att = rep(ap, d)

        o = ttnn.matmul(att, w[f"{lp}.ow"])
        if N_CHIPS > 1:
            o = ttnn.all_reduce(o)
        h = dev_add(h, o, sp, d)

        # Dense MLP
        n2 = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        fc1 = ttnn.matmul(n2, w[f"{lp}.fc1"])
        # Split gate/up, SiLU*mul
        fc1h = rb(fc1)[:sl].to(torch.bfloat16)
        inter = DINTER // N_CHIPS if N_CHIPS > 1 else DINTER
        gh, uh = fc1h[:, :inter], fc1h[:, inter:]
        activated = (torch.nn.functional.silu(gh.float()) * uh.float()).to(torch.bfloat16)
        ap = _p(activated)
        if ap.shape[0] < sp:
            ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))

        fc2 = ttnn.matmul(rep(ap, d), w[f"{lp}.fc2"])
        if N_CHIPS > 1:
            fc2 = ttnn.all_reduce(fc2)
        h = dev_add(h, fc2, sp, d)

    h = dev_norm(h, w["d.fn_w"], sp, w, d)
    return h


# ---------------------------------------------------------------------------
# Speculative decode
# ---------------------------------------------------------------------------
def spec_generate(ids, w, d, max_new=64):
    pl = ids.shape[0]
    sp = ((pl + TILE - 1) // TILE) * TILE

    out = torch.full((pl + max_new + BSIZE,), MASK_ID, dtype=torch.long)
    out[:pl] = ids

    embed_h = w["embed_h"]

    print("Prefill...")
    t0 = time.time()
    h = _p(embed_h[ids])
    if h.shape[0] < sp:
        h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
    h_tt = rep(h, d)

    logits, ths = target_fwd(h_tt, w, pl, sp, d, save_hs=True)
    pft = time.time() - t0
    print(f"Prefill: {pft:.1f}s ({pl} tokens)")

    lh = rb(logits)[:pl, :VOCAB].float()
    out[pl] = torch.argmax(lh[-1]).item()

    # Project target context
    tf = torch.cat([ths[lid] for lid in TLAYER_IDS], dim=-1)
    tf_sp = ((pl + TILE - 1) // TILE) * TILE
    tf_p = _p(tf)
    if tf_p.shape[0] < tf_sp:
        tf_p = F.pad(tf_p, (0, 0, 0, tf_sp - tf_p.shape[0]))
    ctx = ttnn.matmul(rep(tf_p, d), w["d.fc"])
    ctx = dev_norm(ctx, w["d.hn_w"], tf_sp, w, d)

    start = pl
    gen = 0
    ahist = []

    print("Decoding...")
    while start < pl + max_new:
        ts = time.time()
        bids = out[start:start + BSIZE].clone()
        bsp = ((BSIZE + TILE - 1) // TILE) * TILE
        noise = _p(embed_h[bids])
        if noise.shape[0] < bsp:
            noise = F.pad(noise, (0, 0, 0, bsp - noise.shape[0]))

        dout = draft_fwd(rep(noise, d), ctx, w, BSIZE, pl, bsp, d)
        dl = ttnn.matmul(dout, w["lm_head"])
        dlh = rb(dl)[:BSIZE, :VOCAB].float()
        bids[1:] = torch.argmax(dlh[:-1], dim=-1)

        # Verify
        vh = _p(embed_h[bids])
        if vh.shape[0] < bsp:
            vh = F.pad(vh, (0, 0, 0, bsp - vh.shape[0]))
        vl, vhs = target_fwd(rep(vh, d), w, BSIZE, bsp, d, save_hs=True)
        vlh = rb(vl)[:BSIZE, :VOCAB].float()
        post = torch.argmax(vlh, dim=-1)

        acc = (bids[1:] == post[:-1]).to(torch.int64).cumprod(0).sum().item()
        out[start:start+acc+1] = bids[:acc+1]
        out[start+acc+1] = post[acc]
        start += acc + 1
        gen += acc + 1
        ahist.append(acc + 1)

        if vhs:
            vf = torch.cat([vhs[lid] for lid in TLAYER_IDS], dim=-1)
            vf_p = _p(vf)
            if vf_p.shape[0] < bsp:
                vf_p = F.pad(vf_p, (0, 0, 0, bsp - vf_p.shape[0]))
            ctx = ttnn.matmul(rep(vf_p, d), w["d.fc"])
            ctx = dev_norm(ctx, w["d.hn_w"], bsp, w, d)

        el = time.time() - ts
        avg = sum(ahist) / len(ahist)
        print(f"  step {len(ahist)}: acc={acc+1}/{BSIZE} avg={avg:.1f} {el:.1f}s gen={gen}")

        if out[start - 1].item() in (151643, 151645):
            break

    out = out[:start]
    out = out[out != MASK_ID]
    return out


def main():
    print("=" * 60)
    print(f"DFlash on Tenstorrent ({N_CHIPS} chips, all on device)")
    print("=" * 60)

    d = open_dev()
    try:
        w = load_weights(d)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TARGET_DIR)
            prompt = "Write a Python function that computes fibonacci numbers."
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
        except Exception as e:
            print(f"Tokenizer: {e}")
            ids = torch.tensor([151643, 872, 13, 5765, 264, 13325])

        print(f"Prompt: {ids.shape[0]} tokens")
        out = spec_generate(ids, w, d, max_new=64)
        try:
            print(f"\n--- Output ---\n{tok.decode(out, skip_special_tokens=True)}")
        except:
            print(f"Output IDs: {out.tolist()}")
    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
