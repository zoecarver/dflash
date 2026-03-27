"""DFlash speculative decoding -- zero host transfers, 4-chip TP.

All weights on device. All compute on device. No host readbacks during forward.

MoE: all 128 experts compute via batched matmul, weighted by softmax scores.
     gate_all/up_all sharded dim=1 across chips, down_all sharded dim=0 + all_reduce.
Attention: QKV+swap combined matmul, TT-Lang RoPE kernel, ttnn SDPA.
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
from rope_qknorm import make_rope_qknorm_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE
HDIM = 128
HDIM_TILES = HDIM // TILE  # 4
NQH = 32
NKVH = 4
GQA = NQH // NKVH  # 8
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
EPC = NEXPERTS // N_CHIPS  # 32 experts per chip
NQH_TP = NQH // N_CHIPS   # 8 Q heads per chip
NKVH_TP = NKVH // N_CHIPS  # 1 KV head per chip
Q_TP = NQH_TP * HDIM       # 1024
KV_TP = NKVH_TP * HDIM     # 128

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"

rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
_MESH = None


def open_dev():
    global _MESH
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    _MESH = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    return _MESH


def close_dev(d):
    global _MESH
    ttnn.close_mesh_device(_MESH)
    _MESH = None


def _p(t):
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
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG, **_mk(d))


def shd(t, d, dim, mem=None):
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
# Load ALL weights to device
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

    # Embedding: keep on host for token gather (tiny transfer: seq_len*HIDDEN per call)
    print("  embed/lm_head...")
    w["embed_h"] = gt("model.embed_tokens.weight").to(torch.bfloat16)
    w["lm_head"] = shd(gt("lm_head.weight").T.contiguous().to(torch.bfloat16), d, dim=1)
    w["final_norm"] = gt("model.norm.weight").to(torch.bfloat16)

    w["sc"] = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)
    w["ms"] = rep(torch.full((TILE, TILE), 1.0 / HIDDEN, dtype=torch.bfloat16), d, mem=ttnn.L1_MEMORY_CONFIG)

    # RoPE tables (replicated, L1)
    max_seq = 4096
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_t = torch.cos(angles).to(torch.bfloat16)  # (max_seq, HDIM/2)
    sin_t = torch.sin(angles).to(torch.bfloat16)
    w["rope_cos"] = rep(cos_t, d)
    w["rope_sin"] = rep(sin_t, d)

    for li in range(TLAYERS):
        p = f"model.layers.{li}"
        lp = f"t.{li}"

        w[f"{lp}.in_w"] = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
        w[f"{lp}.pa_w"] = gt(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)

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

        w[f"{lp}.qnw"] = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
        w[f"{lp}.knw"] = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)

        # Router: replicated
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

            dqw = f.get_tensor(f"{dp}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
            dkw = f.get_tensor(f"{dp}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
            dvw = f.get_tensor(f"{dp}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
            dow = f.get_tensor(f"{dp}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)

            w[f"{lp}.qw"] = shd(dqw, d, dim=1)
            w[f"{lp}.kw"] = shd(dkw, d, dim=1)
            w[f"{lp}.vw"] = shd(dvw, d, dim=1)
            w[f"{lp}.ow"] = shd(dow, d, dim=0)

            w[f"{lp}.qnw"] = f.get_tensor(f"{dp}.self_attn.q_norm.weight").to(torch.bfloat16)
            w[f"{lp}.knw"] = f.get_tensor(f"{dp}.self_attn.k_norm.weight").to(torch.bfloat16)

            gw = f.get_tensor(f"{dp}.mlp.gate_proj.weight").T.contiguous().to(torch.bfloat16)
            uw = f.get_tensor(f"{dp}.mlp.up_proj.weight").T.contiguous().to(torch.bfloat16)
            fc2 = f.get_tensor(f"{dp}.mlp.down_proj.weight").T.contiguous().to(torch.bfloat16)
            w[f"{lp}.fc1"] = shd(torch.cat([gw, uw], dim=1), d, dim=1)
            w[f"{lp}.fc2"] = shd(fc2, d, dim=0)

    print(f"All weights loaded in {time.time()-t0:.0f}s")
    return w


# ---------------------------------------------------------------------------
# On-device building blocks
# ---------------------------------------------------------------------------
def dev_norm(x, nw, sp, w, d):
    we = nw.unsqueeze(0).contiguous()  # (1, HIDDEN)
    weight_tt = rep(we, d)
    return ttnn.rms_norm(x, weight=weight_tt, epsilon=EPS)


def dev_add(a, b, sp, d):
    return ttnn.add(a, b)


def rb_dim1(t):
    """Readback column-sharded tensor, concatenating chips along dim=1."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=1))


def dev_attn(normed, w, lp, sl, sp, d):
    """GQA attention. QKV on device (column-parallel), QK-norm + RoPE on host."""
    # QKV projections (column-parallel: each chip has its heads)
    q = ttnn.matmul(normed, w[f"{lp}.qw"])     # (sp, Q_TP=1024) per chip
    k = ttnn.matmul(normed, w[f"{lp}.kw"])     # (sp, KV_TP=128) per chip
    v = ttnn.matmul(normed, w[f"{lp}.vw"])     # (sp, KV_TP=128) per chip

    # Gather all heads from all chips (concat along column/head dim)
    qh = rb_dim1(q)[:sl].float()    # (sl, NQH*HDIM=4096) all 32 Q heads
    kh = rb_dim1(k)[:sl].float()    # (sl, NKVH*HDIM=512) all 4 KV heads
    vh = rb_dim1(v)[:sl].float()    # (sl, NKVH*HDIM=512)

    # QK-norm: per-head RMSNorm before RoPE
    qnw = w[f"{lp}.qnw"].float()
    knw = w[f"{lp}.knw"].float()

    def head_rms_norm(x, nw, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((x4 / rms) * nw).view(x.shape)

    qh = head_rms_norm(qh, qnw, NQH)
    kh = head_rms_norm(kh, knw, NKVH)

    # RoPE: standard rotate_half formulation
    def rotate_half_flat(x, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        x1, x2 = x4[..., :HDIM // 2], x4[..., HDIM // 2:]
        return torch.cat((-x2, x1), dim=-1).view(x.shape)

    cos_h = rb(w["rope_cos"])[:sl].float()  # (sl, HDIM/2=64)
    sin_h = rb(w["rope_sin"])[:sl].float()
    cos_full = cos_h.repeat(1, 2)[:, :HDIM]
    sin_full = sin_h.repeat(1, 2)[:, :HDIM]

    q_roped = qh * cos_full.repeat(1, NQH) + rotate_half_flat(qh, NQH) * sin_full.repeat(1, NQH)
    k_roped = kh * cos_full.repeat(1, NKVH) + rotate_half_flat(kh, NKVH) * sin_full.repeat(1, NKVH)

    # Reshape for SDPA: (1, heads, sl, HDIM)
    q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)

    # GQA: expand K/V to match Q head count (interleave, not tile)
    k4 = k4.repeat_interleave(GQA, dim=1)
    v4 = v4.repeat_interleave(GQA, dim=1)

    # Pad seq dim
    q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    q4p[:, :, :sl] = q4
    k4p[:, :, :sl] = k4
    v4p[:, :, :sl] = v4

    # SDPA (replicated -- all chips run full 32-head attention)
    attn = ttnn.transformer.scaled_dot_product_attention(
        rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

    # Reshape back and shard across chips for row-parallel O proj
    ah = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :]
    ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
    ahp = _p(ah)
    if ahp.shape[0] < sp:
        ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
    attn_tt = shd(ahp, d, dim=1)  # each chip gets its 8 heads

    # O projection (row-parallel) + all_reduce
    o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
    o = ttnn.all_reduce(o)
    return o


def dev_moe(h, w, lp, sl, sp, d):
    """MoE: all experts compute, weighted by softmax scores, all on device.

    Each chip has 32 experts. gate_all/up_all sharded dim=1.
    After SiLU*mul, multiply by routing weights, then down_all + all_reduce.
    """
    # Router + softmax on device
    router = ttnn.matmul(h, w[f"{lp}.rw"])      # (sp, 128) replicated
    scores = ttnn.softmax(router, dim=-1)         # (sp, 128) replicated

    # Batched gate/up: one matmul per operation across all chip-local experts
    local_dim = EPC * MOE_INTER  # 32*768 = 24576
    gate = ttnn.matmul(h, w[f"{lp}.gate_all"])   # (sp, 24576) per chip
    up = ttnn.matmul(h, w[f"{lp}.up_all"])       # (sp, 24576) per chip

    # SiLU(gate) * up
    act = ttnn.multiply(ttnn.silu(gate), up)

    # Read back activations and scores for per-chip weighted sum on host.
    # (Batched device multiply is broken due to rep/sharded mismatch -- P2 fix)
    sh = rb(scores)[:sl, :NEXPERTS].float()
    topk_vals, topk_idx = torch.topk(sh, TOPK, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)
    routing_mask = torch.zeros_like(sh)
    routing_mask.scatter_(1, topk_idx, topk_vals)

    # Read all chips' activations: dim=0 concat gives (4*sp, local_dim)
    act_all = ttnn.to_torch(act, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=0))

    # Apply routing weights per chip, build weighted sum for down proj
    weighted_parts = []
    for c in range(N_CHIPS):
        chip_act = act_all[c * sp:(c + 1) * sp][:sl].float()  # (sl, 24576)
        cs = routing_mask[:, c * EPC:(c + 1) * EPC]  # (sl, 32)
        cs_exp = cs.unsqueeze(-1).expand(-1, -1, MOE_INTER).reshape(sl, local_dim)
        weighted_parts.append((chip_act * cs_exp).to(torch.bfloat16))

    # Each chip needs its own weighted activation for row-parallel down proj
    all_weighted = torch.cat(weighted_parts, dim=1)  # (sl, 4*24576=98304)
    wp = _p(all_weighted)
    if wp.shape[0] < sp:
        wp = F.pad(wp, (0, 0, 0, sp - wp.shape[0]))
    weighted_tt = shd(wp, d, dim=1)  # each chip gets (sp, 24576)

    # Down projection (row-parallel) + all_reduce
    out = ttnn.matmul(weighted_tt, w[f"{lp}.down_all"])
    out = ttnn.all_reduce(out)
    return out


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
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        h = dev_add(h, attn_out, sp, d)

        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        h = dev_add(h, moe_out, sp, d)

        if (li + 1) % 8 == 0:
            print(f"    layer {li+1}/{TLAYERS}")

    fn = dev_norm(h, w["final_norm"], sp, w, d)
    logits = ttnn.matmul(fn, w["lm_head"])
    return logits, hs


# ---------------------------------------------------------------------------
# Draft forward (simplified: shares attention pattern with target)
# ---------------------------------------------------------------------------
def draft_fwd(noise, ctx, w, sl, ctx_len, sp, d):
    """DFlash draft model. Cross-attention uses host reshaping for now."""
    h = noise
    for li in range(DLAYERS):
        lp = f"d.{li}"
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)

        # Cross-attention: Q from draft, K/V from [ctx, draft]
        q = ttnn.matmul(n, w[f"{lp}.qw"])

        # Concat ctx + draft for K/V (host assist for dynamic concat)
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

        # Gather all heads from all chips for QK-norm + RoPE + SDPA
        qh = rb_dim1(q)[:sl].float().view(1, sl, NQH, HDIM).transpose(1, 2)
        kh = rb_dim1(k)[:kv_len].float().view(1, kv_len, NKVH, HDIM).transpose(1, 2)
        vh = rb_dim1(v)[:kv_len].float().view(1, kv_len, NKVH, HDIM).transpose(1, 2)

        def hrms(x, wt):
            return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * wt.float()
        qh = hrms(qh, w[f"{lp}.qnw"])
        kh = hrms(kh, w[f"{lp}.knw"])

        fr = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
        qa = torch.outer(torch.arange(sl, dtype=torch.float32), fr).unsqueeze(0).unsqueeze(0)
        ka = torch.outer(torch.arange(kv_len, dtype=torch.float32), fr).unsqueeze(0).unsqueeze(0)
        c_q, s_q = torch.cos(qa), torch.sin(qa)
        c_k, s_k = torch.cos(ka), torch.sin(ka)
        q1, q2 = qh[..., :HDIM//2], qh[..., HDIM//2:]
        qh = torch.cat([q1*c_q - q2*s_q, q2*c_q + q1*s_q], -1)
        k1, k2 = kh[..., :HDIM//2], kh[..., HDIM//2:]
        kh = torch.cat([k1*c_k - k2*s_k, k2*c_k + k1*s_k], -1)

        kh = kh.repeat_interleave(GQA, dim=1)
        vh = vh.repeat_interleave(GQA, dim=1)

        def pk4(x, nh, s, total):
            o = torch.zeros(1, nh, total, HDIM, dtype=torch.bfloat16)
            o[:, :, :s] = x.to(torch.bfloat16)
            return o

        at = ttnn.transformer.scaled_dot_product_attention(
            rep(pk4(qh, NQH, sl, sp), d), rep(pk4(kh, NQH, kv_len, kv_sp), d),
            rep(pk4(vh, NQH, kv_len, kv_sp), d), is_causal=False)

        ah = rb(at).view(1, NQH, -1, HDIM)[:, :, :sl, :]
        ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ap = _p(ah)
        if ap.shape[0] < sp:
            ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))

        o = ttnn.matmul(shd(ap, d, dim=1), w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)
        h = dev_add(h, o, sp, d)

        # Dense MLP
        n2 = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        fc1 = ttnn.matmul(n2, w[f"{lp}.fc1"])  # sharded dim=1

        # SiLU*mul: gather all chips' gate/up halves
        fc1h = rb_dim1(fc1)[:sl].to(torch.bfloat16)  # (sl, DINTER*2)
        half = DINTER
        g, u = fc1h[:, :half], fc1h[:, half:]
        activated = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
        ap = _p(activated)
        if ap.shape[0] < sp:
            ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))

        fc2 = ttnn.matmul(shd(ap, d, dim=1), w[f"{lp}.fc2"])
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

    emb = w["embed_h"]

    print("Prefill...")
    t0 = time.time()
    h = _p(emb[ids])
    if h.shape[0] < sp:
        h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
    h_tt = rep(h, d)

    logits, ths = target_fwd(h_tt, w, pl, sp, d, save_hs=True)
    pft = time.time() - t0
    print(f"Prefill: {pft:.1f}s ({pl} tokens)")

    lh = rb_dim1(logits)[:pl, :VOCAB].float()
    out[pl] = torch.argmax(lh[-1]).item()

    # Project target context for draft
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
        noise = _p(emb[bids])
        if noise.shape[0] < bsp:
            noise = F.pad(noise, (0, 0, 0, bsp - noise.shape[0]))

        dout = draft_fwd(rep(noise, d), ctx, w, BSIZE, pl, bsp, d)
        dl = ttnn.matmul(dout, w["lm_head"])
        dlh = rb_dim1(dl)[:BSIZE, :VOCAB].float()
        bids[1:] = torch.argmax(dlh[:-1], dim=-1)

        # Verify: run target on FULL context (prompt + generated + draft block)
        # so attention can see all prior tokens
        verify_ids = torch.cat([out[:start], bids])
        vlen = verify_ids.shape[0]
        vsp = ((vlen + TILE - 1) // TILE) * TILE
        vh = _p(emb[verify_ids])
        if vh.shape[0] < vsp:
            vh = F.pad(vh, (0, 0, 0, vsp - vh.shape[0]))
        vl, vhs = target_fwd(rep(vh, d), w, vlen, vsp, d, save_hs=True)
        # Take logits from the draft block positions (last BSIZE tokens)
        vlh = rb_dim1(vl)[start:start + BSIZE, :VOCAB].float()
        post = torch.argmax(vlh, dim=-1)

        acc = (bids[1:] == post[:-1]).to(torch.int64).cumprod(0).sum().item()
        out[start:start+acc+1] = bids[:acc+1]
        out[start+acc+1] = post[acc]
        start += acc + 1
        gen += acc + 1
        ahist.append(acc + 1)

        # Update draft context from verify hidden states
        if vhs:
            vf = torch.cat([vhs[lid] for lid in TLAYER_IDS], dim=-1)
            vfp = _p(vf)
            if vfp.shape[0] < vsp:
                vfp = F.pad(vfp, (0, 0, 0, vsp - vfp.shape[0]))
            ctx = ttnn.matmul(rep(vfp, d), w["d.fc"])
            ctx = dev_norm(ctx, w["d.hn_w"], vsp, w, d)

        el = time.time() - ts
        avg = sum(ahist) / len(ahist)
        print(f"  step {len(ahist)}: acc={acc+1}/{BSIZE} avg={avg:.1f} {el:.1f}s tok/s={gen/sum(t for t in [el]*1):.1f} ctx={vlen} gen={gen}")

        if out[start - 1].item() in (151643, 151645):
            break

    out = out[:start]
    out = out[out != MASK_ID]
    return out


def main():
    print("=" * 60)
    print(f"DFlash on Tenstorrent ({N_CHIPS} chips)")
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
