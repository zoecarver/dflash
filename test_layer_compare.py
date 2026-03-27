"""Compare one transformer layer: dflash_device vs PyTorch reference.

Loads layer 0 weights, runs identical input through both paths, compares.
Isolates bugs to attention vs MoE vs norm vs residual.
"""
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

TLAYERS = 48
MOE_INTER = 768
NEXPERTS = 128
TOPK = 8

N_CHIPS = 4
EPC = NEXPERTS // N_CHIPS
NQH_TP = NQH // N_CHIPS
NKVH_TP = NKVH // N_CHIPS
Q_TP = NQH_TP * HDIM
KV_TP = NKVH_TP * HDIM

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)


def _p(t):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w_ = t.shape[-2], t.shape[-1]
    ph, pw = (TILE - h % TILE) % TILE, (TILE - w_ % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def rep(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(d))

def shd(t, d, dim):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(d, dim=dim))

def rb(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:t.shape[0]]

def rb1(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))


# ---- PyTorch reference ----
def rms_norm_ref(x, w):
    xf = x.float()
    return ((xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * w.float()).to(torch.bfloat16)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def ref_attn(hidden, qw, kw, vw, ow, qnw, knw, cos, sin, sl):
    """Pure PyTorch attention."""
    h = hidden[:sl].float()
    q = (h @ qw.float()).view(1, sl, NQH, HDIM).transpose(1, 2)
    k = (h @ kw.float()).view(1, sl, NKVH, HDIM).transpose(1, 2)
    v = (h @ vw.float()).view(1, sl, NKVH, HDIM).transpose(1, 2)

    q = rms_norm_ref(q.to(torch.bfloat16), qnw).float()
    k = rms_norm_ref(k.to(torch.bfloat16), knw).float()

    cos_full = cos[:sl].float().repeat(1, 2)[:, :HDIM].unsqueeze(0).unsqueeze(0)
    sin_full = sin[:sl].float().repeat(1, 2)[:, :HDIM].unsqueeze(0).unsqueeze(0)
    q = q * cos_full + rotate_half(q) * sin_full
    k = k * cos_full + rotate_half(k) * sin_full

    k = k.repeat(1, GQA, 1, 1)
    v = v.repeat(1, GQA, 1, 1)

    scale = 1.0 / math.sqrt(HDIM)
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(sl, NQH * HDIM)
    return (out @ ow.float()).to(torch.bfloat16)

def ref_moe(hidden, router_w, gate_ws, up_ws, down_ws, sl):
    """Pure PyTorch MoE."""
    h = hidden[:sl].float()
    scores = torch.softmax(h @ router_w.float(), dim=-1)
    topk_vals, topk_idx = torch.topk(scores, TOPK, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)

    out = torch.zeros(sl, HIDDEN, dtype=torch.float32)
    for tok in range(sl):
        for ki in range(TOPK):
            ei = topk_idx[tok, ki].item()
            w = topk_vals[tok, ki].item()
            inp = h[tok:tok+1]
            g = inp @ gate_ws[ei].float()
            u = inp @ up_ws[ei].float()
            out[tok] += (torch.nn.functional.silu(g) * u @ down_ws[ei].float()).squeeze(0) * w
    return out.to(torch.bfloat16)


# ---- Device path (from dflash_device.py) ----
def dev_norm(x, nw, sp, sc, ms, d):
    we = nw.unsqueeze(0).expand(sp, -1).contiguous()
    out = rep(torch.zeros(sp, HIDDEN, dtype=torch.bfloat16), d)
    rmsnorm_k(x, rep(we, d), sc, ms, out)
    return out

def dev_attn_test(normed_tt, qw, kw, vw, ow, qnw, knw, rope_cos, rope_sin, sl, sp, d, mesh):
    """Device attention with rb_dim1 fix."""
    q = ttnn.matmul(normed_tt, qw)
    k = ttnn.matmul(normed_tt, kw)
    v = ttnn.matmul(normed_tt, vw)

    # Gather all heads
    qh = rb1(q, mesh)[:sl].float()
    kh = rb1(k, mesh)[:sl].float()
    vh = rb1(v, mesh)[:sl].float()

    # QK-norm
    def head_rms_norm(x, nw, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((x4 / rms) * nw).view(x.shape)

    qh = head_rms_norm(qh, qnw.float(), NQH)
    kh = head_rms_norm(kh, knw.float(), NKVH)

    # RoPE
    def rotate_half_flat(x, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        x1, x2 = x4[..., :HDIM//2], x4[..., HDIM//2:]
        return torch.cat((-x2, x1), dim=-1).view(x.shape)

    cos_h = rb(rope_cos, mesh)[:sl].float()
    sin_h = rb(rope_sin, mesh)[:sl].float()
    cos_full = cos_h.repeat(1, 2)[:, :HDIM]
    sin_full = sin_h.repeat(1, 2)[:, :HDIM]

    q_roped = qh * cos_full.repeat(1, NQH) + rotate_half_flat(qh, NQH) * sin_full.repeat(1, NQH)
    k_roped = kh * cos_full.repeat(1, NKVH) + rotate_half_flat(kh, NKVH) * sin_full.repeat(1, NKVH)

    # Reshape for SDPA
    q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k4.repeat(1, GQA, 1, 1)
    v4 = v4.repeat(1, GQA, 1, 1)

    q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); q4p[:, :, :sl] = q4
    k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); k4p[:, :, :sl] = k4
    v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); v4p[:, :, :sl] = v4

    attn = ttnn.transformer.scaled_dot_product_attention(
        rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

    ah = rb(attn, mesh).view(1, NQH, -1, HDIM)[:, :, :sl, :]
    ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
    ahp = _p(ah)
    if ahp.shape[0] < sp:
        ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
    attn_tt = shd(ahp, d, dim=1)

    o = ttnn.matmul(attn_tt, ow)
    o = ttnn.all_reduce(o)
    return o

def dev_moe_test(h_tt, rw, gate_all, up_all, down_all, sl, sp, d, mesh):
    """Device MoE with top-k masking."""
    local_dim = EPC * MOE_INTER
    router = ttnn.matmul(h_tt, rw)
    scores = ttnn.softmax(router, dim=-1)
    gate = ttnn.matmul(h_tt, gate_all)
    up = ttnn.matmul(h_tt, up_all)
    act = ttnn.multiply(ttnn.silu(gate), up)

    sh = rb(scores, mesh)[:sl, :NEXPERTS].float()
    topk_vals, topk_idx = torch.topk(sh, TOPK, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)
    routing_mask = torch.zeros_like(sh)
    routing_mask.scatter_(1, topk_idx, topk_vals)

    act_all = ttnn.to_torch(act, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    weighted_parts = []
    for c in range(N_CHIPS):
        chip_act = act_all[c * sp:(c + 1) * sp][:sl].float()
        cs = routing_mask[:, c * EPC:(c + 1) * EPC]
        cs_exp = cs.unsqueeze(-1).expand(-1, -1, MOE_INTER).reshape(sl, local_dim)
        weighted_parts.append((chip_act * cs_exp).to(torch.bfloat16))

    all_weighted = torch.cat(weighted_parts, dim=1)
    wp = _p(all_weighted)
    if wp.shape[0] < sp:
        wp = F.pad(wp, (0, 0, 0, sp - wp.shape[0]))
    weighted_tt = shd(wp, d, dim=1)

    out = ttnn.matmul(weighted_tt, down_all)
    out = ttnn.all_reduce(out)
    return out


def test_layer(mesh):
    with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
    def gt(k):
        with safe_open(kf[k], framework="pt") as f:
            return f.get_tensor(k)

    sl = 32
    sp = 32

    print("Loading layer 0 weights...")
    p = "model.layers.0"
    in_w = gt(f"{p}.input_layernorm.weight").to(torch.bfloat16)
    pa_w = gt(f"{p}.post_attention_layernorm.weight").to(torch.bfloat16)
    qw = gt(f"{p}.self_attn.q_proj.weight").T.contiguous().to(torch.bfloat16)
    kw = gt(f"{p}.self_attn.k_proj.weight").T.contiguous().to(torch.bfloat16)
    vw = gt(f"{p}.self_attn.v_proj.weight").T.contiguous().to(torch.bfloat16)
    ow = gt(f"{p}.self_attn.o_proj.weight").T.contiguous().to(torch.bfloat16)
    qnw = gt(f"{p}.self_attn.q_norm.weight").to(torch.bfloat16)
    knw = gt(f"{p}.self_attn.k_norm.weight").to(torch.bfloat16)
    rw = gt(f"{p}.mlp.gate.weight").T.contiguous().to(torch.bfloat16)

    gate_ws, up_ws, down_ws = [], [], []
    for ei in range(NEXPERTS):
        ep = f"{p}.mlp.experts.{ei}"
        gate_ws.append(gt(f"{ep}.gate_proj.weight").T.contiguous().to(torch.bfloat16))
        up_ws.append(gt(f"{ep}.up_proj.weight").T.contiguous().to(torch.bfloat16))
        down_ws.append(gt(f"{ep}.down_proj.weight").T.contiguous().to(torch.bfloat16))
    print("Weights loaded.")

    # RoPE tables
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(4096, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_t = torch.cos(angles).to(torch.bfloat16)
    sin_t = torch.sin(angles).to(torch.bfloat16)

    # Random input
    torch.manual_seed(42)
    embed_w = gt("model.embed_tokens.weight").to(torch.bfloat16)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] * 4)
    hidden = embed_w[input_ids].to(torch.bfloat16)

    d = mesh

    # Device weight setup
    sc = rep(torch.ones(TILE, TILE, dtype=torch.bfloat16), d, )
    ms = rep(torch.full((TILE, TILE), 1.0/HIDDEN, dtype=torch.bfloat16), d)
    # Fix: put sc/ms in a dict-like structure for dev_norm
    qw_d = shd(qw, d, dim=1)
    kw_d = shd(kw, d, dim=1)
    vw_d = shd(vw, d, dim=1)
    ow_d = shd(ow, d, dim=0)
    rope_cos_d = rep(cos_t, d)
    rope_sin_d = rep(sin_t, d)
    rw_d = rep(rw, d)

    gate_all = torch.cat(gate_ws, dim=1)
    up_all = torch.cat(up_ws, dim=1)
    down_all = torch.cat([dw for dw in down_ws], dim=0)
    gate_all_d = shd(gate_all, d, dim=1)
    up_all_d = shd(up_all, d, dim=1)
    down_all_d = shd(down_all, d, dim=0)

    # ---- Step 1: RMSNorm ----
    print("\n=== RMSNorm 1 ===")
    ref_normed = rms_norm_ref(hidden, in_w)

    h_tt = rep(hidden, d)
    normed_tt = dev_norm(h_tt, in_w, sp, sc, ms, d)
    dev_normed = rb(normed_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)

    pcc = torch.corrcoef(torch.stack([ref_normed.float().flatten(), dev_normed.float().flatten()]))[0, 1].item()
    maxe = (ref_normed.float() - dev_normed.float()).abs().max().item()
    print(f"  PCC: {pcc:.6f}  MaxErr: {maxe:.6f}")

    # ---- Step 2: Attention ----
    print("\n=== Attention ===")
    ref_attn_out = ref_attn(ref_normed, qw, kw, vw, ow, qnw, knw, cos_t, sin_t, sl)

    dev_attn_out_tt = dev_attn_test(normed_tt, qw_d, kw_d, vw_d, ow_d, qnw, knw,
                                     rope_cos_d, rope_sin_d, sl, sp, d, mesh)
    dev_attn_out = rb(dev_attn_out_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)

    pcc = torch.corrcoef(torch.stack([ref_attn_out.float().flatten(), dev_attn_out.float().flatten()]))[0, 1].item()
    maxe = (ref_attn_out.float() - dev_attn_out.float()).abs().max().item()
    print(f"  PCC: {pcc:.6f}  MaxErr: {maxe:.6f}")

    # Sub-check: QKV matmul
    print("  Sub-checks:")
    ref_q = (ref_normed[:sl].float() @ qw.float()).to(torch.bfloat16)
    dev_q_raw = rb1(ttnn.matmul(normed_tt, qw_d), mesh)[:sl].to(torch.bfloat16)
    pcc_q = torch.corrcoef(torch.stack([ref_q.float().flatten(), dev_q_raw.float().flatten()]))[0, 1].item()
    print(f"    Q matmul PCC: {pcc_q:.6f}")

    # ---- Step 3: Residual 1 ----
    print("\n=== Residual 1 ===")
    ref_res1 = (hidden.float() + ref_attn_out.float()).to(torch.bfloat16)
    res1_tt = rep(torch.zeros(sp, HIDDEN, dtype=torch.bfloat16), d)
    residual_add_kernel(h_tt, dev_attn_out_tt, res1_tt)
    dev_res1 = rb(res1_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)

    pcc = torch.corrcoef(torch.stack([ref_res1.float().flatten(), dev_res1.float().flatten()]))[0, 1].item()
    print(f"  PCC: {pcc:.6f}")

    # ---- Step 4: RMSNorm 2 + MoE ----
    print("\n=== RMSNorm 2 ===")
    ref_normed2 = rms_norm_ref(ref_res1, pa_w)
    normed2_tt = dev_norm(res1_tt, pa_w, sp, sc, ms, d)
    dev_normed2 = rb(normed2_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)
    pcc = torch.corrcoef(torch.stack([ref_normed2.float().flatten(), dev_normed2.float().flatten()]))[0, 1].item()
    print(f"  PCC: {pcc:.6f}")

    print("\n=== MoE ===")
    ref_moe_out = ref_moe(ref_normed2, rw, gate_ws, up_ws, down_ws, sl)

    # Test 1: device MoE with device-computed input (has RMSNorm drift)
    dev_moe_out_tt = dev_moe_test(normed2_tt, rw_d, gate_all_d, up_all_d, down_all_d, sl, sp, d, mesh)
    dev_moe_out = rb(dev_moe_out_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)
    pcc = torch.corrcoef(torch.stack([ref_moe_out.float().flatten(), dev_moe_out.float().flatten()]))[0, 1].item()
    maxe = (ref_moe_out.float() - dev_moe_out.float()).abs().max().item()
    print(f"  PCC (dev input): {pcc:.6f}  MaxErr: {maxe:.6f}")

    # Test 2: device MoE with SAME ref input (isolates MoE logic from input drift)
    ref_inp_tt = rep(ref_normed2, d)
    dev_moe_out2_tt = dev_moe_test(ref_inp_tt, rw_d, gate_all_d, up_all_d, down_all_d, sl, sp, d, mesh)
    dev_moe_out2 = rb(dev_moe_out2_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)
    pcc2 = torch.corrcoef(torch.stack([ref_moe_out.float().flatten(), dev_moe_out2.float().flatten()]))[0, 1].item()
    maxe2 = (ref_moe_out.float() - dev_moe_out2.float()).abs().max().item()
    print(f"  PCC (ref input): {pcc2:.6f}  MaxErr: {maxe2:.6f}")

    # Sub-check: compare expert selection
    ref_scores_h = torch.softmax((ref_normed2[:sl].float() @ rw.float()), dim=-1)
    _, ref_topk = torch.topk(ref_scores_h, TOPK, dim=-1)
    dev_router = ttnn.matmul(ref_inp_tt, rw_d)
    dev_scores_tt = ttnn.softmax(dev_router, dim=-1)
    dev_scores_h = rb(dev_scores_tt, mesh)[:sl, :NEXPERTS].float()
    _, dev_topk = torch.topk(dev_scores_h, TOPK, dim=-1)
    expert_match = sum(1 for t in range(sl) if set(ref_topk[t].tolist()) == set(dev_topk[t].tolist()))
    print(f"  Expert selection match: {expert_match}/{sl} tokens")
    score_pcc = torch.corrcoef(torch.stack([ref_scores_h.flatten(), dev_scores_h.flatten()]))[0, 1].item()
    print(f"  Router score PCC: {score_pcc:.6f}")

    # ---- Step 5: Final residual ----
    print("\n=== Final (Residual 2) ===")
    ref_final = (ref_res1.float() + ref_moe_out.float()).to(torch.bfloat16)
    final_tt = rep(torch.zeros(sp, HIDDEN, dtype=torch.bfloat16), d)
    residual_add_kernel(res1_tt, dev_moe_out_tt, final_tt)
    dev_final = rb(final_tt, mesh)[:sl, :HIDDEN].to(torch.bfloat16)

    pcc = torch.corrcoef(torch.stack([ref_final.float().flatten(), dev_final.float().flatten()]))[0, 1].item()
    print(f"  PCC: {pcc:.6f}")

    ok = pcc > 0.95
    print(f"\n{'PASS' if ok else 'FAIL'}: Layer 0 final PCC = {pcc:.6f}")
    return ok


if __name__ == "__main__":
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    try:
        test_layer(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
