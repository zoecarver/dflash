"""DFlash draft model (8-layer cross-attention) on Tenstorrent 4-chip TP.

NOT YET MIGRATED to scratch/tracing model.
Will break until draft gets its own scratch preallocation.
"""
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from silu_mul import silu_mul_kernel

from device import (
    TILE, HIDDEN, HDIM, NQH, NKVH, GQA, EPS, ROPE_THETA,
    N_CHIPS, NQH_TP, NKVH_TP,
    rmsnorm_k,
    _p, rep, shd, ztt, rb, rb_dim1,
)
from residual_add import residual_add_kernel

DLAYERS = 8
DINTER = 6144
BSIZE = 16
TLAYER_IDS = [1, 12, 23, 34, 45]
MASK_ID = 151669
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"


# Allocating versions of dev_norm/dev_add for draft model (not yet migrated to scratch).
# These allocate output tensors on each call -- will be replaced with scratch preallocation.
def _draft_norm(x, nw_key, w, d):
    out = ztt((x.shape[0], HIDDEN), d)
    rmsnorm_k(x, w[nw_key], w["sc"], w["ms"], out)
    return out


def _draft_add(a, b, d):
    out = ztt((a.shape[0], HIDDEN), d)
    residual_add_kernel(a, b, out)
    return out


# ---------------------------------------------------------------------------
# Load draft weights to device
# ---------------------------------------------------------------------------
def load_draft_weights(d, w):
    """Load draft model weights into w. Expects w already has target weights."""
    import time
    t0 = time.time()
    print("  draft weights...")
    with safe_open(f"{DRAFT_DIR}/model.safetensors", framework="pt") as f:
        w["d.fc"] = rep(f.get_tensor("fc.weight").T.contiguous().to(torch.bfloat16), d)
        hn_w = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        fn_w = f.get_tensor("norm.weight").to(torch.bfloat16)
        w["d.hn_w"] = hn_w
        w["d.fn_w"] = fn_w
        w["d.hn_w_tt"] = rep(hn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
        w["d.fn_w_tt"] = rep(fn_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

        for li in range(DLAYERS):
            dp = f"layers.{li}"
            lp = f"d.{li}"
            din_w = f.get_tensor(f"{dp}.input_layernorm.weight").to(torch.bfloat16)
            dpa_w = f.get_tensor(f"{dp}.post_attention_layernorm.weight").to(torch.bfloat16)
            w[f"{lp}.in_w"] = din_w
            w[f"{lp}.pa_w"] = dpa_w
            w[f"{lp}.in_w_tt"] = rep(din_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)
            w[f"{lp}.pa_w_tt"] = rep(dpa_w.unsqueeze(0).expand(TILE, -1).contiguous(), d)

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
            w[f"{lp}.gw"] = shd(gw, d, dim=1)
            w[f"{lp}.uw"] = shd(uw, d, dim=1)
            w[f"{lp}.fc2"] = shd(fc2, d, dim=0)

    print(f"  draft weights loaded in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Draft forward -- NOT YET MIGRATED to scratch/tracing model.
# Will break until draft gets its own scratch preallocation.
# ---------------------------------------------------------------------------
def draft_fwd(noise, ctx, w, sl, ctx_len, sp, d):
    """DFlash draft model. Cross-attention uses host reshaping for now."""
    h = noise
    for li in range(DLAYERS):
        lp = f"d.{li}"
        n = _draft_norm(h, f"{lp}.in_w_tt", w, d)

        # Linear: cross-attention Q from draft, K/V from [ctx, draft]
        q = ttnn.matmul(n, w[f"{lp}.qw"])

        # Host: concat ctx + draft for K/V (dynamic-length concat not in TT-Lang)
        nh = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        ch = rb(ctx)[:ctx_len, :HIDDEN].to(torch.bfloat16)
        kv_in = torch.cat([ch, nh], dim=0)
        kv_len = kv_in.shape[0]
        kv_sp = ((kv_len + TILE - 1) // TILE) * TILE
        kv_p = _p(kv_in)
        if kv_p.shape[0] < kv_sp:
            kv_p = F.pad(kv_p, (0, 0, 0, kv_sp - kv_p.shape[0]))
        kv = rep(kv_p, d)

        k = ttnn.matmul(kv, w[f"{lp}.kw"])  # Linear
        v = ttnn.matmul(kv, w[f"{lp}.vw"])  # Linear

        # Host: gather all heads for QK-norm + RoPE (same reasons as dev_attn)
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

        # SDPA (replicated cross-attention)
        at = ttnn.transformer.scaled_dot_product_attention(
            rep(pk4(qh, NQH, sl, sp), d), rep(pk4(kh, NQH, kv_len, kv_sp), d),
            rep(pk4(vh, NQH, kv_len, kv_sp), d), is_causal=False)

        ah = rb(at).view(1, NQH, -1, HDIM)[:, :, :sl, :]
        ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
        ap = _p(ah)
        if ap.shape[0] < sp:
            ap = F.pad(ap, (0, 0, 0, sp - ap.shape[0]))

        # Linear: O projection (row-parallel) + collective all_reduce
        o = ttnn.matmul(shd(ap, d, dim=1), w[f"{lp}.ow"])
        o = ttnn.all_reduce(o)
        h = _draft_add(h, o, d)

        # Dense MLP
        n2 = _draft_norm(h, f"{lp}.pa_w_tt", w, d)
        gate = ttnn.matmul(n2, w[f"{lp}.gw"])  # Linear: gate (column-parallel)
        up = ttnn.matmul(n2, w[f"{lp}.uw"])     # Linear: up (column-parallel)
        # TT-Lang: fused silu(gate) * up (avoids host readback)
        activated = ttnn.zeros_like(gate)
        silu_mul_kernel(gate, up, activated)
        # Linear: down projection (row-parallel) + collective all_reduce
        fc2 = ttnn.matmul(activated, w[f"{lp}.fc2"])
        fc2 = ttnn.all_reduce(fc2)
        h = _draft_add(h, fc2, d)

    h = _draft_norm(h, "d.fn_w_tt", w, d)
    return h
