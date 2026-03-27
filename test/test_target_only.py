"""Minimal target-only generation. No draft model, no speculative decoding.
Just run the target model autoregressively to verify it produces coherent text.
"""
import time
import json
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel

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
N_CHIPS = 4
EPC = NEXPERTS // N_CHIPS
NQH_TP = NQH // N_CHIPS
NKVH_TP = NKVH // N_CHIPS

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
_MESH = None


def _p(t):
    if t.dim() == 1: t = t.unsqueeze(0)
    h, w_ = t.shape[-2], t.shape[-1]
    ph, pw = (TILE - h % TILE) % TILE, (TILE - w_ % TILE) % TILE
    if ph or pw: t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)

def rep(t, d):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(d))

def shd(t, d, dim):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(d, dim=dim))

def rb(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=0))[:t.shape[0]]

def rb1(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=1))

def ztt(shape, d):
    return rep(torch.zeros(shape, dtype=torch.bfloat16), d)


def dev_norm(x, nw, sp, w, d):
    we = nw.unsqueeze(0).contiguous()  # (1, HIDDEN)
    weight_tt = rep(we, d)
    return ttnn.rms_norm(x, weight=weight_tt, epsilon=EPS)

def dev_add(a, b, sp, d):
    out = ztt((sp, HIDDEN), d)
    residual_add_kernel(a, b, out)
    return out

def dev_attn(normed, w, lp, sl, sp, d):
    q = ttnn.matmul(normed, w[f"{lp}.qw"])
    k = ttnn.matmul(normed, w[f"{lp}.kw"])
    v = ttnn.matmul(normed, w[f"{lp}.vw"])

    qh = rb1(q)[:sl].float()
    kh = rb1(k)[:sl].float()
    vh = rb1(v)[:sl].float()

    qnw = w[f"{lp}.qnw"].float()
    knw = w[f"{lp}.knw"].float()

    def head_rms_norm(x, nw, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        rms = torch.sqrt(x4.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((x4 / rms) * nw).view(x.shape)

    qh = head_rms_norm(qh, qnw, NQH)
    kh = head_rms_norm(kh, knw, NKVH)

    def rotate_half_flat(x, n_heads):
        x4 = x.view(-1, n_heads, HDIM)
        x1, x2 = x4[..., :HDIM//2], x4[..., HDIM//2:]
        return torch.cat((-x2, x1), dim=-1).view(x.shape)

    cos_h = rb(w["rope_cos"])[:sl].float()
    sin_h = rb(w["rope_sin"])[:sl].float()
    cos_full = cos_h.repeat(1, 2)[:, :HDIM]
    sin_full = sin_h.repeat(1, 2)[:, :HDIM]

    q_roped = qh * cos_full.repeat(1, NQH) + rotate_half_flat(qh, NQH) * sin_full.repeat(1, NQH)
    k_roped = kh * cos_full.repeat(1, NKVH) + rotate_half_flat(kh, NKVH) * sin_full.repeat(1, NKVH)

    q4 = q_roped.view(1, sl, NQH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k_roped.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    v4 = vh.view(1, sl, NKVH, HDIM).transpose(1, 2).to(torch.bfloat16)
    k4 = k4.repeat_interleave(GQA, dim=1)
    v4 = v4.repeat_interleave(GQA, dim=1)

    q4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); q4p[:, :, :sl] = q4
    k4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); k4p[:, :, :sl] = k4
    v4p = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16); v4p[:, :, :sl] = v4

    attn = ttnn.transformer.scaled_dot_product_attention(
        rep(q4p, d), rep(k4p, d), rep(v4p, d), is_causal=True)

    ah = rb(attn).view(1, NQH, -1, HDIM)[:, :, :sl, :]
    ah = ah.transpose(1, 2).contiguous().view(sl, NQH * HDIM).to(torch.bfloat16)
    ahp = _p(ah)
    if ahp.shape[0] < sp: ahp = F.pad(ahp, (0, 0, 0, sp - ahp.shape[0]))
    attn_tt = shd(ahp, d, dim=1)

    o = ttnn.matmul(attn_tt, w[f"{lp}.ow"])
    o = ttnn.all_reduce(o)
    return o

def dev_moe(h, w, lp, sl, sp, d):
    router = ttnn.matmul(h, w[f"{lp}.rw"])
    scores = ttnn.softmax(router, dim=-1)
    local_dim = EPC * MOE_INTER
    gate = ttnn.matmul(h, w[f"{lp}.gate_all"])
    up = ttnn.matmul(h, w[f"{lp}.up_all"])
    act = ttnn.multiply(ttnn.silu(gate), up)

    sh = rb(scores)[:sl, :NEXPERTS].float()
    topk_vals, topk_idx = torch.topk(sh, TOPK, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)
    routing_mask = torch.zeros_like(sh)
    routing_mask.scatter_(1, topk_idx, topk_vals)

    act_all = ttnn.to_torch(act, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=0))
    weighted_parts = []
    for c in range(N_CHIPS):
        chip_act = act_all[c * sp:(c + 1) * sp][:sl].float()
        cs = routing_mask[:, c * EPC:(c + 1) * EPC]
        cs_exp = cs.unsqueeze(-1).expand(-1, -1, MOE_INTER).reshape(sl, local_dim)
        weighted_parts.append((chip_act * cs_exp).to(torch.bfloat16))

    all_weighted = torch.cat(weighted_parts, dim=1)
    wp = _p(all_weighted)
    if wp.shape[0] < sp: wp = F.pad(wp, (0, 0, 0, sp - wp.shape[0]))
    weighted_tt = shd(wp, d, dim=1)

    out = ttnn.matmul(weighted_tt, w[f"{lp}.down_all"])
    out = ttnn.all_reduce(out)
    return out


def target_fwd(h, w, sl, sp, d):
    for li in range(TLAYERS):
        lp = f"t.{li}"
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
    return logits


def main():
    global _MESH
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    _MESH = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    d = _MESH

    try:
        # Load weights (same as dflash_device.py)
        from dflash_device import load_weights
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

        emb = w["embed_h"]
        print(f"Prompt: {ids.shape[0]} tokens")

        # Simple autoregressive generation -- no draft, no speculation
        max_new = 32
        generated = ids.tolist()

        for step in range(max_new):
            sl = len(generated)
            sp = ((sl + TILE - 1) // TILE) * TILE

            h = _p(emb[torch.tensor(generated)])
            if h.shape[0] < sp:
                h = F.pad(h, (0, 0, 0, sp - h.shape[0]))

            t0 = time.time()
            logits = target_fwd(rep(h, d), w, sl, sp, d)
            el = time.time() - t0

            lh = rb1(logits)[:sl, :VOCAB].float()
            next_tok = torch.argmax(lh[-1]).item()
            generated.append(next_tok)

            try:
                tok_str = tok.decode([next_tok])
            except:
                tok_str = f"<{next_tok}>"
            print(f"  step {step}: tok={next_tok} '{tok_str}' ({el:.1f}s, ctx={sl})")

            if next_tok in (151643, 151645):
                break

        print(f"\n--- Output ---")
        try:
            print(tok.decode(generated, skip_special_tokens=True))
        except:
            print(generated)

    finally:
        ttnn.close_mesh_device(_MESH)


if __name__ == "__main__":
    main()
