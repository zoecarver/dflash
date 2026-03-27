"""Speculative decoding: Qwen3 target (4-chip TP) + DFlash draft (replicated).

Target model runs on 4-chip mesh with full TP.
Draft model runs replicated on the same mesh (each chip does full computation).
"""
import time
import torch
import torch.nn.functional as F
import ttnn

from device import (
    TILE, HIDDEN, VOCAB, N_CHIPS,
    _p, rep, rb, rb_dim1,
    open_dev, close_dev,
)
from qwen3 import (
    load_target_weights, target_fwd, target_fwd_save_hs,
    prealloc_scratch,
)
from dflash_draft import (
    load_draft_weights, draft_fwd, prepare_context, setup_rope_tables,
    to_dev as draft_to_dev,
    BSIZE, TLAYER_IDS, MASK_ID, SP, N_CTX_LAYERS,
    _tile_pad, norm_k,
)

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def spec_generate(ids, w, d, max_new=64):
    pl = ids.shape[0]
    sp = ((pl + TILE - 1) // TILE) * TILE

    out = torch.full((pl + max_new + BSIZE,), MASK_ID, dtype=torch.long)
    out[:pl] = ids

    emb = w["embed_h"]

    # Prefill: run target on prompt, save hidden states for draft context
    print("Prefill...")
    t0 = time.time()
    h = _p(emb[ids])
    if h.shape[0] < sp:
        h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
    h_tt = rep(h, d)

    s = prealloc_scratch(sp, d)
    # Set up target RoPE tables for this sequence length
    cos_host = w["rope_cos_full_host"][:sp].to(torch.bfloat16)
    sin_host = w["rope_sin_adj_host"][:sp].to(torch.bfloat16)
    w["rope_cos_sp"] = rep(cos_host, d, mem=ttnn.L1_MEMORY_CONFIG)
    w["rope_sin_sp"] = rep(sin_host, d, mem=ttnn.L1_MEMORY_CONFIG)

    save_set = set(TLAYER_IDS)
    logits, ths = target_fwd_save_hs(h_tt, w, sp, d, s, save_set)

    pft = time.time() - t0
    print(f"Prefill: {pft:.1f}s ({pl} tokens)")

    # Greedy pick first token from prefill
    lh = rb_dim1(logits)[:pl, :VOCAB].float()
    out[pl] = torch.argmax(lh[-1]).item()

    # Build draft context from target hidden states (all on device)
    tf_dev = ttnn.concat([ths[lid] for lid in TLAYER_IDS], dim=-1)
    ctx = prepare_context(tf_dev, w, d)
    setup_rope_tables(w, sp, d)  # sp = tile-padded context rows

    start = pl
    gen = 0
    ahist = []

    print("Decoding...")
    while start < pl + max_new:
        ts = time.time()
        bids = out[start:start + BSIZE].clone()
        noise = _p(emb[bids])
        if noise.shape[0] < SP:
            noise = F.pad(noise, (0, 0, 0, SP - noise.shape[0]))
        noise_dev = draft_to_dev(noise, d)

        # Draft: propose BSIZE tokens
        dout = draft_fwd(noise_dev, ctx, w, d)
        dl = ttnn.matmul(dout, w["lm_head"])
        dlh = rb_dim1(dl)[:BSIZE, :VOCAB].float()
        bids[1:] = torch.argmax(dlh[:-1], dim=-1)

        # Target: verify full context
        verify_ids = torch.cat([out[:start], bids])
        vlen = verify_ids.shape[0]
        vsp = ((vlen + TILE - 1) // TILE) * TILE
        vh = _p(emb[verify_ids])
        if vh.shape[0] < vsp:
            vh = F.pad(vh, (0, 0, 0, vsp - vh.shape[0]))

        vs = prealloc_scratch(vsp, d)
        cos_v = w["rope_cos_full_host"][:vsp].to(torch.bfloat16)
        sin_v = w["rope_sin_adj_host"][:vsp].to(torch.bfloat16)
        w["rope_cos_sp"] = rep(cos_v, d, mem=ttnn.L1_MEMORY_CONFIG)
        w["rope_sin_sp"] = rep(sin_v, d, mem=ttnn.L1_MEMORY_CONFIG)
        vl, vhs = target_fwd_save_hs(rep(vh, d), w, vsp, d, vs, save_set)
        vlh = rb_dim1(vl)[start:start + BSIZE, :VOCAB].float()
        post = torch.argmax(vlh, dim=-1)

        # Accept/reject
        acc = (bids[1:] == post[:-1]).to(torch.int64).cumprod(0).sum().item()
        out[start:start+acc+1] = bids[:acc+1]
        out[start+acc+1] = post[acc]
        start += acc + 1
        gen += acc + 1
        ahist.append(acc + 1)

        # Update draft context from verify hidden states (all on device)
        vf_dev = ttnn.concat([vhs[lid] for lid in TLAYER_IDS], dim=-1)
        ctx = prepare_context(vf_dev, w, d)
        setup_rope_tables(w, vsp, d)  # vsp = tile-padded verify context rows

        el = time.time() - ts
        avg = sum(ahist) / len(ahist)
        print(f"  step {len(ahist)}: acc={acc+1}/{BSIZE} avg={avg:.1f} "
              f"{el:.1f}s gen={gen}")

        if out[start - 1].item() in (151643, 151645):
            break

    out = out[:start]
    out = out[out != MASK_ID]
    return out


def main():
    print("=" * 60)
    print(f"DFlash Speculative Decoding on Tenstorrent ({N_CHIPS} chips)")
    print("=" * 60)

    d = open_dev()
    try:
        w = load_target_weights(d)
        dw = load_draft_weights(d)
        # Merge draft weights into shared dict
        w.update(dw)

        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TARGET_DIR)
            prompt = "Write a Python function that computes fibonacci numbers."
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
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
