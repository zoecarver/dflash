"""Speculative decoding: Qwen3 target + DFlash draft combined."""
import torch
import torch.nn.functional as F
import ttnn

from device import (
    TILE, HIDDEN, VOCAB, N_CHIPS,
    _p, rep, rb_dim1,
    open_dev, close_dev,
)
from qwen3 import load_target_weights, target_fwd
from dflash_draft import (
    load_draft_weights, draft_fwd, _draft_norm,
    BSIZE, TLAYER_IDS, MASK_ID,
)

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


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
    import time
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
    ctx = ttnn.matmul(rep(tf_p, d), w["d.fc"])  # Linear: project target hidden states
    ctx = _draft_norm(ctx, "d.hn_w_tt", w, d)

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
        dl = ttnn.matmul(dout, w["lm_head"])  # Linear: draft lm_head
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
            ctx = ttnn.matmul(rep(vfp, d), w["d.fc"])  # Linear: re-project context
            ctx = _draft_norm(ctx, "d.hn_w_tt", w, d)

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
    print(f"DFlash Speculative Decoding on Tenstorrent ({N_CHIPS} chips)")
    print("=" * 60)

    d = open_dev()
    try:
        w = load_target_weights(d)
        load_draft_weights(d, w)
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
