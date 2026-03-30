"""Hybrid spec decode: CPU target (transformers) + Tenstorrent draft (TTNN).

Uses the CPU target model for prefill/verify (ground truth hidden states + logits),
but runs the device draft model for proposals. This isolates whether the device
draft diverges from the CPU draft given the same context features.
"""
import sys
sys.path.insert(0, "/tmp")

import time
import json
import torch
import torch.nn.functional as F
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_minimal import extract_context_feature, sample

from device import (
    TILE, HIDDEN, VOCAB, N_CHIPS,
    _p, rep, rb, rb_dim1,
    open_dev, close_dev,
)
from dflash_draft import (
    load_draft_weights, draft_fwd_ttnn, prepare_context_ttnn, setup_rope_tables,
    to_dev as draft_to_dev,
    BSIZE, TLAYER_IDS, MASK_ID, SP, N_CTX_LAYERS, DLAYERS,
    _tile_pad, NQH, NKVH, HDIM, GQA, EPS,
)

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"


def main():
    # Load CPU target model
    print("Loading CPU target model...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    target.eval()
    print(f"  Target loaded in {time.time()-t0:.1f}s")

    # Load device draft model
    print("Opening Tenstorrent device...")
    d = open_dev()
    w = load_draft_weights(d)
    # Also need lm_head on device for draft logits
    lm_head_cpu = target.lm_head.weight.data.T.contiguous().to(torch.bfloat16)  # (hidden, vocab)

    # Tokenize
    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
    input_ids = tok(text, return_tensors="pt")["input_ids"]
    pl = input_ids.shape[1]
    print(f"Prompt: {pl} tokens")

    try:
        # === Prefill on CPU ===
        print("\nPrefill (CPU)...")
        t0 = time.time()
        with torch.inference_mode():
            output = target(
                input_ids,
                output_hidden_states=True,
            )
        print(f"  Prefill: {time.time()-t0:.1f}s")

        # Extract context features from CPU target
        target_hidden_cpu = extract_context_feature(output.hidden_states, TLAYER_IDS)
        # shape: (1, pl, 5*HIDDEN)
        print(f"  target_hidden shape: {target_hidden_cpu.shape}")

        # First token from CPU target
        out = torch.full((pl + 64 + BSIZE,), MASK_ID, dtype=torch.long)
        out[:pl] = input_ids[0]
        first_logits = output.logits[0, -1, :].float()
        out[pl] = torch.argmax(first_logits).item()
        print(f"  First token: {out[pl].item()}")

        # Upload context to device
        ctx_host = target_hidden_cpu.squeeze(0).to(torch.bfloat16)  # (pl, 5*HIDDEN)
        ctx_sp = _tile_pad(pl)
        if ctx_sp > pl:
            pad_rows = ctx_host[-1:].expand(ctx_sp - pl, -1)
            ctx_host = torch.cat([ctx_host, pad_rows], dim=0)
        ctx_dev = rep(ctx_host, d)
        ctx = prepare_context_ttnn(ctx_dev, w, d)
        setup_rope_tables(w, ctx_sp, d, q_start=pl)

        emb_weight = target.model.embed_tokens.weight.data.to(torch.bfloat16)

        start = pl
        gen = 0
        ahist = []

        print("\nDecoding (device draft, CPU verify)...")
        while start < pl + 64:
            ts = time.time()
            bids = out[start:start + BSIZE].clone()

            # Embed on CPU, upload to device
            noise_cpu = emb_weight[bids]  # (BSIZE, HIDDEN)
            noise_padded = _p(noise_cpu)
            if noise_padded.shape[0] < SP:
                noise_padded = F.pad(noise_padded, (0, 0, 0, SP - noise_padded.shape[0]))
            noise_dev = draft_to_dev(noise_padded, d)

            # Draft forward on device
            dout = draft_fwd_ttnn(noise_dev, ctx, w, d)
            dout_cpu = rb(dout)[:BSIZE, :HIDDEN].float()

            # lm_head on CPU (avoid device lm_head sharding issues)
            dlh = (dout_cpu.to(torch.bfloat16) @ lm_head_cpu).float()[:BSIZE, :VOCAB]
            bids[1:] = torch.argmax(dlh[1:BSIZE], dim=-1)

            # Verify on CPU target (no KV cache, full sequence)
            verify_ids = torch.cat([out[:start], bids]).unsqueeze(0)
            with torch.inference_mode():
                vout = target(verify_ids, output_hidden_states=True)
            vlh = vout.logits[0, start:start + BSIZE, :VOCAB].float()
            post = torch.argmax(vlh, dim=-1)

            # Per-position diagnostics (first 5 steps)
            if len(ahist) < 5:
                matches = (bids[1:] == post[:-1]).tolist()
                n_match = sum(matches)
                print(f"    match: {n_match}/15  bids[1:]={bids[1:6].tolist()} post[:-1]={post[:5].tolist()}")
                for pos in range(min(5, BSIZE - 1)):
                    dtok = torch.argmax(dlh[pos + 1]).item()
                    ttok = post[pos].item()
                    d5 = torch.topk(dlh[pos + 1], 5).indices.tolist()
                    t5 = torch.topk(vlh[pos], 5).indices.tolist()
                    overlap = len(set(d5) & set(t5))
                    print(f"    pos {pos}: d={dtok} t={ttok} {'Y' if dtok==ttok else 'N'} top5ovlp={overlap}/5")

            # Accept/reject
            acc = (bids[1:] == post[:-1]).to(torch.int64).cumprod(0).sum().item()
            out[start:start+acc+1] = bids[:acc+1]
            out[start+acc+1] = post[acc]
            start += acc + 1
            gen += acc + 1
            ahist.append(acc + 1)

            # Update context from CPU verify hidden states
            new_ctx_cpu = extract_context_feature(
                vout.hidden_states, TLAYER_IDS
            ).squeeze(0)[:start].to(torch.bfloat16)
            ctx_sp = _tile_pad(start)
            if ctx_sp > start:
                pad_rows = new_ctx_cpu[-1:].expand(ctx_sp - start, -1)
                new_ctx_cpu = torch.cat([new_ctx_cpu, pad_rows], dim=0)
            ctx_dev = rep(new_ctx_cpu, d)
            ctx = prepare_context_ttnn(ctx_dev, w, d)
            setup_rope_tables(w, ctx_sp, d, q_start=start)

            el = time.time() - ts
            avg = sum(ahist) / len(ahist)
            print(f"  step {len(ahist)}: acc={acc+1}/{BSIZE} avg={avg:.1f} {el:.1f}s gen={gen}")

            if out[start - 1].item() in (151643, 151645):
                break

        out_ids = out[:start]
        out_ids = out_ids[out_ids != MASK_ID]
        print(f"\n--- Output ---\n{tok.decode(out_ids, skip_special_tokens=True)}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
