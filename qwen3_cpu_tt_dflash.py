"""Speculative decoding: CPU Qwen3 target + Tenstorrent DFlash draft.

CPU Qwen3 (transformers) produces logits and hidden states — stock model,
nothing custom. Everything DFlash-related runs on Tenstorrent: context
projection, 8-layer draft forward, lm_head, argmax.
"""
import time
import torch
import torch.nn.functional as F
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM

from device import (
    TILE, HIDDEN, VOCAB, N_CHIPS,
    _p, rep, rb, rb_dim1,
    open_dev, close_dev,
)
import dflash_draft
from dflash_draft import (
    load_draft_weights, draft_fwd_ttnn, prepare_context_ttnn, setup_rope_tables,
    to_dev as draft_to_dev,
    BSIZE, TLAYER_IDS, MASK_ID, SP, N_CTX_LAYERS,
    _tile_pad,
)

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"

# Set True to use TT-Lang kernels for softmax, residual_add, silu_mul
USE_TTLANG = True
dflash_draft.TTLANG_ENABLED = USE_TTLANG
# TT-Lang rmsnorm has ~0.63x magnitude bug on mesh -- enable to test the flow
dflash_draft.TTLANG_RMSNORM = False


def extract_context_feature(hidden_states, layer_ids):
    """Extract and concatenate hidden states at target layer indices."""
    offset = 1  # hidden_states[0] is the embedding, layer i is at index i+1
    return torch.cat([hidden_states[lid + offset] for lid in layer_ids], dim=-1)


def main():
    # --- CPU: load stock Qwen3 model ---
    print("Loading Qwen3 (CPU, transformers)...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    target.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # --- Device: load DFlash draft weights ---
    print("Opening Tenstorrent device...")
    d = open_dev()

    try:
        dw = load_draft_weights(d)
        # lm_head from target model, replicated on device for draft logits
        lm_head_host = target.lm_head.weight.data.T.contiguous().to(torch.bfloat16)
        dw["lm_head_rep"] = rep(lm_head_host, d)

        emb_weight = target.model.embed_tokens.weight.data.to(torch.bfloat16)

        # --- Tokenize ---
        tok = AutoTokenizer.from_pretrained(TARGET_DIR)
        prompt = "Write a Python function that computes fibonacci numbers."
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        input_ids = tok(text, return_tensors="pt")["input_ids"]
        pl = input_ids.shape[1]
        print(f"Prompt: {pl} tokens")

        # === CPU: prefill ===
        print("\nPrefill...")
        t0 = time.time()
        with torch.inference_mode():
            cpu_out = target(input_ids, output_hidden_states=True)
        print(f"  {time.time()-t0:.1f}s ({pl} tokens)")

        out = torch.full((pl + 64 + BSIZE,), MASK_ID, dtype=torch.long)
        out[:pl] = input_ids[0]
        out[pl] = torch.argmax(cpu_out.logits[0, -1, :].float()).item()

        # === Device: prepare initial context from CPU hidden states ===
        ctx_cpu = extract_context_feature(
            cpu_out.hidden_states, TLAYER_IDS
        ).squeeze(0).to(torch.bfloat16)  # (pl, 5*HIDDEN)
        ctx_sp = _tile_pad(pl)
        if ctx_sp > pl:
            pad = ctx_cpu[-1:].expand(ctx_sp - pl, -1)
            ctx_cpu = torch.cat([ctx_cpu, pad], dim=0)
        ctx = prepare_context_ttnn(rep(ctx_cpu, d), dw, d)
        setup_rope_tables(dw, ctx_sp, d, q_start=pl)

        start = pl
        gen = 0
        ahist = []

        print("\nDecoding...")
        while start < pl + 64:
            ts = time.time()
            bids = out[start:start + BSIZE].clone()

            # === Device: embed, draft forward, lm_head, argmax ===
            noise = _p(emb_weight[bids])
            if noise.shape[0] < SP:
                noise = F.pad(noise, (0, 0, 0, SP - noise.shape[0]))
            noise_dev = draft_to_dev(noise, d)

            dout = draft_fwd_ttnn(noise_dev, ctx, dw, d)
            dl = ttnn.matmul(dout, dw["lm_head_rep"])
            # Argmax on device, readback just the token IDs
            draft_tokens = ttnn.argmax(dl, dim=-1)
            draft_ids = rb(draft_tokens)[:BSIZE].long().squeeze(-1)
            bids[1:] = draft_ids[1:BSIZE]

            # === CPU: verify (stock Qwen3 forward) ===
            verify_ids = torch.cat([out[:start], bids]).unsqueeze(0)
            with torch.inference_mode():
                cpu_vout = target(verify_ids, output_hidden_states=True)
            vlh = cpu_vout.logits[0, start:start + BSIZE, :VOCAB].float()
            post = torch.argmax(vlh, dim=-1)

            # Accept/reject (15 integers, trivial)
            acc = (bids[1:] == post[:-1]).to(torch.int64).cumprod(0).sum().item()
            out[start:start+acc+1] = bids[:acc+1]
            out[start+acc+1] = post[acc]
            start += acc + 1
            gen += acc + 1
            ahist.append(acc + 1)

            # === Device: update context from CPU hidden states ===
            new_ctx = extract_context_feature(
                cpu_vout.hidden_states, TLAYER_IDS
            ).squeeze(0)[:start].to(torch.bfloat16)
            ctx_sp = _tile_pad(start)
            if ctx_sp > start:
                pad = new_ctx[-1:].expand(ctx_sp - start, -1)
                new_ctx = torch.cat([new_ctx, pad], dim=0)
            ctx = prepare_context_ttnn(rep(new_ctx, d), dw, d)
            setup_rope_tables(dw, ctx_sp, d, q_start=start)

            el = time.time() - ts
            avg = sum(ahist) / len(ahist)
            print(f"  step {len(ahist)}: acc={acc+1}/{BSIZE} avg={avg:.1f} "
                  f"{el:.1f}s gen={gen}")

            if out[start - 1].item() in (151643, 151645):
                break

        out_ids = out[:start]
        out_ids = out_ids[out_ids != MASK_ID]
        print(f"\n--- Output ---\n{tok.decode(out_ids, skip_special_tokens=True)}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
