"""Test: target model with tracing mode and scratch preallocation."""
import time
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
VOCAB = 151936
HIDDEN = 2048


def main():
    from dflash_device import (load_weights, open_dev, close_dev, _p,
                                rb_dim1, rb,
                                prealloc_target_scratch,
                                capture_target_trace,
                                execute_target_trace)

    d = open_dev()
    try:
        w = load_weights(d)

        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("/workspace/qwen-coder-30b-a3b/weights")
            prompt = "Write a Python function that computes fibonacci numbers."
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
        except Exception as e:
            print(f"Tokenizer: {e}")
            ids = torch.tensor([151643, 872, 13, 5765, 264, 13325])

        emb = w["embed_h"]
        max_new = 10
        prompt_len = ids.shape[0]

        # Fixed sp for tracing
        max_ctx = prompt_len + max_new
        sp = ((max_ctx + TILE - 1) // TILE) * TILE
        print(f"Prompt: {prompt_len} tokens, fixed sp={sp} for tracing")

        t0 = time.time()
        scr = prealloc_target_scratch(sp, w, d)
        capture_target_trace(scr, w, d)
        print(f"Prealloc + compile + trace: {time.time()-t0:.1f}s")

        generated = ids.tolist()
        for step in range(max_new):
            sl = len(generated)

            t_emb = time.time()
            h = _p(emb[torch.tensor(generated)])
            if h.shape[0] < sp:
                h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
            h_host = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            t_emb = time.time() - t_emb

            t_fwd = time.time()
            logits = execute_target_trace(scr, h_host, d)
            t_fwd = time.time() - t_fwd

            t_tok = time.time()
            # Device-side argmax, then read back just the indices (not full logits)
            argmax_ids = ttnn.argmax(logits, dim=-1)
            argmax_host = rb(argmax_ids)
            next_tok = int(argmax_host[sl - 1].item())
            t_tok = time.time() - t_tok

            generated.append(next_tok)

            try:
                tok_str = tok.decode([next_tok])
            except:
                tok_str = f"<{next_tok}>"
            print(f"  step {step}: tok={next_tok} '{tok_str}' "
                  f"(emb={t_emb:.3f}s fwd={t_fwd:.3f}s tok={t_tok:.3f}s ctx={sl})")

            if next_tok in (151643, 151645):
                break

        print(f"\n--- Output ---")
        try:
            print(tok.decode(generated, skip_special_tokens=True))
        except:
            print(generated)

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
