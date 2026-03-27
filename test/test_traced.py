"""Test: target model with tracing mode -- zero host ops in hot loop."""
import time
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
VOCAB = 151936
HIDDEN = 2048


def main():
    from dflash_device import (load_weights, open_dev, close_dev, _p,
                                rb_dim1, target_fwd, rep,
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

        # Fixed sp for tracing: pad to max possible context length
        max_ctx = prompt_len + max_new
        sp = ((max_ctx + TILE - 1) // TILE) * TILE
        print(f"Prompt: {prompt_len} tokens, fixed sp={sp} for tracing")

        # Phase 4: preallocate + compile + trace capture
        scr = prealloc_target_scratch(sp, w, d)
        capture_target_trace(scr, w, d)

        # Non-traced baseline for comparison
        generated_baseline = ids.tolist()
        h = _p(emb[torch.tensor(generated_baseline)])
        if h.shape[0] < sp:
            h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
        t0 = time.time()
        logits, _ = target_fwd(rep(h, d), w, prompt_len, sp, d)
        baseline_time = time.time() - t0
        lh = rb_dim1(logits)[:prompt_len, :VOCAB].float()
        baseline_tok = torch.argmax(lh[-1]).item()
        print(f"Baseline (no trace): tok={baseline_tok} ({baseline_time:.2f}s)")

        # Traced generation
        generated = ids.tolist()
        for step in range(max_new):
            sl = len(generated)

            h = _p(emb[torch.tensor(generated)])
            if h.shape[0] < sp:
                h = F.pad(h, (0, 0, 0, sp - h.shape[0]))
            # Prepare host tensor for copy_host_to_device_tensor
            h_host = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            t0 = time.time()
            logits = execute_target_trace(scr, h_host, d)
            el = time.time() - t0

            lh = rb_dim1(logits)[:sl, :VOCAB].float()
            next_tok = torch.argmax(lh[-1]).item()
            generated.append(next_tok)

            try:
                tok_str = tok.decode([next_tok])
            except:
                tok_str = f"<{next_tok}>"
            print(f"  step {step}: tok={next_tok} '{tok_str}' ({el:.3f}s, ctx={sl})")

            if next_tok in (151643, 151645):
                break

        print(f"\n--- Output ---")
        try:
            print(tok.decode(generated, skip_special_tokens=True))
        except:
            print(generated)

        # Verify first token matches baseline
        if generated[prompt_len] == baseline_tok:
            print(f"\nTrace matches baseline: first token {baseline_tok}")
        else:
            print(f"\nMISMATCH: trace={generated[prompt_len]} baseline={baseline_tok}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
