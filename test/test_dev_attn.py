"""Quick test: run target model via dflash_device.target_fwd and check output tokens."""
import time
import torch
import torch.nn.functional as F

TILE = 32
VOCAB = 151936


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb_dim1, _p, target_fwd)

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
        print(f"Prompt: {ids.shape[0]} tokens")

        max_new = 10
        generated = ids.tolist()

        for step in range(max_new):
            sl = len(generated)
            sp = ((sl + TILE - 1) // TILE) * TILE

            h = _p(emb[torch.tensor(generated)])
            if h.shape[0] < sp:
                h = F.pad(h, (0, 0, 0, sp - h.shape[0]))

            t0 = time.time()
            logits, _ = target_fwd(rep(h, d), w, sl, sp, d)
            el = time.time() - t0

            lh = rb_dim1(logits)[:sl, :VOCAB].float()
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
        close_dev(d)


if __name__ == "__main__":
    main()
