"""Compare full target forward: HuggingFace reference vs device implementation.

Uses HF transformers with output_hidden_states=True to get per-layer hidden states.
Compares against our device forward to find exactly where divergence occurs.
"""
import time
import json
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open

TILE = 32
HIDDEN = 2048
VOCAB = 151936
TLAYERS = 48
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return (F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    from dflash_device import (load_weights, target_fwd, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, _MESH)
    from rmsnorm import make_rmsnorm_kernel
    from residual_add import residual_add_kernel

    # Tokenize
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    print(f"Prompt: {sl} tokens")

    # HF reference forward
    print("\n=== HF Reference Forward ===")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()
    print(f"HF model loaded in {time.time()-t0:.0f}s")

    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)

    hf_logits = hf_out.logits[0]  # (sl, vocab)
    hf_hidden = hf_out.hidden_states  # tuple of (1, sl, hidden) -- len = TLAYERS+1
    hf_top = torch.argmax(hf_logits[-1, :VOCAB]).item()
    print(f"HF top token: {hf_top} = '{tok.decode([hf_top])}'")
    print(f"HF hidden states: {len(hf_hidden)} layers")

    # Free HF model to save memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # Device forward
    print("\n=== Device Forward ===")
    d = open_dev()
    try:
        w = load_weights(d)

        sp = ((sl + TILE - 1) // TILE) * TILE
        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))

        # Compare embedding
        hf_emb = hf_hidden[0][0, :sl, :HIDDEN].to(torch.bfloat16)
        dev_emb = h_pt[:sl, :HIDDEN]
        print(f"\nEmbedding PCC: {pcc(dev_emb, hf_emb):.6f}")

        # Run device forward layer by layer, comparing after each
        from dflash_device import (dev_norm, dev_attn, dev_add, dev_moe,
                                    ztt, shd, rmsnorm_k, EPS)
        import dflash_device

        h = rep(h_pt, d)
        print(f"\nPer-layer comparison (device vs HF):")
        print(f"{'Layer':>6} {'PCC':>10} {'MaxDiff':>10} {'Status':>8}")
        print("-" * 40)

        first_bad = None
        for li in range(TLAYERS):
            lp = f"t.{li}"

            n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
            attn_out = dev_attn(n, w, lp, sl, sp, d)
            h = dev_add(h, attn_out, sp, d)

            nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
            moe_out = dev_moe(nm, w, lp, sl, sp, d)
            h = dev_add(h, moe_out, sp, d)

            # Compare with HF hidden state after this layer
            dev_h = rb(h)[:sl, :HIDDEN].float()
            hf_h = hf_hidden[li + 1][0, :sl, :HIDDEN].float()
            layer_pcc = pcc(dev_h, hf_h)
            max_diff = (dev_h - hf_h).abs().max().item()
            status = "OK" if layer_pcc > 0.99 else ("WARN" if layer_pcc > 0.9 else "BAD")

            print(f"{li:>6} {layer_pcc:>10.6f} {max_diff:>10.4f} {status:>8}")

            if first_bad is None and layer_pcc < 0.95:
                first_bad = li

        # Final logits
        fn = dev_norm(h, w["final_norm"], sp, w, d)
        logits = ttnn.matmul(fn, w["lm_head"])
        dev_lh = rb_dim1(logits)[:sl, :VOCAB].float()
        dev_top = torch.argmax(dev_lh[-1]).item()

        print(f"\n=== Results ===")
        print(f"HF  top token: {hf_top} = '{tok.decode([hf_top])}'")
        print(f"Dev top token: {dev_top} = '{tok.decode([dev_top])}'")
        print(f"Logit PCC: {pcc(hf_logits[:, :VOCAB].float(), dev_lh):.6f}")
        print(f"Last-pos logit PCC: {pcc(hf_logits[-1, :VOCAB].float(), dev_lh[-1]):.6f}")
        print(f"Top token match: {hf_top == dev_top}")
        if first_bad is not None:
            print(f"First bad layer: {first_bad}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
