"""Full 48-layer forward pass + greedy token generation to verify correctness."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
EPS = 1e-6
N_CHIPS = 4
TLAYERS = 48
VOCAB = 151936

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p,
                                dev_norm, dev_attn, dev_add, dev_moe)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    # HF reference
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True, use_cache=False)
    hf_logits = hf_out.logits[0, -1, :VOCAB].float()
    hf_token = torch.argmax(hf_logits).item()
    hf_final = hf_out.hidden_states[-1][0, :sl].to(torch.bfloat16)

    # Spot-check a few layers
    hf_layers = {i: hf_out.hidden_states[i+1][0, :sl].to(torch.bfloat16) for i in [0, 11, 23, 35, 47]}

    del model
    import gc; gc.collect()

    # Device
    print("Opening device...")
    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(hf_out.hidden_states[0][0, :sl].to(torch.bfloat16))
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        print(f"\nsl={sl}, sp={sp}")
        print("Running 48 layers...")

        for li in range(TLAYERS):
            lp = f"t.{li}"
            n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
            attn_out = dev_attn(n, w, lp, sl, sp, d)
            h = dev_add(h, attn_out, sp, d)
            nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
            moe_out = dev_moe(nm, w, lp, sl, sp, d)
            h = dev_add(h, moe_out, sp, d)

            if li in hf_layers:
                dev_h = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
                print(f"  Layer {li}: PCC={pcc(dev_h, hf_layers[li]):.6f}")

            if (li + 1) % 8 == 0:
                print(f"  Progress: {li+1}/{TLAYERS}")

        # Final norm + logits
        fn = dev_norm(h, w["final_norm"], sp, w, d)
        logits = ttnn.matmul(fn, w["lm_head"])
        dev_logits = rb_dim1(logits)[sl-1, :VOCAB].float()
        dev_token = torch.argmax(dev_logits).item()

        # Final hidden state comparison
        dev_final = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"\nFinal hidden: PCC={pcc(dev_final, hf_final):.6f}")

        print(f"\n=== Token prediction ===")
        print(f"HF:  '{tok.decode([hf_token])}' (id={hf_token})")
        print(f"Dev: '{tok.decode([dev_token])}' (id={dev_token})")
        print(f"Match: {hf_token == dev_token}")

        # Top-5 comparison
        hf_top5 = torch.topk(hf_logits, 5)
        dev_top5 = torch.topk(dev_logits, 5)
        print(f"\nHF top-5:  {[(tok.decode([t.item()]), f'{v.item():.2f}') for t, v in zip(hf_top5.indices, hf_top5.values)]}")
        print(f"Dev top-5: {[(tok.decode([t.item()]), f'{v.item():.2f}') for t, v in zip(dev_top5.indices, dev_top5.values)]}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
