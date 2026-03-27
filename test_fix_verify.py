"""Verify the repeat_interleave fix: per-layer PCC through first 4 layers."""
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

    # Grab first 5 layer outputs
    hf_hs = [hf_out.hidden_states[i][0, :sl].to(torch.bfloat16) for i in range(6)]  # 0=emb, 1-5=layers
    hf_logits = hf_out.logits[0, -1]
    hf_token = torch.argmax(hf_logits).item()

    del model
    import gc; gc.collect()

    # Device
    print("Opening device...")
    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(hf_hs[0])  # start from HF embedding
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        print(f"\nsl={sl}, sp={sp}")
        print("=" * 50)

        for li in range(5):
            lp = f"t.{li}"
            n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
            attn_out = dev_attn(n, w, lp, sl, sp, d)
            h = dev_add(h, attn_out, sp, d)
            nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
            moe_out = dev_moe(nm, w, lp, sl, sp, d)
            h = dev_add(h, moe_out, sp, d)

            dev_h = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
            p = pcc(dev_h, hf_hs[li + 1])
            print(f"Layer {li}: PCC={p:.6f}")

        # Check logits
        fn = dev_norm(h, w["final_norm"], sp, w, d)
        logits = ttnn.matmul(fn, w["lm_head"])
        dev_logits = rb_dim1(logits)[:sl]
        dev_token = torch.argmax(dev_logits[-1, :151936].float()).item()

        print(f"\nHF next token: {hf_token} = '{tok.decode([hf_token])}'")
        print(f"Dev next token: {dev_token} = '{tok.decode([dev_token])}'")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
