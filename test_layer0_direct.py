"""Direct layer 0 test: no hooks, just compare hidden_states[0] vs hidden_states[1]."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
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

    # HF: just get hidden_states[0] and hidden_states[1]
    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0 = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)
    hf_l1 = hf_out.hidden_states[2][0, :sl].to(torch.bfloat16)
    hf_logits = hf_out.logits[0, :sl]

    del model
    import gc; gc.collect()

    # Device: run layer 0 from HF's exact embedding
    d = open_dev()
    try:
        w = load_weights(d)

        # Start from HF's embedding (guaranteed match)
        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        lp = "t.0"
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        h = dev_add(h, attn_out, sp, d)
        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        h = dev_add(h, moe_out, sp, d)

        dev_l0 = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"Layer 0 output (dev vs HF): PCC={pcc(dev_l0, hf_l0):.6f}  MaxDiff={(dev_l0.float()-hf_l0.float()).abs().max().item():.4f}")
        print(f"Dev[0,:5]: {dev_l0[0,:5].float()}")
        print(f"HF [0,:5]: {hf_l0[0,:5].float()}")

        # Continue to layer 1
        lp = "t.1"
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        h = dev_add(h, attn_out, sp, d)
        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        h = dev_add(h, moe_out, sp, d)

        dev_l1 = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Layer 1 output (dev vs HF): PCC={pcc(dev_l1, hf_l1):.6f}  MaxDiff={(dev_l1.float()-hf_l1.float()).abs().max().item():.4f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
