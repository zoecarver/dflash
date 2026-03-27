"""Test if roundtripping tensors through host fixes the chaining issue."""
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

    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0 = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)
    del model
    import gc; gc.collect()

    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        lp = "t.0"

        # Path A: normal chaining (same as before)
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        h_a = ttnn.add(h, attn_out)

        # Read back h_a and check
        h_a_pt = rb(h_a)[:sp, :HIDDEN].to(torch.bfloat16)
        print(f"h after attn (first 5): {h_a_pt[0,:5].float()}")

        nm = dev_norm(h_a, w[f"{lp}.pa_w"], sp, w, d)
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        h_a = ttnn.add(h_a, moe_out)
        dev_l0_a = rb(h_a)[:sl, :HIDDEN].to(torch.bfloat16)

        # Path B: roundtrip through host after each step
        h = rep(h_pt, d)
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        attn_out = dev_attn(n, w, lp, sl, sp, d)

        # Roundtrip: read back both, add on host, re-send
        h_host = rb(h)[:sp, :HIDDEN].to(torch.bfloat16)
        attn_host = rb(attn_out)[:sp, :HIDDEN].to(torch.bfloat16)
        h_sum = (h_host.float() + attn_host.float()).to(torch.bfloat16)
        h_b = rep(_p(h_sum), d)

        print(f"h after attn roundtrip (first 5): {h_sum[0,:5].float()}")

        nm = dev_norm(h_b, w[f"{lp}.pa_w"], sp, w, d)

        # Roundtrip the norm output too
        nm_host = rb(nm)[:sp, :HIDDEN].to(torch.bfloat16)

        moe_out = dev_moe(rep(_p(nm_host), d), w, lp, sl, sp, d)
        moe_host = rb(moe_out)[:sp, :HIDDEN].to(torch.bfloat16)
        h_sum2 = (h_sum.float() + moe_host.float()).to(torch.bfloat16)
        dev_l0_b = h_sum2[:sl, :HIDDEN]

        print(f"\nPath A (direct chain): PCC={pcc(dev_l0_a, hf_l0):.6f}")
        print(f"Path B (host roundtrip): PCC={pcc(dev_l0_b, hf_l0):.6f}")
        print(f"A vs B: PCC={pcc(dev_l0_a, dev_l0_b):.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
