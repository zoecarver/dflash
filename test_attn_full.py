"""Compare dev_attn output directly against HF attention output.
Run dev_attn exactly as target_fwd does, compare against HF hook."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
HDIM = 128
NQH = 32
NKVH = 4
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, dev_norm, dev_attn)
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
    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()

    hf_data = {}
    def attn_hook(module, args, output):
        hf_data["attn_out"] = output[0].detach().clone()
    def ln1_hook(module, args, output):
        hf_data["ln1_out"] = output.detach().clone()

    layer0 = model.model.layers[0]
    h1 = layer0.input_layernorm.register_forward_hook(ln1_hook)
    h2 = layer0.self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        model(ids, output_hidden_states=True)
    h1.remove(); h2.remove()

    hf_ln1 = hf_data["ln1_out"][0, :sl].to(torch.bfloat16)
    hf_attn = hf_data["attn_out"][0, :sl].to(torch.bfloat16)
    print(f"HF ln1 shape: {hf_ln1.shape}, attn shape: {hf_attn.shape}")

    del model
    import gc; gc.collect()

    # Device
    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        # RMSNorm
        n = dev_norm(h, w["t.0.in_w"], sp, w, d)
        dev_ln1 = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"RMSNorm PCC: {pcc(dev_ln1, hf_ln1):.6f}")

        # Now feed HF's LN output through our attention to isolate the attn logic
        hf_ln1_padded = _p(hf_ln1)
        if hf_ln1_padded.shape[0] < sp:
            hf_ln1_padded = F.pad(hf_ln1_padded, (0, 0, 0, sp - hf_ln1_padded.shape[0]))
        n_from_hf = rep(hf_ln1_padded, d)

        # Run dev_attn with HF's norm output
        attn_hf_input = dev_attn(n_from_hf, w, "t.0", sl, sp, d)
        dev_attn_hf = rb(attn_hf_input)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Attn (HF ln1 input) PCC: {pcc(dev_attn_hf, hf_attn):.6f}")
        print(f"Attn (HF ln1 input) MaxDiff: {(dev_attn_hf.float()-hf_attn.float()).abs().max().item():.4f}")

        # Run dev_attn with our norm output
        attn_dev_input = dev_attn(n, w, "t.0", sl, sp, d)
        dev_attn_dev = rb(attn_dev_input)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"Attn (dev ln1 input) PCC: {pcc(dev_attn_dev, hf_attn):.6f}")
        print(f"Attn (dev ln1 input) MaxDiff: {(dev_attn_dev.float()-hf_attn.float()).abs().max().item():.4f}")

        # Check: are the two dev_attn outputs different?
        print(f"Attn (hf vs dev input) PCC: {pcc(dev_attn_hf, dev_attn_dev):.6f}")

        # Debug: print some actual values
        print(f"\nHF attn[0,:5]:  {hf_attn[0,:5].float()}")
        print(f"Dev attn[0,:5]: {dev_attn_hf[0,:5].float()}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
