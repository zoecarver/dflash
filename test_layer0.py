"""Debug layer 0: compare each sub-step against HuggingFace reference."""
import time
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
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd, ztt,
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
    print(f"Prompt: {sl} tokens, padded to {sp}")

    # HF reference: run layer 0 only and capture intermediates
    print("\nLoading HF model...")
    model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
    model.eval()

    # Get HF intermediates by hooking into layer 0
    hf_intermediates = {}

    def make_hook(name):
        def hook(module, args, output):
            # args can be a tuple or single tensor
            if isinstance(args, tuple) and len(args) > 0:
                inp = args[0]
                if isinstance(inp, torch.Tensor):
                    hf_intermediates[f"{name}_in"] = inp.detach().clone()
            if isinstance(output, tuple):
                hf_intermediates[f"{name}_out"] = output[0].detach().clone()
            elif isinstance(output, torch.Tensor):
                hf_intermediates[f"{name}_out"] = output.detach().clone()
        return hook

    layer0 = model.model.layers[0]
    hooks = []
    hooks.append(layer0.input_layernorm.register_forward_hook(make_hook("ln1")))
    hooks.append(layer0.self_attn.register_forward_hook(make_hook("attn")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(make_hook("ln2")))
    hooks.append(layer0.mlp.register_forward_hook(make_hook("moe")))

    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)

    for h in hooks:
        h.remove()

    hf_hs = hf_out.hidden_states
    print(f"HF hidden states collected: {len(hf_hs)} layers")
    print(f"Hooks captured: {list(hf_intermediates.keys())}")

    # Extract HF intermediates
    hf_emb = hf_hs[0][0].to(torch.bfloat16)  # (sl, hidden)
    hf_after_layer0 = hf_hs[1][0].to(torch.bfloat16)

    hf_ln1_out = hf_intermediates["ln1_out"][0].to(torch.bfloat16)
    hf_attn_out = hf_intermediates["attn_out"][0].to(torch.bfloat16)
    hf_post_attn = hf_emb.float() + hf_attn_out.float()  # residual
    hf_ln2_out = hf_intermediates["ln2_out"][0].to(torch.bfloat16)
    hf_moe_out = hf_intermediates["moe_out"][0].to(torch.bfloat16)

    del model
    import gc; gc.collect()

    # Device forward
    print("\nLoading device weights...")
    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))

        print(f"\n{'Step':<25} {'PCC':>10} {'MaxDiff':>10}")
        print("-" * 50)

        # Embedding
        dev_emb = h_pt[:sl, :HIDDEN]
        print(f"{'Embedding':<25} {pcc(dev_emb, hf_emb):>10.6f} {(dev_emb.float()-hf_emb.float()).abs().max().item():>10.4f}")

        h = rep(h_pt, d)
        lp = "t.0"

        # RMSNorm 1
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        dev_ln1 = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"{'RMSNorm 1':<25} {pcc(dev_ln1, hf_ln1_out):>10.6f} {(dev_ln1.float()-hf_ln1_out.float()).abs().max().item():>10.4f}")

        # Attention
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        dev_attn_out = rb(attn_out)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"{'Attention':<25} {pcc(dev_attn_out, hf_attn_out):>10.6f} {(dev_attn_out.float()-hf_attn_out.float()).abs().max().item():>10.4f}")

        # Residual 1
        h = dev_add(h, attn_out, sp, d)
        dev_post_attn = rb(h)[:sl, :HIDDEN].float()
        print(f"{'Residual (h+attn)':<25} {pcc(dev_post_attn, hf_post_attn.float()):>10.6f} {(dev_post_attn-hf_post_attn.float()).abs().max().item():>10.4f}")

        # RMSNorm 2
        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        dev_ln2 = rb(nm)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"{'RMSNorm 2':<25} {pcc(dev_ln2, hf_ln2_out):>10.6f} {(dev_ln2.float()-hf_ln2_out.float()).abs().max().item():>10.4f}")

        # MoE
        moe_out = dev_moe(nm, w, lp, sl, sp, d)
        dev_moe_out = rb(moe_out)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"{'MoE':<25} {pcc(dev_moe_out, hf_moe_out):>10.6f} {(dev_moe_out.float()-hf_moe_out.float()).abs().max().item():>10.4f}")

        # Final residual
        h = dev_add(h, moe_out, sp, d)
        dev_final = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"{'Final (h+moe)':<25} {pcc(dev_final, hf_after_layer0):>10.6f} {(dev_final.float()-hf_after_layer0.float()).abs().max().item():>10.4f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
