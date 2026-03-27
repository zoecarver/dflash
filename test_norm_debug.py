"""Debug RMSNorm: compare TT-Lang kernel vs PyTorch vs HuggingFace."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
EPS = 1e-6
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, _p, dev_norm)
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
    def ln1_hook(module, args, output):
        hf_data["in"] = args[0].detach().clone()
        hf_data["out"] = output.detach().clone()

    layer0 = model.model.layers[0]
    h = layer0.input_layernorm.register_forward_hook(ln1_hook)
    with torch.no_grad():
        model(ids)
    h.remove()

    hf_in = hf_data["in"][0, :sl].to(torch.bfloat16)
    hf_out = hf_data["out"][0, :sl].to(torch.bfloat16)
    ln_weight = layer0.input_layernorm.weight.detach().to(torch.bfloat16)

    del model
    import gc; gc.collect()

    # PyTorch reference RMSNorm
    x = hf_in.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    pt_out = (x * torch.rsqrt(variance + EPS) * ln_weight.float()).to(torch.bfloat16)

    print(f"PyTorch RMSNorm vs HF: PCC={pcc(pt_out, hf_out):.6f}")

    # Device
    d = open_dev()
    try:
        w = load_weights(d)

        # Use HF's exact input (embedding output)
        h_pt = _p(hf_in)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h_tt = rep(h_pt, d)

        n = dev_norm(h_tt, w["t.0.in_w"], sp, w, d)
        dev_out = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"Device RMSNorm vs HF:      PCC={pcc(dev_out, hf_out):.6f}  MaxDiff={(dev_out.float()-hf_out.float()).abs().max().item():.4f}")
        print(f"Device RMSNorm vs PyTorch:  PCC={pcc(dev_out, pt_out):.6f}  MaxDiff={(dev_out.float()-pt_out.float()).abs().max().item():.4f}")

        # Now use the actual embedding from our weights
        h_ours = _p(w["embed_h"][ids.squeeze(0)])
        if h_ours.shape[0] < sp:
            h_ours = F.pad(h_ours, (0, 0, 0, sp - h_ours.shape[0]))
        h_tt2 = rep(h_ours, d)

        n2 = dev_norm(h_tt2, w["t.0.in_w"], sp, w, d)
        dev_out2 = rb(n2)[:sl, :HIDDEN].to(torch.bfloat16)

        # Compare our embedding input vs HF embedding input
        our_emb = h_ours[:sl, :HIDDEN]
        print(f"\nOur embed vs HF embed:     PCC={pcc(our_emb, hf_in):.6f}  MaxDiff={(our_emb.float()-hf_in.float()).abs().max().item():.4f}")
        print(f"Device RMSNorm (our emb) vs HF: PCC={pcc(dev_out2, hf_out):.6f}")

        # Check: what does dev_norm compute vs what we expect?
        # dev_norm uses TT-Lang rmsnorm kernel with mean_scale = 1/HIDDEN
        # Let's verify the mean_scale
        ms_val = 1.0 / HIDDEN
        print(f"\nmean_scale value: {ms_val}")
        print(f"mean_scale * variance = variance/dim (mean)")

        # Manual check: our kernel does rsqrt(sum(x^2) * mean_scale + eps)
        # = rsqrt(mean(x^2) + eps)  -- correct!

        # TTNN RMSNorm comparison
        weight_tt = rep(ln_weight.unsqueeze(0), d)
        try:
            ttnn_norm = ttnn.rms_norm(h_tt, weight=weight_tt, epsilon=EPS)
            ttnn_out = rb(ttnn_norm)[:sl, :HIDDEN].to(torch.bfloat16)
            print(f"\nTTNN RMSNorm vs HF:        PCC={pcc(ttnn_out, hf_out):.6f}  MaxDiff={(ttnn_out.float()-hf_out.float()).abs().max().item():.4f}")
            print(f"TTNN RMSNorm vs PyTorch:    PCC={pcc(ttnn_out, pt_out):.6f}")
        except Exception as e:
            print(f"\nTTNN rms_norm failed: {e}")

        # Check specific values
        print(f"\nHF out[0,:5]:  {hf_out[0,:5].float()}")
        print(f"Dev out[0,:5]: {dev_out[0,:5].float()}")
        print(f"PT  out[0,:5]: {pt_out[0,:5].float()}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
