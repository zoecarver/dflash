"""Debug MoE: compare dev_moe against HF MoE on same input."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
EPS = 1e-6
MOE_INTER = 768
NEXPERTS = 128
TOPK = 8
N_CHIPS = 4
EPC = NEXPERTS // N_CHIPS

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd,
                                dev_norm, dev_moe, dev_attn, dev_add)
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
    def moe_hook(module, args, output):
        hf_data["moe_in"] = args[0].detach().clone()
        hf_data["moe_out"] = output.detach().clone()

    layer0 = model.model.layers[0]
    h = layer0.mlp.register_forward_hook(moe_hook)
    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True)
    h.remove()

    hf_moe_in = hf_data["moe_in"][0, :sl].to(torch.bfloat16)
    hf_moe_out = hf_data["moe_out"][0, :sl].to(torch.bfloat16)
    print(f"HF MoE input shape: {hf_moe_in.shape}")
    print(f"HF MoE output shape: {hf_moe_out.shape}")

    del model
    import gc; gc.collect()

    # Device
    d = open_dev()
    try:
        w = load_weights(d)

        # Feed HF's exact MoE input to dev_moe
        moe_in_padded = _p(hf_moe_in)
        if moe_in_padded.shape[0] < sp:
            moe_in_padded = F.pad(moe_in_padded, (0, 0, 0, sp - moe_in_padded.shape[0]))
        moe_in_tt = rep(moe_in_padded, d)

        moe_out_tt = dev_moe(moe_in_tt, w, "t.0", sl, sp, d)
        dev_moe_out = rb(moe_out_tt)[:sl, :HIDDEN].to(torch.bfloat16)

        print(f"\nMoE (HF input -> dev_moe) vs HF MoE output:")
        print(f"  PCC: {pcc(dev_moe_out, hf_moe_out):.6f}")
        print(f"  MaxDiff: {(dev_moe_out.float() - hf_moe_out.float()).abs().max().item():.4f}")
        print(f"  Dev mean: {dev_moe_out.float().abs().mean().item():.6f}")
        print(f"  HF  mean: {hf_moe_out.float().abs().mean().item():.6f}")
        print(f"  Dev[0,:5]: {dev_moe_out[0,:5].float()}")
        print(f"  HF [0,:5]: {hf_moe_out[0,:5].float()}")

        # Also test: dev_moe router scores vs HF
        # Check expert selection overlap
        router_tt = ttnn.matmul(moe_in_tt, w["t.0.rw"])
        scores_tt = ttnn.softmax(router_tt, dim=-1)
        dev_scores = rb(scores_tt)[:sl, :NEXPERTS].float()

        # HF router scores
        from safetensors import safe_open
        import json
        with open(f"{TARGET_DIR}/model.safetensors.index.json") as f:
            idx = json.load(f)
        kf = {k: f"{TARGET_DIR}/{v}" for k, v in idx["weight_map"].items()}
        with safe_open(kf["model.layers.0.mlp.gate.weight"], framework="pt") as f:
            rw = f.get_tensor("model.layers.0.mlp.gate.weight").to(torch.bfloat16)
        hf_scores = F.softmax((hf_moe_in.float() @ rw.float().T), dim=-1)

        print(f"\nRouter scores PCC: {pcc(dev_scores, hf_scores):.6f}")

        # Expert selection
        dev_topk = torch.topk(dev_scores, TOPK, dim=-1)
        hf_topk = torch.topk(hf_scores, TOPK, dim=-1)

        match = 0
        total = sl * TOPK
        for i in range(sl):
            dev_set = set(dev_topk.indices[i].tolist())
            hf_set = set(hf_topk.indices[i].tolist())
            match += len(dev_set & hf_set)
        print(f"Expert selection overlap: {match}/{total} ({match/total*100:.1f}%)")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
