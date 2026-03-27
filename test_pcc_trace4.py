"""Trace full pipeline with HF hooks on EVERY submodule output."""
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
ROPE_THETA = 1e7
N_CHIPS = 4
NEXPERTS = 128
TOPK = 8
EPC = NEXPERTS // N_CHIPS
MOE_INTER = 768

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p, shd,
                                dev_norm, dev_attn, dev_add, dev_moe)
    import dflash_device
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    # HF: run layer 0 manually step by step to get exact intermediates
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    layer0 = model.model.layers[0]
    sa = layer0.self_attn

    with torch.no_grad():
        # Get full model output for ground truth
        hf_out = model(ids, output_hidden_states=True)
        hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
        hf_l0_out = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)

        # Now run layer 0 manually
        hidden = model.model.embed_tokens(ids)  # (1, sl, HIDDEN)

        # Step 1: input layernorm
        ln1 = layer0.input_layernorm(hidden)
        hf_ln1 = ln1[0, :sl].to(torch.bfloat16)

        # Step 2: self_attn (full forward with proper causal mask)
        position_ids = torch.arange(sl).unsqueeze(0)
        pos_emb = model.model.rotary_emb(hidden, position_ids)
        # Build causal mask: (1, 1, sl, sl) with -inf upper triangle
        min_val = torch.finfo(torch.bfloat16).min
        causal_mask = torch.triu(torch.full((sl, sl), min_val, dtype=torch.bfloat16), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_out_tuple = sa(ln1, attention_mask=causal_mask, position_embeddings=pos_emb)
        hf_attn_out = attn_out_tuple[0]  # (1, sl, HIDDEN)
        hf_attn_flat = hf_attn_out[0, :sl].to(torch.bfloat16)

        # Step 3: residual add
        hf_resid1 = hidden + hf_attn_out
        hf_resid1_flat = hf_resid1[0, :sl].to(torch.bfloat16)

        # Step 4: post_attention_layernorm
        ln2 = layer0.post_attention_layernorm(hf_resid1)
        hf_ln2 = ln2[0, :sl].to(torch.bfloat16)

        # Step 5: MoE
        moe_out = layer0.mlp(ln2)
        hf_moe_out = moe_out[0, :sl].to(torch.bfloat16)

        # Step 6: residual add
        hf_final = hf_resid1 + moe_out
        hf_final_flat = hf_final[0, :sl].to(torch.bfloat16)

    # Verify our manual matches hidden_states[1]
    print(f"Manual final vs hidden_states[1]: PCC={pcc(hf_final_flat, hf_l0_out):.6f}")

    del model
    import gc; gc.collect()

    # Device pipeline
    print("\nOpening device...")
    d = open_dev()
    try:
        w = load_weights(d)
        lp = "t.0"

        h_pt = _p(hf_emb)
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        print("\n" + "=" * 70)
        print("FULL PIPELINE TRACE")
        print("=" * 70)

        # Step 1: Norm
        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        dev_n = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"1. Norm1:       PCC={pcc(dev_n, hf_ln1):.6f}  MaxDiff={maxdiff(dev_n, hf_ln1):.6f}")

        # Step 2: Attention
        attn_out = dev_attn(n, w, lp, sl, sp, d)
        dev_attn_out = rb(attn_out)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"2. Attn out:    PCC={pcc(dev_attn_out, hf_attn_flat):.6f}  MaxDiff={maxdiff(dev_attn_out, hf_attn_flat):.6f}")

        # Step 3: Residual add
        h = dev_add(h, attn_out, sp, d)
        dev_resid1 = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"3. Resid1:      PCC={pcc(dev_resid1, hf_resid1_flat):.6f}  MaxDiff={maxdiff(dev_resid1, hf_resid1_flat):.6f}")

        # Step 4: Norm2
        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        dev_nm = rb(nm)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"4. Norm2:       PCC={pcc(dev_nm, hf_ln2):.6f}  MaxDiff={maxdiff(dev_nm, hf_ln2):.6f}")

        # Step 5: MoE
        moe = dev_moe(nm, w, lp, sl, sp, d)
        dev_moe_out = rb(moe)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"5. MoE out:     PCC={pcc(dev_moe_out, hf_moe_out):.6f}  MaxDiff={maxdiff(dev_moe_out, hf_moe_out):.6f}")

        # Step 6: Final residual
        h = dev_add(h, moe, sp, d)
        dev_final = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"6. Final:       PCC={pcc(dev_final, hf_final_flat):.6f}  MaxDiff={maxdiff(dev_final, hf_final_flat):.6f}")
        print(f"   vs h_s[1]:   PCC={pcc(dev_final, hf_l0_out):.6f}")

        # Detailed values
        print(f"\n--- Values at [0,:5] ---")
        print(f"HF attn:  {hf_attn_flat[0,:5].float()}")
        print(f"Dev attn: {dev_attn_out[0,:5].float()}")
        print(f"HF resid1:  {hf_resid1_flat[0,:5].float()}")
        print(f"Dev resid1: {dev_resid1[0,:5].float()}")
        print(f"HF moe:   {hf_moe_out[0,:5].float()}")
        print(f"Dev moe:  {dev_moe_out[0,:5].float()}")
        print(f"HF final: {hf_final_flat[0,:5].float()}")
        print(f"Dev final:{dev_final[0,:5].float()}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
