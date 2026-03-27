"""Final trace: use_cache=False to avoid KV cache contamination."""
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
    print(f"sl={sl}, sp={sp}")

    # HF reference with use_cache=False
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    with torch.no_grad():
        hf_out = model(ids, output_hidden_states=True, use_cache=False)
    hf_emb = hf_out.hidden_states[0][0, :sl].to(torch.bfloat16)
    hf_l0 = hf_out.hidden_states[1][0, :sl].to(torch.bfloat16)
    hf_l1 = hf_out.hidden_states[2][0, :sl].to(torch.bfloat16)

    # Now run layer 0 manually with use_cache=False
    layer0 = model.model.layers[0]
    sa = layer0.self_attn

    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)
        ln1 = layer0.input_layernorm(hidden)
        position_ids = torch.arange(sl).unsqueeze(0)
        pos_emb = model.model.rotary_emb(hidden, position_ids)

        # Build causal mask
        min_val = torch.finfo(torch.bfloat16).min
        causal_mask = torch.triu(torch.full((sl, sl), min_val, dtype=torch.bfloat16), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Call sa with NO cache
        attn_out = sa(ln1, attention_mask=causal_mask, position_embeddings=pos_emb,
                      past_key_values=None, use_cache=False)[0]
        hf_attn = attn_out[0, :sl].to(torch.bfloat16)
        hf_resid1 = (hidden + attn_out)[0, :sl].to(torch.bfloat16)

        ln2 = layer0.post_attention_layernorm(hidden + attn_out)
        hf_ln2 = ln2[0, :sl].to(torch.bfloat16)
        moe_out = layer0.mlp(ln2)
        hf_moe = moe_out[0, :sl].to(torch.bfloat16)
        hf_final = (hidden + attn_out + moe_out)[0, :sl].to(torch.bfloat16)

    print(f"Manual final vs h_s[1]: PCC={pcc(hf_final, hf_l0):.6f}")

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
        print("FULL PIPELINE TRACE (use_cache=False)")
        print("=" * 70)

        n = dev_norm(h, w[f"{lp}.in_w"], sp, w, d)
        dev_n = rb(n)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"1. Norm1:    PCC={pcc(dev_n, ln1[0,:sl].to(torch.bfloat16)):.6f}")

        attn_out_dev = dev_attn(n, w, lp, sl, sp, d)
        dev_attn_out = rb(attn_out_dev)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"2. Attn:     PCC={pcc(dev_attn_out, hf_attn):.6f}  MaxDiff={maxdiff(dev_attn_out, hf_attn):.6f}")

        h = dev_add(h, attn_out_dev, sp, d)
        dev_resid1 = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"3. Resid1:   PCC={pcc(dev_resid1, hf_resid1):.6f}")

        nm = dev_norm(h, w[f"{lp}.pa_w"], sp, w, d)
        dev_nm = rb(nm)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"4. Norm2:    PCC={pcc(dev_nm, hf_ln2):.6f}")

        moe_dev = dev_moe(nm, w, lp, sl, sp, d)
        dev_moe_out = rb(moe_dev)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"5. MoE:      PCC={pcc(dev_moe_out, hf_moe):.6f}  MaxDiff={maxdiff(dev_moe_out, hf_moe):.6f}")

        h = dev_add(h, moe_dev, sp, d)
        dev_final = rb(h)[:sl, :HIDDEN].to(torch.bfloat16)
        print(f"6. L0 out:   PCC={pcc(dev_final, hf_final):.6f}")
        print(f"   vs h_s[1]:PCC={pcc(dev_final, hf_l0):.6f}")

        print(f"\n--- Values ---")
        print(f"HF attn[0,:5]:  {hf_attn[0,:5].float()}")
        print(f"Dev attn[0,:5]: {dev_attn_out[0,:5].float()}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
