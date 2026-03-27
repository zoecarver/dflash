"""Find exactly where HF sa.forward() differs from manual reconstruction."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, repeat_kv

HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

model = AutoModelForCausalLM.from_pretrained(
    TARGET_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()

tok = AutoTokenizer.from_pretrained(TARGET_DIR)
prompt = "Write a Python function that computes fibonacci numbers."
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt")["input_ids"]
sl = ids.shape[1]

sa = model.model.layers[0].self_attn

# Print sa attributes
print(f"sa attrs with 'head': {[a for a in dir(sa) if 'head' in a.lower()]}")
print(f"sa attrs with 'num': {[a for a in dir(sa) if 'num' in a.lower()]}")

with torch.no_grad():
    hidden = model.model.embed_tokens(ids)
    ln1 = model.model.layers[0].input_layernorm(hidden)

    position_ids = torch.arange(sl).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden, position_ids)
    cos, sin = pos_emb

    # Build causal mask
    min_val = torch.finfo(torch.bfloat16).min
    causal_mask = torch.triu(torch.full((sl, sl), min_val, dtype=torch.bfloat16), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # === Method A: HF forward with causal mask ===
    attn_a = sa(ln1, attention_mask=causal_mask, position_embeddings=pos_emb)[0]

    # === Method B: Manual with causal mask (same approach as device code) ===
    bsz = 1
    q_proj_out = sa.q_proj(ln1)
    k_proj_out = sa.k_proj(ln1)
    v_proj_out = sa.v_proj(ln1)

    q = sa.q_norm(q_proj_out.view(bsz, sl, NQH, HDIM)).transpose(1, 2)
    k = sa.k_norm(k_proj_out.view(bsz, sl, NKVH, HDIM)).transpose(1, 2)
    v = v_proj_out.view(bsz, sl, NKVH, HDIM).transpose(1, 2)

    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)

    k_exp = repeat_kv(k_r, GQA)
    v_exp = repeat_kv(v, GQA)
    scale = sa.scaling
    attn_w = torch.matmul(q_r, k_exp.transpose(2, 3)) * scale
    attn_w = attn_w + causal_mask
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    sdpa_out = torch.matmul(attn_w, v_exp)
    attn_b = sa.o_proj(sdpa_out.transpose(1, 2).reshape(bsz, sl, NQH * HDIM))

    # === Method C: Manual WITHOUT causal mask (no mask at all) ===
    attn_w_nc = torch.matmul(q_r, k_exp.transpose(2, 3)) * scale
    attn_w_nc = F.softmax(attn_w_nc, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    sdpa_out_nc = torch.matmul(attn_w_nc, v_exp)
    attn_c = sa.o_proj(sdpa_out_nc.transpose(1, 2).reshape(bsz, sl, NQH * HDIM))

print(f"\n=== Results ===")
print(f"A (HF forward + mask)    [0,0,:5]: {attn_a[0,0,:5].float()}")
print(f"B (manual + mask)        [0,0,:5]: {attn_b[0,0,:5].float()}")
print(f"C (manual, no mask)      [0,0,:5]: {attn_c[0,0,:5].float()}")
print(f"\nA vs B (both have mask): PCC={pcc(attn_a, attn_b):.6f}")
print(f"A vs C (A mask, C none): PCC={pcc(attn_a, attn_c):.6f}")
print(f"B vs C (B mask, C none): PCC={pcc(attn_b, attn_c):.6f}")

# Does HF forward use DynamicCache that modifies K/V?
print(f"\n=== Debug: does HF forward modify K/V via cache? ===")
# Run again with use_cache=False by intercepting
print(f"Full model config use_cache: {model.config.use_cache}")
