"""Debug why F.scaled_dot_product_attention differs from HF eager."""
import torch
import torch.nn.functional as F

HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, repeat_kv

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

print(f"sa.scaling = {sa.scaling}")
print(f"1/sqrt(HDIM) = {1.0 / (HDIM ** 0.5)}")
print(f"sa.head_dim = {sa.head_dim}")

with torch.no_grad():
    hidden = model.model.embed_tokens(ids)
    ln1 = model.model.layers[0].input_layernorm(hidden)
    position_ids = torch.arange(sl).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    bsz = 1
    q = sa.q_norm(sa.q_proj(ln1).view(bsz, sl, NQH, HDIM)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(ln1).view(bsz, sl, NKVH, HDIM)).transpose(1, 2)
    v = sa.v_proj(ln1).view(bsz, sl, NKVH, HDIM).transpose(1, 2)
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)

    k_exp = repeat_kv(k_r, GQA)
    v_exp = repeat_kv(v, GQA)

    # Method 1: HF eager (explicit scaling + mask)
    scale = sa.scaling
    attn_w = torch.matmul(q_r, k_exp.transpose(2, 3)) * scale
    min_val = torch.finfo(torch.bfloat16).min
    causal = torch.triu(torch.full((sl, sl), min_val, dtype=torch.bfloat16), diagonal=1)
    attn_w = attn_w + causal.unsqueeze(0).unsqueeze(0)
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    sdpa1 = torch.matmul(attn_w, v_exp)

    # Method 2: PyTorch SDPA with is_causal (uses 1/sqrt(d) scaling by default)
    sdpa2 = F.scaled_dot_product_attention(
        q_r.float(), k_exp.float(), v_exp.float(), is_causal=True).to(torch.bfloat16)

    # Method 3: PyTorch SDPA with explicit scale
    sdpa3 = F.scaled_dot_product_attention(
        q_r.float(), k_exp.float(), v_exp.float(), is_causal=True,
        scale=scale).to(torch.bfloat16)

    # Method 4: manual SDPA with float32 mask
    attn_w4 = torch.matmul(q_r.float(), k_exp.float().transpose(2, 3)) * scale
    causal_f32 = torch.triu(torch.full((sl, sl), float('-inf')), diagonal=1)
    attn_w4 = attn_w4 + causal_f32.unsqueeze(0).unsqueeze(0)
    attn_w4 = F.softmax(attn_w4, dim=-1, dtype=torch.float32)
    sdpa4 = torch.matmul(attn_w4, v_exp.float()).to(torch.bfloat16)

print(f"\n1 (HF eager bf16) vs 2 (PT SDPA default): PCC={pcc(sdpa1, sdpa2):.6f}")
print(f"1 (HF eager bf16) vs 3 (PT SDPA scale={scale}): PCC={pcc(sdpa1, sdpa3):.6f}")
print(f"1 (HF eager bf16) vs 4 (manual f32 -inf mask): PCC={pcc(sdpa1, sdpa4):.6f}")
print(f"2 vs 3 (default scale vs explicit): PCC={pcc(sdpa2, sdpa3):.6f}")

# Check: is sa.scaling different from 1/sqrt(d)?
print(f"\n=== Scaling analysis ===")
print(f"sa.scaling = {sa.scaling}")
print(f"1/sqrt(128) = {1/128**0.5}")
print(f"Are they equal? {abs(sa.scaling - 1/128**0.5) < 1e-10}")

# Check the actual attention weights pattern
print(f"\n=== Attention weights analysis ===")
# For first query (position 0), only position 0 should have weight
print(f"HF attn_w[0,0,0,:5]: {attn_w[0,0,0,:5].float()}")
# For last query, all positions should have some weight
print(f"HF attn_w[0,0,-1,:5]: {attn_w[0,0,-1,:5].float()}")
print(f"HF attn_w[0,0,-1,-5:]: {attn_w[0,0,-1,-5:].float()}")
