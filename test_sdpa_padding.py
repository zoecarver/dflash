"""Test if zero-padding Q/K/V to sp breaks SDPA causal masking."""
import torch
import torch.nn.functional as F

HDIM = 128
NQH = 32
NKVH = 4
GQA = NQH // NKVH
TILE = 32

def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"

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
sp = ((sl + TILE - 1) // TILE) * TILE

sa = model.model.layers[0].self_attn

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

    # Reference: no padding
    ref = F.scaled_dot_product_attention(
        q_r.float(), k_exp.float(), v_exp.float(), is_causal=True).to(torch.bfloat16)

    # Test 1: pad to sp, run SDPA, slice back
    q_pad = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    k_pad = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    v_pad = torch.zeros(1, NQH, sp, HDIM, dtype=torch.bfloat16)
    q_pad[:, :, :sl] = q_r.to(torch.bfloat16)
    k_pad[:, :, :sl] = k_exp.to(torch.bfloat16)
    v_pad[:, :, :sl] = v_exp.to(torch.bfloat16)

    padded = F.scaled_dot_product_attention(
        q_pad.float(), k_pad.float(), v_pad.float(), is_causal=True)
    padded_sl = padded[:, :, :sl].to(torch.bfloat16)

print(f"sl={sl}, sp={sp}")
print(f"\nRef (no pad) vs Padded[:sl]: PCC={pcc(ref, padded_sl):.6f}  MaxDiff={maxdiff(ref, padded_sl):.6f}")
print(f"Ref[0,0,0,:5]:    {ref[0,0,0,:5].float()}")
print(f"Padded[0,0,0,:5]: {padded_sl[0,0,0,:5].float()}")
print(f"Ref[0,0,-1,:5]:    {ref[0,0,-1,:5].float()}")
print(f"Padded[0,0,-1,:5]: {padded_sl[0,0,-1,:5].float()}")

# The key question: does the zero-padding in K/V get attended to?
# With is_causal=True, position i attends to positions 0..i
# Position 17 (first padded) would attend to 0..17, but position 17's Q is zero.
# More critically: position 16 (last real) attends to 0..16 -- no problem.
# BUT: the zero K vectors at positions 17-31 produce dot products of 0,
# which after softmax might steal probability mass from real positions!
# With is_causal, position 16 sees keys 0..16, so zeros at 17-31 don't matter.
# But wait: the causal mask is based on sp, not sl!
# So position 16 can see positions 0..16 only (causal mask blocks 17+) -- this is fine.
# But what about the softmax normalization? Let me check.

# Actually the concern is: with padding, the causal mask shape is (sp, sp).
# For query position i, it attends to keys 0..i.
# For i < sl: attends to keys 0..i, which are real tokens. Keys at 0..i are all real. No issue.
# For i >= sl: attends to keys 0..i, some of which are zero-padded. But we don't use these outputs.
# So padding should be fine for positions 0..sl-1.

# Let's verify this more carefully:
with torch.no_grad():
    # Manual attention computation with padding to see the softmax weights
    scale = 1.0 / (HDIM ** 0.5)
    scores = torch.matmul(q_pad.float(), k_pad.float().transpose(2, 3)) * scale
    # Causal mask (sp x sp)
    causal = torch.triu(torch.full((sp, sp), float('-inf')), diagonal=1)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    weights = F.softmax(scores, dim=-1)

    # Check: for position 16 (last real), what do the attention weights look like?
    last_real = sl - 1
    print(f"\nAttention weights for position {last_real} (last real):")
    print(f"  Sum: {weights[0,0,last_real,:].sum().item():.6f}")
    print(f"  Weight on real tokens (0..{last_real}): {weights[0,0,last_real,:sl].sum().item():.6f}")
    print(f"  Weight on padded tokens ({sl}..{sp-1}): {weights[0,0,last_real,sl:].sum().item():.6f}")

    # Check position 0
    print(f"\nAttention weights for position 0:")
    print(f"  Weight on pos 0: {weights[0,0,0,0].item():.6f}")
    print(f"  Sum: {weights[0,0,0,:].sum().item():.6f}")
