"""Check if HF self_attn.forward() matches manual reconstruction."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

layer0 = model.model.layers[0]
sa = layer0.self_attn

with torch.no_grad():
    hidden = model.model.embed_tokens(ids)
    ln1 = layer0.input_layernorm(hidden)

    position_ids = torch.arange(sl).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden, position_ids)

    # Method A: HF forward
    attn_a = sa(ln1, attention_mask=None, position_embeddings=pos_emb)[0]

    # Method B: manual reconstruction (same as trace2)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

    bsz = 1
    q = sa.q_norm(sa.q_proj(ln1).view(bsz, sl, NQH, HDIM)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(ln1).view(bsz, sl, NKVH, HDIM)).transpose(1, 2)
    v = sa.v_proj(ln1).view(bsz, sl, NKVH, HDIM).transpose(1, 2)

    cos, sin = pos_emb
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)

    # Eager attention (matching HF's eager_attention_forward)
    k_exp = k_r.repeat(1, GQA, 1, 1)
    v_exp = v.repeat(1, GQA, 1, 1)

    scale = sa.scaling
    print(f"scaling = {scale}")

    attn_w = torch.matmul(q_r, k_exp.transpose(2, 3)) * scale
    causal = torch.triu(torch.full((sl, sl), float('-inf')), diagonal=1)
    attn_w = attn_w + causal.unsqueeze(0).unsqueeze(0)
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    sdpa_out = torch.matmul(attn_w, v_exp)

    attn_b = sa.o_proj(sdpa_out.transpose(1, 2).reshape(bsz, sl, NQH * HDIM))

print(f"Method A (HF forward) [0,:5]: {attn_a[0,0,:5].float()}")
print(f"Method B (manual)     [0,:5]: {attn_b[0,0,:5].float()}")
print(f"A vs B: PCC={pcc(attn_a, attn_b):.6f}  MaxDiff={(attn_a.float()-attn_b.float()).abs().max().item():.6f}")

# Check what HF's eager_attention_forward actually does
import inspect
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
print(f"\nAttention functions: {list(ALL_ATTENTION_FUNCTIONS.keys())}")

# Get the eager forward source
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import eager_attention_forward
    src = inspect.getsource(eager_attention_forward)
    print(f"\neager_attention_forward source:\n{src[:2000]}")
except ImportError:
    # Try the generic one
    from transformers.modeling_utils import eager_attention_forward
    src = inspect.getsource(eager_attention_forward)
    print(f"\neager_attention_forward source:\n{src[:2000]}")
