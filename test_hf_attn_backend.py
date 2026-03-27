"""Check what attention backend HF is using and if it matches our SDPA."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"

tok = AutoTokenizer.from_pretrained(TARGET_DIR)
model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)
model.eval()

print(f"Attn implementation: {model.config._attn_implementation}")

layer0 = model.model.layers[0]
attn = layer0.self_attn
print(f"Attn type: {type(attn)}")
print(f"Attn scaling: {attn.scaling}")

# Check what attention function is being used
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
print(f"Available attention functions: {list(ALL_ATTENTION_FUNCTIONS.keys())}")

# Try forcing eager attention and compare
prompt = "Write a Python function that computes fibonacci numbers."
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt")["input_ids"]

# Run with default backend
with torch.no_grad():
    out1 = model(ids, output_hidden_states=True)
hs1 = out1.hidden_states[1][0].clone()

# Try with eager
model2 = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16,
                                               attn_implementation="eager")
model2.eval()
print(f"\nEager attn implementation: {model2.config._attn_implementation}")
with torch.no_grad():
    out2 = model2(ids, output_hidden_states=True)
hs2 = out2.hidden_states[1][0].clone()

# Compare
def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

print(f"\nDefault vs Eager layer 0: PCC={pcc(hs1, hs2):.6f}  MaxDiff={(hs1.float()-hs2.float()).abs().max().item():.4f}")
print(f"Default[0,:5]: {hs1[0,:5].float()}")
print(f"Eager[0,:5]:   {hs2[0,:5].float()}")

del model, model2
