"""Figure out what causal mask HF actually uses in the full forward."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect

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

# Hook self_attn to capture the attention_mask it receives
captured = {}
def capture_attn_inputs(mod, args, kwargs):
    captured["args"] = args
    captured["kwargs"] = kwargs
    return args, kwargs

layer0 = model.model.layers[0]
layer0.self_attn.register_forward_pre_hook(capture_attn_inputs, with_kwargs=True)

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

# What did the attention layer receive?
print(f"Args count: {len(captured['args'])}")
for i, a in enumerate(captured['args']):
    if torch.is_tensor(a):
        print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
    else:
        print(f"  arg[{i}]: {type(a)} = {a}")

print(f"\nKwargs keys: {list(captured['kwargs'].keys())}")
for k, v in captured['kwargs'].items():
    if torch.is_tensor(v):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, tuple):
        print(f"  {k}: tuple of {len(v)}")
        for i, item in enumerate(v):
            if torch.is_tensor(item):
                print(f"    [{i}]: shape={item.shape}")
    else:
        print(f"  {k}: {v}")

if "attention_mask" in captured["kwargs"]:
    mask = captured["kwargs"]["attention_mask"]
    if mask is not None:
        print(f"\nAttention mask shape: {mask.shape}")
        print(f"Attention mask dtype: {mask.dtype}")
        print(f"Attention mask min: {mask.min()}")
        print(f"Attention mask max: {mask.max()}")
        print(f"Attention mask unique values: {mask.unique()[:10]}")
        # Is it causal?
        if mask.dim() >= 2:
            print(f"Mask [0,0,:5,:5]:\n{mask[0,0,:5,:5] if mask.dim()==4 else mask[0,:5,:5]}")
    else:
        print("\nAttention mask is None!")

# Also check: what does the model forward actually do?
print("\n\nQwen3MoeModel.forward source (relevant mask section):")
src = inspect.getsource(type(model.model).forward)
# Find the part about causal_mask
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'mask' in line.lower() or 'causal' in line.lower():
        print(f"  L{i}: {line}")
