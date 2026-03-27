"""Inspect HF Qwen3 attention forward to see how RoPE is applied."""
from transformers import AutoModelForCausalLM
import torch, inspect

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/qwen-coder-30b-a3b/weights",
    torch_dtype=torch.bfloat16, attn_implementation="eager")
sa = model.model.layers[0].self_attn
print("Type:", type(sa))
print("Attrs:", [a for a in dir(sa) if 'rot' in a.lower()])
src = inspect.getsource(type(sa).forward)
print(src[:4000])
