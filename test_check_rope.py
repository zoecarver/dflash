"""Check HF Qwen3 RoPE: compare cos/sin values and application method."""
import torch
from transformers import AutoModelForCausalLM
import inspect

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.bfloat16)

rotary = model.model.rotary_emb
print(f"Rotary type: {type(rotary)}")
print(f"Rotary source: {inspect.getfile(type(rotary))}")

# Get cos/sin
pos_ids = torch.arange(17).unsqueeze(0)
cos, sin = rotary(pos_ids)
print(f"\nHF cos shape: {cos.shape}")
print(f"HF sin shape: {sin.shape}")
print(f"HF cos[0, :8]: {cos[0, :8]}")
print(f"HF sin[0, :8]: {sin[0, :8]}")
print(f"HF cos[1, :8]: {cos[1, :8]}")

# Our precomputation
HDIM = 128
ROPE_THETA = 1e7
freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
pos = torch.arange(17, dtype=torch.float32)
angles = torch.outer(pos, freqs)
our_cos = torch.cos(angles).to(torch.bfloat16)
our_sin = torch.sin(angles).to(torch.bfloat16)

print(f"\nOur cos shape: {our_cos.shape}")  # (17, 64)
print(f"Our cos[0, :8]: {our_cos[0, :8]}")
print(f"Our cos[1, :8]: {our_cos[1, :8]}")

# Compare
hf_c = cos.float()
if hf_c.shape[-1] == HDIM:
    print(f"\nHF returns full HDIM={HDIM} cos/sin")
    # Check if first half == second half (i.e. it's just repeat)
    print(f"First half == second half: {torch.allclose(hf_c[:, :HDIM//2], hf_c[:, HDIM//2:])}")
    hf_half = hf_c[:, :HDIM//2]
elif hf_c.shape[-1] == HDIM // 2:
    print(f"\nHF returns half HDIM={HDIM//2} cos/sin")
    hf_half = hf_c

our_c = our_cos.float()
print(f"\nHF half cos[0, :8]: {hf_half[0, :8]}")
print(f"Our cos[0, :8]:     {our_c[0, :8]}")
max_diff = (hf_half - our_c).abs().max().item()
print(f"Max diff: {max_diff}")
print(f"Match: {max_diff < 0.01}")

# Also check: print the apply_rotary_pos_emb function
attn_module = model.model.layers[0].self_attn
print(f"\nAttn forward source file: {inspect.getfile(type(attn_module))}")

# Find apply_rotary_pos_emb
modeling_file = inspect.getfile(type(model))
print(f"Model source: {modeling_file}")

# Read the source and find apply_rotary_pos_emb
with open(modeling_file) as f:
    src = f.read()

# Find the function
import re
match = re.search(r'def apply_rotary_pos_emb.*?(?=\ndef |\nclass )', src, re.DOTALL)
if match:
    print(f"\napply_rotary_pos_emb:\n{match.group()}")
else:
    # Try rotating_half
    match = re.search(r'def rotate_half.*?(?=\ndef |\nclass )', src, re.DOTALL)
    if match:
        print(f"\nrotate_half:\n{match.group()}")
    print("Could not find apply_rotary_pos_emb in model source")
    # Search in parent modules
    for name in ['rotate_half', 'apply_rotary']:
        for line_no, line in enumerate(src.split('\n')):
            if name in line:
                print(f"  L{line_no}: {line.strip()}")

del model
