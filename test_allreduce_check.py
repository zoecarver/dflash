"""Check if all_reduce output is actually identical across chips."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN = 2048
N_CHIPS = 4

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    from dflash_device import (load_weights, open_dev, close_dev,
                                rep, rb, rb_dim1, _p,
                                dev_norm, dev_attn)
    import dflash_device
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    sl = ids.shape[1]
    sp = ((sl + TILE - 1) // TILE) * TILE

    d = open_dev()
    try:
        w = load_weights(d)

        h_pt = _p(w["embed_h"][ids.squeeze(0)])
        if h_pt.shape[0] < sp:
            h_pt = F.pad(h_pt, (0, 0, 0, sp - h_pt.shape[0]))
        h = rep(h_pt, d)

        n = dev_norm(h, w["t.0.in_w"], sp, w, d)
        attn_out = dev_attn(n, w, "t.0", sl, sp, d)

        # Read back attn_out from all chips via dim=0 concat
        all_chips = ttnn.to_torch(attn_out, mesh_composer=ttnn.ConcatMeshToTensor(dflash_device._MESH, dim=0))
        print(f"all_chips shape: {all_chips.shape}")  # should be (4*sp, HIDDEN) if replicated

        for c in range(N_CHIPS):
            chip_data = all_chips[c*sp:(c+1)*sp, :HIDDEN]
            print(f"  Chip {c} [0,:5]: {chip_data[0,:5].float()}")

        # Check if chips are identical
        chip0 = all_chips[0:sp, :HIDDEN].float()
        for c in range(1, N_CHIPS):
            chipc = all_chips[c*sp:(c+1)*sp, :HIDDEN].float()
            diff = (chip0 - chipc).abs().max().item()
            p = pcc(chip0[:sl], chipc[:sl])
            print(f"  Chip 0 vs Chip {c}: MaxDiff={diff:.6f}  PCC={p:.6f}")

        # Also check: h (the input) from all chips
        h_all = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(dflash_device._MESH, dim=0))
        print(f"\nh (input) all_chips shape: {h_all.shape}")
        h0 = h_all[0:sp, :HIDDEN].float()
        for c in range(1, N_CHIPS):
            hc = h_all[c*sp:(c+1)*sp, :HIDDEN].float()
            diff = (h0 - hc).abs().max().item()
            print(f"  h Chip 0 vs Chip {c}: MaxDiff={diff:.6f}")

        # Check n (norm output) from all chips
        n_all = ttnn.to_torch(n, mesh_composer=ttnn.ConcatMeshToTensor(dflash_device._MESH, dim=0))
        print(f"\nn (norm) all_chips shape: {n_all.shape}")
        n0 = n_all[0:sp, :HIDDEN].float()
        for c in range(1, N_CHIPS):
            nc = n_all[c*sp:(c+1)*sp, :HIDDEN].float()
            diff = (n0 - nc).abs().max().item()
            print(f"  n Chip 0 vs Chip {c}: MaxDiff={diff:.6f}")

        # Now: what does ttnn.add give vs host add?
        h_plus_attn = ttnn.add(h, attn_out)
        hpa_all = ttnn.to_torch(h_plus_attn, mesh_composer=ttnn.ConcatMeshToTensor(dflash_device._MESH, dim=0))
        print(f"\nh+attn all_chips shape: {hpa_all.shape}")
        hpa0 = hpa_all[0:sp, :HIDDEN].float()
        for c in range(1, N_CHIPS):
            hpac = hpa_all[c*sp:(c+1)*sp, :HIDDEN].float()
            diff = (hpa0 - hpac).abs().max().item()
            print(f"  h+attn Chip 0 vs Chip {c}: MaxDiff={diff:.6f}")

        # Host add reference
        host_add = (h_all[0:sp, :HIDDEN].float() + all_chips[0:sp, :HIDDEN].float())
        print(f"\n  h+attn device [0,:5]: {hpa0[0,:5]}")
        print(f"  h+attn host   [0,:5]: {host_add[0,:5]}")
        print(f"  h+attn dev vs host: MaxDiff={(hpa0[:sl]-host_add[:sl]).abs().max().item():.6f}")

    finally:
        close_dev(d)


if __name__ == "__main__":
    main()
