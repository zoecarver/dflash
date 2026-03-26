"""Test MoE expert computation on device using TTNN matmuls + silu_mul kernel."""
import torch
import ttnn
from silu_mul import silu_mul_kernel

TILE = 32
HIDDEN_SIZE = 2048
MOE_INTER = 768
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2


def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pad_to_tile(t, dim=-1):
    size = t.shape[dim]
    pad_size = (TILE - size % TILE) % TILE
    if pad_size == 0:
        return t
    pad_spec = [0] * (2 * len(t.shape))
    pad_spec[-(2 * (dim % len(t.shape)) + 1)] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


def test_moe_on_device(device, seq_len=32):
    torch.manual_seed(42)
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    hidden = torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    router_w = torch.randn(HIDDEN_SIZE, NUM_EXPERTS, dtype=torch.bfloat16) * 0.01
    expert_gate = [torch.randn(HIDDEN_SIZE, MOE_INTER, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]
    expert_up = [torch.randn(HIDDEN_SIZE, MOE_INTER, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]
    expert_down = [torch.randn(MOE_INTER, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01 for _ in range(NUM_EXPERTS)]

    # PyTorch reference
    router_logits = hidden.float() @ router_w.float()
    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    moe_ref = torch.zeros(seq_len, HIDDEN_SIZE)
    for tok in range(seq_len):
        for k in range(NUM_EXPERTS_PER_TOK):
            eidx = topk_indices[tok, k].item()
            w = topk_weights[tok, k].item()
            inp = hidden[tok:tok+1].float()
            g = inp @ expert_gate[eidx].float()
            u = inp @ expert_up[eidx].float()
            activated = torch.nn.functional.silu(g) * u
            moe_ref[tok] += (activated @ expert_down[eidx].float()).squeeze(0) * w
    moe_ref = moe_ref.to(torch.bfloat16)

    # Device implementation: per-expert batched matmul
    hidden_padded = pad_to_tile(hidden, dim=0)
    if hidden_padded.shape[0] < seq_pad:
        hidden_padded = torch.nn.functional.pad(hidden_padded, (0, 0, 0, seq_pad - hidden_padded.shape[0]))
    hidden_tt = to_dev(hidden_padded, device)

    # Router on device
    router_w_tt = to_dev(router_w, device)
    router_tt = ttnn.matmul(hidden_tt, router_w_tt)
    router_host = ttnn.to_torch(router_tt)[:seq_len].float()

    scores_dev = torch.softmax(router_host, dim=-1)
    topk_w_dev, topk_i_dev = torch.topk(scores_dev, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w_dev = topk_w_dev / topk_w_dev.sum(dim=-1, keepdim=True)

    # Group tokens by expert
    expert_to_tokens = {}
    for tok in range(seq_len):
        for k in range(NUM_EXPERTS_PER_TOK):
            eidx = topk_i_dev[tok, k].item()
            w = topk_w_dev[tok, k].item()
            if eidx not in expert_to_tokens:
                expert_to_tokens[eidx] = []
            expert_to_tokens[eidx].append((tok, w))

    # Load expert weights to device
    expert_gate_tt = [to_dev(expert_gate[i], device) for i in range(NUM_EXPERTS)]
    expert_up_tt = [to_dev(expert_up[i], device) for i in range(NUM_EXPERTS)]
    expert_down_tt = [to_dev(expert_down[i], device) for i in range(NUM_EXPERTS)]

    # Run experts on device
    moe_output = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)

    for eidx, token_info in expert_to_tokens.items():
        tok_indices = [t[0] for t in token_info]
        weights = [t[1] for t in token_info]

        # Gather tokens for this expert and pad to tile alignment
        expert_input = hidden[tok_indices]
        n_tok = len(tok_indices)
        n_tok_pad = ((n_tok + TILE - 1) // TILE) * TILE
        inp_padded = torch.nn.functional.pad(expert_input, (0, 0, 0, n_tok_pad - n_tok))
        inp_tt = to_dev(inp_padded, device)

        # gate_proj and up_proj via TTNN matmul
        gate_out_tt = ttnn.matmul(inp_tt, expert_gate_tt[eidx])
        up_out_tt = ttnn.matmul(inp_tt, expert_up_tt[eidx])

        # SiLU(gate) * up via TT-Lang kernel
        activated_tt = to_dev(torch.zeros(n_tok_pad, MOE_INTER, dtype=torch.bfloat16), device)
        silu_mul_kernel(gate_out_tt, up_out_tt, activated_tt)

        # down_proj via TTNN matmul
        expert_out_tt = ttnn.matmul(activated_tt, expert_down_tt[eidx])

        # Readback and scatter with weights
        expert_out = ttnn.to_torch(expert_out_tt)[:n_tok, :HIDDEN_SIZE].to(torch.bfloat16)
        for i, (tok_idx, w) in enumerate(token_info):
            moe_output[tok_idx] += expert_out[i] * w

    # Compare
    r = moe_output.float().flatten()
    e = moe_ref.float().flatten()
    p = torch.corrcoef(torch.stack([r, e]))[0, 1].item()
    max_err = (moe_output.float() - moe_ref.float()).abs().max().item()

    print(f"MoE on-device test (seq_len={seq_len}, experts={NUM_EXPERTS}, top-{NUM_EXPERTS_PER_TOK}):")
    print(f"  PCC: {p:.6f}")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  {'PASS' if p > 0.97 else 'FAIL'}")
    return p > 0.97


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    ok = test_moe_on_device(device, seq_len=32)
    ttnn.close_device(device)
    if not ok:
        exit(1)
