# DFlash on Tenstorrent

Reference implementation of [DFlash](https://arxiv.org/abs/2602.06036) speculative decoding on Tenstorrent hardware, currently tested on a QuietBox with 4 cards. DFlash is a lightweight 8-layer cross-attention draft model that proposes token blocks in parallel, verified by an unmodified target LLM.

The target model runs with 4-chip tensor parallelism. The draft model runs replicated across the same mesh.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                   spec_generate loop                  │
│                                                       │
│  ┌─────────────┐            ┌──────────────────────┐  │
│  │  Draft      │            │  Target (Qwen3)      │  │
│  │  (DFlash)   │            │  (unmodified)        │  │
│  │             │            │                      │  │
│  │  8 layers   │            │  48 layers           │  │
│  │  cross-attn │            │  self-attn + MoE     │  │
│  └─────────────┘            └──────────────────────┘  │
│        ▲                         │          │         │
│        │                         │          │         │
│        │     ┌───────────────────┘          │         │
│        │     │ hidden states                │         │
│        │     │ from layers                  │ logits  │
│        │     │ [1,12,23,34,45]              │         │
│        │     ▼                              ▼         │
│  ┌─────────────┐                 ┌──────────────┐     │
│  │ ctx =       │                 │ argmax →     │     │
│  │ project +   │                 │ accept/reject│     │
│  │ norm        │                 │ draft tokens │     │
│  └─────────────┘                 └──────────────┘     │
└───────────────────────────────────────────────────────┘
```

The target model runs normally and exposes hidden states at 5 layers. DFlash projects these into context features, then uses cross-attention to draft a block of 16 tokens at once. The target verifies and accepts a prefix.

## Setup

Requires Tenstorrent hardware with `tt-metal` and `ttnn` installed. The target model (Qwen3-Coder-30B-A3B) and DFlash weights must be at:

```
/workspace/qwen-coder-30b-a3b/weights/    # Qwen3 target weights + tokenizer
/workspace/qwen-coder-30b-a3b/dflash/     # DFlash draft weights
```

## Entry point: `qwen3_cpu_tt_dflash.py`

Runs the full speculative decoding loop with:
- **CPU**: Stock Qwen3 via `transformers` (prefill, verify, hidden state extraction)
- **Tenstorrent**: DFlash draft model (context projection, 8-layer forward, lm_head, argmax)

This isolates the DFlash implementation on device from the target model, making it the best starting point for integration work.

```bash
python qwen3_cpu_tt_dflash.py
```

Two flags in the script control which ops use TT-Lang fused kernels vs TTNN:

- `USE_TTLANG = True` -- softmax, residual_add, silu_mul use TT-Lang kernels
- `TTLANG_RMSNORM = False` -- rmsnorm stays TTNN (TT-Lang version has a mesh scaling bug)

## Files

| File | Description |
|------|-------------|
| `qwen3_cpu_tt_dflash.py` | Main entry point: CPU target + device DFlash |
| `dflash_draft.py` | DFlash draft model (TTNN + TT-Lang forward) |
| `device.py` | Shared device infra (mesh open/close, tensor placement) |
| `qwen3.py` | Standalone Qwen3 target model on device (4-chip TP) |
| `spec_decode.py` | Full device spec decode (both target + draft on TT) |
| `src/` | TT-Lang fused kernels (rmsnorm, rope, softmax, silu_mul, residual_add) |
| `model/` | Original PyTorch DFlash model |
| `test/` | Unit and integration tests |

## Example run

This is a run using the cpu base model + on device dflash. `acc` is the number of accepted tokens. Timing is incorrect in the snippet below due to host <-> device transfers. 

Profiled runs without device transfers: standalone at 64 token context, dflash runs in 7.9ms. At 250k context it takes 1762ms (kv caching not implemented yet).

```
   step 1: acc=2/16 avg=2.0 1.5s gen=2
   step 2: acc=1/16 avg=1.5 1.0s gen=3
   step 3: acc=2/16 avg=1.7 1.1s gen=5
   step 4: acc=1/16 avg=1.5 1.1s gen=6
   step 5: acc=3/16 avg=1.8 1.1s gen=9
   step 6: acc=2/16 avg=1.8 1.1s gen=11
   step 7: acc=3/16 avg=2.0 1.1s gen=14
   step 8: acc=2/16 avg=2.0 1.2s gen=16
   step 9: acc=3/16 avg=2.1 1.2s gen=19
   step 10: acc=3/16 avg=2.2 1.2s gen=22
   step 11: acc=6/16 avg=2.5 1.3s gen=28
   step 12: acc=6/16 avg=2.8 1.3s gen=34
   step 13: acc=1/16 avg=2.7 1.4s gen=35
   step 14: acc=3/16 avg=2.7 1.3s gen=38
   step 15: acc=5/16 avg=2.9 1.4s gen=43
   step 16: acc=5/16 avg=3.0 1.4s gen=48
   step 17: acc=6/16 avg=3.2 1.5s gen=54
   step 18: acc=7/16 avg=3.4 1.5s gen=61
   step 19: acc=12/16 avg=3.8 1.5s gen=73

--- Output ---
user
Write a Python function that computes fibonacci numbers.
assistant
Here are several Python implementations of a Fibonacci function, from basic to optimized:

## 1. Basic Recursive Approach
```python
def fibonacci_recursive(n):
	"""
	Compute the nth Fibonacci number using recursion.
	Time complexity: O(2^n) - Very slow for large n
	"""
	if n <= 0:
		return 0
```