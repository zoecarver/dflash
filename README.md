# DFlash on Tenstorrent

Reference implementation of [DFlash](https://arxiv.org/abs/2602.06036) speculative decoding on Tenstorrent hardware (QuietBox, 4 cards). DFlash is a lightweight 8-layer cross-attention draft model that proposes 16 tokens in parallel, verified by an unmodified target LLM.

Target model: 4-chip tensor parallelism. Draft model: replicated across the same mesh.

## DFlash model architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      speculative decode loop                       │
  │                                                                    │
  │   target model (eg Qwen3-30B-A3B)                                 │
  │   ┌─────────────────────────┐                                     │
  │   │  48 layers, self-attn   │                                     │
  │   │  + MoE                  │                                     │
  │   │                         │                                     │
  │   │  hidden states at       │                                     │
  │   │  layers [1,12,23,34,45] │                                     │
  │   └──────┬─────────────┬────┘                                     │
  │          │             │                                          │
  │          │ 5 x hidden  │ logits                                   │
  │          │ states      │ (seq, vocab)                             │
  │          ▼             ▼                                          │
  │   ┌────────────┐  ┌──────────────────────────────┐               │
  │   │ context    │  │ argmax target logits          │               │
  │   │ projection │  │ compare with draft proposals: │               │
  │   │ FC + norm  │  │   draft: [A, B, C, D, ...]   │               │
  │   └─────┬──────┘  │   target: [A, B, X, ...]     │               │
  │         │         │   accept prefix [A, B]        │               │
  │         │ ctx     │   take target's token X       │               │
  │         │         └──────────────┬────────────────┘               │
  │         │                        │ accepted tokens                │
  │         ▼                        │ fed back as next prompt        │
  │   ┌─────────────────────────┐    │                                │
  │   │    DFlash layer (x8)    │    │                                │
  │   │                         │    │                                │
  │   │  noise = embed(tokens)  │    │                                │
  │   │  Q from noise           │    │                                │
  │   │  K,V from [ctx; noise]  │    │                                │
  │   │  QK-norm + RoPE         │    │                                │
  │   │  SDPA (non-causal, GQA) │    │                                │
  │   │  O proj + residual      │    │                                │
  │   │  MLP + residual         │    │                                │
  │   │  KV cache: ctx K,V      │    │                                │
  │   └──────────┬──────────────┘    │                                │
  │              ▼                   │                                │
  │   RMSNorm → lm_head → argmax    │                                │
  │              │                   │                                │
  │              │ 16 draft tokens   │                                │
  │              └───────────────────┘                                │
  │              fed to target for verification                       │
  └─────────────────────────────────────────────────────────────────────┘
```

### KV cache: cross-attention, not self-attention

In a standard transformer, the KV cache stores keys/values from **previous tokens in the same sequence** -- it grows by one row per decode step and K/V come from the same input as Q.

DFlash's KV cache is different. K and V come from two sources:

1. **Context** (from the target model's hidden states) -- this is the cross-attention part. The context grows by the number of accepted tokens each step, not by one.
2. **Noise** (the draft token embeddings) -- these 16 rows are recomputed every step and never cached.

The cache stores post-QKnorm + post-RoPE K and raw V for the context portion only. Each step:
- Compute K/V only for **new context + noise** (small matmul, ~64 rows)
- Concat cached K/V with new K/V for attention
- After accept/reject, slice the context portion into the updated cache

At 120k context this gives 9.6x speedup vs recomputing all K/V each step.

## Op mapping: TT-Lang vs TTNN

| Operation | Backend | Notes |
|-----------|---------|-------|
| RoPE | TT-Lang | Element-wise, works on mesh |
| Softmax | TT-Lang | Fused row-wise softmax (non-cached path) |
| Residual add | TT-Lang | Fused element-wise |
| SiLU * mul | TT-Lang | Fused MLP activation |
| RMSNorm | TTNN | TT-Lang version has ~0.63x magnitude bug on mesh |
| QKV/O/MLP projections | TTNN matmul | Large weight matmuls |
| Attention (cached) | TTNN SDPA | Fused flash attention |
| Context projection | TTNN matmul | Split 5-way for bf16 precision |
| Argmax | TTNN | On-device token selection |

Controlled by `TTLANG_ENABLED` and `TTLANG_RMSNORM` flags in `dflash_draft.py`.

## Setup

Requires Tenstorrent hardware with `tt-metal` and `ttnn`. Weights at:

```
/workspace/qwen-coder-30b-a3b/weights/    # Qwen3 target + tokenizer
/workspace/qwen-coder-30b-a3b/dflash/     # DFlash draft weights
```

## Entry point

```bash
python qwen3_cpu_tt_dflash.py
```

CPU Qwen3 (stock `transformers`) for prefill/verify. DFlash entirely on Tenstorrent: context projection, 8-layer forward, lm_head, argmax.

## Files

| File | Description |
|------|-------------|
| `qwen3_cpu_tt_dflash.py` | Main entry point: CPU target + device DFlash |
| `dflash_draft.py` | DFlash draft model (TTNN + TT-Lang) |
| `device.py` | Shared device infra (mesh open/close, tensor placement) |
| `qwen3.py` | Standalone Qwen3 target model on device (4-chip TP) |
| `spec_decode.py` | Full device spec decode (both target + draft on TT) |
| `src/` | TT-Lang fused kernels (rmsnorm, rope, softmax, silu_mul, residual_add) |
| `model/` | Original PyTorch DFlash model |
| `test/` | Unit and integration tests |

## Performance

Profiled standalone DFlash forward (no host transfers):

| Context length | Non-cached | Cached | Speedup |
|---|---|---|---|
| 64 | 7.9ms | -- | -- |
| 120k | 897ms | 93ms | 9.6x |
| 250k | 1750ms | 180ms | 9.7x |

Note: current cached forward bottleneck is 2x concat of cached K/V (250k rows) per layer for SDPA input.

## Example run

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

Average acceptance: 3.6 tokens/step (CPU reference: 5.3). The gap requires further investigation (likely due to bf16 precision). TTNN and TT-Lang impls have same acceptance rate / PCC.
