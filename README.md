# DFlash on Tenstorrent

Reference implementation of [DFlash](https://arxiv.org/abs/2602.06036) speculative decoding on Tenstorrent hardware (QuietBox, 4 cards). DFlash is a lightweight 8-layer cross-attention draft model that proposes 16 tokens in parallel, verified by an unmodified target LLM.

Target model: 4-chip tensor parallelism. Draft model: replicated across the same mesh.

## DFlash model architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      speculative decode loop                        │
  │                                                                     │
  │   target model (eg Qwen3-30B-A3B)                                   │
  │   ┌─────────────────────────┐                                       │
  │   │  48 layers, self-attn   │                                       │
  │   │  + MoE                  │                                       │
  │   │                         │                                       │
  │   │  hidden states at       │                                       │
  │   │  layers [...]           │                                       │
  │   └──────┬─────────────┬────┘                                       │
  │          │             │                                            │
  │          │ 8 x hidden  │ logits                                     │
  │          │ states      │ (seq, vocab)                               │
  │          ▼             ▼                                            │
  │   ┌────────────┐  ┌───────────────────────────────┐                 │
  │   │ context    │  │ argmax target logits          │                 │
  │   │ projection │  │ compare with draft proposals: │                 │
  │   │ FC + norm  │  │   draft: [A, B, C, D, ...]    │                 │
  │   └─────┬──────┘  │   target: [A, B, X, ...]      │                 │
  │         │         │   accept prefix [A, B]        │                 │
  │         │ ctx     │   take target's token X       │                 │
  │         │         └──────────────┬────────────────┘                 │
  │         │                        │ accepted tokens                  │
  │         ▼                        │ fed back as next prompt          │
  │   ┌─────────────────────────┐    │                                  │
  │   │    DFlash layer (x8)    │    │                                  │
  │   │                         │    │                                  │
  │   │  noise = embed(tokens)  │    │                                  │
  │   │  Q from noise           │    │                                  │
  │   │  K,V from [ctx; noise]  │    │                                  │
  │   │  QK-norm + RoPE         │    │                                  │
  │   │  SDPA (non-causal, GQA) │    │                                  │
  │   │  O proj + residual      │    │                                  │
  │   │  MLP + residual         │    │                                  │
  │   │  KV cache (ctx only)    │    │                                  │
  │   └──────────┬──────────────┘    │                                  │
  │              ▼                   │                                  │
  │   RMSNorm → lm_head → argmax     │                                  │
  │              │                   │                                  │
  │              │ 16 draft tokens   │                                  │
  │              └───────────────────┘                                  │
  │              fed to target for verification                         │
  └─────────────────────────────────────────────────────────────────────┘
```

### KV cache cross-attention

In a standard transformer, the KV cache stores keys/values from previous tokens in the same sequence, grows by one row per decode step, and K/V come from the same input as Q.

DFlash's KV cache differs in two ways:

1. **K/V come from two sources**: context (target hidden states projected through k/v_proj) and noise (draft token embeddings through k/v_proj). Both are concatenated before attention.
2. **Context-only caching**: the cache only stores context K/V. Noise K/V is computed fresh each step for SDPA but never persisted. After acceptance, the accepted tokens' K/V are re-computed as new context (from target hidden states) on the next step.

The cache stores post-QKnorm + post-RoPE K and raw V. Each step:
- Compute K/V for new context + noise (not the full history)
- Concat cached K/V with new K/V for SDPA
- Store only the context portion of new K/V in the cache (exclude noise)

This matches the reference implementation which uses `DynamicCache.update()` to append all K/V, then `DynamicCache.crop(start)` to discard noise positions. Net cache growth per step is `acceptance_length + 1`.

## Op mapping: TT-Lang vs TTNN

| Operation | Impl | Notes |
|-----------|---------|-------|
| RoPE | TT-Lang | Element-wise, works on mesh |
| Residual add | TT-Lang | Fused element-wise |
| SiLU * mul | TT-Lang | Fused MLP activation |
| RMSNorm | TTNN | TT-Lang version has ~0.63x magnitude bug on mesh |
| QKV/O/MLP projections | TTNN matmul | Large weight matmuls |
| Attention | TTNN SDPA | Fused flash attention (non-causal, GQA) |
| Context projection | TTNN matmul | Split 5-way for bf16 precision |
| Argmax | TTNN | On-device token selection |
| Softmax | TT-Lang | Row-wise fused kernel; unused (SDPA handles it) |

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
| `qwen3_cpu_tt_dflash.py` | Main entry point: CPU target + device DFlash (KV-cached draft) |
| `dflash_draft.py` | DFlash draft model (TTNN + TT-Lang) |
| `device.py` | Shared device infra (mesh open/close, tensor placement) |
| `qwen3.py` | Standalone Qwen3 target model on device (4-chip TP) |
| `qwen3_inference.py` | Qwen3 target inference entry point (device only, no DFlash) |
| `spec_decode.py` | Full device spec decode (both target + draft on TT) |
| `src/` | TT-Lang fused kernels (rmsnorm, rope, softmax, silu_mul, residual_add) |
| `model/` | Original PyTorch DFlash model |
| `test/` | Unit and integration tests |

## Performance

Profiled standalone DFlash forward (no host transfers). Non-cached recomputes K/V projections, QK-norm, and RoPE for the entire context every step. Cached stores post-RoPE K and raw V, only computing new rows each step.

| Context length | Non-cached (acc 5.3) | Cached ctx+noise (acc 3.2) | Cached ctx only (acc 5.8) |
|---|---|---|---|
| 64 | 7.9ms | -- | -- |
| 120k | 892ms | 80ms | 145ms |
| 250k | 1776ms | 158ms | 294ms |

Cached ctx+noise (`CACHE_NOISE=True`) is faster but has degraded acceptance because noise
K/V (from draft embeddings) differs from context K/V (from target hidden states). Cached
ctx only (`CACHE_NOISE=False`) matches the reference model and achieves full acceptance rate.

## Example run

```
  step 1: acc=5/16 avg=5.0 1.2s gen=5
  step 2: acc=1/16 avg=3.0 1.1s gen=6
  step 3: acc=4/16 avg=3.3 1.1s gen=10
  step 4: acc=3/16 avg=3.2 1.1s gen=13
  step 5: acc=1/16 avg=2.8 1.1s gen=14
  step 6: acc=2/16 avg=2.7 1.2s gen=16
  step 7: acc=3/16 avg=2.7 1.2s gen=19
  step 8: acc=3/16 avg=2.8 1.2s gen=22
  step 9: acc=13/16 avg=3.9 1.2s gen=35
  step 10: acc=8/16 avg=4.3 1.4s gen=43
  step 11: acc=11/16 avg=4.9 1.4s gen=54
  step 12: acc=15/16 avg=5.8 1.5s gen=69

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

Average acceptance: 5.8 tokens/step, matching the CPU reference. Uses KV-cached forward
with `CACHE_NOISE=False` (only cache context K/V, not noise -- matching the reference model).
