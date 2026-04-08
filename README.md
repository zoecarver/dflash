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
| RMSNorm | TT-Lang | Two-pass streaming (sum-of-squares, normalize+scale) |
| RoPE | TT-Lang | Element-wise rotary embeddings |
| Q/K/V projections | TT-Lang | Streaming matmul over K dimension |
| O proj + residual add | TT-Lang | Fused matmul + elementwise add |
| Gate/up + SiLU*mul | TT-Lang | Fused dual matmul + activation |
| Down proj + residual add | TT-Lang | Fused matmul + elementwise add |
| Attention | TTNN SDPA | Fused flash attention (non-causal, GQA) |
| Reshape/transpose | TTNN | SDPA interface (4D head layout) |
| KV cache concat/slice | TTNN | Cache management |
| Context projection | TTNN matmul | Split 5-way for bf16 precision |
| Argmax | TTNN | On-device token selection |
| Softmax | TT-Lang | Row-wise fused kernel; unused (SDPA handles it) |

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
| `dflash_draft.py` | DFlash draft model (TT-Lang compute, DFlashDraft class) |
| `benchmark_draft.py` | Standalone draft forward benchmark (120k context, per-op timing) |
| `device.py` | Shared device infra (mesh open/close, tensor placement) |
| `qwen3.py` | Standalone Qwen3 target model on device (4-chip TP) |
| `qwen3_inference.py` | Qwen3 target inference entry point (device only, no DFlash) |
| `spec_decode.py` | Full device spec decode (both target + draft on TT) |
| `src/` | TT-Lang fused kernels (rmsnorm, rope, softmax, silu_mul, residual_add) |
| `model/` | Original PyTorch DFlash model |
| `test/` | Unit and integration tests |

## Performance

Profiled standalone DFlash forward (no host transfers). Non-cached recomputes K/V projections, QK-norm, and RoPE for the entire context every step. Cached stores post-RoPE K and raw V, only computing new rows each step. Single p300 chip:

```
============================================================
  CACHE_NOISE=False (ctx only)
  2 warmup, 5 timed per context length
============================================================
       ctx  wall (ms)   ttl (ms)  ttnn (ms)   ttnn %
  -------- ---------- ---------- ---------- --------
     2,000       48.2       40.7        7.5      16%
     4,000       48.1       40.7        7.4      15%
     8,000       51.0       40.5       10.6      21%
    16,000       57.4       40.4       17.0      30%
    32,000       70.6       40.9       29.7      42%
    64,000       95.9       40.6       55.3      58%
   120,000      143.1       43.0      100.1      70%

Per-op breakdown at ctx=120,000 (ctx only, last 5 runs):
  Operation                 Total (ms)  Count   Avg (ms)  Kernel
  ------------------------- ---------- ------ ----------  ------
  gate_up+silu                   65.61     40      1.640  src/matmul_silu_mul.py
  down+resadd                    33.48     40      0.837  src/matmul_residual_add.py
  o_proj+resadd                  27.99     40      0.700  src/matmul_residual_add.py
  q_proj                         26.58     40      0.664  src/streaming_matmul.py
  k_proj                         10.98     40      0.275  src/streaming_matmul.py
  pa_rmsnorm                     10.93     40      0.273  src/rmsnorm.py
  qk_rmsnorm                     10.20     80      0.127  src/rmsnorm.py
  v_proj                          9.74     40      0.244  src/streaming_matmul.py
  in_rmsnorm                      9.13     40      0.228  src/rmsnorm.py
  q_rope                          5.46     40      0.136  src/rope.py
  k_rope                          3.60     40      0.090  src/rope.py
  fn_rmsnorm                      1.08      5      0.217  src/rmsnorm.py
  TOTAL                         214.78
```

Note: most ttnn logic is sdpa.

Roofline for top two ops:
```
--- Program 1024 (matmul_silu_mul) ---
grid: 10x10 (100 cores)
  DRAM read:        540.0 MB  (276480 transfers)
  DRAM write:         3.8 MB  (1920 transfers)
  effective BW:   395.4 GB/s (total payload / duration)
  transfer size:  2.0 KB (uniform)
  barriers:       960 read (1 per 288 reads), 480 write (1 per 4 writes)
  noc reads:      NOC_0=153600, NOC_1=122880
  noc writes:     NOC_1=1920
  DRAM channels:  16

ROOFLINE ANALYSIS
====================================================================================================
  Thread            Total   - Sync Waits   =       Work
  ------------ ----------   - ----------   = ----------
  NCRISC       63,914,612   - 59,101,945   =  4,812,667
  BRISC        42,058,956   - 42,009,536   =     49,420
  TRISC_0      133,995,410   -    479,148   = 133,516,262
  TRISC_1      134,709,090   -     47,751   = 134,661,339
  TRISC_2      134,005,535   -  2,266,137   = 131,739,398

  Memory:   4,812,667 cycles  (NCRISC)
  Compute: 134,661,339 cycles  (TRISC_1)

  96% compute bound
  Compute ├─●──────────────────────────────────────┤ Memory
          134,661,339 cycles                            4,812,667 cycles

--- Program 1024 (matmul_residual_add) ---
grid: 8x10 (80 cores)
  DRAM read:        271.2 MB  (138880 transfers)
  DRAM write:         1.2 MB  (640 transfers)
  effective BW:   384.8 GB/s (total payload / duration)
  transfer size:  2.0 KB (uniform)
  barriers:       30800 read (1 per 5 reads), 80 write (1 per 8 writes)
  noc reads:      NOC_0=123520, NOC_1=15360
  noc writes:     NOC_1=640
  DRAM channels:  16

ROOFLINE ANALYSIS
====================================================================================================
  Thread            Total   - Sync Waits   =       Work
  ------------ ----------   - ----------   = ----------
  NCRISC        1,494,230   -    825,848   =    668,382
  BRISC        57,653,282   - 45,988,357   = 11,664,925
  TRISC_0      11,453,391   - 10,672,962   =    780,429
  TRISC_1      11,490,080   -    104,576   = 11,385,504
  TRISC_2      11,461,329   -    122,673   = 11,338,656

  Memory:  11,664,925 cycles  (BRISC)
  Compute: 11,385,504 cycles  (TRISC_1)

  2% memory bound
  Compute ├───────────────────●────────────────────┤ Memory
          11,385,504 cycles                            11,664,925 cycles

====================================================================================================

```

Generated with tt-lang profiling tools, check them out [here](https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/reference/performance-tools.md)!

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
