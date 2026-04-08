"""DFlash cached forward performance benchmark.

Uses DFlashDraft class with pre-allocated scratch buffers.
Sweeps context lengths from 2k to 120k (doubling each step).
Reports wall-clock, TT-Lang kernel time, and TTNN overhead per context length.
"""
import sys
sys.path.insert(0, "/tmp")
import time
import torch
import torch.nn.functional as F
import ttnn

import dflash_draft
from dflash_draft import (
    DFlashDraft, to_dev, _tile_pad,
    enable_op_timing, disable_op_timing, print_op_timing,
    BSIZE, SP, HIDDEN, HDIM, NQH, NKVH, DLAYERS, N_CTX_LAYERS,
    TILE, ROPE_THETA,
)

CTX_LENGTHS = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 120_000]
N_WARMUP = 2
N_TIMED = 5
NEW_ACCEPTED = 3


def extend_rope_tables(draft, max_pos):
    """Extend RoPE cos/sin tables to cover max_pos positions."""
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
    pos = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
    sin_adj = sin_full.clone()
    sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
    draft.w["rope_cos_full"] = cos_full
    draft.w["rope_sin_full"] = sin_adj


def bench_ctx_len(draft, d, ctx_len, noise_dev, cache_noise):
    """Benchmark one context length. Returns (wall_ms, ttl_ms, ttnn_ms)."""
    dflash_draft.CACHE_NOISE = cache_noise
    ctx_sp = _tile_pad(ctx_len)
    new_ctx_sp = _tile_pad(NEW_ACCEPTED)
    new_kv_sp = new_ctx_sp + SP

    # Build cache from random context
    target_hidden = torch.randn(ctx_len, N_CTX_LAYERS * HIDDEN,
                                dtype=torch.bfloat16) * 0.1
    ctx_padded = target_hidden
    if ctx_sp > ctx_len:
        ctx_padded = F.pad(target_hidden, (0, 0, 0, ctx_sp - ctx_len))
    ctx_dev = draft.prepare_context(to_dev(ctx_padded, d))

    # Populate cache (first forward)
    draft.alloc_scratch(ctx_sp + SP)
    draft.setup_rope(0, ctx_sp, q_start=ctx_len, new_ctx_real=ctx_len)
    _, cache = draft.step(noise_dev, ctx_dev, None)
    ttnn.synchronize_device(d)

    # CACHE_NOISE=False: cache has ctx_len rows
    # CACHE_NOISE=True: cache has ctx_sp + SP rows (context + noise, tile-padded)
    cache_rows = cache[0]["k"].shape[2]

    # Prepare steady-state inputs
    new_ctx_data = torch.randn(NEW_ACCEPTED, N_CTX_LAYERS * HIDDEN,
                               dtype=torch.bfloat16) * 0.1
    if new_ctx_sp > NEW_ACCEPTED:
        new_ctx_data = F.pad(new_ctx_data, (0, 0, 0, new_ctx_sp - NEW_ACCEPTED))
    new_ctx_dev = draft.prepare_context(to_dev(new_ctx_data, d))

    draft.alloc_scratch(new_kv_sp)
    draft.setup_rope(cache_rows, new_ctx_sp,
                     q_start=cache_rows + NEW_ACCEPTED,
                     new_ctx_real=NEW_ACCEPTED)

    # Warmup
    for _ in range(N_WARMUP):
        _, _ = draft.step(noise_dev, new_ctx_dev, cache)
        ttnn.synchronize_device(d)

    # Timed
    enable_op_timing()
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        _, _ = draft.step(noise_dev, new_ctx_dev, cache)
        ttnn.synchronize_device(d)
    elapsed = time.perf_counter() - t0
    wall_ms = elapsed / N_TIMED * 1000
    ttl_ms = sum(dflash_draft._op_times.values()) * 1000 / N_TIMED
    ttnn_ms = wall_ms - ttl_ms
    disable_op_timing()

    return wall_ms, ttl_ms, ttnn_ms


def main():
    torch.manual_seed(42)
    max_ctx = max(CTX_LENGTHS)
    max_ctx_sp = _tile_pad(max_ctx)

    d = ttnn.open_device(device_id=0)
    try:
        draft = DFlashDraft(d)
        extend_rope_tables(draft, max_ctx_sp + SP + TILE)

        noise_bf = torch.randn(BSIZE, HIDDEN, dtype=torch.bfloat16) * 0.1
        noise_dev = to_dev(noise_bf, d)

        for cache_noise in [False, True]:
            mode = "ctx+noise" if cache_noise else "ctx only"
            results = []
            for ctx_len in CTX_LENGTHS:
                print(f"\n--- {mode}, ctx={ctx_len:,} ---")
                wall, ttl, ttnn_oh = bench_ctx_len(draft, d, ctx_len, noise_dev, cache_noise)
                results.append((ctx_len, wall, ttl, ttnn_oh))
                print(f"  wall={wall:.1f}ms  ttl={ttl:.1f}ms  ttnn={ttnn_oh:.1f}ms")

            print(f"\n{'='*60}")
            print(f"  CACHE_NOISE={cache_noise} ({mode})")
            print(f"  {N_WARMUP} warmup, {N_TIMED} timed per context length")
            print(f"{'='*60}")
            print(f"  {'ctx':>8} {'wall (ms)':>10} {'ttl (ms)':>10} {'ttnn (ms)':>10} {'ttnn %':>8}")
            print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for ctx_len, wall, ttl, ttnn_oh in results:
                pct = ttnn_oh / wall * 100 if wall > 0 else 0
                print(f"  {ctx_len:>8,} {wall:>10.1f} {ttl:>10.1f} {ttnn_oh:>10.1f} {pct:>7.0f}%")

            print(f"\nPer-op breakdown at ctx={max_ctx:,} ({mode}, last {N_TIMED} runs):")
            print_op_timing()

    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
