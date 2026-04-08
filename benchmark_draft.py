"""DFlash cached forward performance benchmark.

Uses DFlashDraft class with pre-allocated scratch buffers.
Measures steady-state decode at 120k context: scratch allocated once,
noise pre-uploaded, only draft.step() in the timed loop.
Reports wall-clock, per-op breakdown, and TTNN overhead.
"""
import sys
sys.path.insert(0, "/tmp")
import time
import torch
import torch.nn.functional as F
import ttnn

import dflash_draft
from dflash_draft import (
    DFlashDraft, crop_cache, to_dev, _tile_pad,
    enable_op_timing, disable_op_timing, print_op_timing,
    BSIZE, SP, HIDDEN, HDIM, NQH, NKVH, DLAYERS, N_CTX_LAYERS,
    TILE, EPS, ROPE_THETA,
)

dflash_draft.CACHE_NOISE = False

CTX_LEN = 120_000
N_WARMUP = 2
N_TIMED = 5
NEW_ACCEPTED = 3  # typical acceptance per step


def main():
    torch.manual_seed(42)
    ctx_sp = _tile_pad(CTX_LEN)
    new_ctx_sp = _tile_pad(NEW_ACCEPTED)
    new_kv_sp = new_ctx_sp + SP

    print(f"Config: BSIZE={BSIZE}, ctx={CTX_LEN}, new_accepted={NEW_ACCEPTED}")
    print(f"  ctx_sp={ctx_sp}, new_kv_sp={new_kv_sp}")

    # Open device
    d = ttnn.open_device(device_id=0)
    try:
        # --- Init draft model ---
        draft = DFlashDraft(d)

        # Extend RoPE tables for long context (load_draft_weights defaults to 512)
        max_seq = ctx_sp + SP + TILE
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float32) / HDIM))
        pos = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        cos_full = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_full = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)[:, :HDIM]
        sin_adj = sin_full.clone()
        sin_adj[:, :HDIM // 2] = -sin_adj[:, :HDIM // 2]
        draft.w["rope_cos_full"] = cos_full
        draft.w["rope_sin_full"] = sin_adj

        # --- Build cache from random context (simulates prefill) ---
        print(f"\nPopulating cache (ctx={CTX_LEN})...")
        target_hidden = torch.randn(CTX_LEN, N_CTX_LAYERS * HIDDEN,
                                    dtype=torch.bfloat16) * 0.1
        ctx_padded = target_hidden
        if ctx_sp > CTX_LEN:
            ctx_padded = F.pad(target_hidden, (0, 0, 0, ctx_sp - CTX_LEN))
        ctx_dev = draft.prepare_context(to_dev(ctx_padded, d))

        # First forward to populate cache
        draft.alloc_scratch(ctx_sp + SP)
        draft.setup_rope(0, ctx_sp, q_start=CTX_LEN, new_ctx_real=CTX_LEN)
        noise_bf = torch.randn(BSIZE, HIDDEN, dtype=torch.bfloat16) * 0.1
        noise_dev = to_dev(noise_bf, d)
        t0 = time.time()
        _, cache = draft.step(noise_dev, ctx_dev, None)
        ttnn.synchronize_device(d)
        print(f"  Cache populated in {time.time()-t0:.1f}s")
        print(f"  Cache K/V rows: {cache[0]['k'].shape[2]}")

        # CACHE_NOISE=False: cache has exactly CTX_LEN context rows, no crop needed
        cache_rows = CTX_LEN
        print(f"  Cache rows: {cache_rows}")

        # --- Prepare steady-state decode inputs ---
        new_ctx_data = torch.randn(NEW_ACCEPTED, N_CTX_LAYERS * HIDDEN,
                                   dtype=torch.bfloat16) * 0.1
        if new_ctx_sp > NEW_ACCEPTED:
            new_ctx_data = F.pad(new_ctx_data,
                                 (0, 0, 0, new_ctx_sp - NEW_ACCEPTED))
        new_ctx_dev = draft.prepare_context(to_dev(new_ctx_data, d))

        # Re-allocate scratch for steady-state size
        draft.alloc_scratch(new_kv_sp)
        draft.setup_rope(cache_rows, new_ctx_sp,
                         q_start=cache_rows + NEW_ACCEPTED,
                         new_ctx_real=NEW_ACCEPTED)

        # Pre-upload noise (stays on device for all runs)
        noise_dev = to_dev(noise_bf, d)

        # --- Warmup ---
        print(f"\n=== Cached forward benchmark ===")
        print(f"  new_ctx={new_ctx_sp}, cache={cache_rows}, {N_WARMUP} warmup, {N_TIMED} timed")
        for _ in range(N_WARMUP):
            _, _ = draft.step(noise_dev, new_ctx_dev, cache)
            ttnn.synchronize_device(d)

        # --- Timed runs ---
        enable_op_timing()
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            _, _ = draft.step(noise_dev, new_ctx_dev, cache)
            ttnn.synchronize_device(d)
        elapsed = time.perf_counter() - t0
        per_fwd = elapsed / N_TIMED * 1000

        print(f"\n  Wall-clock: {per_fwd:.1f} ms/forward")
        print(f"  Per layer:  {per_fwd / DLAYERS:.1f} ms/layer")
        print(f"\n  Per-op breakdown ({N_TIMED} runs):")
        print_op_timing()
        disable_op_timing()

        ttl_total = per_fwd * N_TIMED
        print(f"\n  TT-Lang kernel time: ~{print_op_total():.0f} ms ({N_TIMED} runs)")
        print(f"  TTNN/host overhead:  ~{ttl_total - print_op_total():.0f} ms ({N_TIMED} runs)")

    finally:
        ttnn.close_device(d)


def print_op_total():
    """Return total TT-Lang kernel time in ms across all timed runs."""
    return sum(dflash_draft._op_times.values()) * 1000


if __name__ == "__main__":
    main()
