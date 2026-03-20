# Marlin Tile Engine Extraction Notes

## Goal

Replace the current scalar qweight helpers in
[`rdquant/csrc/fused_gemv.cu`](../rdquant/csrc/fused_gemv.cu):

- `run_nvfp4_qweight_k_tile(...)`
- `run_fp8_qweight_k_tile(...)`

with a Marlin-style tile engine, while keeping the current mixed split-K outer
scheduler intact.

## Local Marlin Source

The relevant local reference is in:

- [`/root/autodl-tmp/marlin/marlin_fp8/marlin.cuh`](/root/autodl-tmp/marlin/marlin_fp8/marlin.cuh)
- [`/root/autodl-tmp/marlin/marlin_fp8/marlin.cu`](/root/autodl-tmp/marlin/marlin_fp8/marlin.cu)
- [`/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h)
- [`/root/autodl-tmp/marlin/marlin_fp8/generated/kernel_selector.h`](/root/autodl-tmp/marlin/marlin_fp8/generated/kernel_selector.h)

## Shared Decode Config To Target First

The first mixed-engine target should still be the shared small-batch config:

- `threads = 256`
- `thread_m_blocks = 1`
- `thread_n_blocks = 8`
- `thread_k_blocks = 8`
- `stages = 4`

Why this config:

- It exists for both decode paths in
  [`kernel_selector.h`](/root/autodl-tmp/marlin/marlin_fp8/generated/kernel_selector.h#L3).
- It matches the earlier mixed-kernel design choice.
- It avoids trying to support both 128-thread and 256-thread kernels inside one
  fused launch.

## Marlin Pieces We Actually Need

The mixed outer scheduler already exists in RDQuant. We do not need to port all
of Marlin. The extraction target is narrower:

1. `A` staging path

- cp.async helpers live in
  [`marlin.cuh`](/root/autodl-tmp/marlin/marlin_fp8/marlin.cuh#L48).
- The relevant shared/global stride math starts around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L525).

2. `B` staging path for Marlin-packed qweights

- Weight/shared stride logic starts around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L542).
- This is the most important part to replace the current scalar qweight decode.

3. Scale loading path

- Scale/shared stride logic starts around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L555).
- NVFP4 and FP8 differ here, so this is where the two helper bodies will still
  diverge after extraction.

4. Main pipeline loop

- The software pipeline and fetch/compute loop begins around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L1744)
  and continues through the staged compute loop around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L1788).

5. Epilogue write path

- The output writeback logic begins around
  [`marlin_template.h`](/root/autodl-tmp/marlin/marlin_fp8/marlin_template.h#L1576).
- RDQuant still needs its own `inv_perm` scatter after this.

## What We Should Not Port

- Host-side `determine_exec_config(...)` and autotune logic in
  [`marlin.cu`](/root/autodl-tmp/marlin/marlin_fp8/marlin.cu#L430)
- Full Marlin launch path and workspace ownership
- The single-type persistent tile scheduler inside `Marlin<>`

RDQuant already has:

- mixed tile dispatch
- split-K scheduling
- reusable workspace reduction
- mixed epilogue with `inv_perm`

## Immediate Coding Sequence

1. Keep the current outer split-K kernel shape.
2. Keep the current helper interface boundary:
   - `run_nvfp4_qweight_k_tile(...)`
   - `run_fp8_qweight_k_tile(...)`
3. Replace the scalar body of one helper at a time:
   - first `FP8`, because its scale path is simpler
   - then `NVFP4`
4. Only after one helper is Marlin-style, revisit block size and shared-memory
   usage.

Current progress on this sequence:

- `FP8` now has:
  - `half2` fragment multiply/accumulate
  - shared-memory staging for the Marlin-packed `B` tile over one `kBlockK`
    slice
  - `int4` vectorized qweight copies into shared staging
  - `cp.async`-backed qweight global->shared copies
  - a double-buffered shared->register helper pipeline across the eight
    `16-K` subtiles inside one `kBlockK=128` slice
  - on the main split-K kernel, a lighter cross-`kBlockK` FP8 prefetch path
    that only ping-pongs two `16-K` shared chunks instead of duplicating the
    full `kBlockK` qweight tile
- The result is structurally closer to Marlin, but still missing:
  - software pipelining across stages
  - Marlin's fragment/register blocking
- `NVFP4` now also has a staged-qweight/staged-scale version in the base fused
  kernel.
- The split-K path is now benchmarked in two forms:
  - the original scalar/global NVFP4 helper path
  - an alternate staged-NVFP4 path
  The staged-NVFP4 variant is not a uniform win, so both are kept visible in
  the benchmark instead of prematurely replacing the previous best path.
- RDQuant now also has a conservative wrapper that chooses between those two
  split-K variants at launch time. This is not meant to be the final solution;
  it is just a way to preserve the measured win on the few small-tile shapes
  where the staged-NVFP4 path helps, without forcing that path on the broader
  set of layers where the original scalar-NVFP4 split-K kernel remains better.
- The staged qweight loaders now use 16-byte vector copies. This is still not
  Marlin's full multi-stage pipeline, but the loader itself is now using the
  same `cp.async` family of primitives instead of plain vector stores.
- What is still missing relative to Marlin:
  - multi-stage double buffering
  - cp.async-based activation/scales staging
  The main split-K FP8 path now does overlap qweight fetch with compute, but it
  does so with a lighter `16-K` chunk ping-pong instead of a full-tile
  double-buffer. That keeps the overlap structure while pulling resource usage
  back down to a more practical range:
  - scalar-NVFP4 split-K kernel: `REG=56`, `SHARED=5380`
  - staged-NVFP4 split-K kernel: `REG=64`, `SHARED=10500`
  It still does not have Marlin's deeper multi-stage pipeline or cp.async-based
  activation/scales staging, so there is still headroom in the inner loop.
- Remote `ncu` runs on RTX 5070 also now confirm that the current split-K path
  is not uniformly memory-bound:
  - `q_proj` scalar `SplitK` sits around compute `57.7%` / DRAM `31.0%`
  - `down_proj` scalar `SplitK` sits around compute `90.3%` / DRAM `18.4%`
  That is exactly why a single global FP8 chunk size is not ideal.
- Those same `ncu` runs now also point at the next inner-loop target more
  specifically:
  - `q_proj` narrow FP8 path shows long-scoreboard `5.65` and MIO throttle
    `5.02`
  - `down_proj` wide FP8 path shows long-scoreboard `3.70` but much larger MIO
    throttle `14.44`
  So the next profitable FP8 work is reducing repeated helper-side loads and
  conversions, not adding even deeper qweight buffering by default.
- RDQuant therefore now keeps two scalar-NVFP4 split-K FP8 chunk variants:
  - a narrow `16-K` chunk kernel (`SHARED=5380`)
  - a wider `32-K` chunk kernel (`SHARED=9476`)
  The launch path only uses the wider FP8 chunk kernel for small-FP8-group
  shapes (`N_fp8 <= 128`, `K >= 4096`), which matches the current `o_proj` /
  `down_proj` corner without paying that shared-memory cost on the rest of the
  decode layers.
- The Python benchmark now exposes those two variants directly as `SK16` and
  `SK32`, instead of only reporting the heuristic-selected `SplitK` column.
  This makes FP8 chunk heuristic tuning a data problem rather than guesswork.
- That benchmark data also changed the current heuristic. The older rule
  assumed very small FP8 groups should take the wide `32-K` chunk path; the
  measured Qwen3-4B decode shapes show the reverse. The current launch policy
  now uses the wide FP8 chunk kernel only for medium FP8 groups:
  - `384 <= N_fp8 < 4096` -> wide `32-K`
  - otherwise -> narrow `16-K`
- The latest cleanup step hoists FP8 per-channel scale conversion out of the
  split-K chunk helpers. The helper now consumes a precomputed `half2` scale
  instead of reloading/converting it inside every `16-K` or `32-K` chunk. This
  keeps the main split-K kernels at `REG=56` while slightly improving the
  current best-of-two fused latency.
- The next small FP8 cleanup step now also stages chunk-local packed words and
  `x` fragments into registers through a dedicated chunk helper, instead of
  rebuilding them inside each `16-K` chunk compute call. This is still not a
  Marlin fragment engine, but it removes duplicated helper-side shared reads
  and address arithmetic from both the narrow and wide FP8 chunk paths.
- The current measurements line up with that change:
  - local RTX 5090 runs move the per-layer `BestSK` total from the earlier
    `~281.6us` baseline into the `~279-281us` range
  - remote RTX 5070 `ncu` runs show:
    - `q_proj` wide path: long-scoreboard `5.73 -> 4.60`, eligible warps
      `0.50 -> 0.53`
    - `down_proj` narrow path: MIO throttle `18.02 -> 15.19`
  So the helper-side register staging is behaving as intended even though the
  total kernel duration gain is still modest.

## Why FP8 First

- `FP8` uses per-channel scale, so the scale path is much simpler than NVFP4.
- NVFP4 still needs its packed nibble decode and per-16 block scale semantics.
- A successful FP8 engine swap would validate the shared A/B staging and
  pipeline integration before taking on the harder NVFP4 path.
