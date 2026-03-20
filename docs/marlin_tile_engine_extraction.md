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
- The result is structurally closer to Marlin, but still missing:
  - cp.async-based global->shared staging
  - software pipelining across stages
  - Marlin's fragment/register blocking
- `NVFP4` now also has a staged-qweight/staged-scale version in the base fused
  kernel. The split-K kernel still uses the scalar/global NVFP4 helper because
  that remains the safer choice for the current best-latency path.

## Why FP8 First

- `FP8` uses per-channel scale, so the scale path is much simpler than NVFP4.
- NVFP4 still needs its packed nibble decode and per-16 block scale semantics.
- A successful FP8 engine swap would validate the shared A/B staging and
  pipeline integration before taking on the harder NVFP4 path.
