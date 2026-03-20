# Fused Mixed GEMV Plan

## Goal

Replace the current `2x marlin_gemm + cat + index_select` decode path with a
single-launch fused kernel for `M=1`:

- NVFP4 group: dequant + GEMV
- FP8 group: dequant + GEMV
- `inv_perm` scatter on store

Target workload is the 7 Qwen3-4B mixed layers:

| Layer | N_total | K | N_nvfp4 | N_fp8 |
|---|---:|---:|---:|---:|
| q_proj | 4096 | 2560 | 1920 | 2176 |
| k_proj | 1024 | 2560 | 256 | 768 |
| v_proj | 1024 | 2560 | 640 | 384 |
| o_proj | 2560 | 4096 | 2432 | 128 |
| gate_proj | 9728 | 2560 | 5376 | 4352 |
| up_proj | 9728 | 2560 | 8960 | 768 |
| down_proj | 2560 | 9728 | 2432 | 128 |

## Constraints We Are Keeping

- Existing weight packing stays Marlin-compatible:
  - NVFP4: `[K/16, N*2] int32`
  - FP8: `[K/16, N*4] int32`
- Existing scale permutations stay unchanged.
- Fused v1 should use one shared launch configuration for both dtype paths.
- Branching is allowed only at CTA/tile granularity, not inside the hot inner loop.

## Current Status

Implemented:

- A tiled fused prototype in [`rdquant/csrc/fused_gemv.cu`](../rdquant/csrc/fused_gemv.cu)
  with:
  - `1 CTA = 1 output tile`
  - CTA-level dispatch between `NVFP4` and `FP8`
  - shared-memory staging for the activation tile
  - `inv_perm` scatter in the epilogue
- Runtime checks in [`rdquant/csrc/bindings.cpp`](../rdquant/csrc/bindings.cpp)
- A benchmark script aligned to the 7 target layer shapes in
  [`benchmarks/bench_fused_gemv.py`](../benchmarks/bench_fused_gemv.py)

Observed result on RTX 5090:

- Correctness passes for the tiled prototype.
- Performance is still far from target because it still consumes row-major
  fake-quant tensors and uses scalar decode paths.
- Representative numbers from `python benchmarks/bench_fused_gemv.py`:
  - `q_proj`: cuBLAS `20.8us`, tiled prototype `390.6us`
  - `o_proj`: cuBLAS `19.9us`, tiled prototype `593.9us`
  - `down_proj`: cuBLAS `25.0us`, tiled prototype `1411.6us`

Conclusion: the scheduler shape is now closer to the final design, but further
work on this row-major prototype is not worthwhile. The next step has to be
switching the loaders and inner loops to Marlin-repacked weights.

## Shared Config Choice For Mixed-Marlin v1

Marlin decode uses the same small-batch family for both dtype paths:

- `(threads=256, thread_n_blocks=8, thread_k_blocks=8)`
- `(threads=128, thread_n_blocks=8, thread_k_blocks=4)`
- `(threads=128, thread_n_blocks=4, thread_k_blocks=8)`

Because a fused kernel only has one `blockDim.x`, v1 should force both paths to
a shared config.

Recommended v1 choice:

- `threads = 256`
- `thread_n_blocks = 8`
- `thread_k_blocks = 8`
- `stages = 4`

Reason:

- It is a valid Marlin specialization for both FP8 and NVFP4.
- It avoids idle-half-block logic in the first implementation.
- It matches the “single shared tile engine” objective better than mixing
  128-thread and 256-thread blocks in one launch.

## Implementation Plan

### 1. Lock the mixed scheduler contract

Difficulty: low

- Define the tile metadata needed by a fused persistent scheduler:
  - `kind`
  - `slice_col`
  - `logical_out_base`
- Decide whether v1 computes `TileDesc` on the fly or from a small host-built
  descriptor table.

### 2. Extract Marlin tile-entry points

Difficulty: very high

- Split the single-type Marlin path into two device-side entry points:
  - `run_nvfp4_tile(...)`
  - `run_fp8_tile(...)`
- Keep Marlin’s address formulas, scale loads, and shared-memory usage intact.
- Do not try to launch Marlin kernels from inside another kernel.

### 3. Build a mixed persistent outer scheduler

Difficulty: high

- Outer kernel owns one persistent loop over logical tiles.
- Each iteration chooses a tile kind and dispatches one of the two Marlin tile
  paths.
- Only CTA-level branching is allowed.

### 4. Integrate `inv_perm` into the epilogue

Difficulty: medium

- Store each completed channel directly to `y[inv_perm[n]]`.
- Keep v1 simple: no extra output staging unless profiling says store scatter is
  a measurable bottleneck.

### 5. Replace the current Python benchmark tensors with Marlin-packed inputs

Difficulty: medium

- Feed the fused path the same repacked tensors used by the standalone Marlin
  kernels.
- Benchmark against:
  - `2x Marlin`
  - fused mixed kernel
  - cuBLAS BF16

### 6. Profile and tune the shared config

Difficulty: high

- If the shared `256-thread` config is clearly suboptimal for some layers,
  compile 2-3 fused variants from the common small-batch family and select the
  best shared config per shape.

## Immediate Next Code Changes

1. Replace the current row-major `run_fp4_tile` / `run_fp8_tile` loaders with
   Marlin-packed tile loaders.
2. Introduce a mixed tile descriptor or equivalent `slice_col` mapping so a
   persistent kernel can walk `FP4` tiles first and `FP8` tiles second.
3. Keep the current benchmark script, but switch its inputs to repacked test
   tensors so the benchmark reflects the final memory layout.

## What Not To Do

- Do not redesign weight layout from scratch.
- Do not keep optimizing the current row-major prototype.
- Do not put `if (is_fp4)` inside the `K` hot loop for mixed channels.
- Do not try to preserve fully independent autotuned Marlin configs inside one
  launch in v1.
