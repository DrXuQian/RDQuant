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
- A second transition path in the same file that consumes Marlin-repacked
  qweights (`[K/16, N*2]` for NVFP4 and `[K/16, N*4]` for FP8) while still
  using plain scales. This validates the mixed scheduler and qweight address
  formulas independently of Marlin's full inner loop.
- Runtime checks in [`rdquant/csrc/bindings.cpp`](../rdquant/csrc/bindings.cpp)
- A benchmark script aligned to the 7 target layer shapes in
  [`benchmarks/bench_fused_gemv.py`](../benchmarks/bench_fused_gemv.py)
- The scalar Marlin-qweight inner loops have now been extracted into explicit
  dtype-specific device entry points inside
  [`rdquant/csrc/fused_gemv.cu`](../rdquant/csrc/fused_gemv.cu):
  - `run_nvfp4_qweight_k_tile(...)`
  - `run_fp8_qweight_k_tile(...)`
  This does not change the algorithm yet, but it gives the mixed scheduler a
  clean seam where the current scalar decode path can be replaced with a
  Marlin-style tile engine without rewriting the outer split-K scheduler again.
  The current implementation is now explicitly routed through scalar backends,
  so future Marlin-style replacements can land behind the same wrapper names.
- The first actual vectorized replacement is now on the FP8 side:
  `run_fp8_qweight_k_tile(...)` uses `half2` fragment multiplies instead of
  four scalar multiply-adds per packed FP8 word. This is not the full Marlin
  tile engine yet, but it moves the active FP8 backend one step closer to the
  fragment-oriented Marlin compute path.
- The FP8 path now also stages the Marlin-packed `B` tile for one `kBlockK`
  slice through shared memory before fragment decode/accumulate. This makes the
  FP8 helper structurally closer to Marlin's staged `B` path even though the
  current split-K end-to-end latency is still roughly flat. The main value of
  this step is establishing the shared `B` staging seam that a later cp.async /
  software-pipelined FP8 helper can build on.
- The NVFP4 path now has the analogous staged-qweight/staged-scale structure in
  the base fused kernel. For the split-K kernel, the active NVFP4 path is still
  the older scalar/global-load version for now, because the staged NVFP4 path
  did not show a stable end-to-end win on the current split-K benchmark. This
  keeps the best mixed path conservative while still letting the base kernel
  evolve toward the same staged tile structure.
- The staged qweight loaders now use 16-byte vector copies into shared memory
  instead of scalar `int32` copies. This is still short of Marlin's async
  pipeline, but it narrows the gap between the current prototype and Marlin's
  wide staged global->shared transfer pattern.

Observed result on RTX 5090:

- Correctness passes for both:
  - the row-major tiled prototype
  - the Marlin-qweight transition path
- Performance is still far from target because it still consumes row-major
  fake-quant tensors or scalar per-element Marlin-qweight decode paths.
- Representative numbers from `python benchmarks/bench_fused_gemv.py`:
  - row-major prototype:
    - `q_proj`: cuBLAS `20.8us`, tiled prototype `390.6us`
    - `o_proj`: cuBLAS `19.9us`, tiled prototype `593.9us`
    - `down_proj`: cuBLAS `25.0us`, tiled prototype `1411.6us`
  - Marlin-qweight prototype:
    - scalar qvalue lookup version:
      - `q_proj`: cuBLAS `20.6us`, Marlin-qweight prototype `460.0us`
      - `o_proj`: cuBLAS `23.2us`, Marlin-qweight prototype `727.1us`
      - `down_proj`: cuBLAS `25.2us`, Marlin-qweight prototype `1701.3us`
    - compact packed-word decode version:
      - `q_proj`: cuBLAS `20.3us`, fused `164.6us`
      - `k_proj`: cuBLAS `16.9us`, fused `163.1us`
      - `v_proj`: cuBLAS `16.8us`, fused `163.6us`
      - `o_proj`: cuBLAS `20.5us`, fused `257.3us`
      - `gate_proj`: cuBLAS `28.5us`, fused `165.3us`
      - `up_proj`: cuBLAS `29.0us`, fused `166.4us`
      - `down_proj`: cuBLAS `27.0us`, fused `593.9us`
- Correctness on the compact packed-word decode path:
  - `N_fp4=64, N_fp8=64, K=256`
  - max abs error `0.030380`
  - mean abs error `0.003356`
  - relative error `0.2242%`
- `cuobjdump --dump-resource-usage` on
  [`rdquant/csrc/build/rdquant_cuda.so`](../rdquant/csrc/build/rdquant_cuda.so)
  reports the fused Marlin-qweight kernel at:
  - `REG=64`
  - `SHARED=1280`
  - `LOCAL=0`
- A split-K transition kernel with in-kernel reduction is now implemented:
  - `grid = (num_tiles, parallel_k)`
  - partial sums accumulate into a reusable FP32 workspace
  - one CTA per tile observes a completion counter and writes the final FP16
    result plus resets the workspace for the next launch
- With `parallel_k = K / 128`, this split-K prototype materially improves the
  Marlin-qweight transition path:
  - `q_proj`: cuBLAS `20.8us`, base `165.5us`, split-K `31.9us`
  - `k_proj`: cuBLAS `17.0us`, base `163.5us`, split-K `22.9us`
  - `v_proj`: cuBLAS `17.0us`, base `164.6us`, split-K `23.1us`
  - `o_proj`: cuBLAS `20.9us`, base `257.7us`, split-K `37.1us`
  - `gate_proj`: cuBLAS `28.9us`, base `166.1us`, split-K `49.7us`
  - `up_proj`: cuBLAS `28.8us`, base `166.4us`, split-K `65.1us`
  - `down_proj`: cuBLAS `26.6us`, base `592.9us`, split-K `70.6us`
- A direct benchmark against the current `2x marlin_gemm + concat + inv_perm`
  path shows the split-K prototype is already competitive:
  - `q_proj`: `2x Marlin 65.6us`, split-K `31.3us`
  - `k_proj`: `2x Marlin 58.8us`, split-K `22.5us`
  - `v_proj`: `2x Marlin 62.9us`, split-K `22.7us`
  - `o_proj`: `2x Marlin 66.4us`, split-K `35.6us`
  - `gate_proj`: `2x Marlin 62.9us`, split-K `48.7us`
  - `up_proj`: `2x Marlin 61.9us`, split-K `64.6us`
  - `down_proj`: `2x Marlin 99.5us`, split-K `64.3us`
- Breaking the current Marlin baseline into per-group kernel latencies makes the
  remaining gap easier to interpret:
  - `q_proj`: `Marlin NVFP4 23.7us`, `Marlin FP8 24.1us`, kernel sum `47.8us`,
    end-to-end `2x Marlin 57.7us`
  - `k_proj`: `Marlin NVFP4 31.9us`, `Marlin FP8 26.0us`, kernel sum `57.9us`,
    end-to-end `2x Marlin 57.8us`
  - `v_proj`: `Marlin NVFP4 25.4us`, `Marlin FP8 32.7us`, kernel sum `58.1us`,
    end-to-end `2x Marlin 58.0us`
  - `o_proj`: `Marlin NVFP4 23.4us`, `Marlin FP8 40.7us`, kernel sum `64.1us`,
    end-to-end `2x Marlin 63.6us`
  - `gate_proj`: `Marlin NVFP4 23.6us`, `Marlin FP8 24.2us`, kernel sum `47.7us`,
    end-to-end `2x Marlin 57.8us`
  - `up_proj`: `Marlin NVFP4 23.5us`, `Marlin FP8 25.6us`, kernel sum `49.2us`,
    end-to-end `2x Marlin 56.9us`
  - `down_proj`: `Marlin NVFP4 24.7us`, `Marlin FP8 74.3us`, kernel sum `99.0us`,
    end-to-end `2x Marlin 97.0us`
- This confirms the mixed fused path is mostly winning by eliminating a second
  high-fixed-cost launch, with the largest benefit showing up when one group is
  small enough to be a poor standalone Marlin decode kernel.
- A benchmark-side `parallel_k` sweep was added and confirms the current full
  split choice is already the best among a reasonable candidate set for all 7
  target shapes:
  - `q_proj`: best `parallel_k = 20`
  - `k_proj`: best `parallel_k = 20`
  - `v_proj`: best `parallel_k = 20`
  - `o_proj`: best `parallel_k = 32`
  - `gate_proj`: best `parallel_k = 20`
  - `up_proj`: best `parallel_k = 20`
  - `down_proj`: best `parallel_k = 76`
- In other words, `parallel_k = K / 128` is not just a placeholder heuristic for
  this prototype; on the current sweep it is already the best-performing choice.
  That makes the remaining bottleneck much more clearly an inner-loop issue than
  an outer scheduling issue.
- A smoke benchmark after extracting those dtype-specific tile entry points
  still passes correctness and preserves the same qualitative ranking:
  split-K remains far ahead of the base fused kernel, so this refactor is a
  clean structural step toward swapping in a Marlin tile engine.
- `cuobjdump --dump-resource-usage` for the split-K kernel reports:
  - `REG=56`
  - `SHARED=1284`
  - `LOCAL=0`
- The current `1 CTA = 1 output tile` scheduler is also fundamentally
  under-parallelized on RTX 5090 (`170` SMs):
  - `q_proj`: `32` tiles
  - `k_proj`: `8` tiles
  - `v_proj`: `8` tiles
  - `o_proj`: `20` tiles
  - `gate_proj`: `76` tiles
  - `up_proj`: `76` tiles
  - `down_proj`: `20` tiles

Conclusion: recovering K-direction parallelism was necessary and already buys a
large fraction of the gap back. But the remaining gap is still substantial for
large shapes, so the next step has to be switching this split-K transition path
from per-thread dot products to Marlin's actual vectorized shared-memory /
tensorcore inner loop.

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

1. Keep the new split-K outer scheduler and replace its per-thread dot-product
   loop with Marlin's tile engine for one shared decode config.
2. Replace the plain-scale transition path with Marlin-permuted scale loads and
   dtype-specific Marlin dequant/matmul loops.
3. Once the tile engine is in place, revisit the split-K reduction path so its
   workspace/lock semantics match Marlin more closely if that simplifies
   integration.

## What Not To Do

- Do not redesign weight layout from scratch.
- Do not keep optimizing the current row-major prototype.
- Do not put `if (is_fp4)` inside the `K` hot loop for mixed channels.
- Do not try to preserve fully independent autotuned Marlin configs inside one
  launch in v1.
