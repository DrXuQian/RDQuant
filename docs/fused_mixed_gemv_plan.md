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
- The staged qweight loaders now also issue those 16-byte copies through
  `cp.async`-family helpers instead of plain vector stores. This is still only
  a single-stage async copy pattern, not Marlin's full multi-stage pipeline,
  but it brings the qweight staging path one step closer to Marlin's actual
  global->shared transfer structure. On the latest RTX 5090 run this helped the
  base fused path more clearly than the split-K best path, which is consistent
  with the current split-K bottleneck already being deeper inside the compute
  loop.
- The split-K path is now exposed in two explicit variants:
  - the original best path where `NVFP4` still uses the scalar/global helper
  - an alternate `staged NVFP4` path where both dtype groups stage qweights
    through shared memory before the mixed inner loop
  Benchmarking them side by side is important because the staged NVFP4 path is
  not a uniform win: it helps some low-tile-count shapes (`k_proj`, `v_proj`,
  `down_proj`) but regresses others. The current benchmark now prints both
  variants and a per-layer `BestSK` result instead of forcing one policy.
- There is now also a small `splitk_auto` wrapper that dispatches between the
  two split-K variants with a conservative runtime heuristic. The current rule
  only sends shapes with very small total tile count and a non-trivial FP8 tile
  set to the `staged NVFP4` variant; everything else stays on the original
  scalar-NVFP4 path. This is intentionally narrower than the raw `BestSK`
  benchmark table because the per-shape timing deltas are small enough that a
  more aggressive heuristic is not robust yet.

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
- With the split-K benchmark now reporting both the original scalar-NVFP4 path
  and the staged-NVFP4 alternate path, the current per-layer best-of-two on
  RTX 5090 is approximately:
  - `q_proj`: `31.0us` (scalar)
  - `k_proj`: `21.4us` (staged NVFP4)
  - `v_proj`: `20.2us` (staged NVFP4)
  - `o_proj`: `35.8us` (scalar)
  - `gate_proj`: `43.6us` (scalar)
  - `up_proj`: `61.8us` (scalar)
  - `down_proj`: `63.2us` (staged NVFP4)
  This gives a current best-of-two total of roughly `277us` across the 7 target
  layers, which is modestly better than forcing the original split-K path on
  every layer.
- The benchmark now also fixes the random seed before generating test tensors.
  This removes one avoidable source of run-to-run jitter, which matters when
  comparing the two split-K variants because several of the per-layer deltas are
  only on the order of `0.5-1us`.
- The FP8 helper now has a double-buffered shared->register pipeline inside its
  `kBlockK=128` inner loop. Concretely, the helper preloads two `16-K`
  subtiles of packed FP8 words and their corresponding `x` fragments into
  registers, then alternates between consuming the current register stage and
  refilling the slot for the subtile two steps ahead. This is a structural move
  toward Marlin's register pipeline even though the current end-to-end latency
  impact is still mixed.
- The main split-K FP8 path now uses a lighter overlap scheme at `16-K` chunk
  granularity instead of double-buffering an entire `kBlockK=128` qweight tile.
  Concretely, each loop iteration stages one `16-K` FP8 chunk into shared,
  prefetches only the next `16-K` chunk with `cp.async`, and alternates between
  two small chunk buffers while the current chunk is being computed.
- That lighter chunk overlap keeps the FP8 fetch/compute overlap structure
  without paying the cost of duplicating the full FP8 shared tile. After this
  change, `cuobjdump` reports:
  - scalar-NVFP4 split-K kernel: `REG=56`, `SHARED=5380`, `LOCAL=0`
  - staged-NVFP4 split-K kernel: `REG=64`, `SHARED=10500`, `LOCAL=0`
  This is materially smaller than the earlier whole-`kBlockK` overlap version,
  which had pushed the main split-K kernel up to `REG=64`, `SHARED=34052`.
- A smoke benchmark after extracting those dtype-specific tile entry points
  still passes correctness and preserves the same qualitative ranking:
  split-K remains far ahead of the base fused kernel, so this refactor is a
  clean structural step toward swapping in a Marlin tile engine.
- A smoke benchmark on the lighter FP8 chunk-overlap version still passes
  correctness and keeps the split-K path near the current best range:
  - `q_proj`: `SplitK 30.9us`, `Split4S 31.1us`, `AutoSK 30.7us`
  - `k_proj`: `SplitK 20.4us`, `Split4S 20.4us`, `AutoSK 20.4us`
  - `v_proj`: `SplitK 20.4us`, `Split4S 20.3us`, `AutoSK 20.4us`
  - `o_proj`: `SplitK 36.5us`, `Split4S 36.8us`, `AutoSK 36.4us`
  - `gate_proj`: `SplitK 47.1us`, `Split4S 52.8us`, `AutoSK 47.2us`
  - `up_proj`: `SplitK 63.8us`, `Split4S 64.5us`, `AutoSK 63.7us`
  - `down_proj`: `SplitK 70.6us`, `Split4S 64.5us`, `AutoSK 63.2us`
- This means the lighter chunk overlap preserves most of the useful FP8 overlap
  structure while removing the pathological shared-memory blow-up from the
  earlier whole-`kBlockK` ping-pong design.
- Remote `ncu` profiling is now wired up on a separate RTX 5070 WSL machine,
  and the first four targeted reports are in place for:
  - `q_proj` scalar `SplitK`
  - `q_proj` `Split4S`
  - `down_proj` scalar `SplitK`
  - `down_proj` `Split4S`
- The most important observations from those reports are:
  - `q_proj` scalar `SplitK`: `41.09us`, `REG=56`, `static shared=4.36KB`,
    achieved occupancy `58.62%`, compute throughput `57.70%`, DRAM throughput
    `31.04%`
  - `q_proj` `Split4S`: `40.86us`, `REG=64`, `static shared=9.48KB`, achieved
    occupancy `55.64%`
  - `down_proj` scalar `SplitK`: `120.10us`, achieved occupancy `68.59%`,
    compute throughput `90.29%`, DRAM throughput only `18.44%`
  - `down_proj` `Split4S`: `122.91us`, achieved occupancy `61.69%`
- In other words, `down_proj` is clearly not DRAM-saturated; it is much more
  compute/synchronization heavy than memory-bandwidth limited. That made it a
  good target for a slightly wider FP8 overlap granularity, but only when the
  FP8 group is small enough to justify the extra shared-memory footprint.
- The main scalar-NVFP4 split-K path now has two physical FP8 chunk variants:
  - the original `16-K` chunk version at `REG=56`, `SHARED=5380`
  - a `32-K` wide-FP8 version at `REG=56`, `SHARED=9476`
- The host launch path selects the wide-FP8 kernel only for small FP8 groups
  (`N_fp8 <= 128` with `K >= 4096`), which covers the `o_proj` and
  `down_proj`-style shapes without forcing the larger shared footprint on the
  broader set of layers where the original `16-K` chunk version remains the
  safer default.
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
