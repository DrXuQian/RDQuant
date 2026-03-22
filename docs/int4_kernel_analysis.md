# INT4/INT8 Decode Kernel Analysis

## Scope

This note analyzes the `M=1` decode path for the current `INT4/INT8` mixed
deployment and compares it against the existing `NVFP4/FP8 fused GEMV` path.

The goal is to explain why:

- `NVFP4/FP8 fused GEMV` reaches about `4.18 ms/tok`
- `INT4/INT8 Marlin` stays at about `6.63 ms/tok`

and to rank the optimizations that are actually worth pursuing.

The analysis below uses:

- the measured CUDA Graph numbers provided in the task description
- the current `INT4/INT8` implementation in
  [rdquant/int4_marlin.py](/root/autodl-tmp/rdquant/rdquant/int4_marlin.py)
- the fused helper kernels in
  [rdquant/csrc/int4_postprocess.cu](/root/autodl-tmp/rdquant/rdquant/csrc/int4_postprocess.cu)
- the fake-quant / packing path in
  [rdquant/int4_fusion.py](/root/autodl-tmp/rdquant/rdquant/int4_fusion.py)
- the successful `M=1` fused reference architecture in
  [rdquant/csrc/fused_gemv.cu](/root/autodl-tmp/rdquant/rdquant/csrc/fused_gemv.cu)

## Current Decode Paths

### NVFP4/FP8 fused GEMV

The current fused decode path is a decode-oriented, non-persistent, tiled
kernel. Its key structural properties are visible at the top of
[rdquant/csrc/fused_gemv.cu](/root/autodl-tmp/rdquant/rdquant/csrc/fused_gemv.cu#L1):

- `M=1` specialization
- one CTA per output tile
- standard grid launch, not a persistent kernel
- activation tile staged once and reused across channels
- `inv_perm` fused into the write path
- dequant + GEMV + output scatter all stay inside the same kernel family

In other words, it avoids both:

- the fixed persistent-kernel warmup tax
- the extra graph nodes for post-processing / reorder

### INT4/INT8 Marlin path

The current `INT4/INT8` decode path is implemented in
[rdquant/int4_marlin.py](/root/autodl-tmp/rdquant/rdquant/int4_marlin.py#L155)
and has three stages per layer:

1. fused pre-kernel:
   [fused_awq_sum_v2_kernel](/root/autodl-tmp/rdquant/rdquant/csrc/int4_postprocess.cu#L21)
   scales `x` by `1/alpha` and computes `sum_x`
2. one `marlin_gemm(uint4b8)` over the combined weight matrix
3. fused post-kernel:
   [fused_post_v2_kernel](/root/autodl-tmp/rdquant/rdquant/csrc/int4_postprocess.cu#L124)
   reconstructs INT8 output and applies `inv_perm`

The combined weight layout is built in
[pack_for_marlin()](/root/autodl-tmp/rdquant/rdquant/int4_marlin.py#L69):

- `W_combined = [W_int4 | W_int8_high | W_int8_low]`
- `S_combined = [S_int4 | 16*S_int8 | S_int8]`

and the forward path in
[Int4MarlinLinear._forward_fused()](/root/autodl-tmp/rdquant/rdquant/int4_marlin.py#L227)
is:

1. `x <- x / alpha`, `sum_x`
2. `y_combined = marlin_gemm(x, W_combined, S_combined, uint4b8)`
3. `y_int8 = y_high + y_low + 8 * s_int8 * sum_x`
4. `cat/int8-reconstruct + inv_perm`

## Time Breakdown

### Measured totals

| Path | ms/tok | Notes |
|---|---:|---|
| NVFP4/FP8 fused GEMV | 4.18 | 1 launch/layer, non-persistent |
| INT4/INT8 Marlin | 6.63 | 1 Marlin launch + 2 fused helper kernels/layer |
| BF16 cuBLAS | 8.03 | dense baseline |

### INT4/INT8 path decomposition

Using the provided measurements:

| Component | Time | Share |
|---|---:|---:|
| Marlin GEMM | 3.70 ms | dominant |
| Fused pre (AWQ + sum_x) | ~0.25 ms | small |
| Fused post (correction + perm) | ~0.25 ms | small |
| Non-GEMM shared work | ~2.46 ms | same floor as other paths |
| Total | 6.63 ms | |

### What this already implies

The first-order conclusion is immediate:

- the extra pre/post kernels are not the main problem
- the main problem is the Marlin decode kernel itself

Even if pre and post were free, the path would still be around:

- `6.63 - 0.25 - 0.25 = 6.13 ms`

which is still far from `4.18 ms`.

## Root Cause

## 1. The dominant bottleneck is persistent-kernel warmup

The provided observation is the critical one:

- Marlin decode launch cost is about `14.7 us`
- it is roughly flat from `N=1024` to `N=14000`

This is classic persistent-kernel behavior: the warmup / pipeline setup dominates
the tiny `M=1` GEMV work.

For 252 linear layers:

- `252 * 14.7 us = 3.70 ms`

This alone explains most of the `INT4/INT8` GEMM time.

By contrast, the `NVFP4/FP8` fused decode kernels are standard grid launches
with a much smaller per-layer overhead, around a few microseconds.

So the primary gap is:

- not arithmetic throughput
- not memory bandwidth saturation
- not the two tiny helper kernels

It is the fact that `INT4/INT8` still pays a persistent-kernel warmup 252 times.

## 2. The `INT8 -> 2 x UINT4` decomposition is a secondary cost

The current layout in
[rdquant/int4_marlin.py](/root/autodl-tmp/rdquant/rdquant/int4_marlin.py#L69)
expands the INT8 group:

- each INT8 channel becomes two channels: `high` and `low`
- effective output width becomes:
  `N_combined = N_int4 + 2 * N_int8`

That has two consequences:

1. the main GEMM touches more columns than the final logical output width
2. the post-kernel must reconstruct INT8 outputs from `high/low`

This is real extra work, but it is not the current dominant term.

Why:

- the observed Marlin kernel latency is already almost flat vs shape
- so today the persistent warmup hides a lot of the added compute

Still, this decomposition matters for the next stage. If the persistent launch
tax is removed, the `2 x INT8` expansion will become more visible. In other
words:

- today it is a secondary bottleneck
- in a future non-persistent fused kernel it becomes a more important second-order term

## 3. The helper kernels are not the right optimization target

The pre-kernel in
[rdquant/csrc/int4_postprocess.cu](/root/autodl-tmp/rdquant/rdquant/csrc/int4_postprocess.cu#L21)
does:

- in-place `x *= 1 / alpha`
- reduction for `sum_x`

The post-kernel in
[rdquant/csrc/int4_postprocess.cu](/root/autodl-tmp/rdquant/rdquant/csrc/int4_postprocess.cu#L124)
does:

- reconstruct INT8 output
- apply `inv_perm`

These are already reasonably optimized:

- vectorized `half2` loads/stores
- shared-memory path for tiny `M`
- `int32` permutation indices

Their total cost is only about `0.5 ms`.

That means squeezing another `2x` out of pre/post only saves about `0.25 ms`,
which does not change the ranking of the whole decode path.

## Side-by-side Comparison

| Item | NVFP4/FP8 fused GEMV | INT4/INT8 Marlin |
|---|---|---|
| Main launch style | standard grid | persistent Marlin |
| Kernels per layer | 1 | 3 |
| `inv_perm` handling | fused on store | separate post-kernel |
| AWQ scaling | not needed | separate pre-kernel |
| INT8 correction | not needed | separate post step |
| Weight formats inside main kernel | mixed NVFP4 + FP8 | unified UINT4 after INT8 decomposition |
| Effective GEMM width | `N_fp4 + N_fp8` | `N_int4 + 2*N_int8` |
| Dominant overhead | low launch + compute | persistent warmup |

## Bottleneck Attribution

Ordered by importance:

| Rank | Bottleneck | Importance | Why |
|---|---|---:|---|
| 1 | Marlin persistent warmup | very high | `3.70 ms` by itself |
| 2 | `INT8 -> 2 x UINT4` expansion | medium | increases `N_combined`, compute, and postprocess work |
| 3 | pre/post CUDA Graph nodes | low-medium | only about `0.5 ms` total |
| 4 | AWQ scaling implementation details | low | folded into the small pre-kernel term |

## Optimization Directions

Sorted by expected gain.

### 1. Write a fused `INT4` decode kernel and replace Marlin for `M=1`

Expected gain: very high

This is the only change that directly attacks the dominant `3.70 ms` term.

The fused kernel should:

- use a standard grid launch, not a persistent scheduler
- fuse `x / alpha` into the activation load path
- accumulate `sum_x` inside the same kernel
- run the main `INT4` GEMV
- reconstruct INT8 output in the epilogue
- write final output in original order using `inv_perm`

If successful, this should remove:

- the Marlin persistent warmup cost
- the separate pre-kernel
- the separate post-kernel

This is the only realistic path to move the decode time materially toward the
`NVFP4/FP8 fused GEMV` regime.

### 2. Reuse the existing decomposition first, then optimize away the decomposition

Expected gain: high for step 1, medium for step 2

There are two implementation levels:

#### Level A: fastest path to a better kernel

Keep the current deployment contract:

- keep `W_combined = [W_int4 | W_high | W_low]`
- keep `S_combined = [S_int4 | 16*S_int8 | S_int8]`
- keep the current correction formula

Then only replace the execution engine:

- Marlin persistent kernel -> custom fused non-persistent decode kernel

This minimizes checkpoint / packing churn and should already capture most of
the missing decode performance, because it removes the warmup wall.

#### Level B: remove the `2 x INT8` expansion

After Level A, revisit the representation itself:

- avoid materializing `high` and `low` as separate logical output channels
- handle INT8 channels directly in the fused main loop or in a specialized tile path

This is harder, but it attacks the next bottleneck that will remain after the
persistent warmup is gone.

### 3. Fuse pre/post more tightly only if a new main kernel is being written anyway

Expected gain: medium as part of a fused-kernel rewrite, low as a standalone effort

Trying to optimize:

- `fused_awq_sum_v2_kernel`
- `fused_post_v2_kernel`

without replacing Marlin is not attractive. Their maximum upside is too small.

But once a new fused main kernel exists, the right design is to eliminate both
helper kernels entirely by absorbing their work into:

- activation load / staging
- epilogue writeback

### 4. Continue micro-optimizing the helper kernels

Expected gain: low

Possible, but not the right priority:

- more vectorization
- less sync
- slightly better `sum_x` reduction
- slightly better post-kernel scatter

These do not address the `3.70 ms` dominant term.

### 5. Keep Marlin and hope for tuning wins

Expected gain: very low

This is unlikely to close the gap.

The problem is not that the current Marlin call is poorly tuned. The problem is
that the persistent-kernel model is fundamentally mismatched to tiny `M=1`
decode GEMV.

## Is a fused INT4 GEMV kernel worth writing?

Yes, if decode latency is the priority.

Reason:

- the current path already did the easy graph-node fusion work
- the remaining dominant cost is the Marlin launch model itself
- the `NVFP4/FP8` fused path has already demonstrated that a non-persistent,
  single-launch, decode-specialized kernel can beat the persistent Marlin model

In other words, the remaining gap is large enough that a custom fused kernel is
not premature optimization. It is the main optimization still left.

## How would a fused INT4 decode kernel differ from `fused_gemv.cu`?

It would be similar in outer structure but simpler in dtype dispatch.

### Similarities

It should reuse the same big ideas from
[rdquant/csrc/fused_gemv.cu](/root/autodl-tmp/rdquant/rdquant/csrc/fused_gemv.cu#L1):

- `M=1` specialization
- standard grid launch
- CTA-per-output-tile scheduling
- shared-memory staging of `x`
- optional split-K if needed
- epilogue writes in original channel order

### Differences

Unlike `NVFP4/FP8 fused GEMV`, the `INT4/INT8` decode kernel does not need two
fundamentally different quantization engines in the main loop.

If it keeps the current decomposition, then the main GEMV math is actually more
uniform:

- every packed weight is `uint4`
- every channel dequantizes as `(q - 8) * scale`

That means:

- no tile-level branch between NVFP4 and FP8 engines
- no dual scale-format contracts like `NVFP4 block-scale` vs `FP8 channel-scale`

The complexity moves elsewhere:

- AWQ scaling must be fused into activation handling
- `sum_x` must be accumulated
- INT8 logical outputs must be reconstructed from `high/low`
- `inv_perm` must be applied after reconstruction

So the trade-off is:

- main dequant path simpler than `NVFP4/FP8`
- epilogue more specialized than `NVFP4/FP8`

## Recommended Execution Plan

### V1

Build a decode-only fused `INT4` kernel that keeps the current packed
representation:

- same `W_combined`
- same `S_combined`
- same correction rule

This is the lowest-risk way to eliminate:

- persistent Marlin warmup
- pre-kernel launch
- post-kernel launch

### V2

If V1 lands near the `NVFP4/FP8` path but still trails it, the next most likely
remaining reason is the `2 x INT8` expansion. Then the next step is:

- redesign the fused kernel so INT8 channels are not represented as two logical
  output channels

## Bottom Line

The current `INT4/INT8` decode path is slower than `NVFP4/FP8 fused GEMV`
primarily because it still depends on a persistent Marlin kernel whose fixed
warmup cost dominates `M=1`.

The pre/post fused kernels are not the main problem.

The most valuable next optimization is:

- replace the Marlin decode call with a custom non-persistent fused `INT4`
  decode kernel

and the practical first version should:

- keep the current combined weight layout
- fuse AWQ scaling, `sum_x`, INT8 correction, and `inv_perm`
- reuse the successful outer scheduling ideas from `fused_gemv.cu`

That is the clearest path to closing most of the remaining gap.
