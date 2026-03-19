# Prefill Acceleration Experiments

RTX 5090 (Blackwell SM120), K=9728, N=2560

## Problem
RDQuant prefill: 2× Marlin GEMM per layer → 1.6× slower than BF16 cuBLAS.

## Exp 1: Triton single GEMM vs cuBLAS

```
    M     cuBLAS   Triton    ratio
    1      16.5    243.0    14.76x  ❌
   16      30.7    241.8     7.87x  ❌
  128      36.9    244.6     6.64x  ❌
  512     114.6    247.7     2.16x  ❌
 2048     448.7    473.6     1.06x  ≈
```

**Conclusion**: Triton on SM120 has ~240μs fixed overhead. Only competitive at M≥2048. **Triton fused GEMM is NOT viable for prefill.**

## Exp 5: CUDA stream overlap

```
    M    single    seq    overlap  speedup
    1     16.6    47.7     73.3    0.65x  ❌
  128     36.9    55.3     79.3    0.70x  ❌
 2048    463.6   529.3    567.3    0.93x  ❌
```

**Conclusion**: Stream overlap is worse than sequential. Two GEMMs compete for the same SM resources, synchronization adds overhead. **Not viable.**

## Exp 6: FP16 allocation in Qwen3-4B

```
budget=4.0:  NVFP4=100%  FP8=0%  FP16=0%  bpw=4.00
```

FP16 only gets allocated at very high budgets (>12 bpw). At practical budgets (4-8), it's all NVFP4+FP8. FP16 is in the search space but the allocator never chooses it because FP8 is nearly lossless.

## Exp 7: Varying NVFP4/FP8 ratio (M=128)

```
FP4%   FP8%   bpw   mixed(μs)  single(μs)  ratio
 50     50    6.0     53.3       36.8       1.45x
 65     35    5.4     55.2       36.8       1.50x
 80     20    4.8     53.7       36.9       1.46x
 90     10    4.4     53.7       36.9       1.46x
 95      5    4.2     55.3       36.8       1.50x
100      0    4.0     43.0       36.8       1.17x
```

**Key finding**: Even with FP8 at 5%, mixed is still ~1.46× cuBLAS. The overhead is NOT proportional to FP8 group size — it's the **second kernel launch** itself (~15μs) plus cat + index_select. Only at FP4=100% (single GEMM) does overhead drop to 1.17× (just the Marlin dequant cost vs native cuBLAS).

## Conclusions

1. **Triton is not viable on SM120** for M≤512 (6.6× slower than cuBLAS)
2. **Stream overlap doesn't help** (shared GPU resources)
3. **Reducing FP8 ratio doesn't help** (overhead is per-launch, not proportional to N)
4. **The fundamental issue**: any 2-launch approach adds ~15-20μs overhead per layer that scales with #layers (252)

## Remaining Options

| Option | Expected improvement | Complexity |
|---|---|---|
| CUTLASS grouped GEMM | Best: 1 launch, ~1.0× cuBLAS | High (custom CUTLASS) |
| Uniform NVFP4 for prefill | 1.17× cuBLAS (Exp 7) | Zero code change |
| Prefill at FP16 dequanted | 1.0× cuBLAS (pre-materialize) | Low |

The most practical path: **prefill uses pre-materialized FP16 weights** (single cuBLAS GEMM, no overhead), **decode uses Marlin** (2.59× faster). Weight materialization happens once at prefill start.
