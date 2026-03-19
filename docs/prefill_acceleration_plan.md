# Prefill Acceleration Plan — Overnight Execution

## Problem
RDQuant prefill is 1.6× slower than BF16 at kernel level (12.2ms vs 7.6ms GEMM).
Root cause: 2× Marlin GEMM launches per layer (NVFP4 + FP8).

## Experiments

### Exp 1: Triton single GEMM baseline
- Write a Triton FP16 matmul kernel for [M=128, K=9728, N=2560]
- Compare vs cuBLAS F.linear
- Decision: if Triton < 1.5× cuBLAS → proceed with fused; else → skip to CUDA

### Exp 2: Triton fused mixed GEMM (if Exp 1 passes)
- Single Triton kernel that reads from 2 weight buffers (dequanted FP16)
- Different tiles read different weight pointers based on N coordinate
- Fuse inv_perm into the store (scatter output)
- Benchmark vs 2× Marlin and vs cuBLAS

### Exp 3: Triton with FP8 in-kernel dequant
- FP8 group stored as FP8, loaded with tl.load → auto upcast
- NVFP4 group still dequanted FP16
- Measure bandwidth savings

### Exp 4: CUTLASS grouped GEMM
- Use our existing CUTLASS w4a8/w8a8 kernels
- Try sequential launch with CUDA streams for overlap
- Compare vs serial launch

### Exp 5: cuBLAS stream overlap
- Launch NVFP4 Marlin and FP8 Marlin on separate CUDA streams
- They operate on different weight data → can overlap
- Measure with CUDA events

### Exp 6: FP16 allocation check
- Verify FP16 is in the search space and check allocation on Qwen3-4B
- If FP16 channels exist → need 3-way fused kernel

### Exp 7: Reduced FP8 ratio
- Try budget=4.5 bpw → mostly NVFP4, tiny FP8 group
- Measure prefill with this allocation

## Success Criteria
- Find a method that makes prefill ≤ 1.0× BF16 (at least parity)
- Or understand the fundamental limit and document it
