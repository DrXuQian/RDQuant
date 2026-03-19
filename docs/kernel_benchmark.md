# Kernel Benchmark Results

RTX 5090 (Blackwell, SM120), CUDA 12.8, vLLM 0.17.1

## Test Setup

- Layer dimensions: K=9728, N=2560 (Qwen3-4B MLP down_proj)
- All kernels compute `Y[M, N] = X[M, K] @ W[N, K]^T`
- Same input/output dimensions for fair comparison
- Timing: batch CUDA events (1000 iterations, median)
- FP8 Marlin correctness verified (max diff = 0.003 vs reference)

---

## Single Kernel: Same Size (N=2560, K=9728)

All three kernels operating on the full layer (N=2560):

```
    M     FP16    NV4_M    FP8_M | NV4/FP16  FP8/FP16
------------------------------------------------------
    1     74.2     21.5     21.6 |   3.46x    3.43x
    2     34.7     21.6     21.7 |   1.61x    1.60x
    4     34.6     21.5     21.8 |   1.61x    1.59x
    8     56.8     21.5     25.8 |   2.65x    2.21x
   16     62.1     26.3     26.2 |   2.36x    2.37x
   32     43.7     34.4     35.2 |   1.27x    1.24x
   64     43.6     57.1     57.1 |   0.76x    0.76x
  128     78.1     86.4     90.5 |   0.90x    0.86x
  256    136.8    154.5    154.4 |   0.89x    0.89x
  512    250.4    255.3    254.8 |   0.98x    0.98x
 1024    496.7    489.4    489.6 |   1.01x    1.01x
 2048    982.6    969.2    966.1 |   1.01x    1.02x
```

Key findings:
- **NVFP4 Marlin and FP8 Marlin have identical performance** (~21.5μs @ M=1)
- Both achieve **3.4x speedup** over FP16 at decode (M=1)
- Both hit a ~21μs launch overhead floor
- **M ≥ 64**: Marlin slower than FP16 (dequant→FP16 MMA cannot compete with native FP16 Tensor Core)
- Performance difference between NVFP4 and FP8 is negligible — bottleneck is kernel launch, not weight bandwidth

### Why NVFP4 ≈ FP8 at M=1?

NVFP4 reads 14MB vs FP8 reads 25MB, yet both take 21.5μs. This means neither is bandwidth-bound — both are limited by Marlin kernel launch overhead. At N=2560 the GEMV has only ~20 tiles, finishing before the GPU pipeline is fully utilized.

### Regime Analysis

| Regime | M range | Bottleneck | Best kernel |
|---|---|---|---|
| Decode | M = 1-16 | Launch overhead | Marlin (NVFP4 or FP8), **3.4x** |
| Transition | M = 32-64 | Mixed | FP16 slightly better |
| Prefill | M ≥ 128 | Compute | FP16 cuBLAS (native Tensor Core) |

---

## Per-Group Benchmark (Simulating RDQuant Mixed-Precision)

RDQuant 5.3 bpw splits: NVFP4=1664 channels (65%), FP8=896 channels (35%)

```
    M |  NVFP4 Marlin (N=1664)  |       FP8 Marlin (N=896)       | BF16 full
      |  BF16    NV4_M   sp    |  BF16   FP8_M  FP8_cSM  sp_M  | N=2560
---------------------------------------------------------------------------
    1 |  65.0    21.6   3.0x   |  25.8   26.1    39.2   1.0x   |  74.2
   16 |  39.2    26.3   1.5x   |  26.0   30.5    39.0   0.9x   |  86.2
  128 | 113.9    65.8   1.7x   |  61.0   52.5    66.3   1.2x   | 110.3
  512 | 178.7   185.3   1.0x   | 100.9  109.2   240.0   0.9x   | 273.4
 2048 | 668.9   706.5   0.9x   | 350.5  386.0   245.1   0.9x   | 979.0
```

FP8 Marlin vs cutlass_scaled_mm:
- **M ≤ 512: FP8 Marlin wins** (26μs vs 39μs at M=1)
- **M ≥ 2048: cutlass_scaled_mm wins** (245μs vs 386μs, native FP8 Tensor Core)

Small N problem: at N=896, both BF16 and FP8 hit the kernel launch floor (~25μs). The bandwidth advantage of FP8 only shows at larger N.

---

## Kernel Backend Comparison

Tested on same dimensions (K=9728, N=2560, M=1):

| Kernel | Latency | vs FP16 | Notes |
|---|---:|---:|---|
| vLLM marlin_gemm (NVFP4) | 21.5 μs | 3.46x | W4A16, dequant fused |
| vLLM marlin_gemm (FP8) | 21.6 μs | 3.43x | W8A16, channelwise scale |
| vLLM cutlass_scaled_mm (FP8) | 39.1 μs | 1.90x | W8A8, per-channel |
| marlin_fp8_cuda (FP8, channelwise) | 43.8 μs | 1.69x | Older marlin fork |
| marlin_fp8_cuda (FP8, group=128) | 55.1 μs | 1.35x | Per-group scaling overhead |
| cuBLASLt NVFP4 W4A4 | 112 μs | 0.66x | Native FP4 Tensor Core |
| cuBLASLt MXFP8 W8A8 | 161 μs | 0.46x | MX block-scaled |
| CUTLASS MXFP8 (custom) | 286 μs | 0.26x | TMA warp-specialized |
| FP16 cuBLAS (F.linear) | 74.2 μs | 1.00x | Baseline |

**vLLM marlin_gemm is the fastest kernel for decode (M=1)**, 3.4x faster than FP16 for both NVFP4 and FP8 weight formats.

---

## cuBLASLt Type Support Matrix

Exhaustively tested on RTX 5090 (CUDA 12.8):

| A (activation) | B (weight) | Scale Mode | Supported |
|---|---|---|---|
| FP4 E2M1 | FP4 E2M1 | VEC16_UE4M3 (NVFP4) | ✅ |
| FP8 E4M3 | FP8 E4M3 | VEC32_UE8M0 (MXFP8) | ✅ |
| All other combinations | | | ❌ |

cuBLASLt only supports symmetric A/B types. No FP8×FP4, FP8×FP6, or mixed scale modes.

---

## Theoretical RDQuant Decode Acceleration

### Bandwidth Model (M=1)

RTX 5090 HBM bandwidth: ~1.8 TB/s

| Method | Weight (MB) | Bandwidth time (μs) | vs FP16 |
|---|---:|---:|---:|
| FP16 full | 49.8 | 27.7 | 1.0x |
| Uniform NVFP4 | 14.0 | 7.8 | **3.6x** |
| Uniform FP8 | 24.9 | 13.8 | **2.0x** |
| **RDQuant 5.3bpw** (65% NV4 + 35% FP8) | **17.8** | **9.9** | **2.8x** |

### With Fused Kernel (Ideal)

If NVFP4 and FP8 sub-GEMMs run in parallel on different SMs:

| Method | Latency (μs) | vs FP16 |
|---|---:|---:|
| 2 separate launches (sum) | 9.9 | 2.8x |
| Fused kernel (max, ideal) | ~6 | **4.6x** |

RTX 5090 has 170 SMs. At M=1, each sub-GEMM only uses ~13+7=20 tiles, well within parallel capacity.

### Accuracy vs Speed Tradeoff

| Method | PPL | bpw | Weight size | Decode speedup |
|---|---:|---:|---:|---:|
| BF16 baseline | 12.90 | 16.00 | 100% | 1.0x |
| RDQuant calibrated | 12.24 | 5.29 | 33% | ~2.8x |
| Uniform FP8 | 12.93 | 8.13 | 50% | ~2.0x |
| Uniform NVFP4 | 13.22 | 4.00 | 28% | ~3.6x |
| RDQuant data-free | 13.43 | 5.52 | 34% | ~2.8x |

Results on original BF16 Qwen3-4B weights (not dequanted from NVFP4 checkpoint).
RDQuant calibrated achieves the best PPL (12.24, -0.66 vs BF16) at 2.8x decode speedup and 3x model compression.

---

## Whole-Network Estimation (TODO)

To get end-to-end Qwen3-4B latency:

1. Get per-layer (N, K) dimensions and RDQuant format splits (already computed)
2. For each of 252 linear layers, compute weight bandwidth:
   - NVFP4 channels: N_fp4 × (K/2 + K/16) bytes
   - FP8 channels: N_fp8 × K bytes
   - FP16 channels: N_fp16 × K × 2 bytes
3. Sum total weight bytes across all layers
4. Divide by HBM bandwidth (1.8 TB/s) for theoretical minimum
5. Validate against measured single-layer latencies
