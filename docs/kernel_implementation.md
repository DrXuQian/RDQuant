# RDQuant Kernel Implementation

## Overview

RDQuant uses CUTLASS 3.x block-scaled GEMM kernels on NVIDIA Blackwell (SM120) to execute mixed-precision inference. Each linear layer's output channels are grouped by format (MXFP4/MXFP6/MXFP8), and one GEMM kernel is launched per group. Activations are uniformly quantized to MXFP8 once per layer.

```
Inference path for one linear layer:

x_bf16 [M, K]
   │
   ▼ quantize_act_mxfp8()                    ← CUDA kernel, ~3μs
x_mxfp8 [M, K] + x_sf [M, K/32]
   │
   ├─► mx_gemm_w4a8(x, W_fp4)  → y4 [M, N4]  ← CUTLASS tcgen05.mma.blockscaled
   ├─► mx_gemm_w6a8(x, W_fp6)  → y6 [M, N6]  ← CUTLASS tcgen05.mma.blockscaled
   ├─► mx_gemm_w8a8(x, W_fp8)  → y8 [M, N8]  ← CUTLASS tcgen05.mma.blockscaled
   │
   ▼ cat + index_select(inv_perm)
y_bf16 [M, N]
```

---

## Hardware

- **GPU**: NVIDIA RTX 5090 (Blackwell, SM120)
- **Native instruction**: `tcgen05.mma.blockscaled` — hardware decodes sub-byte MX formats (FP4/FP6/FP8) with UE8M0 per-32-element shared exponents directly in the MMA unit, no separate dequantization kernel needed
- **CUDA Toolkit**: 12.8
- **Compilation target**: `sm_120a` (the `a` suffix is required to define `__CUDA_ARCH_FEAT_SM120_ALL`, which CUTLASS block-scaled kernels check at runtime)

---

## Kernel Inventory

| Kernel | Function | A (activation) | B (weight) | Output |
|---|---|---|---|---|
| `mx_gemm_w4a8` | MXFP8 × MXFP4 GEMM | E4M3 + UE8M0 | E2M1 + UE8M0 | BF16 |
| `mx_gemm_w6a8` | MXFP8 × MXFP6 GEMM | E4M3 + UE8M0 | E3M2 + UE8M0 | BF16 |
| `mx_gemm_w8a8` | MXFP8 × MXFP8 GEMM | E4M3 + UE8M0 | E4M3 + UE8M0 | BF16 |
| `quantize_act_mxfp8` | BF16 → MXFP8 | BF16 input | — | E4M3 + UE8M0 |
| `reorder_sf` | Scale factor layout conversion | row-major | — | CUTLASS interleaved |

---

## CUTLASS Template Configuration

All three GEMM kernels share the same CUTLASS template structure, differing only in the weight element type and alignment.

### Common parameters

```cpp
using ArchTag        = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;  // M, N, K tile
using ClusterShape     = Shape<_1, _1, _1>;
using ElementAccumulator = float;                   // FP32 accumulation
using ElementD           = cutlass::bfloat16_t;     // BF16 output

// Epilogue: alpha=1.0, beta=0.0 (no accumulation — RDQuant concatenates outputs)
```

### Per-kernel element types

| Kernel | ElementA | ElementB | AlignmentA | AlignmentB |
|---|---|---|---|---|
| w4a8 | `mx_float8_t<float_e4m3_t>` | `mx_float4_t<float_e2m1_t>` | 16 | 32 |
| w6a8 | `mx_float8_t<float_e4m3_t>` | `mx_float6_t<float_e3m2_t>` | 16 | 128 |
| w8a8 | `mx_float8_t<float_e4m3_t>` | `mx_float8_t<float_e4m3_t>` | 16 | 16 |

### CUTLASS builder pattern

```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,        // activation: RowMajor
    ElementB, LayoutB, AlignmentB,        // weight: ColumnMajor
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    StageCountAutoCarveout<...>,
    KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

### Key design choice: alpha=1, beta=0

Unlike MicroMix (which splits along K and uses `beta=1.0` to accumulate partial sums), RDQuant splits along N — each GEMM produces a disjoint subset of output channels. Therefore:
- **No accumulation needed** — `beta=0.0`, output written directly
- Results are **concatenated** along the output dimension, not summed
- An `inv_permutation` restores the original channel order

---

## Scale Factor Layout

### The problem

CUTLASS SM120 block-scaled MMA expects scale factors in a specific interleaved layout (`SfKMajorAtom`), not simple row-major storage. Our Python quantizer produces row-major `[dim0, K/32]` UE8M0 scales, so a conversion kernel is needed.

### CUTLASS expected layout

For tile parameters BM=128, BN=128, BK=128, SFVecSize=32:

```
SfKMajorAtom:
  Shape:  ((BM/4, 4), (SFVecSize, 4))  = ((32, 4), (32, 4))
  Stride: ((16,   4), (0,          1))
```

Physical offset within one atom (512 bytes):
```
offset = (row_in_tile % 32) * 16 + ((row_in_tile / 32) % 4) * 4 + k_in_block
```

Atoms are tiled with `Step<_2, _1>` — K-blocks vary fastest (inner), row-tiles vary slowest (outer).

### Buffer sizes

```
SFA (activation): ceil(M / 128) * 128 * (K / 32)  bytes
SFB (weight):     N * (K / 32)  bytes
```

Note: activation SF is padded to 128-row tiles. Weight SF is not (N is already 128-aligned by the allocator).

### Reorder kernel

`reorder_sf_kernel` in `sf_layout.cu` converts row-major → CUTLASS interleaved:

```cuda
int row_tile    = row / 128;
int row_in_tile = row % 128;
int k_block     = kt / 4;
int k_in_block  = kt % 4;

int offset_in_atom = (row_in_tile % 32) * 16
                   + ((row_in_tile / 32) % 4) * 4
                   + k_in_block;

int dst_idx = (row_tile * num_k_blocks + k_block) * 512 + offset_in_atom;
```

---

## Weight Packing Format

| Format | Packing | Tensor shape | Bytes per K elements |
|---|---|---|---|
| MXFP4 (E2M1) | 2 values per byte | `[N, K/2]` | K/2 |
| MXFP6 (E3M2) | 4 values per 3 bytes | `[N, K*3/4]` | 3K/4 |
| MXFP8 (E4M3) | 1 value per byte | `[N, K]` | K |

Weight is stored in **ColumnMajor** for CUTLASS: the `[N, K]` contiguous tensor's transpose `[K, N]` is column-major.

---

## Activation Quantization Kernel

`quantize_act_mxfp8_kernel` in `sf_layout.cu` converts BF16 activations to MXFP8 online:

```
Input:  x_bf16  [M, K]
Output: x_fp8   [M, K]  (uint8, FP8 E4M3 codes)
        x_sf    [M, K/32] (uint8, UE8M0 shared exponents)
```

Algorithm per 32-element block:
1. Load 32 BF16 values, compute `absmax`
2. Compute UE8M0 shared exponent: `biased_exp = ceil(log2(absmax / 448)) + 127`
3. Scale: `inv_scale = 2^(-(biased_exp - 127))`
4. Round each value to nearest FP8 E4M3 representable value
5. Store packed FP8 codes and UE8M0 exponent

Grid: one block per row (M), threads = K/32 (one thread per 32-element group).

---

## Benchmark Results

### Setup

- GPU: NVIDIA RTX 5090 (Blackwell, SM120)
- Layer dimensions: K=9728, N=2560 (Qwen3-4B MLP down_proj)
- Latency: median of 500 iterations after 50 warmup

### Single GEMM latency (prefill)

| M | BF16 (μs) | MXFP4 (μs) | MXFP6 (μs) | MXFP8 (μs) | FP4 speedup | FP6 speedup | FP8 speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 23.2 | 63.6 | 71.2 | 61.5 | 0.37x | 0.33x | 0.38x |
| 64 | 40.7 | 63.6 | 61.5 | 64.8 | 0.64x | 0.66x | 0.63x |
| 128 | 51.7 | 63.6 | 61.5 | 61.5 | 0.81x | 0.84x | 0.84x |
| 256 | 63.6 | 73.3 | 61.5 | 61.6 | 0.87x | 1.03x | 1.03x |
| **512** | **127.3** | **63.6** | **61.6** | **63.6** | **2.00x** | **2.07x** | **2.00x** |
| **1024** | **231.4** | **65.7** | **71.5** | **83.7** | **3.52x** | **3.24x** | **2.77x** |
| **2048** | **458.2** | **132.6** | **141.0** | **168.1** | **3.46x** | **3.25x** | **2.73x** |

### Analysis

**Compute-bound regime (M ≥ 512)**: Block-scaled MX kernels significantly outperform BF16:
- **MXFP4**: 3.5x speedup at M=1024 — lowest bit-width, least weight bandwidth
- **MXFP6**: 3.2x speedup — intermediate
- **MXFP8**: 2.7x speedup — still nearly 2x better than BF16 due to higher Tensor Core throughput

**Bandwidth-bound regime (M < 128)**: CUTLASS TMA warp-specialized kernel has ~60μs fixed overhead. cuBLAS BF16 GEMM is faster for small M. This is expected — the TMA pipeline setup cost is amortized only at larger M.

**Crossover point**: M ≈ 256 for MXFP6/MXFP8, M ≈ 512 for MXFP4.

### Weight memory footprint per layer (K=9728, N=2560)

| Format | Weight size | Scale size | Total | vs BF16 |
|---|---|---|---|---|
| BF16 | 49.8 MB | — | 49.8 MB | 1.0x |
| MXFP8 | 24.9 MB | 0.8 MB | 25.7 MB | 0.52x |
| MXFP6 | 18.7 MB | 0.8 MB | 19.5 MB | 0.39x |
| MXFP4 | 12.5 MB | 0.8 MB | 13.3 MB | 0.27x |

### Mixed-precision layer (40% MXFP4 / 30% MXFP6 / 30% MXFP8, ~5.4 avg bpw)

Weight memory = 0.4×13.3 + 0.3×19.5 + 0.3×25.7 = **18.9 MB** (0.38x of BF16).

Three-launch latency (M=512): ~190μs — dominated by launch overhead, not compute. For prefill with large M, this is still faster than a single BF16 GEMM.

---

## Build Instructions

### Prerequisites

- CUDA Toolkit ≥ 12.8
- CUTLASS 3.x with SM120 block-scaled support
- PyTorch ≥ 2.1
- pybind11

### Build

```bash
cd rdquant/csrc
mkdir build && cd build

# CUTLASS_ROOT must point to CUTLASS 3.x with SM120 support
cmake .. -DCUTLASS_ROOT=/path/to/cutlass
make -j$(nproc)
```

The build produces `rdquant_cuda.so` which can be imported directly:

```python
import sys
sys.path.insert(0, 'rdquant/csrc/build')
import rdquant_cuda

# Quantize activation
x_fp8, x_sf = rdquant_cuda.quantize_act_mxfp8(x_bf16)

# Reorder scale factors for CUTLASS
x_sf_r = rdquant_cuda.reorder_sf(x_sf, M, K)
w_sf_r = rdquant_cuda.reorder_sf(w_sf, N, K)

# Run GEMM
y = rdquant_cuda.mx_gemm_w8a8(x_fp8, x_sf_r, w_fp8, w_sf_r, M, N, K)
```

### Build note: sm_120a

The CMake uses `-gencode arch=compute_120a,code=sm_120a` directly instead of `CMAKE_CUDA_ARCHITECTURES=120a`, because CMake strips the `a` suffix. The `a` suffix is critical — it defines `__CUDA_ARCH_FEAT_SM120_ALL`, which CUTLASS checks at kernel launch time.

---

## File Structure

```
rdquant/csrc/
├── CMakeLists.txt          # Build config (SM120a, CUTLASS, PyTorch, pybind11)
├── rdquant_kernels.h       # Full CUTLASS includes (only in .cu files)
├── rdquant_ops.h           # void* function declarations (for bindings.cpp)
├── w4a8.cu                 # MXFP8 × MXFP4 GEMM kernel
├── w6a8.cu                 # MXFP8 × MXFP6 GEMM kernel
├── w8a8.cu                 # MXFP8 × MXFP8 GEMM kernel
├── sf_layout.cu            # Scale factor reorder + activation quantization
├── bindings.cpp            # PyBind11 Python interface
└── build/                  # Build output (gitignored)
    └── rdquant_cuda.so     # Importable Python module
```

### Design: separating CUDA and C++ compilation

The `.cu` files include `rdquant_kernels.h` (which pulls in all CUTLASS headers). The `bindings.cpp` file includes only `rdquant_ops.h` which declares `void*` function signatures — this avoids compiling CUTLASS headers in C++ mode (only CUDA compilation can handle CUTLASS).

---

## Comparison with MicroMix

| Aspect | RDQuant | MicroMix |
|---|---|---|
| Split dimension | Output channels (N) | Input channels (K) |
| # GEMM launches | 3 (one per format group) | 3 (one per precision) |
| Output merge | Concat + inv_perm | Accumulate (beta=1.0) |
| Epilogue beta | 0.0 (no accumulation) | 1.0 (accumulate partial sums) |
| Activation quant | Uniform MXFP8, one pass | Mixed FP4/FP6/FP8 with reorder |
| Output bandwidth | 1× (disjoint channels) | 3× (full [M,N] per GEMM) |
| Weight quant | Data-free (R-D optimal on weight MSE) | Requires activation statistics |

RDQuant's N-dim split avoids writing the full [M, N] output 3 times. MicroMix's K-dim split avoids the inv_permutation but requires activation reordering and 3× output bandwidth.

---

## Next Steps

1. **Integrate with QuantizedLayer**: Replace the fake-quant forward with real CUTLASS kernels when GPU is available
2. **Weight packing**: Convert Python `MXQuantizedTensor` to the packed byte formats CUTLASS expects (FP4: 2/byte, FP6: 4/3 bytes)
3. **Fused act quant + SF reorder**: Combine activation quantization and scale factor reorder into one kernel (saves one global memory pass)
4. **GEMV kernel for decode (M=1)**: The TMA warp-specialized kernel has high launch overhead; a simpler GEMV kernel would be faster for decode
5. **Benchmark on real model**: End-to-end Qwen3-4B inference latency with mixed-precision weights
