# RDQuant Operator Implementation Plan

## Goal

Replace the current fake-quantization inference path (dequant weight → FP16 GEMM) with real low-bit GEMM kernels on NVIDIA Blackwell (SM120), achieving actual inference speedup.

Current path (no speedup):
```
x_bf16 → dequant(W_mxfp4) → W_bf16 → F.linear(x_bf16, W_bf16)  # BF16×BF16, slow
```

Target path (real speedup):
```
x_bf16 → quant_mxfp8(x) → x_mxfp8 → mxfp8_x_mxfpX_gemm(x_mxfp8, W_packed)  # Tensor Core
```

---

## Hardware Context

**RTX 5090 (Blackwell, SM120)**:
- Native block-scaled MMA: `tcgen05.mma.blockscaled`
- Supported precision pairs: MXFP8×MXFP4, MXFP8×MXFP6, MXFP8×MXFP8
- Per-32-element UE8M0 shared exponents consumed natively by hardware
- No dequantization kernel needed — hardware decodes sub-byte formats in the MMA unit

**vLLM already has the building blocks**:
- `torch.ops._C.cutlass_scaled_mm(...)` — FP8 GEMM via CUTLASS
- `torch.ops._C.nvfp4_gemm(...)` — NVFP4 GEMM (different from MXFP4 but similar path)
- CUTLASS 3.x with SM120 support for block-scaled MMA

---

## Operator Inventory

### Hot path (CUDA, latency-critical)

| # | Operator | Input | Output | Notes |
|---|---|---|---|---|
| 1 | `mxfp8_quantize_act` | x_bf16 [M, K] | x_mxfp8 [M, K], scales [M, K/32] | Online, per-token |
| 2 | `mxfp8_x_mxfp4_gemm` | x_mxfp8 [M, K], W_mxfp4 [N, K/2], scales | y_bf16 [M, N] | Block-scaled MMA |
| 3 | `mxfp8_x_mxfp6_gemm` | x_mxfp8 [M, K], W_mxfp6 [N, K*6/8], scales | y_bf16 [M, N] | Block-scaled MMA |
| 4 | `mxfp8_x_mxfp8_gemm` | x_mxfp8 [M, K], W_mxfp8 [N, K], scales | y_bf16 [M, N] | Block-scaled MMA |
| 5 | `channel_permute` | y_cat [M, N], inv_perm [N] | y [M, N] | `index_select`, trivial |

### Offline path (Python/CPU, already implemented)

| # | Operator | Status |
|---|---|---|
| 6 | `mxfp4_quantize_weight` | ✅ `rdquant/core/formats.py` |
| 7 | `mxfp6_quantize_weight` | ✅ `rdquant/core/formats.py` |
| 8 | `mxfp8_quantize_weight` | ✅ `rdquant/core/formats.py` |
| 9 | `mxfpX_dequantize_weight` | ✅ `rdquant/core/formats.py` |

---

## Implementation Phases

### Phase 1: Use vLLM's existing CUTLASS kernels (zero custom CUDA)

**Approach**: Wrap vLLM's `cutlass_scaled_mm` and `nvfp4_gemm` ops.

```python
# rdquant/kernels/vllm_backend.py

def mxfp8_x_mxfp8_gemm(x_mxfp8, w_mxfp8_t, x_scale, w_scale):
    """Wrap vLLM's CUTLASS FP8 GEMM."""
    return torch.ops._C.cutlass_scaled_mm(
        x_mxfp8, w_mxfp8_t,
        scale_a=x_scale, scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )
```

**Key questions to answer**:
1. Does vLLM's `cutlass_scaled_mm` support per-32-element block scales (MX format), or only per-tensor/per-channel scales?
2. Does `nvfp4_gemm` accept MX-style UE8M0 exponents, or only NVIDIA's FP8 block scales?
3. What are the weight packing formats expected by each kernel?

**Investigation tasks**:
- [ ] Read vLLM source: `vllm/model_executor/layers/quantization/` and `csrc/quantization/`
- [ ] Find the CUTLASS kernel template instantiations for SM120
- [ ] Check if `cutlass::float_e2m1_t` / `float_e3m2_t` are used anywhere
- [ ] Test: can we call `cutlass_scaled_mm` with block-scale tensors?
- [ ] Check MicroMix's CUTLASS wrapper for reference (they solved this already)

**Deliverables**:
- `rdquant/kernels/__init__.py`
- `rdquant/kernels/vllm_backend.py` — wrapper around vLLM ops
- `rdquant/kernels/packing.py` — convert our `MXQuantizedTensor` to the packed format expected by vLLM kernels
- `tests/test_kernels.py` — correctness vs fake-quant reference
- `benchmarks/bench_kernels.py` — latency comparison

### Phase 2: Direct CUTLASS integration (if vLLM kernels don't fit)

If vLLM's kernels don't support MX block scaling, use CUTLASS 3.x directly.

**Approach**: Write a thin C++/CUDA wrapper around CUTLASS's block-scaled GEMM template, exposed via PyTorch custom op.

```
rdquant/csrc/
├── mx_gemm.cu          # CUTLASS block-scaled GEMM instantiation
├── mx_gemm.h           # C++ interface
├── act_quant_kernel.cu  # MXFP8 activation quantization kernel
└── bindings.cpp         # PyBind11 / torch.library registration
```

**CUTLASS template to use** (from MicroMix reference):
```cpp
using MMA_OP = cute::rr_blockscaled_op_selector_sm120<
    cutlass::float_e4m3_t,   // A (activation, MXFP8)
    cutlass::float_e2m1_t,   // B (weight, MXFP4)
    float,                    // Accumulator
    cutlass::float_ue8m0_t,  // Scale factor type
    32                        // Block size
>();
```

**Three kernel variants needed**:
1. `mx_gemm<float_e4m3_t, float_e2m1_t>` — MXFP8 × MXFP4
2. `mx_gemm<float_e4m3_t, float_e3m2_t>` — MXFP8 × MXFP6
3. `mx_gemm<float_e4m3_t, float_e4m3_t>` — MXFP8 × MXFP8

All share the same kernel template, only the element type differs.

**Build system**:
- Use `torch.utils.cpp_extension` for JIT compilation
- Or use CMake + setuptools for ahead-of-time compilation
- Require CUTLASS 3.x headers (submodule or pip install)

**Deliverables**:
- `rdquant/csrc/` directory with CUDA/C++ source
- `rdquant/kernels/cutlass_backend.py` — Python wrapper
- Updated `setup.py` / `pyproject.toml` with CUDA extension
- Correctness tests vs fake-quant
- Latency benchmarks

### Phase 3: Activation quantization kernel

The online MXFP8 activation quantization is currently pure Python. For production inference, it needs a CUDA kernel.

```
Input:  x_bf16 [M, K]
Output: x_mxfp8 [M, K] (packed uint8), scales [M, ceil(K/32)] (uint8 UE8M0)

Algorithm per 32-element block:
  1. Compute absmax of block
  2. shared_exp = floor(log2(absmax))
  3. scale = 2^shared_exp
  4. x_quant = round_to_fp8(x / scale)
  5. Store packed codes + scale
```

This is a bandwidth-bound kernel (~3 μs for a typical layer). Options:
- Use vLLM's existing FP8 quantization kernel (if it supports block scaling)
- Write a simple CUDA kernel (one block per 32-element group)
- Use Triton (easier to write, decent performance)

### Phase 4: Fused grouped GEMM (optional)

If 3 separate GEMM launches have significant overhead:

```
Current:  launch_gemm(MXFP4) → launch_gemm(MXFP6) → launch_gemm(MXFP8) → cat → perm
Fused:    launch_grouped_gemm(MXFP4|MXFP6|MXFP8) → perm
```

This is the most complex part and probably not needed initially. From our benchmarks, the multi-launch overhead is ~40 μs, which is acceptable for decode (total ~70 μs per layer). For prefill (batch > 1), GEMMs dominate and overhead is <31%.

**Only pursue this if Phase 1-3 benchmarks show launch overhead is the bottleneck.**

---

## Investigation Order

```
Week 1: Research
  ├── Read vLLM quantization source code
  ├── Read CUTLASS 3.x block-scaled GEMM examples
  ├── Read MicroMix csrc/ for SM120 kernel patterns
  ├── Test if vLLM kernels accept MX block scales
  └── Document: which existing kernels can we reuse?

Week 2: Phase 1 — vLLM wrapper
  ├── Implement packing conversion (MXQuantizedTensor → vLLM format)
  ├── Wrap vLLM GEMM ops
  ├── Correctness test vs fake-quant
  └── Benchmark: mixed GEMM vs uniform FP16

Week 3: Phase 2 — CUTLASS direct (if needed)
  ├── Set up CUTLASS 3.x build
  ├── Instantiate block-scaled GEMM template
  ├── PyBind11 bindings
  └── Correctness + benchmark

Week 4: Phase 3 — Act quant kernel + integration
  ├── CUDA/Triton activation quantization kernel
  ├── End-to-end integration in QuantizedLayer
  ├── Full model inference benchmark
  └── Compare: RDQuant vs uniform MXFP4 vs uniform MXFP8 latency
```

---

## Key Files to Study

### vLLM source
- `vllm/model_executor/layers/quantization/fp8.py` — FP8 quantization config
- `vllm/model_executor/layers/quantization/compressed_tensors/` — MX format support?
- `csrc/quantization/cutlass_w8a8/` — CUTLASS FP8 GEMM wrapper
- `csrc/quantization/fp4/` — NVFP4 GEMM wrapper

### CUTLASS
- `include/cutlass/float_e2m1_t.h` — MXFP4 element type
- `include/cutlass/float_e3m2_t.h` — MXFP6 element type
- `examples/cute/blackwell/` — SM120 block-scaled examples
- `test/unit/cute/ampere/` → `test/unit/cute/blackwell/` — block-scaled MMA tests

### MicroMix
- `mgemm/src/gemm.cu` — 3-way GEMM dispatch (w4a4, w6a6, w8a8)
- `mgemm/include/sm120_multistage_tma.h` — kernel template
- `mgemm/src/reorder.cu` — fused reorder + quantize kernel

---

## Success Criteria

| Metric | Target |
|---|---|
| Correctness | Max abs error < 1e-3 vs fake-quant reference |
| Decode latency (seq=1) | ≤ 1.5× uniform MXFP8 per layer |
| Prefill latency (seq=512) | ≤ 1.2× uniform MXFP8 |
| End-to-end tokens/s | > uniform MXFP4 (smaller model = less memory bandwidth) |
| PPL | Identical to fake-quant (bit-exact weight reconstruction) |
