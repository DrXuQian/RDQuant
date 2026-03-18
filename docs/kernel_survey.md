# Kernel Survey: Available GEMM Backends on RTX 5090 (SM120)

## Test Conditions
- GPU: RTX 5090 (Blackwell, SM120, 32GB)
- CUDA: 12.8
- PyTorch: 2.8.0
- Layer dimensions: K=9728, N=2560 (Qwen3-4B MLP down_proj)

---

## Available Kernels

### 1. cuBLASLt NVFP4 (W4A4)
- **Format**: FP4 E2M1 weight + FP4 E2M1 activation
- **Scale**: per-16-element FP8(E4M3) block scale (`VEC16_UE4M3`)
- **Status**: ✅ Working
- **Note**: Both A and B must be FP4. Cannot do W4A16 (FP16 activation).

### 2. cuBLASLt MXFP8 (W8A8)
- **Format**: FP8 E4M3 weight + FP8 E4M3 activation
- **Scale**: per-32-element UE8M0 block scale (`VEC32_UE8M0`)
- **Status**: ✅ Working
- **Note**: Both A and B must be FP8.

### 3. torch._scaled_mm (W8A8, per-channel)
- **Format**: FP8 E4M3 weight + FP8 E4M3 activation
- **Scale**: per-row (activation) + per-column (weight) FP32 scale
- **Status**: ✅ Working
- **Note**: Per-channel scale, NOT per-block. Different from MX.

### 4. CUTLASS block-scaled (W4A8, W6A8, W8A8)
- **Format**: MXFP8 activation × MXFP4/6/8 weight
- **Scale**: per-32-element UE8M0 (OCP MX standard)
- **Status**: ✅ Working (our custom build)
- **Note**: Only backend that supports mixed A×B precision with block scaling.

### 5. cuBLAS BF16 (W16A16)
- **Format**: BF16 weight + BF16 activation
- **Status**: ✅ PyTorch `F.linear`

### 6. TensorRT-LLM nvfp4_gemm (W4A16)
- **Format**: NVFP4 weight + FP16 activation (weight-only)
- **Status**: ❌ Requires tensorrt_llm (not installed)
- **Note**: This is the ideal W4A16 kernel. Dequant FP4→FP16 fused in-register.

### 7. Marlin FP8 (W8A16)
- **Format**: FP8 weight + FP16 activation (weight-only)
- **Scale**: Channelwise only (b_scales.size(0) == 1)
- **Status**: Available in optimum-quanto but import issues.

---

## cuBLASLt Type Support Matrix

Tested exhaustively on RTX 5090 with CUDA 12.8:

| A (activation) | B (weight) | Scale Mode | Supported |
|---|---|---|---|
| FP4 | FP4 | VEC16_UE4M3 | ✅ |
| FP8 | FP8 | VEC32_UE8M0 | ✅ |
| FP8 | FP4 | any | ❌ |
| FP4 | FP8 | any | ❌ |
| FP8 | FP6 | any | ❌ |
| FP6 | FP6 | any | ❌ |
| FP4 | FP4 | VEC32_UE8M0 | ❌ |
| FP8 | FP8 | VEC16_UE4M3 | ❌ |
| Mixed scale modes | | | ❌ |

**Only two combinations work**: NVFP4×NVFP4 with VEC16, and MXFP8×MXFP8 with VEC32.

---

## Benchmark Results (Prefill, μs)

### cuBLASLt: NVFP4 vs MXFP8

```
    M     NVFP4     MXFP8   NV4/MX8
           (μs)      (μs)     ratio
--------------------------------------
    1     160.9     165.5     1.03x
    8     111.9     164.6     1.47x
   16     113.9     161.4     1.42x
   64     112.1     161.1     1.44x
  128     111.9     161.2     1.44x
  256     112.9     166.0     1.47x
  512     167.3     224.6     1.34x
 1024     266.4     388.4     1.46x
 2048     435.2     723.8     1.66x
```

**NVFP4 consistently 1.3-1.7x faster than MXFP8** — less weight data to read.

### CUTLASS block-scaled (our custom kernels)

```
    M     MXFP4     MXFP6     MXFP8
           (μs)      (μs)      (μs)
--------------------------------------
   32     330.3     285.5     285.5
  128     332.4     285.4     285.5
  512     330.7     286.4     333.5
 1024     336.4     340.0     396.8
 2048     670.1     673.1     836.3
```

CUTLASS has ~285μs fixed overhead (TMA pipeline setup). Faster than cuBLASLt at large M for MXFP4/6.

### torch._scaled_mm (FP8 per-channel)

```
    M   _scaled_mm     BF16   ratio
           (μs)        (μs)
--------------------------------------
    1     330.0       214      0.65x
  128     330.5       277      0.84x
  512     332.2       668      2.01x
 2048    1265.6      2438      1.93x
```

~330μs fixed overhead. Only beneficial at M≥512.

---

## Recommended Kernel Strategy for RDQuant

### Option A: NVFP4 (W4A4) + FP8 (W8A8) + FP16 (cuBLASLt only)

Uses only cuBLASLt — no custom CUDA code, but requires activation quantization.

| Weight format | Kernel | Activation | Performance |
|---|---|---|---|
| NVFP4 | cuBLASLt W4A4 VEC16 | FP4 (online quant) | 112-435μs |
| FP8 | cuBLASLt W8A8 VEC32 | FP8 (online quant) | 161-724μs |
| FP16 | cuBLAS F.linear | BF16 | 109-2438μs |

**Pros**: No custom CUDA. cuBLASLt auto-tunes kernel selection.
**Cons**: Requires activation quantization to FP4/FP8 (precision loss).

### Option B: NVFP4 (W4A16) + FP8 (W8A16) + FP16 (TRT-LLM + Marlin)

Weight-only quantization with FP16 activation. Requires TensorRT-LLM.

| Weight format | Kernel | Activation | Source |
|---|---|---|---|
| NVFP4 | TRT-LLM nvfp4_gemm | FP16 | TensorRT-LLM |
| FP8 | fp8_marlin | FP16 | optimum-quanto / vLLM |
| FP16 | cuBLAS F.linear | BF16 | PyTorch |

**Pros**: No activation quantization. Best accuracy.
**Cons**: Requires TensorRT-LLM install. FP8 Marlin is channelwise only (no per-block).

### Option C: Mixed backend (current best)

Use the fastest available kernel for each format:

| Weight format | Kernel | Notes |
|---|---|---|
| NVFP4 group | cuBLASLt W4A4 (best) or CUTLASS W4A8 | Need FP4 or FP8 activation |
| FP8 group | cuBLASLt MXFP8 (small M) or CUTLASS (large M) | Need FP8 activation |
| FP16 group | cuBLAS F.linear | No quantization |

---

## FP8 Scale Format: Per-Channel vs Per-Block

| Scale type | Format | Group size | Kernel support |
|---|---|---|---|
| Per-channel | 1 scale per output channel | N | torch._scaled_mm, fp8_marlin |
| Per-block MX | UE8M0 per 32 elements along K | 32 | cuBLASLt MXFP8, CUTLASS |
| Per-block NVFP4 | FP8 E4M3 per 16 elements | 16 | cuBLASLt NVFP4 |

For RDQuant with per-channel FP8 scales:
- `torch._scaled_mm`: works, but 330μs overhead
- `fp8_marlin`: works (channelwise), FP16 activation (no quant needed)

For per-block (group_size=32 or 16):
- cuBLASLt: only W8A8 (MXFP8) or W4A4 (NVFP4)
- CUTLASS: all combinations (our custom build)

---

## Key Takeaway

**No single backend covers all needs.** The practical approach is:

1. **Install TensorRT-LLM** for NVFP4 W4A16 kernel (ideal for weight-only FP4)
2. **Use cuBLASLt** for NVFP4 W4A4 (when TRT-LLM unavailable) and MXFP8 W8A8
3. **Use CUTLASS** for mixed-precision (MXFP8×MXFP4/6) and as fallback
4. **Use cuBLAS** for FP16 passthrough
