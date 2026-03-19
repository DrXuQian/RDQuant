# RDQuant: Rate-Distortion Optimal Mixed-Precision Quantization for LLMs

## Paper Outline (arxiv format)

### Abstract
- Mixed-precision quantization assigns different numeric formats to different output channels
- Formulate as Rate-Distortion optimization with Lagrangian binary search
- Data-free + calibrated modes
- Qwen3-4B results: calibrated 12.24 PPL @ 5.29 bpw (beats BF16 12.90)
- 2.59x decode speedup with Marlin kernels on RTX 5090

---

### 1. Introduction
- LLM inference bottleneck: memory bandwidth (decode) and compute (prefill)
- Existing: uniform quantization (GPTQ, AWQ, NVFP4) — same precision for all channels
- Observation: output channels have vastly different sensitivity to quantization
- Our contribution: R-D optimal per-channel format assignment
  - NVFP4 (4-bit) for easy channels, FP8 (8-bit) for sensitive channels, FP16 for outliers
  - Data-free: purely based on weight MSE
  - Calibrated: perturb-based layer importance with 2 samples
- Key results table

### 2. Background and Related Work
- Uniform quantization: GPTQ, AWQ, SqueezeLLM, QuIP
- Mixed-precision: HAWQ, MicroMix, FLUTE
  - HAWQ: layer-level mixed precision (not channel-level)
  - MicroMix: input-channel split (K-dim), requires activation reorder + 3x output write
  - FLUTE: lookup-table based, non-uniform quantization
- Rate-Distortion theory in quantization
- NVIDIA formats: NVFP4 (per-16 FP8 scale), MX formats (per-32 UE8M0)

### 3. Method

#### 3.1 Problem Formulation
- Weight matrix W ∈ R^{N×K} for each linear layer
- Output channels j = 1..N, each row W_j ∈ R^K
- Format set F = {NVFP4, FP8, FP16} with costs c_f = {4, 8, 16} bits/element
- Distortion D_j(f) = MSE of quantizing W_j to format f
- **Theorem/Proposition**: Output channel sensitivity is format-dependent and varies significantly across channels
  - Channels with large outlier weights have disproportionately high NVFP4 MSE due to block-scale sharing
  - Prove: for a channel with outlier ratio r = max(|w|)/std(w), the NVFP4 MSE scales as O(r^2 / 2^{2b}) where b depends on block-scale precision
  - This motivates per-channel format assignment

#### 3.2 R-D Optimal Allocation
- Lagrangian formulation:
  min Σ_j D_j(f_j) s.t. Σ_j c_{f_j} · K ≤ B
- Decomposition via Lagrange multiplier λ:
  f_j* = argmin_f [D_j(f) + λ · c_f · K]
- Binary search on λ (64 iterations)
- **Global budget**: single λ* across all layers (cross-layer allocation)
- **128-channel alignment**: pad format groups for hardware efficiency

#### 3.3 Calibrated Layer Importance
- Data-free mode: w_layer = 1 for all layers (equal importance)
- Calibrated mode: quantize each layer to NVFP4 independently, measure loss delta
  - importance_layer = L(model with layer_j quantized) - L(model baseline)
- Weighted R-D: min Σ_j w_layer · D_j(f_j) + λ · c_{f_j} · K
- **Key finding**: act_norm (||X||²) is a poor proxy — dominated by residual stream growth. Perturb is more accurate.

#### 3.4 Output Channel vs Input Channel Splitting

**Proposition**: Output-channel and input-channel mixed-precision are orthogonal approaches.

Output-channel split (RDQuant):
- Split along N (output) dimension
- Each group: different weight precision, same activation
- Result: concat disjoint output channels + permute
- No activation reorder needed
- Activation uniformly FP16 — no precision loss on activation side

Input-channel split (MicroMix):
- Split along K (input) dimension
- Each group: different weight AND activation precision
- Result: sum partial products (accumulate)
- Requires activation reorder + quantization to different precisions
- 3x output write bandwidth

Comparison table:
| | RDQuant | MicroMix |
|---|---|---|
| Split dim | N (output) | K (input) |
| Activation | FP16 uniform | Mixed FP4/FP6/FP8 |
| Output merge | Concat + permute | Sum (accumulate) |
| Activation reorder | No | Yes |
| Output bandwidth | 1x | 3x |
| Can combine | Yes (orthogonal) | Yes (orthogonal) |

**Combined approach** (future work): split both N and K dimensions for maximum compression.

### 4. Supported Formats and Kernel Integration

#### 4.1 Format Specifications
- NVFP4: E2M1, per-16 FP8(E4M3) block scale + FP32 global scale
- FP8: E4M3, per-channel FP32 scale
- FP16: passthrough
- Why this hierarchy: matches vLLM kernel interfaces (marlin_gemm, cutlass_scaled_mm)

#### 4.2 Inference Architecture
- Per-layer: 2 Marlin GEMM (NVFP4 + FP8) + cat + index_select(inv_perm)
- CUDA Graph eliminates Python dispatch overhead
- Decode: bandwidth-bound → 2.59x speedup from weight compression
- Prefill: compute-bound → limited by 2x launch count (1.25x with CUDA Graph)

### 5. Experiments

#### 5.1 Setup
- Models: Qwen3-4B (original BF16), [Llama-3.2-1B/3B — TODO]
- Hardware: RTX 5090 (Blackwell, SM120, 32GB)
- Evaluation: WikiText-2 PPL, [MMLU, GSM8K, MBPP — TODO]
- Baselines: BF16, Uniform NVFP4, Uniform FP8

#### 5.2 Accuracy Results (Qwen3-4B)

| Method | Avg Bits | WikiText-2 PPL | vs BF16 |
|---|---:|---:|---:|
| RDQuant calibrated | 5.29 | 12.24 | -0.66 |
| BF16 baseline | 16.00 | 12.90 | — |
| Uniform FP8 | 8.13 | 12.93 | +0.03 |
| Uniform NVFP4 | 4.00 | 13.22 | +0.32 |
| RDQuant data-free | 5.52 | 13.43 | +0.53 |

Key observations:
- Calibrated beats BF16 (quantization as regularization)
- NVFP4 at 4 bpw is surprisingly good (+0.32 only)
- R-D allocation gives meaningful improvement over uniform

#### 5.3 Layer Importance Analysis
- Perturb vs act_norm comparison
- Per-layer sensitivity visualization
- Allocation heatmap across layers

#### 5.4 Inference Performance

| Scenario | BF16 | RDQuant (Eager) | RDQuant (CUDA Graph) |
|---|---|---|---|
| Decode (M=1) | 32.6 ms | 50.8 ms | 12.6 ms (2.59x) |
| Prefill (SEQ=128) | 23.6 ms | 45.2 ms | 18.9 ms (1.25x) |

Kernel-level analysis:
- Single Marlin kernel: 21.5 μs (3.4x vs BF16 cuBLAS)
- Overhead sources: 2x launch, cat, index_select, Python dispatch
- CUDA Graph eliminates Python dispatch → decode wins

#### 5.5 Kernel Backend Comparison
- vLLM marlin_gemm vs cuBLASLt vs CUTLASS
- Table of all tested backends

#### 5.6 Ablation Studies
- Budget sweep (4.0 → 8.0 bpw)
- Data-free vs calibrated allocation
- Calibration metrics (perturb vs fisher vs act_norm)
- Group alignment (128 vs none)

### 6. Discussion

#### 6.1 When RDQuant Helps
- Models with high channel sensitivity variance
- Decode-dominated workloads (long generation)
- Memory-constrained deployment

#### 6.2 Limitations
- Prefill overhead from 2x GEMM launch (mitigated by CUDA Graph)
- Requires channel permutation (index_select overhead)
- Calibrated mode needs 2 forward passes per layer

#### 6.3 Future Work
- Fused grouped GEMM kernel (single launch for all format groups)
- Combined N+K dimension splitting (RDQuant + MicroMix)
- Extension to MoE models
- Llama-3 / Qwen2.5 / DeepSeek-V3 evaluation

### 7. Conclusion
- R-D optimal mixed-precision quantization is effective and practical
- Calibrated RDQuant at ~5 bpw matches or beats BF16 precision
- 2.59x decode speedup with production vLLM kernels
- Open-source implementation with one-click quantize + export

---

## TODO Items for Paper

### Must-have
- [ ] Theoretical analysis of output-channel sensitivity (Section 3.1)
  - Prove MSE dependency on block-scale and outlier ratio
  - Show why per-channel format assignment is strictly better than uniform
- [ ] Formal proof of Lagrangian decomposition optimality
- [ ] Llama-3.2-1B/3B results (at minimum WikiText-2 PPL)
- [ ] lm-eval results: MMLU, GSM8K on Qwen3-4B
- [ ] Orthogonality proof/experiment: RDQuant + MicroMix combined

### Nice-to-have
- [ ] More models: Qwen2.5-7B, Llama-3.1-8B
- [ ] Comparison with GPTQ, AWQ, SqueezeLLM
- [ ] Prefill optimization (CUTLASS grouped GEMM or dequant-at-prefill strategy)
- [ ] Throughput benchmark (tokens/s end-to-end)
