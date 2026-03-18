# RDQuant

**Rate-Distortion Optimal Mixed-Precision Quantization for LLMs**

RDQuant assigns different numeric formats (NVFP4 / FP8 / FP16) to different output channels of each linear layer, optimally trading off quantization error against bit cost. The format hierarchy matches vLLM kernel interfaces: NVFP4 uses marlin_gemm with float4_e2m1f, FP8 uses cutlass_scaled_mm with per-channel scale, and FP16 is passthrough.

Two allocation modes:
- **Data-free**: allocation based purely on weight MSE (no calibration data needed)
- **Calibrated**: uses a small calibration set to measure per-layer sensitivity to quantization, producing significantly better bit allocation

---

## Results

WikiText-2 perplexity on **Qwen3-4B** (stride 2048, 32k tokens):

| Method | Avg Bits | PPL | vs BF16 |
|---|---:|---:|---:|
| **RDQuant calibrated** | **5.19** | **12.57** | **-0.59** |
| BF16 baseline | 16.00 | 13.16 | — |
| RDQuant data-free | 5.32 | 13.18 | +0.02 |
| Uniform NVFP4 | 4.00 | 13.27 | +0.11 |
| Uniform FP8 | 8.15 | 13.28 | +0.12 |

Key findings:
- **Calibrated RDQuant at 5.19 bpw beats BF16** (12.57 vs 13.16 PPL) — quantization acts as regularization
- **Data-free RDQuant at 5.32 bpw is near-lossless** (+0.02 PPL vs BF16) at 3x compression
- **Uniform NVFP4 at 4 bpw is near-lossless** (+0.11 PPL) — NVFP4's per-16 FP8 block scale is highly accurate
- Format split at 5.3 bpw: 57% NVFP4 + 43% FP8 (data-free) or 67% NVFP4 + 32% FP8 + 1% FP16 (calibrated)

---

## Algorithm

### Core: R-D Optimal Format Allocation

For every output channel *j* of every linear layer, compute the MSE from quantizing that channel to each candidate format. This produces a Rate-Distortion table:

```
channel j -> [(rate=4, D_nvfp4), (rate=8, D_fp8), (rate=16, D_fp16)]
```

Given a bit budget *B*, solve the constrained optimization:

```
min  Sigma_j  w_layer * D_j(f_j)    subject to  Sigma_j C_j(f_j) <= B
```

where `w_layer` is the per-layer importance weight. This decomposes via a Lagrange multiplier lambda -- each channel independently picks the format minimising `w * D_j(f) + lambda * C_j(f)`. Binary search on lambda (64 iterations, precision ~10^-19) finds the optimal lambda*.

By default, a **single global lambda*** is searched across all layers simultaneously, allowing sensitive layers to receive more bits while easy layers are compressed further.

After allocation, format groups are aligned to multiples of 128 channels for hardware efficiency.

### Inference Path

```
x_fp16 -> [as-is, no activation quantization]

y_nvfp4 = marlin_gemm(x_fp16, W_nvfp4, block_scales, global_scale)  # float4_e2m1f
y_fp8   = cutlass_scaled_mm(x_fp16, W_fp8, channel_scales)          # per-channel FP8
y_fp16  = F.linear(x_fp16, W_fp16)                                  # passthrough

y = cat(y_nvfp4, y_fp8, y_fp16)[:, inv_perm]     # restore channel order
```

No activation quantization. No activation reorder. One GEMM per format group.

---

## Supported Formats

| Format | Bits | Element Type | Block Scale | Scale Type | vLLM Kernel |
|---|---:|---|---|---|---|
| NVFP4 | 4 | E2M1 | per-16-element | FP8 E4M3 + FP32 global | marlin_gemm (float4_e2m1f) |
| FP8 | 8 | E4M3 | per-channel | FP32 | cutlass_scaled_mm |
| FP16 | 16 | float16 | none | none | F.linear (passthrough) |

---

## Project Structure

```
rdquant/
  rdquant/
    core/
      formats.py          # NVFP4/FP8/FP16 quantize/dequantize
      sensitivity.py      # Data-free channel sensitivity + R-D profiling
      allocator.py        # Lagrangian R-D allocation + 128-channel alignment
      calibrate.py        # Calibrated layer importance (perturb/fisher/act_norm)
    quantize.py           # End-to-end model quantization
    ops.py                # Mixed-precision grouped-GEMM linear operator
    eval.py               # Perplexity & zero-shot evaluation
    integrations/
      hf_export.py        # HuggingFace checkpoint save/load
      vllm_linear.py      # vLLM inference stub
  examples/
    quickstart.py         # Minimal quantize + eval
    sweep_budget.py       # Sweep bit budgets, R-D curve
  tests/                  # Comprehensive tests
    test_formats.py       # Format correctness
    test_sensitivity.py   # Sensitivity metrics
    test_allocator.py     # Allocation + alignment
    test_calibrate.py     # Calibrated importance
    test_quantize.py      # End-to-end quantization
    test_ops.py           # Grouped-GEMM operator
  benchmarks/
    bench_ops.py          # GPU latency benchmark
  pyproject.toml
  LICENSE                 # Apache 2.0
  README.md
```

---

## Installation

```bash
git clone https://github.com/DrXuQian/RDQuant.git
cd RDQuant
pip install -e ".[dev]"
```

---

## Quick Start

### Data-free quantization (no calibration data needed)

```python
from transformers import AutoModelForCausalLM
from rdquant import quantize_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
qmodel = quantize_model(model, budget_avg_bits=5.3, ignore=["lm_head", "embed_tokens"])
qmodel.print_summary()
```

### Calibrated quantization (recommended)

```python
from rdquant import quantize_model
from rdquant.core.calibrate import compute_layer_importance

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

# Compute importance from 2 calibration samples
importance = compute_layer_importance(
    model, tokenizer,
    calib_texts=["The transformer architecture...", "Large language models..."],
    metric="perturb", max_samples=2,
    ignore=["lm_head", "embed_tokens"],
)

qmodel = quantize_model(
    model, budget_avg_bits=5.3,
    ignore=["lm_head", "embed_tokens"],
    layer_importance=importance,
)
```

---

## API Reference

### `quantize_model`

```python
qmodel = quantize_model(
    model,
    budget_avg_bits=5.3,                    # target avg bits per weight
    formats=["NVFP4", "FP8", "FP16"],      # candidate formats
    ignore=["lm_head", "embed_tokens"],     # skip these layers
    per_layer_budget=False,                 # global cross-layer allocation
    layer_importance=None,                  # calibrated weights (or None for data-free)
)
```

### `QuantizedModel`

```python
qmodel.print_summary()                         # per-layer allocation table
qmodel.save_pretrained("path/")                 # save checkpoint
qmodel = QuantizedModel.from_pretrained("path/") # load checkpoint
y = qmodel(input_ids)                           # forward with fake quantization
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

Apache 2.0 -- see [LICENSE](LICENSE).
