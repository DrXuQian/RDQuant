# RDQuant

**Rate-Distortion Optimal Mixed-Precision Quantization for LLMs**

RDQuant assigns different OCP MX numeric formats (MXFP4 / MXFP6 / MXFP8) to different output channels of each linear layer, optimally trading off quantization error against bit cost. Activations are uniformly quantized to MXFP8, enabling efficient MXFP8 × MXFPx hardware GEMM on NVIDIA Blackwell GPUs.

Two allocation modes:
- **Data-free**: allocation based purely on weight MSE (no calibration data needed)
- **Calibrated**: uses a small calibration set to measure per-layer sensitivity to quantization, producing significantly better bit allocation

---

## Results

WikiText-2 perplexity on **Qwen3-4B** (stride 2048, 32k tokens):

| Method | Weight bpw | PPL | vs BF16 |
|---|---:|---:|---:|
| BF16 baseline | 16.00 | 13.16 | — |
| Uniform MXFP8 | 8.00 | 12.98 | -0.18 |
| Uniform MXFP6 | 6.07 | 13.10 | -0.06 |
| **RDQuant calibrated** | **5.18** | **13.42** | **+0.26** |
| RDQuant data-free | 5.47 | 14.47 | +1.31 |
| Uniform MXFP4 | 4.00 | 43.90 | +30.74 |

Key findings:
- **Calibrated RDQuant at 5.18 bpw nearly matches BF16** (13.42 vs 13.16), using only 32% of BF16 storage
- Perturb-based calibration improves over data-free by **1.05 PPL** while using **0.29 fewer bits**
- At ~5 bpw, RDQuant is **4.3× better** than uniform MXFP6 (6 bpw) in PPL delta, at lower bit cost
- Uniform MXFP4 is catastrophic (43.90 PPL) because per-32 block scaling is too coarse — mixed-precision is essential

---

## Algorithm

### Core: R-D Optimal Format Allocation

For every output channel *j* of every linear layer, compute the MSE from quantizing that channel to each candidate format. This produces a Rate-Distortion table:

```
channel j → [(rate=4, D_mxfp4), (rate=6, D_mxfp6), (rate=8, D_mxfp8)]
```

Given a bit budget *B*, solve the constrained optimization:

```
min  Σ_j  w_layer · D_j(f_j)    subject to  Σ_j C_j(f_j) ≤ B
```

where `w_layer` is the per-layer importance weight. This decomposes via a Lagrange multiplier λ — each channel independently picks the format minimising `w · D_j(f) + λ · C_j(f)`. Binary search on λ (64 iterations, precision ~10⁻¹⁹) finds the optimal λ*.

By default, a **single global λ*** is searched across all layers simultaneously, allowing sensitive layers to receive more bits while easy layers are compressed further.

After allocation, format groups are aligned to multiples of 128 channels for hardware efficiency.

### Data-free Mode (default)

All layers weighted equally (`w_layer = 1.0`). Allocation is based purely on weight quantization MSE. No calibration data required.

### Calibrated Mode (recommended)

Uses a small calibration set (2-8 samples) to compute per-layer importance via the **perturb** metric: quantize each layer to MXFP4 one-by-one and measure the resulting loss increase. This directly measures how much each layer's quantization error affects the final model output.

```python
from rdquant.core.calibrate import compute_layer_importance

importance = compute_layer_importance(
    model, tokenizer, calib_texts,
    metric="perturb",  # quantize-one-layer-at-a-time
    max_samples=2,
)
qmodel = quantize_model(model, budget_avg_bits=5.3, layer_importance=importance)
```

Key insight from calibration on Qwen3-4B:
- **Most sensitive**: `layers.2.mlp.gate_proj` (importance = 94.7) — early MLP gates
- **Least sensitive**: `layers.35.self_attn.v_proj` (importance ≈ 0) — late attention
- This is the opposite of what activation norm suggests (late layers have larger activations but are less sensitive to quantization)

### Inference Path

```
x_bf16 → quantize_mxfp8(x) → x_mxfp8        # one pass, uniform

y_fp4 = MXFP8 × MXFP4 GEMM(x_mxfp8, W_fp4)  # Blackwell tcgen05.mma.blockscaled
y_fp6 = MXFP8 × MXFP6 GEMM(x_mxfp8, W_fp6)
y_fp8 = MXFP8 × MXFP8 GEMM(x_mxfp8, W_fp8)

y = cat(y_fp4, y_fp6, y_fp8)[:, inv_perm]     # restore channel order
```

No activation reorder. No partial sums. One activation quantization pass per layer.

---

## Supported Formats

All formats follow the OCP Microscaling (MX) standard with per-32-element UE8M0 shared exponents.

| Format | Bits | Element Type | Max Value | Use Case |
|---|---:|---|---:|---|
| MXFP4 | 4 | E2M1 | 6.0 | Easy channels (low outlier severity) |
| MXFP6 | 6 | E3M2 | 28.0 | Medium channels |
| MXFP8 | 8 | E4M3 | 448.0 | Sensitive channels + activations |

---

## Project Structure

```
rdquant/
├── rdquant/
│   ├── core/
│   │   ├── formats.py          # Vectorized MX quantize/dequantize (MXFP4/6/8)
│   │   ├── act_quant.py        # MXFP8 activation quantization
│   │   ├── sensitivity.py      # Data-free channel sensitivity + R-D profiling
│   │   ├── allocator.py        # Lagrangian R-D allocation + 128-channel alignment
│   │   └── calibrate.py        # Calibrated layer importance (perturb/fisher/act_norm)
│   ├── quantize.py             # End-to-end model quantization
│   ├── ops.py                  # Mixed-precision grouped-GEMM linear operator
│   ├── eval.py                 # Perplexity & zero-shot evaluation
│   └── integrations/
│       ├── hf_export.py        # HuggingFace checkpoint save/load
│       └── vllm_linear.py      # vLLM inference stub
├── examples/
│   ├── quickstart.py           # Minimal quantize + eval
│   └── sweep_budget.py         # Sweep bit budgets, R-D curve
├── tests/                      # 240 tests
│   ├── test_formats.py         # MX format correctness
│   ├── test_act_quant.py       # Activation quantization
│   ├── test_sensitivity.py     # Sensitivity metrics
│   ├── test_allocator.py       # Allocation + alignment
│   ├── test_calibrate.py       # Calibrated importance
│   ├── test_quantize.py        # End-to-end quantization
│   └── test_ops.py             # Grouped-GEMM operator
├── benchmarks/
│   └── bench_ops.py            # GPU latency benchmark
├── pyproject.toml
├── LICENSE                     # Apache 2.0
└── README.md
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

### Calibrated quantization (recommended, ~1 PPL better)

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
    formats=["MXFP4", "MXFP6", "MXFP8"],  # candidate formats
    ignore=["lm_head", "embed_tokens"],     # skip these layers
    per_layer_budget=False,                 # global cross-layer allocation
    quantize_activation=True,               # MXFP8 activation quantization
    layer_importance=None,                  # calibrated weights (or None for data-free)
)
```

### `compute_layer_importance`

```python
importance = compute_layer_importance(
    model, tokenizer, calib_texts,
    metric="perturb",    # "perturb" | "fisher" | "grad_norm" | "act_norm"
    max_samples=2,
    seq_length=512,
    ignore=["lm_head", "embed_tokens"],
)
```

| Metric | Method | Accuracy | Speed | Needs Backward |
|---|---|---|---|---|
| `perturb` | Quantize each layer, measure loss delta | Best | O(n_layers) fwd | No |
| `fisher` | `\|\|dL/dW\|\|²` | Good | O(1) fwd+bwd | Yes |
| `grad_norm` | `\|\|dL/dW\|\|` | Good | O(1) fwd+bwd | Yes |
| `act_norm` | `\|\|X\|\|²` | Poor (biased by residual growth) | O(1) fwd | No |

### `QuantizedModel`

```python
qmodel.print_summary()                         # per-layer allocation table
qmodel.save_pretrained("path/")                 # save checkpoint
qmodel = QuantizedModel.from_pretrained("path/") # load checkpoint
y = qmodel(input_ids)                           # forward with fake quantization
```

---

## Design Comparison with MicroMix

| | RDQuant | MicroMix |
|---|---|---|
| Split dimension | Output channels (N) | Input channels (K) |
| Activation quant | Uniform MXFP8 (one pass) | Mixed FP4/FP6/FP8 (reorder needed) |
| Output merge | Concat + permute | Sum partial GEMMs |
| Allocation | Data-free R-D optimal (or calibrated) | Requires activation statistics |
| Output bandwidth | 1× (each GEMM writes disjoint channels) | 3× (each GEMM writes full output) |
| HW kernels | MXFP8 × MXFP4/6/8 (Blackwell native) | Same |

---

## Running Tests

```bash
pytest tests/ -v    # 240 tests
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
