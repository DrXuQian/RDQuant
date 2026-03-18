# RDQuant

**Rate-Distortion Optimal Mixed-Precision Quantization for LLMs**

RDQuant assigns different MX numeric formats (MXFP4 / MXFP6 / MXFP8) to
different output channels of each linear layer in a large language model,
optimally trading off quantization error against bit cost — without requiring
any calibration data.

Activations are uniformly quantized to MXFP8 before each GEMM, enabling
efficient MXFP8 x MXFPx hardware execution.

---

## Algorithm

RDQuant formulates mixed-precision format assignment as a Rate-Distortion optimisation problem and solves it with a Lagrangian binary search. The entire pipeline is **data-free** — only the weight tensors are needed.

### Step 1: Per-channel R-D profiling (`rdquant/core/formats.py`, `sensitivity.py`)

For every output channel *j* of every linear layer, compute the MSE
introduced by quantizing that channel to each candidate format
(MXFP4 / MXFP6 / MXFP8). This produces an R-D table:

```
channel j -> [(rate=4, distortion=D_mxfp4), (rate=6, distortion=D_mxfp6),
              (rate=8, distortion=D_mxfp8)]
```

All quantizers are **fully vectorized** with PyTorch — no per-element Python
loops. All MX formats use per-32-element UE8M0 shared exponents following
the OCP MX standard. MXFP4 uses nearest-LUT quantization (16 E2M1 values).
MXFP6/MXFP8 use direct bit-field encoding.

### Step 2: Lagrangian allocation (`rdquant/core/allocator.py`)

Given a bit budget *B* (e.g. 5.3 avg bits), solve:

```
min  sum_j D_j(f_j)   subject to  sum_j C_j(f_j) <= B
```

This decomposes via a Lagrange multiplier lambda: each channel independently picks
the format minimising `D_j(f) + lambda * C_j(f)`. Binary search on lambda (64
iterations) finds the optimal lambda* where total cost meets the budget.

After allocation, format groups are aligned to multiples of 128 channels for
efficient hardware execution.

### Step 3: Global cross-layer budget (default)

When `per_layer_budget=False`, a single lambda* is searched across **all** layers
simultaneously. This allows sensitive layers to receive more bits while easy
layers are compressed further.

### Step 4: Inference with MXFP8 activation quantization

Activations are quantized once to MXFP8, then each format group performs an
MXFP8 x MXFPx GEMM. Channels are permuted to group by format (all MXFP4
channels first, then MXFP6, MXFP8), and the inverse permutation restores
original channel order.

---

## Supported Formats

| Format | Bits | Standard | Block Scale |
|---|---:|---|---|
| MXFP4 | 4 | OCP MX FP4 E2M1 | per-32-element UE8M0 shared exponent |
| MXFP6 | 6 | OCP MX FP6 E3M2 | per-32-element UE8M0 shared exponent |
| MXFP8 | 8 | OCP MX FP8 E4M3 | per-32-element UE8M0 shared exponent |

---

## Project Structure

```
rdquant/
├── rdquant/
│   ├── core/
│   │   ├── formats.py          # Vectorized quantize/dequantize for MXFP4/MXFP6/MXFP8
│   │   ├── act_quant.py        # MXFP8 activation quantization
│   │   ├── sensitivity.py      # Data-free channel sensitivity metrics + R-D profiling
│   │   └── allocator.py        # Lagrangian R-D optimal format allocation + 128-alignment
│   ├── quantize.py             # End-to-end model quantization (QuantizedModel)
│   ├── ops.py                  # Mixed-precision linear operators
│   ├── eval.py                 # Perplexity & zero-shot evaluation wrappers
│   └── integrations/
│       ├── hf_export.py        # Save/load quantized models
│       └── vllm_linear.py      # vLLM inference stub (MXFP8 x MXFPx GEMM)
├── examples/
│   ├── quickstart.py           # Minimal quantize + eval example
│   └── sweep_budget.py         # Sweep bit budgets, print R-D table
├── tests/
│   ├── test_formats.py         # MX format round-trip, MSE monotonicity, edge cases
│   ├── test_act_quant.py       # MXFP8 activation quantization tests
│   ├── test_sensitivity.py     # Ordering, correlation, determinism
│   ├── test_allocator.py       # Optimality, budget, permutation, alignment
│   ├── test_quantize.py        # Model quantization end-to-end + act quant
│   └── test_ops.py             # Mixed-precision linear + activation quantization
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

Optional extras:

```bash
pip install -e ".[eval]"      # lm-eval-harness + datasets for perplexity eval
pip install matplotlib        # for sweep_budget.py plots
```

---

## Quick Start

```python
from transformers import AutoModelForCausalLM
from rdquant import quantize_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

# Quantize to ~5.3 avg bits per weight with MXFP8 activation quantization
qmodel = quantize_model(
    model,
    budget_avg_bits=5.3,
    ignore=["lm_head", "embed_tokens"],
    quantize_activation=True,  # MXFP8 activation quantization (default)
)
qmodel.print_summary()
qmodel.save_pretrained("./Qwen3-4B-RDQuant-5.3bit")
```

Command-line:

```bash
python examples/quickstart.py \
    --model Qwen/Qwen3-4B \
    --budget 5.3 \
    --output ./Qwen3-4B-RDQuant \
    --eval
```

---

## API Reference

### `quantize_model`

```python
from rdquant import quantize_model

qmodel = quantize_model(
    model,
    budget_avg_bits=5.3,         # target average bits per weight element
    formats=["MXFP4","MXFP6","MXFP8"],
    sensitivity_metric="mse",    # "mse" | "weighted_mse" | "kurtosis" | ...
    ignore=["lm_head"],          # layer name patterns to skip
    per_layer_budget=False,      # True = per-layer, False = global lambda
    quantize_activation=True,    # MXFP8 activation quantization
)
```

### `QuantizedModel`

```python
qmodel.print_summary()              # per-layer format allocation table
qmodel.save_pretrained("path/")     # save to directory
qmodel = QuantizedModel.from_pretrained("path/")  # load back
```

### Evaluation

```python
from rdquant.eval import eval_perplexity, eval_zero_shot

ppl = eval_perplexity(qmodel, tokenizer, dataset="wikitext")
scores = eval_zero_shot(qmodel, tokenizer, tasks=["arc_easy", "hellaswag"])
```

### Sensitivity Metrics

```python
from rdquant.core.sensitivity import compute_sensitivity

scores = compute_sensitivity(weight, metric="mse")       # direct quant MSE
scores = compute_sensitivity(weight, metric="kurtosis")   # tail heaviness
scores = compute_sensitivity(weight, metric="max_over_std") # outlier severity
```

---

## Running Tests

```bash
pytest tests/ -v          # 231 tests covering all formats + allocator + end-to-end
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
