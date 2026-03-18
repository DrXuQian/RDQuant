# RDQuant

**Rate-Distortion Optimal Mixed-Precision Quantization for LLMs**

RDQuant assigns different numeric formats (NVFP4 / MXFP6 / MXFP8 / FP16) to
different output channels of each linear layer in a large language model,
optimally trading off quantization error against bit cost — without requiring
any calibration data.

---

## Results

WikiText-2 perplexity on **Qwen3-4B** (stride 2048, 32k tokens):

| Method | Avg Bits | WikiText-2 PPL | Size vs FP16 |
|---|---:|---:|---:|
| BF16 (baseline) | 16.00 | 13.16 | 100% |
| Uniform MXFP8 | 8.05 | 12.83 | 50% |
| **RDQuant mixed-precision** | **5.45** | **13.26** | **34%** |
| Uniform NVFP4 | 4.00 | 14.74 | 25% |

Key findings:
- **RDQuant at 5.45 bpw recovers near-BF16 quality** (13.26 vs 13.16 PPL), using only 34% of BF16 storage.
- Compared to uniform NVFP4 (14.74 PPL), RDQuant reduces PPL by **1.48** (+10%) with only 1.45 extra bits per weight.
- At equivalent quality (~13.2 PPL), RDQuant uses **5.45 bits** vs MXFP8's **8 bits** — a **32% size reduction**.

---

## Algorithm

RDQuant formulates mixed-precision format assignment as a Rate-Distortion optimisation problem and solves it with a Lagrangian binary search. The entire pipeline is **data-free** — only the weight tensors are needed.

### Step 1: Per-channel R-D profiling (`rdquant/core/formats.py`, `sensitivity.py`)

For every output channel *j* of every linear layer, compute the MSE
introduced by quantizing that channel to each candidate format
(NVFP4 / MXFP6 / MXFP8 / FP16). This produces an R-D table:

```
channel j → [(rate=4, distortion=D_fp4), (rate=6, distortion=D_fp6),
             (rate=8, distortion=D_fp8), (rate=16, distortion=D_fp16)]
```

All quantizers are **fully vectorized** with PyTorch — no per-element Python
loops. NVFP4 uses a 16-entry E2M1 LUT with FP8(E4M3) block scales (group
size 16). MXFP6/MXFP8 use direct bit-field encoding with per-32-element
shared exponents following the OCP MX standard.

### Step 2: Lagrangian allocation (`rdquant/core/allocator.py`)

Given a bit budget *B* (e.g. 5.3 avg bits), solve:

```
min  Σ_j D_j(f_j)   subject to  Σ_j C_j(f_j) ≤ B
```

This decomposes via a Lagrange multiplier λ: each channel independently picks
the format minimising `D_j(f) + λ · C_j(f)`. Binary search on λ (64
iterations, precision ~10⁻¹⁹) finds the optimal λ* where total cost meets the
budget.

### Step 3: Global cross-layer budget (default)

When `per_layer_budget=False`, a single λ* is searched across **all** layers
simultaneously. This allows sensitive layers (e.g. attention V-proj) to
receive more bits while easy layers (e.g. FFN down-proj) are compressed
further — yielding better quality than per-layer budgets at the same total
size.

### Step 4: Fake-quantization inference (`rdquant/quantize.py`)

Channels are permuted to group by format (all NVFP4 channels first, then
MXFP6, MXFP8, FP16). During inference, each group is dequantized, the
inverse permutation restores original channel order, and a standard
`F.linear` GEMM is executed. A future vLLM integration will replace this with
native per-format Tensor Core kernels on NVIDIA Blackwell GPUs.

---

## Supported Formats

| Format | Bits | Standard | Block Scale |
|---|---:|---|---|
| NVFP4 | 4 | NVIDIA FP4 E2M1 | per-16-element FP8(E4M3) block scale |
| MXFP6 | 6 | OCP MX FP6 E3M2 | per-32-element shared exponent |
| MXFP8 | 8 | OCP MX FP8 E4M3 | per-32-element shared exponent |
| FP16 | 16 | IEEE half-precision | lossless passthrough |

---

## Project Structure

```
rdquant/
├── rdquant/
│   ├── core/
│   │   ├── formats.py          # Vectorized quantize/dequantize for all 4 formats
│   │   ├── sensitivity.py      # Data-free channel sensitivity metrics + R-D profiling
│   │   └── allocator.py        # Lagrangian R-D optimal format allocation
│   ├── quantize.py             # End-to-end model quantization (QuantizedModel)
│   ├── eval.py                 # Perplexity & zero-shot evaluation wrappers
│   └── integrations/
│       ├── hf_export.py        # Save/load quantized models (safetensors)
│       └── vllm_linear.py      # vLLM inference stub (Phase 4)
├── examples/
│   ├── quickstart.py           # Minimal quantize + eval example
│   └── sweep_budget.py         # Sweep bit budgets, print R-D table
├── tests/
│   ├── test_formats.py         # 49 tests: round-trip, MSE monotonicity, edge cases
│   ├── test_sensitivity.py     # 56 tests: ordering, correlation, determinism
│   ├── test_allocator.py       # 48 tests: optimality, budget, permutation
│   └── test_quantize.py        # 14 tests: model quantization end-to-end
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

# Quantize to ~5.3 avg bits per weight
qmodel = quantize_model(
    model,
    budget_avg_bits=5.3,
    ignore=["lm_head", "embed_tokens"],
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
    formats=["NVFP4","MXFP6","MXFP8","FP16"],
    sensitivity_metric="mse",    # "mse" | "weighted_mse" | "kurtosis" | ...
    ignore=["lm_head"],          # layer name patterns to skip
    per_layer_budget=False,      # True = per-layer, False = global lambda
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
pytest tests/ -v          # 167 tests, all formats + allocator + end-to-end
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
