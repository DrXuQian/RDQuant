# RDQuant

**Rate-Distortion Optimal Mixed-Precision Quantization for LLMs**

RDQuant assigns different numeric formats (NVFP4 / MXFP6 / MXFP8 / FP16) to
different output channels of each linear layer in a large language model,
optimally trading off quantization error against bit cost — without requiring
any calibration data.

---

## Installation

```bash
git clone https://github.com/rdquant/rdquant
cd rdquant
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[eval]"      # lm-eval-harness + datasets for perplexity / accuracy eval
pip install matplotlib        # for sweep_budget.py plots
```

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdquant import quantize_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Quantize to ~5.3 avg bits (roughly 25 % smaller than FP16)
qmodel = quantize_model(model, budget_avg_bits=5.3)
qmodel.print_summary()
qmodel.save_pretrained("./llama2-7b-rdquant-5.3bit")
```

Command-line quickstart:

```bash
python examples/quickstart.py \
    --model sshleifer/tiny-gpt2 \
    --budget 5.3 \
    --output ./tiny-gpt2-rdquant \
    --eval
```

---

## How It Works

1. **Sensitivity analysis** (`rdquant/core/sensitivity.py`):
   For every output channel of every linear layer, compute the MSE introduced
   by quantizing that channel to each candidate format (NVFP4/MXFP6/MXFP8/FP16).
   This produces a Rate-Distortion (R-D) table: `{channel: [(rate, distortion), ...]}`.

2. **R-D optimal allocation** (`rdquant/core/allocator.py`):
   Formulate the problem as a Lagrangian minimisation:

   ```
   min  Σ_j D_j(f_j)   subject to  Σ_j C_j(f_j) ≤ B
   ```

   A binary search over the Lagrange multiplier λ finds the optimal λ* such
   that total bit cost equals the budget B. Each channel then independently
   picks the format that minimises `D_j(f) + λ* · C_j(f)`.

3. **Global budget** (default):
   When `per_layer_budget=False`, a single λ* is searched across **all** layers
   simultaneously, allowing aggressive layers to receive more bits while
   easy-to-quantize layers are compressed further.

4. **Fake-quantization forward** (`rdquant/quantize.py`):
   Quantized weights are stored in packed format. During inference the weight
   is dequantized to FP32, inverse-permuted to restore original channel order,
   and a standard `F.linear` GEMM is executed. A future vLLM integration
   (`rdquant/integrations/vllm_linear.py`) will replace this with per-format
   hardware kernels.

### Supported Formats

| Format | Bits | Standard |
|--------|------|----------|
| NVFP4  | 4    | NVIDIA FP4 E2M1 with per-16-element FP8 block scale |
| MXFP6  | 6    | OCP MX FP6 E3M2 with per-32-element shared exponent |
| MXFP8  | 8    | OCP MX FP8 E4M3 with per-32-element shared exponent |
| FP16   | 16   | IEEE half-precision (lossless passthrough)           |

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
    per_layer_budget=False,      # True = independent per-layer, False = global lambda
)
```

### `QuantizedModel`

```python
qmodel.print_summary()              # per-layer format allocation table
qmodel.save_pretrained("path/")     # save to directory
qmodel = QuantizedModel.from_pretrained("path/")   # load back
```

### Evaluation

```python
from rdquant.eval import eval_perplexity, eval_zero_shot

ppl = eval_perplexity(qmodel, tokenizer, dataset="wikitext")
scores = eval_zero_shot(qmodel, tokenizer, tasks=["arc_easy", "hellaswag"])
```

---

## Accuracy Results

*Table placeholder — results will be added after benchmarking on LLaMA-2/3 models.*

| Model         | Budget (bits) | WikiText-2 PPL | ARC-E | HellaSwag |
|---------------|---------------|----------------|-------|-----------|
| LLaMA-2-7B    | FP16 (ref)    | —              | —     | —         |
| LLaMA-2-7B    | 5.3           | —              | —     | —         |
| LLaMA-2-7B    | 4.5           | —              | —     | —         |

---

## Citation

If you use RDQuant in your research, please cite:

```bibtex
@software{rdquant2024,
  title  = {{RDQuant}: Rate-Distortion Optimal Mixed-Precision Quantization for LLMs},
  year   = {2024},
  url    = {https://github.com/rdquant/rdquant},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
