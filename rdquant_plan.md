# RDQuant: Rate-Distortion Optimal Mixed-Precision Quantization for LLMs

## Implementation Plan for Claude Code

---

## Project Overview

RDQuant is a data-free, channel-wise mixed-precision quantization method for LLMs. It assigns different numeric formats (NVFP4/MXFP6/MXFP8/FP16) to different output channels based on Rate-Distortion optimal allocation. The key insight: channels with flat weight distributions can tolerate aggressive low-bit quantization, while channels with sharp/outlier-heavy distributions need higher precision. By using a Lagrangian λ-sweep, we find the globally optimal format assignment under any given bit budget — without needing any calibration data.

Target hardware: NVIDIA Blackwell (RTX 5070), where all four formats have native Tensor Core support.

---

## Directory Structure

```
rdquant/
├── README.md
├── LICENSE                          # Apache 2.0
├── setup.py
├── pyproject.toml
├── rdquant/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── sensitivity.py           # Phase 1: data-free channel sensitivity metrics
│   │   ├── allocator.py             # Phase 1: R-D Lagrangian allocation
│   │   └── formats.py              # Phase 1: quantize/dequantize for each format
│   ├── quantize.py                  # Phase 2: end-to-end model quantization
│   ├── eval.py                      # Phase 2: evaluation wrapper
│   └── integrations/
│       ├── __init__.py
│       ├── hf_export.py             # Phase 3: HuggingFace checkpoint export
│       └── vllm_linear.py           # Phase 4: vLLM inference integration (stub)
├── examples/
│   ├── quickstart.py                # Phase 2: minimal usage example
│   ├── quantize_llama3.py           # Phase 2: full example with Llama-3.2-1B/3B
│   └── sweep_budget.py             # Phase 2: sweep avg bits and plot accuracy vs size
├── tests/
│   ├── test_formats.py              # Phase 1: format correctness tests
│   ├── test_sensitivity.py          # Phase 1: sensitivity computation tests
│   ├── test_allocator.py            # Phase 1: allocation optimality tests
│   └── test_quantize.py             # Phase 2: end-to-end quantization tests
└── benchmarks/
    └── accuracy/
        └── run_eval.sh              # Phase 2: lm-eval-harness benchmark script
```

---

## Phase 1: Core Algorithm (implement first, test thoroughly)

### Task 1.1: `rdquant/core/formats.py`

Implement quantize and dequantize functions for each format. All functions operate on 1D torch tensors (a single output channel's weight vector). No calibration data needed.

```python
"""
Numeric format definitions and quantize/dequantize implementations.

Supported formats:
  - NVFP4:  E2M1 with per-16-element FP8(E4M3) block scale + per-tensor FP32 global scale
  - MXFP6:  E3M2 or E2M3 with per-32-element shared exponent (OCP MX standard)
  - MXFP8:  E4M3 or E5M2 with per-32-element shared exponent
  - FP16:   IEEE half-precision, no quantization needed

Each format exposes:
  - bits_per_element: int — the nominal bit-width (4, 6, 8, 16)
  - quantize(tensor) -> QuantizedTensor  — returns packed data + metadata
  - dequantize(qtensor) -> tensor — reconstruct full-precision approximation
  - compute_mse(tensor) -> float — quantize, dequantize, compute MSE (convenience)
"""
```

Implementation notes:
- NVFP4: Use a lookup table of 16 FP4 E2M1 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {+, -}. Block scale = absmax of each 16-element group, quantized to FP8 E4M3. Global scale = max of all block scales.
- MXFP6/MXFP8: Follow OCP Microscaling spec. Shared exponent per 32-element group.
- FP16: Identity (passthrough). MSE = 0.
- All quantize functions should work in pure PyTorch (no CUDA kernels needed at this stage).
- Return a `QuantizedTensor` dataclass with: `data` (packed bytes), `scales`, `format_name`, `original_shape`.

Testing (test_formats.py):
- Round-trip correctness: quantize then dequantize, check values match LUT entries
- MSE monotonicity: MSE(NVFP4) >= MSE(MXFP6) >= MSE(MXFP8) >= MSE(FP16) ≈ 0
- Edge cases: all-zero tensor, single-value tensor, tensor with large outlier
- Block scale correctness: verify block scales are valid FP8 E4M3 values
- Test with random normal tensors of various sizes (128, 1024, 4096)

### Task 1.2: `rdquant/core/sensitivity.py`

Implement data-free channel sensitivity metrics. All metrics take a 2D weight matrix [N_out, N_in] and return a 1D tensor of per-channel sensitivity scores [N_out].

```python
"""
Data-free channel sensitivity metrics.

All functions take a weight matrix W of shape [N_out, N_in] and return
a sensitivity score per output channel of shape [N_out].

Higher sensitivity = channel is harder to quantize = needs more bits.

Available metrics:
  - "mse": Direct quantization MSE at the lowest format (NVFP4).
           Most accurate, slightly slower.
  - "weighted_mse": MSE × ||W_j||^2. Accounts for weight magnitude.
  - "max_over_std": max(|W_j|) / std(W_j). Measures outlier severity.
  - "kurtosis": Excess kurtosis of W_j. Measures tail heaviness.
  - "range_ratio": (max - min) / mean_abs. Measures dynamic range.
  
Recommended default: "mse" (best correlation with actual quantization loss).
"""

def compute_sensitivity(
    weight: torch.Tensor,          # [N_out, N_in]
    metric: str = "mse",           # sensitivity metric name
    base_format: str = "NVFP4",    # compute MSE at this format
) -> torch.Tensor:                 # [N_out] sensitivity scores
    ...

def compute_rd_points(
    weight: torch.Tensor,          # [N_out, N_in]
    formats: list[str] = ["NVFP4", "MXFP6", "MXFP8", "FP16"],
) -> dict:
    """
    For each output channel, compute the (rate, distortion) pair
    at each format.
    
    Returns:
        rd_table: dict mapping channel_idx -> list of {format, rate, distortion, cost}
        
    This is the input to the R-D allocator.
    cost = bits_per_element * N_in (total bits for this channel at this format)
    """
    ...
```

Implementation notes:
- `compute_rd_points` calls `formats.compute_mse()` for each channel × format combination.
- For a 4096×4096 matrix with 4 formats, this is 4096×4 = 16384 MSE computations.
- Each MSE computation is O(N_in). Total: O(N_out × N_formats × N_in).
- For LLaMA-7B (all linear layers), this should take < 2 minutes on CPU.
- Use `torch.no_grad()` everywhere.

Testing (test_sensitivity.py):
- Sensitivity ordering: channel with large outlier should have higher sensitivity than uniform channel
- Metric agreement: different metrics should produce similar relative orderings (rank correlation > 0.7)
- Determinism: same weight -> same sensitivity
- Shape correctness: output shape = [N_out]

### Task 1.3: `rdquant/core/allocator.py`

Implement the R-D Lagrangian allocation algorithm. This is the core of the project.

```python
"""
Rate-Distortion optimal channel-wise format allocation.

Given:
  - R-D points for each channel (from sensitivity.py)
  - A bit budget (average bits per element)
  
Find:
  - Format assignment for each channel that minimizes total distortion
    subject to the bit budget constraint.

Algorithm:
  1. For each channel, compute Lagrangian cost: D_j(f) + λ * C_j(f)
     for each format f, where D = distortion (MSE), C = bit cost.
  2. Binary search on λ to find the value where total cost = budget.
  3. At optimal λ*, each channel independently picks its best format.
  
Returns:
  - AllocationResult with per-channel format assignments, permutation
    indices (sorted by format for efficient grouped GEMM), and summary stats.
"""

@dataclass
class AllocationResult:
    assignments: dict[int, str]      # channel_idx -> format_name
    permutation: torch.Tensor        # [N_out] reorder indices (group by format)
    inv_permutation: torch.Tensor    # [N_out] inverse reorder
    splits: dict[str, int]           # format_name -> num_channels
    avg_bits: float                  # actual average bits achieved
    total_distortion: float          # sum of MSE across all channels
    lambda_star: float               # optimal Lagrange multiplier
    
    # Per-format stats
    format_stats: dict[str, dict]    # {format: {n_channels, avg_mse, total_bits}}

def allocate(
    rd_table: dict,                  # from compute_rd_points()
    budget_avg_bits: float,          # target average bits per element (e.g. 5.3)
    formats: list[str] = ["NVFP4", "MXFP6", "MXFP8", "FP16"],
    n_elements_per_channel: int = None,  # N_in, inferred if not given
) -> AllocationResult:
    """
    Main allocation function.
    
    The λ sweep uses binary search:
      - λ=0: all channels pick FP16 (max bits, min distortion)
      - λ=∞: all channels pick NVFP4 (min bits, max distortion)
      - Binary search finds λ* where total_bits ≈ budget
      
    After finding λ*, channels are sorted and grouped by assigned format.
    The permutation puts all NVFP4 channels first, then MXFP6, etc.
    """
    ...

def allocate_layer(
    weight: torch.Tensor,            # [N_out, N_in]
    budget_avg_bits: float,
    formats: list[str] = ["NVFP4", "MXFP6", "MXFP8", "FP16"],
    sensitivity_metric: str = "mse",
) -> AllocationResult:
    """
    Convenience: compute R-D points + allocate in one call for a single layer.
    """
    rd_table = compute_rd_points(weight, formats)
    return allocate(rd_table, budget_avg_bits, formats, weight.shape[1])

def sweep_budgets(
    weight: torch.Tensor,
    budgets: list[float],            # e.g. [4, 5, 6, 7, 8]
    formats: list[str] = ["NVFP4", "MXFP6", "MXFP8", "FP16"],
) -> list[AllocationResult]:
    """
    Run allocation at multiple budgets. Useful for plotting R-D curves.
    Computes rd_table once, then sweeps λ.
    """
    ...
```

Implementation notes for the λ binary search:
- Initialize `lambda_lo = 0`, `lambda_hi = 1.0`.
- First, find upper bound: double `lambda_hi` until total_cost < budget.
- Then binary search for 50 iterations (gives precision ~1e-15).
- At each λ, for each channel, pick format f that minimizes D_j(f) + λ * C_j(f).
  This is just comparing 4 numbers per channel.
- The permutation should order channels as: [all NVFP4 channels | all MXFP6 | all MXFP8 | all FP16].
  Within each format group, preserve original order (or sort by sensitivity for cache locality).

Testing (test_allocator.py):
- Budget satisfaction: assert actual avg_bits ≈ budget (within ±0.1)
- Optimality: for 2 channels, enumerate all possible assignments, verify allocator picks the best
- Monotonicity: larger budget → lower total distortion
- Extreme budgets: budget=4 → all NVFP4; budget=16 → all FP16
- Permutation correctness: inv_permutation[permutation] = identity
- Splits sum to N_out
- Lambda monotonicity: higher lambda → fewer bits used

---

## Phase 2: End-to-End Model Quantization

### Task 2.1: `rdquant/quantize.py`

End-to-end quantization of a HuggingFace model.

```python
"""
End-to-end mixed-precision quantization for HuggingFace models.

Usage:
    from rdquant import quantize_model
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    result = quantize_model(
        model,
        budget_avg_bits=5.3,
        formats=["NVFP4", "MXFP6", "MXFP8", "FP16"],
        ignore=["lm_head", "embed_tokens"],
    )
    result.save_pretrained("Llama-3.2-1B-RDQuant")
"""

def quantize_model(
    model: PreTrainedModel,
    budget_avg_bits: float = 5.3,
    formats: list[str] = ["NVFP4", "MXFP6", "MXFP8", "FP16"],
    sensitivity_metric: str = "mse",
    ignore: list[str] = None,        # layer name patterns to skip
    per_layer_budget: bool = False,   # True: each layer gets same budget
                                      # False: global budget across all layers
) -> QuantizedModel:
    """
    Quantize all Linear layers in the model.
    
    Steps:
    1. Iterate over all nn.Linear layers (excluding ignored ones)
    2. For each layer, call allocate_layer() to get format assignments
    3. Apply quantization: reorder channels, quantize each group
    4. Replace original weight with QuantizedWeight containing:
       - Packed weight tensors per format group
       - Scales per format group
       - Permutation index
       - Format splits
    5. Return QuantizedModel that can be saved/loaded/evaluated
    
    When per_layer_budget=False (default, recommended):
    - Compute R-D points for ALL layers first
    - Run a single global λ sweep across all layers
    - This allows aggressive layers (e.g. attention V proj) to get more bits
      while easy layers (e.g. FFN down proj) get fewer bits
    - Total model size matches budget
    """
    ...
```

The `QuantizedModel` should support:
- `save_pretrained(path)`: Save in HuggingFace-compatible format
- `forward(input_ids)`: Run inference using fake quantization (dequant → FP16 matmul)
  This is for accuracy evaluation, NOT for fast inference.
- Print a summary table showing per-layer format allocation

### Task 2.2: `rdquant/eval.py`

Evaluation wrapper that runs standard benchmarks.

```python
"""
Evaluation utilities for quantized models.

Supports:
  - Perplexity on WikiText-2
  - Zero-shot accuracy via lm-eval-harness (if installed)
  - Custom eval on any HuggingFace dataset
"""

def eval_perplexity(
    model: QuantizedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: str = "wikitext",       # "wikitext" or path to dataset
    max_samples: int = None,
    seq_length: int = 2048,
) -> float:
    """
    Compute perplexity using sliding window.
    Returns PPL as float.
    """
    ...

def eval_zero_shot(
    model: QuantizedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: list[str] = ["arc_easy", "arc_challenge", "hellaswag", "mmlu", "winogrande"],
    num_fewshot: int = 0,
) -> dict[str, float]:
    """
    Run zero-shot evaluation using lm-eval-harness.
    Returns dict of task -> accuracy.
    Requires: pip install lm-eval
    """
    ...
```

### Task 2.3: `examples/quickstart.py`

```python
"""
Minimal example: quantize a small model and evaluate.

Usage:
    python examples/quickstart.py --model meta-llama/Llama-3.2-1B \
                                   --budget 5.3 \
                                   --output ./llama-1b-rdquant
"""
```

This script should:
1. Load model and tokenizer
2. Print original model size
3. Run quantize_model()
4. Print allocation summary (per-layer table)
5. Evaluate PPL on WikiText-2
6. Compare with uniform NVFP4 (budget=4.0) and uniform MXFP8 (budget=8.0)
7. Save quantized model

### Task 2.4: `examples/sweep_budget.py`

Sweep average bits from 4.0 to 8.0 in 0.5 steps, evaluate PPL at each point, and print a table + save a matplotlib plot showing the R-D curve of the entire model.

Compare against:
- Uniform NVFP4 (4 bits)
- Uniform MXFP8 (8 bits)
- GPTQ INT4 (4 bits, if available)

This script demonstrates the key selling point: at the same average bit-width, RDQuant achieves lower perplexity than uniform quantization.

---

## Phase 3: HuggingFace Export

### Task 3.1: `rdquant/integrations/hf_export.py`

Save and load quantized models in a format compatible with HuggingFace Hub.

```python
"""
Export quantized model as HuggingFace checkpoint.

Directory layout:
  model_dir/
  ├── config.json                    # original model config
  ├── tokenizer.json                 # original tokenizer
  ├── quantization_config.json       # RDQuant-specific metadata
  ├── model.safetensors              # quantized weights
  └── README.md                      # auto-generated model card

quantization_config.json contains:
  - quant_method: "rdquant"
  - budget_avg_bits: 5.3
  - formats_used: ["NVFP4", "MXFP6", "MXFP8", "FP16"]
  - per-layer metadata: permutation, splits, format assignments
"""

def save_quantized(model: QuantizedModel, path: str):
    ...

def load_quantized(path: str, device: str = "cpu") -> QuantizedModel:
    ...
```

Weight naming convention in safetensors:
```
model.layers.0.self_attn.q_proj.weight_nvfp4     # packed FP4 data
model.layers.0.self_attn.q_proj.weight_mxfp6     # packed FP6 data
model.layers.0.self_attn.q_proj.weight_mxfp8     # packed FP8 data
model.layers.0.self_attn.q_proj.weight_fp16       # FP16 data
model.layers.0.self_attn.q_proj.scales_nvfp4      # FP8 block scales for FP4
model.layers.0.self_attn.q_proj.scales_mxfp6      # shared exponents for FP6
model.layers.0.self_attn.q_proj.scales_mxfp8      # shared exponents for FP8
model.layers.0.self_attn.q_proj.permutation        # int32 channel reorder index
model.layers.0.self_attn.q_proj.format_splits      # int32 [N_fp4, N_fp6, N_fp8, N_fp16]
```

---

## Phase 4: vLLM Integration (stub for now)

### Task 4.1: `rdquant/integrations/vllm_linear.py`

Write a stub for the vLLM quantized linear layer. This will be filled in later when we have Blackwell hardware access, but the interface should be defined now.

```python
"""
vLLM integration for RDQuant mixed-precision inference.

Phase 4A (multi-kernel, no new CUDA code):
  - Each format group uses vLLM's existing quantized GEMM kernel
  - Output is concatenated and permuted back to original order
  - 4 kernel launches per layer (can overlap with CUDA streams)

Phase 4B (future, single-kernel):
  - Based on MxMoE's grouped GEMM with micro-kernel specialization
  - Single persistent kernel launch for all format groups
"""

class RDQuantLinear(torch.nn.Module):
    """
    Mixed-precision linear layer.
    
    Forward pass (Phase 4A):
      y_fp4 = fp4_gemm(x, w_fp4, scales_fp4)
      y_fp6 = fp6_gemm(x, w_fp6, scales_fp6)
      y_fp8 = fp8_gemm(x, w_fp8, scales_fp8)
      y_fp16 = F.linear(x, w_fp16)
      y = cat([y_fp4, y_fp6, y_fp8, y_fp16])[inv_perm]
    """
    
    @classmethod
    def from_quantized(cls, config, weight_data) -> "RDQuantLinear":
        """Load from quantized checkpoint."""
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

---

## Implementation Order

Execute in this exact sequence. Each task should be a separate commit.

```
1. rdquant/core/formats.py           + tests/test_formats.py
2. rdquant/core/sensitivity.py       + tests/test_sensitivity.py
3. rdquant/core/allocator.py         + tests/test_allocator.py
4. rdquant/quantize.py               + tests/test_quantize.py
5. rdquant/eval.py
6. examples/quickstart.py
7. examples/sweep_budget.py
8. rdquant/integrations/hf_export.py
9. rdquant/integrations/vllm_linear.py (stub)
10. setup.py + pyproject.toml + README.md + LICENSE
```

---

## Technical Constraints

- Python >= 3.10
- PyTorch >= 2.1
- transformers >= 4.40
- All core code (rdquant/core/) must work on CPU. No CUDA required for quantization.
- Use torch.no_grad() and torch.inference_mode() everywhere during quantization.
- Type hints on all public functions.
- Docstrings on all public functions (Google style).
- No global state. All functions should be pure (or explicitly take/return state).

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.1",
    "transformers>=4.40",
    "safetensors>=0.4",
    "numpy>=1.24",
]

[project.optional-dependencies]
eval = ["lm-eval>=0.4"]
dev = ["pytest>=7.0", "matplotlib>=3.7"]
```

---

## Key Design Decisions (for Claude Code to follow)

1. **Data-free by default.** Never require calibration data for the core algorithm. The `eval.py` uses data for evaluation only.

2. **Global budget, not per-layer.** The default mode pools bit budget across all layers. This is better because some layers (e.g. attention output projection) are much more sensitive than others (e.g. FFN intermediate).

3. **Channel = output channel.** We partition along the output dimension (rows of W). This means each partition is a contiguous block of output neurons, which maps naturally to grouped GEMM.

4. **Permutation is per-layer.** Each linear layer has its own channel permutation. The inverse permutation is applied to the output of each layer, so the rest of the model sees the original channel order.

5. **Format hierarchy is fixed.** NVFP4 < MXFP6 < MXFP8 < FP16 in bits. The allocator exploits this ordering — if a channel is assigned MXFP6, all channels with lower sensitivity get NVFP4 or MXFP6 (never MXFP8).

6. **Fake quantization for eval.** Phase 2 uses dequantized FP16 weights for inference. This is slow but correct. Real acceleration comes in Phase 4 with vLLM.

7. **Reproducibility.** Set `torch.manual_seed(42)` in all tests. The core algorithm is deterministic (no randomness).
