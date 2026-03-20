"""
INT4/INT8 mixed-precision quantization with AWQ scaling and R-D allocation.

Combines:
  1. AWQ per-input-channel scaling (improves INT4 precision)
  2. RDQuant R-D allocation (chooses INT4 vs INT8 per output channel)
  3. INT8 -> 2x INT4 decomposition (single Marlin kernel launch)

This module is additive -- existing NVFP4/FP8/FP16 code is untouched.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from rdquant.core.formats import compute_mse_2d, get_bits_per_element
from rdquant.core.allocator import allocate, AllocationResult
from rdquant.core.sensitivity import compute_rd_points
from rdquant.int4_fusion import (
    Int4FusedLinear,
    quantize_to_int4_groupwise,
    quantize_to_int8_channelwise,
)


# ---------------------------------------------------------------------------
# QuantizedModelInt4: wrapper for INT4/INT8 quantized models
# ---------------------------------------------------------------------------

@dataclass
class Int4LayerInfo:
    """Per-layer information for INT4/INT8 quantized layers."""
    allocation: AllocationResult
    n_int4: int
    n_int8: int
    avg_bits: float


class QuantizedModelInt4(nn.Module):
    """Wrapper around a model with nn.Linear layers replaced by Int4FusedLinear.

    Args:
        model: The original (now modified in-place) model.
        layer_info: Dict mapping layer_name -> Int4LayerInfo.
        budget_avg_bits: The target budget used.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_info: dict[str, Int4LayerInfo],
        budget_avg_bits: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.layer_info = layer_info
        self.budget_avg_bits = budget_avg_bits

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_summary(self) -> None:
        """Print a per-layer format allocation table."""
        header = f"{'Layer':<50} {'N_out':>6} {'INT4':>6} {'INT8':>6} {'avg_bits':>9}"
        print(header)
        print("-" * len(header))
        total_bits = 0.0
        total_params = 0
        for name, info in self.layer_info.items():
            n_out = info.n_int4 + info.n_int8
            alloc = info.allocation
            # Infer n_in from allocation
            total_ch_bits = sum(
                stats["total_bits"]
                for stats in alloc.format_stats.values()
            )
            n_in = int(total_ch_bits / (n_out * alloc.avg_bits)) if (n_out * alloc.avg_bits) > 0 else 0
            print(f"{name:<50} {n_out:>6} {info.n_int4:>6} {info.n_int8:>6} {info.avg_bits:>9.2f}")
            total_bits += total_ch_bits
            total_params += n_out * n_in
        if total_params > 0:
            overall_bits = total_bits / total_params
        else:
            overall_bits = 0.0
        print("-" * len(header))
        print(f"{'TOTAL':<50} {'':>6} {'':>6} {'':>6} {overall_bits:>9.2f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_ignore(name: str, ignore: list[str]) -> bool:
    """Return True if layer name matches any pattern in ignore list."""
    if not ignore:
        return False
    for pattern in ignore:
        if fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def _get_named_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Collect all nn.Linear layers from a model recursively."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            result.append((name, module))
    return result


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule by dotted name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# quantize_model_int4
# ---------------------------------------------------------------------------

def quantize_model_int4(
    model: nn.Module,
    budget_avg_bits: float = 5.3,
    formats: Optional[list[str]] = None,
    ignore: Optional[list[str]] = None,
    awq_scales: Optional[dict[str, torch.Tensor]] = None,
    int4_group_size: int = 128,
    layer_importance: Optional[dict[str, float]] = None,
) -> QuantizedModelInt4:
    """Quantize model using INT4/INT8 with AWQ scaling.

    Steps:
      1. For each linear layer:
         a. Apply AWQ scaling: W' = W * diag(alpha) (if awq_scales provided)
         b. Compute R-D points for INT4 and INT8
         c. R-D allocate output channels to INT4 or INT8
      2. Quantize each group
      3. For INT8 channels: decompose to 2x UINT4 nibbles
      4. Return model with Int4FusedLinear layers

    Args:
        model: Any nn.Module (HuggingFace PreTrainedModel or custom).
        budget_avg_bits: Target average bits per element across all layers.
        formats: Allowed format names. Defaults to ["INT4", "INT8"].
        ignore: List of layer name patterns to skip quantization.
        awq_scales: Dict from compute_awq_scales(). Per-input-channel scales.
        int4_group_size: Group size for INT4 quantization. Default 128.
        layer_importance: Optional dict mapping layer name to importance weight.

    Returns:
        QuantizedModelInt4 wrapping the modified model.
    """
    if formats is None:
        formats = ["INT4", "INT8"]

    if ignore is None:
        ignore = []

    linear_layers = _get_named_linears(model)
    to_quantize = [
        (name, layer)
        for name, layer in linear_layers
        if not _should_ignore(name, ignore)
    ]

    layer_info: dict[str, Int4LayerInfo] = {}

    # Global lambda sweep: collect all R-D tables
    all_rd_tables: dict[str, dict] = {}
    all_n_in: dict[str, int] = {}
    all_weights_scaled: dict[str, torch.Tensor] = {}
    all_layer_awq: dict[str, Optional[torch.Tensor]] = {}

    with torch.inference_mode():
        for name, layer in to_quantize:
            weight = layer.weight.data.float()

            # Apply AWQ scaling if provided
            layer_awq = None
            if awq_scales is not None and name in awq_scales:
                alpha = awq_scales[name].to(weight.device).float()
                layer_awq = alpha
                weight_scaled = weight * alpha.unsqueeze(0)  # W' = W * diag(alpha)
            else:
                weight_scaled = weight

            all_weights_scaled[name] = weight_scaled
            all_layer_awq[name] = layer_awq

            # Compute R-D points on the scaled weight
            rd_table = compute_rd_points(weight_scaled, formats)
            all_rd_tables[name] = rd_table
            all_n_in[name] = weight.shape[1]

    # Global allocation across all layers
    n_elements = {name: all_n_in[name] for name, _ in to_quantize}
    total_params = sum(
        len(rd) * n_elements[name]
        for name, rd in all_rd_tables.items()
    )
    budget_total_bits = budget_avg_bits * total_params

    # Per-layer distortion weights
    layer_weights: dict[str, float] = {}
    for name, _ in to_quantize:
        layer_weights[name] = (
            layer_importance.get(name, 1.0) if layer_importance else 1.0
        )

    from rdquant.core.allocator import _pick_formats, _total_bits, _build_result, _align_groups

    def _global_bits(lam: float) -> float:
        total = 0.0
        for name, rd in all_rd_tables.items():
            assignments = _pick_formats(rd, lam, formats, layer_weights[name])
            total += _total_bits(rd, assignments)
        return total

    # Check feasibility
    min_bits = _global_bits(1e18)
    max_bits = _global_bits(0.0)

    if budget_total_bits <= min_bits:
        lambda_star = 1e18
    elif budget_total_bits >= max_bits:
        lambda_star = 0.0
    else:
        lam_lo, lam_hi = 0.0, 1.0
        for _ in range(64):
            if _global_bits(lam_hi) <= budget_total_bits:
                break
            lam_hi *= 2.0
        for _ in range(64):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _global_bits(lam_mid) > budget_total_bits:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lambda_star = (lam_lo + lam_hi) / 2.0

    # Allocate each layer at the global lambda_star and replace with Int4FusedLinear
    with torch.no_grad():
        for name, layer in to_quantize:
            rd = all_rd_tables[name]
            n_in = all_n_in[name]
            w = layer_weights[name]
            weight_scaled = all_weights_scaled[name]
            layer_awq = all_layer_awq[name]

            assignments = _pick_formats(rd, lambda_star, formats, w)
            assignments = _align_groups(assignments, rd, formats)
            result = _build_result(rd, assignments, n_in, lambda_star, formats)

            # Separate channels into INT4 and INT8 groups
            perm = result.permutation
            weight_perm = weight_scaled[perm]  # [N_out, K] in format-grouped order

            n_int4 = result.splits.get("INT4", 0)
            n_int8 = result.splits.get("INT8", 0)

            if n_int4 > 0:
                w_int4_group = weight_perm[:n_int4]  # [N_int4, K]
                w_int4, s_int4 = quantize_to_int4_groupwise(w_int4_group, int4_group_size)
            else:
                w_int4 = torch.zeros(0, n_in, dtype=torch.int8, device=weight_scaled.device)
                s_int4 = torch.zeros(0, n_in // int4_group_size, dtype=torch.float32, device=weight_scaled.device)

            if n_int8 > 0:
                w_int8_group = weight_perm[n_int4:n_int4 + n_int8]  # [N_int8, K]
                w_int8, s_int8 = quantize_to_int8_channelwise(w_int8_group)
            else:
                w_int8 = torch.zeros(0, n_in, dtype=torch.int8, device=weight_scaled.device)
                s_int8 = torch.zeros(0, dtype=torch.float32, device=weight_scaled.device)

            bias = layer.bias.data if layer.bias is not None else None

            qlayer = Int4FusedLinear(
                w_int4=w_int4,
                s_int4=s_int4,
                w_int8=w_int8,
                s_int8=s_int8,
                inv_perm=result.inv_permutation,
                bias=bias,
                group_size=int4_group_size,
                awq_scales=layer_awq,
            )
            _set_module(model, name, qlayer)

            layer_info[name] = Int4LayerInfo(
                allocation=result,
                n_int4=n_int4,
                n_int8=n_int8,
                avg_bits=result.avg_bits,
            )

    return QuantizedModelInt4(
        model=model,
        layer_info=layer_info,
        budget_avg_bits=budget_avg_bits,
    )
