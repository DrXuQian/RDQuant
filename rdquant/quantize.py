"""
End-to-end mixed-precision quantization for HuggingFace (and generic nn.Module) models.

Implements quantize_model(), QuantizedWeight, QuantizedLayer, and QuantizedModel.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdquant.core.formats import quantize, dequantize, get_bits_per_element, QuantizedTensor
from rdquant.core.allocator import allocate_layer, allocate, AllocationResult
from rdquant.core.sensitivity import compute_rd_points


# ---------------------------------------------------------------------------
# QuantizedWeight: stores per-format quantized tensors + permutation info
# ---------------------------------------------------------------------------

@dataclass
class QuantizedWeight:
    """Packed quantized weight data for a single linear layer.

    Attributes:
        qtensors: Dict mapping format_name -> QuantizedTensor (for the group of
            channels assigned to that format).
        permutation: [N_out] index tensor that reorders channels by format.
        inv_permutation: [N_out] inverse permutation to restore original order.
        splits: Dict format_name -> number of channels for that format.
        original_shape: (N_out, N_in) shape of the original weight.
        avg_bits: Average bits per element achieved.
    """
    qtensors: dict[str, QuantizedTensor]
    permutation: torch.Tensor
    inv_permutation: torch.Tensor
    splits: dict[str, int]
    original_shape: torch.Size
    avg_bits: float

    def dequantize(self) -> torch.Tensor:
        """Reconstruct the full float32 weight tensor in original channel order.

        Returns:
            Float32 tensor of shape original_shape.
        """
        n_out, n_in = self.original_shape
        pieces = []
        for fmt, qtensor in self.qtensors.items():
            n_ch = self.splits[fmt]
            if n_ch == 0:
                continue
            # dequantize gives shape [n_ch * n_in] (1-D stored)
            # We stored the group as a 2-D original_shape [n_ch, n_in]
            deq = dequantize(qtensor)  # [n_ch, n_in]
            pieces.append(deq)

        # Concatenate in permuted order then apply inv_permutation
        reordered = torch.cat(pieces, dim=0)  # [N_out, N_in] in permuted order
        # Apply inverse permutation to restore original channel order
        inv_perm = self.inv_permutation
        weight = reordered[inv_perm]
        return weight.float()


# ---------------------------------------------------------------------------
# QuantizedLayer: replaces nn.Linear
# ---------------------------------------------------------------------------

class QuantizedLayer(nn.Module):
    """Fake-quantization replacement for nn.Linear.

    Stores a QuantizedWeight and reconstructs the full weight on each forward
    pass using dequantize + inv_permutation before running the standard linear.

    Args:
        quantized_weight: The quantized weight data.
        bias: Optional bias tensor (copied from original layer if present).
        in_features: Number of input features.
        out_features: Number of output features.
    """

    def __init__(
        self,
        quantized_weight: QuantizedWeight,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.quantized_weight = quantized_weight
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixed-precision forward: per-format dequant → GEMM → concat → permute.

        Uses grouped GEMMs (one per format) so each can later be swapped for
        a native Tensor Core kernel.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        from rdquant.ops import mixed_precision_linear

        qw = self.quantized_weight
        return mixed_precision_linear(
            x, qw.qtensors, qw.splits, qw.inv_permutation, self.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"avg_bits={self.quantized_weight.avg_bits:.2f}"
        )


# ---------------------------------------------------------------------------
# QuantizedModel
# ---------------------------------------------------------------------------

class QuantizedModel(nn.Module):
    """Wrapper around a model with some nn.Linear layers replaced by QuantizedLayer.

    Args:
        model: The original (now modified in-place) model.
        layer_info: Dict mapping layer_name -> AllocationResult for bookkeeping.
        formats: Format list used during quantization.
        budget_avg_bits: The target budget used.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_info: dict[str, AllocationResult],
        formats: list[str],
        budget_avg_bits: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.layer_info = layer_info
        self.formats = formats
        self.budget_avg_bits = budget_avg_bits

    def forward(self, *args, **kwargs):
        """Pass-through forward to the underlying model.

        Args:
            *args: Positional arguments forwarded to model.forward().
            **kwargs: Keyword arguments forwarded to model.forward().

        Returns:
            Model output (same as original model.forward()).
        """
        return self.model(*args, **kwargs)

    def print_summary(self) -> None:
        """Print a per-layer format allocation table to stdout."""
        header = f"{'Layer':<40} {'N_out':>6} {'N_in':>6} {'avg_bits':>9} " + \
                 "  ".join(f"{f:>8}" for f in self.formats)
        print(header)
        print("-" * len(header))
        total_bits = 0.0
        total_params = 0
        for name, result in self.layer_info.items():
            n_out = len(result.assignments)
            # Infer n_in from splits and total_bits via avg_bits
            # avg_bits = total_bits / (n_out * n_in)
            # We can get n_in from the first rd entry cost / rate
            # But we don't store rd_table here. Use format_stats.
            total_ch_bits = sum(
                stats["total_bits"]
                for stats in result.format_stats.values()
            )
            n_in = int(total_ch_bits / (n_out * result.avg_bits)) if (n_out * result.avg_bits) > 0 else 0
            counts = "  ".join(
                f"{result.splits.get(f, 0):>8}" for f in self.formats
            )
            print(f"{name:<40} {n_out:>6} {n_in:>6} {result.avg_bits:>9.2f}  {counts}")
            total_bits += total_ch_bits
            total_params += n_out * n_in
        if total_params > 0:
            overall_bits = total_bits / total_params
        else:
            overall_bits = 0.0
        print("-" * len(header))
        print(f"{'TOTAL':<40} {'':>6} {'':>6} {overall_bits:>9.2f}")

    def save_pretrained(self, path: str) -> None:
        """Save the quantized model to a directory.

        Args:
            path: Directory path to save into. Created if it doesn't exist.
        """
        from rdquant.integrations.hf_export import save_quantized
        save_quantized(self, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "QuantizedModel":
        """Load a quantized model previously saved with save_pretrained.

        Args:
            path: Directory path previously saved by save_pretrained().

        Returns:
            Loaded QuantizedModel.
        """
        from rdquant.integrations.hf_export import load_quantized
        return load_quantized(path)


# ---------------------------------------------------------------------------
# Helper: quantize a single weight tensor into QuantizedWeight
# ---------------------------------------------------------------------------

def _quantize_weight(
    weight: torch.Tensor,
    result: AllocationResult,
) -> QuantizedWeight:
    """Pack a 2-D weight [N_out, N_in] into a QuantizedWeight using AllocationResult.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        result: AllocationResult from allocate_layer().

    Returns:
        QuantizedWeight with per-format quantized tensors.
    """
    n_out, n_in = weight.shape
    perm = result.permutation  # [N_out]

    # Reorder channels by format group
    weight_perm = weight[perm]  # [N_out, N_in] in format-grouped order

    qtensors: dict[str, QuantizedTensor] = {}
    offset = 0
    for fmt in result.splits:
        n_ch = result.splits[fmt]
        if n_ch == 0:
            continue
        group = weight_perm[offset: offset + n_ch]  # [n_ch, n_in]
        # Flatten to 1-D for quantize, store original_shape as [n_ch, n_in]
        flat = group.flatten().float()
        qt = quantize(flat, fmt)
        # Override original_shape to [n_ch, n_in] so dequantize restores 2-D
        from dataclasses import replace as dc_replace
        qt = QuantizedTensor(
            data=qt.data,
            scales=qt.scales,
            format_name=qt.format_name,
            original_shape=torch.Size([n_ch, n_in]),
        )
        qtensors[fmt] = qt
        offset += n_ch

    return QuantizedWeight(
        qtensors=qtensors,
        permutation=perm.clone(),
        inv_permutation=result.inv_permutation.clone(),
        splits=dict(result.splits),
        original_shape=weight.shape,
        avg_bits=result.avg_bits,
    )


# ---------------------------------------------------------------------------
# quantize_model
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


def quantize_model(
    model: nn.Module,
    budget_avg_bits: float = 5.3,
    formats: list[str] = None,
    sensitivity_metric: str = "mse",
    ignore: list[str] = None,
    per_layer_budget: bool = False,
) -> QuantizedModel:
    """Quantize all nn.Linear layers of a model using R-D optimal allocation.

    Args:
        model: Any nn.Module (HuggingFace PreTrainedModel or custom).
        budget_avg_bits: Target average bits per element across all (or per) layers.
        formats: Allowed format names. Defaults to ["NVFP4","MXFP6","MXFP8","FP16"].
        sensitivity_metric: Sensitivity metric passed to allocate_layer().
        ignore: List of layer name patterns (fnmatch-style) to skip quantization.
        per_layer_budget: If True, apply budget independently to each layer.
            If False (default), do a global lambda sweep across all layers for
            better cross-layer bit allocation.

    Returns:
        QuantizedModel wrapping the modified model.
    """
    if formats is None:
        formats = ["NVFP4", "MXFP6", "MXFP8", "FP16"]

    if ignore is None:
        ignore = []

    linear_layers = _get_named_linears(model)

    # Filter ignored layers
    to_quantize = [
        (name, layer)
        for name, layer in linear_layers
        if not _should_ignore(name, ignore)
    ]

    layer_info: dict[str, AllocationResult] = {}

    if per_layer_budget or len(to_quantize) == 0:
        # Simple per-layer budget allocation
        for name, layer in to_quantize:
            weight = layer.weight.data.float()
            with torch.inference_mode():
                result = allocate_layer(weight, budget_avg_bits, formats, sensitivity_metric)
            layer_info[name] = result
    else:
        # Global lambda sweep: collect all R-D tables, then do single global allocation
        all_rd_tables: dict[str, dict] = {}
        all_n_in: dict[str, int] = {}

        with torch.inference_mode():
            for name, layer in to_quantize:
                weight = layer.weight.data.float()
                rd_table = compute_rd_points(weight, formats)
                all_rd_tables[name] = rd_table
                all_n_in[name] = weight.shape[1]

        # Combine all rd_tables with globally unique channel keys and run one allocate
        # Key structure: (layer_name, channel_idx)
        # We need to flatten into a single rd_table with integer keys for the allocator.
        # The allocator assumes all channels have the same n_elements_per_channel.
        # For global allocation with different n_in per layer, we do it per-layer
        # but share a single lambda_star.

        # Find a common lambda by doing a global bit budget calculation.
        # We sweep lambda and compute total bits vs budget across all layers.
        n_elements = {name: all_n_in[name] for name, _ in to_quantize}

        # Compute total params across all layers to quantize
        total_params = sum(
            len(rd) * n_elements[name]
            for name, rd in all_rd_tables.items()
        )
        budget_total_bits = budget_avg_bits * total_params

        # Binary search for global lambda
        from rdquant.core.allocator import _pick_formats, _total_bits

        def _global_bits(lam: float) -> float:
            total = 0.0
            for name, rd in all_rd_tables.items():
                assignments = _pick_formats(rd, lam, formats)
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

        # Now allocate each layer at the global lambda_star
        from rdquant.core.allocator import _build_result, _pick_formats
        for name, layer in to_quantize:
            rd = all_rd_tables[name]
            n_in = all_n_in[name]
            assignments = _pick_formats(rd, lambda_star, formats)
            result = _build_result(rd, assignments, n_in, lambda_star, formats)
            layer_info[name] = result

    # Replace nn.Linear layers with QuantizedLayer
    with torch.inference_mode():
        for name, layer in to_quantize:
            result = layer_info[name]
            weight = layer.weight.data.float()
            qw = _quantize_weight(weight, result)
            bias = layer.bias.data if layer.bias is not None else None
            qlayer = QuantizedLayer(
                quantized_weight=qw,
                bias=bias,
                in_features=layer.in_features,
                out_features=layer.out_features,
            )
            _set_module(model, name, qlayer)

    return QuantizedModel(
        model=model,
        layer_info=layer_info,
        formats=formats,
        budget_avg_bits=budget_avg_bits,
    )
