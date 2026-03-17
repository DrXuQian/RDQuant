"""
HuggingFace-compatible save/load for QuantizedModel.

Saves:
  - quantization_config.json: metadata (formats, budget, layer info)
  - weights.pt: full serialised QuantizedModel (torch .pt format)
    Individual weight tensors are also accessible under the same archive.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rdquant.quantize import QuantizedModel


def save_quantized(model: "QuantizedModel", path: str) -> None:
    """Save quantized model to a directory.

    Writes:
      - ``quantization_config.json``: format list, budget, layer metadata.
      - ``model.pt``: complete serialised :class:`~rdquant.quantize.QuantizedModel`.
      - ``weights.pt``: individual quantized tensors keyed by layer/format for
        inspection or custom loading.

    Args:
        model: A :class:`~rdquant.quantize.QuantizedModel` instance.
        path: Directory path to save into. Created if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)

    # Build config
    layer_configs: dict[str, dict] = {}
    for layer_name, result in model.layer_info.items():
        layer_configs[layer_name] = {
            "splits": result.splits,
            "avg_bits": result.avg_bits,
            "total_distortion": result.total_distortion,
            "lambda_star": result.lambda_star,
        }

    config = {
        "formats": model.formats,
        "budget_avg_bits": model.budget_avg_bits,
        "layers": layer_configs,
    }
    with open(os.path.join(path, "quantization_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save the full model for exact round-trip loading
    torch.save(model, os.path.join(path, "model.pt"))

    # Also save individual weight tensors for inspection
    from rdquant.quantize import QuantizedLayer

    state: dict[str, object] = {}
    for mod_name, module in model.model.named_modules():
        if isinstance(module, QuantizedLayer):
            qw = module.quantized_weight
            prefix = f"qlayer.{mod_name}."
            state[prefix + "permutation"] = qw.permutation
            state[prefix + "inv_permutation"] = qw.inv_permutation
            state[prefix + "original_shape"] = torch.tensor(list(qw.original_shape))
            state[prefix + "avg_bits"] = torch.tensor(qw.avg_bits)
            for fmt, qt in qw.qtensors.items():
                fp = prefix + f"qtensor.{fmt}."
                state[fp + "data"] = qt.data
                state[fp + "scales"] = qt.scales
                state[fp + "original_shape"] = torch.tensor(list(qt.original_shape))
            if module.bias is not None:
                state[prefix + "bias"] = module.bias.data

    torch.save(state, os.path.join(path, "weights.pt"))


def load_quantized(path: str, original_model=None) -> "QuantizedModel":
    """Load a quantized model from a directory saved by :func:`save_quantized`.

    Args:
        path: Directory path previously saved by :func:`save_quantized`.
        original_model: Unused (kept for API compatibility).

    Returns:
        Loaded :class:`~rdquant.quantize.QuantizedModel`.
    """
    model_pt = os.path.join(path, "model.pt")
    if os.path.isfile(model_pt):
        return torch.load(model_pt, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"Cannot find model.pt in {path}. "
        "Please save the model using save_quantized() (or QuantizedModel.save_pretrained())."
    )
