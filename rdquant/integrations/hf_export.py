"""
HuggingFace-compatible save/load for QuantizedModel.

Two formats supported:
  1. Legacy: model.pt (torch pickle) for exact round-trip.
  2. Packed: safetensors with quantized weights in compact binary format.

Packed format stores:
  - NVFP4: nibble-packed uint8 indices + FP8 E4M3 scales (as uint8 view) + global_scale
  - FP8: float8_e4m3fn data (as uint8 view) + per-channel float32 scale
  - FP16: bfloat16 weights
  - Metadata: inv_permutation (int32), splits (int32 [3])
  - Optional bias

Also writes quantization_config.json with per-layer metadata.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rdquant.quantize import QuantizedModel


# ---------------------------------------------------------------------------
# Packed checkpoint save (safetensors)
# ---------------------------------------------------------------------------

def save_packed(model: "QuantizedModel", path: str, source_model_dir: str | None = None) -> None:
    """Save quantized model in packed safetensors format.

    Stores weights in their quantized representation (not dequantized BF16)
    for compact checkpoints compatible with vLLM loading.

    Args:
        model: A QuantizedModel instance.
        path: Directory to save into. Created if it doesn't exist.
        source_model_dir: Optional path to original HF model directory.
            If provided, tokenizer files and config.json are copied over.
    """
    from safetensors.torch import save_file
    from rdquant.quantize import QuantizedLayer

    os.makedirs(path, exist_ok=True)

    # Canonical format order for splits tensor
    _FMT_ORDER = ["NVFP4", "FP8", "FP16"]

    tensors: dict[str, torch.Tensor] = {}
    layer_configs: dict[str, dict] = {}

    for mod_name, module in model.model.named_modules():
        if not isinstance(module, QuantizedLayer):
            continue

        qw = module.quantized_weight
        prefix = mod_name

        n_out, n_in = qw.original_shape

        # splits: [N_nvfp4, N_fp8, N_fp16] as int32
        splits_list = [qw.splits.get(f, 0) for f in _FMT_ORDER]
        tensors[f"{prefix}.splits"] = torch.tensor(splits_list, dtype=torch.int32)

        # inv_permutation: [N_out] int32
        tensors[f"{prefix}.inv_permutation"] = qw.inv_permutation.to(torch.int32)

        # Per-format tensors
        for fmt, qt in qw.qtensors.items():
            n_ch = qw.splits[fmt]
            if n_ch == 0:
                continue

            if fmt == "NVFP4":
                # Pack 2 indices per byte: low nibble=even col, high nibble=odd col
                indices_2d = qt.data.reshape(n_ch, n_in)  # long indices 0..15
                even = indices_2d[:, 0::2].to(torch.uint8)
                odd = indices_2d[:, 1::2].to(torch.uint8)
                packed = (odd << 4) | even  # [n_ch, n_in//2] uint8
                tensors[f"{prefix}.weight_nvfp4"] = packed

                # Scales: [n_blocks] float32 -> [n_ch, n_in//16] -> cast to fp8 -> view as uint8
                blocks_per_row = n_in // 16
                scales_2d = qt.scales.reshape(n_ch, blocks_per_row)
                scales_fp8 = scales_2d.to(torch.float8_e4m3fn)
                tensors[f"{prefix}.weight_nvfp4_scale"] = scales_fp8.view(torch.uint8)

                # Global scale: [1] float32
                gs = qt.global_scale if qt.global_scale is not None else 1.0
                tensors[f"{prefix}.nvfp4_global_scale"] = torch.tensor([gs], dtype=torch.float32)

            elif fmt == "FP8":
                # Data: float32 (FP8-representable) -> cast to fp8 -> view as uint8
                data_2d = qt.data.reshape(n_ch, n_in)
                data_fp8 = data_2d.to(torch.float8_e4m3fn)
                tensors[f"{prefix}.weight_fp8"] = data_fp8.view(torch.uint8)

                # Scale: [1] float32 per-channel scale -> expand to [n_ch]
                scale_val = qt.scales.item()
                tensors[f"{prefix}.weight_fp8_scale"] = torch.full(
                    (n_ch,), scale_val, dtype=torch.float32
                )

            elif fmt == "FP16":
                # Data: float16 -> store as bfloat16
                data_2d = qt.data.reshape(n_ch, n_in)
                tensors[f"{prefix}.weight_fp16"] = data_2d.to(torch.bfloat16)

        # Bias
        if module.bias is not None:
            tensors[f"{prefix}.bias"] = module.bias.data.clone()

        # Build per-layer config
        layer_configs[mod_name] = {
            "in_features": n_in,
            "out_features": n_out,
            "splits": {f: qw.splits.get(f, 0) for f in _FMT_ORDER},
            "avg_bits": qw.avg_bits,
        }

    # Save tensors as safetensors
    save_file(tensors, os.path.join(path, "model.safetensors"))

    # Save quantization config
    config = {
        "quant_method": "rdquant",
        "formats": model.formats,
        "budget_avg_bits": model.budget_avg_bits,
        "format_order": _FMT_ORDER,
        "layers": layer_configs,
    }
    # Also include layer_info summary if available
    if hasattr(model, "layer_info"):
        info_summary = {}
        for layer_name, result in model.layer_info.items():
            info_summary[layer_name] = {
                "splits": dict(result.splits),
                "avg_bits": result.avg_bits,
                "total_distortion": result.total_distortion,
                "lambda_star": result.lambda_star,
            }
        config["layer_info"] = info_summary

    with open(os.path.join(path, "quantization_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer and config.json from source model if available
    if source_model_dir and os.path.isdir(source_model_dir):
        _copy_model_files(source_model_dir, path)


def _copy_model_files(src: str, dst: str) -> None:
    """Copy tokenizer files and config.json from source HF model dir."""
    patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
    ]
    for fname in patterns:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst, fname))


# ---------------------------------------------------------------------------
# Packed checkpoint load
# ---------------------------------------------------------------------------

def load_packed(path: str, device: str = "cpu") -> "QuantizedModel":
    """Load a packed safetensors checkpoint into a QuantizedModel.

    Reconstructs QuantizedWeight/QuantizedLayer objects from the packed
    binary representation.

    Args:
        path: Directory containing model.safetensors and quantization_config.json.
        device: Target device for tensors.

    Returns:
        QuantizedModel wrapping a simple container module with QuantizedLayers.
    """
    from safetensors import safe_open
    from rdquant.core.formats import QuantizedTensor
    from rdquant.quantize import QuantizedWeight, QuantizedLayer, QuantizedModel

    # Load config
    with open(os.path.join(path, "quantization_config.json")) as f:
        config = json.load(f)

    fmt_order = config.get("format_order", ["NVFP4", "FP8", "FP16"])
    layer_configs = config["layers"]

    # Load safetensors
    st_path = os.path.join(path, "model.safetensors")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(st_path, framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Reconstruct layers
    layers: dict[str, QuantizedLayer] = {}

    for layer_name, lconf in layer_configs.items():
        n_in = lconf["in_features"]
        n_out = lconf["out_features"]
        splits_dict = lconf["splits"]

        prefix = layer_name

        # Read inv_permutation
        inv_perm = tensors[f"{prefix}.inv_permutation"].to(torch.int64)

        # Reconstruct permutation from inv_permutation
        perm = torch.empty_like(inv_perm)
        perm[inv_perm] = torch.arange(n_out, dtype=torch.int64)

        qtensors: dict[str, QuantizedTensor] = {}

        # NVFP4
        n_nvfp4 = splits_dict.get("NVFP4", 0)
        if n_nvfp4 > 0:
            packed = tensors[f"{prefix}.weight_nvfp4"]  # [n_ch, n_in//2] uint8
            # Unpack: low nibble = even, high nibble = odd
            even = (packed & 0x0F).to(torch.int64)
            odd = ((packed >> 4) & 0x0F).to(torch.int64)
            # Interleave back: [n_ch, n_in]
            n_ch = packed.shape[0]
            half_k = packed.shape[1]
            indices_2d = torch.empty(n_ch, half_k * 2, dtype=torch.int64, device=packed.device)
            indices_2d[:, 0::2] = even
            indices_2d[:, 1::2] = odd
            indices_flat = indices_2d.reshape(-1)  # [n_ch * n_in]

            # Scales: uint8 view -> fp8 -> float32
            scales_uint8 = tensors[f"{prefix}.weight_nvfp4_scale"]  # [n_ch, blocks_per_row] as uint8
            scales_fp8 = scales_uint8.view(torch.float8_e4m3fn)
            scales_flat = scales_fp8.to(torch.float32).reshape(-1)  # [n_ch * blocks_per_row]

            global_scale = tensors[f"{prefix}.nvfp4_global_scale"].item()

            qt_nvfp4 = QuantizedTensor(
                data=indices_flat,
                scales=scales_flat,
                format_name="NVFP4",
                original_shape=torch.Size([n_nvfp4, n_in]),
                bits_per_element=4,
                global_scale=global_scale,
            )
            qtensors["NVFP4"] = qt_nvfp4

        # FP8
        n_fp8 = splits_dict.get("FP8", 0)
        if n_fp8 > 0:
            data_uint8 = tensors[f"{prefix}.weight_fp8"]  # [n_ch, n_in] as uint8
            data_fp8 = data_uint8.view(torch.float8_e4m3fn)
            data_f32 = data_fp8.to(torch.float32).reshape(-1)  # [n_ch * n_in]

            scale_vec = tensors[f"{prefix}.weight_fp8_scale"]  # [n_ch] float32
            # FP8 format uses a single per-channel scale; store as [1] with the first value
            # (all rows share the same scale in current implementation)
            scale_val = scale_vec[0:1]

            qt_fp8 = QuantizedTensor(
                data=data_f32,
                scales=scale_val,
                format_name="FP8",
                original_shape=torch.Size([n_fp8, n_in]),
                bits_per_element=8,
            )
            qtensors["FP8"] = qt_fp8

        # FP16
        n_fp16 = splits_dict.get("FP16", 0)
        if n_fp16 > 0:
            data_bf16 = tensors[f"{prefix}.weight_fp16"]  # [n_ch, n_in] bfloat16
            data_f16 = data_bf16.to(torch.float16).reshape(-1)  # [n_ch * n_in]

            qt_fp16 = QuantizedTensor(
                data=data_f16,
                scales=torch.tensor([], dtype=torch.float32),
                format_name="FP16",
                original_shape=torch.Size([n_fp16, n_in]),
                bits_per_element=16,
            )
            qtensors["FP16"] = qt_fp16

        splits_ordered = {f: splits_dict.get(f, 0) for f in fmt_order}

        qw = QuantizedWeight(
            qtensors=qtensors,
            permutation=perm,
            inv_permutation=inv_perm,
            splits=splits_ordered,
            original_shape=torch.Size([n_out, n_in]),
            avg_bits=lconf["avg_bits"],
        )

        # Bias
        bias_key = f"{prefix}.bias"
        bias = tensors.get(bias_key, None)

        ql = QuantizedLayer(
            quantized_weight=qw,
            bias=bias,
            in_features=n_in,
            out_features=n_out,
        )
        layers[layer_name] = ql

    # Build a simple container model
    container = _LayerContainer(layers)

    # Reconstruct layer_info stub for QuantizedModel
    from rdquant.core.allocator import AllocationResult
    layer_info: dict[str, AllocationResult] = {}
    if "layer_info" in config:
        for lname, info in config["layer_info"].items():
            # Reconstruct a minimal AllocationResult for bookkeeping
            splits = info["splits"]
            n_out = sum(splits.values())
            assignments = {}
            idx = 0
            for fmt in fmt_order:
                n_ch = splits.get(fmt, 0)
                for i in range(n_ch):
                    assignments[idx] = fmt
                    idx += 1
            # Use dummy permutation/inv_permutation (the real ones are in the layers)
            layer_info[lname] = AllocationResult(
                assignments=assignments,
                permutation=torch.arange(n_out),
                inv_permutation=torch.arange(n_out),
                splits=splits,
                avg_bits=info["avg_bits"],
                total_distortion=info.get("total_distortion", 0.0),
                lambda_star=info.get("lambda_star", 0.0),
                format_stats={},
            )

    return QuantizedModel(
        model=container,
        layer_info=layer_info,
        formats=config.get("formats", fmt_order),
        budget_avg_bits=config.get("budget_avg_bits", 0.0),
    )


class _LayerContainer(torch.nn.Module):
    """Simple container that holds QuantizedLayers as named submodules."""

    def __init__(self, layers: dict[str, torch.nn.Module]):
        super().__init__()
        for name, layer in layers.items():
            # Convert dots to a nested module structure
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    setattr(parent, part, torch.nn.Module())
                parent = getattr(parent, part)
            setattr(parent, parts[-1], layer)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This container is for weight storage only. "
            "Use the individual QuantizedLayers for inference."
        )


# ---------------------------------------------------------------------------
# Legacy save/load (torch pickle)
# ---------------------------------------------------------------------------

def save_quantized(model: "QuantizedModel", path: str) -> None:
    """Save quantized model to a directory.

    Writes both legacy (model.pt) and packed (safetensors) formats.

    Args:
        model: A QuantizedModel instance.
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
                # Use lowercase format names: nvfp4, fp8, fp16
                fp = prefix + f"qtensor.{fmt.lower()}."
                state[fp + "data"] = qt.data
                state[fp + "scales"] = qt.scales
                state[fp + "original_shape"] = torch.tensor(list(qt.original_shape))
            if module.bias is not None:
                state[prefix + "bias"] = module.bias.data

    torch.save(state, os.path.join(path, "weights.pt"))

    # Also save packed safetensors format
    try:
        save_packed(model, path)
    except ImportError:
        pass  # safetensors not installed, skip packed format


def load_quantized(path: str, original_model=None) -> "QuantizedModel":
    """Load a quantized model from a directory saved by save_quantized.

    Tries legacy model.pt first, then falls back to packed safetensors.

    Args:
        path: Directory path previously saved by save_quantized.
        original_model: Unused (kept for API compatibility).

    Returns:
        Loaded QuantizedModel.
    """
    model_pt = os.path.join(path, "model.pt")
    if os.path.isfile(model_pt):
        return torch.load(model_pt, map_location="cpu", weights_only=False)

    # Try packed safetensors
    st_path = os.path.join(path, "model.safetensors")
    if os.path.isfile(st_path):
        return load_packed(path)

    raise FileNotFoundError(
        f"Cannot find model.pt or model.safetensors in {path}. "
        "Please save the model using save_quantized() or save_packed()."
    )
