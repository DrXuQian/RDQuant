"""
Numeric format definitions and quantize/dequantize implementations.

Supported formats (NVFP4/FP8/FP16 hierarchy matching vLLM kernel interfaces):
  - NVFP4: E2M1 with per-16-element FP8 E4M3 block scale + per-tensor FP32 global scale
  - FP8:   E4M3 with per-channel FP32 scale
  - FP16:  Passthrough (no quantization)
  - INT4:  Symmetric per-group (group_size=128) quantization, range [-8, 7]
  - INT8:  Symmetric per-channel quantization, range [-128, 127]

Each format exposes:
  - bits_per_element: int -- the nominal bit-width (4, 8, 16)
  - quantize(tensor) -> QuantizedTensor  -- returns packed data + metadata
  - dequantize(qtensor) -> tensor -- reconstruct full-precision approximation
  - compute_mse(tensor) -> float -- quantize, dequantize, compute MSE (convenience)

All quantize functions are fully vectorized (no Python element loops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QuantizedTensor:
    """Container for quantized data and associated metadata."""
    data: torch.Tensor        # packed weight data
    scales: torch.Tensor      # block scales (NVFP4) or channel scales (FP8)
    format_name: str          # "NVFP4", "FP8", "FP16"
    original_shape: torch.Size
    bits_per_element: int     # 4, 8, 16
    global_scale: Optional[float] = None  # only for NVFP4


# ---------------------------------------------------------------------------
# NVFP4 block size
# ---------------------------------------------------------------------------

_NVFP4_BLOCK_SIZE = 16

# ---------------------------------------------------------------------------
# FP8 E4M3 helper: quantize a float tensor to FP8 E4M3 representable values
# ---------------------------------------------------------------------------

_FP8_E4M3_MAX = 448.0


def _quantize_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize float values to the nearest FP8 E4M3 representable value.

    Clamps to [-448, 448], casts to float8_e4m3fn and back to float32
    so the result is exactly FP8-representable.

    Args:
        x: Float tensor of any shape.

    Returns:
        Float32 tensor with values that are exactly representable in FP8 E4M3.
    """
    clamped = x.float().clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    fp8 = clamped.to(torch.float8_e4m3fn)
    return fp8.to(torch.float32)


# ---------------------------------------------------------------------------
# NVFP4 (E2M1) -- per-16-element FP8 E4M3 block scale, FP32 global scale
# ---------------------------------------------------------------------------

# Magnitude grid for E2M1: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# Full signed LUT: 16 entries (8 positive including 0, 8 negative including -0)
_NVFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_NVFP4_POS_VALUES = _NVFP4_LUT[:8]
_NVFP4_MAX_VAL = 6.0


def nvfp4_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to NVFP4 E2M1 with per-16-element FP8 E4M3 block scale.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=indices (long), scales=FP8 block scales (float32),
        global_scale=max of all block scales (float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    blocks = flat.view(-1, _NVFP4_BLOCK_SIZE)  # [n_blocks, 16]

    # Per-block absmax -> raw scale
    absmax = blocks.abs().amax(dim=1)  # [n_blocks]
    raw_scale = absmax / _NVFP4_MAX_VAL  # 6.0 is max FP4 E2M1 value

    # Quantize scale to FP8 E4M3
    scale_fp8 = _quantize_to_fp8_e4m3(raw_scale)  # [n_blocks] float32 but FP8-representable

    # Global scale = max of all block scales
    global_scale = scale_fp8.max().item()
    if global_scale == 0:
        global_scale = 1.0

    # Safe scale for division (avoid div-by-zero)
    safe_scale = scale_fp8.clone()
    safe_scale[safe_scale == 0] = 1.0

    # Normalize and find nearest LUT entry
    normalized = blocks / safe_scale.unsqueeze(1)  # [n_blocks, 16]

    # Nearest LUT entry: separate sign and magnitude
    norm_flat = normalized.reshape(-1)
    sign = (norm_flat < 0).long()
    abs_norm = norm_flat.abs()

    # Distances to each positive LUT value -- [N, 8]
    pos_vals = _NVFP4_POS_VALUES.to(flat.device)
    dists = (abs_norm.unsqueeze(1) - pos_vals).abs()
    mag_idx = dists.argmin(dim=1)  # [N]

    indices = (mag_idx + sign * 8)[:n]

    return QuantizedTensor(
        data=indices,
        scales=scale_fp8,
        format_name="NVFP4",
        original_shape=original_shape,
        bits_per_element=4,
        global_scale=global_scale,
    )


def nvfp4_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize NVFP4 QuantizedTensor back to float32."""
    codes = qtensor.data
    block_scales = qtensor.scales
    n = codes.shape[0]

    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype, device=codes.device)])

    n_blocks = codes.shape[0] // _NVFP4_BLOCK_SIZE
    lut = _NVFP4_LUT.to(codes.device)
    values = lut[codes].view(n_blocks, _NVFP4_BLOCK_SIZE)
    values = values * block_scales.to(values.device).unsqueeze(1)
    return values.view(-1)[:n].reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# FP8 (E4M3) -- per-channel FP32 scale
# ---------------------------------------------------------------------------

def fp8_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to FP8 E4M3 with per-channel (single) scale.

    For a single channel (1D tensor), scale = absmax / 448.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=FP8 codes (float32 but FP8-representable),
        scales=per-channel scale [1] (float32).
    """
    original_shape = tensor.shape
    t = tensor.float()
    amax = t.abs().max().item()
    scale = amax / _FP8_E4M3_MAX
    if scale == 0:
        scale = 1.0

    fp8_data = _quantize_to_fp8_e4m3(t / scale)

    return QuantizedTensor(
        data=fp8_data,
        scales=torch.tensor([scale], dtype=torch.float32, device=tensor.device),
        format_name="FP8",
        original_shape=original_shape,
        bits_per_element=8,
    )


def fp8_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize FP8 QuantizedTensor back to float32."""
    scale = qtensor.scales.item()
    return (qtensor.data.float() * scale).reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# FP16 -- passthrough (no quantization)
# ---------------------------------------------------------------------------

def fp16_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """'Quantize' a tensor to FP16 (lossless passthrough via float16 cast).

    Args:
        tensor: Float tensor.

    Returns:
        QuantizedTensor with data=float16 tensor, scales=empty, format="FP16".
    """
    original_shape = tensor.shape
    data = tensor.float().to(torch.float16)
    return QuantizedTensor(
        data=data,
        scales=torch.tensor([], dtype=torch.float32, device=tensor.device),
        format_name="FP16",
        original_shape=original_shape,
        bits_per_element=16,
    )


def fp16_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize FP16 QuantizedTensor back to float32."""
    return qtensor.data.float().reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# INT4 (symmetric per-group, group_size=128, range [-8, 7])
# ---------------------------------------------------------------------------

_INT4_GROUP_SIZE = 128


def int4_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to INT4 symmetric with per-group scale (group_size=128).

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=int8 (values -8..7), scales=per-group float32.
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]
    group_size = _INT4_GROUP_SIZE

    pad = (-n) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    groups = flat.view(-1, group_size)  # [n_groups, group_size]
    absmax = groups.abs().amax(dim=1)   # [n_groups]
    scale = absmax / 7.0
    scale[scale == 0] = 1.0

    quantized = (groups / scale.unsqueeze(1)).round().clamp(-8, 7)
    data = quantized.flatten()[:n].to(torch.int8)

    return QuantizedTensor(
        data=data,
        scales=scale,
        format_name="INT4",
        original_shape=original_shape,
        bits_per_element=4,
    )


def int4_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize INT4 QuantizedTensor back to float32."""
    data = qtensor.data.float()
    scales = qtensor.scales
    n = data.shape[0]
    group_size = _INT4_GROUP_SIZE

    pad = (-n) % group_size
    if pad > 0:
        data = torch.cat([data, torch.zeros(pad, dtype=data.dtype, device=data.device)])

    groups = data.view(-1, group_size)
    result = groups * scales.to(groups.device).unsqueeze(1)
    return result.flatten()[:n].reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# INT8 (symmetric per-channel, range [-128, 127])
# ---------------------------------------------------------------------------

def int8_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to INT8 symmetric with per-channel (single) scale.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=int8 (values -128..127), scales=[1] float32.
    """
    original_shape = tensor.shape
    t = tensor.float()
    amax = t.abs().max().item()
    scale = amax / 127.0
    if scale == 0:
        scale = 1.0

    quantized = (t / scale).round().clamp(-128, 127).to(torch.int8)

    return QuantizedTensor(
        data=quantized,
        scales=torch.tensor([scale], dtype=torch.float32, device=tensor.device),
        format_name="INT8",
        original_shape=original_shape,
        bits_per_element=8,
    )


def int8_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize INT8 QuantizedTensor back to float32."""
    scale = qtensor.scales.item()
    return (qtensor.data.float() * scale).reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# Unified format registry
# ---------------------------------------------------------------------------

_FORMATS: dict[str, dict] = {
    "NVFP4": {
        "bits_per_element": 4,
        "quantize": nvfp4_quantize,
        "dequantize": nvfp4_dequantize,
    },
    "FP8": {
        "bits_per_element": 8,
        "quantize": fp8_quantize,
        "dequantize": fp8_dequantize,
    },
    "FP16": {
        "bits_per_element": 16,
        "quantize": fp16_quantize,
        "dequantize": fp16_dequantize,
    },
    "INT4": {
        "bits_per_element": 4,
        "quantize": int4_quantize,
        "dequantize": int4_dequantize,
    },
    "INT8": {
        "bits_per_element": 8,
        "quantize": int8_quantize,
        "dequantize": int8_dequantize,
    },
}


def get_bits_per_element(format_name: str) -> int:
    """Return nominal bits-per-element for the given format name.

    Args:
        format_name: One of "NVFP4", "FP8", "FP16".

    Returns:
        Integer bit-width.
    """
    return _FORMATS[format_name]["bits_per_element"]


def quantize(tensor: torch.Tensor, format_name: str) -> QuantizedTensor:
    """Quantize a 1D tensor using the specified format.

    Args:
        tensor: 1D float tensor (single output channel's weights).
        format_name: One of "NVFP4", "FP8", "FP16".

    Returns:
        QuantizedTensor containing packed data and metadata.
    """
    with torch.no_grad():
        return _FORMATS[format_name]["quantize"](tensor)


def dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize a QuantizedTensor back to float32.

    Args:
        qtensor: QuantizedTensor produced by quantize().

    Returns:
        Float32 tensor of original shape.
    """
    with torch.no_grad():
        return _FORMATS[qtensor.format_name]["dequantize"](qtensor)


def compute_mse(tensor: torch.Tensor, format_name: str) -> float:
    """Quantize then dequantize a tensor and compute MSE.

    Args:
        tensor: 1D float tensor.
        format_name: One of "NVFP4", "FP8", "FP16".

    Returns:
        Mean squared error as a Python float.
    """
    with torch.no_grad():
        qtensor = quantize(tensor, format_name)
        reconstructed = dequantize(qtensor)
        return ((tensor.float() - reconstructed.float()) ** 2).mean().item()


def compute_mse_2d(weight: torch.Tensor, format_name: str) -> torch.Tensor:
    """Batch quantize/dequantize a 2-D weight matrix and return per-row MSE.

    Flattens the entire matrix, quantizes once as a single 1-D tensor, then
    reshapes and computes per-channel (per-row) MSE.  Much faster than calling
    compute_mse() in a loop over rows.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        format_name: One of "NVFP4", "FP8", "FP16".

    Returns:
        Float32 tensor of shape [N_out] with per-row MSE values.
    """
    with torch.no_grad():
        n_out, n_in = weight.shape
        flat = weight.reshape(-1)
        qtensor = quantize(flat, format_name)
        recon = dequantize(qtensor).reshape(n_out, n_in)
        return ((weight.float() - recon.float()) ** 2).mean(dim=1)
