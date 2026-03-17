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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QuantizedTensor:
    """Container for quantized data and associated metadata."""
    data: torch.Tensor        # packed/quantized values (indices or float approximations)
    scales: torch.Tensor      # block scales
    format_name: str          # "NVFP4", "MXFP6", "MXFP8", "FP16"
    original_shape: torch.Size


# ---------------------------------------------------------------------------
# FP8 E4M3 helpers (used for NVFP4 block scales)
# ---------------------------------------------------------------------------

# FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
# Max normal value: 448.0  (bias=7, max_exp=14, max_mantissa=1.875)
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN_NORMAL = 2.0 ** (-9)  # min normal: exp=1, mantissa=1.0


def _quantize_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize float tensor to FP8 E4M3 representable values (returned as float32)."""
    sign = torch.sign(x)
    abs_x = x.abs().clamp(min=0.0)

    # Clamp to FP8 E4M3 max
    abs_x = abs_x.clamp(max=_FP8_E4M3_MAX)

    # For zeros and subnormals below min normal, treat as zero
    # For normal range: round to nearest FP8 E4M3 value
    # E4M3: bias=7, exponents 0..14 (15 is NaN), mantissa 3 bits
    result = torch.zeros_like(abs_x)
    nonzero = abs_x >= _FP8_E4M3_MIN_NORMAL / 2.0  # include subnormal rounding

    if nonzero.any():
        v = abs_x[nonzero]
        # Compute floor log2
        exp_f = torch.floor(torch.log2(v.clamp(min=1e-38)))
        exp_biased = exp_f + 7  # bias=7

        # Handle normal range (exp_biased in [1, 14])
        # Clamp to valid range
        exp_biased = exp_biased.clamp(1, 14)
        exp_f_clamped = exp_biased - 7

        # Compute mantissa: v / 2^exp * 8 rounded to integer (3 mantissa bits)
        scale = 2.0 ** exp_f_clamped
        mantissa_f = (v / scale - 1.0) * 8.0  # 0..7
        mantissa_i = mantissa_f.round().clamp(0, 7).long()

        reconstructed = (1.0 + mantissa_i.float() / 8.0) * scale
        result[nonzero] = reconstructed

    return sign * result


# ---------------------------------------------------------------------------
# NVFP4 (E2M1) format
# ---------------------------------------------------------------------------

# E2M1 lookup table: positive values {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# Code 0b0000 = 0, 0b0001 = 0.5, ... (see OCP FP4 spec)
_NVFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_NVFP4_POS_VALUES = _NVFP4_LUT[:8]   # positive codes 0-7
_NVFP4_MAX_VAL = 6.0
_NVFP4_BLOCK_SIZE = 16


def _nvfp4_quantize_block(block: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a 16-element block to NVFP4 indices and return (indices, scale).

    Args:
        block: 1D tensor of length 16.

    Returns:
        indices: int tensor of shape [16], values in 0..15
        scale: float, the FP8 E4M3 block scale
    """
    absmax = block.abs().max().item()
    if absmax == 0.0:
        return torch.zeros(len(block), dtype=torch.long), 0.0

    # Scale so that max absolute value maps to _NVFP4_MAX_VAL
    scale = absmax / _NVFP4_MAX_VAL

    # Quantize scale to FP8 E4M3
    scale_fp8 = _quantize_to_fp8_e4m3(torch.tensor([scale])).item()
    if scale_fp8 == 0.0:
        scale_fp8 = scale  # fallback (shouldn't happen with normal data)

    # Normalize block
    normalized = block / scale_fp8  # values in [-6, 6]

    # For each element, find nearest LUT entry
    # Positive and negative halves share the same magnitude table
    sign = (normalized < 0).long()  # 0=positive, 1=negative
    abs_norm = normalized.abs()

    # Distances to each positive LUT value
    dists = (abs_norm.unsqueeze(1) - _NVFP4_POS_VALUES.to(block.device)).abs()
    mag_idx = dists.argmin(dim=1)  # 0..7

    # Encode: sign bit in bit 3 (negative -> add 8)
    indices = mag_idx + sign * 8

    return indices, scale_fp8


def nvfp4_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to NVFP4 format.

    Args:
        tensor: 1D float tensor (single output channel weights).

    Returns:
        QuantizedTensor with data=indices (long), scales=block FP8 scales (float32).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    # Pad to multiple of block size
    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    n_padded = flat.shape[0]
    n_blocks = n_padded // _NVFP4_BLOCK_SIZE

    blocks = flat.view(n_blocks, _NVFP4_BLOCK_SIZE)
    indices_list = []
    scales_list = []

    for i in range(n_blocks):
        idx, sc = _nvfp4_quantize_block(blocks[i])
        indices_list.append(idx)
        scales_list.append(sc)

    indices = torch.cat(indices_list)[:n]  # trim padding
    scales = torch.tensor(scales_list, dtype=torch.float32)

    return QuantizedTensor(
        data=indices,
        scales=scales,
        format_name="NVFP4",
        original_shape=original_shape,
    )


def nvfp4_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize NVFP4 QuantizedTensor back to float32.

    Args:
        qtensor: QuantizedTensor produced by nvfp4_quantize.

    Returns:
        Float32 tensor of original shape.
    """
    indices = qtensor.data  # long, values 0..15
    scales = qtensor.scales  # [n_blocks]
    n = indices.shape[0]

    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(pad, dtype=indices.dtype)])

    n_padded = indices.shape[0]
    n_blocks = n_padded // _NVFP4_BLOCK_SIZE

    lut = _NVFP4_LUT.to(indices.device)
    values = lut[indices]  # dequantized without scale

    # Apply block scales
    values = values.view(n_blocks, _NVFP4_BLOCK_SIZE)
    values = values * scales.unsqueeze(1)
    values = values.view(-1)[:n]

    return values.reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# MXFP6 (OCP MX, E3M2) format
# ---------------------------------------------------------------------------

# E3M2: 1 sign, 3 exp (bias=3), 2 mantissa bits
# Normal range: exp 1..6, max = (1 + 3/4) * 2^3 = 28.0
# Subnormal: exp=0
_MXFP6_BLOCK_SIZE = 32
_MXFP6_BIAS = 3
_MXFP6_MAX_EXP = 6  # 7 is reserved (NaN/Inf in some specs)
_MXFP6_MANTISSA_BITS = 2
_MXFP6_MAX_VAL = (1.0 + (2**_MXFP6_MANTISSA_BITS - 1) / 2**_MXFP6_MANTISSA_BITS) * (2 ** _MXFP6_MAX_EXP)  # 28.0


def _build_fp6_e3m2_lut() -> torch.Tensor:
    """Build lookup table for all 64 FP6 E3M2 values (as float32)."""
    values = []
    for code in range(64):
        sign_bit = (code >> 5) & 1
        exp_bits = (code >> 2) & 0b111
        mantissa_bits = code & 0b11

        sign = -1.0 if sign_bit else 1.0

        if exp_bits == 0:
            # Subnormal
            val = sign * (mantissa_bits / 4.0) * (2 ** (1 - _MXFP6_BIAS))
        elif exp_bits == 7:
            # NaN/special -> treat as max
            val = sign * _MXFP6_MAX_VAL
        else:
            val = sign * (1.0 + mantissa_bits / 4.0) * (2 ** (exp_bits - _MXFP6_BIAS))
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


_MXFP6_LUT = _build_fp6_e3m2_lut()


def _float_to_fp6_e3m2(x: float) -> int:
    """Encode a single float as FP6 E3M2 code (0..63)."""
    sign_bit = 1 if x < 0 else 0
    abs_x = abs(x)

    if abs_x == 0.0:
        return sign_bit << 5

    # Clamp to representable range
    abs_x = min(abs_x, _MXFP6_MAX_VAL)

    log2_x = math.log2(abs_x) if abs_x > 0 else -100
    exp_unbiased = math.floor(log2_x)
    exp_biased = exp_unbiased + _MXFP6_BIAS

    if exp_biased <= 0:
        # Subnormal
        mantissa_f = abs_x / (2 ** (1 - _MXFP6_BIAS)) * 4.0
        mantissa_i = min(3, max(0, round(mantissa_f)))
        return (sign_bit << 5) | mantissa_i
    else:
        exp_biased = min(exp_biased, _MXFP6_MAX_EXP)
        mantissa_f = (abs_x / (2 ** (exp_biased - _MXFP6_BIAS)) - 1.0) * 4.0
        mantissa_i = min(3, max(0, round(mantissa_f)))
        # Handle rounding overflow
        if mantissa_i == 4:
            mantissa_i = 0
            exp_biased = min(exp_biased + 1, _MXFP6_MAX_EXP)
        return (sign_bit << 5) | (exp_biased << 2) | mantissa_i


def mxfp6_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to MXFP6 E3M2 with per-32-element shared exponent.

    The OCP MX shared exponent is the floor(log2(absmax)) of the block,
    encoded as uint8. Each element is then scaled by 2^(-shared_exp) before
    encoding as FP6 E3M2.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=fp6 codes (long), scales=shared_exponents (int8 as float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    pad = (-n) % _MXFP6_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    n_padded = flat.shape[0]
    n_blocks = n_padded // _MXFP6_BLOCK_SIZE
    blocks = flat.view(n_blocks, _MXFP6_BLOCK_SIZE)

    codes_list = []
    shared_exps = []

    for i in range(n_blocks):
        block = blocks[i]
        absmax = block.abs().max().item()

        if absmax == 0.0:
            shared_exp = 0
        else:
            shared_exp = int(math.floor(math.log2(absmax)))

        shared_exps.append(float(shared_exp))
        scale = 2.0 ** shared_exp

        # Normalize and encode each element
        normalized = (block / scale).tolist()
        block_codes = [_float_to_fp6_e3m2(v) for v in normalized]
        codes_list.extend(block_codes)

    codes = torch.tensor(codes_list[:n], dtype=torch.long)
    scales = torch.tensor(shared_exps, dtype=torch.float32)

    return QuantizedTensor(
        data=codes,
        scales=scales,
        format_name="MXFP6",
        original_shape=original_shape,
    )


def mxfp6_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP6 QuantizedTensor back to float32."""
    codes = qtensor.data
    shared_exps = qtensor.scales
    n = codes.shape[0]

    pad = (-n) % _MXFP6_BLOCK_SIZE
    if pad > 0:
        codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype)])

    n_padded = codes.shape[0]
    n_blocks = n_padded // _MXFP6_BLOCK_SIZE

    lut = _MXFP6_LUT.to(codes.device)
    values = lut[codes]  # normalized values

    # Apply shared exponent scales
    values = values.view(n_blocks, _MXFP6_BLOCK_SIZE)
    block_scales = (2.0 ** shared_exps).unsqueeze(1)
    values = values * block_scales
    values = values.view(-1)[:n]

    return values.reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# MXFP8 (OCP MX, E4M3) format
# ---------------------------------------------------------------------------

_MXFP8_BLOCK_SIZE = 32
_MXFP8_E4M3_BIAS = 7
_MXFP8_E4M3_MAX_EXP = 14  # 15 is NaN
_MXFP8_E4M3_MANTISSA_BITS = 3
_MXFP8_E4M3_MAX_VAL = (1.0 + 7.0 / 8.0) * (2 ** _MXFP8_E4M3_MAX_EXP)  # 448.0


def _build_fp8_e4m3_lut() -> torch.Tensor:
    """Build lookup table for all 256 FP8 E4M3 values (as float32)."""
    values = []
    for code in range(256):
        sign_bit = (code >> 7) & 1
        exp_bits = (code >> 3) & 0b1111
        mantissa_bits = code & 0b111

        sign = -1.0 if sign_bit else 1.0

        if exp_bits == 0:
            # Subnormal
            val = sign * (mantissa_bits / 8.0) * (2 ** (1 - _MXFP8_E4M3_BIAS))
        elif exp_bits == 15:
            if mantissa_bits == 7:
                val = float('nan')
            else:
                val = sign * _MXFP8_E4M3_MAX_VAL  # treat inf-like as max
        else:
            val = sign * (1.0 + mantissa_bits / 8.0) * (2 ** (exp_bits - _MXFP8_E4M3_BIAS))
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


_MXFP8_E4M3_LUT = _build_fp8_e4m3_lut()
# Replace NaN with 0 for safe LUT usage
_MXFP8_E4M3_LUT = torch.where(torch.isnan(_MXFP8_E4M3_LUT),
                                torch.zeros_like(_MXFP8_E4M3_LUT),
                                _MXFP8_E4M3_LUT)


def _float_to_fp8_e4m3(x: float) -> int:
    """Encode a single float as FP8 E4M3 code (0..255)."""
    sign_bit = 1 if x < 0 else 0
    abs_x = abs(x)

    if abs_x == 0.0:
        return sign_bit << 7

    abs_x = min(abs_x, _MXFP8_E4M3_MAX_VAL)
    log2_x = math.log2(abs_x)
    exp_unbiased = math.floor(log2_x)
    exp_biased = exp_unbiased + _MXFP8_E4M3_BIAS

    if exp_biased <= 0:
        # Subnormal
        mantissa_f = abs_x / (2 ** (1 - _MXFP8_E4M3_BIAS)) * 8.0
        mantissa_i = min(7, max(0, round(mantissa_f)))
        return (sign_bit << 7) | mantissa_i
    else:
        exp_biased = min(exp_biased, _MXFP8_E4M3_MAX_EXP)
        mantissa_f = (abs_x / (2 ** (exp_biased - _MXFP8_E4M3_BIAS)) - 1.0) * 8.0
        mantissa_i = min(7, max(0, round(mantissa_f)))
        if mantissa_i == 8:
            mantissa_i = 0
            exp_biased = min(exp_biased + 1, _MXFP8_E4M3_MAX_EXP)
        return (sign_bit << 7) | (exp_biased << 3) | mantissa_i


def mxfp8_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to MXFP8 E4M3 with per-32-element shared exponent.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=fp8 codes (long), scales=shared_exponents (float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    pad = (-n) % _MXFP8_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    n_padded = flat.shape[0]
    n_blocks = n_padded // _MXFP8_BLOCK_SIZE
    blocks = flat.view(n_blocks, _MXFP8_BLOCK_SIZE)

    codes_list = []
    shared_exps = []

    for i in range(n_blocks):
        block = blocks[i]
        absmax = block.abs().max().item()

        if absmax == 0.0:
            shared_exp = 0
        else:
            shared_exp = int(math.floor(math.log2(absmax)))

        shared_exps.append(float(shared_exp))
        scale = 2.0 ** shared_exp

        normalized = (block / scale).tolist()
        block_codes = [_float_to_fp8_e4m3(v) for v in normalized]
        codes_list.extend(block_codes)

    codes = torch.tensor(codes_list[:n], dtype=torch.long)
    scales = torch.tensor(shared_exps, dtype=torch.float32)

    return QuantizedTensor(
        data=codes,
        scales=scales,
        format_name="MXFP8",
        original_shape=original_shape,
    )


def mxfp8_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP8 QuantizedTensor back to float32."""
    codes = qtensor.data
    shared_exps = qtensor.scales
    n = codes.shape[0]

    pad = (-n) % _MXFP8_BLOCK_SIZE
    if pad > 0:
        codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype)])

    n_padded = codes.shape[0]
    n_blocks = n_padded // _MXFP8_BLOCK_SIZE

    lut = _MXFP8_E4M3_LUT.to(codes.device)
    values = lut[codes]

    values = values.view(n_blocks, _MXFP8_BLOCK_SIZE)
    block_scales = (2.0 ** shared_exps).unsqueeze(1)
    values = values * block_scales
    values = values.view(-1)[:n]

    return values.reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# FP16 (passthrough)
# ---------------------------------------------------------------------------

def fp16_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """'Quantize' to FP16 — just cast and store.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=float16 tensor, scales=empty.
    """
    return QuantizedTensor(
        data=tensor.flatten().half(),
        scales=torch.empty(0),
        format_name="FP16",
        original_shape=tensor.shape,
    )


def fp16_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize FP16 QuantizedTensor back to float32."""
    return qtensor.data.float().reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# Unified format registry
# ---------------------------------------------------------------------------

_FORMATS: dict[str, dict] = {
    "NVFP4": {
        "bits_per_element": 4,
        "quantize": nvfp4_quantize,
        "dequantize": nvfp4_dequantize,
    },
    "MXFP6": {
        "bits_per_element": 6,
        "quantize": mxfp6_quantize,
        "dequantize": mxfp6_dequantize,
    },
    "MXFP8": {
        "bits_per_element": 8,
        "quantize": mxfp8_quantize,
        "dequantize": mxfp8_dequantize,
    },
    "FP16": {
        "bits_per_element": 16,
        "quantize": fp16_quantize,
        "dequantize": fp16_dequantize,
    },
}


def get_bits_per_element(format_name: str) -> int:
    """Return nominal bits-per-element for the given format name.

    Args:
        format_name: One of "NVFP4", "MXFP6", "MXFP8", "FP16".

    Returns:
        Integer bit-width.
    """
    return _FORMATS[format_name]["bits_per_element"]


def quantize(tensor: torch.Tensor, format_name: str) -> QuantizedTensor:
    """Quantize a 1D tensor using the specified format.

    Args:
        tensor: 1D float tensor (single output channel's weights).
        format_name: One of "NVFP4", "MXFP6", "MXFP8", "FP16".

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
        format_name: One of "NVFP4", "MXFP6", "MXFP8", "FP16".

    Returns:
        Mean squared error as a Python float.
    """
    with torch.no_grad():
        qtensor = quantize(tensor, format_name)
        reconstructed = dequantize(qtensor)
        return ((tensor.float() - reconstructed.float()) ** 2).mean().item()
