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

All quantize functions are fully vectorized (no Python element loops).
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

_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN_NORMAL = 2.0 ** (-9)


def _quantize_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize float tensor to FP8 E4M3 representable values (returned as float32)."""
    sign = torch.sign(x)
    abs_x = x.abs().clamp(max=_FP8_E4M3_MAX)

    result = torch.zeros_like(abs_x)
    nonzero = abs_x >= _FP8_E4M3_MIN_NORMAL / 2.0

    if nonzero.any():
        v = abs_x[nonzero]
        exp_f = torch.floor(torch.log2(v.clamp(min=1e-38)))
        exp_biased = (exp_f + 7).clamp(1, 14)
        exp_f_clamped = exp_biased - 7
        scale = 2.0 ** exp_f_clamped
        mantissa_i = ((v / scale - 1.0) * 8.0).round().clamp(0, 7).long()
        result[nonzero] = (1.0 + mantissa_i.float() / 8.0) * scale

    return sign * result


# ---------------------------------------------------------------------------
# NVFP4 (E2M1) format — fully vectorized
# ---------------------------------------------------------------------------

_NVFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_NVFP4_POS_VALUES = _NVFP4_LUT[:8]
_NVFP4_MAX_VAL = 6.0
_NVFP4_BLOCK_SIZE = 16


def nvfp4_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to NVFP4 format (vectorized).

    Args:
        tensor: 1D float tensor (single output channel weights).

    Returns:
        QuantizedTensor with data=indices (long), scales=block FP8 scales (float32).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    n_blocks = flat.shape[0] // _NVFP4_BLOCK_SIZE
    blocks = flat.view(n_blocks, _NVFP4_BLOCK_SIZE)  # [B, 16]

    # Per-block absmax → scale
    absmax = blocks.abs().amax(dim=1)  # [B]
    raw_scale = absmax / _NVFP4_MAX_VAL
    scale_fp8 = _quantize_to_fp8_e4m3(raw_scale)  # [B]
    # Avoid div-by-zero for all-zero blocks
    safe_scale = scale_fp8.clone()
    safe_scale[safe_scale == 0.0] = 1.0

    # Normalize blocks
    normalized = blocks / safe_scale.unsqueeze(1)  # [B, 16]

    # Nearest LUT entry: separate sign and magnitude
    sign = (normalized < 0).long()     # [B, 16]
    abs_norm = normalized.abs()        # [B, 16]

    # Distances to each positive LUT value — [B, 16, 8]
    pos_vals = _NVFP4_POS_VALUES.to(flat.device)
    dists = (abs_norm.unsqueeze(2) - pos_vals).abs()
    mag_idx = dists.argmin(dim=2)      # [B, 16]

    indices = (mag_idx + sign * 8).view(-1)[:n]  # [n]

    return QuantizedTensor(
        data=indices,
        scales=scale_fp8,
        format_name="NVFP4",
        original_shape=original_shape,
    )


def nvfp4_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize NVFP4 QuantizedTensor back to float32."""
    indices = qtensor.data
    scales = qtensor.scales
    n = indices.shape[0]

    pad = (-n) % _NVFP4_BLOCK_SIZE
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(pad, dtype=indices.dtype)])

    n_blocks = indices.shape[0] // _NVFP4_BLOCK_SIZE
    lut = _NVFP4_LUT.to(indices.device)
    values = lut[indices].view(n_blocks, _NVFP4_BLOCK_SIZE)
    values = values * scales.unsqueeze(1)
    return values.view(-1)[:n].reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# Vectorized MX-format quantizer (shared by MXFP6 and MXFP8)
# ---------------------------------------------------------------------------

def _mx_quantize_blocks(
    flat: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-block shared exponent and return (normalized blocks, shared_exp).

    Args:
        flat: 1-D float tensor, length already padded to multiple of block_size.
        block_size: Block size (32 for MX formats).

    Returns:
        normalized: [B, block_size] tensor, each element divided by 2^shared_exp.
        shared_exp: [B] float tensor of shared exponents.
    """
    n_blocks = flat.shape[0] // block_size
    blocks = flat.view(n_blocks, block_size)

    absmax = blocks.abs().amax(dim=1)
    shared_exp = torch.zeros(n_blocks, dtype=torch.float32)
    nonzero = absmax > 0
    if nonzero.any():
        shared_exp[nonzero] = torch.floor(torch.log2(absmax[nonzero]))

    scale = (2.0 ** shared_exp).unsqueeze(1)
    normalized = blocks / scale
    return normalized, shared_exp


def _vectorized_encode_fp(
    x: torch.Tensor,
    bias: int,
    max_exp: int,
    mantissa_bits: int,
    max_val: float,
) -> torch.Tensor:
    """Vectorized floating-point encode: float → integer code.

    Directly computes (sign, exponent, mantissa) bit fields for each element.
    O(n) instead of O(n × LUT_size).

    Args:
        x: Flat float32 tensor of normalized values.
        bias: Exponent bias.
        max_exp: Maximum biased exponent (NaN/special exponents excluded).
        mantissa_bits: Number of mantissa bits.
        max_val: Maximum representable absolute value.

    Returns:
        Long tensor of integer codes, same shape as x.
    """
    total_bits = 1 + int(max_exp).bit_length() + mantissa_bits  # not used, just doc
    mantissa_levels = 1 << mantissa_bits  # 2^m

    sign_bit = (x < 0).long()
    abs_x = x.abs().clamp(max=max_val)

    # Subnormal threshold
    min_normal = 2.0 ** (1 - bias)
    is_zero = abs_x < min_normal / (2 * mantissa_levels)  # below half-ULP of smallest subnormal

    # Subnormal path: exp_biased=0, mantissa = round(abs_x / (2^(1-bias)) * mantissa_levels)
    subnorm_mantissa = (abs_x / min_normal * mantissa_levels).round().clamp(0, mantissa_levels - 1).long()

    # Normal path
    log2_val = torch.log2(abs_x.clamp(min=1e-38))
    exp_unbiased = torch.floor(log2_val)
    exp_biased = (exp_unbiased + bias).clamp(1, max_exp).long()
    exp_actual = exp_biased - bias  # reconvert after clamping
    scale = 2.0 ** exp_actual.float()
    mantissa_f = (abs_x / scale - 1.0) * mantissa_levels
    mantissa_i = mantissa_f.round().clamp(0, mantissa_levels).long()

    # Handle mantissa rounding overflow
    overflow = mantissa_i == mantissa_levels
    mantissa_i[overflow] = 0
    exp_biased[overflow] = (exp_biased[overflow] + 1).clamp(max=max_exp)

    # Determine exp_bit_width from max_exp
    exp_bit_width = max_exp.bit_length()  # number of bits needed for exponent field

    # Assemble code
    is_subnormal = (abs_x < 2.0 ** (1 - bias)) & ~is_zero
    code = torch.where(
        is_zero,
        sign_bit << (exp_bit_width + mantissa_bits),
        torch.where(
            is_subnormal,
            (sign_bit << (exp_bit_width + mantissa_bits)) | subnorm_mantissa,
            (sign_bit << (exp_bit_width + mantissa_bits)) | (exp_biased << mantissa_bits) | mantissa_i,
        ),
    )
    return code


def _mx_dequantize(qtensor: QuantizedTensor, lut: torch.Tensor, block_size: int) -> torch.Tensor:
    """Dequantize an MX-format QuantizedTensor back to float32."""
    codes = qtensor.data
    shared_exps = qtensor.scales
    n = codes.shape[0]

    pad = (-n) % block_size
    if pad > 0:
        codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype)])

    n_blocks = codes.shape[0] // block_size
    lut_d = lut.to(codes.device)
    values = lut_d[codes].view(n_blocks, block_size)
    values = values * (2.0 ** shared_exps).unsqueeze(1)
    return values.view(-1)[:n].reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# MXFP6 (OCP MX, E3M2)
# ---------------------------------------------------------------------------

_MXFP6_BLOCK_SIZE = 32
_MXFP6_BIAS = 3
_MXFP6_MAX_EXP = 6
_MXFP6_MANTISSA_BITS = 2
_MXFP6_MAX_VAL = (1.0 + (2**_MXFP6_MANTISSA_BITS - 1) / 2**_MXFP6_MANTISSA_BITS) * (2 ** _MXFP6_MAX_EXP)


def _build_fp6_e3m2_lut() -> torch.Tensor:
    """Build lookup table for all 64 FP6 E3M2 values (as float32)."""
    values = []
    for code in range(64):
        sign_bit = (code >> 5) & 1
        exp_bits = (code >> 2) & 0b111
        mantissa_bits = code & 0b11
        sign = -1.0 if sign_bit else 1.0
        if exp_bits == 0:
            val = sign * (mantissa_bits / 4.0) * (2 ** (1 - _MXFP6_BIAS))
        elif exp_bits == 7:
            val = sign * _MXFP6_MAX_VAL
        else:
            val = sign * (1.0 + mantissa_bits / 4.0) * (2 ** (exp_bits - _MXFP6_BIAS))
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


_MXFP6_LUT = _build_fp6_e3m2_lut()


def mxfp6_quantize(tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize a 1D tensor to MXFP6 E3M2 with per-32-element shared exponent.

    Args:
        tensor: 1D float tensor.

    Returns:
        QuantizedTensor with data=fp6 codes (long), scales=shared_exponents (float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]
    pad = (-n) % _MXFP6_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    normalized, shared_exp = _mx_quantize_blocks(flat, _MXFP6_BLOCK_SIZE)
    codes = _vectorized_encode_fp(
        normalized.reshape(-1),
        bias=_MXFP6_BIAS, max_exp=_MXFP6_MAX_EXP,
        mantissa_bits=_MXFP6_MANTISSA_BITS, max_val=_MXFP6_MAX_VAL,
    )[:n]

    return QuantizedTensor(data=codes, scales=shared_exp,
                           format_name="MXFP6", original_shape=original_shape)


def mxfp6_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP6 QuantizedTensor back to float32."""
    return _mx_dequantize(qtensor, _MXFP6_LUT, _MXFP6_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# MXFP8 (OCP MX, E4M3)
# ---------------------------------------------------------------------------

_MXFP8_BLOCK_SIZE = 32
_MXFP8_E4M3_BIAS = 7
_MXFP8_E4M3_MAX_EXP = 14
_MXFP8_E4M3_MANTISSA_BITS = 3
_MXFP8_E4M3_MAX_VAL = (1.0 + 7.0 / 8.0) * (2 ** _MXFP8_E4M3_MAX_EXP)


def _build_fp8_e4m3_lut() -> torch.Tensor:
    """Build lookup table for all 256 FP8 E4M3 values (as float32)."""
    values = []
    for code in range(256):
        sign_bit = (code >> 7) & 1
        exp_bits = (code >> 3) & 0b1111
        mantissa_bits = code & 0b111
        sign = -1.0 if sign_bit else 1.0
        if exp_bits == 0:
            val = sign * (mantissa_bits / 8.0) * (2 ** (1 - _MXFP8_E4M3_BIAS))
        elif exp_bits == 15:
            val = float('nan') if mantissa_bits == 7 else sign * _MXFP8_E4M3_MAX_VAL
        else:
            val = sign * (1.0 + mantissa_bits / 8.0) * (2 ** (exp_bits - _MXFP8_E4M3_BIAS))
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


_MXFP8_E4M3_LUT = _build_fp8_e4m3_lut()
_MXFP8_E4M3_LUT = torch.where(torch.isnan(_MXFP8_E4M3_LUT),
                                torch.zeros_like(_MXFP8_E4M3_LUT),
                                _MXFP8_E4M3_LUT)


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

    normalized, shared_exp = _mx_quantize_blocks(flat, _MXFP8_BLOCK_SIZE)
    codes = _vectorized_encode_fp(
        normalized.reshape(-1),
        bias=_MXFP8_E4M3_BIAS, max_exp=_MXFP8_E4M3_MAX_EXP,
        mantissa_bits=_MXFP8_E4M3_MANTISSA_BITS, max_val=_MXFP8_E4M3_MAX_VAL,
    )[:n]

    return QuantizedTensor(data=codes, scales=shared_exp,
                           format_name="MXFP8", original_shape=original_shape)


def mxfp8_dequantize(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP8 QuantizedTensor back to float32."""
    return _mx_dequantize(qtensor, _MXFP8_E4M3_LUT, _MXFP8_BLOCK_SIZE)


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


def compute_mse_2d(weight: torch.Tensor, format_name: str) -> torch.Tensor:
    """Batch quantize/dequantize a 2-D weight matrix and return per-row MSE.

    Flattens the entire matrix, quantizes once as a single 1-D tensor, then
    reshapes and computes per-channel (per-row) MSE.  Much faster than calling
    compute_mse() in a loop over rows.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        format_name: One of "NVFP4", "MXFP6", "MXFP8", "FP16".

    Returns:
        Float32 tensor of shape [N_out] with per-row MSE values.
    """
    with torch.no_grad():
        n_out, n_in = weight.shape
        flat = weight.reshape(-1)
        qtensor = quantize(flat, format_name)
        recon = dequantize(qtensor).reshape(n_out, n_in)
        return ((weight.float() - recon.float()) ** 2).mean(dim=1)
