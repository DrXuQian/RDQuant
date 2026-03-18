"""
Numeric format definitions and quantize/dequantize implementations.

Supported formats (MX-only):
  - MXFP4:  E2M1 with per-32-element UE8M0 shared exponent
  - MXFP6:  E3M2 with per-32-element UE8M0 shared exponent
  - MXFP8:  E4M3 with per-32-element UE8M0 shared exponent

Each format exposes:
  - bits_per_element: int — the nominal bit-width (4, 6, 8)
  - quantize(tensor) -> MXQuantizedTensor  — returns packed data + metadata
  - dequantize(qtensor) -> tensor — reconstruct full-precision approximation
  - compute_mse(tensor) -> float — quantize, dequantize, compute MSE (convenience)

All quantize functions are fully vectorized (no Python element loops).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MXQuantizedTensor:
    """Container for MX-format quantized data and associated metadata."""
    data: torch.Tensor        # quantized codes (long)
    scales: torch.Tensor      # shared exponents (float32), one per 32 elements
    format_name: str          # "MXFP4", "MXFP6", "MXFP8"
    original_shape: torch.Size
    bits_per_element: int     # 4, 6, 8


# ---------------------------------------------------------------------------
# Shared MX block quantization (per-32-element shared exponent)
# ---------------------------------------------------------------------------

_MX_BLOCK_SIZE = 32


def _mx_quantize_blocks(
    flat: torch.Tensor,
    block_size: int = _MX_BLOCK_SIZE,
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
    shared_exp = torch.zeros(n_blocks, dtype=torch.float32, device=flat.device)
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
    """Vectorized floating-point encode: float -> integer code.

    Directly computes (sign, exponent, mantissa) bit fields for each element.
    O(n) instead of O(n x LUT_size).

    Args:
        x: Flat float32 tensor of normalized values.
        bias: Exponent bias.
        max_exp: Maximum biased exponent (NaN/special exponents excluded).
        mantissa_bits: Number of mantissa bits.
        max_val: Maximum representable absolute value.

    Returns:
        Long tensor of integer codes, same shape as x.
    """
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


def _mx_dequantize(qtensor: MXQuantizedTensor, lut: torch.Tensor, block_size: int) -> torch.Tensor:
    """Dequantize an MX-format MXQuantizedTensor back to float32."""
    codes = qtensor.data
    shared_exps = qtensor.scales
    n = codes.shape[0]

    pad = (-n) % block_size
    if pad > 0:
        codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype, device=codes.device)])

    n_blocks = codes.shape[0] // block_size
    lut_d = lut.to(codes.device)
    values = lut_d[codes].view(n_blocks, block_size)
    values = values * (2.0 ** shared_exps.to(values.device)).unsqueeze(1)
    return values.view(-1)[:n].reshape(qtensor.original_shape)


# ---------------------------------------------------------------------------
# MXFP4 (E2M1) — per-32-element shared exponent, nearest-LUT quantization
# ---------------------------------------------------------------------------

# Magnitude grid for E2M1: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# Full signed LUT: 16 entries (8 positive including 0, 8 negative including -0)
_MXFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_MXFP4_POS_VALUES = _MXFP4_LUT[:8]
_MXFP4_MAX_VAL = 6.0


def mxfp4_quantize(tensor: torch.Tensor) -> MXQuantizedTensor:
    """Quantize a 1D tensor to MXFP4 E2M1 with per-32-element shared exponent.

    Args:
        tensor: 1D float tensor.

    Returns:
        MXQuantizedTensor with data=indices (long), scales=shared exponents (float32).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]

    pad = (-n) % _MX_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    normalized, shared_exp = _mx_quantize_blocks(flat, _MX_BLOCK_SIZE)

    # Nearest LUT entry: separate sign and magnitude
    norm_flat = normalized.reshape(-1)
    sign = (norm_flat < 0).long()
    abs_norm = norm_flat.abs()

    # Distances to each positive LUT value — [N, 8]
    pos_vals = _MXFP4_POS_VALUES.to(flat.device)
    dists = (abs_norm.unsqueeze(1) - pos_vals).abs()
    mag_idx = dists.argmin(dim=1)  # [N]

    indices = (mag_idx + sign * 8)[:n]

    return MXQuantizedTensor(
        data=indices,
        scales=shared_exp,
        format_name="MXFP4",
        original_shape=original_shape,
        bits_per_element=4,
    )


def mxfp4_dequantize(qtensor: MXQuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP4 MXQuantizedTensor back to float32."""
    return _mx_dequantize(qtensor, _MXFP4_LUT, _MX_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# MXFP6 (OCP MX, E3M2)
# ---------------------------------------------------------------------------

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


def mxfp6_quantize(tensor: torch.Tensor) -> MXQuantizedTensor:
    """Quantize a 1D tensor to MXFP6 E3M2 with per-32-element shared exponent.

    Args:
        tensor: 1D float tensor.

    Returns:
        MXQuantizedTensor with data=fp6 codes (long), scales=shared_exponents (float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]
    pad = (-n) % _MX_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    normalized, shared_exp = _mx_quantize_blocks(flat, _MX_BLOCK_SIZE)
    codes = _vectorized_encode_fp(
        normalized.reshape(-1),
        bias=_MXFP6_BIAS, max_exp=_MXFP6_MAX_EXP,
        mantissa_bits=_MXFP6_MANTISSA_BITS, max_val=_MXFP6_MAX_VAL,
    )[:n]

    return MXQuantizedTensor(
        data=codes, scales=shared_exp,
        format_name="MXFP6", original_shape=original_shape,
        bits_per_element=6,
    )


def mxfp6_dequantize(qtensor: MXQuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP6 MXQuantizedTensor back to float32."""
    return _mx_dequantize(qtensor, _MXFP6_LUT, _MX_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# MXFP8 (OCP MX, E4M3)
# ---------------------------------------------------------------------------

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


def mxfp8_quantize(tensor: torch.Tensor) -> MXQuantizedTensor:
    """Quantize a 1D tensor to MXFP8 E4M3 with per-32-element shared exponent.

    Args:
        tensor: 1D float tensor.

    Returns:
        MXQuantizedTensor with data=fp8 codes (long), scales=shared_exponents (float).
    """
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.shape[0]
    pad = (-n) % _MX_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    normalized, shared_exp = _mx_quantize_blocks(flat, _MX_BLOCK_SIZE)
    codes = _vectorized_encode_fp(
        normalized.reshape(-1),
        bias=_MXFP8_E4M3_BIAS, max_exp=_MXFP8_E4M3_MAX_EXP,
        mantissa_bits=_MXFP8_E4M3_MANTISSA_BITS, max_val=_MXFP8_E4M3_MAX_VAL,
    )[:n]

    return MXQuantizedTensor(
        data=codes, scales=shared_exp,
        format_name="MXFP8", original_shape=original_shape,
        bits_per_element=8,
    )


def mxfp8_dequantize(qtensor: MXQuantizedTensor) -> torch.Tensor:
    """Dequantize MXFP8 MXQuantizedTensor back to float32."""
    return _mx_dequantize(qtensor, _MXFP8_E4M3_LUT, _MX_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Unified format registry
# ---------------------------------------------------------------------------

_FORMATS: dict[str, dict] = {
    "MXFP4": {
        "bits_per_element": 4,
        "quantize": mxfp4_quantize,
        "dequantize": mxfp4_dequantize,
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
}


def get_bits_per_element(format_name: str) -> int:
    """Return nominal bits-per-element for the given format name.

    Args:
        format_name: One of "MXFP4", "MXFP6", "MXFP8".

    Returns:
        Integer bit-width.
    """
    return _FORMATS[format_name]["bits_per_element"]


def quantize(tensor: torch.Tensor, format_name: str) -> MXQuantizedTensor:
    """Quantize a 1D tensor using the specified format.

    Args:
        tensor: 1D float tensor (single output channel's weights).
        format_name: One of "MXFP4", "MXFP6", "MXFP8".

    Returns:
        MXQuantizedTensor containing packed data and metadata.
    """
    with torch.no_grad():
        return _FORMATS[format_name]["quantize"](tensor)


def dequantize(qtensor: MXQuantizedTensor) -> torch.Tensor:
    """Dequantize a MXQuantizedTensor back to float32.

    Args:
        qtensor: MXQuantizedTensor produced by quantize().

    Returns:
        Float32 tensor of original shape.
    """
    with torch.no_grad():
        return _FORMATS[qtensor.format_name]["dequantize"](qtensor)


def compute_mse(tensor: torch.Tensor, format_name: str) -> float:
    """Quantize then dequantize a tensor and compute MSE.

    Args:
        tensor: 1D float tensor.
        format_name: One of "MXFP4", "MXFP6", "MXFP8".

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
        format_name: One of "MXFP4", "MXFP6", "MXFP8".

    Returns:
        Float32 tensor of shape [N_out] with per-row MSE values.
    """
    with torch.no_grad():
        n_out, n_in = weight.shape
        flat = weight.reshape(-1)
        qtensor = quantize(flat, format_name)
        recon = dequantize(qtensor).reshape(n_out, n_in)
        return ((weight.float() - recon.float()) ** 2).mean(dim=1)
