"""
MXFP8 activation quantization.

Provides quantize/dequantize for activations using MXFP8 E4M3 format
with per-32-element UE8M0 shared exponent. Applied uniformly to all
activations before the MXFP8 x MXFPx GEMM.
"""

from __future__ import annotations

import math

import torch

from rdquant.core.formats import (
    _MX_BLOCK_SIZE,
    _mx_quantize_blocks,
    _vectorized_encode_fp,
    _MXFP8_E4M3_BIAS,
    _MXFP8_E4M3_MAX_EXP,
    _MXFP8_E4M3_MANTISSA_BITS,
    _MXFP8_E4M3_MAX_VAL,
    _MXFP8_E4M3_LUT,
)


@torch.no_grad()
def quantize_activation_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP32 activations to MXFP8 E4M3 with per-32 block scaling.

    Args:
        x: Float tensor of shape [*, K] where K is the last dimension.

    Returns:
        Tuple of:
            x_codes: Long tensor of shape [*, K] with MXFP8 codes.
            x_scales: Float32 tensor of shape [*, ceil(K/32)] with shared exponents.
    """
    original_shape = x.shape
    K = original_shape[-1]
    # Reshape to [M, K] where M is product of all leading dims
    leading = x.reshape(-1, K)
    M = leading.shape[0]

    flat = leading.float()
    n = flat.shape[1]  # K

    # Pad K to multiple of block_size
    pad = (-n) % _MX_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(M, pad, dtype=flat.dtype, device=flat.device)], dim=1)

    K_padded = flat.shape[1]
    n_blocks_per_row = K_padded // _MX_BLOCK_SIZE

    # Reshape to [M * n_blocks_per_row, block_size] for block processing
    blocks = flat.reshape(M * n_blocks_per_row, _MX_BLOCK_SIZE)

    # Compute shared exponents
    absmax = blocks.abs().amax(dim=1)  # [M * n_blocks_per_row]
    shared_exp = torch.zeros_like(absmax)
    nonzero = absmax > 0
    if nonzero.any():
        shared_exp[nonzero] = torch.floor(torch.log2(absmax[nonzero]))

    scale = (2.0 ** shared_exp).unsqueeze(1)  # [M * n_blocks_per_row, 1]
    normalized = blocks / scale  # [M * n_blocks_per_row, block_size]

    # Encode to MXFP8 codes
    codes_flat = _vectorized_encode_fp(
        normalized.reshape(-1),
        bias=_MXFP8_E4M3_BIAS,
        max_exp=_MXFP8_E4M3_MAX_EXP,
        mantissa_bits=_MXFP8_E4M3_MANTISSA_BITS,
        max_val=_MXFP8_E4M3_MAX_VAL,
    )

    # Reshape codes back to [M, K_padded] and trim
    codes_2d = codes_flat.reshape(M, K_padded)[:, :K]
    # Reshape scales to [M, n_blocks_per_row]
    scales_2d = shared_exp.reshape(M, n_blocks_per_row)

    # Restore leading dimensions
    x_codes = codes_2d.reshape(*original_shape)
    n_blocks_k = math.ceil(K / _MX_BLOCK_SIZE)
    x_scales = scales_2d[:, :n_blocks_k].reshape(*original_shape[:-1], n_blocks_k)

    return x_codes, x_scales


@torch.no_grad()
def dequantize_activation_mxfp8(
    x_codes: torch.Tensor,
    x_scales: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Reconstruct float tensor from MXFP8 codes + scales.

    Args:
        x_codes: Long tensor of shape [*, K] with MXFP8 codes.
        x_scales: Float32 tensor of shape [*, ceil(K/32)] with shared exponents.
        original_shape: Original shape of the activation tensor.

    Returns:
        Float32 tensor of shape original_shape.
    """
    K = original_shape[-1]
    # Reshape to 2D
    codes_2d = x_codes.reshape(-1, K)
    M = codes_2d.shape[0]
    n_blocks_per_row = x_scales.reshape(M, -1).shape[1]

    # Pad K to multiple of block_size
    pad = (-K) % _MX_BLOCK_SIZE
    if pad > 0:
        codes_2d = torch.cat([
            codes_2d,
            torch.zeros(M, pad, dtype=codes_2d.dtype, device=codes_2d.device),
        ], dim=1)

    K_padded = codes_2d.shape[1]
    n_blocks_padded = K_padded // _MX_BLOCK_SIZE

    # Pad scales if needed
    scales_2d = x_scales.reshape(M, n_blocks_per_row)
    if n_blocks_padded > n_blocks_per_row:
        extra = n_blocks_padded - n_blocks_per_row
        scales_2d = torch.cat([
            scales_2d,
            torch.zeros(M, extra, dtype=scales_2d.dtype, device=scales_2d.device),
        ], dim=1)

    # LUT lookup
    lut = _MXFP8_E4M3_LUT.to(codes_2d.device)
    values = lut[codes_2d]  # [M, K_padded]

    # Apply shared exponents per block
    blocks = values.reshape(M * n_blocks_padded, _MX_BLOCK_SIZE)
    exp_flat = scales_2d.reshape(M * n_blocks_padded)
    blocks = blocks * (2.0 ** exp_flat).unsqueeze(1)

    # Reshape and trim
    result = blocks.reshape(M, K_padded)[:, :K]
    return result.reshape(original_shape)
