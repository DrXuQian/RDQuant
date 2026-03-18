"""
Mixed-precision linear operators for RDQuant.

Execution backends:

  - ``pytorch``:  Pure PyTorch.  Dequant each format group -> ``F.linear`` ->
    concat -> inv-permute.  Correct on any device, used for validation.

  - ``vllm`` (future):  Drop-in replacement using vLLM's native
    Tensor-Core kernels on Blackwell:
        MXFP8 x MXFPx GEMM per group
        Activations quantized to MXFP8 before each GEMM

All backends produce identical results (up to floating-point ordering).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rdquant.core.formats import dequantize, MXQuantizedTensor
from rdquant.core.act_quant import quantize_activation_mxfp8, dequantize_activation_mxfp8


# --------------------------------------------------------------------------
#  Backend: pure-PyTorch
# --------------------------------------------------------------------------

def mixed_precision_linear(
    x: torch.Tensor,
    qtensors: dict[str, MXQuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    quantize_activation: bool = True,
) -> torch.Tensor:
    """Mixed-precision linear: per-format dequant -> GEMM -> concat -> permute.

    Optionally applies MXFP8 activation quantization before the GEMMs.

    Args:
        x: Input activations ``[..., in_features]``.
        qtensors: Format-name -> MXQuantizedTensor for each group's weights,
            stored as ``[n_channels, in_features]`` (via ``original_shape``).
        splits: Format-name -> number of output channels in that group.
        inv_perm: ``[out_features]`` inverse-permutation to restore original
            channel order after concatenation.
        bias: Optional bias ``[out_features]`` in original channel order.
        quantize_activation: If True, apply MXFP8 quantize->dequantize on
            activations before the GEMMs. Default True.

    Returns:
        ``[..., out_features]`` output tensor.
    """
    if quantize_activation:
        # Straight-through estimator: quantize-dequantize the activation values
        # but allow gradients to flow through unchanged.
        with torch.no_grad():
            x_codes, x_scales = quantize_activation_mxfp8(x)
            x_hat = dequantize_activation_mxfp8(x_codes, x_scales, x.shape).to(x.dtype)
        # Straight-through: forward uses quantized values, backward passes gradient through
        x = x + (x_hat - x).detach()

    parts: list[torch.Tensor] = []

    for fmt, qt in qtensors.items():
        n_ch = splits[fmt]
        if n_ch == 0:
            continue
        # Dequantize this group's weight block  ->  [n_ch, in_features]
        w_group = dequantize(qt).to(x.dtype)
        # GEMM for this group  ->  [..., n_ch]
        parts.append(F.linear(x, w_group))

    # Concat along output dim (permuted order) -> [..., out_features]
    y_permuted = torch.cat(parts, dim=-1)

    # Restore original channel order (clone inv_perm to escape inference_mode
    # so autograd can save it for backward)
    y = y_permuted.index_select(-1, inv_perm.to(y_permuted.device).clone())

    if bias is not None:
        y = y + bias

    return y


# --------------------------------------------------------------------------
#  Backend: vLLM native kernels (stub)
# --------------------------------------------------------------------------

def mixed_precision_linear_vllm(
    x: torch.Tensor,
    qtensors: dict[str, MXQuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    quantize_activation: bool = True,
) -> torch.Tensor:
    """Same interface as :func:`mixed_precision_linear`, but uses vLLM's
    native CUDA kernels.  Requires Blackwell GPU + vLLM installed.

    Kernel mapping::

        MXFP8 activation quantization applied once
        MXFP8 x MXFPx GEMM per format group

    Falls back to :func:`mixed_precision_linear` if vLLM is not available.
    """
    # TODO: implement when vLLM kernels are available
    return mixed_precision_linear(x, qtensors, splits, inv_perm, bias, quantize_activation)
