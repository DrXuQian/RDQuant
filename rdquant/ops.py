"""
Mixed-precision linear operators for RDQuant.

Execution backends:

  - ``pytorch``:  Pure PyTorch.  Dequant each format group → ``F.linear`` →
    concat → inv-permute.  Correct on any device, used for validation.

  - ``vllm`` (Step 2, future):  Drop-in replacement using vLLM's native
    Tensor-Core kernels on Blackwell:
        NVFP4 → torch.ops._C.nvfp4_gemm
        MXFP8 → torch.ops._C.cutlass_scaled_mm
        FP16  → F.linear

  - ``fused`` (Step 4, optional):  Single grouped-GEMM kernel if multi-launch
    overhead is significant.

All backends produce identical results (up to floating-point ordering).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rdquant.core.formats import dequantize, QuantizedTensor


# ──────────────────────────────────────────────────────────────────────────────
#  Backend: pure-PyTorch (Step 1)
# ──────────────────────────────────────────────────────────────────────────────

def mixed_precision_linear(
    x: torch.Tensor,
    qtensors: dict[str, QuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mixed-precision linear: per-format dequant → GEMM → concat → permute.

    Instead of dequantizing the entire weight matrix and running one big GEMM,
    this performs one smaller GEMM per format group and concatenates the partial
    outputs.  Mathematically identical, but structured so each GEMM can be
    swapped for a native kernel in Step 2.

    Args:
        x: Input activations ``[..., in_features]``.
        qtensors: Format-name → QuantizedTensor for each group's weights,
            stored as ``[n_channels, in_features]`` (via ``original_shape``).
        splits: Format-name → number of output channels in that group.
        inv_perm: ``[out_features]`` inverse-permutation to restore original
            channel order after concatenation.
        bias: Optional bias ``[out_features]`` in original channel order.

    Returns:
        ``[..., out_features]`` output tensor.
    """
    parts: list[torch.Tensor] = []

    for fmt, qt in qtensors.items():
        n_ch = splits[fmt]
        if n_ch == 0:
            continue
        # Dequantize this group's weight block  →  [n_ch, in_features]
        w_group = dequantize(qt).to(x.dtype)
        # GEMM for this group  →  [..., n_ch]
        parts.append(F.linear(x, w_group))

    # Concat along output dim (permuted order) → [..., out_features]
    y_permuted = torch.cat(parts, dim=-1)

    # Restore original channel order (clone inv_perm to escape inference_mode
    # so autograd can save it for backward)
    y = y_permuted.index_select(-1, inv_perm.to(y_permuted.device).clone())

    if bias is not None:
        y = y + bias

    return y


# ──────────────────────────────────────────────────────────────────────────────
#  Backend: vLLM native kernels (Step 2 — stub)
# ──────────────────────────────────────────────────────────────────────────────

def mixed_precision_linear_vllm(
    x: torch.Tensor,
    qtensors: dict[str, QuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Same interface as :func:`mixed_precision_linear`, but uses vLLM's
    native CUDA kernels.  Requires Blackwell GPU + vLLM installed.

    Kernel mapping::

        NVFP4 → torch.ops._C.nvfp4_gemm(x, w_packed, scales, global_scale)
        MXFP8 → torch.ops._C.cutlass_scaled_mm(x_fp8, w_fp8.T, ...)
        FP16  → F.linear(x, w_fp16)

    Falls back to :func:`mixed_precision_linear` if vLLM is not available.
    """
    # TODO: implement when vLLM kernels are available
    # For now, fall back to pure-PyTorch path
    return mixed_precision_linear(x, qtensors, splits, inv_perm, bias)
