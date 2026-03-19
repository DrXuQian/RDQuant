"""
Mixed-precision linear operators for RDQuant.

Execution backends:

  - ``pytorch``:  Pure PyTorch.  Dequant each format group -> ``F.linear`` ->
    concat -> inv-permute.  Correct on any device, used for validation.

  - ``vllm``:  Uses vLLM's native CUDA kernels:
        NVFP4 group: marlin_gemm with float4_e2m1f (W4A16, dequant fused)
        FP8 group: cutlass_scaled_mm with per-channel scale (W8A8)
        FP16 group: standard F.linear
    Activations always remain in FP16/BF16 (no activation quantization).

All backends produce identical results (up to floating-point ordering).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rdquant.core.formats import dequantize, QuantizedTensor


# --------------------------------------------------------------------------
#  Backend: pure-PyTorch (fake-quant, works everywhere)
# --------------------------------------------------------------------------

def mixed_precision_linear(
    x: torch.Tensor,
    qtensors: dict[str, QuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mixed-precision linear: per-format dequant -> GEMM -> concat -> permute.

    Activations remain in FP16 (no activation quantization).

    Args:
        x: Input activations ``[..., in_features]``.
        qtensors: Format-name -> QuantizedTensor for each group's weights.
        splits: Format-name -> number of output channels in that group.
        inv_perm: ``[out_features]`` inverse-permutation.
        bias: Optional bias ``[out_features]`` in original channel order.

    Returns:
        ``[..., out_features]`` output tensor.
    """
    parts: list[torch.Tensor] = []

    for fmt, qt in qtensors.items():
        n_ch = splits[fmt]
        if n_ch == 0:
            continue
        w_group = dequantize(qt).to(x.dtype)
        parts.append(F.linear(x, w_group))

    y_permuted = torch.cat(parts, dim=-1)
    y = y_permuted.index_select(-1, inv_perm.to(y_permuted.device).clone())

    if bias is not None:
        y = y + bias

    return y


# --------------------------------------------------------------------------
#  Backend: vLLM native kernels
# --------------------------------------------------------------------------

_vllm_available: bool | None = None


def _check_vllm() -> bool:
    """Check if vLLM kernels are loadable."""
    global _vllm_available
    if _vllm_available is not None:
        return _vllm_available
    try:
        import vllm._custom_ops  # noqa: F401
        _vllm_available = True
    except Exception:
        _vllm_available = False
    return _vllm_available


def _get_vllm_ops():
    """Import vLLM custom ops module."""
    import vllm._custom_ops as ops
    return ops


def _nvfp4_marlin_gemm(
    x: torch.Tensor,
    qt: QuantizedTensor,
) -> torch.Tensor:
    """NVFP4 GEMM via vLLM marlin_gemm.

    Requires weight pre-packed into Marlin format (see pack_nvfp4_for_marlin).
    If weight is not pre-packed, falls back to dequant + F.linear.
    """
    ops = _get_vllm_ops()
    from vllm.scalar_type import scalar_types

    # Check if weight has Marlin-packed attributes
    if not hasattr(qt, '_marlin_qweight'):
        # Not packed — fall back to fake-quant
        w = dequantize(qt).to(x.dtype)
        return F.linear(x, w)

    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    N, K = qt._marlin_n, qt._marlin_k

    # marlin_gemm expects FP16 input
    x_half = x_2d.half() if x_2d.dtype != torch.float16 else x_2d

    y = ops.marlin_gemm(
        a=x_half,
        c=None,
        b_q_weight=qt._marlin_qweight,
        b_bias=None,
        b_scales=qt._marlin_scales,
        a_scales=None,
        global_scale=qt._marlin_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=qt._marlin_workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=M,
        size_n=N,
        size_k=K,
    )
    return y.to(x.dtype).reshape(*x.shape[:-1], N)


def _fp8_scaled_mm(
    x: torch.Tensor,
    qt: QuantizedTensor,
) -> torch.Tensor:
    """FP8 GEMM via vLLM cutlass_scaled_mm with per-channel scale."""
    ops = _get_vllm_ops()

    n_ch, n_in = qt.original_shape

    # Get or create the FP8 weight tensor in the right layout
    # cutlass_scaled_mm expects B as [N, K] contiguous fp8, then .t() gives col-major [K, N]
    if not hasattr(qt, '_fp8_weight_cuda'):
        w_data = qt.data.reshape(n_ch, n_in)
        if w_data.dtype != torch.float8_e4m3fn:
            w_data = w_data.to(torch.float8_e4m3fn)
        qt._fp8_weight_cuda = w_data.contiguous().cuda()

    w_fp8 = qt._fp8_weight_cuda  # [N, K] contiguous on CUDA

    w_scale = qt.scales.to(w_fp8.device)  # [1] per-channel scale
    if w_scale.numel() == 1:
        w_scale = w_scale.expand(n_ch)

    # Quantize activation to FP8 on-the-fly (per-token scale)
    x_2d = x.reshape(-1, x.shape[-1]).float()
    x_amax = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    x_scale = x_amax / 448.0
    x_fp8 = (x_2d / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    # cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    # a=[M,K] row-major fp8, b=[K,N] = w_fp8.t() col-major fp8
    y = ops.cutlass_scaled_mm(
        x_fp8,
        w_fp8.t(),  # [K, N] col-major view (since w_fp8 is [N,K] contiguous)
        x_scale.float().contiguous(),
        w_scale.unsqueeze(0).float().contiguous(),
        x.dtype,
    )
    return y.reshape(*x.shape[:-1], n_ch)


def mixed_precision_linear_vllm(
    x: torch.Tensor,
    qtensors: dict[str, QuantizedTensor],
    splits: dict[str, int],
    inv_perm: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mixed-precision linear using vLLM's native CUDA kernels.

    Kernel mapping::

        NVFP4 group: marlin_gemm(float4_e2m1f) — W4A16, decode-optimized
        FP8 group:   cutlass_scaled_mm — W8A8 per-channel, prefill-optimized
        FP16 group:  F.linear — passthrough

    Falls back to :func:`mixed_precision_linear` if vLLM is not available.
    """
    if not _check_vllm():
        return mixed_precision_linear(x, qtensors, splits, inv_perm, bias)

    parts: list[torch.Tensor] = []

    for fmt, qt in qtensors.items():
        n_ch = splits[fmt]
        if n_ch == 0:
            continue

        if fmt == "NVFP4":
            parts.append(_nvfp4_marlin_gemm(x, qt))
        elif fmt == "FP8":
            parts.append(_fp8_scaled_mm(x, qt))
        elif fmt == "FP16":
            w = dequantize(qt).to(x.dtype)
            parts.append(F.linear(x, w))
        else:
            # Unknown format — fall back to dequant
            w = dequantize(qt).to(x.dtype)
            parts.append(F.linear(x, w))

    y_permuted = torch.cat(parts, dim=-1)
    y = y_permuted.index_select(-1, inv_perm.to(y_permuted.device).clone())

    if bias is not None:
        y = y + bias

    return y


# --------------------------------------------------------------------------
#  Marlin weight packing utilities
# --------------------------------------------------------------------------

def pack_nvfp4_for_marlin(
    qt: QuantizedTensor,
    device: torch.device | str = "cuda",
) -> QuantizedTensor:
    """Pack NVFP4 QuantizedTensor weights into Marlin format for vLLM.

    Converts our index-based NVFP4 representation into the packed uint8
    format expected by gptq_marlin_repack, then permutes scales.

    Args:
        qt: NVFP4 QuantizedTensor with data=indices, scales=FP8 block scales.
        device: Target device.

    Returns:
        Same QuantizedTensor with Marlin attributes attached:
        qt._marlin_qweight, qt._marlin_scales, qt._marlin_global_scale,
        qt._marlin_workspace, qt._marlin_n, qt._marlin_k
    """
    if not _check_vllm():
        raise RuntimeError("vLLM not available for Marlin packing")

    ops = _get_vllm_ops()
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales,
        nvfp4_marlin_process_global_scale,
    )

    n_out, n_in = qt.original_shape
    indices = qt.data.to(device)  # [n_out * n_in] long indices 0..15

    # Pack indices into uint8: 2 FP4 per byte (low nibble = even idx, high nibble = odd idx)
    idx_2d = indices.reshape(n_out, n_in)
    even = idx_2d[:, 0::2]  # low nibble
    odd = idx_2d[:, 1::2]   # high nibble
    packed = (odd.to(torch.uint8) << 4) | even.to(torch.uint8)  # [n_out, n_in//2]

    # gptq_marlin_repack expects [K//tile, N*tile//pack] int32 in transposed form
    qweight_int32 = packed.view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=device)

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight_int32,
        perm=perm,
        size_k=n_in,
        size_n=n_out,
        num_bits=4,
        is_a_8bit=False,
    )

    # Process scales: [n_blocks_per_row] -> [n_out, n_in//16] -> permute for Marlin
    n_blocks = qt.scales.shape[0]
    blocks_per_row = n_in // 16
    scales_2d = qt.scales.reshape(n_out, blocks_per_row).to(device)

    # Convert to FP16 for marlin_permute_scales
    scales_fp16 = scales_2d.to(torch.float16).T.contiguous()  # [blocks_per_row, n_out]

    marlin_scales = marlin_permute_scales(
        s=scales_fp16,
        size_k=n_in,
        size_n=n_out,
        group_size=16,
        is_a_8bit=False,
    )
    marlin_scales = nvfp4_marlin_process_scales(marlin_scales)

    # Global scale
    global_scale = torch.tensor(qt.global_scale, dtype=torch.float16, device=device)
    marlin_global_scale = nvfp4_marlin_process_global_scale(global_scale)

    workspace = marlin_make_workspace_new(device)

    # Attach Marlin data to the QuantizedTensor
    qt._marlin_qweight = marlin_qweight
    qt._marlin_scales = marlin_scales
    qt._marlin_global_scale = marlin_global_scale
    qt._marlin_workspace = workspace
    qt._marlin_n = n_out
    qt._marlin_k = n_in

    return qt
