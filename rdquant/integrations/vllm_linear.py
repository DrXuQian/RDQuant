"""
Mixed-precision linear layer for vLLM inference.

NVFP4/FP8/FP16 GEMM per format group:
  y_nvfp4 = marlin_gemm(x_fp16, w_nvfp4, block_scales, global_scale)  # float4_e2m1f
  y_fp8   = cutlass_scaled_mm(x_fp16, w_fp8, channel_scales)          # per-channel FP8
  y_fp16  = F.linear(x_fp16, w_fp16)                                  # passthrough
  y = cat([y_nvfp4, y_fp8, y_fp16])[inv_perm]

Activations remain in FP16 (no activation quantization).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDQuantLinear(nn.Module):
    """Mixed-precision linear layer for vLLM inference.

    Stores per-format packed weight tensors and reconstructs the output by
    running one GEMM per format group, then reassembling in original channel
    order via the inverse permutation.

    Currently falls back to fake-quantization (dequantize then FP32 GEMM)
    because real vLLM kernels require custom CUDA extensions. Once
    vLLM's NVFP4 marlin_gemm and FP8 cutlass_scaled_mm kernels are
    available this class will be updated to use them.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        quantized_data: A :class:`~rdquant.quantize.QuantizedWeight` instance.
        bias: Optional bias tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantized_data,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized_data = quantized_data
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    @classmethod
    def from_quantized(cls, config: dict, weight_data) -> "RDQuantLinear":
        """Construct an RDQuantLinear from a config dict and QuantizedWeight.

        Args:
            config: Dict with keys ``in_features``, ``out_features``,
                and optionally ``bias`` (a tensor or ``None``).
            weight_data: A :class:`~rdquant.quantize.QuantizedWeight` instance.

        Returns:
            Constructed :class:`RDQuantLinear`.
        """
        in_features = config["in_features"]
        out_features = config["out_features"]
        bias = config.get("bias", None)
        return cls(
            in_features=in_features,
            out_features=out_features,
            quantized_data=weight_data,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Stub: dequantizes the weight and uses a standard FP32 GEMM.
        In a future version this will dispatch to per-format vLLM kernels:
          - NVFP4: marlin_gemm with float4_e2m1f
          - FP8: cutlass_scaled_mm with per-channel scale
          - FP16: standard F.linear

        Args:
            x: Input tensor of shape ``[..., in_features]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        weight = self.quantized_data.dequantize()  # [out_features, in_features]
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"avg_bits={self.quantized_data.avg_bits:.2f}"
        )
