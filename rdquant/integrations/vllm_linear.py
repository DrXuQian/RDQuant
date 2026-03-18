"""
Mixed-precision linear layer for vLLM inference.

MXFP8 x MXFPx GEMM per format group:
  y_mxfp4 = mxfp8_x_mxfp4_gemm(x_mxfp8, w_mxfp4, scales)
  y_mxfp6 = mxfp8_x_mxfp6_gemm(x_mxfp8, w_mxfp6, scales)
  y_mxfp8 = mxfp8_x_mxfp8_gemm(x_mxfp8, w_mxfp8, scales)
  y = cat([y_mxfp4, y_mxfp6, y_mxfp8])[inv_perm]

Activations are quantized once to MXFP8, then each group performs
an MXFP8 x MXFPx GEMM.
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
    because real MX GEMM kernels require custom CUDA extensions. Once
    vLLM's MXFP8 x MXFPx grouped GEMM kernels are available this class
    will be updated to use them.

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
        In a future version this will dispatch to per-format vLLM MXFP8 x MXFPx kernels.

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
