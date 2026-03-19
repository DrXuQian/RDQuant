"""
Mixed-precision linear layer for vLLM inference.

NVFP4/FP8/FP16 GEMM per format group:
  y_nvfp4 = marlin_gemm(x_fp16, w_nvfp4, block_scales, global_scale)  # float4_e2m1f
  y_fp8   = marlin_gemm(x_fp16, w_fp8, channel_scales, ...)           # float8_e4m3fn
  y_fp16  = F.linear(x_fp16, w_fp16)                                  # passthrough
  y = cat([y_nvfp4, y_fp8, y_fp16])[inv_perm]

Activations remain in FP16 (no activation quantization).
When vLLM is not available, falls back to fake-quant dequantize + F.linear.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# vLLM availability check
# ---------------------------------------------------------------------------

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


class RDQuantLinear(nn.Module):
    """Mixed-precision linear layer for vLLM inference.

    Stores per-format weight tensors and runs one GEMM per format group,
    then reassembles output in original channel order via inverse permutation.

    Supports two construction paths:
      1. from_quantized() -- from a QuantizedWeight (fake-quant fallback)
      2. from_packed_checkpoint() -- from packed checkpoint tensors (Marlin kernels)

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        quantized_data: Optional QuantizedWeight for fake-quant path.
        bias: Optional bias tensor.
        inv_perm: Inverse permutation tensor [out_features].
        splits: Dict format_name -> n_channels.
        marlin_data: Optional dict with pre-packed Marlin tensors.
        w_fp16: Optional BF16/FP16 weight for FP16 group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantized_data=None,
        bias: Optional[torch.Tensor] = None,
        inv_perm: Optional[torch.Tensor] = None,
        splits: Optional[dict[str, int]] = None,
        marlin_data: Optional[dict] = None,
        w_fp16: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized_data = quantized_data

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

        # For Marlin-based forward
        if inv_perm is not None:
            self.register_buffer("inv_perm", inv_perm)
        else:
            self.inv_perm = None

        self.splits = splits or {}
        self.marlin_data = marlin_data
        if w_fp16 is not None:
            self.register_buffer("w_fp16", w_fp16)
        else:
            self.w_fp16 = None

    @classmethod
    def from_quantized(cls, config: dict, weight_data) -> "RDQuantLinear":
        """Construct from a config dict and QuantizedWeight (fake-quant path).

        Args:
            config: Dict with keys ``in_features``, ``out_features``,
                and optionally ``bias`` (a tensor or None).
            weight_data: A QuantizedWeight instance.

        Returns:
            Constructed RDQuantLinear.
        """
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            quantized_data=weight_data,
            bias=config.get("bias", None),
        )

    @classmethod
    def from_packed_checkpoint(
        cls,
        layer_data: dict[str, torch.Tensor],
        layer_config: dict,
        device: str = "cuda",
    ) -> "RDQuantLinear":
        """Load from packed checkpoint data and optionally repack for Marlin.

        Args:
            layer_data: Dict of tensors for this layer (from safetensors).
            layer_config: Layer config from quantization_config.json.
            device: Target device.

        Returns:
            Constructed RDQuantLinear with Marlin data if vLLM available,
            otherwise with QuantizedWeight for fake-quant fallback.
        """
        from rdquant.core.formats import QuantizedTensor
        from rdquant.quantize import QuantizedWeight

        n_in = layer_config["in_features"]
        n_out = layer_config["out_features"]
        splits = layer_config["splits"]
        prefix = ""  # tensors are already keyed without prefix

        inv_perm = layer_data["inv_permutation"].to(torch.int64).to(device)
        perm = torch.empty_like(inv_perm)
        perm[inv_perm] = torch.arange(n_out, dtype=torch.int64, device=device)

        n_nvfp4 = splits.get("NVFP4", 0)
        n_fp8 = splits.get("FP8", 0)
        n_fp16 = splits.get("FP16", 0)

        # Try Marlin path first
        use_marlin = _check_vllm() and torch.cuda.is_available()
        marlin_data = None
        w_fp16_buf = None

        if use_marlin:
            marlin_data = _build_marlin_data(layer_data, splits, n_in, device)
            # FP16 group as BF16 for F.linear
            if n_fp16 > 0 and "weight_fp16" in layer_data:
                w_fp16_buf = layer_data["weight_fp16"].to(torch.float16).to(device)
        else:
            # Fake-quant fallback: reconstruct QuantizedWeight
            pass

        # Always reconstruct QuantizedWeight for fallback
        qtensors = _reconstruct_qtensors(layer_data, splits, n_in, device)

        qw = QuantizedWeight(
            qtensors=qtensors,
            permutation=perm,
            inv_permutation=inv_perm,
            splits={f: splits.get(f, 0) for f in ["NVFP4", "FP8", "FP16"]},
            original_shape=torch.Size([n_out, n_in]),
            avg_bits=layer_config.get("avg_bits", 0.0),
        )

        bias = layer_data.get("bias", None)
        if bias is not None:
            bias = bias.to(device)

        return cls(
            in_features=n_in,
            out_features=n_out,
            quantized_data=qw,
            bias=bias,
            inv_perm=inv_perm,
            splits=splits,
            marlin_data=marlin_data,
            w_fp16=w_fp16_buf,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Uses Marlin kernels if available and data is pre-packed,
        otherwise falls back to dequant + F.linear.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        if self.marlin_data is not None and _check_vllm():
            return self._forward_marlin(x)
        elif self.quantized_data is not None:
            return self._forward_fakequant(x)
        else:
            raise RuntimeError("No weight data available for forward pass")

    def _forward_fakequant(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quant forward: dequantize all groups + F.linear."""
        weight = self.quantized_data.dequantize()
        return F.linear(x, weight, self.bias)

    def _forward_marlin(self, x: torch.Tensor) -> torch.Tensor:
        """Marlin kernel forward: per-format GEMM -> concat -> inv_perm."""
        import vllm._custom_ops as ops
        from vllm.scalar_type import scalar_types

        parts = []
        md = self.marlin_data

        n_nvfp4 = self.splits.get("NVFP4", 0)
        n_fp8 = self.splits.get("FP8", 0)
        n_fp16 = self.splits.get("FP16", 0)

        x_2d = x.reshape(-1, x.shape[-1])
        M = x_2d.shape[0]
        x_half = x_2d.half()

        if n_nvfp4 > 0 and "nvfp4_qweight" in md:
            y = ops.marlin_gemm(
                a=x_half,
                c=None,
                b_q_weight=md["nvfp4_qweight"],
                b_bias=None,
                b_scales=md["nvfp4_scales"],
                a_scales=None,
                global_scale=md["nvfp4_global_scale"],
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=md["workspace"],
                b_q_type=scalar_types.float4_e2m1f,
                size_m=M,
                size_n=n_nvfp4,
                size_k=self.in_features,
            )
            parts.append(y.to(x.dtype))

        if n_fp8 > 0 and "fp8_qweight" in md:
            y = ops.marlin_gemm(
                a=x_half,
                c=None,
                b_q_weight=md["fp8_qweight"],
                b_bias=None,
                b_scales=md["fp8_scales"],
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=md["workspace"],
                b_q_type=scalar_types.float8_e4m3fn,
                size_m=M,
                size_n=n_fp8,
                size_k=self.in_features,
            )
            parts.append(y.to(x.dtype))

        if n_fp16 > 0 and self.w_fp16 is not None:
            parts.append(F.linear(x, self.w_fp16))

        y_permuted = torch.cat(parts, dim=-1)
        y = y_permuted.index_select(-1, self.inv_perm)

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        avg_bits = 0.0
        if self.quantized_data is not None:
            avg_bits = self.quantized_data.avg_bits
        backend = "marlin" if self.marlin_data is not None else "fakequant"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"avg_bits={avg_bits:.2f}, backend={backend}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reconstruct_qtensors(
    layer_data: dict[str, torch.Tensor],
    splits: dict[str, int],
    n_in: int,
    device: str,
) -> dict:
    """Reconstruct QuantizedTensor objects from packed checkpoint data."""
    from rdquant.core.formats import QuantizedTensor

    qtensors = {}
    n_nvfp4 = splits.get("NVFP4", 0)
    n_fp8 = splits.get("FP8", 0)
    n_fp16 = splits.get("FP16", 0)

    if n_nvfp4 > 0 and "weight_nvfp4" in layer_data:
        packed = layer_data["weight_nvfp4"].to(device)
        even = (packed & 0x0F).to(torch.int64)
        odd = ((packed >> 4) & 0x0F).to(torch.int64)
        n_ch = packed.shape[0]
        half_k = packed.shape[1]
        indices_2d = torch.empty(n_ch, half_k * 2, dtype=torch.int64, device=device)
        indices_2d[:, 0::2] = even
        indices_2d[:, 1::2] = odd

        scales_uint8 = layer_data["weight_nvfp4_scale"].to(device)
        scales_fp8 = scales_uint8.view(torch.float8_e4m3fn)
        scales_flat = scales_fp8.to(torch.float32).reshape(-1)

        global_scale = layer_data["nvfp4_global_scale"].item()

        qtensors["NVFP4"] = QuantizedTensor(
            data=indices_2d.reshape(-1),
            scales=scales_flat,
            format_name="NVFP4",
            original_shape=torch.Size([n_nvfp4, n_in]),
            bits_per_element=4,
            global_scale=global_scale,
        )

    if n_fp8 > 0 and "weight_fp8" in layer_data:
        data_uint8 = layer_data["weight_fp8"].to(device)
        data_fp8 = data_uint8.view(torch.float8_e4m3fn)
        data_f32 = data_fp8.to(torch.float32).reshape(-1)

        scale_vec = layer_data["weight_fp8_scale"].to(device)

        qtensors["FP8"] = QuantizedTensor(
            data=data_f32,
            scales=scale_vec[0:1],
            format_name="FP8",
            original_shape=torch.Size([n_fp8, n_in]),
            bits_per_element=8,
        )

    if n_fp16 > 0 and "weight_fp16" in layer_data:
        data_bf16 = layer_data["weight_fp16"].to(device)
        data_f16 = data_bf16.to(torch.float16).reshape(-1)

        qtensors["FP16"] = QuantizedTensor(
            data=data_f16,
            scales=torch.tensor([], dtype=torch.float32, device=device),
            format_name="FP16",
            original_shape=torch.Size([n_fp16, n_in]),
            bits_per_element=16,
        )

    return qtensors


def _build_marlin_data(
    layer_data: dict[str, torch.Tensor],
    splits: dict[str, int],
    n_in: int,
    device: str,
) -> dict | None:
    """Build Marlin-repacked weight tensors for vLLM kernels.

    Returns None if repacking fails (e.g., dimension constraints not met).
    """
    try:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales,
            nvfp4_marlin_process_global_scale,
        )
    except ImportError:
        return None

    md: dict = {}
    n_nvfp4 = splits.get("NVFP4", 0)
    n_fp8 = splits.get("FP8", 0)

    try:
        workspace = marlin_make_workspace_new(torch.device(device))
        md["workspace"] = workspace

        if n_nvfp4 > 0 and "weight_nvfp4" in layer_data:
            packed = layer_data["weight_nvfp4"].to(device)  # [n_ch, n_in//2] uint8
            # Repack for Marlin: view as int32, transpose
            qweight_int32 = packed.view(torch.int32).T.contiguous()
            perm = torch.empty(0, dtype=torch.int, device=device)

            marlin_qw = ops.gptq_marlin_repack(
                b_q_weight=qweight_int32,
                perm=perm,
                size_k=n_in,
                size_n=n_nvfp4,
                num_bits=4,
                is_a_8bit=False,
            )
            md["nvfp4_qweight"] = marlin_qw

            # Scales
            scales_uint8 = layer_data["weight_nvfp4_scale"].to(device)
            scales_fp8 = scales_uint8.view(torch.float8_e4m3fn)
            blocks_per_row = n_in // 16
            scales_2d = scales_fp8.to(torch.float16).reshape(n_nvfp4, blocks_per_row)
            scales_t = scales_2d.T.contiguous()

            marlin_sc = marlin_permute_scales(
                s=scales_t,
                size_k=n_in,
                size_n=n_nvfp4,
                group_size=16,
                is_a_8bit=False,
            )
            marlin_sc = nvfp4_marlin_process_scales(marlin_sc)
            md["nvfp4_scales"] = marlin_sc

            # Global scale
            gs = layer_data["nvfp4_global_scale"].to(torch.float16).to(device)
            md["nvfp4_global_scale"] = nvfp4_marlin_process_global_scale(gs)

        if n_fp8 > 0 and "weight_fp8" in layer_data:
            data_uint8 = layer_data["weight_fp8"].to(device)  # [n_ch, n_in] uint8 (fp8 view)
            data_fp8 = data_uint8.view(torch.float8_e4m3fn)

            # Pack FP8 into int32 for gptq_marlin_repack
            packed_int32 = data_uint8.view(torch.int32).T.contiguous()
            perm = torch.empty(0, dtype=torch.int, device=device)

            marlin_qw = ops.gptq_marlin_repack(
                b_q_weight=packed_int32,
                perm=perm,
                size_k=n_in,
                size_n=n_fp8,
                num_bits=8,
                is_a_8bit=True,
            )
            md["fp8_qweight"] = marlin_qw

            # Scales: per-channel -> [1, n_fp8] for marlin
            scale_vec = layer_data["weight_fp8_scale"].to(torch.float16).to(device)
            scale_2d = scale_vec.unsqueeze(0)  # [1, n_fp8]

            marlin_sc = marlin_permute_scales(
                s=scale_2d,
                size_k=n_in,
                size_n=n_fp8,
                group_size=-1,  # per-channel
                is_a_8bit=True,
            )
            md["fp8_scales"] = marlin_sc

    except Exception:
        return None

    return md if len(md) > 1 else None  # must have workspace + at least one format
