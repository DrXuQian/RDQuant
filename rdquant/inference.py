"""
High-level inference and quantization-export helpers.

Provides two one-click interfaces:

  - :func:`load_for_inference` — load a packed RDQuant checkpoint with real
    Marlin kernels (or fake-quant fallback) for fast inference.
  - :func:`quantize_and_export` — quantize a HuggingFace model end-to-end
    and save as a packed checkpoint.

Also defines :class:`MarlinMixedLinear`, a drop-in ``nn.Module`` that runs
two Marlin GEMMs (NVFP4 + FP8) and concatenates/permutes the results.

vLLM dependency is **conditional**: Marlin mode requires vLLM custom ops,
but fake-quant mode works without it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  vLLM availability
# ---------------------------------------------------------------------------

_vllm_available: bool | None = None


def _ensure_vllm_path() -> None:
    """Add the vLLM site-packages directory to sys.path if present."""
    vllm_site = "/root/autodl-tmp/vllm_site"
    if os.path.isdir(vllm_site) and vllm_site not in sys.path:
        sys.path.insert(0, vllm_site)


def _check_vllm() -> bool:
    global _vllm_available
    if _vllm_available is not None:
        return _vllm_available
    _ensure_vllm_path()
    try:
        import vllm._custom_ops  # noqa: F401
        _vllm_available = True
    except Exception:
        _vllm_available = False
    return _vllm_available


# ---------------------------------------------------------------------------
#  MarlinMixedLinear — real Marlin kernel execution
# ---------------------------------------------------------------------------

class MarlinMixedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using Marlin kernels.

    Internally holds two Marlin-repacked weight matrices (NVFP4 and FP8)
    and executes two ``marlin_gemm`` / ``marlin_quant_fp8`` calls whose
    outputs are concatenated and permuted back to original channel order.

    Args:
        w_nvfp4_fp16: Dequantized NVFP4 group weights ``[n_nvfp4, K]`` in FP16.
        w_fp8_fp16: Dequantized FP8 group weights ``[n_fp8, K]`` in FP16.
        w_fp16_fp16: FP16 group weights ``[n_fp16, K]`` in FP16 (pass-through).
        inv_perm: ``[N_out]`` inverse permutation tensor.
        bias: Optional bias ``[N_out]``.
        raw_nvfp4_scales: Per-block absmax/6.0 scales ``[n_nvfp4, K//16]`` float32.
            Used for Marlin global_scale normalisation.
        nvfp4_packed_indices: Nibble-packed uint8 ``[n_nvfp4, K//2]``. If provided,
            skips re-quantization and packs directly.
    """

    def __init__(
        self,
        w_nvfp4_fp16: Optional[torch.Tensor],
        w_fp8_fp16: Optional[torch.Tensor],
        w_fp16_fp16: Optional[torch.Tensor],
        inv_perm: torch.Tensor,
        bias: Optional[torch.Tensor],
        raw_nvfp4_scales: Optional[torch.Tensor] = None,
        nvfp4_packed_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        _ensure_vllm_path()
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales,
            nvfp4_marlin_process_global_scale,
        )
        from vllm.scalar_type import scalar_types

        device = inv_perm.device

        self.n_nvfp4 = w_nvfp4_fp16.shape[0] if w_nvfp4_fp16 is not None else 0
        self.n_fp8 = w_fp8_fp16.shape[0] if w_fp8_fp16 is not None else 0
        self.n_fp16 = w_fp16_fp16.shape[0] if w_fp16_fp16 is not None else 0
        self.N = self.n_nvfp4 + self.n_fp8 + self.n_fp16

        if w_nvfp4_fp16 is not None:
            K = w_nvfp4_fp16.shape[1]
        elif w_fp8_fp16 is not None:
            K = w_fp8_fp16.shape[1]
        elif w_fp16_fp16 is not None:
            K = w_fp16_fp16.shape[1]
        else:
            raise ValueError("At least one weight group must be provided")
        self.K = K

        self.register_buffer("inv_perm", inv_perm.to(torch.int64))

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

        self._scalar_type_nvfp4 = scalar_types.float4_e2m1f

        # --- NVFP4 Marlin repack ---
        if self.n_nvfp4 > 0 and w_nvfp4_fp16 is not None:
            N_nvfp4 = self.n_nvfp4

            if nvfp4_packed_indices is not None and raw_nvfp4_scales is not None:
                # Use pre-packed indices directly
                packed = nvfp4_packed_indices.to(device)  # [N_nvfp4, K//2] uint8
            else:
                # Re-quantize from dequantized FP16 weights
                # Quantize to NVFP4 indices
                from rdquant.core.formats import _NVFP4_LUT, _NVFP4_POS_VALUES, _NVFP4_BLOCK_SIZE
                w = w_nvfp4_fp16.float().to(device)
                blocks = w.reshape(-1, _NVFP4_BLOCK_SIZE)
                absmax = blocks.abs().amax(dim=1)
                raw_scale = absmax / 6.0
                safe_scale = raw_scale.clone()
                safe_scale[safe_scale == 0] = 1.0
                normalized = blocks / safe_scale.unsqueeze(1)
                norm_flat = normalized.reshape(-1)
                sign = (norm_flat < 0).long()
                abs_norm = norm_flat.abs()
                pos_vals = _NVFP4_POS_VALUES.to(device)
                dists = (abs_norm.unsqueeze(1) - pos_vals).abs()
                mag_idx = dists.argmin(dim=1)
                indices = mag_idx + sign * 8
                indices_2d = indices.reshape(N_nvfp4, K)
                even = indices_2d[:, 0::2].to(torch.uint8)
                odd = indices_2d[:, 1::2].to(torch.uint8)
                packed = (odd << 4) | even
                raw_nvfp4_scales = raw_scale.reshape(N_nvfp4, K // 16)

            # gptq_marlin_repack
            qweight_int32 = packed.view(torch.int32).T.contiguous()
            perm_empty = torch.empty(0, dtype=torch.int, device=device)
            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight_int32,
                perm=perm_empty,
                size_k=K,
                size_n=N_nvfp4,
                num_bits=4,
                is_a_8bit=False,
            )
            self.register_buffer("_nvfp4_qweight", marlin_qweight)

            # Process scales with global_scale normalisation
            scales_2d = raw_nvfp4_scales.float().to(device)  # [N_nvfp4, K//16]
            global_scale_val = scales_2d.max().item()
            if global_scale_val == 0:
                global_scale_val = 1.0
            global_scale_for_norm = global_scale_val / 448.0
            block_scales_fp8 = (scales_2d / global_scale_for_norm).clamp(max=448.0).to(torch.float8_e4m3fn)
            block_scales_fp16 = block_scales_fp8.to(torch.float16)

            # marlin_permute_scales expects [K//group_size, N] layout
            scales_T = block_scales_fp16.T.contiguous()  # [K//16, N_nvfp4]
            marlin_scales = marlin_permute_scales(
                s=scales_T,
                size_k=K,
                size_n=N_nvfp4,
                group_size=16,
                is_a_8bit=False,
            )
            marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            self.register_buffer("_nvfp4_scales", marlin_scales)

            global_scale_tensor = torch.tensor(global_scale_for_norm, dtype=torch.float16, device=device)
            marlin_global_scale = nvfp4_marlin_process_global_scale(global_scale_tensor)
            self.register_buffer("_nvfp4_global_scale", marlin_global_scale)

            self.register_buffer("_nvfp4_workspace", marlin_make_workspace_new(device))

        # --- FP8 Marlin repack ---
        if self.n_fp8 > 0 and w_fp8_fp16 is not None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                marlin_quant_fp8_torch,
            )
            w = w_fp8_fp16.half().to(device)
            # Returns (weight_ref_T, marlin_qweight, marlin_scales)
            _, fp8_qw, fp8_scales = marlin_quant_fp8_torch(w, group_size=-1)
            self.register_buffer("_fp8_qweight", fp8_qw)
            self.register_buffer("_fp8_scales", fp8_scales)
            self.register_buffer("_fp8_workspace", marlin_make_workspace_new(device))

        # --- FP16 passthrough ---
        if self.n_fp16 > 0 and w_fp16_fp16 is not None:
            self.register_buffer("_fp16_weight", w_fp16_fp16.half().to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run mixed-precision Marlin forward.

        Args:
            x: Input ``[..., K]``.

        Returns:
            Output ``[..., N_out]`` in original channel order.
        """
        _ensure_vllm_path()
        import vllm._custom_ops as ops

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K)
        M = x_2d.shape[0]
        x_half = x_2d.half() if x_2d.dtype != torch.float16 else x_2d

        parts = []

        # NVFP4 group
        if self.n_nvfp4 > 0:
            y_nvfp4 = ops.marlin_gemm(
                a=x_half,
                c=None,
                b_q_weight=self._nvfp4_qweight,
                b_bias=None,
                b_scales=self._nvfp4_scales,
                a_scales=None,
                global_scale=self._nvfp4_global_scale,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=self._nvfp4_workspace,
                b_q_type=self._scalar_type_nvfp4,
                size_m=M,
                size_n=self.n_nvfp4,
                size_k=self.K,
            )
            parts.append(y_nvfp4)

        # FP8 group
        if self.n_fp8 > 0:
            from vllm.scalar_type import scalar_types
            y_fp8 = ops.marlin_gemm(
                a=x_half,
                c=None,
                b_q_weight=self._fp8_qweight,
                b_bias=None,
                b_scales=self._fp8_scales,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=self._fp8_workspace,
                b_q_type=scalar_types.float8_e4m3fn,
                size_m=M,
                size_n=self.n_fp8,
                size_k=self.K,
            )
            parts.append(y_fp8)

        # FP16 group
        if self.n_fp16 > 0:
            y_fp16 = F.linear(x_half, self._fp16_weight)
            parts.append(y_fp16)

        y_permuted = torch.cat(parts, dim=-1)  # [M, N_out] in permuted order
        y = y_permuted.index_select(-1, self.inv_perm)

        if self.bias is not None:
            y = y + self.bias

        y = y.to(x.dtype)
        return y.reshape(*orig_shape[:-1], self.N)


# ---------------------------------------------------------------------------
#  load_for_inference
# ---------------------------------------------------------------------------

def load_for_inference(
    checkpoint_dir: str,
    model_class=None,
    device: str = "cuda",
    use_marlin: bool = True,
) -> nn.Module:
    """Load a packed RDQuant checkpoint for inference.

    Returns a HuggingFace model with all quantized layers replaced by
    :class:`MarlinMixedLinear` (real Marlin kernels) or materialised
    fake-quant ``nn.Linear``.

    Args:
        checkpoint_dir: Path to packed checkpoint directory containing
            ``model.safetensors``, ``quantization_config.json``, and
            ``config.json``.
        model_class: HuggingFace model class. Defaults to
            ``AutoModelForCausalLM``.
        device: Target device (``"cuda"`` or ``"cpu"``).
        use_marlin: If True and CUDA is available, replace quantized
            layers with :class:`MarlinMixedLinear`.  If False, materialise
            dequantized weights into plain ``nn.Linear``.

    Returns:
        The HuggingFace model ready for inference.
    """
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoConfig

    if model_class is None:
        model_class = AutoModelForCausalLM

    # 1. Load quantization config
    qconfig_path = os.path.join(checkpoint_dir, "quantization_config.json")
    with open(qconfig_path) as f:
        qconfig = json.load(f)

    layer_configs = qconfig["layers"]
    fmt_order = qconfig.get("format_order", ["NVFP4", "FP8", "FP16"])

    # 2. Load HF model architecture (empty weights)
    config = AutoConfig.from_pretrained(checkpoint_dir)
    with torch.device("meta"):
        model = model_class.from_config(config)
    # Materialise on CPU first with empty weights
    model = model.to_empty(device="cpu")
    model.eval()

    # 3. Load safetensors
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # 4. Check Marlin feasibility
    can_marlin = use_marlin and torch.cuda.is_available() and _check_vllm()

    # 5. Replace each quantized layer
    for layer_name, lconf in layer_configs.items():
        n_in = lconf["in_features"]
        n_out = lconf["out_features"]
        splits = lconf["splits"]
        prefix = layer_name

        # Read inv_permutation
        inv_perm = tensors[f"{prefix}.inv_permutation"].to(torch.int64)

        # --- Dequantize each group ---
        w_nvfp4 = None
        w_fp8 = None
        w_fp16 = None
        raw_nvfp4_scales = None
        nvfp4_packed_indices = None

        n_nvfp4 = splits.get("NVFP4", 0)
        if n_nvfp4 > 0:
            packed = tensors[f"{prefix}.weight_nvfp4"]  # [n_ch, K//2] uint8
            scales_uint8 = tensors[f"{prefix}.weight_nvfp4_scale"]  # [n_ch, K//16] as uint8
            scales_fp8 = scales_uint8.view(torch.float8_e4m3fn)
            scales_f32 = scales_fp8.to(torch.float32)  # [n_ch, K//16]
            global_scale = tensors[f"{prefix}.nvfp4_global_scale"].item()

            if can_marlin:
                # For Marlin: pass packed indices + raw_scales directly
                nvfp4_packed_indices = packed
                raw_nvfp4_scales = scales_f32  # [n_ch, K//16] — these are already absmax/6.0

            # Dequantize for fake-quant or FP16 materialisation
            # Unpack nibbles
            even = (packed & 0x0F).to(torch.int64)
            odd = ((packed >> 4) & 0x0F).to(torch.int64)
            n_ch = packed.shape[0]
            half_k = packed.shape[1]
            indices_2d = torch.empty(n_ch, half_k * 2, dtype=torch.int64)
            indices_2d[:, 0::2] = even
            indices_2d[:, 1::2] = odd

            # Dequantize using LUT
            from rdquant.core.formats import _NVFP4_LUT, _NVFP4_BLOCK_SIZE
            lut = _NVFP4_LUT
            values = lut[indices_2d.reshape(-1)].reshape(n_ch, n_in)
            # Apply block scales: each scale covers 16 elements
            blocks_per_row = n_in // _NVFP4_BLOCK_SIZE
            scales_expanded = scales_f32.unsqueeze(-1).expand(
                n_ch, blocks_per_row, _NVFP4_BLOCK_SIZE
            ).reshape(n_ch, n_in)
            w_nvfp4 = (values * scales_expanded).half()

        n_fp8 = splits.get("FP8", 0)
        if n_fp8 > 0:
            data_uint8 = tensors[f"{prefix}.weight_fp8"]  # [n_ch, K] as uint8
            data_fp8 = data_uint8.view(torch.float8_e4m3fn)
            data_f32 = data_fp8.to(torch.float32)
            scale_vec = tensors[f"{prefix}.weight_fp8_scale"]  # [n_ch] float32
            w_fp8 = (data_f32 * scale_vec.unsqueeze(1)).half()

        n_fp16 = splits.get("FP16", 0)
        if n_fp16 > 0:
            data_bf16 = tensors[f"{prefix}.weight_fp16"]  # [n_ch, K] bfloat16
            w_fp16 = data_bf16.half()

        if can_marlin:
            # Build MarlinMixedLinear
            target_device = torch.device(device)
            inv_perm_dev = inv_perm.to(target_device)

            bias_key = f"{prefix}.bias"
            bias_tensor = tensors.get(bias_key, None)
            if bias_tensor is not None:
                bias_tensor = bias_tensor.to(target_device)

            # Move weight tensors to device
            if w_nvfp4 is not None:
                w_nvfp4 = w_nvfp4.to(target_device)
            if w_fp8 is not None:
                w_fp8 = w_fp8.to(target_device)
            if w_fp16 is not None:
                w_fp16 = w_fp16.to(target_device)
            if raw_nvfp4_scales is not None:
                raw_nvfp4_scales = raw_nvfp4_scales.to(target_device)
            if nvfp4_packed_indices is not None:
                nvfp4_packed_indices = nvfp4_packed_indices.to(target_device)

            marlin_layer = MarlinMixedLinear(
                w_nvfp4_fp16=w_nvfp4,
                w_fp8_fp16=w_fp8,
                w_fp16_fp16=w_fp16,
                inv_perm=inv_perm_dev,
                bias=bias_tensor,
                raw_nvfp4_scales=raw_nvfp4_scales,
                nvfp4_packed_indices=nvfp4_packed_indices,
            )
            _set_module(model, layer_name, marlin_layer)
        else:
            # Fake-quant: dequantize all groups, apply inv_perm, set as nn.Linear
            pieces = []
            if w_nvfp4 is not None:
                pieces.append(w_nvfp4)
            if w_fp8 is not None:
                pieces.append(w_fp8)
            if w_fp16 is not None:
                pieces.append(w_fp16)

            w_permuted = torch.cat(pieces, dim=0)  # [N_out, K] in permuted order
            w_original = w_permuted[inv_perm].to(torch.bfloat16)  # restore original channel order

            bias_key = f"{prefix}.bias"
            bias_tensor = tensors.get(bias_key, None)

            linear = nn.Linear(n_in, n_out, bias=(bias_tensor is not None))
            linear.weight = nn.Parameter(w_original)
            if bias_tensor is not None:
                linear.bias = nn.Parameter(bias_tensor)

            _set_module(model, layer_name, linear)

    # 6. Move non-quantized parameters to device
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
#  quantize_and_export
# ---------------------------------------------------------------------------

def quantize_and_export(
    model_path: str,
    output_dir: str,
    budget_avg_bits: float = 5.3,
    ignore: Optional[list[str]] = None,
    calibrate: bool = False,
    calib_texts: Optional[list[str]] = None,
    calib_metric: str = "perturb",
    device: str = "cuda",
) -> None:
    """Quantize a HuggingFace model and save as packed checkpoint.

    End-to-end pipeline: load model -> (optionally calibrate) -> quantize
    -> save in packed safetensors format.

    Args:
        model_path: Path to a HuggingFace model directory.
        output_dir: Directory to save the packed checkpoint.
        budget_avg_bits: Target average bits per element.
        ignore: Layer name patterns to skip.
        calibrate: If True, run calibration to compute layer importance.
        calib_texts: Calibration texts (required if ``calibrate=True``).
        calib_metric: Calibration metric (``"perturb"``, ``"fisher"``, etc.).
        device: Device for quantization.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rdquant.quantize import quantize_model
    from rdquant.integrations.hf_export import save_packed

    # 1. Load model
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map=device,
    )
    model.eval()

    # 2. Optionally calibrate
    layer_importance = None
    if calibrate:
        from rdquant.core.calibrate import compute_layer_importance

        if calib_texts is None:
            raise ValueError("calib_texts must be provided when calibrate=True")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Running calibration with metric={calib_metric} ...")
        layer_importance = compute_layer_importance(
            model, tokenizer, calib_texts,
            metric=calib_metric, device=device,
        )

    # 3. Quantize
    print(f"Quantizing with budget={budget_avg_bits} bits ...")
    qmodel = quantize_model(
        model, budget_avg_bits=budget_avg_bits,
        ignore=ignore, layer_importance=layer_importance,
    )

    # 4. Save packed
    print(f"Saving packed checkpoint to {output_dir} ...")
    save_packed(qmodel, output_dir, source_model_dir=model_path)
    print("Done.")


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------

def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule by dotted name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
