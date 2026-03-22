"""Helpers for building fused mixed-GEMV and Marlin fallback buffers.

These utilities convert packed RDQuant checkpoint tensors into:

- Marlin-repacked qweights + processed scale tensors for the existing
  ``marlin_gemm`` fallback path.
- The lighter-weight scale/layout metadata expected by the current
  fused mixed-GEMV split-K prototype.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

K_TILE = 128

_GROUP_MAP_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_RDQUANT_CUDA_AVAILABLE: bool | None = None


def _ensure_vllm_path() -> None:
    vllm_site = "/root/autodl-tmp/vllm_site"
    if os.path.isdir(vllm_site) and vllm_site not in sys.path:
        sys.path.insert(0, vllm_site)


def _ensure_rdquant_cuda_path() -> None:
    ext_dir = os.path.join(os.path.dirname(__file__), "csrc", "build")
    if os.path.isdir(ext_dir) and ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)


def get_rdquant_cuda():
    """Import the rdquant CUDA extension from the local build directory."""
    _ensure_rdquant_cuda_path()
    import rdquant_cuda

    return rdquant_cuda


def fused_gemv_available() -> bool:
    """Return whether the local rdquant CUDA extension is importable."""
    global _RDQUANT_CUDA_AVAILABLE
    if _RDQUANT_CUDA_AVAILABLE is not None:
        return _RDQUANT_CUDA_AVAILABLE
    try:
        mod = get_rdquant_cuda()
        _RDQUANT_CUDA_AVAILABLE = hasattr(
            mod, "fused_mixed_gemv_marlin_weights_splitk_auto"
        )
    except Exception:
        _RDQUANT_CUDA_AVAILABLE = False
    return _RDQUANT_CUDA_AVAILABLE


def choose_parallel_k(num_tiles: int, k: int) -> int:
    """Current fixed split-K heuristic used by the stable benchmark path."""
    del num_tiles
    return k // K_TILE


def _get_weight_perm(num_bits: int) -> torch.Tensor:
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    if num_bits == 4:
        interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    elif num_bits == 8:
        interleave = [0, 2, 1, 3]
    else:
        raise ValueError(f"Unsupported num_bits={num_bits}")

    perm = torch.tensor(perm_list, dtype=torch.int64)
    perm = perm.view(-1, len(interleave))[:, interleave].reshape(-1)
    return perm


def _make_marlin_weight_map(num_bits: int) -> torch.Tensor:
    perm = _get_weight_perm(num_bits)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv.to(dtype=torch.int32)


def make_marlin_group_maps(
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the compact lookup tables used by the fused qweight kernels."""
    device_key = str(torch.device(device))
    cached = _GROUP_MAP_CACHE.get(device_key)
    if cached is not None:
        return cached

    fp4_inv = _make_marlin_weight_map(4)
    fp8_inv = _make_marlin_weight_map(8)

    fp4_word_offsets = torch.empty((64, 4), dtype=torch.int32)
    fp4_slot_map = torch.empty((64, 4, 4), dtype=torch.int32)
    fp8_word_offsets = torch.empty((64, 4), dtype=torch.int32)

    for n_in_tile in range(64):
        for group in range(4):
            sub = group * 2
            base_idx = (n_in_tile // 16) * 256 + (n_in_tile % 16)

            fp4_cols = [
                fp4_inv[base_idx + (sub + 0) * 16].item(),
                fp4_inv[base_idx + (sub + 1) * 16].item(),
                fp4_inv[base_idx + (sub + 8) * 16].item(),
                fp4_inv[base_idx + (sub + 9) * 16].item(),
            ]
            fp8_cols = [
                fp8_inv[base_idx + (sub + 0) * 16].item(),
                fp8_inv[base_idx + (sub + 1) * 16].item(),
                fp8_inv[base_idx + (sub + 8) * 16].item(),
                fp8_inv[base_idx + (sub + 9) * 16].item(),
            ]

            fp4_word_offsets[n_in_tile, group] = fp4_cols[0] // 8
            fp8_word_offsets[n_in_tile, group] = fp8_cols[0] // 4
            fp4_slot_map[n_in_tile, group, 0] = fp4_cols[0] % 8
            fp4_slot_map[n_in_tile, group, 1] = fp4_cols[1] % 8
            fp4_slot_map[n_in_tile, group, 2] = fp4_cols[2] % 8
            fp4_slot_map[n_in_tile, group, 3] = fp4_cols[3] % 8

    result = (
        fp4_word_offsets.to(device=device),
        fp4_slot_map.to(device=device),
        fp8_word_offsets.to(device=device),
    )
    _GROUP_MAP_CACHE[device_key] = result
    return result


def pack_fp4_to_marlin_qweight(w_fp4_packed: torch.Tensor) -> torch.Tensor:
    """Convert RDQuant nibble-packed NVFP4 bytes into Marlin qweight."""
    _ensure_vllm_path()
    import vllm._custom_ops as vllm_ops

    qweight = w_fp4_packed.contiguous().view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=w_fp4_packed.device)
    return vllm_ops.gptq_marlin_repack(
        qweight, perm, w_fp4_packed.size(1) * 2, w_fp4_packed.size(0), 4, False
    )


def pack_fp8_to_marlin_qweight(w_fp8_raw: torch.Tensor) -> torch.Tensor:
    """Convert RDQuant raw FP8 bytes into Marlin qweight."""
    _ensure_vllm_path()
    import vllm._custom_ops as vllm_ops

    qweight = w_fp8_raw.contiguous().view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=w_fp8_raw.device)
    return vllm_ops.gptq_marlin_repack(
        qweight, perm, w_fp8_raw.size(1), w_fp8_raw.size(0), 8, False
    )


def _decode_fp8_e4m3_tensor(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(torch.float8_e4m3fn).to(torch.float32)


def prepare_marlin_nvfp4_scales(
    w_fp4_scales_raw: torch.Tensor,
    global_scale: float,
    k: int,
    n_fp4: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw uint8-view FP8 NVFP4 block scales into Marlin layout."""
    _ensure_vllm_path()
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    fp4_scales = _decode_fp8_e4m3_tensor(w_fp4_scales_raw)
    fp4_scales = fp4_scales.T.contiguous().to(torch.float16)
    fp4_scales = marlin_permute_scales(
        fp4_scales, size_k=k, size_n=n_fp4, group_size=16
    )
    fp4_scales = nvfp4_marlin_process_scales(fp4_scales)
    marlin_global_scale = nvfp4_marlin_process_global_scale(
        torch.tensor(global_scale, device=w_fp4_scales_raw.device, dtype=torch.float16)
    )
    return fp4_scales, marlin_global_scale


def prepare_marlin_fp8_scales(
    w_fp8_scales: torch.Tensor,
    k: int,
    n_fp8: int,
) -> torch.Tensor:
    """Convert raw per-channel FP8 scales into the Marlin FP8 layout."""
    _ensure_vllm_path()
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )

    fp8_scales = w_fp8_scales.view(1, n_fp8).to(torch.float16)
    fp8_scales = marlin_permute_scales(
        fp8_scales, size_k=k, size_n=n_fp8, group_size=-1
    )
    return fp8_fused_exponent_bias_into_scales(fp8_scales)


def build_marlin_data(
    layer_data: dict[str, torch.Tensor],
    splits: dict[str, int],
    k: int,
    device: torch.device | str,
) -> Optional[dict[str, torch.Tensor]]:
    """Build fallback tensors for the existing vLLM Marlin inference path."""
    _ensure_vllm_path()
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    device = torch.device(device)
    n_fp4 = splits.get("NVFP4", 0)
    n_fp8 = splits.get("FP8", 0)

    marlin_data: dict[str, torch.Tensor] = {}

    if n_fp4 > 0 and "weight_nvfp4" in layer_data:
        w_fp4_packed = layer_data["weight_nvfp4"].to(
            device=device, dtype=torch.uint8
        ).contiguous()
        w_fp4_scales = layer_data["weight_nvfp4_scale"].to(
            device=device, dtype=torch.uint8
        ).contiguous()
        global_scale = float(layer_data["nvfp4_global_scale"].item())
        marlin_data["nvfp4_qweight"] = pack_fp4_to_marlin_qweight(w_fp4_packed)
        (
            marlin_data["nvfp4_scales"],
            marlin_data["nvfp4_global_scale"],
        ) = prepare_marlin_nvfp4_scales(w_fp4_scales, global_scale, k, n_fp4)
        marlin_data["nvfp4_workspace"] = marlin_make_workspace_new(device)

    if n_fp8 > 0 and "weight_fp8" in layer_data:
        w_fp8_raw = layer_data["weight_fp8"].to(
            device=device, dtype=torch.uint8
        ).contiguous()
        w_fp8_scales = layer_data["weight_fp8_scale"].to(
            device=device, dtype=torch.float32
        ).contiguous()
        marlin_data["fp8_qweight"] = pack_fp8_to_marlin_qweight(w_fp8_raw)
        marlin_data["fp8_scales"] = prepare_marlin_fp8_scales(w_fp8_scales, k, n_fp8)
        marlin_data["fp8_workspace"] = marlin_make_workspace_new(device)

    return marlin_data or None


def pack_for_fused_gemv(
    layer_data: dict[str, torch.Tensor],
    splits: dict[str, int],
    k: int,
    device: torch.device | str,
) -> Optional[dict[str, torch.Tensor | float | int]]:
    """Build the buffer set expected by the fused mixed-GEMV split-K kernel.

    Returns ``None`` when the current layer is not eligible for the stable
    fused path, which presently requires:

    - both NVFP4 and FP8 groups present
    - no FP16 group
    - ``K`` multiple of 128
    - both output-group sizes multiples of 64
    """
    if not fused_gemv_available():
        return None

    device = torch.device(device)
    n_fp4 = splits.get("NVFP4", 0)
    n_fp8 = splits.get("FP8", 0)
    n_fp16 = splits.get("FP16", 0)

    if (
        n_fp4 <= 0
        or n_fp8 <= 0
        or n_fp16 != 0
        or k % K_TILE != 0
        or n_fp4 % 64 != 0
        or n_fp8 % 64 != 0
    ):
        return None

    w_fp4_packed = layer_data["weight_nvfp4"].to(
        device=device, dtype=torch.uint8
    ).contiguous()
    w_fp8_raw = layer_data["weight_fp8"].to(
        device=device, dtype=torch.uint8
    ).contiguous()
    w_fp4_scales_raw = layer_data["weight_nvfp4_scale"].to(
        device=device, dtype=torch.uint8
    ).contiguous()
    (
        w_fp4_scales_marlin,
        w_fp4_global_scale_marlin,
    ) = prepare_marlin_nvfp4_scales(
        w_fp4_scales_raw,
        float(layer_data["nvfp4_global_scale"].item()),
        k,
        n_fp4,
    )

    fp4_word_offsets, fp4_slot_map, fp8_word_offsets = make_marlin_group_maps(device)
    n_total = n_fp4 + n_fp8
    num_tiles = (n_fp4 + 127) // 128 + (n_fp8 + 127) // 128

    return {
        "w_fp4_q": pack_fp4_to_marlin_qweight(w_fp4_packed),
        "w_fp4_scales": w_fp4_scales_raw,
        "w_fp4_global_scale": float(layer_data["nvfp4_global_scale"].item()),
        "w_fp4_scales_marlin": w_fp4_scales_marlin,
        "w_fp4_global_scale_marlin": w_fp4_global_scale_marlin,
        "w_fp8_q": pack_fp8_to_marlin_qweight(w_fp8_raw),
        "w_fp8_scales": layer_data["weight_fp8_scale"].to(
            device=device, dtype=torch.float32
        ).contiguous(),
        "fp4_word_offsets": fp4_word_offsets,
        "fp4_slot_map": fp4_slot_map,
        "fp8_word_offsets": fp8_word_offsets,
        "workspace": torch.zeros(n_total, device=device, dtype=torch.float32),
        "tile_counters": torch.zeros(num_tiles, device=device, dtype=torch.int32),
        "parallel_k": choose_parallel_k(num_tiles, k),
    }
