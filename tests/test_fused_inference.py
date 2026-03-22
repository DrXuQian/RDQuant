"""Tests for fused decode inference helpers and module dispatch."""

from __future__ import annotations

import json
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import PreTrainedModel

from rdquant.fused_gemv_pack import (
    fused_gemv_available,
    get_rdquant_cuda,
    make_marlin_group_maps,
    pack_for_fused_gemv,
    build_marlin_data,
)
from rdquant.inference import FusedMixedLinear, _check_vllm, load_for_inference


class _DummyConfig:
    model_type = "bert"


class _DummyHFModel(PreTrainedModel):
    config_class = _DummyConfig
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(256, 128, bias=True)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def forward(self, x):
        return self.fc(x)


def _make_valid_layer_data(n_fp4: int, n_fp8: int, k: int) -> dict[str, torch.Tensor]:
    fp4_scale_f32 = torch.rand(n_fp4, k // 16, dtype=torch.float32) * 2.0
    fp8_weight = (torch.randn(n_fp8, k, dtype=torch.float32) * 0.5).to(
        torch.float8_e4m3fn
    )
    return {
        "weight_nvfp4": torch.randint(
            0, 256, (n_fp4, k // 2), device="cpu", dtype=torch.uint8
        ),
        "weight_nvfp4_scale": fp4_scale_f32.to(torch.float8_e4m3fn).view(torch.uint8),
        "nvfp4_global_scale": torch.tensor([1.0], dtype=torch.float32),
        "weight_fp8": fp8_weight.view(torch.uint8).cpu(),
        "weight_fp8_scale": torch.rand(n_fp8, device="cpu", dtype=torch.float32) + 0.1,
        "inv_permutation": torch.randperm(n_fp4 + n_fp8, dtype=torch.int32),
    }


def _write_dummy_checkpoint(tmpdir: str, layer_data: dict[str, torch.Tensor]) -> None:
    save_file(
        {
            f"fc.{name}": tensor.contiguous().cpu()
            for name, tensor in layer_data.items()
        },
        f"{tmpdir}/model.safetensors",
    )
    with open(f"{tmpdir}/quantization_config.json", "w") as f:
        json.dump(
            {
                "quant_method": "rdquant",
                "formats": ["NVFP4", "FP8", "FP16"],
                "budget_avg_bits": 5.0,
                "format_order": ["NVFP4", "FP8", "FP16"],
                "layers": {
                    "fc": {
                        "in_features": 256,
                        "out_features": 128,
                        "splits": {"NVFP4": 64, "FP8": 64, "FP16": 0},
                        "avg_bits": 6.0,
                    }
                },
            },
            f,
        )
    with open(f"{tmpdir}/config.json", "w") as f:
        json.dump({"model_type": "bert"}, f)


def test_make_marlin_group_maps_shapes():
    fp4_word_offsets, fp4_slot_map, fp8_word_offsets = make_marlin_group_maps("cpu")
    assert fp4_word_offsets.shape == (64, 4)
    assert fp4_slot_map.shape == (64, 4, 4)
    assert fp8_word_offsets.shape == (64, 4)
    assert fp4_word_offsets.dtype == torch.int32
    assert fp4_slot_map.dtype == torch.int32
    assert fp8_word_offsets.dtype == torch.int32


def test_pack_for_fused_gemv_rejects_ineligible_shapes():
    layer_data = _make_valid_layer_data(64, 64, 256)
    splits = {"NVFP4": 64, "FP8": 64, "FP16": 16}
    assert pack_for_fused_gemv(layer_data, splits, 256, "cpu") is None

    splits = {"NVFP4": 96, "FP8": 64, "FP16": 0}
    assert pack_for_fused_gemv(layer_data, splits, 256, "cpu") is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_vllm() or not fused_gemv_available(),
    reason="requires CUDA + vLLM + rdquant CUDA extension",
)
def test_pack_for_fused_gemv_includes_processed_marlin_scales():
    layer_data = _make_valid_layer_data(64, 64, 256)
    splits = {"NVFP4": 64, "FP8": 64, "FP16": 0}
    fused_data = pack_for_fused_gemv(layer_data, splits, 256, "cuda")

    assert fused_data is not None
    assert "w_fp4_scales_marlin" in fused_data
    assert "w_fp4_global_scale_marlin" in fused_data
    assert "w_fp8_scales_marlin" in fused_data
    assert fused_data["w_fp4_scales_marlin"].is_cuda
    assert fused_data["w_fp4_global_scale_marlin"].is_cuda
    assert fused_data["w_fp8_scales_marlin"].is_cuda


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_vllm() or not fused_gemv_available(),
    reason="requires CUDA + vLLM + rdquant CUDA extension",
)
def test_fused_mixed_linear_decode_matches_direct_kernel():
    sys.path.insert(0, "/root/autodl-tmp/vllm_site")
    rdquant_cuda = get_rdquant_cuda()

    device = "cuda"
    n_fp4, n_fp8, k = 64, 64, 256
    splits = {"NVFP4": n_fp4, "FP8": n_fp8, "FP16": 0}
    layer_data = _make_valid_layer_data(n_fp4, n_fp8, k)
    marlin_data = build_marlin_data(layer_data, splits, k, device)
    fused_data = pack_for_fused_gemv(layer_data, splits, k, device)

    mod = FusedMixedLinear(
        n_nvfp4=n_fp4,
        n_fp8=n_fp8,
        n_fp16=0,
        k=k,
        inv_perm=layer_data["inv_permutation"].to(device=device, dtype=torch.int64),
        bias=None,
        marlin_data=marlin_data,
        fused_data=fused_data,
        w_fp16_fp16=None,
    ).to(device)
    mod.eval()

    x = torch.randn(1, k, device=device, dtype=torch.float16)
    y_mod = mod(x)
    lane = mod._select_decode_fused_lane()
    if lane == "n4m":
        y_direct = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_nvfp4_marlin(
            x,
            mod._fused_w_fp4_q,
            mod._fused_w_fp4_scales_marlin,
            mod._fused_w_fp4_global_scale_marlin,
            mod._fused_w_fp8_q,
            mod._fused_w_fp8_scales,
            mod._fused_fp4_word_offsets,
            mod._fused_fp4_slot_map,
            mod._fused_fp8_word_offsets,
            mod._inv_perm_i32,
            mod._fused_workspace,
            mod._fused_tile_counters,
            n_fp4,
            n_fp8,
            k,
            mod._fused_parallel_k,
        )
    elif lane == "f8m":
        y_direct = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_fp8_marlin(
            x,
            mod._fused_w_fp4_q,
            mod._fused_w_fp4_scales_marlin,
            mod._fused_w_fp4_global_scale_marlin,
            mod._fused_w_fp8_q,
            mod._fused_w_fp8_scales_marlin,
            mod._fused_fp4_word_offsets,
            mod._fused_fp4_slot_map,
            mod._fused_fp8_word_offsets,
            mod._inv_perm_i32,
            mod._fused_workspace,
            mod._fused_tile_counters,
            n_fp4,
            n_fp8,
            k,
            mod._fused_parallel_k,
        )
    else:
        y_direct = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_auto(
            x,
            mod._fused_w_fp4_q,
            mod._fused_w_fp4_scales,
            mod._fused_w_fp4_global_scale,
            mod._fused_w_fp8_q,
            mod._fused_w_fp8_scales,
            mod._fused_fp4_word_offsets,
            mod._fused_fp4_slot_map,
            mod._fused_fp8_word_offsets,
            mod._inv_perm_i32,
            mod._fused_workspace,
            mod._fused_tile_counters,
            n_fp4,
            n_fp8,
            k,
            mod._fused_parallel_k,
        )
    assert torch.equal(y_mod, y_direct)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_vllm() or not fused_gemv_available(),
    reason="requires CUDA + vLLM + rdquant CUDA extension",
)
def test_fused_mixed_linear_prefill_matches_marlin_fallback():
    device = "cuda"
    n_fp4, n_fp8, k = 64, 64, 256
    splits = {"NVFP4": n_fp4, "FP8": n_fp8, "FP16": 0}
    layer_data = _make_valid_layer_data(n_fp4, n_fp8, k)
    marlin_data = build_marlin_data(layer_data, splits, k, device)
    fused_data = pack_for_fused_gemv(layer_data, splits, k, device)

    mod = FusedMixedLinear(
        n_nvfp4=n_fp4,
        n_fp8=n_fp8,
        n_fp16=0,
        k=k,
        inv_perm=layer_data["inv_permutation"].to(device=device, dtype=torch.int64),
        bias=None,
        marlin_data=marlin_data,
        fused_data=fused_data,
        w_fp16_fp16=None,
    ).to(device)
    mod.eval()

    x = torch.randn(2, k, device=device, dtype=torch.float16)
    y = mod(x)
    y_ref = mod._forward_marlin(x)
    assert torch.equal(y, y_ref)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_vllm() or not fused_gemv_available(),
    reason="requires CUDA + vLLM + rdquant CUDA extension",
)
def test_load_for_inference_replaces_layer_with_fused_module():
    device = "cuda"
    layer_data = _make_valid_layer_data(64, 64, 256)
    layer_data["bias"] = torch.randn(128, dtype=torch.float16)

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_checkpoint(tmpdir, layer_data)
        model = load_for_inference(
            tmpdir,
            model_class=_DummyHFModel,
            device=device,
            use_marlin=True,
            use_fused_gemv=True,
        )

    assert isinstance(model.fc, FusedMixedLinear)
    assert model.fc._has_marlin
    assert model.fc._has_fused

    x_decode = torch.randn(1, 256, device=device, dtype=torch.float16)
    x_prefill = torch.randn(2, 256, device=device, dtype=torch.float16)

    y_decode = model(x_decode)
    y_prefill = model(x_prefill)

    y_decode_ref = model.fc._forward_fused_gemv(x_decode)
    y_prefill_ref = model.fc._forward_marlin(x_prefill)

    assert torch.equal(y_decode, y_decode_ref + model.fc.bias)
    assert torch.equal(y_prefill, y_prefill_ref + model.fc.bias)
