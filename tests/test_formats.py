"""
Tests for rdquant/core/formats.py

Covers:
  - Round-trip correctness for all formats (NVFP4, FP8, FP16)
  - MSE monotonicity: NVFP4 >= FP8 >= FP16 ~= 0
  - Edge cases: all-zero, single-value, large outlier
  - NVFP4 block scale is FP8 E4M3 representable
  - FP8 per-channel scale
  - Various tensor sizes
  - QuantizedTensor fields
"""

import math

import pytest
import torch

from rdquant.core.formats import (
    QuantizedTensor,
    _NVFP4_LUT,
    _NVFP4_POS_VALUES,
    _NVFP4_BLOCK_SIZE,
    _FP8_E4M3_MAX,
    _quantize_to_fp8_e4m3,
    compute_mse,
    compute_mse_2d,
    dequantize,
    get_bits_per_element,
    nvfp4_quantize,
    nvfp4_dequantize,
    fp8_quantize,
    fp8_dequantize,
    fp16_quantize,
    fp16_dequantize,
    quantize,
)

torch.manual_seed(42)

FORMATS = ["NVFP4", "FP8", "FP16"]
SIZES = [16, 32, 64, 128, 256, 1024, 4096]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_tensor(n: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n, generator=g)


# ---------------------------------------------------------------------------
# bits_per_element
# ---------------------------------------------------------------------------

def test_bits_per_element_nvfp4():
    assert get_bits_per_element("NVFP4") == 4

def test_bits_per_element_fp8():
    assert get_bits_per_element("FP8") == 8

def test_bits_per_element_fp16():
    assert get_bits_per_element("FP16") == 16

def test_bits_per_element_unknown_raises():
    with pytest.raises(KeyError):
        get_bits_per_element("MXFP4")


# ---------------------------------------------------------------------------
# Round-trip: NVFP4
# ---------------------------------------------------------------------------

class TestNVFP4RoundTrip:
    def test_basic_shape(self):
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(16)
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([3.14])
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_large_outlier(self):
        t = torch.zeros(16)
        t[0] = 1000.0
        t[1] = -1000.0
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape
        assert recon[0] > 0
        assert recon[1] < 0

    @pytest.mark.parametrize("n", SIZES)
    def test_various_sizes(self, n):
        t = _rand_tensor(n)
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_indices_in_range(self):
        t = _rand_tensor(64)
        qt = nvfp4_quantize(t)
        assert qt.data.min() >= 0
        assert qt.data.max() <= 15

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = nvfp4_quantize(t)
        assert qt.format_name == "NVFP4"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = nvfp4_quantize(t)
        assert qt.bits_per_element == 4

    def test_scale_count(self):
        for n in [16, 17, 32, 64, 100, 128]:
            t = _rand_tensor(n)
            qt = nvfp4_quantize(t)
            expected_blocks = math.ceil(n / _NVFP4_BLOCK_SIZE)
            assert qt.scales.shape[0] == expected_blocks

    def test_scale_fp8_representable(self):
        """Block scales should be FP8 E4M3 representable."""
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        # Verify scales survive FP8 round-trip
        scales_rt = _quantize_to_fp8_e4m3(qt.scales)
        assert torch.allclose(qt.scales, scales_rt)

    def test_global_scale_present(self):
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        assert qt.global_scale is not None
        assert qt.global_scale > 0

    def test_zero_block_scale(self):
        t = torch.zeros(16)
        qt = nvfp4_quantize(t)
        assert qt.scales[0].item() == 0.0

    def test_mixed_sign(self):
        t = torch.tensor([-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0,
                           6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert (recon[:4] <= 0).all()
        assert (recon[5:9] >= 0).all()


# ---------------------------------------------------------------------------
# Round-trip: FP8
# ---------------------------------------------------------------------------

class TestFP8RoundTrip:
    def test_basic(self):
        t = _rand_tensor(128)
        qt = fp8_quantize(t)
        recon = fp8_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(64)
        qt = fp8_quantize(t)
        recon = fp8_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([2.5])
        qt = fp8_quantize(t)
        recon = fp8_dequantize(qt)
        assert recon.shape == t.shape

    def test_per_channel_scale(self):
        """FP8 should have a per-channel (single) FP32 scale."""
        t = _rand_tensor(128)
        qt = fp8_quantize(t)
        assert qt.scales.shape == (1,)
        assert qt.scales.dtype == torch.float32

    def test_low_error(self):
        """FP8 should have very low MSE for standard normal data."""
        t = _rand_tensor(1024)
        qt = fp8_quantize(t)
        recon = fp8_dequantize(qt)
        mse = ((t - recon) ** 2).mean().item()
        assert mse < 0.001, f"FP8 MSE too high: {mse}"

    @pytest.mark.parametrize("n", SIZES)
    def test_various_sizes(self, n):
        t = _rand_tensor(n)
        qt = fp8_quantize(t)
        recon = fp8_dequantize(qt)
        assert recon.shape == t.shape

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = fp8_quantize(t)
        assert qt.format_name == "FP8"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = fp8_quantize(t)
        assert qt.bits_per_element == 8


# ---------------------------------------------------------------------------
# Round-trip: FP16
# ---------------------------------------------------------------------------

class TestFP16RoundTrip:
    def test_basic(self):
        t = _rand_tensor(128)
        qt = fp16_quantize(t)
        recon = fp16_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(64)
        qt = fp16_quantize(t)
        recon = fp16_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_nearly_lossless(self):
        """FP16 should have near-zero MSE for normal-range data."""
        t = _rand_tensor(1024)
        qt = fp16_quantize(t)
        recon = fp16_dequantize(qt)
        mse = ((t - recon) ** 2).mean().item()
        assert mse < 1e-6, f"FP16 MSE too high: {mse}"

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = fp16_quantize(t)
        assert qt.format_name == "FP16"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = fp16_quantize(t)
        assert qt.bits_per_element == 16


# ---------------------------------------------------------------------------
# QuantizedTensor fields
# ---------------------------------------------------------------------------

def test_quantized_tensor_fields():
    t = _rand_tensor(128)
    for fmt in FORMATS:
        qt = quantize(t, fmt)
        assert isinstance(qt, QuantizedTensor)
        assert qt.format_name == fmt
        assert qt.original_shape == t.shape
        assert isinstance(qt.data, torch.Tensor)
        assert isinstance(qt.scales, torch.Tensor)
        assert qt.bits_per_element == get_bits_per_element(fmt)


# ---------------------------------------------------------------------------
# MSE monotonicity: NVFP4 >= FP8 >= FP16 ~= 0
# ---------------------------------------------------------------------------

class TestMSEMonotonicity:
    @pytest.mark.parametrize("n", [128, 1024])
    def test_monotonicity(self, n):
        t = _rand_tensor(n)
        mse_nvfp4 = compute_mse(t, "NVFP4")
        mse_fp8 = compute_mse(t, "FP8")
        mse_fp16 = compute_mse(t, "FP16")

        assert mse_nvfp4 >= mse_fp8, f"NVFP4 MSE ({mse_nvfp4}) should >= FP8 MSE ({mse_fp8})"
        assert mse_fp8 >= mse_fp16 - 1e-12, f"FP8 MSE ({mse_fp8}) should >= FP16 MSE ({mse_fp16})"

    def test_nvfp4_higher_error_than_fp16(self):
        t = _rand_tensor(1024)
        assert compute_mse(t, "NVFP4") > compute_mse(t, "FP16")

    def test_zero_tensor_all_zero_mse(self):
        t = torch.zeros(128)
        for fmt in FORMATS:
            mse = compute_mse(t, fmt)
            assert mse == 0.0, f"{fmt}: expected 0 MSE for zero tensor, got {mse}"

    def test_fp16_near_zero_error(self):
        """FP16 should have very low MSE for standard normal data."""
        t = _rand_tensor(1024)
        mse = compute_mse(t, "FP16")
        assert mse < 1e-6, f"FP16 MSE too high: {mse}"


# ---------------------------------------------------------------------------
# Unified quantize/dequantize API
# ---------------------------------------------------------------------------

def test_unified_api_roundtrip():
    t = _rand_tensor(256)
    for fmt in FORMATS:
        qt = quantize(t, fmt)
        recon = dequantize(qt)
        assert recon.shape == t.shape
        assert not torch.isnan(recon).any(), f"{fmt}: NaN in dequantized output"


def test_unified_api_invalid_format():
    t = _rand_tensor(128)
    with pytest.raises(KeyError):
        quantize(t, "MXFP4")


# ---------------------------------------------------------------------------
# compute_mse convenience function
# ---------------------------------------------------------------------------

def test_compute_mse_nonnegative():
    t = _rand_tensor(256)
    for fmt in FORMATS:
        mse = compute_mse(t, fmt)
        assert mse >= 0.0


def test_compute_mse_deterministic():
    t = _rand_tensor(128)
    for fmt in FORMATS:
        mse1 = compute_mse(t, fmt)
        mse2 = compute_mse(t, fmt)
        assert mse1 == mse2


# ---------------------------------------------------------------------------
# compute_mse_2d
# ---------------------------------------------------------------------------

class TestComputeMSE2D:
    def test_shape(self):
        w = torch.randn(8, 64)
        for fmt in FORMATS:
            mse = compute_mse_2d(w, fmt)
            assert mse.shape == (8,)

    def test_nonnegative(self):
        w = torch.randn(8, 64)
        for fmt in FORMATS:
            mse = compute_mse_2d(w, fmt)
            assert (mse >= 0).all()

    def test_zero_weight(self):
        w = torch.zeros(4, 32)
        for fmt in FORMATS:
            mse = compute_mse_2d(w, fmt)
            assert torch.allclose(mse, torch.zeros(4))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_same_value(self):
        for fmt in FORMATS:
            t = torch.full((64,), 3.0)
            qt = quantize(t, fmt)
            recon = dequantize(qt)
            assert recon.shape == t.shape
            assert not torch.isnan(recon).any()

    def test_single_element_tensor(self):
        for fmt in FORMATS:
            t = torch.tensor([1.5])
            qt = quantize(t, fmt)
            recon = dequantize(qt)
            assert recon.shape == (1,)

    def test_very_small_values(self):
        t = torch.randn(128) * 1e-6
        for fmt in FORMATS:
            qt = quantize(t, fmt)
            recon = dequantize(qt)
            assert not torch.isnan(recon).any()

    def test_large_values(self):
        t = torch.randn(128) * 100.0
        for fmt in FORMATS:
            qt = quantize(t, fmt)
            recon = dequantize(qt)
            assert not torch.isnan(recon).any()

    def test_alternating_sign(self):
        t = torch.zeros(64)
        for i in range(64):
            t[i] = ((-1) ** i) * (i + 1) * 0.1
        for fmt in FORMATS:
            qt = quantize(t, fmt)
            recon = dequantize(qt)
            assert not torch.isnan(recon).any()

    def test_non_multiple_of_block_size(self):
        """Tensor length not a multiple of 16."""
        for n in [1, 7, 17, 33, 65, 100]:
            for fmt in FORMATS:
                t = _rand_tensor(n)
                qt = quantize(t, fmt)
                recon = dequantize(qt)
                assert recon.shape == t.shape


# ---------------------------------------------------------------------------
# LUT sanity checks
# ---------------------------------------------------------------------------

def test_nvfp4_lut_symmetry():
    """LUT should be antisymmetric: lut[i+8] == -lut[i]."""
    lut = _NVFP4_LUT
    for i in range(8):
        assert lut[i] == -lut[i + 8], f"LUT[{i}]={lut[i]}, LUT[{i+8}]={lut[i+8]}"


def test_nvfp4_lut_zero_at_index_0():
    assert _NVFP4_LUT[0].item() == 0.0


def test_nvfp4_lut_max_value():
    assert _NVFP4_LUT[7].item() == 6.0


def test_nvfp4_lut_size():
    assert len(_NVFP4_LUT) == 16


# ---------------------------------------------------------------------------
# Block size consistency
# ---------------------------------------------------------------------------

def test_block_size_is_16():
    assert _NVFP4_BLOCK_SIZE == 16


# ---------------------------------------------------------------------------
# FP8 E4M3 helper
# ---------------------------------------------------------------------------

def test_fp8_e4m3_clamp():
    """Values beyond 448 should be clamped."""
    t = torch.tensor([500.0, -500.0, 448.0, -448.0, 0.0])
    result = _quantize_to_fp8_e4m3(t)
    assert result[0].item() <= 448.0
    assert result[1].item() >= -448.0
    assert result[4].item() == 0.0
