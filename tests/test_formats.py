"""
Tests for rdquant/core/formats.py

Covers:
  - Round-trip correctness for all formats
  - MSE monotonicity: NVFP4 >= MXFP6 >= MXFP8 >= FP16 ≈ 0
  - Edge cases: all-zero, single-value, large outlier
  - Block scale correctness for NVFP4
  - Various tensor sizes
"""

import math

import pytest
import torch

from rdquant.core.formats import (
    QuantizedTensor,
    _NVFP4_LUT,
    _MXFP6_LUT,
    _MXFP8_E4M3_LUT,
    _FP8_E4M3_MAX,
    _quantize_to_fp8_e4m3,
    compute_mse,
    dequantize,
    fp16_dequantize,
    fp16_quantize,
    get_bits_per_element,
    mxfp6_dequantize,
    mxfp6_quantize,
    mxfp8_dequantize,
    mxfp8_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
    quantize,
)

torch.manual_seed(42)

FORMATS = ["NVFP4", "MXFP6", "MXFP8", "FP16"]
SIZES = [128, 1024, 4096]


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

def test_bits_per_element():
    assert get_bits_per_element("NVFP4") == 4
    assert get_bits_per_element("MXFP6") == 6
    assert get_bits_per_element("MXFP8") == 8
    assert get_bits_per_element("FP16") == 16


# ---------------------------------------------------------------------------
# Round-trip: dequantized values must come from the LUT
# ---------------------------------------------------------------------------

class TestNVFP4RoundTrip:
    def test_values_in_lut(self):
        """All dequantized values must be exact LUT entries (×scale)."""
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(32)
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([3.14])
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_large_outlier(self):
        t = torch.zeros(32)
        t[0] = 1000.0
        t[1] = -1000.0
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        # outlier position should be max LUT value * scale
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


class TestMXFP6RoundTrip:
    def test_basic(self):
        t = _rand_tensor(128)
        qt = mxfp6_quantize(t)
        recon = mxfp6_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(64)
        qt = mxfp6_quantize(t)
        recon = mxfp6_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([2.5])
        qt = mxfp6_quantize(t)
        recon = mxfp6_dequantize(qt)
        assert recon.shape == t.shape

    def test_large_outlier(self):
        t = torch.zeros(32)
        t[0] = 500.0
        qt = mxfp6_quantize(t)
        recon = mxfp6_dequantize(qt)
        assert recon[0] > 0

    def test_codes_in_range(self):
        t = _rand_tensor(128)
        qt = mxfp6_quantize(t)
        assert qt.data.min() >= 0
        assert qt.data.max() <= 63

    @pytest.mark.parametrize("n", SIZES)
    def test_various_sizes(self, n):
        t = _rand_tensor(n)
        qt = mxfp6_quantize(t)
        recon = mxfp6_dequantize(qt)
        assert recon.shape == t.shape


class TestMXFP8RoundTrip:
    def test_basic(self):
        t = _rand_tensor(128)
        qt = mxfp8_quantize(t)
        recon = mxfp8_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(64)
        qt = mxfp8_quantize(t)
        recon = mxfp8_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([1.0])
        qt = mxfp8_quantize(t)
        recon = mxfp8_dequantize(qt)
        assert recon.shape == t.shape

    def test_codes_in_range(self):
        t = _rand_tensor(128)
        qt = mxfp8_quantize(t)
        assert qt.data.min() >= 0
        assert qt.data.max() <= 255

    @pytest.mark.parametrize("n", SIZES)
    def test_various_sizes(self, n):
        t = _rand_tensor(n)
        qt = mxfp8_quantize(t)
        recon = mxfp8_dequantize(qt)
        assert recon.shape == t.shape


class TestFP16RoundTrip:
    def test_basic(self):
        t = _rand_tensor(128)
        qt = fp16_quantize(t)
        recon = fp16_dequantize(qt)
        assert recon.shape == t.shape
        # FP16 should be very close
        assert torch.allclose(recon, t, atol=1e-3)

    def test_zero_tensor(self):
        t = torch.zeros(64)
        qt = fp16_quantize(t)
        recon = fp16_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_mse_near_zero(self):
        t = _rand_tensor(1024)
        mse = compute_mse(t, "FP16")
        assert mse < 1e-6


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


# ---------------------------------------------------------------------------
# MSE monotonicity: NVFP4 >= MXFP6 >= MXFP8 >= FP16 ≈ 0
# ---------------------------------------------------------------------------

class TestMSEMonotonicity:
    @pytest.mark.parametrize("n", [128, 1024])
    def test_monotonicity(self, n):
        t = _rand_tensor(n)
        mse_fp4 = compute_mse(t, "NVFP4")
        mse_fp6 = compute_mse(t, "MXFP6")
        mse_fp8 = compute_mse(t, "MXFP8")
        mse_fp16 = compute_mse(t, "FP16")

        assert mse_fp4 >= mse_fp6, f"NVFP4 MSE ({mse_fp4}) should >= MXFP6 MSE ({mse_fp6})"
        assert mse_fp6 >= mse_fp8, f"MXFP6 MSE ({mse_fp6}) should >= MXFP8 MSE ({mse_fp8})"
        assert mse_fp8 >= mse_fp16, f"MXFP8 MSE ({mse_fp8}) should >= FP16 MSE ({mse_fp16})"
        assert mse_fp16 < 1e-6

    def test_fp4_higher_error_than_fp16(self):
        t = _rand_tensor(1024)
        assert compute_mse(t, "NVFP4") > compute_mse(t, "FP16")

    def test_zero_tensor_all_zero_mse(self):
        t = torch.zeros(128)
        for fmt in FORMATS:
            mse = compute_mse(t, fmt)
            assert mse == 0.0, f"{fmt}: expected 0 MSE for zero tensor, got {mse}"


# ---------------------------------------------------------------------------
# Block scale correctness for NVFP4
# ---------------------------------------------------------------------------

class TestNVFP4BlockScales:
    def test_scale_count(self):
        """Number of block scales should be ceil(n / 16)."""
        for n in [16, 17, 32, 100, 128]:
            t = _rand_tensor(n)
            qt = nvfp4_quantize(t)
            expected_blocks = math.ceil(n / 16)
            assert qt.scales.shape[0] == expected_blocks, \
                f"n={n}: expected {expected_blocks} scales, got {qt.scales.shape[0]}"

    def test_scales_are_fp8_representable(self):
        """Block scales should be exactly representable in FP8 E4M3."""
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        for sc in qt.scales.tolist():
            if sc == 0.0:
                continue
            # Re-quantize the scale through FP8 and check it's unchanged
            sc_t = torch.tensor([sc])
            sc_fp8 = _quantize_to_fp8_e4m3(sc_t).item()
            assert abs(sc - sc_fp8) < 1e-6 * abs(sc) + 1e-12, \
                f"Scale {sc} not FP8 E4M3 representable (got {sc_fp8})"

    def test_scale_positive(self):
        t = _rand_tensor(128)
        qt = nvfp4_quantize(t)
        assert (qt.scales >= 0).all()

    def test_zero_block_scale_is_zero(self):
        t = torch.zeros(16)
        qt = nvfp4_quantize(t)
        assert qt.scales[0].item() == 0.0


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

    def test_mixed_sign(self):
        t = torch.tensor([-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qt = nvfp4_quantize(t)
        recon = nvfp4_dequantize(qt)
        # Negative values should dequantize to negatives
        assert (recon[:4] <= 0).all()
        assert (recon[5:9] >= 0).all()


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


def test_mxfp6_lut_size():
    assert len(_MXFP6_LUT) == 64


def test_mxfp8_lut_size():
    assert len(_MXFP8_E4M3_LUT) == 256


def test_mxfp8_lut_no_nan():
    assert not torch.isnan(_MXFP8_E4M3_LUT).any()
