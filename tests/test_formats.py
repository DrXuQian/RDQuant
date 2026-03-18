"""
Tests for rdquant/core/formats.py

Covers:
  - Round-trip correctness for all MX formats (MXFP4, MXFP6, MXFP8)
  - MSE monotonicity: MXFP4 >= MXFP6 >= MXFP8
  - Edge cases: all-zero, single-value, large outlier
  - Block scale correctness
  - Various tensor sizes
  - MXQuantizedTensor fields
"""

import math

import pytest
import torch

from rdquant.core.formats import (
    MXQuantizedTensor,
    _MXFP4_LUT,
    _MXFP6_LUT,
    _MXFP8_E4M3_LUT,
    _MX_BLOCK_SIZE,
    compute_mse,
    compute_mse_2d,
    dequantize,
    get_bits_per_element,
    mxfp4_dequantize,
    mxfp4_quantize,
    mxfp6_dequantize,
    mxfp6_quantize,
    mxfp8_dequantize,
    mxfp8_quantize,
    quantize,
)

torch.manual_seed(42)

FORMATS = ["MXFP4", "MXFP6", "MXFP8"]
SIZES = [32, 64, 128, 256, 1024, 4096]


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

def test_bits_per_element_mxfp4():
    assert get_bits_per_element("MXFP4") == 4

def test_bits_per_element_mxfp6():
    assert get_bits_per_element("MXFP6") == 6

def test_bits_per_element_mxfp8():
    assert get_bits_per_element("MXFP8") == 8

def test_bits_per_element_unknown_raises():
    with pytest.raises(KeyError):
        get_bits_per_element("FP16")


# ---------------------------------------------------------------------------
# Round-trip: MXFP4
# ---------------------------------------------------------------------------

class TestMXFP4RoundTrip:
    def test_basic_shape(self):
        t = _rand_tensor(128)
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_zero_tensor(self):
        t = torch.zeros(32)
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert torch.allclose(recon, t)

    def test_single_element(self):
        t = torch.tensor([3.14])
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_large_outlier(self):
        t = torch.zeros(32)
        t[0] = 1000.0
        t[1] = -1000.0
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert recon.shape == t.shape
        assert recon[0] > 0
        assert recon[1] < 0

    @pytest.mark.parametrize("n", SIZES)
    def test_various_sizes(self, n):
        t = _rand_tensor(n)
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert recon.shape == t.shape

    def test_indices_in_range(self):
        t = _rand_tensor(64)
        qt = mxfp4_quantize(t)
        assert qt.data.min() >= 0
        assert qt.data.max() <= 15

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = mxfp4_quantize(t)
        assert qt.format_name == "MXFP4"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = mxfp4_quantize(t)
        assert qt.bits_per_element == 4

    def test_scale_count(self):
        for n in [32, 33, 64, 100, 128]:
            t = _rand_tensor(n)
            qt = mxfp4_quantize(t)
            expected_blocks = math.ceil(n / _MX_BLOCK_SIZE)
            assert qt.scales.shape[0] == expected_blocks

    def test_scale_positive_or_zero(self):
        t = _rand_tensor(128)
        qt = mxfp4_quantize(t)
        # shared exponents can be negative (floor(log2(absmax)))
        # just check they're finite
        assert torch.isfinite(qt.scales).all()

    def test_zero_block_scale(self):
        t = torch.zeros(32)
        qt = mxfp4_quantize(t)
        assert qt.scales[0].item() == 0.0

    def test_mixed_sign(self):
        t = torch.tensor([-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0,
                           6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qt = mxfp4_quantize(t)
        recon = mxfp4_dequantize(qt)
        assert (recon[:4] <= 0).all()
        assert (recon[5:9] >= 0).all()


# ---------------------------------------------------------------------------
# Round-trip: MXFP6
# ---------------------------------------------------------------------------

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

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = mxfp6_quantize(t)
        assert qt.format_name == "MXFP6"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = mxfp6_quantize(t)
        assert qt.bits_per_element == 6


# ---------------------------------------------------------------------------
# Round-trip: MXFP8
# ---------------------------------------------------------------------------

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

    def test_format_name(self):
        t = _rand_tensor(64)
        qt = mxfp8_quantize(t)
        assert qt.format_name == "MXFP8"

    def test_bits_per_element_field(self):
        t = _rand_tensor(64)
        qt = mxfp8_quantize(t)
        assert qt.bits_per_element == 8


# ---------------------------------------------------------------------------
# MXQuantizedTensor fields
# ---------------------------------------------------------------------------

def test_quantized_tensor_fields():
    t = _rand_tensor(128)
    for fmt in FORMATS:
        qt = quantize(t, fmt)
        assert isinstance(qt, MXQuantizedTensor)
        assert qt.format_name == fmt
        assert qt.original_shape == t.shape
        assert isinstance(qt.data, torch.Tensor)
        assert isinstance(qt.scales, torch.Tensor)
        assert qt.bits_per_element == get_bits_per_element(fmt)


# ---------------------------------------------------------------------------
# MSE monotonicity: MXFP4 >= MXFP6 >= MXFP8
# ---------------------------------------------------------------------------

class TestMSEMonotonicity:
    @pytest.mark.parametrize("n", [128, 1024])
    def test_monotonicity(self, n):
        t = _rand_tensor(n)
        mse_fp4 = compute_mse(t, "MXFP4")
        mse_fp6 = compute_mse(t, "MXFP6")
        mse_fp8 = compute_mse(t, "MXFP8")

        assert mse_fp4 >= mse_fp6, f"MXFP4 MSE ({mse_fp4}) should >= MXFP6 MSE ({mse_fp6})"
        assert mse_fp6 >= mse_fp8, f"MXFP6 MSE ({mse_fp6}) should >= MXFP8 MSE ({mse_fp8})"

    def test_fp4_higher_error_than_fp8(self):
        t = _rand_tensor(1024)
        assert compute_mse(t, "MXFP4") > compute_mse(t, "MXFP8")

    def test_zero_tensor_all_zero_mse(self):
        t = torch.zeros(128)
        for fmt in FORMATS:
            mse = compute_mse(t, fmt)
            assert mse == 0.0, f"{fmt}: expected 0 MSE for zero tensor, got {mse}"

    def test_mxfp8_low_error(self):
        """MXFP8 should have very low MSE for standard normal data."""
        t = _rand_tensor(1024)
        mse = compute_mse(t, "MXFP8")
        assert mse < 0.01, f"MXFP8 MSE too high: {mse}"


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
        quantize(t, "FP16")


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
        """Tensor length not a multiple of 32."""
        for n in [1, 7, 33, 65, 100]:
            for fmt in FORMATS:
                t = _rand_tensor(n)
                qt = quantize(t, fmt)
                recon = dequantize(qt)
                assert recon.shape == t.shape


# ---------------------------------------------------------------------------
# LUT sanity checks
# ---------------------------------------------------------------------------

def test_mxfp4_lut_symmetry():
    """LUT should be antisymmetric: lut[i+8] == -lut[i]."""
    lut = _MXFP4_LUT
    for i in range(8):
        assert lut[i] == -lut[i + 8], f"LUT[{i}]={lut[i]}, LUT[{i+8}]={lut[i+8]}"


def test_mxfp4_lut_zero_at_index_0():
    assert _MXFP4_LUT[0].item() == 0.0


def test_mxfp4_lut_max_value():
    assert _MXFP4_LUT[7].item() == 6.0


def test_mxfp4_lut_size():
    assert len(_MXFP4_LUT) == 16


def test_mxfp6_lut_size():
    assert len(_MXFP6_LUT) == 64


def test_mxfp8_lut_size():
    assert len(_MXFP8_E4M3_LUT) == 256


def test_mxfp8_lut_no_nan():
    assert not torch.isnan(_MXFP8_E4M3_LUT).any()


def test_mxfp6_lut_zero_at_index_0():
    assert _MXFP6_LUT[0].item() == 0.0


def test_mxfp8_lut_zero_at_index_0():
    assert _MXFP8_E4M3_LUT[0].item() == 0.0


# ---------------------------------------------------------------------------
# Block size consistency
# ---------------------------------------------------------------------------

def test_block_size_is_32():
    assert _MX_BLOCK_SIZE == 32
