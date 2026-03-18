"""
Tests for rdquant/core/act_quant.py

Covers:
  - Round-trip correctness for MXFP8 activation quantization
  - Error bounds
  - Various shapes (1D, 2D, 3D)
  - Edge cases: zero, constant, large values
"""

import math

import pytest
import torch

from rdquant.core.act_quant import quantize_activation_mxfp8, dequantize_activation_mxfp8

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_2d_shape(self):
        x = torch.randn(4, 128)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert recon.shape == x.shape

    def test_3d_shape(self):
        x = torch.randn(2, 8, 64)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert recon.shape == x.shape

    def test_1d_shape(self):
        x = torch.randn(128)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert recon.shape == x.shape

    def test_values_are_finite(self):
        x = torch.randn(4, 128)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert torch.isfinite(recon).all()

    def test_codes_dtype(self):
        x = torch.randn(2, 64)
        codes, scales = quantize_activation_mxfp8(x)
        assert codes.dtype == torch.long

    def test_scales_dtype(self):
        x = torch.randn(2, 64)
        codes, scales = quantize_activation_mxfp8(x)
        assert scales.dtype == torch.float32

    def test_scales_shape(self):
        x = torch.randn(4, 128)
        codes, scales = quantize_activation_mxfp8(x)
        expected_n_blocks = math.ceil(128 / 32)
        assert scales.shape == (4, expected_n_blocks)

    def test_codes_shape(self):
        x = torch.randn(4, 128)
        codes, scales = quantize_activation_mxfp8(x)
        assert codes.shape == x.shape


# ---------------------------------------------------------------------------
# Error bounds
# ---------------------------------------------------------------------------

class TestErrorBounds:
    def test_mse_small(self):
        """MXFP8 should have very low reconstruction error for standard normal."""
        x = torch.randn(8, 256)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        mse = ((x - recon) ** 2).mean().item()
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_error_bounded_per_element(self):
        """Reconstruction error should be bounded relative to scale."""
        x = torch.randn(4, 128)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        max_err = (x - recon).abs().max().item()
        # Max error should be less than 2x the max value (generous bound)
        assert max_err < x.abs().max().item() * 0.5 + 0.01


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_tensor(self):
        x = torch.zeros(4, 64)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert torch.allclose(recon, x)

    def test_constant_tensor(self):
        x = torch.full((2, 32), 1.5)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert torch.isfinite(recon).all()

    def test_large_values(self):
        x = torch.randn(2, 64) * 100.0
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert torch.isfinite(recon).all()

    def test_very_small_values(self):
        x = torch.randn(2, 64) * 1e-6
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert torch.isfinite(recon).all()

    def test_non_multiple_of_32(self):
        """K not a multiple of 32."""
        x = torch.randn(3, 50)
        codes, scales = quantize_activation_mxfp8(x)
        recon = dequantize_activation_mxfp8(codes, scales, x.shape)
        assert recon.shape == x.shape
        assert torch.isfinite(recon).all()

    def test_deterministic(self):
        x = torch.randn(2, 64)
        codes1, scales1 = quantize_activation_mxfp8(x)
        codes2, scales2 = quantize_activation_mxfp8(x)
        assert torch.equal(codes1, codes2)
        assert torch.equal(scales1, scales2)
