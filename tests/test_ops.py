"""
Tests for rdquant/ops.py — mixed_precision_linear correctness.

Verifies that the grouped-GEMM path (per-format dequant → matmul → cat →
inv_perm) produces the same output as the monolithic dequant → single matmul.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rdquant import quantize_model, QuantizedLayer
from rdquant.ops import mixed_precision_linear

torch.manual_seed(42)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _quantize_tiny(budget: float = 6.0) -> tuple:
    """Return (quantized_model, fresh_input)."""
    model = _TinyModel()
    qm = quantize_model(model, budget_avg_bits=budget)
    x = torch.randn(2, 128)
    return qm, x


# ──────────────────────────────────────────────────────────────────────────
#  Correctness: grouped vs monolithic
# ──────────────────────────────────────────────────────────────────────────

class TestGroupedVsMonolithic:
    """Compare per-format grouped GEMM against full-dequant single GEMM."""

    def _monolithic_forward(self, qlayer: QuantizedLayer, x: torch.Tensor):
        """Old path: dequant entire weight → one F.linear."""
        w = qlayer.quantized_weight.dequantize()
        return F.linear(x, w, qlayer.bias)

    @pytest.mark.parametrize("budget", [4.0, 6.0, 8.0, 16.0])
    def test_outputs_match(self, budget):
        qm, _ = _quantize_tiny(budget)
        for name, mod in qm.model.named_modules():
            if not isinstance(mod, QuantizedLayer):
                continue
            x_layer = torch.randn(2, mod.in_features)
            y_grouped = mod(x_layer)
            y_mono = self._monolithic_forward(mod, x_layer)
            assert torch.allclose(y_grouped, y_mono, atol=1e-5), (
                f"{name}: max diff = {(y_grouped - y_mono).abs().max():.2e}"
            )

    def test_end_to_end_shape(self):
        qm, x = _quantize_tiny()
        y = qm(x)
        assert y.shape == (2, 32)

    def test_output_finite(self):
        qm, x = _quantize_tiny()
        y = qm(x)
        assert torch.isfinite(y).all()

    def test_grad_flows(self):
        """Backward should work through the grouped path."""
        qm, x = _quantize_tiny()
        x.requires_grad_(True)
        y = qm(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ──────────────────────────────────────────────────────────────────────────
#  Direct mixed_precision_linear API
# ──────────────────────────────────────────────────────────────────────────

class TestDirectAPI:
    def test_with_bias(self):
        qm, _ = _quantize_tiny()
        for name, mod in qm.model.named_modules():
            if isinstance(mod, QuantizedLayer) and mod.bias is not None:
                x_layer = torch.randn(2, mod.in_features)
                qw = mod.quantized_weight
                y = mixed_precision_linear(
                    x_layer, qw.qtensors, qw.splits, qw.inv_permutation, mod.bias
                )
                assert y.shape[-1] == mod.out_features
                return
        pytest.skip("No biased layer found")

    def test_without_bias(self):
        qm, _ = _quantize_tiny()
        for name, mod in qm.model.named_modules():
            if isinstance(mod, QuantizedLayer) and mod.bias is None:
                x_layer = torch.randn(2, mod.in_features)
                qw = mod.quantized_weight
                y = mixed_precision_linear(
                    x_layer, qw.qtensors, qw.splits, qw.inv_permutation, None
                )
                assert y.shape[-1] == mod.out_features
                return
        pytest.skip("No unbiased layer found")

    def test_batch_dims(self):
        """Works with arbitrary leading batch dimensions."""
        qm, _ = _quantize_tiny()
        x = torch.randn(3, 4, 128)
        y = qm(x)
        assert y.shape == (3, 4, 32)


# ──────────────────────────────────────────────────────────────────────────
#  Edge: all-one-format (budget 4 → all NVFP4, budget 16 → all FP16)
# ──────────────────────────────────────────────────────────────────────────

class TestSingleFormat:
    def test_all_nvfp4(self):
        qm, x = _quantize_tiny(budget=4.0)
        y = qm(x)
        assert y.shape == (2, 32)
        assert torch.isfinite(y).all()

    def test_all_fp16(self):
        qm, x = _quantize_tiny(budget=16.0)
        y = qm(x)
        assert y.shape == (2, 32)
        assert torch.isfinite(y).all()
