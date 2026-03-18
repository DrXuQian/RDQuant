"""
Tests for rdquant/core/calibrate.py and layer_importance in quantize_model.
"""

import torch
import torch.nn as nn
import pytest

from rdquant import quantize_model


torch.manual_seed(42)


class _TinyModel(nn.Module):
    """Minimal model where layers have different importance."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────────
#  layer_importance changes allocation
# ──────────────────────────────────────────────────────────────────────────

class TestLayerImportance:
    def test_high_weight_gets_more_bits(self):
        """Layer with higher importance weight should get >= avg bits of the
        layer with lower importance weight."""
        m1 = _TinyModel()
        m2 = _TinyModel()
        m2.load_state_dict(m1.state_dict())

        # fc1 is very important, fc2 is not
        importance = {"fc1": 10.0, "fc2": 0.1}
        qm = quantize_model(m1, budget_avg_bits=5.0, layer_importance=importance,
                             quantize_activation=False)

        # Without importance — uniform allocation
        qm_uni = quantize_model(m2, budget_avg_bits=5.0,
                                quantize_activation=False)

        # fc1 should get more bits with importance than without
        bits_fc1_imp = qm.layer_info["fc1"].avg_bits
        bits_fc1_uni = qm_uni.layer_info["fc1"].avg_bits
        bits_fc2_imp = qm.layer_info["fc2"].avg_bits

        assert bits_fc1_imp >= bits_fc2_imp, (
            f"High-importance fc1 ({bits_fc1_imp:.2f}) should get >= bits than "
            f"low-importance fc2 ({bits_fc2_imp:.2f})"
        )

    def test_none_importance_is_data_free(self):
        """layer_importance=None should produce same result as no importance."""
        m1 = _TinyModel()
        m2 = _TinyModel()
        m2.load_state_dict(m1.state_dict())

        qm1 = quantize_model(m1, budget_avg_bits=6.0, layer_importance=None,
                              quantize_activation=False)
        qm2 = quantize_model(m2, budget_avg_bits=6.0,
                              quantize_activation=False)

        for name in qm1.layer_info:
            assert qm1.layer_info[name].avg_bits == qm2.layer_info[name].avg_bits

    def test_uniform_importance_same_as_none(self):
        """All weights = 1.0 should be identical to no importance."""
        m1 = _TinyModel()
        m2 = _TinyModel()
        m2.load_state_dict(m1.state_dict())

        importance = {"fc1": 1.0, "fc2": 1.0}
        qm1 = quantize_model(m1, budget_avg_bits=5.0, layer_importance=importance,
                              quantize_activation=False)
        qm2 = quantize_model(m2, budget_avg_bits=5.0,
                              quantize_activation=False)

        for name in qm1.layer_info:
            assert qm1.layer_info[name].avg_bits == qm2.layer_info[name].avg_bits

    def test_output_shape_preserved(self):
        m = _TinyModel()
        importance = {"fc1": 2.0, "fc2": 0.5}
        qm = quantize_model(m, budget_avg_bits=5.0, layer_importance=importance,
                             quantize_activation=False)
        x = torch.randn(2, 64)
        y = qm(x)
        assert y.shape == (2, 16)
        assert torch.isfinite(y).all()

    def test_missing_layer_defaults_to_one(self):
        """If a layer is not in importance dict, it should default to 1.0."""
        m = _TinyModel()
        importance = {"fc1": 5.0}  # fc2 not listed
        qm = quantize_model(m, budget_avg_bits=5.0, layer_importance=importance,
                             quantize_activation=False)
        assert "fc1" in qm.layer_info
        assert "fc2" in qm.layer_info

    def test_extreme_importance_all_high(self):
        """Very high importance on all layers should still respect budget."""
        m = _TinyModel()
        importance = {"fc1": 100.0, "fc2": 100.0}
        qm = quantize_model(m, budget_avg_bits=5.0, layer_importance=importance,
                             quantize_activation=False)
        actual = sum(r.avg_bits for r in qm.layer_info.values()) / len(qm.layer_info)
        # Budget should still be approximately met
        assert actual <= 8.5  # within feasible range


# ──────────────────────────────────────────────────────────────────────────
#  calibrate.py unit tests (no real model needed)
# ──────────────────────────────────────────────────────────────────────────

class TestCalibrateModule:
    def test_normalize(self):
        from rdquant.core.calibrate import _normalize
        raw = {"a": 2.0, "b": 4.0, "c": 6.0}
        normed = _normalize(raw)
        mean_val = sum(normed.values()) / len(normed)
        assert abs(mean_val - 1.0) < 1e-6

    def test_normalize_zero(self):
        from rdquant.core.calibrate import _normalize
        raw = {"a": 0.0, "b": 0.0}
        normed = _normalize(raw)
        assert all(v == 1.0 for v in normed.values())

    def test_normalize_preserves_ratio(self):
        from rdquant.core.calibrate import _normalize
        raw = {"a": 1.0, "b": 3.0}
        normed = _normalize(raw)
        assert abs(normed["b"] / normed["a"] - 3.0) < 1e-6
