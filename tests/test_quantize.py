"""Tests for rdquant/quantize.py.

Uses hand-crafted small models only (no HuggingFace downloads).
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

from rdquant.quantize import quantize_model, QuantizedModel, QuantizedLayer


# ---------------------------------------------------------------------------
# Small test model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class TinyModelNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_tiny() -> TinyModel:
    torch.manual_seed(42)
    model = TinyModel()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_quantize_model_returns_quantized_model():
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    assert isinstance(qmodel, QuantizedModel)


def test_output_shape_unchanged():
    torch.manual_seed(0)
    model = _make_tiny()
    x = torch.randn(4, 64)
    with torch.no_grad():
        original_out = model(x)

    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        q_out = qmodel(x)

    assert q_out.shape == original_out.shape


def test_forward_produces_finite_outputs():
    torch.manual_seed(1)
    model = _make_tiny()
    x = torch.randn(4, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()


def test_budget_approximately_met():
    torch.manual_seed(2)
    model = _make_tiny()
    target_bits = 5.3
    qmodel = quantize_model(model, budget_avg_bits=target_bits)

    total_bits = 0.0
    total_params = 0
    for name, result in qmodel.layer_info.items():
        for fmt, stats in result.format_stats.items():
            total_bits += stats["total_bits"]
        n_out = sum(result.splits.values())
        if n_out > 0 and result.avg_bits > 0:
            n_in = int(round(
                sum(s["total_bits"] for s in result.format_stats.values()) /
                (n_out * result.avg_bits)
            ))
            total_params += n_out * n_in

    if total_params > 0:
        actual_bits = total_bits / total_params
        assert abs(actual_bits - target_bits) < 1.5


def test_ignore_patterns():
    torch.manual_seed(3)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3, ignore=["fc1"])

    assert "fc1" not in qmodel.layer_info
    assert "fc2" in qmodel.layer_info
    assert "fc3" in qmodel.layer_info
    assert isinstance(qmodel.model.fc1, nn.Linear)
    assert isinstance(qmodel.model.fc2, QuantizedLayer)


def test_ignore_all_layers():
    torch.manual_seed(4)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3, ignore=["fc1", "fc2", "fc3"])
    assert len(qmodel.layer_info) == 0
    assert isinstance(qmodel.model.fc1, nn.Linear)
    assert isinstance(qmodel.model.fc2, nn.Linear)
    assert isinstance(qmodel.model.fc3, nn.Linear)


def test_no_bias_model():
    torch.manual_seed(5)
    model = TinyModelNoBias()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 16)


def test_per_layer_budget():
    torch.manual_seed(6)
    model = _make_tiny()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3, per_layer_budget=True)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 8)


def test_save_load_roundtrip():
    torch.manual_seed(7)
    model = _make_tiny()
    x = torch.randn(2, 64)

    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out1 = qmodel(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        qmodel.save_pretrained(tmpdir)
        assert os.path.isfile(os.path.join(tmpdir, "quantization_config.json"))

        qmodel2 = QuantizedModel.from_pretrained(tmpdir)
        with torch.no_grad():
            out2 = qmodel2(x)

    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=1e-5)


def test_print_summary_runs(capsys):
    torch.manual_seed(8)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    qmodel.print_summary()
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_formats_subset():
    torch.manual_seed(9)
    model = _make_tiny()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=7.0, formats=["MXFP6", "MXFP8"])
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()


def test_low_budget_mostly_mxfp4():
    torch.manual_seed(10)
    model = TinyModel()
    qmodel = quantize_model(model, budget_avg_bits=4.0)
    for name, result in qmodel.layer_info.items():
        total_ch = sum(result.splits.values())
        mxfp4_ch = result.splits.get("MXFP4", 0)
        assert mxfp4_ch >= total_ch * 0.5


def test_high_budget_mostly_mxfp8():
    torch.manual_seed(11)
    model = TinyModel()
    qmodel = quantize_model(model, budget_avg_bits=8.0)
    for name, result in qmodel.layer_info.items():
        total_ch = sum(result.splits.values())
        mxfp8_ch = result.splits.get("MXFP8", 0)
        assert mxfp8_ch == total_ch


def test_quantized_layer_repr():
    torch.manual_seed(12)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    repr_str = repr(qmodel.model.fc1)
    assert "avg_bits" in repr_str
    assert "act_quant" in repr_str


def test_activation_quantization_disabled():
    """quantize_activation=False should skip MXFP8 act quantization."""
    torch.manual_seed(13)
    model = _make_tiny()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3, quantize_activation=False)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()
    # Check that QuantizedLayer has act quant disabled
    for name, mod in qmodel.model.named_modules():
        if isinstance(mod, QuantizedLayer):
            assert not mod.quantize_activation


def test_activation_quantization_enabled_by_default():
    """By default, activation quantization should be enabled."""
    torch.manual_seed(14)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    for name, mod in qmodel.model.named_modules():
        if isinstance(mod, QuantizedLayer):
            assert mod.quantize_activation
