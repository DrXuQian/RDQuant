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
    """quantize_model() should return a QuantizedModel."""
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    assert isinstance(qmodel, QuantizedModel)


def test_output_shape_unchanged():
    """Output shape should be identical before and after quantization."""
    torch.manual_seed(0)
    model = _make_tiny()
    x = torch.randn(4, 64)
    with torch.no_grad():
        original_out = model(x)

    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        q_out = qmodel(x)

    assert q_out.shape == original_out.shape, (
        f"Shape mismatch: {q_out.shape} vs {original_out.shape}"
    )


def test_forward_produces_finite_outputs():
    """Forward pass should produce finite (non-NaN, non-Inf) outputs."""
    torch.manual_seed(1)
    model = _make_tiny()
    x = torch.randn(4, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_budget_approximately_met():
    """Average bits across quantized layers should be within ±1 of target."""
    torch.manual_seed(2)
    model = _make_tiny()
    target_bits = 5.3
    qmodel = quantize_model(model, budget_avg_bits=target_bits)

    # Compute actual average bits across all quantized layers
    total_bits = 0.0
    total_params = 0
    for name, result in qmodel.layer_info.items():
        for fmt, stats in result.format_stats.items():
            total_bits += stats["total_bits"]
        # n_params = n_out * n_in
        n_out = sum(result.splits.values())
        if n_out > 0 and result.avg_bits > 0:
            n_in = int(round(
                sum(s["total_bits"] for s in result.format_stats.values()) /
                (n_out * result.avg_bits)
            ))
            total_params += n_out * n_in

    if total_params > 0:
        actual_bits = total_bits / total_params
        assert abs(actual_bits - target_bits) < 1.0, (
            f"avg_bits={actual_bits:.2f} is more than 1 bit away from target={target_bits}"
        )


def test_ignore_patterns():
    """Layers matching ignore patterns should remain as nn.Linear."""
    torch.manual_seed(3)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3, ignore=["fc1"])

    # fc1 should not be in layer_info (was ignored)
    assert "fc1" not in qmodel.layer_info, "fc1 should be ignored"
    # fc2 and fc3 should be quantized
    assert "fc2" in qmodel.layer_info
    assert "fc3" in qmodel.layer_info
    # fc1 should still be an nn.Linear in the model
    assert isinstance(qmodel.model.fc1, nn.Linear), "fc1 should remain as nn.Linear"
    # fc2 should be a QuantizedLayer
    assert isinstance(qmodel.model.fc2, QuantizedLayer), "fc2 should be QuantizedLayer"


def test_ignore_all_layers():
    """Ignoring all layers should return a model with no quantized layers."""
    torch.manual_seed(4)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3, ignore=["fc1", "fc2", "fc3"])
    assert len(qmodel.layer_info) == 0
    # All layers remain as nn.Linear
    assert isinstance(qmodel.model.fc1, nn.Linear)
    assert isinstance(qmodel.model.fc2, nn.Linear)
    assert isinstance(qmodel.model.fc3, nn.Linear)


def test_no_bias_model():
    """Quantization should work for layers without bias."""
    torch.manual_seed(5)
    model = TinyModelNoBias()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 16)


def test_per_layer_budget():
    """per_layer_budget=True should also return a QuantizedModel with finite outputs."""
    torch.manual_seed(6)
    model = _make_tiny()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=5.3, per_layer_budget=True)
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 8)


def test_save_load_roundtrip():
    """save_pretrained / from_pretrained round-trip should produce matching outputs."""
    torch.manual_seed(7)
    model = _make_tiny()
    x = torch.randn(2, 64)

    qmodel = quantize_model(model, budget_avg_bits=5.3)
    with torch.no_grad():
        out1 = qmodel(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        qmodel.save_pretrained(tmpdir)
        # Check that config file was written
        assert os.path.isfile(os.path.join(tmpdir, "quantization_config.json"))

        qmodel2 = QuantizedModel.from_pretrained(tmpdir)
        with torch.no_grad():
            out2 = qmodel2(x)

    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=1e-5), (
        f"Outputs differ after save/load: max diff = {(out1 - out2).abs().max().item()}"
    )


def test_print_summary_runs(capsys):
    """print_summary() should not crash and should print something."""
    torch.manual_seed(8)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    qmodel.print_summary()
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "print_summary produced no output"


def test_formats_subset():
    """Quantization should work with a subset of formats."""
    torch.manual_seed(9)
    model = _make_tiny()
    x = torch.randn(2, 64)
    qmodel = quantize_model(model, budget_avg_bits=7.0, formats=["MXFP8", "FP16"])
    with torch.no_grad():
        out = qmodel(x)
    assert torch.isfinite(out).all()


def test_low_budget_mostly_nvfp4():
    """At very low budget, most channels should be assigned NVFP4."""
    torch.manual_seed(10)
    model = TinyModel()
    qmodel = quantize_model(model, budget_avg_bits=4.0)
    # Check that NVFP4 channels dominate
    for name, result in qmodel.layer_info.items():
        total_ch = sum(result.splits.values())
        nvfp4_ch = result.splits.get("NVFP4", 0)
        assert nvfp4_ch >= total_ch * 0.5, (
            f"Layer {name}: expected NVFP4 dominance at budget=4.0, "
            f"got {nvfp4_ch}/{total_ch}"
        )


def test_high_budget_mostly_fp16():
    """At very high budget, most channels should be assigned FP16."""
    torch.manual_seed(11)
    model = TinyModel()
    qmodel = quantize_model(model, budget_avg_bits=16.0)
    for name, result in qmodel.layer_info.items():
        total_ch = sum(result.splits.values())
        fp16_ch = result.splits.get("FP16", 0)
        assert fp16_ch == total_ch, (
            f"Layer {name}: expected all FP16 at budget=16.0, "
            f"got {fp16_ch}/{total_ch}"
        )


def test_quantized_layer_repr():
    """QuantizedLayer extra_repr should include avg_bits."""
    torch.manual_seed(12)
    model = _make_tiny()
    qmodel = quantize_model(model, budget_avg_bits=5.3)
    repr_str = repr(qmodel.model.fc1)
    assert "avg_bits" in repr_str
