"""
Tests for rdquant/core/sensitivity.py

Covers:
  - Sensitivity ordering: outlier channel > uniform channel
  - Metric agreement: rank correlation > 0.7 across metrics
  - Determinism: same weight -> same sensitivity
  - Shape correctness: output shape == [N_out]
  - compute_rd_points: structure, cost formula, distortion ordering
"""

from __future__ import annotations

import torch
import pytest

from rdquant.core.sensitivity import compute_sensitivity, compute_rd_points
from rdquant.core.formats import get_bits_per_element

torch.manual_seed(42)

METRICS = ["mse", "weighted_mse", "max_over_std", "kurtosis", "range_ratio"]
FORMATS = ["NVFP4", "FP8", "FP16"]


def _rand_weight(n_out: int, n_in: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n_out, n_in, generator=g)


def _spearman_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    n = a.shape[0]
    rank_a = a.argsort().argsort().float()
    rank_b = b.argsort().argsort().float()
    d = rank_a - rank_b
    return 1.0 - 6.0 * (d ** 2).sum().item() / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestShape:
    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("n_out,n_in", [(8, 32), (64, 128), (256, 512)])
    def test_shape(self, metric, n_out, n_in):
        w = _rand_weight(n_out, n_in)
        scores = compute_sensitivity(w, metric=metric)
        assert scores.shape == (n_out,), \
            f"{metric}: expected ({n_out},), got {scores.shape}"

    @pytest.mark.parametrize("metric", METRICS)
    def test_dtype_float32(self, metric):
        w = _rand_weight(16, 64)
        scores = compute_sensitivity(w, metric=metric)
        assert scores.dtype == torch.float32


# ---------------------------------------------------------------------------
# Sensitivity ordering: outlier channel should score higher
# ---------------------------------------------------------------------------

class TestSensitivityOrdering:
    def _make_outlier_weight(self) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(0)
        w = torch.randn(2, 256, generator=g)
        w[1, 0] = 1000.0
        return w

    @pytest.mark.parametrize("metric", METRICS)
    def test_outlier_channel_higher_sensitivity(self, metric):
        w = self._make_outlier_weight()
        scores = compute_sensitivity(w, metric=metric)
        assert scores[1] > scores[0], (
            f"{metric}: outlier channel score {scores[1]:.4f} "
            f"should exceed uniform channel score {scores[0]:.4f}"
        )

    def test_mse_base_format_kwarg(self):
        """mse metric should accept different base formats."""
        w = _rand_weight(8, 64)
        for fmt in FORMATS:
            scores = compute_sensitivity(w, metric="mse", base_format=fmt)
            assert scores.shape == (8,)

    def test_default_base_format_is_nvfp4(self):
        """Default base_format should be NVFP4."""
        w = _rand_weight(8, 64)
        s1 = compute_sensitivity(w, metric="mse")
        s2 = compute_sensitivity(w, metric="mse", base_format="NVFP4")
        assert torch.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Metric agreement: rank correlation > 0.7 for most pairs
# ---------------------------------------------------------------------------

def _make_outlier_gradient_weight(n_out: int = 20, n_in: int = 256, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    w = torch.randn(n_out, n_in, generator=g)
    for j in range(n_out):
        w[j, 0] = (j + 1) * 2.0
    return w


class TestMetricAgreement:
    def test_rank_correlation_mse_weighted_mse(self):
        w = _rand_weight(32, 128)
        s1 = compute_sensitivity(w, metric="mse")
        s2 = compute_sensitivity(w, metric="weighted_mse")
        rho = _spearman_rho(s1, s2)
        assert rho > 0.7, f"mse vs weighted_mse rank corr = {rho:.3f} < 0.7"

    def test_rank_correlation_mse_max_over_std(self):
        w = _make_outlier_gradient_weight()
        s1 = compute_sensitivity(w, metric="mse")
        s2 = compute_sensitivity(w, metric="max_over_std")
        rho = _spearman_rho(s1, s2)
        assert rho > 0.7, f"mse vs max_over_std rank corr = {rho:.3f} < 0.7"

    def test_rank_correlation_mse_kurtosis(self):
        w = _make_outlier_gradient_weight()
        s1 = compute_sensitivity(w, metric="mse")
        s2 = compute_sensitivity(w, metric="kurtosis")
        rho = _spearman_rho(s1, s2)
        assert rho > 0.7, f"mse vs kurtosis rank corr = {rho:.3f} < 0.7"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("metric", METRICS)
    def test_same_input_same_output(self, metric):
        w = _rand_weight(16, 64)
        s1 = compute_sensitivity(w, metric=metric)
        s2 = compute_sensitivity(w, metric=metric)
        assert torch.allclose(s1, s2), f"{metric}: non-deterministic output"


# ---------------------------------------------------------------------------
# Invalid metric raises
# ---------------------------------------------------------------------------

def test_invalid_metric_raises():
    w = _rand_weight(4, 16)
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_sensitivity(w, metric="bogus")


# ---------------------------------------------------------------------------
# compute_rd_points
# ---------------------------------------------------------------------------

class TestComputeRDPoints:
    def test_returns_dict(self):
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w, FORMATS)
        assert isinstance(rd, dict)

    def test_keys_are_channel_indices(self):
        n_out = 8
        w = _rand_weight(n_out, 32)
        rd = compute_rd_points(w, FORMATS)
        assert set(rd.keys()) == set(range(n_out))

    def test_entries_per_channel(self):
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            assert len(entries) == len(FORMATS)

    def test_entry_fields(self):
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            for entry in entries:
                assert "format" in entry
                assert "rate" in entry
                assert "distortion" in entry
                assert "cost" in entry

    def test_cost_formula(self):
        n_in = 64
        w = _rand_weight(4, n_in)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            for entry in entries:
                expected_cost = get_bits_per_element(entry["format"]) * n_in
                assert entry["cost"] == expected_cost

    def test_rate_matches_bits_per_element(self):
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            for entry in entries:
                assert entry["rate"] == get_bits_per_element(entry["format"])

    def test_distortion_ordering(self):
        """For each channel: D(NVFP4) >= D(FP8) >= D(FP16)."""
        w = _rand_weight(16, 128)
        rd = compute_rd_points(w, FORMATS)
        fmt_order = {f: i for i, f in enumerate(FORMATS)}
        for j, entries in rd.items():
            sorted_entries = sorted(entries, key=lambda e: fmt_order[e["format"]])
            distortions = [e["distortion"] for e in sorted_entries]
            for k in range(len(distortions) - 1):
                assert distortions[k] >= distortions[k + 1] - 1e-12

    def test_distortion_nonnegative(self):
        w = _rand_weight(8, 64)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            for entry in entries:
                assert entry["distortion"] >= 0.0

    def test_fp16_distortion_near_zero(self):
        w = _rand_weight(8, 64)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            fp16_entry = next(e for e in entries if e["format"] == "FP16")
            assert fp16_entry["distortion"] < 1e-6

    def test_custom_format_subset(self):
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w, formats=["NVFP4", "FP16"])
        for j, entries in rd.items():
            assert len(entries) == 2
            fmts = {e["format"] for e in entries}
            assert fmts == {"NVFP4", "FP16"}

    @pytest.mark.parametrize("n_out,n_in", [(4, 32), (16, 128)])
    def test_various_shapes(self, n_out, n_in):
        w = _rand_weight(n_out, n_in)
        rd = compute_rd_points(w, FORMATS)
        assert len(rd) == n_out

    def test_zero_weight_matrix(self):
        w = torch.zeros(4, 32)
        rd = compute_rd_points(w, FORMATS)
        for j, entries in rd.items():
            for entry in entries:
                assert entry["distortion"] == 0.0

    def test_default_formats(self):
        """Default formats should be NVFP4, FP8, FP16."""
        w = _rand_weight(4, 32)
        rd = compute_rd_points(w)
        for j, entries in rd.items():
            fmts = {e["format"] for e in entries}
            assert fmts == {"NVFP4", "FP8", "FP16"}


# ---------------------------------------------------------------------------
# Sensitivity scores are finite
# ---------------------------------------------------------------------------

class TestFiniteScores:
    @pytest.mark.parametrize("metric", METRICS)
    def test_no_nan_or_inf(self, metric):
        w = _rand_weight(16, 128)
        scores = compute_sensitivity(w, metric=metric)
        assert torch.isfinite(scores).all(), \
            f"{metric}: got non-finite scores: {scores}"

    @pytest.mark.parametrize("metric", ["max_over_std", "kurtosis", "range_ratio"])
    def test_constant_channel_no_crash(self, metric):
        w = torch.zeros(4, 64)
        scores = compute_sensitivity(w, metric=metric)
        assert scores.shape == (4,)
        assert not torch.isnan(scores).any()
