"""
Tests for rdquant/core/allocator.py

Covers:
  - Budget satisfaction: actual avg_bits ~ budget (within +/-0.5)
  - Optimality: 2-channel exhaustive check
  - Monotonicity: larger budget -> lower total distortion
  - Extreme budgets: budget=4 -> all MXFP4; budget=8 -> all MXFP8
  - Permutation correctness: inv_permutation[permutation] = identity
  - Splits sum to N_out
  - Lambda monotonicity: higher lambda -> fewer bits
  - Group alignment to 128 channels
"""

from __future__ import annotations

import itertools

import pytest
import torch

from rdquant.core.allocator import (
    AllocationResult,
    _GROUP_ALIGNMENT,
    _pick_formats,
    _total_bits,
    _total_distortion,
    _align_groups,
    allocate,
    allocate_layer,
    sweep_budgets,
)
from rdquant.core.sensitivity import compute_rd_points

torch.manual_seed(42)

FORMATS = ["MXFP4", "MXFP6", "MXFP8"]
BITS = {"MXFP4": 4, "MXFP6": 6, "MXFP8": 8}


def _rand_weight(n_out: int, n_in: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n_out, n_in, generator=g)


def _rd_table(n_out: int, n_in: int, seed: int = 42) -> dict:
    w = _rand_weight(n_out, n_in, seed)
    return compute_rd_points(w, FORMATS)


# ---------------------------------------------------------------------------
# Budget satisfaction
# ---------------------------------------------------------------------------

class TestBudgetSatisfaction:
    @pytest.mark.parametrize("budget", [4.5, 5.0, 6.0, 7.0, 8.0])
    def test_avg_bits_within_tolerance(self, budget):
        rd = _rd_table(32, 128)
        result = allocate(rd, budget, FORMATS, 128, align_groups=False)
        assert abs(result.avg_bits - budget) <= 0.5, (
            f"budget={budget}: avg_bits={result.avg_bits:.3f}"
        )

    def test_budget_4_all_mxfp4(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 4.0, FORMATS, 64, align_groups=False)
        for j, fmt in result.assignments.items():
            assert fmt == "MXFP4"

    def test_budget_8_all_mxfp8(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 8.0, FORMATS, 64, align_groups=False)
        for j, fmt in result.assignments.items():
            assert fmt == "MXFP8"

    def test_budget_just_above_minimum(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 4.01, FORMATS, 64, align_groups=False)
        assert result.avg_bits >= 4.0

    def test_budget_just_below_maximum(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 7.99, FORMATS, 64, align_groups=False)
        assert result.avg_bits <= 8.0


# ---------------------------------------------------------------------------
# Optimality: 2-channel exhaustive check
# ---------------------------------------------------------------------------

class TestOptimality:
    def _brute_force_best(self, rd_table: dict, budget_avg_bits: float, n_in: int) -> float:
        channels = sorted(rd_table.keys())
        n_out = len(channels)
        budget_total = budget_avg_bits * n_out * n_in
        best_dist = float("inf")
        for assignment in itertools.product(FORMATS, repeat=n_out):
            fmt_map = {ch: assignment[i] for i, ch in enumerate(channels)}
            total_bits = _total_bits(rd_table, fmt_map)
            if total_bits <= budget_total + 1e-6:
                dist = _total_distortion(rd_table, fmt_map)
                if dist < best_dist:
                    best_dist = dist
        return best_dist

    @pytest.mark.parametrize("budget", [5.0, 6.0, 7.0, 8.0])
    def test_2_channel_optimal(self, budget):
        rd = _rd_table(2, 64)
        result = allocate(rd, budget, FORMATS, 64, align_groups=False)
        brute_dist = self._brute_force_best(rd, budget, 64)
        assert result.total_distortion <= brute_dist + 1e-9 * max(brute_dist, 1.0)

    @pytest.mark.parametrize("budget", [5.0, 7.0])
    def test_4_channel_optimal(self, budget):
        rd = _rd_table(4, 64)
        result = allocate(rd, budget, FORMATS, 64, align_groups=False)
        brute_dist = self._brute_force_best(rd, budget, 64)
        assert result.total_distortion <= brute_dist + 1e-9 * max(brute_dist, 1.0)


# ---------------------------------------------------------------------------
# Monotonicity: larger budget -> lower total distortion
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_distortion_decreases_with_budget(self):
        rd = _rd_table(16, 64)
        budgets = [4.5, 5.0, 6.0, 7.0, 8.0]
        distortions = [allocate(rd, b, FORMATS, 64, align_groups=False).total_distortion for b in budgets]
        for i in range(len(distortions) - 1):
            assert distortions[i] >= distortions[i + 1] - 1e-12

    def test_avg_bits_increases_with_budget(self):
        rd = _rd_table(16, 64)
        budgets = [4.5, 5.0, 6.0, 7.0]
        avg_bits = [allocate(rd, b, FORMATS, 64, align_groups=False).avg_bits for b in budgets]
        for i in range(len(avg_bits) - 1):
            assert avg_bits[i] <= avg_bits[i + 1] + 0.5


# ---------------------------------------------------------------------------
# Permutation correctness
# ---------------------------------------------------------------------------

class TestPermutation:
    def _check_permutation(self, result: AllocationResult, n_out: int):
        perm = result.permutation
        inv = result.inv_permutation
        assert perm.shape == (n_out,)
        assert inv.shape == (n_out,)
        identity = inv[perm]
        expected = torch.arange(n_out)
        assert torch.equal(identity, expected)
        identity2 = perm[inv]
        assert torch.equal(identity2, expected)

    @pytest.mark.parametrize("budget", [4.5, 6.0, 8.0])
    def test_permutation_is_inverse(self, budget):
        rd = _rd_table(32, 64)
        result = allocate(rd, budget, FORMATS, 64, align_groups=False)
        self._check_permutation(result, 32)

    def test_permutation_groups_by_format(self):
        rd = _rd_table(32, 64)
        result = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
        perm = result.permutation.tolist()
        prev_fmt_idx = -1
        for pos, orig_ch in enumerate(perm):
            fmt = result.assignments[orig_ch]
            fmt_idx = FORMATS.index(fmt) if fmt in FORMATS else -1
            assert fmt_idx >= prev_fmt_idx

    def test_splits_sum_to_n_out(self):
        n_out = 32
        rd = _rd_table(n_out, 64)
        result = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
        total = sum(result.splits.values())
        assert total == n_out

    def test_splits_nonnegative(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
        for fmt, cnt in result.splits.items():
            assert cnt >= 0


# ---------------------------------------------------------------------------
# Lambda monotonicity: higher lambda -> fewer bits
# ---------------------------------------------------------------------------

class TestLambdaMonotonicity:
    def test_higher_lambda_fewer_bits(self):
        rd = _rd_table(16, 64)
        lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
        bits_list = []
        for lam in lambdas:
            a = _pick_formats(rd, lam, FORMATS)
            bits_list.append(_total_bits(rd, a))
        for i in range(len(bits_list) - 1):
            assert bits_list[i] >= bits_list[i + 1] - 1e-6


# ---------------------------------------------------------------------------
# AllocationResult fields
# ---------------------------------------------------------------------------

class TestAllocationResultFields:
    def test_all_channels_assigned(self):
        n_out = 16
        rd = _rd_table(n_out, 64)
        result = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
        assert len(result.assignments) == n_out

    def test_assignments_valid_formats(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
        for j, fmt in result.assignments.items():
            assert fmt in FORMATS

    def test_avg_bits_nonnegative(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64)
        assert result.avg_bits >= 0.0

    def test_total_distortion_nonnegative(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64)
        assert result.total_distortion >= 0.0

    def test_lambda_star_nonnegative(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64)
        assert result.lambda_star >= 0.0

    def test_format_stats_keys(self):
        rd = _rd_table(16, 64)
        result = allocate(rd, 6.0, FORMATS, 64)
        for fmt in FORMATS:
            assert fmt in result.format_stats
            stats = result.format_stats[fmt]
            assert "n_channels" in stats
            assert "avg_mse" in stats
            assert "total_bits" in stats

    def test_format_stats_n_channels_sum(self):
        n_out = 16
        rd = _rd_table(n_out, 64)
        result = allocate(rd, 6.0, FORMATS, 64)
        total = sum(s["n_channels"] for s in result.format_stats.values())
        assert total == n_out

    def test_format_stats_total_bits_consistent(self):
        n_in = 64
        rd = _rd_table(16, n_in)
        result = allocate(rd, 6.0, FORMATS, n_in)
        for fmt, stats in result.format_stats.items():
            expected = BITS[fmt] * n_in * stats["n_channels"]
            assert stats["total_bits"] == expected


# ---------------------------------------------------------------------------
# allocate_layer convenience function
# ---------------------------------------------------------------------------

class TestAllocateLayer:
    def test_returns_allocation_result(self):
        w = _rand_weight(16, 64)
        result = allocate_layer(w, 6.0, align_groups=False)
        assert isinstance(result, AllocationResult)

    def test_shape_consistency(self):
        n_out, n_in = 16, 64
        w = _rand_weight(n_out, n_in)
        result = allocate_layer(w, 6.0, align_groups=False)
        assert result.permutation.shape == (n_out,)
        assert result.inv_permutation.shape == (n_out,)
        assert len(result.assignments) == n_out

    @pytest.mark.parametrize("budget", [4.5, 6.0, 8.0])
    def test_budget_satisfaction(self, budget):
        w = _rand_weight(32, 128)
        result = allocate_layer(w, budget, align_groups=False)
        assert abs(result.avg_bits - budget) <= 0.5


# ---------------------------------------------------------------------------
# sweep_budgets
# ---------------------------------------------------------------------------

class TestSweepBudgets:
    def test_returns_list_of_correct_length(self):
        w = _rand_weight(16, 64)
        budgets = [4.0, 6.0, 8.0]
        results = sweep_budgets(w, budgets, align_groups=False)
        assert len(results) == len(budgets)

    def test_all_results_are_allocation_result(self):
        w = _rand_weight(16, 64)
        results = sweep_budgets(w, [4.0, 6.0, 8.0], align_groups=False)
        for r in results:
            assert isinstance(r, AllocationResult)

    def test_distortion_monotone_over_budgets(self):
        w = _rand_weight(16, 64)
        budgets = [4.0, 5.0, 6.0, 7.0, 8.0]
        results = sweep_budgets(w, budgets, align_groups=False)
        for i in range(len(results) - 1):
            assert results[i].total_distortion >= results[i + 1].total_distortion - 1e-12

    def test_extreme_budgets(self):
        w = _rand_weight(16, 64)
        results = sweep_budgets(w, [4.0, 8.0], align_groups=False)
        for fmt in results[0].assignments.values():
            assert fmt == "MXFP4"
        for fmt in results[1].assignments.values():
            assert fmt == "MXFP8"


# ---------------------------------------------------------------------------
# n_elements_per_channel inference
# ---------------------------------------------------------------------------

def test_infer_n_elements_per_channel():
    rd = _rd_table(8, 64)
    result = allocate(rd, 6.0, FORMATS, align_groups=False)
    assert isinstance(result, AllocationResult)
    assert result.avg_bits > 0


# ---------------------------------------------------------------------------
# Format subset
# ---------------------------------------------------------------------------

class TestFormatSubset:
    def test_two_format_subset(self):
        rd = _rd_table(8, 64)
        r4 = allocate(rd, 4.0, ["MXFP4", "MXFP8"], 64, align_groups=False)
        r8 = allocate(rd, 8.0, ["MXFP4", "MXFP8"], 64, align_groups=False)
        for fmt in r4.assignments.values():
            assert fmt == "MXFP4"
        for fmt in r8.assignments.values():
            assert fmt == "MXFP8"

    def test_single_format(self):
        rd = _rd_table(8, 64)
        result = allocate(rd, 6.0, ["MXFP6"], 64, align_groups=False)
        for fmt in result.assignments.values():
            assert fmt == "MXFP6"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    rd = _rd_table(16, 64)
    r1 = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
    r2 = allocate(rd, 6.0, FORMATS, 64, align_groups=False)
    assert r1.assignments == r2.assignments
    assert abs(r1.avg_bits - r2.avg_bits) < 1e-12
    assert abs(r1.lambda_star - r2.lambda_star) < 1e-12


# ---------------------------------------------------------------------------
# Group alignment (128-channel)
# ---------------------------------------------------------------------------

class TestGroupAlignment:
    def test_alignment_constant(self):
        assert _GROUP_ALIGNMENT == 128

    def test_aligned_groups_are_multiples_of_128(self):
        """When align_groups=True and n_out is large enough, each group should be
        a multiple of 128 (or zero)."""
        # Use 256 channels = 2 * 128
        w = _rand_weight(256, 64, seed=0)
        result = allocate_layer(w, 6.0, align_groups=True)
        for fmt, n_ch in result.splits.items():
            if n_ch > 0:
                assert n_ch % 128 == 0, (
                    f"Format {fmt}: {n_ch} channels not aligned to 128"
                )

    def test_aligned_splits_sum(self):
        w = _rand_weight(256, 64, seed=0)
        result = allocate_layer(w, 6.0, align_groups=True)
        total = sum(result.splits.values())
        assert total == 256

    def test_alignment_with_small_matrix_skipped(self):
        """For matrices smaller than 128, alignment should not crash."""
        w = _rand_weight(16, 64)
        result = allocate_layer(w, 6.0, align_groups=True)
        assert isinstance(result, AllocationResult)
        assert sum(result.splits.values()) == 16

    def test_alignment_preserves_permutation_validity(self):
        """Permutation should still be valid after alignment."""
        w = _rand_weight(256, 64, seed=0)
        result = allocate_layer(w, 6.0, align_groups=True)
        perm = result.permutation
        inv = result.inv_permutation
        assert torch.equal(inv[perm], torch.arange(256))

    def test_align_groups_function_direct(self):
        """Direct test of _align_groups."""
        # Create assignments with non-aligned groups
        assignments = {i: "MXFP4" for i in range(200)}
        for i in range(200, 256):
            assignments[i] = "MXFP8"
        rd = _rd_table(256, 64, seed=0)
        aligned = _align_groups(assignments, rd, FORMATS)
        # Count per format
        counts = {}
        for fmt in aligned.values():
            counts[fmt] = counts.get(fmt, 0) + 1
        for fmt, cnt in counts.items():
            if cnt > 0:
                assert cnt % 128 == 0 or len(aligned) < 128, (
                    f"Format {fmt}: {cnt} channels not aligned"
                )

    def test_align_groups_no_change_when_already_aligned(self):
        """If groups are already aligned, should not change assignments."""
        assignments = {i: "MXFP4" for i in range(128)}
        for i in range(128, 256):
            assignments[i] = "MXFP8"
        rd = _rd_table(256, 64, seed=0)
        aligned = _align_groups(assignments, rd, FORMATS)
        counts = {}
        for fmt in aligned.values():
            counts[fmt] = counts.get(fmt, 0) + 1
        assert counts.get("MXFP4", 0) == 128
        assert counts.get("MXFP8", 0) == 128

    def test_alignment_disabled(self):
        """align_groups=False should skip alignment."""
        w = _rand_weight(256, 64, seed=0)
        result = allocate_layer(w, 6.0, align_groups=False)
        # Don't require alignment
        assert isinstance(result, AllocationResult)
