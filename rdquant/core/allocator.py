"""
Rate-Distortion optimal channel-wise format allocation.

Given:
  - R-D points for each channel (from sensitivity.py)
  - A bit budget (average bits per element)

Find:
  - Format assignment for each channel that minimizes total distortion
    subject to the bit budget constraint.

Algorithm:
  1. For each channel, compute Lagrangian cost: D_j(f) + lambda * C_j(f)
     for each format f, where D = distortion (MSE), C = bit cost.
  2. Binary search on lambda to find the value where total cost = budget.
  3. At optimal lambda*, each channel independently picks its best format.
  4. After allocation, pad each format group to a multiple of 128 channels
     by promoting channels at boundaries to the next-higher format.

Returns:
  - AllocationResult with per-channel format assignments, permutation
    indices (sorted by format for efficient grouped GEMM), and summary stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rdquant.core.sensitivity import compute_rd_points
from rdquant.core.formats import get_bits_per_element

# Canonical format order: lowest bits first
_FORMAT_ORDER = ["NVFP4", "INT4", "FP8", "INT8", "FP16"]

# Group alignment: each format group must be a multiple of this many channels
_GROUP_ALIGNMENT = 128


@dataclass
class AllocationResult:
    """Result of R-D optimal format allocation for one linear layer."""

    assignments: dict[int, str]       # channel_idx -> format_name
    permutation: torch.Tensor         # [N_out] reorder indices (group by format)
    inv_permutation: torch.Tensor     # [N_out] inverse reorder
    splits: dict[str, int]            # format_name -> num_channels
    avg_bits: float                   # actual average bits achieved
    total_distortion: float           # sum of MSE across all channels
    lambda_star: float                # optimal Lagrange multiplier

    # Per-format stats: {format: {n_channels, avg_mse, total_bits}}
    format_stats: dict[str, dict] = field(default_factory=dict)


def _pick_formats(
    rd_table: dict,
    lam: float,
    formats: list[str],
    dist_weight: float = 1.0,
) -> dict[int, str]:
    """For each channel, pick the format that minimises w*D_j(f) + lambda*C_j(f).

    Args:
        rd_table: Output of compute_rd_points().
        lam: Lagrange multiplier (non-negative).
        formats: Allowed format names.
        dist_weight: Scalar multiplier on distortion for this layer.
            Higher weight makes the layer prefer higher-precision formats.

    Returns:
        Dict mapping channel_idx -> chosen format name.
    """
    assignments: dict[int, str] = {}
    format_set = set(formats)
    for j, entries in rd_table.items():
        best_fmt = None
        best_cost = float("inf")
        for e in entries:
            if e["format"] not in format_set:
                continue
            lagrangian = dist_weight * e["distortion"] + lam * e["cost"]
            if lagrangian < best_cost:
                best_cost = lagrangian
                best_fmt = e["format"]
        assignments[j] = best_fmt
    return assignments


def _total_bits(rd_table: dict, assignments: dict[int, str]) -> float:
    """Sum of bit costs across all channels under the given assignment."""
    total = 0.0
    for j, fmt in assignments.items():
        for e in rd_table[j]:
            if e["format"] == fmt:
                total += e["cost"]
                break
    return total


def _total_distortion(
    rd_table: dict,
    assignments: dict[int, str],
    dist_weight: float = 1.0,
) -> float:
    """Weighted sum of distortions across all channels under the given assignment."""
    total = 0.0
    for j, fmt in assignments.items():
        for e in rd_table[j]:
            if e["format"] == fmt:
                total += dist_weight * e["distortion"]
                break
    return total


def _align_groups(
    assignments: dict[int, str],
    rd_table: dict,
    formats: list[str],
    alignment: int = _GROUP_ALIGNMENT,
) -> dict[int, str]:
    """Pad each format group to a multiple of `alignment` channels.

    Promotion strategy: for each format group that is not aligned, promote
    channels at the boundary (those with lowest distortion difference to
    the next-higher format) to the next-higher format. The highest format
    group absorbs any remainder from lower groups.

    Args:
        assignments: Current channel -> format assignments.
        rd_table: R-D table for distortion lookups.
        formats: Ordered format list (lowest bits first).
        alignment: Target alignment (default 128).

    Returns:
        Updated assignments dict.
    """
    if alignment <= 1:
        return assignments

    n_out = len(assignments)
    if n_out < alignment:
        # If total channels < alignment, no alignment needed
        return assignments

    # Build format groups preserving channel order
    active_formats = [f for f in _FORMAT_ORDER if f in formats]
    if len(active_formats) <= 1:
        return assignments

    groups: dict[str, list[int]] = {f: [] for f in active_formats}
    for j in sorted(assignments.keys()):
        groups[assignments[j]].append(j)

    # Get distortion per channel per format for promotion decisions
    def _get_distortion(ch: int, fmt: str) -> float:
        for e in rd_table[ch]:
            if e["format"] == fmt:
                return e["distortion"]
        return float("inf")

    # Process from lowest to highest format
    new_assignments = dict(assignments)
    for i in range(len(active_formats) - 1):
        fmt_lo = active_formats[i]
        fmt_hi = active_formats[i + 1]
        group = groups[fmt_lo]
        n_ch = len(group)
        if n_ch == 0:
            continue

        remainder = n_ch % alignment
        if remainder == 0:
            continue

        # Need to promote `remainder` channels to fmt_hi
        n_promote = remainder

        # Pick channels with smallest distortion increase when promoted
        promote_costs = []
        for ch in group:
            d_lo = _get_distortion(ch, fmt_lo)
            d_hi = _get_distortion(ch, fmt_hi)
            promote_costs.append((d_hi - d_lo, ch))

        # Sort by distortion increase (ascending) — promote cheapest first
        promote_costs.sort()
        to_promote = [ch for _, ch in promote_costs[:n_promote]]

        for ch in to_promote:
            new_assignments[ch] = fmt_hi
            groups[fmt_lo].remove(ch)
            groups[fmt_hi].append(ch)

    return new_assignments


def _build_result(
    rd_table: dict,
    assignments: dict[int, str],
    n_elements_per_channel: int,
    lambda_star: float,
    formats: list[str],
) -> AllocationResult:
    """Build AllocationResult from assignments."""
    n_out = len(assignments)
    n_total_elements = n_out * n_elements_per_channel

    # Compute permutation: group by format in canonical order
    # Within each group preserve original channel order
    active_formats = [f for f in _FORMAT_ORDER if f in formats]

    groups: dict[str, list[int]] = {f: [] for f in active_formats}
    for j in sorted(assignments.keys()):
        groups[assignments[j]].append(j)

    perm_list: list[int] = []
    for f in active_formats:
        perm_list.extend(groups[f])

    permutation = torch.tensor(perm_list, dtype=torch.long)
    inv_permutation = torch.empty_like(permutation)
    inv_permutation[permutation] = torch.arange(n_out, dtype=torch.long)

    splits = {f: len(groups[f]) for f in active_formats}

    # Compute total bits used
    total_bits_used = _total_bits(rd_table, assignments)
    avg_bits = total_bits_used / n_total_elements if n_total_elements > 0 else 0.0

    total_dist = _total_distortion(rd_table, assignments)

    # Per-format stats
    format_stats: dict[str, dict] = {}
    for f in active_formats:
        ch_list = groups[f]
        if not ch_list:
            format_stats[f] = {"n_channels": 0, "avg_mse": 0.0, "total_bits": 0}
            continue
        mses = []
        for j in ch_list:
            for e in rd_table[j]:
                if e["format"] == f:
                    mses.append(e["distortion"])
                    break
        n_ch = len(ch_list)
        format_stats[f] = {
            "n_channels": n_ch,
            "avg_mse": sum(mses) / n_ch if n_ch else 0.0,
            "total_bits": get_bits_per_element(f) * n_elements_per_channel * n_ch,
        }

    return AllocationResult(
        assignments=assignments,
        permutation=permutation,
        inv_permutation=inv_permutation,
        splits=splits,
        avg_bits=avg_bits,
        total_distortion=total_dist,
        lambda_star=lambda_star,
        format_stats=format_stats,
    )


def allocate(
    rd_table: dict,
    budget_avg_bits: float,
    formats: list[str] = None,
    n_elements_per_channel: int = None,
    align_groups: bool = True,
) -> AllocationResult:
    """Find format assignment minimising total distortion subject to a bit budget.

    Uses a Lagrangian lambda-sweep with binary search:
      - lambda=0  -> all channels pick FP16 (max bits, min distortion)
      - lambda=inf -> all channels pick NVFP4 (min bits, max distortion)
      - Binary search finds lambda* where total_bits ~ budget

    After allocation, format groups are aligned to multiples of 128 channels
    by promoting boundary channels to the next-higher format.

    Args:
        rd_table: Output of :func:`~rdquant.core.sensitivity.compute_rd_points`.
        budget_avg_bits: Target average bits per element (e.g. 5.3).
        formats: Allowed format names. Defaults to ["NVFP4","FP8","FP16"].
        n_elements_per_channel: Number of elements per channel (N_in). Inferred
            from rd_table cost entries if not provided.
        align_groups: Whether to align format groups to multiples of 128 channels.
            Defaults to True.

    Returns:
        :class:`AllocationResult` with per-channel assignments and statistics.
    """
    if formats is None:
        formats = ["NVFP4", "FP8", "FP16"]

    n_out = len(rd_table)
    if n_out == 0:
        raise ValueError("rd_table is empty")

    # Infer n_elements_per_channel from the cost of the first format entry
    if n_elements_per_channel is None:
        first_entries = next(iter(rd_table.values()))
        for e in first_entries:
            if e["rate"] > 0:
                n_elements_per_channel = e["cost"] // e["rate"]
                break

    budget_total_bits = budget_avg_bits * n_out * n_elements_per_channel

    # Compute min/max possible total bits
    min_bits_assignments = _pick_formats(rd_table, lam=1e18, formats=formats)
    max_bits_assignments = _pick_formats(rd_table, lam=0.0, formats=formats)
    min_total = _total_bits(rd_table, min_bits_assignments)
    max_total = _total_bits(rd_table, max_bits_assignments)

    # If budget is outside feasible range, clamp and return immediately
    if budget_total_bits <= min_total:
        assignments = min_bits_assignments
        if align_groups:
            assignments = _align_groups(assignments, rd_table, formats)
        return _build_result(rd_table, assignments, n_elements_per_channel, 1e18, formats)

    if budget_total_bits >= max_total:
        assignments = max_bits_assignments
        if align_groups:
            assignments = _align_groups(assignments, rd_table, formats)
        return _build_result(rd_table, assignments, n_elements_per_channel, 0.0, formats)

    # Binary search on lambda
    lambda_lo = 0.0
    lambda_hi = 1.0

    # Expand upper bound until total_bits(lambda_hi) <= budget
    for _ in range(64):
        a = _pick_formats(rd_table, lambda_hi, formats)
        if _total_bits(rd_table, a) <= budget_total_bits:
            break
        lambda_hi *= 2.0

    # Binary search for 64 iterations
    for _ in range(64):
        lam_mid = (lambda_lo + lambda_hi) / 2.0
        a = _pick_formats(rd_table, lam_mid, formats)
        bits = _total_bits(rd_table, a)
        if bits > budget_total_bits:
            lambda_lo = lam_mid
        else:
            lambda_hi = lam_mid

    lambda_star = (lambda_lo + lambda_hi) / 2.0
    assignments = _pick_formats(rd_table, lambda_star, formats)

    if align_groups:
        assignments = _align_groups(assignments, rd_table, formats)

    return _build_result(rd_table, assignments, n_elements_per_channel, lambda_star, formats)


def allocate_layer(
    weight: torch.Tensor,
    budget_avg_bits: float,
    formats: list[str] = None,
    sensitivity_metric: str = "mse",
    align_groups: bool = True,
) -> AllocationResult:
    """Compute R-D points and allocate formats for a single layer.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        budget_avg_bits: Target average bits per element.
        formats: Allowed format names. Defaults to ["NVFP4","FP8","FP16"].
        sensitivity_metric: Unused (allocation uses MSE directly via rd_table).
        align_groups: Whether to align format groups to multiples of 128 channels.

    Returns:
        :class:`AllocationResult` for this layer.
    """
    if formats is None:
        formats = ["NVFP4", "FP8", "FP16"]
    with torch.inference_mode():
        rd_table = compute_rd_points(weight, formats)
    return allocate(rd_table, budget_avg_bits, formats, weight.shape[1], align_groups=align_groups)


def sweep_budgets(
    weight: torch.Tensor,
    budgets: list[float],
    formats: list[str] = None,
    align_groups: bool = True,
) -> list[AllocationResult]:
    """Run allocation at multiple bit budgets, computing rd_table only once.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        budgets: List of target average bits per element (e.g. [4, 5, 6, 7, 8]).
        formats: Allowed format names. Defaults to ["NVFP4","FP8","FP16"].
        align_groups: Whether to align format groups to multiples of 128 channels.

    Returns:
        List of :class:`AllocationResult`, one per budget entry.
    """
    if formats is None:
        formats = ["NVFP4", "FP8", "FP16"]
    with torch.inference_mode():
        rd_table = compute_rd_points(weight, formats)
    n_in = weight.shape[1]
    return [allocate(rd_table, b, formats, n_in, align_groups=align_groups) for b in budgets]
