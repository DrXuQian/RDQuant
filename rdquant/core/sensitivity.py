"""
Data-free channel sensitivity metrics.

All functions take a weight matrix W of shape [N_out, N_in] and return
a sensitivity score per output channel of shape [N_out].

Higher sensitivity = channel is harder to quantize = needs more bits.

Available metrics:
  - "mse": Direct quantization MSE at the lowest format (MXFP4).
           Most accurate, slightly slower.
  - "weighted_mse": MSE x ||W_j||^2. Accounts for weight magnitude.
  - "max_over_std": max(|W_j|) / std(W_j). Measures outlier severity.
  - "kurtosis": Excess kurtosis of W_j. Measures tail heaviness.
  - "range_ratio": (max - min) / mean_abs. Measures dynamic range.

Recommended default: "mse" (best correlation with actual quantization loss).
"""

from __future__ import annotations

import torch

from rdquant.core.formats import compute_mse, compute_mse_2d, get_bits_per_element


def _sensitivity_mse(weight: torch.Tensor, base_format: str) -> torch.Tensor:
    """Per-channel quantization MSE at base_format."""
    return compute_mse_2d(weight, base_format)


def _sensitivity_weighted_mse(weight: torch.Tensor, base_format: str) -> torch.Tensor:
    """Per-channel MSE x ||W_j||^2."""
    mse = _sensitivity_mse(weight, base_format)
    norm_sq = (weight.float() ** 2).sum(dim=1)
    return mse * norm_sq


def _sensitivity_max_over_std(weight: torch.Tensor) -> torch.Tensor:
    """max(|W_j|) / std(W_j). Measures outlier severity."""
    abs_max = weight.float().abs().max(dim=1).values
    std = weight.float().std(dim=1)
    # Avoid division by zero
    std = std.clamp(min=1e-12)
    return abs_max / std


def _sensitivity_kurtosis(weight: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis of each output channel."""
    w = weight.double()  # use float64 to avoid underflow in var**2
    mean = w.mean(dim=1, keepdim=True)
    centered = w - mean
    var = (centered ** 2).mean(dim=1)
    m4 = (centered ** 4).mean(dim=1)
    # Excess kurtosis = m4/var^2 - 3; zero-variance channels get 0 (Gaussian default)
    var = var.clamp(min=1e-40)
    result = (m4 / (var ** 2) - 3.0).float()
    return result.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)


def _sensitivity_range_ratio(weight: torch.Tensor) -> torch.Tensor:
    """(max - min) / mean_abs. Measures dynamic range."""
    w = weight.float()
    rng = w.max(dim=1).values - w.min(dim=1).values
    mean_abs = w.abs().mean(dim=1).clamp(min=1e-12)
    return rng / mean_abs


_METRIC_FNS = {
    "mse":          lambda w, fmt: _sensitivity_mse(w, fmt),
    "weighted_mse": lambda w, fmt: _sensitivity_weighted_mse(w, fmt),
    "max_over_std": lambda w, _fmt: _sensitivity_max_over_std(w),
    "kurtosis":     lambda w, _fmt: _sensitivity_kurtosis(w),
    "range_ratio":  lambda w, _fmt: _sensitivity_range_ratio(w),
}


@torch.inference_mode()
def compute_sensitivity(
    weight: torch.Tensor,
    metric: str = "mse",
    base_format: str = "MXFP4",
) -> torch.Tensor:
    """Compute per-channel sensitivity scores for a weight matrix.

    Args:
        weight: Float tensor of shape [N_out, N_in].
        metric: One of "mse", "weighted_mse", "max_over_std", "kurtosis",
            "range_ratio".
        base_format: Format used for MSE-based metrics (default "MXFP4").

    Returns:
        Float32 tensor of shape [N_out] with per-channel sensitivity scores.
        Higher score = harder to quantize = needs more bits.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_METRIC_FNS)}")
    return _METRIC_FNS[metric](weight, base_format)


@torch.inference_mode()
def compute_rd_points(
    weight: torch.Tensor,
    formats: list[str] = ["MXFP4", "MXFP6", "MXFP8"],
) -> dict:
    """Compute (rate, distortion) pairs for every channel x format combination.

    For each output channel j and each format f, computes:
      - rate    = bits_per_element(f)   (bits per weight element)
      - distortion = MSE(W_j, f)
      - cost    = bits_per_element(f) * N_in   (total bits for this channel)

    Args:
        weight: Float tensor of shape [N_out, N_in].
        formats: Ordered list of format names from lowest to highest precision.

    Returns:
        rd_table: dict mapping channel_idx (int) -> list of dicts, each with keys
            {"format", "rate", "distortion", "cost"}.
    """
    n_out, n_in = weight.shape

    # Batch: compute per-channel MSE for each format in one shot
    mse_per_fmt: dict[str, torch.Tensor] = {}
    for fmt in formats:
        mse_per_fmt[fmt] = compute_mse_2d(weight, fmt)  # [N_out]

    rd_table: dict[int, list[dict]] = {}
    for j in range(n_out):
        entries = []
        for fmt in formats:
            bits = get_bits_per_element(fmt)
            entries.append({
                "format": fmt,
                "rate": bits,
                "distortion": mse_per_fmt[fmt][j].item(),
                "cost": bits * n_in,
            })
        rd_table[j] = entries

    return rd_table
