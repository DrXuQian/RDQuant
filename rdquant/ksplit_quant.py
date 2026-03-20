"""
K-dimension mixed-precision quantization: different K-groups get INT4 or INT8.

Unlike N-split (per output channel), K-split assigns precision per group_size=128
block along the input (reduction) dimension. The assignment is uniform across all
output channels — every output channel uses the same K-group precision map.

Combined with AWQ scaling, activation-important K-groups automatically receive
INT8 (AWQ upscales them → larger MSE under INT4 → R-D allocator picks INT8).

For single-kernel deployment: INT8 K-groups are decomposed into 2× INT4 K-groups
(same math as N-split decomposition), expanding K to K_combined.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# K-group R-D allocation
# ---------------------------------------------------------------------------

@dataclass
class KSplitAllocation:
    """Result of K-dimension R-D allocation for one layer."""
    group_formats: list[str]       # [G] format per K-group ("INT4" or "INT8")
    n_int4_groups: int
    n_int8_groups: int
    k_perm: torch.Tensor           # [K] reorder: INT4 groups first, INT8 groups last
    k_inv_perm: torch.Tensor       # [K] inverse reorder
    avg_bits: float
    total_distortion: float
    lambda_star: float


def _compute_kgroup_rd(
    weight: torch.Tensor,   # [N, K] float, AWQ-scaled
    group_size: int = 128,
) -> list[dict]:
    """Compute R-D points for each K-group.

    For each group g, distortion = sum of per-channel MSE under INT4/INT8
    quantization of that group's 128 columns across all N output channels.

    Returns:
        List of G dicts, each with keys 'INT4' and 'INT8' mapping to
        {'distortion': float, 'rate': float}.
    """
    N, K = weight.shape
    G = K // group_size
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"

    # Reshape to [N, G, group_size]
    w_grouped = weight.reshape(N, G, group_size)

    rd_points = []
    for g in range(G):
        w_g = w_grouped[:, g, :]  # [N, group_size]

        # INT4: per-(N,group) scale, range [-8, 7]
        absmax_4 = w_g.abs().amax(dim=1, keepdim=True)  # [N, 1]
        scale_4 = absmax_4 / 7.0
        scale_4[scale_4 == 0] = 1.0
        q4 = (w_g / scale_4).round().clamp(-8, 7)
        deq4 = q4 * scale_4
        mse_4 = ((w_g - deq4) ** 2).sum().item()  # total MSE across all N

        # INT8: per-(N,group) scale, range [-128, 127]
        absmax_8 = w_g.abs().amax(dim=1, keepdim=True)  # [N, 1]
        scale_8 = absmax_8 / 127.0
        scale_8[scale_8 == 0] = 1.0
        q8 = (w_g / scale_8).round().clamp(-128, 127)
        deq8 = q8 * scale_8
        mse_8 = ((w_g - deq8) ** 2).sum().item()

        # Rate: bits per element (including scale overhead amortized)
        # INT4: 4 bits/elem + 32 bits scale / 128 elems = 4.25 bpw
        # INT8: 8 bits/elem + 32 bits scale / 128 elems = 8.25 bpw
        rd_points.append({
            'INT4': {'distortion': mse_4, 'rate': 4.25},
            'INT8': {'distortion': mse_8, 'rate': 8.25},
        })

    return rd_points


def allocate_kgroups(
    weight: torch.Tensor,          # [N, K] float, AWQ-scaled
    budget_avg_bits: float = 5.3,
    group_size: int = 128,
) -> KSplitAllocation:
    """R-D allocate K-groups to INT4 or INT8 under a bit budget.

    Args:
        weight: AWQ-scaled weight tensor [N, K].
        budget_avg_bits: Target average bits per weight element.
        group_size: Group size along K (must be 128 for Marlin).

    Returns:
        KSplitAllocation with per-group format assignments and K permutation.
    """
    N, K = weight.shape
    G = K // group_size

    with torch.inference_mode():
        rd_points = _compute_kgroup_rd(weight, group_size)

    # Total elements per group = N * group_size
    elems_per_group = N * group_size
    total_elems = N * K
    budget_total_bits = budget_avg_bits * total_elems

    def _pick(lam: float) -> list[str]:
        fmts = []
        for g in range(G):
            cost_4 = rd_points[g]['INT4']['distortion'] + lam * rd_points[g]['INT4']['rate'] * elems_per_group
            cost_8 = rd_points[g]['INT8']['distortion'] + lam * rd_points[g]['INT8']['rate'] * elems_per_group
            fmts.append('INT4' if cost_4 <= cost_8 else 'INT8')
        return fmts

    def _total_bits(fmts: list[str]) -> float:
        return sum(rd_points[g][f]['rate'] * elems_per_group for g, f in enumerate(fmts))

    # Bisection on lambda
    min_fmts = _pick(1e18)
    max_fmts = _pick(0.0)
    min_bits = _total_bits(min_fmts)
    max_bits = _total_bits(max_fmts)

    if budget_total_bits <= min_bits:
        chosen = min_fmts
        lam_star = 1e18
    elif budget_total_bits >= max_bits:
        chosen = max_fmts
        lam_star = 0.0
    else:
        lam_lo, lam_hi = 0.0, 1.0
        for _ in range(64):
            if _total_bits(_pick(lam_hi)) <= budget_total_bits:
                break
            lam_hi *= 2.0
        for _ in range(64):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _total_bits(_pick(lam_mid)) > budget_total_bits:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam_star = (lam_lo + lam_hi) / 2.0
        chosen = _pick(lam_star)

    # Build K permutation: INT4 groups first, INT8 groups last
    int4_groups = [g for g in range(G) if chosen[g] == 'INT4']
    int8_groups = [g for g in range(G) if chosen[g] == 'INT8']

    # Build element-level permutation
    perm_list = []
    for g in int4_groups:
        perm_list.extend(range(g * group_size, (g + 1) * group_size))
    for g in int8_groups:
        perm_list.extend(range(g * group_size, (g + 1) * group_size))

    k_perm = torch.tensor(perm_list, dtype=torch.long)
    k_inv_perm = torch.empty_like(k_perm)
    k_inv_perm[k_perm] = torch.arange(K, dtype=torch.long)

    # Stats
    actual_bits = _total_bits(chosen)
    avg_bits = actual_bits / total_elems
    total_dist = sum(rd_points[g][chosen[g]]['distortion'] for g in range(G))

    return KSplitAllocation(
        group_formats=chosen,
        n_int4_groups=len(int4_groups),
        n_int8_groups=len(int8_groups),
        k_perm=k_perm,
        k_inv_perm=k_inv_perm,
        avg_bits=avg_bits,
        total_distortion=total_dist,
        lambda_star=lam_star,
    )


# ---------------------------------------------------------------------------
# KSplit quantized linear layer (fake-quant)
# ---------------------------------------------------------------------------

class KSplitLinear(nn.Module):
    """Fake-quant linear layer with K-dimension mixed INT4/INT8 precision.

    Stores dequantized weights for fake-quant forward pass.
    """

    def __init__(
        self,
        weight: torch.Tensor,          # [N, K] original float weight
        allocation: KSplitAllocation,
        group_size: int = 128,
        awq_scales: Optional[torch.Tensor] = None,  # [K]
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        N, K = weight.shape
        self.N = N
        self.K = K
        self.group_size = group_size
        self.n_int4_groups = allocation.n_int4_groups
        self.n_int8_groups = allocation.n_int8_groups
        self.K_int4 = allocation.n_int4_groups * group_size
        self.K_int8 = allocation.n_int8_groups * group_size

        # Apply AWQ scaling
        w = weight.float()
        if awq_scales is not None:
            w = w * awq_scales.float().unsqueeze(0)  # W' = W * diag(α)

        # Reorder K
        w_reordered = w[:, allocation.k_perm]  # [N, K] with INT4 groups first

        # Quantize INT4 K-groups
        w_int4_part = w_reordered[:, :self.K_int4]  # [N, K_int4]
        w_int4_grouped = w_int4_part.reshape(N, -1, group_size)
        absmax_4 = w_int4_grouped.abs().amax(dim=2, keepdim=True)
        scale_4 = absmax_4 / 7.0
        scale_4[scale_4 == 0] = 1.0
        q4 = (w_int4_grouped / scale_4).round().clamp(-8, 7)
        deq_int4 = (q4 * scale_4).reshape(N, self.K_int4)

        # Quantize INT8 K-groups (per-group scale, group_size=128)
        w_int8_part = w_reordered[:, self.K_int4:]  # [N, K_int8]
        if self.K_int8 > 0:
            w_int8_grouped = w_int8_part.reshape(N, -1, group_size)
            absmax_8 = w_int8_grouped.abs().amax(dim=2, keepdim=True)
            scale_8 = absmax_8 / 127.0
            scale_8[scale_8 == 0] = 1.0
            q8 = (w_int8_grouped / scale_8).round().clamp(-128, 127)
            deq_int8 = (q8 * scale_8).reshape(N, self.K_int8)
        else:
            deq_int8 = torch.zeros(N, 0, dtype=w.dtype)
            q8 = torch.zeros(N, 0, dtype=torch.int8)
            scale_8 = torch.zeros(N, 0, 1, dtype=torch.float32)

        # Reconstruct full dequantized weight in reordered K order
        w_deq_reordered = torch.cat([deq_int4, deq_int8], dim=1)  # [N, K]

        # Undo AWQ scaling: W_deq_orig = W_deq_awq / α
        if awq_scales is not None:
            awq_reordered = awq_scales.float()[allocation.k_perm]
            w_deq_reordered = w_deq_reordered / awq_reordered.unsqueeze(0)

        # Undo K reorder to get weight in original K order
        w_deq = w_deq_reordered[:, allocation.k_inv_perm]

        self.register_buffer('w_deq', w_deq.half())
        self.register_buffer('k_perm', allocation.k_perm)
        self.register_buffer('k_inv_perm', allocation.k_inv_perm)

        # Store raw quantized data for Marlin conversion
        self.register_buffer('q4_data', q4.to(torch.int8).reshape(N, self.K_int4) if self.K_int4 > 0
                             else torch.zeros(N, 0, dtype=torch.int8))
        self.register_buffer('s4_data', scale_4.squeeze(-1) if self.K_int4 > 0
                             else torch.zeros(N, 0, dtype=torch.float32))
        self.register_buffer('q8_data', q8.to(torch.int8).reshape(N, self.K_int8) if self.K_int8 > 0
                             else torch.zeros(N, 0, dtype=torch.int8))
        self.register_buffer('s8_data', scale_8.squeeze(-1) if self.K_int8 > 0
                             else torch.zeros(N, 0, dtype=torch.float32))

        if awq_scales is not None:
            self.register_buffer('awq_scales', awq_scales.float())
        else:
            self.awq_scales = None

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quant forward: uses stored dequantized weight."""
        orig_dtype = x.dtype
        y = F.linear(x.to(self.w_deq.dtype), self.w_deq)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(orig_dtype)


# ---------------------------------------------------------------------------
# Model-level K-split quantization
# ---------------------------------------------------------------------------

@dataclass
class KSplitLayerInfo:
    allocation: KSplitAllocation
    avg_bits: float


class QuantizedModelKSplit(nn.Module):
    """Wrapper for model with K-split quantized linear layers."""

    def __init__(self, model: nn.Module, layer_info: dict[str, KSplitLayerInfo],
                 budget_avg_bits: float):
        super().__init__()
        self.model = model
        self.layer_info = layer_info
        self.budget_avg_bits = budget_avg_bits

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_summary(self):
        header = f"{'Layer':<50} {'K':>6} {'G4':>5} {'G8':>5} {'avg_bits':>9}"
        print(header)
        print("-" * len(header))
        for name, info in self.layer_info.items():
            alloc = info.allocation
            K = (alloc.n_int4_groups + alloc.n_int8_groups) * 128
            print(f"{name:<50} {K:>6} {alloc.n_int4_groups:>5} {alloc.n_int8_groups:>5} {alloc.avg_bits:>9.2f}")
        print("-" * len(header))
        total_bits = sum(
            info.allocation.avg_bits * (info.allocation.n_int4_groups + info.allocation.n_int8_groups) * 128
            for info in self.layer_info.values()
        )
        total_elems = sum(
            (info.allocation.n_int4_groups + info.allocation.n_int8_groups) * 128
            for info in self.layer_info.values()
        )
        if total_elems > 0:
            print(f"{'TOTAL':<50} {'':>6} {'':>5} {'':>5} {total_bits/total_elems:>9.2f}")


def _should_ignore(name: str, ignore: list[str]) -> bool:
    if not ignore:
        return False
    for pattern in ignore:
        if fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def quantize_model_ksplit(
    model: nn.Module,
    budget_avg_bits: float = 5.3,
    ignore: Optional[list[str]] = None,
    awq_scales: Optional[dict[str, torch.Tensor]] = None,
    group_size: int = 128,
) -> QuantizedModelKSplit:
    """Quantize model with K-dimension INT4/INT8 mixed precision.

    Args:
        model: HuggingFace model (on CPU).
        budget_avg_bits: Target average bits per weight element.
        ignore: Layer name patterns to skip.
        awq_scales: Per-layer AWQ scales from compute_awq_scales().
        group_size: Group size along K (128).

    Returns:
        QuantizedModelKSplit wrapping modified model.
    """
    if ignore is None:
        ignore = []

    layer_info: dict[str, KSplitLayerInfo] = {}

    # Collect all linear layers
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    to_quantize = [(n, m) for n, m in linears if not _should_ignore(n, ignore)]

    # Global allocation: collect R-D tables for all layers, then solve jointly
    all_rd: dict[str, list[dict]] = {}
    all_N: dict[str, int] = {}

    with torch.inference_mode():
        for name, layer in to_quantize:
            w = layer.weight.data.float()
            N, K = w.shape
            if K % group_size != 0:
                continue  # skip layers with K not divisible by group_size

            # Apply AWQ scaling
            if awq_scales is not None and name in awq_scales:
                alpha = awq_scales[name].cpu().float()
                w_scaled = w * alpha.unsqueeze(0)
            else:
                w_scaled = w

            rd = _compute_kgroup_rd(w_scaled, group_size)
            all_rd[name] = rd
            all_N[name] = N

    # Global Lagrangian: find single λ* across all layers
    total_elems = sum(
        all_N[name] * len(all_rd[name]) * group_size
        for name in all_rd
    )
    budget_total_bits = budget_avg_bits * total_elems

    def _global_pick(lam: float) -> dict[str, list[str]]:
        result = {}
        for name, rd in all_rd.items():
            N = all_N[name]
            epg = N * group_size
            fmts = []
            for g, pts in enumerate(rd):
                c4 = pts['INT4']['distortion'] + lam * pts['INT4']['rate'] * epg
                c8 = pts['INT8']['distortion'] + lam * pts['INT8']['rate'] * epg
                fmts.append('INT4' if c4 <= c8 else 'INT8')
            result[name] = fmts
        return result

    def _global_bits(fmts_dict: dict[str, list[str]]) -> float:
        total = 0.0
        for name, fmts in fmts_dict.items():
            N = all_N[name]
            epg = N * group_size
            for g, f in enumerate(fmts):
                total += all_rd[name][g][f]['rate'] * epg
        return total

    min_fmts = _global_pick(1e18)
    max_fmts = _global_pick(0.0)
    min_bits = _global_bits(min_fmts)
    max_bits = _global_bits(max_fmts)

    if budget_total_bits <= min_bits:
        lam_star = 1e18
    elif budget_total_bits >= max_bits:
        lam_star = 0.0
    else:
        lam_lo, lam_hi = 0.0, 1.0
        for _ in range(64):
            if _global_bits(_global_pick(lam_hi)) <= budget_total_bits:
                break
            lam_hi *= 2.0
        for _ in range(64):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _global_bits(_global_pick(lam_mid)) > budget_total_bits:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam_star = (lam_lo + lam_hi) / 2.0

    chosen_all = _global_pick(lam_star)

    # Apply allocations and replace layers
    with torch.no_grad():
        for name, layer in to_quantize:
            if name not in all_rd:
                continue

            w = layer.weight.data.float()
            N, K = w.shape
            G = K // group_size
            chosen = chosen_all[name]

            # Build allocation result
            int4_groups = [g for g in range(G) if chosen[g] == 'INT4']
            int8_groups = [g for g in range(G) if chosen[g] == 'INT8']

            perm_list = []
            for g in int4_groups:
                perm_list.extend(range(g * group_size, (g + 1) * group_size))
            for g in int8_groups:
                perm_list.extend(range(g * group_size, (g + 1) * group_size))

            k_perm = torch.tensor(perm_list, dtype=torch.long)
            k_inv_perm = torch.empty_like(k_perm)
            k_inv_perm[k_perm] = torch.arange(K, dtype=torch.long)

            total_d = sum(all_rd[name][g][chosen[g]]['distortion'] for g in range(G))
            epg = N * group_size
            actual_bits = sum(all_rd[name][g][chosen[g]]['rate'] * epg for g in range(G))

            alloc = KSplitAllocation(
                group_formats=chosen,
                n_int4_groups=len(int4_groups),
                n_int8_groups=len(int8_groups),
                k_perm=k_perm,
                k_inv_perm=k_inv_perm,
                avg_bits=actual_bits / (N * K),
                total_distortion=total_d,
                lambda_star=lam_star,
            )

            awq = awq_scales.get(name).cpu() if (awq_scales and name in awq_scales) else None
            bias = layer.bias.data if layer.bias is not None else None

            qlayer = KSplitLinear(
                weight=w,
                allocation=alloc,
                group_size=group_size,
                awq_scales=awq,
                bias=bias,
            )

            # Replace layer in model
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], qlayer)

            layer_info[name] = KSplitLayerInfo(
                allocation=alloc,
                avg_bits=alloc.avg_bits,
            )

    return QuantizedModelKSplit(model, layer_info, budget_avg_bits)
