"""
2D joint N+K mixed-precision quantization.

Combines N-split (per output channel) and K-split (per K-group) allocation
for maximum R-D flexibility. Each (output_channel, K_group) cell gets the
precision dictated by both its N-assignment and K-assignment:

    effective_bits(n, g) = max(a_n, b_g)

Where a_n is the N-channel precision and b_g is the K-group precision.
Both are solved via alternating Lagrangian optimization.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JointAllocation:
    """Result of 2D N+K allocation for one layer."""
    n_formats: list[str]        # [N] format per output channel
    k_formats: list[str]        # [G] format per K-group
    n_int4_channels: int
    n_int8_channels: int
    n_int4_kgroups: int
    n_int8_kgroups: int
    n_perm: torch.Tensor        # [N] output channel reorder
    n_inv_perm: torch.Tensor
    k_perm: torch.Tensor        # [K] K reorder
    k_inv_perm: torch.Tensor
    avg_bits: float


class JointLinear(nn.Module):
    """Fake-quant linear with 2D N+K mixed precision."""

    def __init__(self, weight: torch.Tensor, alloc: JointAllocation,
                 group_size: int = 128, awq_scales=None, bias=None):
        super().__init__()
        N, K = weight.shape
        G = K // group_size
        self.N, self.K, self.group_size = N, K, group_size

        w = weight.float()
        if awq_scales is not None:
            w = w * awq_scales.float().unsqueeze(0)

        # Quantize each (n, g) cell at its effective precision
        w_deq = torch.zeros_like(w)
        w_grouped = w.reshape(N, G, group_size)

        for g in range(G):
            for n in range(N):
                cell_bits = max(
                    4 if alloc.n_formats[n] == 'INT4' else 8,
                    4 if alloc.k_formats[g] == 'INT4' else 8,
                )
                w_ng = w_grouped[n, g, :]  # [group_size]
                absmax = w_ng.abs().max()
                if cell_bits == 4:
                    s = absmax / 7.0 if absmax > 0 else 1.0
                    q = (w_ng / s).round().clamp(-8, 7)
                else:
                    s = absmax / 127.0 if absmax > 0 else 1.0
                    q = (w_ng / s).round().clamp(-128, 127)
                w_deq[n, g*group_size:(g+1)*group_size] = q * s

        # Undo AWQ
        if awq_scales is not None:
            w_deq = w_deq / awq_scales.float().unsqueeze(0)

        self.register_buffer('w_deq', w_deq.half())
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_dtype = x.dtype
        y = F.linear(x.to(self.w_deq.dtype), self.w_deq)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(orig_dtype)


def _compute_cell_mse(weight: torch.Tensor, group_size: int = 128):
    """Compute MSE for each (n, g) cell under INT4 and INT8.

    Returns:
        mse4: [N, G] float tensor
        mse8: [N, G] float tensor
    """
    N, K = weight.shape
    G = K // group_size
    w = weight.reshape(N, G, group_size)

    # Vectorized INT4
    absmax4 = w.abs().amax(dim=2, keepdim=True)
    s4 = absmax4 / 7.0
    s4[s4 == 0] = 1.0
    q4 = (w / s4).round().clamp(-8, 7)
    mse4 = ((w - q4 * s4) ** 2).mean(dim=2)  # [N, G]

    # Vectorized INT8
    absmax8 = w.abs().amax(dim=2, keepdim=True)
    s8 = absmax8 / 127.0
    s8[s8 == 0] = 1.0
    q8 = (w / s8).round().clamp(-128, 127)
    mse8 = ((w - q8 * s8) ** 2).mean(dim=2)  # [N, G]

    return mse4, mse8


def quantize_model_joint(
    model: nn.Module,
    budget_avg_bits: float = 5.3,
    ignore: Optional[list[str]] = None,
    awq_scales: Optional[dict[str, torch.Tensor]] = None,
    group_size: int = 128,
    n_alternations: int = 3,
) -> nn.Module:
    """2D joint N+K mixed-precision quantization.

    Uses alternating optimization:
    1. Fix K-formats, optimize N-formats
    2. Fix N-formats, optimize K-formats
    Repeat for n_alternations rounds.
    """
    if ignore is None:
        ignore = []

    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and not _should_ignore(n, ignore)]

    # Precompute per-cell MSE for all layers
    all_mse4: dict[str, torch.Tensor] = {}
    all_mse8: dict[str, torch.Tensor] = {}
    layer_shapes: dict[str, tuple] = {}

    with torch.inference_mode():
        for name, layer in linears:
            w = layer.weight.data.float()
            N, K = w.shape
            if K % group_size != 0:
                continue

            if awq_scales is not None and name in awq_scales:
                alpha = awq_scales[name].cpu().float()
                w = w * alpha.unsqueeze(0)

            mse4, mse8 = _compute_cell_mse(w, group_size)
            all_mse4[name] = mse4
            all_mse8[name] = mse8
            layer_shapes[name] = (N, K)

    # Initialize: all INT4 (then allocator will upgrade some to INT8)
    all_n_fmt: dict[str, list[str]] = {}
    all_k_fmt: dict[str, list[str]] = {}
    for name in all_mse4:
        N, K = layer_shapes[name]
        G = K // group_size
        all_n_fmt[name] = ['INT4'] * N
        all_k_fmt[name] = ['INT4'] * G

    total_elems = sum(N * K for N, K in layer_shapes.values())
    budget_total = budget_avg_bits * total_elems

    for iteration in range(n_alternations):
        # --- Step A: Fix K-formats, optimize N-formats ---
        # For each N-channel, its effective precision depends on K-format:
        # If K-group g is INT8, then (n,g) is already INT8 regardless of n's format.
        # So upgrading n to INT8 only helps for K-groups that are INT4.
        # Distortion for channel n at INT4: sum_g d(n,g, eff_bits(4, k_fmt[g]))
        # Distortion for channel n at INT8: sum_g d(n,g, eff_bits(8, k_fmt[g]))

        def _n_channel_rd(name):
            """Returns per-channel (rate_diff, dist_diff) for upgrading from INT4 to INT8."""
            N, K = layer_shapes[name]
            G = K // group_size
            mse4 = all_mse4[name]  # [N, G]
            mse8 = all_mse8[name]
            k_fmt = all_k_fmt[name]

            # Distortion at n=INT4: for each g, eff = max(4, k_bits[g])
            # Distortion at n=INT8: for each g, eff = max(8, k_bits[g]) = 8
            d_n4 = torch.zeros(N)
            d_n8 = torch.zeros(N)
            r_n4 = torch.zeros(N)
            r_n8 = torch.zeros(N)

            for g in range(G):
                k_is_8 = (k_fmt[g] == 'INT8')
                if k_is_8:
                    # Both n=INT4 and n=INT8 give eff=INT8 for this group
                    d_n4 += mse8[:, g] * group_size
                    d_n8 += mse8[:, g] * group_size
                    r_n4 += 8.25 * group_size
                    r_n8 += 8.25 * group_size
                else:
                    # n=INT4: eff=INT4; n=INT8: eff=INT8
                    d_n4 += mse4[:, g] * group_size
                    d_n8 += mse8[:, g] * group_size
                    r_n4 += 4.25 * group_size
                    r_n8 += 8.25 * group_size

            return d_n4, r_n4, d_n8, r_n8

        # Lagrangian over all N-channels across all layers
        all_n_rd = {}
        for name in all_mse4:
            all_n_rd[name] = _n_channel_rd(name)

        def _n_pick(lam):
            bits = 0.0
            for name, (d4, r4, d8, r8) in all_n_rd.items():
                cost4 = d4 + lam * r4
                cost8 = d8 + lam * r8
                pick8 = cost8 < cost4
                bits += (r8 * pick8.float() + r4 * (~pick8).float()).sum().item()
                all_n_fmt[name] = ['INT8' if pick8[n] else 'INT4' for n in range(len(d4))]
            return bits

        # Bisection for N-formats
        lo, hi = 0.0, 1.0
        for _ in range(64):
            if _n_pick(hi) <= budget_total:
                break
            hi *= 2.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _n_pick(mid) > budget_total:
                lo = mid
            else:
                hi = mid
        _n_pick((lo + hi) / 2.0)

        # --- Step B: Fix N-formats, optimize K-formats ---
        def _k_group_rd(name):
            N, K = layer_shapes[name]
            G = K // group_size
            mse4 = all_mse4[name]
            mse8 = all_mse8[name]
            n_fmt = all_n_fmt[name]

            d_g4 = torch.zeros(G)
            d_g8 = torch.zeros(G)
            r_g4 = torch.zeros(G)
            r_g8 = torch.zeros(G)

            for n in range(N):
                n_is_8 = (n_fmt[n] == 'INT8')
                if n_is_8:
                    # Both g=INT4 and g=INT8 give eff=INT8
                    d_g4 += mse8[n, :] * group_size
                    d_g8 += mse8[n, :] * group_size
                    r_g4 += 8.25 * group_size
                    r_g8 += 8.25 * group_size
                else:
                    d_g4 += mse4[n, :] * group_size
                    d_g8 += mse8[n, :] * group_size
                    r_g4 += 4.25 * group_size
                    r_g8 += 8.25 * group_size

            return d_g4, r_g4, d_g8, r_g8

        all_k_rd = {}
        for name in all_mse4:
            all_k_rd[name] = _k_group_rd(name)

        def _k_pick(lam):
            bits = 0.0
            for name, (d4, r4, d8, r8) in all_k_rd.items():
                cost4 = d4 + lam * r4
                cost8 = d8 + lam * r8
                pick8 = cost8 < cost4
                bits += (r8 * pick8.float() + r4 * (~pick8).float()).sum().item()
                G = len(d4)
                all_k_fmt[name] = ['INT8' if pick8[g] else 'INT4' for g in range(G)]
            return bits

        lo, hi = 0.0, 1.0
        for _ in range(64):
            if _k_pick(hi) <= budget_total:
                break
            hi *= 2.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _k_pick(mid) > budget_total:
                lo = mid
            else:
                hi = mid
        _k_pick((lo + hi) / 2.0)

        # Compute current total bits
        total_bits = 0
        for name in all_mse4:
            N, K = layer_shapes[name]
            G = K // group_size
            for n in range(N):
                for g in range(G):
                    eff = max(4 if all_n_fmt[name][n] == 'INT4' else 8,
                              4 if all_k_fmt[name][g] == 'INT4' else 8)
                    total_bits += (eff + 0.25) * group_size  # +0.25 for scale overhead

        avg = total_bits / total_elems
        n8_n = sum(1 for name in all_n_fmt for f in all_n_fmt[name] if f == 'INT8')
        n8_k = sum(1 for name in all_k_fmt for f in all_k_fmt[name] if f == 'INT8')
        print(f'  Iter {iteration+1}: avg_bits={avg:.3f}, N-INT8={n8_n}, K-INT8={n8_k}')

    # Build quantized layers
    layer_info = {}
    with torch.no_grad():
        for name, layer in linears:
            if name not in all_mse4:
                continue

            w = layer.weight.data.float()
            N, K = w.shape
            G = K // group_size

            n_fmt = all_n_fmt[name]
            k_fmt = all_k_fmt[name]

            # Build permutations
            n_int4_ch = [n for n in range(N) if n_fmt[n] == 'INT4']
            n_int8_ch = [n for n in range(N) if n_fmt[n] == 'INT8']
            n_perm = torch.tensor(n_int4_ch + n_int8_ch, dtype=torch.long)
            n_inv_perm = torch.empty_like(n_perm)
            n_inv_perm[n_perm] = torch.arange(N, dtype=torch.long)

            k_int4_g = [g for g in range(G) if k_fmt[g] == 'INT4']
            k_int8_g = [g for g in range(G) if k_fmt[g] == 'INT8']
            k_perm_list = []
            for g in k_int4_g:
                k_perm_list.extend(range(g*group_size, (g+1)*group_size))
            for g in k_int8_g:
                k_perm_list.extend(range(g*group_size, (g+1)*group_size))
            k_perm = torch.tensor(k_perm_list, dtype=torch.long)
            k_inv_perm = torch.empty_like(k_perm)
            k_inv_perm[k_perm] = torch.arange(K, dtype=torch.long)

            # Compute effective bits per cell
            total_b = 0
            for n in range(N):
                for g in range(G):
                    eff = max(4 if n_fmt[n] == 'INT4' else 8,
                              4 if k_fmt[g] == 'INT4' else 8)
                    total_b += (eff + 0.25) * group_size
            avg_b = total_b / (N * K)

            alloc = JointAllocation(
                n_formats=n_fmt, k_formats=k_fmt,
                n_int4_channels=len(n_int4_ch), n_int8_channels=len(n_int8_ch),
                n_int4_kgroups=len(k_int4_g), n_int8_kgroups=len(k_int8_g),
                n_perm=n_perm, n_inv_perm=n_inv_perm,
                k_perm=k_perm, k_inv_perm=k_inv_perm,
                avg_bits=avg_b,
            )

            awq = awq_scales.get(name).cpu() if (awq_scales and name in awq_scales) else None
            bias = layer.bias.data if layer.bias is not None else None

            qlayer = JointLinear(w, alloc, group_size, awq, bias)

            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], qlayer)
            layer_info[name] = alloc

    # Wrap
    class QuantizedModelJoint(nn.Module):
        def __init__(self, model, layer_info, budget):
            super().__init__()
            self.model = model
            self.layer_info = layer_info
            self.budget_avg_bits = budget
        def forward(self, *a, **kw):
            return self.model(*a, **kw)

    return QuantizedModelJoint(model, layer_info, budget_avg_bits)


def _should_ignore(name, ignore):
    for p in ignore:
        if fnmatch.fnmatch(name, p) or p in name:
            return True
    return False
