"""
INT8 → 2× INT4 decomposition for single-kernel mixed-precision inference.

Math:
  w_int8 = 16 * h + l - 128    (h, l are UINT4 in [0,15])
  y = Σ w_int8 * scale * x
    = scale * (16 * Σ h*x + Σ l*x - 128 * Σ x)
    = scale * (16 * y_h + y_l - 128 * sum_x)

By concatenating INT4 weights + INT8 high/low nibbles along N,
we run a SINGLE Marlin INT4 kernel and post-process to reconstruct.

Layout (single Marlin call):
  W_combined = [W_int4_orig | W_int8_high | W_int8_low]
  N_combined = N_int4 + 2 * N_int8

  y_combined = Marlin_INT4(x, W_combined)  # 1 launch

Post-process:
  y_int4 = y_combined[:N_int4]                         # direct
  y_int8 = 16 * y_combined[N_int4:N_int4+N8] + y_combined[N_int4+N8:]
  y_int8 = y_int8 * int8_scale - 128 * int8_scale * sum_x
  y = cat(y_int4, y_int8)[inv_perm]

This module is additive — existing NVFP4/FP8/FP16 code is untouched.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


def decompose_int8_to_uint4_pair(
    w_int8: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split INT8 weights into high and low UINT4 nibbles.

    Args:
        w_int8: [N, K] int8 tensor

    Returns:
        w_high: [N, K] uint8 (values 0-15, high nibble)
        w_low:  [N, K] uint8 (values 0-15, low nibble)
    """
    w_uint8 = (w_int8.to(torch.int16) + 128).to(torch.uint8)
    w_high = (w_uint8 >> 4) & 0x0F
    w_low = w_uint8 & 0x0F
    return w_high, w_low


def reconstruct_int8_output(
    y_high: torch.Tensor,
    y_low: torch.Tensor,
    int8_scales: torch.Tensor,
    sum_x: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct INT8 GEMV output from UINT4 partial results.

    Args:
        y_high: [M, N_int8] — Marlin output for high nibbles
        y_low:  [M, N_int8] — Marlin output for low nibbles
        int8_scales: [N_int8] — per-channel INT8 scales
        sum_x: [M, 1] — sum of activation elements per token

    Returns:
        y_int8: [M, N_int8] — reconstructed INT8 GEMV output
    """
    # y = scale * (16 * y_high + y_low - 128 * sum_x)
    y_raw = 16.0 * y_high + y_low - 128.0 * sum_x
    return y_raw * int8_scales.unsqueeze(0)


def quantize_to_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 weight to INT8 with per-channel scale.

    Args:
        w: [N, K] float weight

    Returns:
        w_int8: [N, K] int8
        scales: [N] float32 per-channel scale
    """
    w_f = w.float()
    absmax = w_f.abs().amax(dim=1)  # [N]
    scales = absmax / 127.0
    scales[scales == 0] = 1.0
    w_int8 = (w_f / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scales


def quantize_to_int4_groupwise(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 weight to INT4 with per-group scale.

    Args:
        w: [N, K] float weight
        group_size: elements per scale group

    Returns:
        w_int4: [N, K] int8 (values in [-8, 7])
        scales: [N, K//group_size] float32
    """
    w_f = w.float()
    N, K = w_f.shape
    n_groups = K // group_size
    w_grouped = w_f.reshape(N, n_groups, group_size)
    absmax = w_grouped.abs().amax(dim=2)  # [N, n_groups]
    scales = absmax / 7.0
    scales[scales == 0] = 1.0
    w_norm = w_grouped / scales.unsqueeze(2)
    w_int4 = w_norm.round().clamp(-8, 7).to(torch.int8).reshape(N, K)
    return w_int4, scales


class Int4FusedLinear(torch.nn.Module):
    """Mixed INT4/INT8 linear layer using single-kernel INT4 decomposition.

    Stores:
        - INT4 channels: original INT4 weights (low-precision group)
        - INT8 channels: decomposed into high/low UINT4 nibbles

    Forward: 1× Marlin INT4 GEMV on concatenated weights + post-process
    """

    def __init__(
        self,
        w_int4: torch.Tensor,       # [N_int4, K] int8 (values -8..7)
        s_int4: torch.Tensor,        # [N_int4, K//group_size] float32
        w_int8: torch.Tensor,        # [N_int8, K] int8 (values -128..127)
        s_int8: torch.Tensor,        # [N_int8] float32 per-channel
        inv_perm: torch.Tensor,      # [N_total] int64
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ):
        super().__init__()

        N_int4, K = w_int4.shape
        N_int8 = w_int8.shape[0]
        N_total = N_int4 + N_int8

        self.N_int4 = N_int4
        self.N_int8 = N_int8
        self.K = K
        self.group_size = group_size

        # Decompose INT8 → high/low UINT4
        w_high, w_low = decompose_int8_to_uint4_pair(w_int8)

        # Convert INT4 weights to UINT4 (add 8 to shift from [-8,7] to [0,15])
        w_int4_uint = (w_int4.to(torch.int16) + 8).to(torch.uint8)

        # Concatenate: [W_int4_uint | W_high | W_low] along N
        # Shape: [N_int4 + 2*N_int8, K] uint8 (all values 0-15)
        w_combined = torch.cat([w_int4_uint, w_high, w_low], dim=0)

        # For fake-quant: dequantize combined weight to FP16 for F.linear
        # INT4 part: (uint4 - 8) * scale_per_group
        n_groups = K // group_size
        int4_deq = (w_int4_uint.float() - 8) * s_int4.repeat_interleave(group_size, dim=1)

        # INT8 high/low: stored as UINT4, will be combined in post-process
        # For fake-quant, just dequant the original INT8 directly
        int8_deq = w_int8.float() * s_int8.unsqueeze(1)

        # Store dequantized weights for fake-quant forward
        self.register_buffer('w_int4_deq', int4_deq.half())
        self.register_buffer('w_int8_deq', int8_deq.half())
        self.register_buffer('inv_perm', inv_perm)
        self.register_buffer('int8_scales', s_int8)

        # Store packed combined weight for future Marlin integration
        self.register_buffer('w_combined_uint4', w_combined)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quant forward (dequant + F.linear per group, then combine).

        TODO: Replace with single Marlin INT4 kernel + post-process.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K)

        # INT4 group
        y_int4 = F.linear(x_2d, self.w_int4_deq)  # [M, N_int4]

        # INT8 group
        y_int8 = F.linear(x_2d, self.w_int8_deq)  # [M, N_int8]

        # Concat + inv_perm
        y = torch.cat([y_int4, y_int8], dim=-1)
        y = y.index_select(-1, self.inv_perm)

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*orig_shape[:-1], y.shape[-1])

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward using INT8→2×INT4 decomposition.

        Single GEMV on w_combined, then post-process to reconstruct INT8 output.
        Currently uses fake F.linear; will be replaced by Marlin kernel.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K).float()

        # Pre-compute sum_x for INT8 reconstruction
        sum_x = x_2d.sum(dim=1, keepdim=True)  # [M, 1]

        # Single GEMV on combined weight
        w_comb_float = self.w_combined_uint4.float()  # [N_int4 + 2*N_int8, K]
        y_combined = F.linear(x_2d, w_comb_float)  # [M, N_int4 + 2*N_int8]

        N4 = self.N_int4
        N8 = self.N_int8

        # Split output
        y_int4_raw = y_combined[:, :N4]
        y_int8_high = y_combined[:, N4:N4+N8]
        y_int8_low = y_combined[:, N4+N8:]

        # INT4 reconstruction: (raw_uint4_dot - 8*sum_x) * scale_per_group
        # Since we stored UINT4 = INT4 + 8, the dot product is offset:
        # Σ (uint4_k * x_k) = Σ ((int4_k + 8) * x_k) = Σ int4_k*x_k + 8*sum_x
        # So: y_int4 = (y_raw - 8 * sum_x) * scale
        # But scale is per-group, not per-channel... need group-wise correction
        # For simplicity, just use the direct dequant path for INT4
        y_int4 = F.linear(x_2d, self.w_int4_deq.float())

        # INT8 reconstruction
        y_int8 = reconstruct_int8_output(y_int8_high, y_int8_low,
                                          self.int8_scales, sum_x)

        y = torch.cat([y_int4.half(), y_int8.half()], dim=-1)
        y = y.index_select(-1, self.inv_perm)

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*orig_shape[:-1], y.shape[-1])


def verify_int8_decomposition():
    """Verify INT8 → 2×INT4 decomposition is exact."""
    torch.manual_seed(42)

    N, K, M = 256, 512, 4

    # Random INT8 weight
    w_int8 = torch.randint(-128, 128, (N, K), dtype=torch.int8)
    scale = torch.randn(N).abs() * 0.01

    # Reference: direct INT8 GEMV
    x = torch.randn(M, K)
    y_ref = (x @ (w_int8.float() * scale.unsqueeze(1)).T)

    # Decomposed
    w_high, w_low = decompose_int8_to_uint4_pair(w_int8)
    sum_x = x.sum(dim=1, keepdim=True)

    y_high = x @ w_high.float().T
    y_low = x @ w_low.float().T
    y_recon = reconstruct_int8_output(y_high, y_low, scale, sum_x)

    diff = (y_ref - y_recon).abs().max().item()
    print(f"INT8 → 2×INT4 decomposition verification:")
    print(f"  Max diff: {diff:.8f}")
    print(f"  Status: {'PASS' if diff < 1e-4 else 'FAIL'}")
    return diff < 1e-4


if __name__ == "__main__":
    verify_int8_decomposition()
