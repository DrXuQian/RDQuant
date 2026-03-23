"""
Single-Marlin-launch fused inference for AWQ + INT4/INT8 RDQuant.

All INT4 and INT8 weights are packed as UINT4 and run through a SINGLE
marlin_gemm call.  INT8 channels are decomposed into high/low nibbles,
with scales pre-multiplied by 16 for high nibbles.

Key insight about uint4b8
-------------------------
vLLM's ``uint4b8`` scalar type means "unsigned 4-bit with implicit
zero-point of 8".  The Marlin dequant for uint4b8 does::

    value = (uint4 - 8) * scale

This is exactly what we want for INT4 channels (stored as
UINT4 = INT4 + 8).  For INT8 high/low nibbles (raw UINT4, NOT shifted
by 8), Marlin also subtracts 8.  We compensate analytically:

    y_h_marlin = sum_k (h_k - 8) * (s8 * 16) * x_k
    y_l_marlin = sum_k (l_k - 8) * s8 * x_k
    y_h_marlin + y_l_marlin
      = s8 * sum (16*(h-8) + (l-8)) * x
      = s8 * sum (16h + l - 136) * x
      = s8 * sum w_int8 * x   -  8 * s8 * sum x

So the correction is simply  ``+ 8 * s8 * sum_x``.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# vLLM imports
# ---------------------------------------------------------------------------
sys.path.insert(0, "/root/autodl-tmp/vllm_site")

import vllm._custom_ops as ops  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
    marlin_permute_scales,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_uint4_to_int32(w_uint4: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] uint8 (values 0-15) into [N, K//8] int32 (8 nibbles each).

    The packing order matches what ``gptq_marlin_repack`` expects:
    bits  0-3  = col 0, bits 4-7 = col 1, ... bits 28-31 = col 7.
    """
    N, K = w_uint4.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8"
    w = w_uint4.to(torch.int32)
    packed = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= w[:, i::8] << (4 * i)
    return packed


def pack_for_marlin(
    w_int4_uint4: torch.Tensor,  # [N_int4, K] uint8 0-15
    s_int4: torch.Tensor,         # [N_int4, K//group_size] float32
    w_int8: torch.Tensor,         # [N_int8, K] int8
    s_int8: torch.Tensor,         # [N_int8] float32
    group_size: int = 128,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Offline packing for the single-Marlin mixed-precision kernel.

    Returns dict with keys:
      - marlin_qweight / marlin_scales for the persistent Marlin fallback
      - uint4_packed_rowwise / uint4_scales_rowwise for the non-persistent
        decode GEMV path
      - int8_correction, N_int4, N_int8, K, N_combined
    """
    N_int4, K = w_int4_uint4.shape
    N_int8 = w_int8.shape[0]
    N_combined = N_int4 + 2 * N_int8
    n_groups = K // group_size

    # --- 1. Decompose INT8 into high/low nibbles ---
    w_uint8 = (w_int8.to(torch.int16) + 128).to(torch.uint8)
    w_high = (w_uint8 >> 4) & 0x0F  # [N_int8, K]
    w_low = w_uint8 & 0x0F           # [N_int8, K]

    # --- 2. Concatenate all UINT4 weights ---
    w_combined = torch.cat([w_int4_uint4, w_high, w_low], dim=0)  # [N_combined, K]

    # --- 3. Build unified scale tensor [K//group_size, N_combined] ---
    s_int4_t = s_int4.float().T.contiguous()  # [n_groups, N_int4]

    s_high = (s_int8.float() * 16.0).unsqueeze(0).expand(n_groups, -1)  # [n_groups, N_int8]
    s_low = s_int8.float().unsqueeze(0).expand(n_groups, -1).contiguous()  # [n_groups, N_int8]

    scales_combined = torch.cat([s_int4_t, s_high, s_low], dim=1)  # [n_groups, N_combined]

    # --- 4. Pack UINT4 weights for Marlin ---
    # Pack [N_combined, K] uint4 -> [N_combined, K//8] int32
    w_int32 = _pack_uint4_to_int32(w_combined.to(device))  # [N_combined, K//8]
    # Transpose to [K//8, N_combined] for gptq_marlin_repack
    w_int32_t = w_int32.T.contiguous()

    perm = torch.empty(0, dtype=torch.int, device=device)
    marlin_qweight = ops.gptq_marlin_repack(
        w_int32_t, perm, K, N_combined, 4, False
    )

    # --- 5. Permute scales for Marlin ---
    marlin_scales = marlin_permute_scales(
        scales_combined.to(torch.half).to(device),
        K, N_combined, group_size=group_size, is_a_8bit=False,
    )

    # --- 6. INT8 correction: +8 * s8 ---
    int8_correction = (8.0 * s_int8.float()).to(torch.half).to(device)  # [N_int8]

    return dict(
        marlin_qweight=marlin_qweight,
        marlin_scales=marlin_scales,
        uint4_packed_rowwise=w_int32.contiguous(),
        uint4_scales_rowwise=scales_combined.T.to(torch.half).to(device).contiguous(),
        int8_correction=int8_correction,
        N_int4=N_int4,
        N_int8=N_int8,
        K=K,
        N_combined=N_combined,
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def _load_fused_kernels():
    """Load fused pre/post-processing CUDA kernels (cached after first call)."""
    if not hasattr(_load_fused_kernels, "_module"):
        import os
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "int4_postprocess.cu")
        _load_fused_kernels._module = load(
            name="int4_postprocess",
            sources=[src],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _load_fused_kernels._module


def _choose_uint4_parallel_k(n_combined: int, k: int) -> int:
    """Heuristic for the non-persistent UINT4 decode kernel.

    Larger-K / smaller-N shapes need extra K-parallelism to keep enough CTAs
    in flight; smaller K benefits more from keeping the path single-kernel.
    """
    parallel_k = 1
    if k >= 2048:
        parallel_k = 2
    if k >= 4096:
        parallel_k = 4
    if k >= 8192:
        parallel_k = 8
    if n_combined >= 6144 and k >= 2048:
        parallel_k *= 2
    return min(parallel_k, 16)


class Int4MarlinLinear(nn.Module):
    """Single-Marlin-launch mixed INT4/INT8 linear layer."""

    def __init__(
        self,
        w_int4_uint4: torch.Tensor,   # [N_int4, K] uint8 0-15
        s_int4: torch.Tensor,          # [N_int4, K//group_size] float32
        w_int8: torch.Tensor,          # [N_int8, K] int8
        s_int8: torch.Tensor,          # [N_int8] float32
        inv_perm: torch.Tensor,        # [N_total] int64
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
        awq_scales: Optional[torch.Tensor] = None,  # [K] float32
        use_fused: bool = True,
        use_nonpersistent_gemv: bool = True,
    ):
        super().__init__()

        device = "cuda"
        packed = pack_for_marlin(
            w_int4_uint4, s_int4, w_int8, s_int8,
            group_size=group_size, device=device,
        )

        self.N_int4 = packed["N_int4"]
        self.N_int8 = packed["N_int8"]
        self.N_total = self.N_int4 + self.N_int8
        self.N_combined = packed["N_combined"]
        self.K = packed["K"]
        self.group_size = group_size
        self.use_fused = use_fused
        self.use_nonpersistent_gemv = use_nonpersistent_gemv
        self.uint4_parallel_k = _choose_uint4_parallel_k(self.N_combined, self.K)

        self.register_buffer("marlin_qweight", packed["marlin_qweight"])
        self.register_buffer("marlin_scales", packed["marlin_scales"])
        self.register_buffer("uint4_packed_rowwise", packed["uint4_packed_rowwise"])
        self.register_buffer("uint4_scales_rowwise", packed["uint4_scales_rowwise"])
        self.register_buffer("int8_correction", packed["int8_correction"])
        # int32 perm for v2 fused kernel (halves bandwidth vs int64)
        self.register_buffer("inv_perm", inv_perm.to(torch.int32).to(device))
        self.register_buffer("workspace", marlin_make_workspace_new(torch.device(device)))

        # Empty tensors Marlin needs
        self.register_buffer("g_idx", torch.empty(0, dtype=torch.int, device=device))
        self.register_buffer("sort_indices", torch.empty(0, dtype=torch.int, device=device))
        self.register_buffer("zp", torch.empty(0, dtype=torch.int, device=device))

        if awq_scales is not None:
            self.register_buffer("awq_scales", awq_scales.to(torch.half).to(device))
            # Precompute 1/α for v2 fused kernel (avoid per-element division)
            self.register_buffer("inv_awq_scales",
                                 (1.0 / awq_scales.float()).to(torch.half).to(device))
        else:
            self.awq_scales = None
            self.inv_awq_scales = None

        if bias is not None:
            self.register_buffer("bias", bias.to(torch.half).to(device))
        else:
            self.bias = None

        # Load fused kernels
        if use_fused:
            self._fused = _load_fused_kernels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K).half()
        M = x_2d.shape[0]

        if self.use_fused and self._fused is not None:
            return self._forward_fused(x_2d, M, orig_dtype, orig_shape)
        else:
            return self._forward_unfused(x_2d, M, orig_dtype, orig_shape)

    def _forward_fused(self, x_2d, M, orig_dtype, orig_shape):
        """Forward with fused pre/post-processing CUDA kernels (v2)."""
        if self.use_nonpersistent_gemv and M == 1:
            inv_awq = self.inv_awq_scales if self.inv_awq_scales is not None else x_2d.new_empty(0)
            y_out = self._fused.fused_uint4_decode(
                x_2d, inv_awq,
                self.uint4_packed_rowwise, self.uint4_scales_rowwise,
                self.int8_correction, self.inv_perm,
                self.N_int4, self.N_int8,
                self.group_size, self.uint4_parallel_k,
            )
            if self.bias is not None:
                y_out = y_out + self.bias
            return y_out.to(orig_dtype).reshape(*orig_shape[:-1], self.N_total)

        # Fused pre: in-place AWQ scaling (x *= 1/α) + sum_x — one kernel
        if self.inv_awq_scales is not None:
            x_2d = x_2d.contiguous()   # ensure contiguous for in-place
            x_2d, sum_x = self._fused.fused_awq_sum(x_2d, self.inv_awq_scales)
        else:
            sum_x = self._fused.fused_sum_only(x_2d)

        y = ops.marlin_gemm(
            a=x_2d, c=None,
            b_q_weight=self.marlin_qweight, b_bias=None,
            b_scales=self.marlin_scales,
            a_scales=None, global_scale=None,
            b_zeros=self.zp, g_idx=self.g_idx,
            perm=self.sort_indices, workspace=self.workspace,
            b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=self.N_combined, size_k=self.K,
            is_k_full=True,
        )

        # Fused post: INT8 correction + inv_perm reorder in one kernel
        y_out = self._fused.fused_post(
            y, self.int8_correction, sum_x, self.inv_perm,
            self.N_int4, self.N_int8,
        )

        if self.bias is not None:
            y_out = y_out + self.bias

        return y_out.to(orig_dtype).reshape(*orig_shape[:-1], self.N_total)

    def _forward_unfused(self, x_2d, M, orig_dtype, orig_shape):
        """Forward with standard PyTorch ops (fallback)."""
        if self.awq_scales is not None:
            x_2d = x_2d / self.awq_scales.unsqueeze(0)

        sum_x = x_2d.sum(dim=1, keepdim=True)

        y = ops.marlin_gemm(
            a=x_2d, c=None,
            b_q_weight=self.marlin_qweight, b_bias=None,
            b_scales=self.marlin_scales,
            a_scales=None, global_scale=None,
            b_zeros=self.zp, g_idx=self.g_idx,
            perm=self.sort_indices, workspace=self.workspace,
            b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=self.N_combined, size_k=self.K,
            is_k_full=True,
        )

        y_int4 = y[:, :self.N_int4]
        y_high = y[:, self.N_int4:self.N_int4 + self.N_int8]
        y_low = y[:, self.N_int4 + self.N_int8:]
        y_int8 = y_high + y_low + self.int8_correction.unsqueeze(0) * sum_x

        y_out = torch.cat([y_int4, y_int8], dim=-1)
        y_out = y_out.index_select(-1, self.inv_perm)

        if self.bias is not None:
            y_out = y_out + self.bias

        return y_out.to(orig_dtype).reshape(*orig_shape[:-1], self.N_total)


# ===================================================================
# Tests
# ===================================================================

def _fake_quant_reference(
    x: torch.Tensor,         # [M, K] float
    w_int4: torch.Tensor,    # [N_int4, K] int8 values -8..7
    s_int4: torch.Tensor,    # [N_int4, K//gs] float
    w_int8: torch.Tensor,    # [N_int8, K] int8
    s_int8: torch.Tensor,    # [N_int8] float
    inv_perm: torch.Tensor,  # [N_total]
    group_size: int = 128,
    awq_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for correctness checking."""
    x_f = x.float()
    if awq_scales is not None:
        x_f = x_f / awq_scales.float().unsqueeze(0)

    K = x_f.shape[1]
    n_groups = K // group_size

    # INT4 dequant: per-group
    w4_deq = w_int4.float() * s_int4.float().repeat_interleave(group_size, dim=1)
    y_int4 = x_f @ w4_deq.T  # [M, N_int4]

    # INT8 dequant: per-channel
    w8_deq = w_int8.float() * s_int8.float().unsqueeze(1)
    y_int8 = x_f @ w8_deq.T  # [M, N_int8]

    y = torch.cat([y_int4, y_int8], dim=-1)
    y = y.index_select(-1, inv_perm)
    return y


def test_correctness(M=4, N_int4=128, N_int8=128, K=512, group_size=128, use_awq=False):
    """Pack weights -> marlin_gemm -> post-process -> compare with reference."""
    torch.manual_seed(42)
    device = "cuda"

    # Generate random quantized weights
    w_int4_vals = torch.randint(-8, 8, (N_int4, K), dtype=torch.int8)
    w_int4_uint4 = (w_int4_vals.to(torch.int16) + 8).to(torch.uint8)
    s_int4 = torch.randn(N_int4, K // group_size).abs().float() * 0.01

    w_int8 = torch.randint(-128, 128, (N_int8, K), dtype=torch.int8)
    s_int8 = torch.randn(N_int8).abs().float() * 0.01

    N_total = N_int4 + N_int8
    inv_perm = torch.randperm(N_total, dtype=torch.int64)

    awq_scales = None
    if use_awq:
        awq_scales = torch.randn(K).abs().float() * 0.5 + 0.5

    x = torch.randn(M, K, device=device, dtype=torch.half)

    # Reference
    ref = _fake_quant_reference(
        x.float().cpu(), w_int4_vals, s_int4, w_int8, s_int8,
        inv_perm, group_size, awq_scales,
    ).to(device).half()

    # Marlin
    layer = Int4MarlinLinear(
        w_int4_uint4, s_int4, w_int8, s_int8, inv_perm,
        group_size=group_size, awq_scales=awq_scales,
    )
    out = layer(x)

    diff = (ref.float() - out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = (diff / (ref.float().abs() + 1e-8)).mean().item()

    tag = "AWQ+" if use_awq else ""
    status = "PASS" if rel_err < 0.02 else "FAIL"
    print(f"[{tag}Correctness] M={M}, N4={N_int4}, N8={N_int8}, K={K}, gs={group_size}")
    print(f"  max_abs_diff={max_diff:.6f}  mean_abs_diff={mean_diff:.6f}  "
          f"mean_rel_err={rel_err:.6f}  -> {status}")
    return status == "PASS"


def test_correctness_suite():
    """Run multiple correctness tests."""
    print("=" * 70)
    print("Correctness tests")
    print("=" * 70)
    configs = [
        dict(M=1, N_int4=128, N_int8=128, K=512, group_size=128, use_awq=False),
        dict(M=4, N_int4=256, N_int8=128, K=1024, group_size=128, use_awq=False),
        dict(M=16, N_int4=128, N_int8=256, K=512, group_size=128, use_awq=False),
        dict(M=4, N_int4=128, N_int8=128, K=512, group_size=128, use_awq=True),
        dict(M=8, N_int4=256, N_int8=256, K=1024, group_size=128, use_awq=True),
        # Larger sizes
        dict(M=1, N_int4=2048, N_int8=2048, K=4096, group_size=128, use_awq=False),
        dict(M=32, N_int4=2048, N_int8=2048, K=4096, group_size=128, use_awq=True),
    ]
    all_pass = True
    for cfg in configs:
        ok = test_correctness(**cfg)
        all_pass = all_pass and ok
        print()
    return all_pass


def benchmark(M=1, N_int4=2048, N_int8=2048, K=4096, group_size=128,
              warmup=50, iters=200):
    """Benchmark: non-persistent UINT4 GEMV vs persistent Marlin vs baselines."""
    torch.manual_seed(0)
    device = "cuda"

    # --- Setup weights ---
    w_int4_vals = torch.randint(-8, 8, (N_int4, K), dtype=torch.int8)
    w_int4_uint4 = (w_int4_vals.to(torch.int16) + 8).to(torch.uint8)
    s_int4 = torch.randn(N_int4, K // group_size).abs().float() * 0.01

    w_int8 = torch.randint(-128, 128, (N_int8, K), dtype=torch.int8)
    s_int8 = torch.randn(N_int8).abs().float() * 0.01

    N_total = N_int4 + N_int8
    inv_perm = torch.arange(N_total, dtype=torch.int64)  # identity for benchmark

    x = torch.randn(M, K, device=device, dtype=torch.half)

    layer_new = Int4MarlinLinear(
        w_int4_uint4, s_int4, w_int8, s_int8, inv_perm,
        group_size=group_size, use_nonpersistent_gemv=True,
    )
    layer_old = Int4MarlinLinear(
        w_int4_uint4, s_int4, w_int8, s_int8, inv_perm,
        group_size=group_size, use_nonpersistent_gemv=False,
    )

    def _time_layer(mod):
        for _ in range(warmup):
            mod(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mod(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    new_ms = _time_layer(layer_new)
    old_ms = _time_layer(layer_old)

    # --- 2x Marlin (separate INT4 + INT8-as-INT4 calls) ---
    # Pack INT4 part separately
    packed_int4 = pack_for_marlin(
        w_int4_uint4, s_int4,
        torch.empty(0, K, dtype=torch.int8), torch.empty(0, dtype=torch.float32),
        group_size=group_size, device=device,
    )
    # Pack INT8 part separately (decomposed into h/l)
    # Create fake "int4" from int8 high/low
    w_u8 = (w_int8.to(torch.int16) + 128).to(torch.uint8)
    w_h = (w_u8 >> 4) & 0x0F
    w_l = w_u8 & 0x0F
    w_hl = torch.cat([w_h, w_l], dim=0)
    s_hl_high = (s_int8.float() * 16.0).unsqueeze(0).expand(K // group_size, -1)
    s_hl_low = s_int8.float().unsqueeze(0).expand(K // group_size, -1)
    s_hl = torch.cat([s_hl_high, s_hl_low], dim=1).contiguous()

    N_hl = 2 * N_int8
    w_hl_int32 = _pack_uint4_to_int32(w_hl.cuda())
    perm_empty = torch.empty(0, dtype=torch.int, device=device)
    marlin_qw_int4 = packed_int4["marlin_qweight"]
    marlin_sc_int4 = packed_int4["marlin_scales"]

    marlin_qw_hl = ops.gptq_marlin_repack(
        w_hl_int32.T.contiguous(), perm_empty, K, N_hl, 4, False)
    marlin_sc_hl = marlin_permute_scales(
        s_hl.half().cuda(), K, N_hl, group_size=group_size, is_a_8bit=False)
    ws2 = marlin_make_workspace_new(torch.device(device))
    zp2 = torch.empty(0, dtype=torch.int, device=device)
    g2 = torch.empty(0, dtype=torch.int, device=device)
    corr2 = (8.0 * s_int8.float()).half().cuda()

    def two_marlin_forward(x_in):
        x2 = x_in.reshape(-1, K).half()
        m = x2.shape[0]
        sum_x = x2.sum(dim=1, keepdim=True)

        y4 = ops.marlin_gemm(
            x2, None, marlin_qw_int4, None, marlin_sc_int4,
            None, None, zp2, g2, g2, ws2, scalar_types.uint4b8,
            m, N_int4, K, True)

        y_hl = ops.marlin_gemm(
            x2, None, marlin_qw_hl, None, marlin_sc_hl,
            None, None, zp2, g2, g2, ws2, scalar_types.uint4b8,
            m, N_hl, K, True)
        y_h = y_hl[:, :N_int8]
        y_l = y_hl[:, N_int8:]
        y8 = y_h + y_l + corr2.unsqueeze(0) * sum_x
        return torch.cat([y4, y8], dim=-1)

    for _ in range(warmup):
        two_marlin_forward(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        two_marlin_forward(x)
    torch.cuda.synchronize()
    two_ms = (time.perf_counter() - t0) / iters * 1000

    # --- BF16 cuBLAS baseline ---
    w_bf16 = torch.randn(N_total, K, device=device, dtype=torch.bfloat16)
    x_bf = x.bfloat16()

    for _ in range(warmup):
        F.linear(x_bf, w_bf16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        F.linear(x_bf, w_bf16)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"Benchmark  M={M}  N_total={N_total}  K={K}")
    print(f"  New UINT4 GEMV:      {new_ms:.4f} ms")
    print(f"  1x Marlin (old):     {old_ms:.4f} ms")
    print(f"  2x Marlin (split):   {two_ms:.4f} ms")
    print(f"  BF16 cuBLAS:         {bf16_ms:.4f} ms")
    print(f"  Speedup new/old:     {old_ms/new_ms:.2f}x")
    print(f"  Speedup new/split:   {two_ms/new_ms:.2f}x")
    print(f"  Speedup new/BF16:    {bf16_ms/new_ms:.2f}x")


if __name__ == "__main__":
    ok = test_correctness_suite()
    print()

    print("=" * 70)
    print("Benchmarks")
    print("=" * 70)
    for M in [1, 4, 16, 64]:
        benchmark(M=M)
        print()

    if not ok:
        print("WARNING: Some correctness tests FAILED!")
        sys.exit(1)
    else:
        print("All correctness tests passed.")
