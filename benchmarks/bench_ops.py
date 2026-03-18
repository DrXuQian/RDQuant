"""
Benchmark mixed-precision linear ops on GPU.

Measures per-group GEMM latency and compares against uniform baselines.

Usage:
    python benchmarks/bench_ops.py [--n_out 2560] [--n_in 9728] [--batch 1]
"""

import argparse
import time

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bench(fn, *args, warmup=20, repeat=200, **kwargs):
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]  # median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_out", type=int, default=2560)
    parser.add_argument("--n_in", type=int, default=9728)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1,
                        help="Sequence length (decode=1, prefill=e.g.512)")
    args = parser.parse_args()

    n_out, n_in = args.n_out, args.n_in
    B, S = args.batch, args.seq
    dtype = torch.float16

    print(f"Benchmarking on {DEVICE}  "
          f"shape=[{B}×{S}, {n_in}] → [{n_out}]  dtype={dtype}\n")

    x = torch.randn(B, S, n_in, device=DEVICE, dtype=dtype)

    # ── Simulate a 40/30/30 format split (NVFP4/MXFP6 or MXFP8/FP16) ────
    n_fp4 = int(n_out * 0.40)
    n_fp8 = int(n_out * 0.30)
    n_fp16 = n_out - n_fp4 - n_fp8

    w_fp4 = torch.randn(n_fp4, n_in, device=DEVICE, dtype=dtype)
    w_fp8 = torch.randn(n_fp8, n_in, device=DEVICE, dtype=dtype)
    w_fp16 = torch.randn(n_fp16, n_in, device=DEVICE, dtype=dtype)
    w_full = torch.randn(n_out, n_in, device=DEVICE, dtype=dtype)
    inv_perm = torch.randperm(n_out, device=DEVICE)

    # ── Individual GEMMs ──────────────────────────────────────────────────
    t_fp4 = bench(F.linear, x, w_fp4)
    t_fp8 = bench(F.linear, x, w_fp8)
    t_fp16 = bench(F.linear, x, w_fp16)

    # ── Cat + permute ─────────────────────────────────────────────────────
    y_fp4 = F.linear(x, w_fp4)
    y_fp8 = F.linear(x, w_fp8)
    y_fp16_ = F.linear(x, w_fp16)
    t_cat = bench(torch.cat, [y_fp4, y_fp8, y_fp16_], dim=-1)
    y_cat = torch.cat([y_fp4, y_fp8, y_fp16_], dim=-1)
    t_perm = bench(lambda: y_cat.index_select(-1, inv_perm))

    # ── Full mixed path ───────────────────────────────────────────────────
    def mixed_path():
        a = F.linear(x, w_fp4)
        b = F.linear(x, w_fp8)
        c = F.linear(x, w_fp16)
        y = torch.cat([a, b, c], dim=-1)
        return y.index_select(-1, inv_perm)

    t_mixed = bench(mixed_path)

    # ── Uniform baselines ─────────────────────────────────────────────────
    t_uniform = bench(F.linear, x, w_full)

    total_parts = t_fp4 + t_fp8 + t_fp16 + t_cat + t_perm

    print(f"{'Operation':<25} {'Latency (μs)':>12}  {'% of total':>10}")
    print("-" * 52)
    print(f"{'NVFP4 group GEMM':<25} {t_fp4:>12.1f}  {t_fp4/total_parts*100:>9.1f}%")
    print(f"{'MXFP8 group GEMM':<25} {t_fp8:>12.1f}  {t_fp8/total_parts*100:>9.1f}%")
    print(f"{'FP16  group GEMM':<25} {t_fp16:>12.1f}  {t_fp16/total_parts*100:>9.1f}%")
    print(f"{'cat':<25} {t_cat:>12.1f}  {t_cat/total_parts*100:>9.1f}%")
    print(f"{'inv_perm':<25} {t_perm:>12.1f}  {t_perm/total_parts*100:>9.1f}%")
    print("-" * 52)
    print(f"{'sum of parts':<25} {total_parts:>12.1f}")
    print(f"{'mixed (end-to-end)':<25} {t_mixed:>12.1f}")
    print(f"{'uniform FP16 (baseline)':<25} {t_uniform:>12.1f}")
    print()

    overhead_pct = (t_mixed - t_uniform) / t_uniform * 100
    print(f"Mixed vs uniform overhead: {overhead_pct:+.1f}%")
    print(f"Multi-launch overhead:     {t_mixed - t_uniform:.1f} μs")


if __name__ == "__main__":
    main()
