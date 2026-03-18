"""
Benchmark mixed-precision linear ops on GPU.

Measures per-group GEMM latency and compares against uniform baselines.
Uses MX-only format groups: MXFP4/MXFP6/MXFP8.

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
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if DEVICE == "cuda":
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
          f"shape=[{B}x{S}, {n_in}] -> [{n_out}]  dtype={dtype}\n")

    x = torch.randn(B, S, n_in, device=DEVICE, dtype=dtype)

    # Simulate a 40/30/30 format split (MXFP4/MXFP6/MXFP8)
    n_mxfp4 = int(n_out * 0.40)
    n_mxfp6 = int(n_out * 0.30)
    n_mxfp8 = n_out - n_mxfp4 - n_mxfp6

    w_mxfp4 = torch.randn(n_mxfp4, n_in, device=DEVICE, dtype=dtype)
    w_mxfp6 = torch.randn(n_mxfp6, n_in, device=DEVICE, dtype=dtype)
    w_mxfp8 = torch.randn(n_mxfp8, n_in, device=DEVICE, dtype=dtype)
    w_full = torch.randn(n_out, n_in, device=DEVICE, dtype=dtype)
    inv_perm = torch.randperm(n_out, device=DEVICE)

    # Individual GEMMs
    t_mxfp4 = bench(F.linear, x, w_mxfp4)
    t_mxfp6 = bench(F.linear, x, w_mxfp6)
    t_mxfp8 = bench(F.linear, x, w_mxfp8)

    # Cat + permute
    y_mxfp4 = F.linear(x, w_mxfp4)
    y_mxfp6 = F.linear(x, w_mxfp6)
    y_mxfp8_ = F.linear(x, w_mxfp8)
    t_cat = bench(torch.cat, [y_mxfp4, y_mxfp6, y_mxfp8_], dim=-1)
    y_cat = torch.cat([y_mxfp4, y_mxfp6, y_mxfp8_], dim=-1)
    t_perm = bench(lambda: y_cat.index_select(-1, inv_perm))

    # Full mixed path
    def mixed_path():
        a = F.linear(x, w_mxfp4)
        b = F.linear(x, w_mxfp6)
        c = F.linear(x, w_mxfp8)
        y = torch.cat([a, b, c], dim=-1)
        return y.index_select(-1, inv_perm)

    t_mixed = bench(mixed_path)

    # Uniform baseline
    t_uniform = bench(F.linear, x, w_full)

    total_parts = t_mxfp4 + t_mxfp6 + t_mxfp8 + t_cat + t_perm

    print(f"{'Operation':<25} {'Latency (us)':>12}  {'% of total':>10}")
    print("-" * 52)
    print(f"{'MXFP4 group GEMM':<25} {t_mxfp4:>12.1f}  {t_mxfp4/total_parts*100:>9.1f}%")
    print(f"{'MXFP6 group GEMM':<25} {t_mxfp6:>12.1f}  {t_mxfp6/total_parts*100:>9.1f}%")
    print(f"{'MXFP8 group GEMM':<25} {t_mxfp8:>12.1f}  {t_mxfp8/total_parts*100:>9.1f}%")
    print(f"{'cat':<25} {t_cat:>12.1f}  {t_cat/total_parts*100:>9.1f}%")
    print(f"{'inv_perm':<25} {t_perm:>12.1f}  {t_perm/total_parts*100:>9.1f}%")
    print("-" * 52)
    print(f"{'sum of parts':<25} {total_parts:>12.1f}")
    print(f"{'mixed (end-to-end)':<25} {t_mixed:>12.1f}")
    print(f"{'uniform FP16 (baseline)':<25} {t_uniform:>12.1f}")
    print()

    overhead_pct = (t_mixed - t_uniform) / t_uniform * 100
    print(f"Mixed vs uniform overhead: {overhead_pct:+.1f}%")
    print(f"Multi-launch overhead:     {t_mixed - t_uniform:.1f} us")


if __name__ == "__main__":
    main()
