"""Benchmark the tiled fused mixed-precision GEMV prototype."""

import sys

import torch
import torch.nn.functional as F
import vllm._custom_ops as vllm_ops

sys.path.insert(0, "rdquant/csrc/build")
import rdquant_cuda

DEVICE = "cuda"

# Qwen3-4B mixed layers: (name, N_total, K, N_fp4, N_fp8)
QWEN3_4B_SHAPES = [
    ("q_proj",     4096,   2560, 1920, 2176),
    ("k_proj",     1024,   2560,  256,  768),
    ("v_proj",     1024,   2560,  640,  384),
    ("o_proj",     2560,   4096, 2432,  128),
    ("gate_proj",  9728,   2560, 5376, 4352),
    ("up_proj",    9728,   2560, 8960,  768),
    ("down_proj",  2560,   9728, 2432,  128),
]

def bench(fn, warmup=200, repeat=1000):
    """Return average latency in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        end.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms * 1000.0 / repeat


def fp8_e4m3_encode(val_f32):
    """Encode a float32 value to FP8 E4M3 raw byte (simplified)."""
    # Use torch's float8 support if available, otherwise manual
    if val_f32 == 0:
        return 0
    sign = 1 if val_f32 < 0 else 0
    val_f32 = abs(val_f32)
    # Clamp to FP8 E4M3 range: max = 448
    val_f32 = min(val_f32, 448.0)
    # Find exponent
    import math
    if val_f32 < 2**(-6) * 0.125:  # min subnormal
        return sign << 7
    if val_f32 < 2**(-6):  # subnormal
        mant = int(round(val_f32 / (2**(-9))))
        mant = max(0, min(7, mant))
        return (sign << 7) | mant
    exp = int(math.floor(math.log2(val_f32)))
    exp_biased = exp + 7
    exp_biased = max(1, min(14, exp_biased))
    mant_val = val_f32 / (2**(exp_biased - 7)) - 1.0
    mant = int(round(mant_val * 8))
    mant = max(0, min(7, mant))
    if exp_biased == 15 and mant == 7:
        mant = 6  # avoid NaN encoding
    return (sign << 7) | (exp_biased << 3) | mant


def fp8_e4m3_decode(raw):
    """Decode FP8 E4M3 raw byte to float."""
    sign = (raw >> 7) & 1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7
    if exp == 0:
        val = mant * (2**(-10))
    elif exp == 15 and mant == 7:
        val = 0.0
    else:
        val = (1.0 + mant / 8.0) * (2**(exp - 7))
    return -val if sign else val


FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def get_weight_perm(num_bits):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    if num_bits == 4:
        interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    elif num_bits == 8:
        interleave = [0, 2, 1, 3]
    else:
        raise ValueError(f"Unsupported num_bits={num_bits}")

    perm = torch.tensor(perm_list, dtype=torch.int64)
    perm = perm.view(-1, len(interleave))[:, interleave].reshape(-1)
    return perm


def make_marlin_weight_map(num_bits, device=DEVICE):
    perm = get_weight_perm(num_bits)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv.to(device=device, dtype=torch.int32)


def pack_fp4_to_marlin_qweight(w_fp4_packed):
    qweight = w_fp4_packed.contiguous().view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=w_fp4_packed.device)
    return vllm_ops.gptq_marlin_repack(qweight, perm, w_fp4_packed.size(1) * 2,
                                       w_fp4_packed.size(0), 4, False)


def pack_fp8_to_marlin_qweight(w_fp8_raw):
    qweight = w_fp8_raw.contiguous().view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=w_fp8_raw.device)
    return vllm_ops.gptq_marlin_repack(qweight, perm, w_fp8_raw.size(1),
                                       w_fp8_raw.size(0), 8, False)


def make_test_data(N_fp4, N_fp8, K, device=DEVICE):
    """Create fake quantized weights and matching dequantized reference."""
    N_total = N_fp4 + N_fp8

    # --- FP4 weights ---
    # Generate random FP4 indices (0-15) and pack into bytes
    fp4_indices = torch.randint(0, 16, (N_fp4, K), device="cpu", dtype=torch.uint8)
    # Pack pairs into bytes: low nibble = even k, high nibble = odd k
    w_fp4_packed = torch.zeros(N_fp4, K // 2, dtype=torch.uint8)
    for i in range(K // 2):
        w_fp4_packed[:, i] = fp4_indices[:, 2*i] | (fp4_indices[:, 2*i+1] << 4)

    # Block scales (FP8 E4M3, one per 16 elements)
    # Use simple power-of-2 scales for easy verification
    n_blocks = K // 16
    block_scale_values = torch.rand(N_fp4, n_blocks) * 0.1 + 0.01  # small positive
    w_fp4_scales_raw = torch.zeros(N_fp4, n_blocks, dtype=torch.uint8)
    for i in range(N_fp4):
        for j in range(n_blocks):
            w_fp4_scales_raw[i, j] = fp8_e4m3_encode(block_scale_values[i, j].item())

    global_scale = 1.0

    # Dequant FP4 reference
    w_fp4_deq = torch.zeros(N_fp4, K, dtype=torch.float32)
    for i in range(N_fp4):
        for j in range(K):
            idx = fp4_indices[i, j].item()
            block_idx = j // 16
            bs = fp8_e4m3_decode(w_fp4_scales_raw[i, block_idx].item())
            w_fp4_deq[i, j] = FP4_LUT[idx] * bs * global_scale

    # --- FP8 weights ---
    # Generate random FP8 values
    w_fp8_float = torch.randn(N_fp8, K) * 0.1
    w_fp8_raw = torch.zeros(N_fp8, K, dtype=torch.uint8)
    for i in range(N_fp8):
        for j in range(K):
            w_fp8_raw[i, j] = fp8_e4m3_encode(w_fp8_float[i, j].item())

    # Per-channel scales
    w_fp8_scales = torch.rand(N_fp8) * 0.5 + 0.1

    # Dequant FP8 reference
    w_fp8_deq = torch.zeros(N_fp8, K, dtype=torch.float32)
    for i in range(N_fp8):
        for j in range(K):
            w_fp8_deq[i, j] = fp8_e4m3_decode(w_fp8_raw[i, j].item()) * w_fp8_scales[i].item()

    # Inverse permutation (identity for correctness test)
    inv_perm = torch.arange(N_total, dtype=torch.int32)

    # Full dequanted weight for reference
    w_ref = torch.cat([w_fp4_deq, w_fp8_deq], dim=0)  # [N_total, K]

    return {
        "w_fp4": w_fp4_packed.to(device),
        "w_fp4_scales": w_fp4_scales_raw.to(device),
        "global_scale": global_scale,
        "w_fp8": w_fp8_raw.to(device),
        "w_fp8_scales": w_fp8_scales.to(device),
        "inv_perm": inv_perm.to(device),
        "w_ref": w_ref.to(device),
        "N_fp4": N_fp4,
        "N_fp8": N_fp8,
        "K": K,
    }


def test_correctness():
    """Verify fused GEMV matches dequant reference."""
    print("=" * 60)
    print("Correctness test")
    print("=" * 60)

    # Use a small shape for fast CPU-side dequant
    N_fp4, N_fp8, K = 64, 64, 256
    data = make_test_data(N_fp4, N_fp8, K)
    x = torch.randn(1, K, device=DEVICE, dtype=torch.float16)
    fp4_map = make_marlin_weight_map(4)
    fp8_map = make_marlin_weight_map(8)
    w_fp4_q = pack_fp4_to_marlin_qweight(data["w_fp4"])
    w_fp8_q = pack_fp8_to_marlin_qweight(data["w_fp8"])

    # Fused kernel result
    y_fused = rdquant_cuda.fused_mixed_gemv_marlin_weights(
        x,
        w_fp4_q, data["w_fp4_scales"], data["global_scale"],
        w_fp8_q, data["w_fp8_scales"], fp4_map, fp8_map, data["inv_perm"],
        N_fp4, N_fp8, K
    )

    # Reference: dequant + matmul
    w_ref_fp16 = data["w_ref"].half()
    y_ref = F.linear(x, w_ref_fp16)  # [1, N_total]

    max_err = (y_fused.float() - y_ref.float()).abs().max().item()
    mean_err = (y_fused.float() - y_ref.float()).abs().mean().item()
    ref_norm = y_ref.float().abs().mean().item()

    print(f"  Shape: N_fp4={N_fp4}, N_fp8={N_fp8}, K={K}")
    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  Reference mean |y|:  {ref_norm:.6f}")
    print(f"  Relative error:      {mean_err / (ref_norm + 1e-8):.4%}")
    ok = max_err < 0.5  # FP4 has limited precision
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def bench_shapes():
    """Benchmark fused GEMV vs cuBLAS BF16 for all Qwen3-4B shapes."""
    print("=" * 60)
    print("Performance benchmark: Marlin-qweight fused GEMV vs cuBLAS BF16 GEMV")
    print("=" * 60)
    print(f"{'Layer':<14} {'N':>6} {'K':>6} {'N4':>5} {'N8':>5} "
          f"{'cuBLAS':>8} {'Fused':>8} {'Speedup':>8}")
    print("-" * 70)

    for name, N, K, N_fp4, N_fp8 in QWEN3_4B_SHAPES:
        N_total = N_fp4 + N_fp8
        fp4_map = make_marlin_weight_map(4)
        fp8_map = make_marlin_weight_map(8)

        # --- cuBLAS BF16 baseline ---
        x_bf16 = torch.randn(1, K, device=DEVICE, dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
        t_cublas = bench(lambda: F.linear(x_bf16, w_bf16))

        # --- Fused GEMV ---
        x_fp16 = torch.randn(1, K, device=DEVICE, dtype=torch.float16)
        w_fp4 = torch.randint(0, 256, (N_fp4, K // 2), device=DEVICE, dtype=torch.uint8)
        w_fp4_scales = torch.randint(0, 128, (N_fp4, K // 16), device=DEVICE, dtype=torch.uint8)
        w_fp8 = torch.randint(0, 256, (N_fp8, K), device=DEVICE, dtype=torch.uint8)
        w_fp8_scales = torch.rand(N_fp8, device=DEVICE, dtype=torch.float32)
        inv_perm = torch.randperm(N_total, device=DEVICE, dtype=torch.int32)
        w_fp4_q = pack_fp4_to_marlin_qweight(w_fp4)
        w_fp8_q = pack_fp8_to_marlin_qweight(w_fp8)

        t_fused = bench(lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights(
            x_fp16, w_fp4_q, w_fp4_scales, 1.0,
            w_fp8_q, w_fp8_scales, fp4_map, fp8_map, inv_perm,
            N_fp4, N_fp8, K
        ))

        speedup = t_cublas / t_fused if t_fused > 0 else float('inf')
        print(f"{name:<14} {N:>6} {K:>6} {N_fp4:>5} {N_fp8:>5} "
              f"{t_cublas:>7.1f}us {t_fused:>7.1f}us {speedup:>7.2f}x")

    print()


def main():
    print(f"Device: {torch.cuda.get_device_name()}\n")

    ok = test_correctness()
    if not ok:
        print("Correctness test FAILED — skipping benchmarks")
        return

    bench_shapes()


if __name__ == "__main__":
    main()
