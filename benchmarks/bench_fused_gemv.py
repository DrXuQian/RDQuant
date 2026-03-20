"""Benchmark the tiled fused mixed-precision GEMV prototype."""

import argparse
import sys

import torch
import torch.nn.functional as F
import vllm._custom_ops as vllm_ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    fp8_fused_exponent_bias_into_scales,
)
from vllm.scalar_type import scalar_types

sys.path.insert(0, "rdquant/csrc/build")
import rdquant_cuda

DEVICE = "cuda"
K_TILE = 128
BENCH_SEED = 20260320

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


def choose_parallel_k(n_tiles, k, target_ctas=None):
    del n_tiles, target_ctas
    return k // K_TILE


def parallel_k_candidates(num_tiles, k, sm_count):
    max_parallel_k = k // K_TILE
    candidates = {
        1,
        2,
        4,
        8,
        12,
        16,
        20,
        24,
        32,
        40,
        48,
        64,
        76,
        max_parallel_k,
        max(1, (2 * sm_count + num_tiles - 1) // num_tiles),
        max(1, (4 * sm_count + num_tiles - 1) // num_tiles),
    }
    return sorted(p for p in candidates if p <= max_parallel_k)


def autotune_parallel_k(
    x_fp16,
    w_fp4_q,
    w_fp4_scales,
    w_fp8_q,
    w_fp8_scales,
    fp4_word_offsets,
    fp4_slot_map,
    fp8_word_offsets,
    inv_perm,
    n_fp4,
    n_fp8,
    k,
    sweep_warmup,
    sweep_repeat,
):
    num_tiles = (n_fp4 + 127) // 128 + (n_fp8 + 127) // 128
    sm_count = torch.cuda.get_device_properties(x_fp16.device).multi_processor_count
    best_parallel_k = None
    best_latency = None

    for parallel_k in parallel_k_candidates(num_tiles, k, sm_count):
        workspace = torch.zeros(n_fp4 + n_fp8, device=x_fp16.device, dtype=torch.float32)
        tile_counters = torch.zeros(num_tiles, device=x_fp16.device, dtype=torch.int32)
        latency = bench(
            lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk(
                x_fp16, w_fp4_q, w_fp4_scales, 1.0,
                w_fp8_q, w_fp8_scales,
                fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
                inv_perm, workspace, tile_counters,
                n_fp4, n_fp8, k, parallel_k
            ),
            warmup=sweep_warmup,
            repeat=sweep_repeat,
        )
        if best_latency is None or latency < best_latency:
            best_latency = latency
            best_parallel_k = parallel_k

    return best_parallel_k, best_latency


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
FP8_DECODE_LUT = torch.tensor([fp8_e4m3_decode(i) for i in range(256)], dtype=torch.float32)


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


def make_marlin_group_maps(device=DEVICE):
    fp4_inv = make_marlin_weight_map(4, device="cpu")
    fp8_inv = make_marlin_weight_map(8, device="cpu")

    fp4_word_offsets = torch.empty((64, 4), dtype=torch.int32)
    fp4_slot_map = torch.empty((64, 4, 4), dtype=torch.int32)
    fp8_word_offsets = torch.empty((64, 4), dtype=torch.int32)

    for n_in_tile in range(64):
        for group in range(4):
            sub = group * 2
            base_idx = (n_in_tile // 16) * 256 + (n_in_tile % 16)

            fp4_cols = [
                fp4_inv[base_idx + (sub + 0) * 16].item(),
                fp4_inv[base_idx + (sub + 1) * 16].item(),
                fp4_inv[base_idx + (sub + 8) * 16].item(),
                fp4_inv[base_idx + (sub + 9) * 16].item(),
            ]
            fp8_cols = [
                fp8_inv[base_idx + (sub + 0) * 16].item(),
                fp8_inv[base_idx + (sub + 1) * 16].item(),
                fp8_inv[base_idx + (sub + 8) * 16].item(),
                fp8_inv[base_idx + (sub + 9) * 16].item(),
            ]

            fp4_words = {col // 8 for col in fp4_cols}
            fp8_words = {col // 4 for col in fp8_cols}
            assert len(fp4_words) == 1
            assert len(fp8_words) == 1

            fp4_word_offsets[n_in_tile, group] = fp4_cols[0] // 8
            fp8_word_offsets[n_in_tile, group] = fp8_cols[0] // 4
            fp4_slot_map[n_in_tile, group, 0] = fp4_cols[0] % 8
            fp4_slot_map[n_in_tile, group, 1] = fp4_cols[1] % 8
            fp4_slot_map[n_in_tile, group, 2] = fp4_cols[2] % 8
            fp4_slot_map[n_in_tile, group, 3] = fp4_cols[3] % 8

    return (
        fp4_word_offsets.to(device=device),
        fp4_slot_map.to(device=device),
        fp8_word_offsets.to(device=device),
    )


def decode_fp8_e4m3_tensor(raw):
    lut = FP8_DECODE_LUT.to(device=raw.device)
    return lut[raw.to(torch.long)]


def prepare_marlin_nvfp4_scales(w_fp4_scales_raw, global_scale, k, n_fp4):
    fp4_scales = decode_fp8_e4m3_tensor(w_fp4_scales_raw).T.contiguous().to(torch.float16)
    fp4_scales = marlin_permute_scales(
        fp4_scales, size_k=k, size_n=n_fp4, group_size=16
    )
    fp4_scales = nvfp4_marlin_process_scales(fp4_scales)
    marlin_global_scale = nvfp4_marlin_process_global_scale(
        torch.tensor(global_scale, device=w_fp4_scales_raw.device, dtype=torch.float16)
    )
    return fp4_scales, marlin_global_scale


def prepare_marlin_fp8_scales(w_fp8_scales, k, n_fp8):
    fp8_scales = w_fp8_scales.view(1, n_fp8).to(torch.float16)
    fp8_scales = marlin_permute_scales(
        fp8_scales, size_k=k, size_n=n_fp8, group_size=-1
    )
    return fp8_fused_exponent_bias_into_scales(fp8_scales)


def run_marlin_nvfp4(x_fp16, w_fp4_q, marlin_fp4_scales, marlin_global_scale,
                     marlin_ws4, n_fp4, k):
    return vllm_ops.marlin_gemm(
        a=x_fp16,
        c=None,
        b_q_weight=w_fp4_q,
        b_bias=None,
        b_scales=marlin_fp4_scales,
        a_scales=None,
        global_scale=marlin_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=marlin_ws4,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=1,
        size_n=n_fp4,
        size_k=k,
    )


def run_marlin_fp8(x_fp16, w_fp8_q, marlin_fp8_scales, marlin_ws8, n_fp8, k):
    return vllm_ops.marlin_gemm(
        a=x_fp16,
        c=None,
        b_q_weight=w_fp8_q,
        b_bias=None,
        b_scales=marlin_fp8_scales,
        a_scales=None,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=marlin_ws8,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=1,
        size_n=n_fp8,
        size_k=k,
    )


def run_marlin_mixed(x_fp16, w_fp4_q, marlin_fp4_scales, marlin_global_scale,
                     marlin_ws4, n_fp4, w_fp8_q, marlin_fp8_scales, marlin_ws8,
                     n_fp8, k, inv_perm):
    return torch.cat(
        [
            run_marlin_nvfp4(
                x_fp16, w_fp4_q, marlin_fp4_scales, marlin_global_scale,
                marlin_ws4, n_fp4, k
            ),
            run_marlin_fp8(
                x_fp16, w_fp8_q, marlin_fp8_scales, marlin_ws8, n_fp8, k
            ),
        ],
        dim=-1,
    ).index_select(-1, inv_perm)


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
    torch.manual_seed(BENCH_SEED)
    torch.cuda.manual_seed_all(BENCH_SEED)
    print("=" * 60)
    print("Correctness test")
    print("=" * 60)

    # Use a small shape for fast CPU-side dequant
    N_fp4, N_fp8, K = 64, 64, 256
    data = make_test_data(N_fp4, N_fp8, K)
    x = torch.randn(1, K, device=DEVICE, dtype=torch.float16)
    fp4_word_offsets, fp4_slot_map, fp8_word_offsets = make_marlin_group_maps()
    w_fp4_q = pack_fp4_to_marlin_qweight(data["w_fp4"])
    w_fp8_q = pack_fp8_to_marlin_qweight(data["w_fp8"])
    marlin_fp4_scales, marlin_global_scale = prepare_marlin_nvfp4_scales(
        data["w_fp4_scales"], data["global_scale"], K, N_fp4
    )
    marlin_fp8_scales = prepare_marlin_fp8_scales(data["w_fp8_scales"], K, N_fp8)
    marlin_ws4 = marlin_make_workspace_new(torch.device(DEVICE))
    marlin_ws8 = marlin_make_workspace_new(torch.device(DEVICE))
    num_tiles = (N_fp4 + 127) // 128 + (N_fp8 + 127) // 128
    parallel_k = choose_parallel_k(num_tiles, K)
    workspace = torch.zeros(N_fp4 + N_fp8, device=DEVICE, dtype=torch.float32)
    tile_counters = torch.zeros(num_tiles, device=DEVICE, dtype=torch.int32)

    # Fused kernel result
    y_fused = rdquant_cuda.fused_mixed_gemv_marlin_weights(
        x,
        w_fp4_q, data["w_fp4_scales"], data["global_scale"],
        w_fp8_q, data["w_fp8_scales"],
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
        data["inv_perm"],
        N_fp4, N_fp8, K
    )
    y_splitk = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk(
        x,
        w_fp4_q, data["w_fp4_scales"], data["global_scale"],
        w_fp8_q, data["w_fp8_scales"],
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
        data["inv_perm"], workspace, tile_counters,
        N_fp4, N_fp8, K, parallel_k
    )
    workspace_staged = torch.zeros_like(workspace)
    tile_counters_staged = torch.zeros_like(tile_counters)
    y_splitk_staged = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
        x,
        w_fp4_q, data["w_fp4_scales"], data["global_scale"],
        w_fp8_q, data["w_fp8_scales"],
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
        data["inv_perm"], workspace_staged, tile_counters_staged,
        N_fp4, N_fp8, K, parallel_k
    )
    workspace_auto = torch.zeros_like(workspace)
    tile_counters_auto = torch.zeros_like(tile_counters)
    y_splitk_auto = rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_auto(
        x,
        w_fp4_q, data["w_fp4_scales"], data["global_scale"],
        w_fp8_q, data["w_fp8_scales"],
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
        data["inv_perm"], workspace_auto, tile_counters_auto,
        N_fp4, N_fp8, K, parallel_k
    )
    y_marlin = run_marlin_mixed(
        x, w_fp4_q, marlin_fp4_scales, marlin_global_scale, marlin_ws4, N_fp4,
        w_fp8_q, marlin_fp8_scales, marlin_ws8, N_fp8, K, data["inv_perm"]
    )

    # Reference: dequant + matmul
    w_ref_fp16 = data["w_ref"].half()
    y_ref = F.linear(x, w_ref_fp16)  # [1, N_total]

    max_err = (y_fused.float() - y_ref.float()).abs().max().item()
    mean_err = (y_fused.float() - y_ref.float()).abs().mean().item()
    splitk_max_err = (y_splitk.float() - y_ref.float()).abs().max().item()
    splitk_mean_err = (y_splitk.float() - y_ref.float()).abs().mean().item()
    splitk_staged_max_err = (y_splitk_staged.float() - y_ref.float()).abs().max().item()
    splitk_staged_mean_err = (y_splitk_staged.float() - y_ref.float()).abs().mean().item()
    splitk_auto_max_err = (y_splitk_auto.float() - y_ref.float()).abs().max().item()
    splitk_auto_mean_err = (y_splitk_auto.float() - y_ref.float()).abs().mean().item()
    marlin_max_err = (y_marlin.float() - y_ref.float()).abs().max().item()
    marlin_mean_err = (y_marlin.float() - y_ref.float()).abs().mean().item()
    ref_norm = y_ref.float().abs().mean().item()

    print(f"  Shape: N_fp4={N_fp4}, N_fp8={N_fp8}, K={K}")
    print(f"  Base max absolute error:   {max_err:.6f}")
    print(f"  Base mean absolute error:  {mean_err:.6f}")
    print(f"  Split-K max absolute error:{splitk_max_err:.6f}")
    print(f"  Split-K mean absolute error:{splitk_mean_err:.6f}")
    print(f"  Split-K-S4 max abs error:  {splitk_staged_max_err:.6f}")
    print(f"  Split-K-S4 mean abs error: {splitk_staged_mean_err:.6f}")
    print(f"  Split-K-Auto max abs err: {splitk_auto_max_err:.6f}")
    print(f"  Split-K-Auto mean err:    {splitk_auto_mean_err:.6f}")
    print(f"  2xMarlin max abs error:    {marlin_max_err:.6f}")
    print(f"  2xMarlin mean abs error:   {marlin_mean_err:.6f}")
    print(f"  Reference mean |y|:  {ref_norm:.6f}")
    print(f"  Base relative error: {mean_err / (ref_norm + 1e-8):.4%}")
    print(f"  Split-K rel. error:  {splitk_mean_err / (ref_norm + 1e-8):.4%}")
    print(f"  Split-K-S4 rel. err: {splitk_staged_mean_err / (ref_norm + 1e-8):.4%}")
    print(f"  Split-K-Auto rel.err:{splitk_auto_mean_err / (ref_norm + 1e-8):.4%}")
    print(f"  2xMarlin rel. error: {marlin_mean_err / (ref_norm + 1e-8):.4%}")
    ok = max(max_err, splitk_max_err, splitk_staged_max_err, splitk_auto_max_err, marlin_max_err) < 0.5
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def bench_shapes(parallel_k_mode, warmup, repeat, sweep_warmup, sweep_repeat):
    """Benchmark fused GEMV vs cuBLAS BF16 for all Qwen3-4B shapes."""
    print("=" * 60)
    print("Performance benchmark: 2x Marlin vs fused split-K vs cuBLAS BF16 GEMV")
    print("=" * 60)
    print(f"{'Layer':<14} {'N':>6} {'K':>6} {'N4':>5} {'N8':>5} "
          f"{'cuBLAS':>8} {'Mrl4':>8} {'Mrl8':>8} {'M4+8':>8} {'2xMrl':>8} "
          f"{'Base':>8} {'SplitK':>8} {'Split4S':>8} {'AutoSK':>8} {'BestSK':>8} {'P_K':>5} {'Mode':>8}")
    print("-" * 176)

    fp4_word_offsets, fp4_slot_map, fp8_word_offsets = make_marlin_group_maps()

    for shape_idx, (name, N, K, N_fp4, N_fp8) in enumerate(QWEN3_4B_SHAPES):
        seed = BENCH_SEED + shape_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        N_total = N_fp4 + N_fp8
        num_tiles = (N_fp4 + 127) // 128 + (N_fp8 + 127) // 128

        # --- cuBLAS BF16 baseline ---
        x_bf16 = torch.randn(1, K, device=DEVICE, dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
        t_cublas = bench(lambda: F.linear(x_bf16, w_bf16), warmup=warmup, repeat=repeat)

        # --- Fused GEMV ---
        x_fp16 = torch.randn(1, K, device=DEVICE, dtype=torch.float16)
        w_fp4 = torch.randint(0, 256, (N_fp4, K // 2), device=DEVICE, dtype=torch.uint8)
        w_fp4_scales = torch.randint(0, 128, (N_fp4, K // 16), device=DEVICE, dtype=torch.uint8)
        w_fp8 = torch.randint(0, 256, (N_fp8, K), device=DEVICE, dtype=torch.uint8)
        w_fp8_scales = torch.rand(N_fp8, device=DEVICE, dtype=torch.float32)
        inv_perm = torch.randperm(N_total, device=DEVICE, dtype=torch.int32)
        w_fp4_q = pack_fp4_to_marlin_qweight(w_fp4)
        w_fp8_q = pack_fp8_to_marlin_qweight(w_fp8)
        marlin_fp4_scales, marlin_global_scale = prepare_marlin_nvfp4_scales(
            w_fp4_scales, 1.0, K, N_fp4
        )
        marlin_fp8_scales = prepare_marlin_fp8_scales(w_fp8_scales, K, N_fp8)
        marlin_ws4 = marlin_make_workspace_new(torch.device(DEVICE))
        marlin_ws8 = marlin_make_workspace_new(torch.device(DEVICE))
        if parallel_k_mode == "sweep":
            parallel_k, _ = autotune_parallel_k(
                x_fp16, w_fp4_q, w_fp4_scales,
                w_fp8_q, w_fp8_scales,
                fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
                inv_perm, N_fp4, N_fp8, K,
                sweep_warmup, sweep_repeat,
            )
        else:
            parallel_k = choose_parallel_k(num_tiles, K)
        workspace = torch.zeros(N_total, device=DEVICE, dtype=torch.float32)
        tile_counters = torch.zeros(num_tiles, device=DEVICE, dtype=torch.int32)

        t_marlin_fp4 = bench(lambda: run_marlin_nvfp4(
            x_fp16, w_fp4_q, marlin_fp4_scales, marlin_global_scale,
            marlin_ws4, N_fp4, K
        ), warmup=warmup, repeat=repeat)
        t_marlin_fp8 = bench(lambda: run_marlin_fp8(
            x_fp16, w_fp8_q, marlin_fp8_scales, marlin_ws8, N_fp8, K
        ), warmup=warmup, repeat=repeat)
        t_marlin_sum = t_marlin_fp4 + t_marlin_fp8
        t_marlin = bench(lambda: run_marlin_mixed(
            x_fp16, w_fp4_q, marlin_fp4_scales, marlin_global_scale, marlin_ws4,
            N_fp4, w_fp8_q, marlin_fp8_scales, marlin_ws8, N_fp8, K, inv_perm
        ), warmup=warmup, repeat=repeat)
        t_fused = bench(lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights(
            x_fp16, w_fp4_q, w_fp4_scales, 1.0,
            w_fp8_q, w_fp8_scales,
            fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
            inv_perm,
            N_fp4, N_fp8, K
        ), warmup=warmup, repeat=repeat)
        t_splitk = bench(lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk(
            x_fp16, w_fp4_q, w_fp4_scales, 1.0,
            w_fp8_q, w_fp8_scales,
            fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
            inv_perm, workspace, tile_counters,
            N_fp4, N_fp8, K, parallel_k
        ), warmup=warmup, repeat=repeat)
        workspace_staged = torch.zeros_like(workspace)
        tile_counters_staged = torch.zeros_like(tile_counters)
        t_splitk_staged = bench(
            lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
                x_fp16, w_fp4_q, w_fp4_scales, 1.0,
                w_fp8_q, w_fp8_scales,
                fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
                inv_perm, workspace_staged, tile_counters_staged,
                N_fp4, N_fp8, K, parallel_k
            ),
            warmup=warmup,
            repeat=repeat,
        )
        workspace_auto = torch.zeros_like(workspace)
        tile_counters_auto = torch.zeros_like(tile_counters)
        t_splitk_auto = bench(
            lambda: rdquant_cuda.fused_mixed_gemv_marlin_weights_splitk_auto(
                x_fp16, w_fp4_q, w_fp4_scales, 1.0,
                w_fp8_q, w_fp8_scales,
                fp4_word_offsets, fp4_slot_map, fp8_word_offsets,
                inv_perm, workspace_auto, tile_counters_auto,
                N_fp4, N_fp8, K, parallel_k
            ),
            warmup=warmup,
            repeat=repeat,
        )
        best_splitk = min(t_splitk, t_splitk_staged)
        best_mode = "scalar" if t_splitk <= t_splitk_staged else "stg4"
        print(f"{name:<14} {N:>6} {K:>6} {N_fp4:>5} {N_fp8:>5} "
              f"{t_cublas:>7.1f}us {t_marlin_fp4:>7.1f}us {t_marlin_fp8:>7.1f}us "
              f"{t_marlin_sum:>7.1f}us {t_marlin:>7.1f}us {t_fused:>7.1f}us "
              f"{t_splitk:>7.1f}us {t_splitk_staged:>8.1f}us {t_splitk_auto:>8.1f}us {best_splitk:>8.1f}us "
              f"{parallel_k:>5} {best_mode:>8}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-k-mode", choices=("fixed", "sweep"), default="fixed")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--sweep-warmup", type=int, default=50)
    parser.add_argument("--sweep-repeat", type=int, default=200)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}\n")

    ok = test_correctness()
    if not ok:
        print("Correctness test FAILED — skipping benchmarks")
        return

    bench_shapes(
        args.parallel_k_mode,
        args.warmup,
        args.repeat,
        args.sweep_warmup,
        args.sweep_repeat,
    )


if __name__ == "__main__":
    main()
