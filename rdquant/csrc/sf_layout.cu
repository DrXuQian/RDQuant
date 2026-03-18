#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>

// ============================================================================
// Scale factor reorder kernel: row-major [dim0, K/32] -> CUTLASS interleaved
//
// The CUTLASS Sm120 block-scaled MMA expects scale factors in an interleaved
// layout defined by SfKMajorAtom (BM=128, BN=128, BK=128, SFVecSize=32):
//   Shape:   ((BM/4, 4), (SFVecSize, 4))  = ((32, 4), (32, 4))
//   Stride:  ((16,   4), (0,          1))
//
// Within each 128-row tile and each BK=128 chunk (= 4 K-tiles of 32):
//   Physical offset = (row_in_tile % 32) * 16 + ((row_in_tile / 32) % 4) * 4 + k_in_block
// Atom physical size = 512 (with zero-stride dimension expanded).
// Tiles arranged with Step<_2,_1>: K blocks inner, row tiles outer.
// ============================================================================

__global__ void reorder_sf_kernel(
    const uint8_t* __restrict__ src,  // row-major [dim0, K_tiles]
    uint8_t* __restrict__ dst,        // CUTLASS interleaved layout
    int dim0,
    int K_tiles                       // K / 32
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim0 * K_tiles;
    if (idx >= total) return;

    int row = idx / K_tiles;
    int kt  = idx % K_tiles;

    // Tile coordinates
    int row_tile = row / 128;
    int row_in_tile = row % 128;
    int k_block = kt / 4;
    int k_in_block = kt % 4;

    // SfKMajorAtom mapping within tile
    int r0 = row_in_tile % 32;
    int r1 = (row_in_tile / 32) % 4;
    int offset_in_atom = r0 * 16 + r1 * 4 + k_in_block;

    int num_k_blocks = (K_tiles + 3) / 4;
    int dst_idx = (row_tile * num_k_blocks + k_block) * 512 + offset_in_atom;

    dst[dst_idx] = src[idx];
}

void reorder_sf_for_cutlass(
    const void *src,
    void *dst,
    int dim0,
    int K
)
{
    int K_tiles = K / 32;
    int total = dim0 * K_tiles;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    reorder_sf_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t*>(src),
        reinterpret_cast<uint8_t*>(dst),
        dim0,
        K_tiles
    );
}

// ============================================================================
// Online MXFP8 activation quantization kernel
//
// Converts BF16 [M, K] to MXFP8 per-32 block scaled:
//   - For each 32-element block along K: compute absmax, derive shared exponent
//   - Scale and round to FP8 (e4m3)
//   - Output: FP8 data [M, K] + row-major scales [M, K/32] in UE8M0
// ============================================================================

__device__ __forceinline__ float bf16_to_float_dev(uint16_t v) {
    float f;
    uint32_t u = static_cast<uint32_t>(v) << 16;
    memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3_dev(float v) {
    const float FP8_MAX_VAL = 448.0f;
    v = fmaxf(-FP8_MAX_VAL, fminf(FP8_MAX_VAL, v));

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint8_t sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (v == 0.0f) return sign << 7;

    // FP8 e4m3: bias=7, exp range [-6, 8], 3 mantissa bits, no inf
    int fp8_exp = exp + 7;
    if (fp8_exp <= 0) {
        int shift = 1 - fp8_exp;
        if (shift > 24) return sign << 7;
        uint32_t sub_mantissa = (0x800000 | mantissa);
        sub_mantissa >>= (20 + shift);
        uint8_t m = (sub_mantissa >> 1) & 0x7;
        if ((sub_mantissa & 1) && ((m & 1) || (sub_mantissa & ~1u))) {
            m++;
            if (m > 7) { m = 0; fp8_exp = 1; }
            else fp8_exp = 0;
        } else {
            fp8_exp = 0;
        }
        return (sign << 7) | ((fp8_exp & 0xF) << 3) | (m & 0x7);
    } else if (fp8_exp >= 15) {
        return (sign << 7) | 0x7E;  // max normal
    } else {
        uint8_t m = (mantissa >> 20) & 0x7;
        uint32_t round_bit = (mantissa >> 19) & 1;
        uint32_t sticky = mantissa & 0x7FFFF;
        if (round_bit && (sticky || (m & 1))) {
            m++;
            if (m > 7) { m = 0; fp8_exp++; }
            if (fp8_exp >= 15) return (sign << 7) | 0x7E;
        }
        return (sign << 7) | ((fp8_exp & 0xF) << 3) | (m & 0x7);
    }
}

// Each thread handles one 32-element group
__global__ void quantize_act_mxfp8_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int M, int K
)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int K_tiles = K / 32;
    int tid = threadIdx.x;
    if (tid >= K_tiles) return;

    const uint16_t* row_in = input + row * K;
    uint8_t* row_out = output + row * K;
    int base = tid * 32;

    // Find absmax
    float absmax = 0.0f;
    float vals[32];
    for (int i = 0; i < 32; i++) {
        vals[i] = bf16_to_float_dev(row_in[base + i]);
        absmax = fmaxf(absmax, fabsf(vals[i]));
    }

    // Compute shared exponent (UE8M0)
    const float FP8_MAX_VAL = 448.0f;
    float scale;
    uint8_t scale_bits;
    if (absmax == 0.0f) {
        scale = 1.0f;
        scale_bits = 127;  // 2^0
    } else {
        int e = static_cast<int>(ceilf(log2f(absmax / FP8_MAX_VAL)));
        int biased = e + 127;
        biased = max(0, min(254, biased));
        scale_bits = static_cast<uint8_t>(biased);
        scale = ldexpf(1.0f, biased - 127);
    }

    scales[row * K_tiles + tid] = scale_bits;

    // Quantize
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < 32; i++) {
        float scaled_val = vals[i] * inv_scale;
        row_out[base + i] = float_to_fp8_e4m3_dev(scaled_val);
    }
}

void quantize_act_mxfp8(
    const void *input,
    void *output,
    void *scales,
    int M, int K
)
{
    assert(K % 32 == 0);
    int K_tiles = K / 32;

    int threads = K_tiles < 1024 ? K_tiles : 1024;
    dim3 grid(M);
    dim3 block(threads);

    quantize_act_mxfp8_kernel<<<grid, block>>>(
        reinterpret_cast<const uint16_t*>(input),
        reinterpret_cast<uint8_t*>(output),
        reinterpret_cast<uint8_t*>(scales),
        M, K
    );
}
