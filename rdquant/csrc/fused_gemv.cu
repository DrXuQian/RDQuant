/*
 * Fused mixed-precision GEMV kernel for RDQuant.
 *
 * This is a decode-oriented M=1 kernel with a tile scheduler that assigns
 * one CTA to one output tile. The CTA dispatches at tile granularity:
 *   - leading tiles consume NVFP4-packed channels
 *   - trailing tiles consume FP8 channels
 *
 * The input activation tile is staged through shared memory and reused across
 * all channels in the CTA. This keeps the control flow aligned with the fused
 * mixed scheduler we ultimately want for Marlin-packed weights.
 *
 * Current interface intentionally keeps the original row-major fake-quant test
 * tensors used by bench_fused_gemv.py. The next step is swapping the tile
 * loaders to consume Marlin-repacked weights without changing the scheduler.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kThreadsPerBlock = kBlockN;

enum TileKind : int {
  kTileNVFP4 = 0,
  kTileFP8 = 1,
};

struct TileDesc {
  int kind;
  int tile_base;
  int valid_channels;
  int logical_base;
};

static __constant__ float c_fp4_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t raw) {
  int sign = (raw >> 7) & 1;
  int exp = (raw >> 3) & 0xF;
  int mant = raw & 0x7;

  float val;
  if (exp == 0) {
    val = mant * 9.765625e-4f;
  } else if (exp == 15 && mant == 7) {
    val = 0.0f;
  } else {
    val = (1.0f + mant * 0.125f) * ldexpf(1.0f, exp - 7);
  }
  return sign ? -val : val;
}

__device__ __forceinline__ TileDesc resolve_tile(
    int tile_id, int n_fp4, int n_fp8) {
  int num_fp4_tiles = (n_fp4 + kBlockN - 1) / kBlockN;
  if (tile_id < num_fp4_tiles) {
    int tile_base = tile_id * kBlockN;
    int valid_channels = n_fp4 - tile_base;
    if (valid_channels > kBlockN) {
      valid_channels = kBlockN;
    }
    return TileDesc{kTileNVFP4, tile_base, valid_channels, tile_base};
  }

  int fp8_tile = tile_id - num_fp4_tiles;
  int tile_base = fp8_tile * kBlockN;
  int valid_channels = n_fp8 - tile_base;
  if (valid_channels > kBlockN) {
    valid_channels = kBlockN;
  }
  return TileDesc{kTileFP8, tile_base, valid_channels, n_fp4 + tile_base};
}

template <int BLOCK_K>
__device__ __forceinline__ void stage_x_tile(
    const half* __restrict__ x, int k0, half* __restrict__ x_tile) {
  x_tile[threadIdx.x] = x[k0 + threadIdx.x];
}

template <int BLOCK_K>
__device__ __forceinline__ float run_fp4_tile(
    const uint8_t* __restrict__ w_fp4,
    const uint8_t* __restrict__ w_fp4_scales,
    float w_fp4_global_scale,
    const half* __restrict__ x_tile,
    int channel,
    int k0,
    int k) {
  constexpr int kPairsPerScaleBlock = 8;
  constexpr int kScaleBlockK = 16;

  const int half_k = k / 2;
  const int scales_per_row = k / kScaleBlockK;
  const uint8_t* __restrict__ row_ptr = w_fp4 + channel * half_k;
  const uint8_t* __restrict__ scale_ptr = w_fp4_scales + channel * scales_per_row;

  float acc = 0.0f;
  int byte_base = k0 / 2;
  int scale_base = k0 / kScaleBlockK;

  #pragma unroll
  for (int scale_block = 0; scale_block < BLOCK_K / kScaleBlockK; ++scale_block) {
    float scale =
        fp8_e4m3_to_float(scale_ptr[scale_base + scale_block]) * w_fp4_global_scale;

    const uint8_t* __restrict__ block_ptr =
        row_ptr + byte_base + scale_block * kPairsPerScaleBlock;

    #pragma unroll
    for (int pair = 0; pair < kPairsPerScaleBlock; ++pair) {
      uint8_t packed = block_ptr[pair];
      int x_idx = scale_block * kScaleBlockK + pair * 2;

      acc += c_fp4_lut[packed & 0x0F] * scale * __half2float(x_tile[x_idx]);
      acc += c_fp4_lut[(packed >> 4) & 0x0F] * scale *
             __half2float(x_tile[x_idx + 1]);
    }
  }

  return acc;
}

template <int BLOCK_K>
__device__ __forceinline__ float run_fp8_tile(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ w_fp8_scales,
    const half* __restrict__ x_tile,
    int channel,
    int k0,
    int k) {
  const uint8_t* __restrict__ row_ptr = w_fp8 + channel * k + k0;
  float channel_scale = w_fp8_scales[channel];
  float acc = 0.0f;

  #pragma unroll 8
  for (int kk = 0; kk < BLOCK_K; ++kk) {
    acc += fp8_e4m3_to_float(row_ptr[kk]) * channel_scale *
           __half2float(x_tile[kk]);
  }

  return acc;
}

template <int NUM_BITS, int PACK_FACTOR>
__device__ __forceinline__ uint32_t load_marlin_qvalue(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ perm_map,
    int row_stride,
    int channel,
    int k_idx) {
  constexpr int kMarlinKTile = 16;
  constexpr int kMarlinNTile = 64;
  constexpr int kPackedColsPerNTile = (kMarlinKTile * kMarlinNTile) / PACK_FACTOR;

  int row = k_idx / kMarlinKTile;
  int k_in_tile = k_idx % kMarlinKTile;
  int n_tile = channel / kMarlinNTile;
  int n_in_tile = channel % kMarlinNTile;

  int perm_col = perm_map[k_in_tile * kMarlinNTile + n_in_tile];
  int packed_col = n_tile * kPackedColsPerNTile + perm_col / PACK_FACTOR;
  int shift = NUM_BITS * (perm_col % PACK_FACTOR);
  uint32_t packed = static_cast<uint32_t>(qweight[row * row_stride + packed_col]);
  return (packed >> shift) & ((1u << NUM_BITS) - 1);
}

__global__ void fused_mixed_gemv_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ w_fp4,
    const uint8_t* __restrict__ w_fp4_scales,
    float w_fp4_global_scale,
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ w_fp8_scales,
    const int32_t* __restrict__ inv_perm,
    half* __restrict__ y,
    int n_fp4,
    int n_fp8,
    int k) {
  __shared__ half x_tile[kBlockK];

  TileDesc tile = resolve_tile(blockIdx.x, n_fp4, n_fp8);
  if (tile.valid_channels <= 0) {
    return;
  }

  int lane_channel = tile.tile_base + threadIdx.x;
  bool active = threadIdx.x < tile.valid_channels;
  float acc = 0.0f;

  for (int k0 = 0; k0 < k; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (tile.kind == kTileNVFP4) {
        acc += run_fp4_tile<kBlockK>(
            w_fp4, w_fp4_scales, w_fp4_global_scale, x_tile, lane_channel, k0, k);
      } else {
        acc += run_fp8_tile<kBlockK>(
            w_fp8, w_fp8_scales, x_tile, lane_channel, k0, k);
      }
    }

    __syncthreads();
  }

  if (active) {
    int out_idx = inv_perm[tile.logical_base + threadIdx.x];
    y[out_idx] = __float2half(acc);
  }
}

__global__ void fused_mixed_gemv_marlin_qweight_kernel(
    const half* __restrict__ x,
    const int32_t* __restrict__ w_fp4_q,
    const uint8_t* __restrict__ w_fp4_scales,
    float w_fp4_global_scale,
    const int32_t* __restrict__ w_fp8_q,
    const float* __restrict__ w_fp8_scales,
    const int32_t* __restrict__ fp4_perm_map,
    const int32_t* __restrict__ fp8_perm_map,
    const int32_t* __restrict__ inv_perm,
    half* __restrict__ y,
    int n_fp4,
    int n_fp8,
    int k) {
  __shared__ half x_tile[kBlockK];

  TileDesc tile = resolve_tile(blockIdx.x, n_fp4, n_fp8);
  if (tile.valid_channels <= 0) {
    return;
  }

  int lane_channel = tile.tile_base + threadIdx.x;
  bool active = threadIdx.x < tile.valid_channels;
  float acc = 0.0f;
  int fp4_row_stride = n_fp4 * 2;
  int fp8_row_stride = n_fp8 * 4;

  for (int k0 = 0; k0 < k; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (tile.kind == kTileNVFP4) {
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
          float scale = fp8_e4m3_to_float(
              w_fp4_scales[lane_channel * (k / 16) + (k0 + kk) / 16]) *
              w_fp4_global_scale;
          #pragma unroll
          for (int inner = 0; inner < 16; ++inner) {
            uint32_t q = load_marlin_qvalue<4, 8>(
                w_fp4_q, fp4_perm_map, fp4_row_stride, lane_channel, k0 + kk + inner);
            acc += c_fp4_lut[q] * scale * __half2float(x_tile[kk + inner]);
          }
        }
      } else {
        float channel_scale = w_fp8_scales[lane_channel];
        #pragma unroll 8
        for (int kk = 0; kk < kBlockK; ++kk) {
          uint32_t q = load_marlin_qvalue<8, 4>(
              w_fp8_q, fp8_perm_map, fp8_row_stride, lane_channel, k0 + kk);
          acc += fp8_e4m3_to_float(static_cast<uint8_t>(q)) * channel_scale *
                 __half2float(x_tile[kk]);
        }
      }
    }

    __syncthreads();
  }

  if (active) {
    int out_idx = inv_perm[tile.logical_base + threadIdx.x];
    y[out_idx] = __float2half(acc);
  }
}

}  // namespace

void fused_mixed_gemv(
    const void* x,
    const void* w_fp4,
    const void* w_fp4_scales,
    float w_fp4_global_scale,
    const void* w_fp8,
    const void* w_fp8_scales,
    const void* inv_perm,
    void* y,
    int n_fp4,
    int n_fp8,
    int k) {
  int num_fp4_tiles = (n_fp4 + kBlockN - 1) / kBlockN;
  int num_fp8_tiles = (n_fp8 + kBlockN - 1) / kBlockN;
  dim3 grid(num_fp4_tiles + num_fp8_tiles);
  dim3 block(kThreadsPerBlock);

  fused_mixed_gemv_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const uint8_t*>(w_fp4),
      reinterpret_cast<const uint8_t*>(w_fp4_scales),
      w_fp4_global_scale,
      reinterpret_cast<const uint8_t*>(w_fp8),
      reinterpret_cast<const float*>(w_fp8_scales),
      reinterpret_cast<const int32_t*>(inv_perm),
      reinterpret_cast<half*>(y),
      n_fp4,
      n_fp8,
      k);
}

void fused_mixed_gemv_marlin_weights(
    const void* x,
    const void* w_fp4_q,
    const void* w_fp4_scales,
    float w_fp4_global_scale,
    const void* w_fp8_q,
    const void* w_fp8_scales,
    const void* fp4_perm_map,
    const void* fp8_perm_map,
    const void* inv_perm,
    void* y,
    int n_fp4,
    int n_fp8,
    int k) {
  int num_fp4_tiles = (n_fp4 + kBlockN - 1) / kBlockN;
  int num_fp8_tiles = (n_fp8 + kBlockN - 1) / kBlockN;
  dim3 grid(num_fp4_tiles + num_fp8_tiles);
  dim3 block(kThreadsPerBlock);

  fused_mixed_gemv_marlin_qweight_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const int32_t*>(w_fp4_q),
      reinterpret_cast<const uint8_t*>(w_fp4_scales),
      w_fp4_global_scale,
      reinterpret_cast<const int32_t*>(w_fp8_q),
      reinterpret_cast<const float*>(w_fp8_scales),
      reinterpret_cast<const int32_t*>(fp4_perm_map),
      reinterpret_cast<const int32_t*>(fp8_perm_map),
      reinterpret_cast<const int32_t*>(inv_perm),
      reinterpret_cast<half*>(y),
      n_fp4,
      n_fp8,
      k);
}
