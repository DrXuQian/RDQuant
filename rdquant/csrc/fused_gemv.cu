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

__device__ __forceinline__ void dequant_fp8_word_to_half2_pairs(
    int q, half2* frag_b) {
  constexpr int kFp8Exponent = 4;
  constexpr int kFp16Exponent = 5;
  constexpr int kRightShift = kFp16Exponent - kFp8Exponent;
  constexpr int kMask = 0x7F007F00;
  constexpr int kBiasOffset =
      (1 << (kFp16Exponent - 1)) - (1 << (kFp8Exponent - 1));

  int out1 = (q & 0x80008000) | ((q & kMask) >> kRightShift);
  q <<= 8;
  int out2 = (q & 0x80008000) | ((q & kMask) >> kRightShift);

  frag_b[1] = *reinterpret_cast<const half2*>(&out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&out2);

  const half2 bias_reg = __float2half2_rn(float(1 << kBiasOffset));
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
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
    const int32_t* __restrict__ fp4_word_offsets,
    const int32_t* __restrict__ fp4_slot_map,
    const int32_t* __restrict__ fp8_word_offsets,
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
  int n_tile = lane_channel / 64;
  int n_in_tile = lane_channel % 64;
  const int32_t* fp4_word_row = fp4_word_offsets + n_in_tile * 4;
  const int32_t* fp4_slot_row = fp4_slot_map + n_in_tile * 16;
  const int32_t* fp8_word_row = fp8_word_offsets + n_in_tile * 4;

  for (int k0 = 0; k0 < k; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (tile.kind == kTileNVFP4) {
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
          int row_base = ((k0 + kk) / 16) * fp4_row_stride + n_tile * 128;
          float scale = fp8_e4m3_to_float(
              w_fp4_scales[lane_channel * (k / 16) + (k0 + kk) / 16]) *
              w_fp4_global_scale;
          #pragma unroll
          for (int group = 0; group < 4; ++group) {
            int packed = w_fp4_q[row_base + fp4_word_row[group]];
            const int32_t* slot_group = fp4_slot_row + group * 4;

            int sub = group * 2;
            acc += c_fp4_lut[(packed >> (4 * slot_group[0])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[1])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 1]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[2])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 8]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[3])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 9]);
          }
        }
      } else {
        float channel_scale = w_fp8_scales[lane_channel];
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
          int row_base = ((k0 + kk) / 16) * fp8_row_stride + n_tile * 256;
          #pragma unroll
          for (int group = 0; group < 4; ++group) {
            half2 frag_b[2];
            int packed = w_fp8_q[row_base + fp8_word_row[group]];
            dequant_fp8_word_to_half2_pairs(packed, frag_b);

            int sub = group * 2;
            half x0 = x_tile[kk + sub];
            half x1 = x_tile[kk + sub + 1];
            half x8 = x_tile[kk + sub + 8];
            half x9 = x_tile[kk + sub + 9];

            acc += __half2float(__low2half(frag_b[0])) * __half2float(x0) * channel_scale;
            acc += __half2float(__high2half(frag_b[0])) * __half2float(x1) * channel_scale;
            acc += __half2float(__low2half(frag_b[1])) * __half2float(x8) * channel_scale;
            acc += __half2float(__high2half(frag_b[1])) * __half2float(x9) * channel_scale;
          }
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

__global__ void fused_mixed_gemv_marlin_qweight_splitk_kernel(
    const half* __restrict__ x,
    const int32_t* __restrict__ w_fp4_q,
    const uint8_t* __restrict__ w_fp4_scales,
    float w_fp4_global_scale,
    const int32_t* __restrict__ w_fp8_q,
    const float* __restrict__ w_fp8_scales,
    const int32_t* __restrict__ fp4_word_offsets,
    const int32_t* __restrict__ fp4_slot_map,
    const int32_t* __restrict__ fp8_word_offsets,
    const int32_t* __restrict__ inv_perm,
    float* __restrict__ workspace,
    int32_t* __restrict__ tile_counters,
    half* __restrict__ y,
    int n_fp4,
    int n_fp8,
    int k,
    int parallel_k) {
  __shared__ half x_tile[kBlockK];
  __shared__ int last_slice_flag;

  TileDesc tile = resolve_tile(blockIdx.x, n_fp4, n_fp8);
  if (tile.valid_channels <= 0) {
    return;
  }

  int total_k_tiles = k / kBlockK;
  int slice_id = blockIdx.y;
  int slice_tile_begin = (total_k_tiles * slice_id) / parallel_k;
  int slice_tile_end = (total_k_tiles * (slice_id + 1)) / parallel_k;
  if (slice_tile_begin >= slice_tile_end) {
    return;
  }
  int k_begin = slice_tile_begin * kBlockK;
  int k_end = slice_tile_end * kBlockK;

  int lane_channel = tile.tile_base + threadIdx.x;
  bool active = threadIdx.x < tile.valid_channels;
  float acc = 0.0f;
  int fp4_row_stride = n_fp4 * 2;
  int fp8_row_stride = n_fp8 * 4;
  int n_tile = lane_channel / 64;
  int n_in_tile = lane_channel % 64;
  const int32_t* fp4_word_row = fp4_word_offsets + n_in_tile * 4;
  const int32_t* fp4_slot_row = fp4_slot_map + n_in_tile * 16;
  const int32_t* fp8_word_row = fp8_word_offsets + n_in_tile * 4;

  for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (tile.kind == kTileNVFP4) {
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
          int row_base = ((k0 + kk) / 16) * fp4_row_stride + n_tile * 128;
          float scale = fp8_e4m3_to_float(
              w_fp4_scales[lane_channel * (k / 16) + (k0 + kk) / 16]) *
              w_fp4_global_scale;
          #pragma unroll
          for (int group = 0; group < 4; ++group) {
            int packed = w_fp4_q[row_base + fp4_word_row[group]];
            const int32_t* slot_group = fp4_slot_row + group * 4;

            int sub = group * 2;
            acc += c_fp4_lut[(packed >> (4 * slot_group[0])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[1])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 1]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[2])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 8]);
            acc += c_fp4_lut[(packed >> (4 * slot_group[3])) & 0xF] *
                   scale * __half2float(x_tile[kk + sub + 9]);
          }
        }
      } else {
        float channel_scale = w_fp8_scales[lane_channel];
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
          int row_base = ((k0 + kk) / 16) * fp8_row_stride + n_tile * 256;
          #pragma unroll
          for (int group = 0; group < 4; ++group) {
            half2 frag_b[2];
            int packed = w_fp8_q[row_base + fp8_word_row[group]];
            dequant_fp8_word_to_half2_pairs(packed, frag_b);

            int sub = group * 2;
            half x0 = x_tile[kk + sub];
            half x1 = x_tile[kk + sub + 1];
            half x8 = x_tile[kk + sub + 8];
            half x9 = x_tile[kk + sub + 9];

            acc += __half2float(__low2half(frag_b[0])) * __half2float(x0) * channel_scale;
            acc += __half2float(__high2half(frag_b[0])) * __half2float(x1) * channel_scale;
            acc += __half2float(__low2half(frag_b[1])) * __half2float(x8) * channel_scale;
            acc += __half2float(__high2half(frag_b[1])) * __half2float(x9) * channel_scale;
          }
        }
      }
    }

    __syncthreads();
  }

  if (active) {
    atomicAdd(workspace + tile.logical_base + threadIdx.x, acc);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int prev = atomicAdd(tile_counters + blockIdx.x, 1);
    last_slice_flag = (prev + 1 == parallel_k) ? 1 : 0;
  }
  __syncthreads();

  if (last_slice_flag) {
    if (active) {
      int logical_idx = tile.logical_base + threadIdx.x;
      float total = workspace[logical_idx];
      int out_idx = inv_perm[logical_idx];
      y[out_idx] = __float2half(total);
      workspace[logical_idx] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      tile_counters[blockIdx.x] = 0;
    }
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
    const void* fp4_word_offsets,
    const void* fp4_slot_map,
    const void* fp8_word_offsets,
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
      reinterpret_cast<const int32_t*>(fp4_word_offsets),
      reinterpret_cast<const int32_t*>(fp4_slot_map),
      reinterpret_cast<const int32_t*>(fp8_word_offsets),
      reinterpret_cast<const int32_t*>(inv_perm),
      reinterpret_cast<half*>(y),
      n_fp4,
      n_fp8,
      k);
}

void fused_mixed_gemv_marlin_weights_splitk(
    const void* x,
    const void* w_fp4_q,
    const void* w_fp4_scales,
    float w_fp4_global_scale,
    const void* w_fp8_q,
    const void* w_fp8_scales,
    const void* fp4_word_offsets,
    const void* fp4_slot_map,
    const void* fp8_word_offsets,
    const void* inv_perm,
    void* workspace,
    void* tile_counters,
    void* y,
    int n_fp4,
    int n_fp8,
    int k,
    int parallel_k) {
  int num_fp4_tiles = (n_fp4 + kBlockN - 1) / kBlockN;
  int num_fp8_tiles = (n_fp8 + kBlockN - 1) / kBlockN;
  dim3 grid(num_fp4_tiles + num_fp8_tiles, parallel_k);
  dim3 block(kThreadsPerBlock);

  fused_mixed_gemv_marlin_qweight_splitk_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const int32_t*>(w_fp4_q),
      reinterpret_cast<const uint8_t*>(w_fp4_scales),
      w_fp4_global_scale,
      reinterpret_cast<const int32_t*>(w_fp8_q),
      reinterpret_cast<const float*>(w_fp8_scales),
      reinterpret_cast<const int32_t*>(fp4_word_offsets),
      reinterpret_cast<const int32_t*>(fp4_slot_map),
      reinterpret_cast<const int32_t*>(fp8_word_offsets),
      reinterpret_cast<const int32_t*>(inv_perm),
      reinterpret_cast<float*>(workspace),
      reinterpret_cast<int32_t*>(tile_counters),
      reinterpret_cast<half*>(y),
      n_fp4,
      n_fp8,
      k,
      parallel_k);
}
