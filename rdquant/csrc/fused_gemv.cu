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
constexpr int kQweightTileKChunks = kBlockK / 16;
constexpr int kFp4WordsPerSubtile = 128;
constexpr int kFp8WordsPerSubtile = 256;
constexpr int kFp8ChunkSubtiles = 2;
constexpr int kFp4MaxWordsPerKTile = kQweightTileKChunks * 2 * kFp4WordsPerSubtile;
constexpr int kFp8MaxWordsPerKTile = kQweightTileKChunks * 2 * kFp8WordsPerSubtile;
constexpr int kFp8MaxWordsPer16Chunk = 2 * kFp8WordsPerSubtile;
constexpr int kFp8MaxWordsPerChunk =
    kFp8ChunkSubtiles * 2 * kFp8WordsPerSubtile;
constexpr int kFp4MaxScalesPerKTile = kBlockN * kQweightTileKChunks;
constexpr int kFp8RegisterStages = 2;
constexpr int kNvfp4RegisterStages = 2;

static_assert(kQweightTileKChunks % kFp8ChunkSubtiles == 0,
              "FP8 chunk staging expects an integer number of chunks per kBlockK");

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

__host__ __device__ __forceinline__ bool
should_use_staged_nvfp4_splitk_variant(int n_fp4, int n_fp8, int k) {
  (void)k;
  const int num_tiles = (n_fp4 + kBlockN - 1) / kBlockN +
                        (n_fp8 + kBlockN - 1) / kBlockN;
  const int num_fp8_tiles = (n_fp8 + kBlockN - 1) / kBlockN;
  if (num_tiles <= 8 && num_fp8_tiles >= 2) {
    return true;
  }
  return false;
}

__host__ __device__ __forceinline__ bool
should_use_wide_fp8_splitk_variant(int n_fp4, int n_fp8, int k) {
  const int num_tiles = (n_fp4 + kBlockN - 1) / kBlockN +
                        (n_fp8 + kBlockN - 1) / kBlockN;
  const int num_fp8_tiles = (n_fp8 + kBlockN - 1) / kBlockN;
  if (num_tiles <= 8 && num_fp8_tiles >= 4) {
    return true;
  }
  return num_fp8_tiles == 1 && k >= 4096;
}


static __constant__ float c_fp4_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ void cp_async4(void* smem_ptr, const void* glob_ptr) {
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

__device__ __forceinline__ void cp_async_fence() {}

template <int n>
__device__ __forceinline__ void cp_async_wait() {}
#else
__device__ __forceinline__ void cp_async4(void* smem_ptr, const void* glob_ptr) {
  constexpr int kBytes = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(kBytes));
}

__device__ __forceinline__ void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}
#endif

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

__device__ __forceinline__ int fp8_words_per_k_chunk(int valid_channels) {
  return ((valid_channels + 63) / 64) * kFp8WordsPerSubtile;
}

__device__ __forceinline__ int fp4_words_per_k_chunk(int valid_channels) {
  return ((valid_channels + 63) / 64) * kFp4WordsPerSubtile;
}

__device__ __forceinline__ int fp4_word_offset(int n_in_tile, int group) {
  return ((n_in_tile & 7) << 4) + (n_in_tile >> 4) + (group << 2);
}

__device__ __forceinline__ int fp8_word_offset(int n_in_tile, int group) {
  return ((n_in_tile & 7) << 5) + (n_in_tile >> 3) + (group << 3);
}

__device__ __forceinline__ int fp4_slot_base(int n_in_tile) {
  return ((n_in_tile >> 3) & 1) << 1;
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

__device__ __forceinline__ void load_nvfp4_scalar_register_stage(
    const int32_t* __restrict__ row_ptr,
    int32_t* __restrict__ packed_stage);

__device__ __forceinline__ float consume_nvfp4_scalar_register_stage(
    const int32_t* __restrict__ packed_stage,
    float scale,
    const half* __restrict__ x_tile,
    int kk,
    int shift_lo_01,
    int shift_hi_01,
    int shift_lo_89,
    int shift_hi_89);

__device__ __forceinline__ float run_nvfp4_qweight_k_tile_scalar(
    const int32_t* __restrict__ w_fp4_q,
    const uint8_t* __restrict__ w_fp4_scales,
    float w_fp4_global_scale,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int channel,
    int n_tile,
    int fp4_row_stride,
    int k,
    int k0) {
  float acc = 0.0f;
  const int slot_base = fp4_slot_base(n_in_tile);
  const int shift_lo_01 = 4 * (slot_base + 0);
  const int shift_hi_01 = 4 * (slot_base + 4);
  const int shift_lo_89 = 4 * (slot_base + 1);
  const int shift_hi_89 = 4 * (slot_base + 5);
  const int scale_base = channel * (k / 16) + (k0 / 16);
  const uint2 packed_scales =
      reinterpret_cast<const uint2*>(w_fp4_scales + scale_base)[0];
  const int qweight_base =
      (k0 / 16) * fp4_row_stride + n_tile * 128 + fp4_word_offset(n_in_tile, 0);
  const int32_t* __restrict__ row_ptr = w_fp4_q + qweight_base;
  int32_t packed_regs[kNvfp4RegisterStages][4];

  load_nvfp4_scalar_register_stage(row_ptr, packed_regs[0]);
  if constexpr (kQweightTileKChunks > 1) {
    load_nvfp4_scalar_register_stage(row_ptr + fp4_row_stride, packed_regs[1]);
  }

  #pragma unroll
  for (int kk_block = 0; kk_block < kQweightTileKChunks; ++kk_block) {
    const uint32_t scale_word =
        (kk_block < 4) ? packed_scales.x : packed_scales.y;
    const int scale_shift = (kk_block & 3) * 8;
    const float scale =
        fp8_e4m3_to_float((scale_word >> scale_shift) & 0xFF) *
        w_fp4_global_scale;
    const int pipe = kk_block % kNvfp4RegisterStages;
    const int kk = kk_block * 16;

    acc += consume_nvfp4_scalar_register_stage(
        packed_regs[pipe], scale, x_tile, kk,
        shift_lo_01, shift_hi_01, shift_lo_89, shift_hi_89);

    const int next_block = kk_block + kNvfp4RegisterStages;
    if (next_block < kQweightTileKChunks) {
      load_nvfp4_scalar_register_stage(
          row_ptr + next_block * fp4_row_stride, packed_regs[pipe]);
    }
  }

  return acc;
}

__device__ __forceinline__ float run_nvfp4_qweight_k_tile_staged(
    const int32_t* __restrict__ sh_fp4_q_tile,
    const uint8_t* __restrict__ sh_fp4_scales,
    float w_fp4_global_scale,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_channel,
    int local_n_tile,
    int words_per_k_chunk) {
  float acc = 0.0f;
  const int slot_base = fp4_slot_base(n_in_tile);

  #pragma unroll
  for (int kk = 0; kk < kBlockK; kk += 16) {
    const int row_base = (kk / 16) * words_per_k_chunk +
                         local_n_tile * kFp4WordsPerSubtile;
    const float scale = fp8_e4m3_to_float(
        sh_fp4_scales[local_channel * kQweightTileKChunks + (kk / 16)]) *
        w_fp4_global_scale;
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int packed =
          sh_fp4_q_tile[row_base + fp4_word_offset(n_in_tile, group)];
      const int sub = group * 2;
      acc += c_fp4_lut[(packed >> (4 * (slot_base + 0))) & 0xF] *
             scale * __half2float(x_tile[kk + sub]);
      acc += c_fp4_lut[(packed >> (4 * (slot_base + 4))) & 0xF] *
             scale * __half2float(x_tile[kk + sub + 1]);
      acc += c_fp4_lut[(packed >> (4 * (slot_base + 1))) & 0xF] *
             scale * __half2float(x_tile[kk + sub + 8]);
      acc += c_fp4_lut[(packed >> (4 * (slot_base + 5))) & 0xF] *
             scale * __half2float(x_tile[kk + sub + 9]);
    }
  }

  return acc;
}

__device__ __forceinline__ void load_nvfp4_scalar_register_stage(
    const int32_t* __restrict__ row_ptr,
    int32_t* __restrict__ packed_stage) {
  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    packed_stage[group] = row_ptr[group * 4];
  }
}

__device__ __forceinline__ float consume_nvfp4_scalar_register_stage(
    const int32_t* __restrict__ packed_stage,
    float scale,
    const half* __restrict__ x_tile,
    int kk,
    int shift_lo_01,
    int shift_hi_01,
    int shift_lo_89,
    int shift_hi_89) {
  float acc = 0.0f;

  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int packed = packed_stage[group];
    const int sub = group * 2;
    acc += c_fp4_lut[(packed >> shift_lo_01) & 0xF] *
           scale * __half2float(x_tile[kk + sub]);
    acc += c_fp4_lut[(packed >> shift_hi_01) & 0xF] *
           scale * __half2float(x_tile[kk + sub + 1]);
    acc += c_fp4_lut[(packed >> shift_lo_89) & 0xF] *
           scale * __half2float(x_tile[kk + sub + 8]);
    acc += c_fp4_lut[(packed >> shift_hi_89) & 0xF] *
           scale * __half2float(x_tile[kk + sub + 9]);
  }

  return acc;
}

__device__ __forceinline__ float sum_half2(half2 x) {
  return __half2float(__low2half(x)) + __half2float(__high2half(x));
}

__device__ __forceinline__ void load_fp8_register_stage(
    const int32_t* __restrict__ sh_fp8_q_tile,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_n_tile,
    int words_per_k_chunk,
    int kk,
    int32_t* __restrict__ packed_stage,
    half2* __restrict__ x01_stage,
    half2* __restrict__ x89_stage) {
  const int row_base = (kk / 16) * words_per_k_chunk +
                       local_n_tile * kFp8WordsPerSubtile;

  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    packed_stage[group] = sh_fp8_q_tile[row_base + fp8_word_offset(n_in_tile, group)];
    const int sub = group * 2;
    x01_stage[group] = __halves2half2(x_tile[kk + sub], x_tile[kk + sub + 1]);
    x89_stage[group] = __halves2half2(x_tile[kk + sub + 8], x_tile[kk + sub + 9]);
  }
}

__device__ __forceinline__ void load_fp8_chunk_register_stage(
    const int32_t* __restrict__ sh_fp8_q_chunk,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_n_tile,
    int words_per_k_chunk,
    int chunk_subtile,
    int x_kk,
    int32_t* __restrict__ packed_stage,
    half2* __restrict__ x01_stage,
    half2* __restrict__ x89_stage) {
  const int row_base = chunk_subtile * words_per_k_chunk +
                       local_n_tile * kFp8WordsPerSubtile;

  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    packed_stage[group] =
        sh_fp8_q_chunk[row_base + fp8_word_offset(n_in_tile, group)];
    const int sub = group * 2;
    x01_stage[group] =
        __halves2half2(x_tile[x_kk + sub], x_tile[x_kk + sub + 1]);
    x89_stage[group] =
        __halves2half2(x_tile[x_kk + sub + 8], x_tile[x_kk + sub + 9]);
  }
}

__device__ __forceinline__ float consume_fp8_register_stage(
    const int32_t* __restrict__ packed_stage,
    const half2* __restrict__ x01_stage,
    const half2* __restrict__ x89_stage,
    half2 scale_h2) {
  float acc = 0.0f;

  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    half2 frag_b[2];
    dequant_fp8_word_to_half2_pairs(packed_stage[group], frag_b);

    frag_b[0] = __hmul2(frag_b[0], scale_h2);
    frag_b[1] = __hmul2(frag_b[1], scale_h2);

    acc += sum_half2(__hmul2(frag_b[0], x01_stage[group]));
    acc += sum_half2(__hmul2(frag_b[1], x89_stage[group]));
  }

  return acc;
}

__device__ __forceinline__ float run_fp8_qweight_k_tile_half2(
    const int32_t* __restrict__ sh_fp8_q_tile,
    const float* __restrict__ w_fp8_scales,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int channel,
    int local_n_tile,
    int words_per_k_chunk) {
  const half2 scale_h2 = __float2half2_rn(w_fp8_scales[channel]);
  constexpr int kKSubtiles = kBlockK / 16;
  int32_t packed_regs[kFp8RegisterStages][4];
  half2 x01_regs[kFp8RegisterStages][4];
  half2 x89_regs[kFp8RegisterStages][4];
  float acc = 0.0f;

  load_fp8_register_stage(
      sh_fp8_q_tile, x_tile, n_in_tile, local_n_tile, words_per_k_chunk, 0,
      packed_regs[0], x01_regs[0], x89_regs[0]);
  if constexpr (kKSubtiles > 1) {
    load_fp8_register_stage(
        sh_fp8_q_tile, x_tile, n_in_tile, local_n_tile, words_per_k_chunk,
        16, packed_regs[1], x01_regs[1], x89_regs[1]);
  }

  #pragma unroll
  for (int kk_block = 0; kk_block < kKSubtiles; ++kk_block) {
    const int pipe = kk_block % kFp8RegisterStages;
    acc += consume_fp8_register_stage(
        packed_regs[pipe], x01_regs[pipe], x89_regs[pipe], scale_h2);

    const int next_block = kk_block + kFp8RegisterStages;
    if (next_block < kKSubtiles) {
      load_fp8_register_stage(
          sh_fp8_q_tile, x_tile, n_in_tile, local_n_tile, words_per_k_chunk,
          next_block * 16, packed_regs[pipe], x01_regs[pipe], x89_regs[pipe]);
    }
  }

  return acc;
}

__device__ __forceinline__ float run_fp8_qweight_16_chunk_half2(
    const int32_t* __restrict__ sh_fp8_q_chunk,
    half2 scale_h2,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_n_tile,
    int words_per_k_chunk,
    int kk) {
  int32_t packed_regs[4];
  half2 x01_regs[4];
  half2 x89_regs[4];
  load_fp8_chunk_register_stage(
      sh_fp8_q_chunk, x_tile, n_in_tile, local_n_tile, words_per_k_chunk, 0, kk,
      packed_regs, x01_regs, x89_regs);
  return consume_fp8_register_stage(packed_regs, x01_regs, x89_regs, scale_h2);
}

__device__ __forceinline__ float run_fp8_qweight_chunk_half2(
    const int32_t* __restrict__ sh_fp8_q_chunk,
    half2 scale_h2,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_n_tile,
    int words_per_k_chunk,
    int kk_base) {
  int32_t packed_regs[kFp8ChunkSubtiles][4];
  half2 x01_regs[kFp8ChunkSubtiles][4];
  half2 x89_regs[kFp8ChunkSubtiles][4];
  float acc = 0.0f;

  #pragma unroll
  for (int subtile = 0; subtile < kFp8ChunkSubtiles; ++subtile) {
    load_fp8_chunk_register_stage(
        sh_fp8_q_chunk, x_tile, n_in_tile, local_n_tile, words_per_k_chunk,
        subtile, kk_base + subtile * 16, packed_regs[subtile],
        x01_regs[subtile], x89_regs[subtile]);
  }

  #pragma unroll
  for (int subtile = 0; subtile < kFp8ChunkSubtiles; ++subtile) {
    acc += consume_fp8_register_stage(
        packed_regs[subtile], x01_regs[subtile], x89_regs[subtile], scale_h2);
  }

  return acc;
}

__device__ __forceinline__ void stage_fp8_qweight_k_tile(
    const int32_t* __restrict__ w_fp8_q,
    int32_t* __restrict__ sh_fp8_q_tile,
    int tile_base,
  int valid_channels,
  int fp8_row_stride,
  int k0) {
  const int words_per_k_chunk = fp8_words_per_k_chunk(valid_channels);
  const int tile_word_base = (tile_base / 64) * kFp8WordsPerSubtile;
  const int vecs_per_k_chunk = words_per_k_chunk / 4;

  for (int vec_idx = threadIdx.x; vec_idx < kQweightTileKChunks * vecs_per_k_chunk;
       vec_idx += blockDim.x) {
    const int kk_block = vec_idx / vecs_per_k_chunk;
    const int local_vec = vec_idx % vecs_per_k_chunk;
    const int global_row_base =
        ((k0 / 16) + kk_block) * fp8_row_stride + tile_word_base;
    void* smem_ptr = reinterpret_cast<void*>(
        reinterpret_cast<int4*>(sh_fp8_q_tile + kk_block * words_per_k_chunk) +
        local_vec);
    const void* glob_ptr = reinterpret_cast<const void*>(
        reinterpret_cast<const int4*>(w_fp8_q + global_row_base) + local_vec);
    cp_async4(smem_ptr, glob_ptr);
  }
}

__device__ __forceinline__ void stage_fp8_qweight_16_chunk(
    const int32_t* __restrict__ w_fp8_q,
    int32_t* __restrict__ sh_fp8_q_chunk,
    int tile_base,
    int valid_channels,
    int fp8_row_stride,
    int k0,
    int kk_block) {
  const int words_per_k_chunk = fp8_words_per_k_chunk(valid_channels);
  const int tile_word_base = (tile_base / 64) * kFp8WordsPerSubtile;
  const int vecs_per_k_chunk = words_per_k_chunk / 4;
  const int global_row_base =
      ((k0 / 16) + kk_block) * fp8_row_stride + tile_word_base;

  for (int vec_idx = threadIdx.x; vec_idx < vecs_per_k_chunk;
       vec_idx += blockDim.x) {
    void* smem_ptr = reinterpret_cast<void*>(
        reinterpret_cast<int4*>(sh_fp8_q_chunk) + vec_idx);
    const void* glob_ptr = reinterpret_cast<const void*>(
        reinterpret_cast<const int4*>(w_fp8_q + global_row_base) + vec_idx);
    cp_async4(smem_ptr, glob_ptr);
  }
}

__device__ __forceinline__ void stage_fp8_qweight_32_chunk(
    const int32_t* __restrict__ w_fp8_q,
    int32_t* __restrict__ sh_fp8_q_chunk,
    int tile_base,
    int valid_channels,
    int fp8_row_stride,
    int k0,
    int kk_block) {
  const int words_per_k_chunk = fp8_words_per_k_chunk(valid_channels);
  const int tile_word_base = (tile_base / 64) * kFp8WordsPerSubtile;
  const int vecs_per_k_chunk = words_per_k_chunk / 4;
  const int total_vecs = kFp8ChunkSubtiles * vecs_per_k_chunk;

  for (int vec_idx = threadIdx.x; vec_idx < total_vecs;
       vec_idx += blockDim.x) {
    const int subtile = vec_idx / vecs_per_k_chunk;
    const int local_vec = vec_idx % vecs_per_k_chunk;
    const int global_row_base =
        ((k0 / 16) + kk_block + subtile) * fp8_row_stride + tile_word_base;
    void* smem_ptr = reinterpret_cast<void*>(
        reinterpret_cast<int4*>(sh_fp8_q_chunk + subtile * words_per_k_chunk) +
        local_vec);
    const void* glob_ptr = reinterpret_cast<const void*>(
        reinterpret_cast<const int4*>(w_fp8_q + global_row_base) + local_vec);
    cp_async4(smem_ptr, glob_ptr);
  }
}

__device__ __forceinline__ void stage_fp4_qweight_k_tile(
    const int32_t* __restrict__ w_fp4_q,
    int32_t* __restrict__ sh_fp4_q_tile,
    int tile_base,
  int valid_channels,
  int fp4_row_stride,
  int k0) {
  const int words_per_k_chunk = fp4_words_per_k_chunk(valid_channels);
  const int tile_word_base = (tile_base / 64) * kFp4WordsPerSubtile;
  const int vecs_per_k_chunk = words_per_k_chunk / 4;

  for (int vec_idx = threadIdx.x; vec_idx < kQweightTileKChunks * vecs_per_k_chunk;
       vec_idx += blockDim.x) {
    const int kk_block = vec_idx / vecs_per_k_chunk;
    const int local_vec = vec_idx % vecs_per_k_chunk;
    const int global_row_base =
        ((k0 / 16) + kk_block) * fp4_row_stride + tile_word_base;
    void* smem_ptr = reinterpret_cast<void*>(
        reinterpret_cast<int4*>(sh_fp4_q_tile + kk_block * words_per_k_chunk) +
        local_vec);
    const void* glob_ptr = reinterpret_cast<const void*>(
        reinterpret_cast<const int4*>(w_fp4_q + global_row_base) + local_vec);
    cp_async4(smem_ptr, glob_ptr);
  }
}

__device__ __forceinline__ void stage_fp4_scales_k_tile(
    const uint8_t* __restrict__ w_fp4_scales,
    uint8_t* __restrict__ sh_fp4_scales,
    int tile_base,
    int valid_channels,
    int k,
    int k0) {
  const int scales_per_channel = k / 16;
  const int total_scales = valid_channels * kQweightTileKChunks;

  for (int scale_idx = threadIdx.x; scale_idx < total_scales; scale_idx += blockDim.x) {
    const int local_channel = scale_idx / kQweightTileKChunks;
    const int kk_block = scale_idx % kQweightTileKChunks;
    sh_fp4_scales[scale_idx] =
        w_fp4_scales[(tile_base + local_channel) * scales_per_channel +
                     (k0 / 16) + kk_block];
  }
}

// Keep a stable entry point for each dtype path so the mixed scheduler does not
// need to change again when the scalar path is replaced with a Marlin-style
// tile engine.
__device__ __forceinline__ float run_nvfp4_qweight_k_tile(
    const int32_t* __restrict__ sh_fp4_q_tile,
    const uint8_t* __restrict__ sh_fp4_scales,
    float w_fp4_global_scale,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int local_channel,
    int local_n_tile,
    int words_per_k_chunk) {
  return run_nvfp4_qweight_k_tile_staged(
      sh_fp4_q_tile, sh_fp4_scales, w_fp4_global_scale, x_tile, n_in_tile,
      local_channel, local_n_tile, words_per_k_chunk);
}

__device__ __forceinline__ float run_fp8_qweight_k_tile(
    const int32_t* __restrict__ sh_fp8_q_tile,
    const float* __restrict__ w_fp8_scales,
    const half* __restrict__ x_tile,
    int n_in_tile,
    int channel,
    int local_n_tile,
    int words_per_k_chunk) {
  return run_fp8_qweight_k_tile_half2(
      sh_fp8_q_tile, w_fp8_scales, x_tile, n_in_tile, channel,
      local_n_tile, words_per_k_chunk);
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
  __shared__ union {
    int32_t fp4_q[kFp4MaxWordsPerKTile];
    int32_t fp8_q[kFp8MaxWordsPerKTile];
  } sh_q_tile;
  __shared__ uint8_t sh_fp4_scales[kFp4MaxScalesPerKTile];

  TileDesc tile = resolve_tile(blockIdx.x, n_fp4, n_fp8);
  if (tile.valid_channels <= 0) {
    return;
  }

  int lane_channel = tile.tile_base + threadIdx.x;
  bool active = threadIdx.x < tile.valid_channels;
  float acc = 0.0f;
  int fp4_row_stride = n_fp4 * 2;
  int fp8_row_stride = n_fp8 * 4;
  int local_channel = threadIdx.x;
  int local_n_tile = threadIdx.x / 64;
  int n_in_tile = lane_channel % 64;
  (void)fp4_word_offsets;
  (void)fp4_slot_map;
  (void)fp8_word_offsets;

  for (int k0 = 0; k0 < k; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    if (tile.kind == kTileNVFP4) {
      stage_fp4_qweight_k_tile(
          w_fp4_q, sh_q_tile.fp4_q, tile.tile_base, tile.valid_channels,
          fp4_row_stride, k0);
      stage_fp4_scales_k_tile(
          w_fp4_scales, sh_fp4_scales, tile.tile_base, tile.valid_channels, k, k0);
      cp_async_fence();
      cp_async_wait<0>();
    } else {
      stage_fp8_qweight_k_tile(
          w_fp8_q, sh_q_tile.fp8_q, tile.tile_base, tile.valid_channels,
          fp8_row_stride, k0);
      cp_async_fence();
      cp_async_wait<0>();
    }
    __syncthreads();

    if (active) {
      if (tile.kind == kTileNVFP4) {
        acc += run_nvfp4_qweight_k_tile(
            sh_q_tile.fp4_q, sh_fp4_scales, w_fp4_global_scale,
            x_tile, n_in_tile, local_channel, local_n_tile,
            fp4_words_per_k_chunk(tile.valid_channels));
      } else {
        acc += run_fp8_qweight_k_tile(
            sh_q_tile.fp8_q, w_fp8_scales, x_tile, n_in_tile, lane_channel,
            local_n_tile, fp8_words_per_k_chunk(tile.valid_channels));
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
  __shared__ int32_t sh_fp8_q_chunk[2][kFp8MaxWordsPer16Chunk];
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
  int local_n_tile = threadIdx.x / 64;
  int n_in_tile = lane_channel % 64;
  const bool is_fp8_tile = tile.kind == kTileFP8;
  (void)fp4_word_offsets;
  (void)fp4_slot_map;
  (void)fp8_word_offsets;
  const half2 fp8_scale_h2 =
      (active && is_fp8_tile) ? __float2half2_rn(w_fp8_scales[lane_channel])
                              : __float2half2_rn(0.0f);

  for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (!is_fp8_tile) {
        acc += run_nvfp4_qweight_k_tile_scalar(
            w_fp4_q, w_fp4_scales, w_fp4_global_scale,
            x_tile, n_in_tile, lane_channel, n_tile,
            fp4_row_stride, k, k0);
      }
    }
    if (is_fp8_tile) {
      const int words_per_k_chunk = fp8_words_per_k_chunk(tile.valid_channels);
      stage_fp8_qweight_16_chunk(
          w_fp8_q, sh_fp8_q_chunk[0], tile.tile_base, tile.valid_channels,
          fp8_row_stride, k0, 0);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      #pragma unroll
      for (int kk_block = 0, pipe = 0; kk_block < kQweightTileKChunks;
           ++kk_block, pipe ^= 1) {
        const int next_kk_block = kk_block + 1;
        if (next_kk_block < kQweightTileKChunks) {
          stage_fp8_qweight_16_chunk(
              w_fp8_q, sh_fp8_q_chunk[pipe ^ 1], tile.tile_base,
              tile.valid_channels, fp8_row_stride, k0, next_kk_block);
          cp_async_fence();
        }

        if (active) {
          acc += run_fp8_qweight_16_chunk_half2(
              sh_fp8_q_chunk[pipe], fp8_scale_h2, x_tile, n_in_tile,
              local_n_tile, words_per_k_chunk, kk_block * 16);
        }

        if (next_kk_block < kQweightTileKChunks) {
          cp_async_wait<0>();
        }
        __syncthreads();
      }
    } else {
      __syncthreads();
    }
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

__global__ void fused_mixed_gemv_marlin_qweight_splitk_wide_fp8_kernel(
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
  __shared__ int32_t sh_fp8_q_chunk[2][kFp8MaxWordsPerChunk];
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
  int local_n_tile = threadIdx.x / 64;
  int n_in_tile = lane_channel % 64;
  const bool is_fp8_tile = tile.kind == kTileFP8;
  (void)fp4_word_offsets;
  (void)fp4_slot_map;
  (void)fp8_word_offsets;
  const half2 fp8_scale_h2 =
      (active && is_fp8_tile) ? __float2half2_rn(w_fp8_scales[lane_channel])
                              : __float2half2_rn(0.0f);

  for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    __syncthreads();

    if (active) {
      if (!is_fp8_tile) {
        acc += run_nvfp4_qweight_k_tile_scalar(
            w_fp4_q, w_fp4_scales, w_fp4_global_scale,
            x_tile, n_in_tile, lane_channel, n_tile,
            fp4_row_stride, k, k0);
      }
    }
    if (is_fp8_tile) {
      const int words_per_k_chunk = fp8_words_per_k_chunk(tile.valid_channels);
      stage_fp8_qweight_32_chunk(
          w_fp8_q, sh_fp8_q_chunk[0], tile.tile_base, tile.valid_channels,
          fp8_row_stride, k0, 0);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      #pragma unroll
      for (int kk_block = 0, pipe = 0; kk_block < kQweightTileKChunks;
           kk_block += kFp8ChunkSubtiles, pipe ^= 1) {
        const int next_kk_block = kk_block + kFp8ChunkSubtiles;
        if (next_kk_block < kQweightTileKChunks) {
          stage_fp8_qweight_32_chunk(
              w_fp8_q, sh_fp8_q_chunk[pipe ^ 1], tile.tile_base,
              tile.valid_channels, fp8_row_stride, k0, next_kk_block);
          cp_async_fence();
        }

        if (active) {
          acc += run_fp8_qweight_chunk_half2(
              sh_fp8_q_chunk[pipe], fp8_scale_h2, x_tile, n_in_tile,
              local_n_tile, words_per_k_chunk, kk_block * 16);
        }

        if (next_kk_block < kQweightTileKChunks) {
          cp_async_wait<0>();
        }
        __syncthreads();
      }
    } else {
      __syncthreads();
    }
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

__global__ void fused_mixed_gemv_marlin_qweight_splitk_staged_nvfp4_kernel(
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
  __shared__ union {
    int32_t fp4_q[kFp4MaxWordsPerKTile];
    int32_t fp8_q_chunk[2][kFp8MaxWordsPerChunk];
  } sh_q_tile;
  __shared__ uint8_t sh_fp4_scales[kFp4MaxScalesPerKTile];
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
  int local_channel = threadIdx.x;
  int local_n_tile = threadIdx.x / 64;
  int n_in_tile = lane_channel % 64;
  (void)fp4_word_offsets;
  (void)fp4_slot_map;
  (void)fp8_word_offsets;
  const bool is_fp8_tile = tile.kind == kTileFP8;
  const half2 fp8_scale_h2 =
      (active && is_fp8_tile) ? __float2half2_rn(w_fp8_scales[lane_channel])
                              : __float2half2_rn(0.0f);

  for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
    stage_x_tile<kBlockK>(x, k0, x_tile);
    if (!is_fp8_tile) {
      stage_fp4_qweight_k_tile(
          w_fp4_q, sh_q_tile.fp4_q, tile.tile_base, tile.valid_channels,
          fp4_row_stride, k0);
      stage_fp4_scales_k_tile(
          w_fp4_scales, sh_fp4_scales, tile.tile_base, tile.valid_channels, k, k0);
      cp_async_fence();
      cp_async_wait<0>();
    }
    __syncthreads();

    if (active) {
      if (!is_fp8_tile) {
        acc += run_nvfp4_qweight_k_tile(
            sh_q_tile.fp4_q, sh_fp4_scales, w_fp4_global_scale,
            x_tile, n_in_tile, local_channel, local_n_tile,
            fp4_words_per_k_chunk(tile.valid_channels));
      }
    }
    if (is_fp8_tile) {
      const int words_per_k_chunk = fp8_words_per_k_chunk(tile.valid_channels);
      stage_fp8_qweight_32_chunk(
          w_fp8_q, sh_q_tile.fp8_q_chunk[0], tile.tile_base, tile.valid_channels,
          fp8_row_stride, k0, 0);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      #pragma unroll
      for (int kk_block = 0, pipe = 0; kk_block < kQweightTileKChunks;
           kk_block += kFp8ChunkSubtiles, pipe ^= 1) {
        const int next_kk_block = kk_block + kFp8ChunkSubtiles;
        if (next_kk_block < kQweightTileKChunks) {
          stage_fp8_qweight_32_chunk(
              w_fp8_q, sh_q_tile.fp8_q_chunk[pipe ^ 1], tile.tile_base,
              tile.valid_channels, fp8_row_stride, k0, next_kk_block);
          cp_async_fence();
        }

        if (active) {
          acc += run_fp8_qweight_chunk_half2(
              sh_q_tile.fp8_q_chunk[pipe], fp8_scale_h2, x_tile, n_in_tile,
              local_n_tile, words_per_k_chunk, kk_block * 16);
        }

        if (next_kk_block < kQweightTileKChunks) {
          cp_async_wait<0>();
        }
        __syncthreads();
      }
    } else {
      __syncthreads();
    }
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

void fused_mixed_gemv_marlin_weights_splitk_narrow_fp8(
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
    int parallel_k);

void fused_mixed_gemv_marlin_weights_splitk_wide_fp8(
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
    int parallel_k);

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
  if (should_use_wide_fp8_splitk_variant(n_fp4, n_fp8, k)) {
    fused_mixed_gemv_marlin_weights_splitk_wide_fp8(
        x, w_fp4_q, w_fp4_scales, w_fp4_global_scale, w_fp8_q, w_fp8_scales,
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets, inv_perm, workspace,
        tile_counters, y, n_fp4, n_fp8, k, parallel_k);
  } else {
    fused_mixed_gemv_marlin_weights_splitk_narrow_fp8(
        x, w_fp4_q, w_fp4_scales, w_fp4_global_scale, w_fp8_q, w_fp8_scales,
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets, inv_perm, workspace,
        tile_counters, y, n_fp4, n_fp8, k, parallel_k);
  }
}

void fused_mixed_gemv_marlin_weights_splitk_narrow_fp8(
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

void fused_mixed_gemv_marlin_weights_splitk_wide_fp8(
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

  fused_mixed_gemv_marlin_qweight_splitk_wide_fp8_kernel<<<grid, block>>>(
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

void fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
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

  fused_mixed_gemv_marlin_qweight_splitk_staged_nvfp4_kernel<<<grid, block>>>(
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

void fused_mixed_gemv_marlin_weights_splitk_auto(
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
  if (should_use_staged_nvfp4_splitk_variant(n_fp4, n_fp8, k)) {
    fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
        x, w_fp4_q, w_fp4_scales, w_fp4_global_scale, w_fp8_q, w_fp8_scales,
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets, inv_perm, workspace,
        tile_counters, y, n_fp4, n_fp8, k, parallel_k);
  } else {
    fused_mixed_gemv_marlin_weights_splitk(
        x, w_fp4_q, w_fp4_scales, w_fp4_global_scale, w_fp8_q, w_fp8_scales,
        fp4_word_offsets, fp4_slot_map, fp8_word_offsets, inv_perm, workspace,
        tile_counters, y, n_fp4, n_fp8, k, parallel_k);
  }
}
