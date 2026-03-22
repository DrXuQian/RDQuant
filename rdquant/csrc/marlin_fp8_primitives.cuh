#pragma once

// Fixed FP8 Marlin helpers used to build an experimental FP8-dominant fused
// lane inside RDQuant. Reuse the vendored rdq_marlin namespace instantiated by
// marlin_nvfp4_primitives.cuh so we can share the same minimal Marlin subset.

#include "marlin_nvfp4_primitives.cuh"

namespace rdquant_marlin_fp8 {

using FragA = rdq_marlin::MarlinScalarType<vllm::kFloat16.id()>::FragA;
using FragB = rdq_marlin::MarlinScalarType<vllm::kFloat16.id()>::FragB;
using FragC = rdq_marlin::MarlinScalarType<vllm::kFloat16.id()>::FragC;
using FragS = rdq_marlin::MarlinScalarType<vllm::kFloat16.id()>::FragS;

struct Tile128x128x128 {
  static constexpr int kThreads = 256;
  static constexpr int kThreadMBlocks = 1;
  static constexpr int kThreadNBlocks = 8;
  static constexpr int kThreadKBlocks = 8;
  static constexpr int kStages = 4;
  static constexpr bool kMBlockSize8 = true;
  static constexpr int kMBlockSize = 8;
  static constexpr int kTbNWarps = kThreadNBlocks / 4;
  static constexpr int kAShStride = 16;
  static constexpr int kAShStage = kAShStride * kMBlockSize;
  static constexpr int kBThreadVecs = 2;
  static constexpr int kBShStride = 128;
  static constexpr int kBShStage = kBShStride * kThreadKBlocks;
  static constexpr int kBShWrIters = 2;
  static constexpr int kSShStride = 16;
  static constexpr int kSShStage = kSShStride;

  __device__ static inline int transform_a(int i) {
    int row = i / kAShStride;
    return kAShStride * row + ((i % kAShStride) ^ (row % 8));
  }

  __device__ static inline int a_sh_rd(int lane) {
    int warp = lane / 32;
    int lane_in_warp = lane % 32;
    return kAShStride * (lane_in_warp % 8) + lane_in_warp / 8 +
           2 * (warp / kTbNWarps) * kBShWrIters;
  }

  __device__ static inline int a_sh_rd_trans(int lane, int k_step) {
    return transform_a(2 * (k_step % kBShWrIters) + a_sh_rd(lane));
  }

  __device__ static inline int b_sh_rd(int lane) {
    int rd = lane * kBThreadVecs;
    return rd + rd / kBShStride * (kBShStride * (kBShWrIters - 1));
  }

  __device__ static inline int s_sh_rd(int lane) {
    return 8 * ((lane / 32) % kTbNWarps) + (lane % 32) / 8;
  }
};

template <int count>
__device__ inline void ldsm_a(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  }
}

__device__ inline void dequant_fp8(int q, half2* frag_b) {
  rdq_marlin::dequant<half2, vllm::kFE4M3fn.id(), true>(q, frag_b);
}

__device__ inline void mma_fp16_trans(
    const FragA& frag_a,
    const FragB& frag_b0,
    const FragB& frag_b1,
    FragC& frag_c) {
  rdq_marlin::mma_trans<vllm::kFloat16.id(), false>(
      frag_a, frag_b0, frag_b1, frag_c);
}

__device__ inline void zero_frag_c(FragC& frag_c) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    frag_c[i] = 0.0f;
  }
}

__device__ inline void load_frag_a_tile128(
    const int4* sh_a, int lane, int k_step, FragA& frag_a) {
  ldsm_a<2>(frag_a, &sh_a[Tile128x128x128::a_sh_rd_trans(lane, k_step)]);
}

__device__ inline void load_qweight_pair_tile128(
    const int4* sh_b,
    int lane,
    int k_step,
    int j,
    int* b_quant_0,
    int* b_quant_1) {
  const int base =
      Tile128x128x128::kBShStride *
          (k_step % Tile128x128x128::kBShWrIters) +
      Tile128x128x128::b_sh_rd(lane);
  const int4 raw0 = sh_b[base + 0];
  const int4 raw1 = sh_b[base + 1];
  const int* ints0 = reinterpret_cast<const int*>(&raw0);
  const int* ints1 = reinterpret_cast<const int*>(&raw1);
  if (j < 2) {
    *b_quant_0 = ints0[j * 2 + 0];
    *b_quant_1 = ints0[j * 2 + 1];
  } else {
    *b_quant_0 = ints1[(j - 2) * 2 + 0];
    *b_quant_1 = ints1[(j - 2) * 2 + 1];
  }
}

__device__ inline void load_channel_scales_tile128(
    const int4* sh_s, int lane, FragS frag_s[2][4]) {
  const int s_sh_rd = Tile128x128x128::s_sh_rd(lane);
  reinterpret_cast<int4*>(&frag_s[0][0])[0] = sh_s[s_sh_rd + 0];
  reinterpret_cast<int4*>(&frag_s[0][0])[1] = sh_s[s_sh_rd + 4];
  const int idx = (lane / 4) % 2;
  half2* frag_s_half2 = reinterpret_cast<half2*>(&frag_s[0][0]);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    frag_s_half2[i] = __half2half2(reinterpret_cast<half*>(&frag_s_half2[i])[idx]);
  }
}

template <int offset>
__device__ inline void scale_float_pair(float* c, const FragS& s) {
  const half2 scale = reinterpret_cast<const half2*>(&s)[0];
  const half* scale_half = reinterpret_cast<const half*>(&scale);
  const float s0 = __half2float(scale_half[offset + 0]);
  const float s1 = __half2float(scale_half[offset + 1]);
  c[0] *= s0;
  c[1] *= s1;
}

__device__ inline void scale_frag_c_channelwise(
    FragC frag_c[4][2], FragS frag_s[2][4]) {
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float* c = reinterpret_cast<float*>(&frag_c[j][0]);
    scale_float_pair<0>(c + 0, frag_s[j / 2][2 * (j % 2) + 0]);
    scale_float_pair<0>(c + 2, frag_s[j / 2][2 * (j % 2) + 1]);
  }
}

}  // namespace rdquant_marlin_fp8
