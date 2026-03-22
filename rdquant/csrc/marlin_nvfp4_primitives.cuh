#pragma once

// Minimal subset of Marlin NVFP4 helpers used to build a fused NVFP4 tile
// engine inside RDQuant without importing the full Marlin kernel scheduler.

#define MARLIN_NAMESPACE_NAME rdq_marlin
#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "dequant.h"
#include "marlin_mma.h"
#undef MARLIN_NAMESPACE_NAME

namespace rdquant_marlin_nvfp4 {

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
  static constexpr int kTbNWarps = kThreadNBlocks / 4;
  static constexpr int kBShWrIters = 2;
  static constexpr int kAShStride = 16;
  static constexpr int kAShStage = 256;
  static constexpr int kBShStride = 64;
  static constexpr int kBShStage = 512;
  static constexpr int kSShStride = 8;
  static constexpr int kSShStage = 64;

  __device__ static inline int transform_a(int i) {
    int row = i / kAShStride;
    return kAShStride * row + ((i % kAShStride) ^ (row % 8));
  }

  __device__ static inline int a_sh_rd(int lane) {
    int warp = lane / 32;
    int lane_in_warp = lane % 32;
    return kAShStride * (lane_in_warp % 16) + lane_in_warp / 16 +
           2 * (warp / kTbNWarps) * kBShWrIters;
  }

  __device__ static inline int a_sh_rd_trans(int lane, int k_step) {
    return transform_a(2 * (k_step % kBShWrIters) + a_sh_rd(lane));
  }

  __device__ static inline int b_sh_rd(int lane) {
    int rd = lane;
    return rd + (rd / kBShStride) * kBShStride;
  }

  __device__ static inline int scale_group(int lane, int k_step) {
    int warp_row = (lane / 32) / kTbNWarps;
    return kBShWrIters * warp_row + (k_step % kBShWrIters);
  }

  __device__ static inline int s_sh_rd(int lane) {
    return kSShStride * ((lane / 32) % kTbNWarps) + ((lane % 32) / 4);
  }

  __device__ static inline int s_sh_rd_pair(int lane, int k_step) {
    return s_sh_rd(lane) + scale_group(lane, k_step) * (2 * kSShStride);
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

__device__ inline void dequant_nvfp4(int q, half2* frag_b) {
  rdq_marlin::dequant<half2, vllm::kFE2M1f.id(), true>(q, frag_b);
}

__device__ inline void dequant_nvfp4_scales(int q, half2* frag_s) {
  rdq_marlin::dequant_fp8_scales<half2, vllm::kFE4M3fn.id()>(q, frag_s);
}

__device__ inline void scale_frag(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

__device__ inline void mma_fp16(
    const FragA& frag_a, const FragB& frag_b, FragC& frag_c) {
  rdq_marlin::mma<vllm::kFloat16.id(), false>(frag_a, frag_b, frag_c);
}

__device__ inline void zero_frag_c(FragC& frag_c) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    frag_c[i] = 0.0f;
  }
}

__device__ inline void load_frag_a_tile128(
    const int4* sh_a, int lane, int k_step, FragA& frag_a) {
  ldsm_a<4>(frag_a, &sh_a[Tile128x128x128::a_sh_rd_trans(lane, k_step)]);
}

__device__ inline int load_qweight_int_tile128(
    const int4* sh_b, int lane, int k_step, int j) {
  const int* sh_b_int = reinterpret_cast<const int*>(sh_b);
  return sh_b_int[Tile128x128x128::kBShStride * (k_step % Tile128x128x128::kBShWrIters) +
                  Tile128x128x128::b_sh_rd(lane) + j];
}

__device__ inline void load_scale_frags_tile128(
    const int4* sh_s, int lane, int k_step, FragS* frag_s) {
  const int2 raw_pair =
      reinterpret_cast<const int2*>(sh_s)[Tile128x128x128::s_sh_rd_pair(lane, k_step)];
  reinterpret_cast<int2*>(frag_s)[0] = raw_pair;

  const int s_quant_0 = reinterpret_cast<int*>(frag_s)[0];
  const int s_quant_1 = reinterpret_cast<int*>(frag_s)[1];
  dequant_nvfp4_scales(s_quant_0, reinterpret_cast<half2*>(frag_s));
  dequant_nvfp4_scales(s_quant_1, reinterpret_cast<half2*>(frag_s) + 2);
}

}  // namespace rdquant_marlin_nvfp4
