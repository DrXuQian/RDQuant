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

}  // namespace rdquant_marlin_nvfp4
