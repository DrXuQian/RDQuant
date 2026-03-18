#pragma once

// Forward declarations for rdquant CUDA operations.
// This header does NOT include CUTLASS — safe to include from .cpp files.

#include <cstdint>

// ============================================================================
// GEMM functions (implemented in .cu files)
// All pointer types are void* to avoid CUTLASS type dependencies.
// ============================================================================

// MXFP8 activation x MXFP4 weight -> BF16 output
void mx_gemm_w4a8(
    const void *A,       // MXFP8 activation data [M, K]
    const void *SFA,     // activation scales, CUTLASS layout
    const void *B,       // MXFP4 weight data
    const void *SFB,     // weight scales, CUTLASS layout
    int M, int N, int K,
    void *D              // BF16 output [M, N]
);

// MXFP8 activation x MXFP6 weight -> BF16 output
void mx_gemm_w6a8(
    const void *A,
    const void *SFA,
    const void *B,
    const void *SFB,
    int M, int N, int K,
    void *D
);

// MXFP8 activation x MXFP8 weight -> BF16 output
void mx_gemm_w8a8(
    const void *A,
    const void *SFA,
    const void *B,
    const void *SFB,
    int M, int N, int K,
    void *D
);

// Scale factor reorder: row-major [dim0, K/32] -> CUTLASS interleaved
void reorder_sf_for_cutlass(
    const void *src,
    void *dst,
    int dim0,
    int K
);

// Online MXFP8 activation quantization: BF16 [M,K] -> FP8 [M,K] + scales [M,K/32]
void quantize_act_mxfp8(
    const void *input,
    void *output,
    void *scales,
    int M, int K
);

// cuBLASLt MX block-scaled GEMMs (row-major scales, auto kernel selection)
// Scales are in row-major [dim0, K/32] UE8M0 — NOT CUTLASS interleaved layout

void cublas_mxfp8_gemm(   // MXFP8 act × MXFP8 weight
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C);

void cublas_mx_gemm_w4a8( // MXFP8 act × MXFP4 weight
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C);

void cublas_mx_gemm_w6a8( // MXFP8 act × MXFP6 weight
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C);
