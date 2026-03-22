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

// Fused mixed-precision GEMV for M=1 decode (NVFP4 + FP8 in one launch)
void fused_mixed_gemv(
    const void *x,               // [1, K] FP16 activation
    const void *w_fp4,           // [N_fp4, K/2] packed FP4 E2M1 nibbles
    const void *w_fp4_scales,    // [N_fp4, K/16] FP8 E4M3 block scales
    float w_fp4_global_scale,    // scalar global scale for FP4 weights
    const void *w_fp8,           // [N_fp8, K] FP8 E4M3 weights
    const void *w_fp8_scales,    // [N_fp8] per-channel FP32 scales
    const void *inv_perm,        // [N_fp4+N_fp8] output permutation (int32)
    void *y,                     // [1, N_fp4+N_fp8] FP16 output
    int N_fp4, int N_fp8, int K
);

// Fused mixed-precision GEMV using Marlin-repacked qweights and plain scales.
// This is a transition kernel used to validate mixed scheduling and Marlin
// qweight address formulas before switching to Marlin's full inner loop.
void fused_mixed_gemv_marlin_weights(
    const void *x,               // [1, K] FP16 activation
    const void *w_fp4_q,         // [K/16, N_fp4*2] int32 Marlin qweight
    const void *w_fp4_scales,    // [N_fp4, K/16] FP8 E4M3 block scales
    float w_fp4_global_scale,    // scalar global scale for FP4 weights
    const void *w_fp8_q,         // [K/16, N_fp8*4] int32 Marlin qweight
    const void *w_fp8_scales,    // [N_fp8] FP32 channel scales
    const void *fp4_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *fp4_slot_map,    // [64,4,4] int32 slot ids for {k,k+1,k+8,k+9}
    const void *fp8_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *inv_perm,        // [N_fp4+N_fp8] output permutation (int32)
    void *y,                     // [1, N_fp4+N_fp8] FP16 output
    int N_fp4, int N_fp8, int K
);

// Same transition kernel, but with split-K style parallel scheduling across K
// tiles and an in-kernel reduction into a reusable workspace.
void fused_mixed_gemv_marlin_weights_splitk(
    const void *x,               // [1, K] FP16 activation
    const void *w_fp4_q,         // [K/16, N_fp4*2] int32 Marlin qweight
    const void *w_fp4_scales,    // [N_fp4, K/16] FP8 E4M3 block scales
    float w_fp4_global_scale,    // scalar global scale for FP4 weights
    const void *w_fp8_q,         // [K/16, N_fp8*4] int32 Marlin qweight
    const void *w_fp8_scales,    // [N_fp8] FP32 channel scales
    const void *fp4_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *fp4_slot_map,    // [64,4,4] int32 slot ids for {k,k+1,k+8,k+9}
    const void *fp8_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *inv_perm,        // [N_fp4+N_fp8] output permutation (int32)
    void *workspace,             // [N_fp4+N_fp8] FP32 accumulation buffer
    void *tile_counters,         // [num_tiles] int32 completion counters
    void *y,                     // [1, N_fp4+N_fp8] FP16 output
    int N_fp4, int N_fp8, int K, int parallel_k
);

// Explicit narrow-FP8 split-K variant (16-K FP8 chunks).
void fused_mixed_gemv_marlin_weights_splitk_narrow_fp8(
    const void *x,
    const void *w_fp4_q,
    const void *w_fp4_scales,
    float w_fp4_global_scale,
    const void *w_fp8_q,
    const void *w_fp8_scales,
    const void *fp4_word_offsets,
    const void *fp4_slot_map,
    const void *fp8_word_offsets,
    const void *inv_perm,
    void *workspace,
    void *tile_counters,
    void *y,
    int N_fp4, int N_fp8, int K, int parallel_k
);

// Explicit wide-FP8 split-K variant (32-K FP8 chunks).
void fused_mixed_gemv_marlin_weights_splitk_wide_fp8(
    const void *x,
    const void *w_fp4_q,
    const void *w_fp4_scales,
    float w_fp4_global_scale,
    const void *w_fp8_q,
    const void *w_fp8_scales,
    const void *fp4_word_offsets,
    const void *fp4_slot_map,
    const void *fp8_word_offsets,
    const void *inv_perm,
    void *workspace,
    void *tile_counters,
    void *y,
    int N_fp4, int N_fp8, int K, int parallel_k
);

// Experimental split-K mixed GEMV lane where NVFP4 tiles use a local
// Marlin-style tile engine with processed Marlin scales/global scale.
void fused_mixed_gemv_marlin_weights_splitk_nvfp4_marlin(
    const void *x,
    const void *w_fp4_q,
    const void *w_fp4_scales_marlin,
    const void *w_fp4_global_scale_marlin,
    const void *w_fp8_q,
    const void *w_fp8_scales,
    const void *fp4_word_offsets,
    const void *fp4_slot_map,
    const void *fp8_word_offsets,
    const void *inv_perm,
    void *workspace,
    void *tile_counters,
    void *y,
    int N_fp4, int N_fp8, int K, int parallel_k
);

// Alternate split-K transition kernel where the NVFP4 path also stages qweight
// and block scales through shared memory before the mixed inner loop.
void fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
    const void *x,               // [1, K] FP16 activation
    const void *w_fp4_q,         // [K/16, N_fp4*2] int32 Marlin qweight
    const void *w_fp4_scales,    // [N_fp4, K/16] FP8 E4M3 block scales
    float w_fp4_global_scale,    // scalar global scale for FP4 weights
    const void *w_fp8_q,         // [K/16, N_fp8*4] int32 Marlin qweight
    const void *w_fp8_scales,    // [N_fp8] FP32 channel scales
    const void *fp4_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *fp4_slot_map,    // [64,4,4] int32 slot ids for {k,k+1,k+8,k+9}
    const void *fp8_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *inv_perm,        // [N_fp4+N_fp8] output permutation (int32)
    void *workspace,             // [N_fp4+N_fp8] FP32 accumulation buffer
    void *tile_counters,         // [num_tiles] int32 completion counters
    void *y,                     // [1, N_fp4+N_fp8] FP16 output
    int N_fp4, int N_fp8, int K, int parallel_k
);

// Heuristic wrapper that selects between the two split-K mixed GEMV variants
// based on the current Qwen3-4B decode benchmark results.
void fused_mixed_gemv_marlin_weights_splitk_auto(
    const void *x,               // [1, K] FP16 activation
    const void *w_fp4_q,         // [K/16, N_fp4*2] int32 Marlin qweight
    const void *w_fp4_scales,    // [N_fp4, K/16] FP8 E4M3 block scales
    float w_fp4_global_scale,    // scalar global scale for FP4 weights
    const void *w_fp8_q,         // [K/16, N_fp8*4] int32 Marlin qweight
    const void *w_fp8_scales,    // [N_fp8] FP32 channel scales
    const void *fp4_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *fp4_slot_map,    // [64,4,4] int32 slot ids for {k,k+1,k+8,k+9}
    const void *fp8_word_offsets,// [64,4] int32 packed-word offsets inside one 16x64 tile
    const void *inv_perm,        // [N_fp4+N_fp8] output permutation (int32)
    void *workspace,             // [N_fp4+N_fp8] FP32 accumulation buffer
    void *tile_counters,         // [num_tiles] int32 completion counters
    void *y,                     // [1, N_fp4+N_fp8] FP16 output
    int N_fp4, int N_fp8, int K, int parallel_k
);
