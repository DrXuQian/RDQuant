#include <torch/extension.h>
#include "rdquant_ops.h"

// ============================================================================
// Torch wrapper: MXFP8 x MXFP4 GEMM
// ============================================================================
torch::Tensor mx_gemm_w4a8_torch(
    const torch::Tensor &x_fp8,    // [M, K] uint8
    const torch::Tensor &x_sf,     // CUTLASS-layout activation scales, uint8
    const torch::Tensor &w_fp4,    // packed FP4 weight data, uint8
    const torch::Tensor &w_sf,     // CUTLASS-layout weight scales, uint8
    int M, int N, int K
)
{
    auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_fp8.device()));

    mx_gemm_w4a8(
        x_fp8.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        w_fp4.data_ptr<uint8_t>(),
        w_sf.data_ptr<uint8_t>(),
        M, N, K,
        D.data_ptr<at::BFloat16>()
    );

    return D;
}

// ============================================================================
// Torch wrapper: MXFP8 x MXFP6 GEMM
// ============================================================================
torch::Tensor mx_gemm_w6a8_torch(
    const torch::Tensor &x_fp8,
    const torch::Tensor &x_sf,
    const torch::Tensor &w_fp6,
    const torch::Tensor &w_sf,
    int M, int N, int K
)
{
    auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_fp8.device()));

    mx_gemm_w6a8(
        x_fp8.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        w_fp6.data_ptr<uint8_t>(),
        w_sf.data_ptr<uint8_t>(),
        M, N, K,
        D.data_ptr<at::BFloat16>()
    );

    return D;
}

// ============================================================================
// Torch wrapper: MXFP8 x MXFP8 GEMM
// ============================================================================
torch::Tensor mx_gemm_w8a8_torch(
    const torch::Tensor &x_fp8,
    const torch::Tensor &x_sf,
    const torch::Tensor &w_fp8,
    const torch::Tensor &w_sf,
    int M, int N, int K
)
{
    auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_fp8.device()));

    mx_gemm_w8a8(
        x_fp8.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        w_fp8.data_ptr<uint8_t>(),
        w_sf.data_ptr<uint8_t>(),
        M, N, K,
        D.data_ptr<at::BFloat16>()
    );

    return D;
}

// ============================================================================
// Torch wrapper: Scale factor reorder (row-major -> CUTLASS interleaved)
// ============================================================================
torch::Tensor reorder_sf_torch(
    const torch::Tensor &sf_rowmajor,   // [dim0, K/32] uint8
    int dim0,
    int K
)
{
    int K_tiles = K / 32;
    // Compute output size: padded to tile boundaries
    int num_row_tiles = (dim0 + 127) / 128;
    int num_k_blocks = (K_tiles + 3) / 4;
    int output_size = num_row_tiles * num_k_blocks * 512;

    auto sf_reordered = torch::zeros({output_size},
        torch::dtype(torch::kUInt8).device(sf_rowmajor.device()));

    reorder_sf_for_cutlass(
        sf_rowmajor.data_ptr<uint8_t>(),
        sf_reordered.data_ptr<uint8_t>(),
        dim0,
        K
    );

    return sf_reordered;
}

// ============================================================================
// Torch wrapper: Online MXFP8 activation quantization
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> quantize_act_mxfp8_torch(
    const torch::Tensor &x_bf16   // [M, K] BFloat16
)
{
    int M = x_bf16.size(0);
    int K = x_bf16.size(1);

    auto x_fp8 = torch::empty({M, K},
        torch::dtype(torch::kUInt8).device(x_bf16.device()));
    auto x_sf = torch::empty({M, K / 32},
        torch::dtype(torch::kUInt8).device(x_bf16.device()));

    quantize_act_mxfp8(
        x_bf16.data_ptr<at::BFloat16>(),
        x_fp8.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        M, K
    );

    return std::make_tuple(x_fp8, x_sf);
}

// ============================================================================
// Torch wrappers: cuBLASLt MX block-scaled GEMMs
// ============================================================================
#define DEFINE_CUBLAS_MX_WRAPPER(name, c_func)                                \
torch::Tensor name(                                                           \
    const torch::Tensor &x_fp8, const torch::Tensor &x_sf,                   \
    const torch::Tensor &w, const torch::Tensor &w_sf,                        \
    int M, int N, int K)                                                      \
{                                                                             \
    auto D = torch::empty({M, N},                                             \
        torch::dtype(torch::kBFloat16).device(x_fp8.device()));               \
    c_func(x_fp8.data_ptr<uint8_t>(), x_sf.data_ptr<uint8_t>(),              \
           w.data_ptr<uint8_t>(), w_sf.data_ptr<uint8_t>(),                   \
           M, N, K, D.data_ptr<at::BFloat16>());                              \
    return D;                                                                 \
}

DEFINE_CUBLAS_MX_WRAPPER(cublas_mxfp8_gemm_torch,  cublas_mxfp8_gemm)
DEFINE_CUBLAS_MX_WRAPPER(cublas_mx_w4a8_torch,     cublas_mx_gemm_w4a8)
DEFINE_CUBLAS_MX_WRAPPER(cublas_mx_w6a8_torch,     cublas_mx_gemm_w6a8)

// ============================================================================
// Torch wrapper: Fused mixed-precision GEMV (NVFP4 + FP8, single launch)
// ============================================================================
torch::Tensor fused_mixed_gemv_torch(
    const torch::Tensor &x,              // [1, K] float16
    const torch::Tensor &w_fp4,          // [N_fp4, K/2] uint8 packed nibbles
    const torch::Tensor &w_fp4_scales,   // [N_fp4, K/16] uint8 FP8 E4M3
    double w_fp4_global_scale,            // scalar
    const torch::Tensor &w_fp8,          // [N_fp8, K] uint8 FP8 E4M3
    const torch::Tensor &w_fp8_scales,   // [N_fp8] float32 per-channel
    const torch::Tensor &inv_perm,       // [N_total] int32
    int N_fp4, int N_fp8, int K
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_fp4.is_cuda() && w_fp4_scales.is_cuda() && w_fp8.is_cuda() &&
                    w_fp8_scales.is_cuda() && inv_perm.is_cuda(),
                "all fused_mixed_gemv inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "fused_mixed_gemv currently only supports M=1");
    TORCH_CHECK(x.size(1) == K, "x.size(1) must match K");
    TORCH_CHECK(K % 128 == 0, "fused_mixed_gemv v1 requires K to be a multiple of 128");
    TORCH_CHECK(w_fp4.size(0) == N_fp4 && w_fp8.size(0) == N_fp8,
                "weight rows must match N_fp4/N_fp8");
    TORCH_CHECK(w_fp4.size(1) * 2 == K, "w_fp4 must have shape [N_fp4, K/2]");
    TORCH_CHECK(w_fp4_scales.size(0) == N_fp4 && w_fp4_scales.size(1) * 16 == K,
                "w_fp4_scales must have shape [N_fp4, K/16]");
    TORCH_CHECK(w_fp8.size(1) == K, "w_fp8 must have shape [N_fp8, K]");
    TORCH_CHECK(w_fp8_scales.numel() == N_fp8, "w_fp8_scales must have N_fp8 elements");
    TORCH_CHECK(inv_perm.numel() == N_fp4 + N_fp8, "inv_perm must have N_fp4 + N_fp8 elements");

    int N_total = N_fp4 + N_fp8;
    auto y = torch::empty({1, N_total},
        torch::dtype(torch::kFloat16).device(x.device()));

    fused_mixed_gemv(
        x.data_ptr(),
        w_fp4.data_ptr<uint8_t>(),
        w_fp4_scales.data_ptr<uint8_t>(),
        static_cast<float>(w_fp4_global_scale),
        w_fp8.data_ptr<uint8_t>(),
        w_fp8_scales.data_ptr<float>(),
        inv_perm.data_ptr<int32_t>(),
        y.data_ptr(),
        N_fp4, N_fp8, K
    );

    return y;
}

torch::Tensor fused_mixed_gemv_marlin_weights_torch(
    const torch::Tensor &x,              // [1, K] float16
    const torch::Tensor &w_fp4_q,        // [K/16, N_fp4*2] int32 Marlin qweight
    const torch::Tensor &w_fp4_scales,   // [N_fp4, K/16] uint8 FP8 E4M3
    double w_fp4_global_scale,           // scalar
    const torch::Tensor &w_fp8_q,        // [K/16, N_fp8*4] int32 Marlin qweight
    const torch::Tensor &w_fp8_scales,   // [N_fp8] float32 per-channel
    const torch::Tensor &fp4_word_offsets, // [64, 4] int32
    const torch::Tensor &fp4_slot_map,     // [64, 4, 4] int32
    const torch::Tensor &fp8_word_offsets, // [64, 4] int32
    const torch::Tensor &inv_perm,       // [N_total] int32
    int N_fp4, int N_fp8, int K
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_fp4_q.is_cuda() && w_fp4_scales.is_cuda() && w_fp8_q.is_cuda() &&
                    w_fp8_scales.is_cuda() && fp4_word_offsets.is_cuda() &&
                    fp4_slot_map.is_cuda() && fp8_word_offsets.is_cuda() &&
                    inv_perm.is_cuda(),
                "all fused_mixed_gemv_marlin_weights inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused_mixed_gemv_marlin_weights currently only supports M=1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N_fp4 % 64 == 0 && N_fp8 % 64 == 0,
                "N_fp4 and N_fp8 must be multiples of 64 for the Marlin-packed prototype");
    TORCH_CHECK(w_fp4_q.dtype() == torch::kInt32 && w_fp8_q.dtype() == torch::kInt32,
                "Marlin qweights must be int32");
    TORCH_CHECK(fp4_word_offsets.dtype() == torch::kInt32 &&
                    fp4_slot_map.dtype() == torch::kInt32 &&
                    fp8_word_offsets.dtype() == torch::kInt32,
                "compact Marlin maps must be int32");
    TORCH_CHECK(w_fp4_q.size(0) * 16 == K && w_fp4_q.size(1) == N_fp4 * 2,
                "w_fp4_q must have shape [K/16, N_fp4*2]");
    TORCH_CHECK(w_fp4_scales.size(0) == N_fp4 && w_fp4_scales.size(1) * 16 == K,
                "w_fp4_scales must have shape [N_fp4, K/16]");
    TORCH_CHECK(w_fp8_q.size(0) * 16 == K && w_fp8_q.size(1) == N_fp8 * 4,
                "w_fp8_q must have shape [K/16, N_fp8*4]");
    TORCH_CHECK(w_fp8_scales.numel() == N_fp8, "w_fp8_scales must have N_fp8 elements");
    TORCH_CHECK(fp4_word_offsets.dim() == 2 &&
                    fp4_word_offsets.size(0) == 64 &&
                    fp4_word_offsets.size(1) == 4,
                "fp4_word_offsets must have shape [64, 4]");
    TORCH_CHECK(fp4_slot_map.dim() == 3 &&
                    fp4_slot_map.size(0) == 64 &&
                    fp4_slot_map.size(1) == 4 &&
                    fp4_slot_map.size(2) == 4,
                "fp4_slot_map must have shape [64, 4, 4]");
    TORCH_CHECK(fp8_word_offsets.dim() == 2 &&
                    fp8_word_offsets.size(0) == 64 &&
                    fp8_word_offsets.size(1) == 4,
                "fp8_word_offsets must have shape [64, 4]");
    TORCH_CHECK(inv_perm.numel() == N_fp4 + N_fp8, "inv_perm must have N_fp4 + N_fp8 elements");

    int N_total = N_fp4 + N_fp8;
    auto y = torch::empty({1, N_total},
        torch::dtype(torch::kFloat16).device(x.device()));

    fused_mixed_gemv_marlin_weights(
        x.data_ptr(),
        w_fp4_q.data_ptr<int32_t>(),
        w_fp4_scales.data_ptr<uint8_t>(),
        static_cast<float>(w_fp4_global_scale),
        w_fp8_q.data_ptr<int32_t>(),
        w_fp8_scales.data_ptr<float>(),
        fp4_word_offsets.data_ptr<int32_t>(),
        fp4_slot_map.data_ptr<int32_t>(),
        fp8_word_offsets.data_ptr<int32_t>(),
        inv_perm.data_ptr<int32_t>(),
        y.data_ptr(),
        N_fp4, N_fp8, K
    );

    return y;
}

torch::Tensor fused_mixed_gemv_marlin_weights_splitk_torch(
    const torch::Tensor &x,              // [1, K] float16
    const torch::Tensor &w_fp4_q,        // [K/16, N_fp4*2] int32 Marlin qweight
    const torch::Tensor &w_fp4_scales,   // [N_fp4, K/16] uint8 FP8 E4M3
    double w_fp4_global_scale,           // scalar
    const torch::Tensor &w_fp8_q,        // [K/16, N_fp8*4] int32 Marlin qweight
    const torch::Tensor &w_fp8_scales,   // [N_fp8] float32 per-channel
    const torch::Tensor &fp4_word_offsets, // [64, 4] int32
    const torch::Tensor &fp4_slot_map,     // [64, 4, 4] int32
    const torch::Tensor &fp8_word_offsets, // [64, 4] int32
    const torch::Tensor &inv_perm,       // [N_total] int32
    const torch::Tensor &workspace,      // [N_total] float32
    const torch::Tensor &tile_counters,  // [num_tiles] int32
    int N_fp4, int N_fp8, int K, int parallel_k
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_fp4_q.is_cuda() && w_fp4_scales.is_cuda() && w_fp8_q.is_cuda() &&
                    w_fp8_scales.is_cuda() && fp4_word_offsets.is_cuda() &&
                    fp4_slot_map.is_cuda() && fp8_word_offsets.is_cuda() &&
                    inv_perm.is_cuda(),
                "all fused_mixed_gemv_marlin_weights_splitk inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused_mixed_gemv_marlin_weights_splitk currently only supports M=1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N_fp4 % 64 == 0 && N_fp8 % 64 == 0,
                "N_fp4 and N_fp8 must be multiples of 64 for the Marlin-packed prototype");
    TORCH_CHECK(w_fp4_q.dtype() == torch::kInt32 && w_fp8_q.dtype() == torch::kInt32,
                "Marlin qweights must be int32");
    TORCH_CHECK(fp4_word_offsets.dtype() == torch::kInt32 &&
                    fp4_slot_map.dtype() == torch::kInt32 &&
                    fp8_word_offsets.dtype() == torch::kInt32,
                "compact Marlin maps must be int32");
    TORCH_CHECK(w_fp4_q.size(0) * 16 == K && w_fp4_q.size(1) == N_fp4 * 2,
                "w_fp4_q must have shape [K/16, N_fp4*2]");
    TORCH_CHECK(w_fp4_scales.size(0) == N_fp4 && w_fp4_scales.size(1) * 16 == K,
                "w_fp4_scales must have shape [N_fp4, K/16]");
    TORCH_CHECK(w_fp8_q.size(0) * 16 == K && w_fp8_q.size(1) == N_fp8 * 4,
                "w_fp8_q must have shape [K/16, N_fp8*4]");
    TORCH_CHECK(w_fp8_scales.numel() == N_fp8, "w_fp8_scales must have N_fp8 elements");
    TORCH_CHECK(fp4_word_offsets.dim() == 2 &&
                    fp4_word_offsets.size(0) == 64 &&
                    fp4_word_offsets.size(1) == 4,
                "fp4_word_offsets must have shape [64, 4]");
    TORCH_CHECK(fp4_slot_map.dim() == 3 &&
                    fp4_slot_map.size(0) == 64 &&
                    fp4_slot_map.size(1) == 4 &&
                    fp4_slot_map.size(2) == 4,
                "fp4_slot_map must have shape [64, 4, 4]");
    TORCH_CHECK(fp8_word_offsets.dim() == 2 &&
                    fp8_word_offsets.size(0) == 64 &&
                    fp8_word_offsets.size(1) == 4,
                "fp8_word_offsets must have shape [64, 4]");
    TORCH_CHECK(inv_perm.numel() == N_fp4 + N_fp8, "inv_perm must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(workspace.is_cuda() && tile_counters.is_cuda(),
                "workspace and tile_counters must be CUDA tensors");
    TORCH_CHECK(workspace.dtype() == torch::kFloat32, "workspace must be float32");
    TORCH_CHECK(tile_counters.dtype() == torch::kInt32, "tile_counters must be int32");
    TORCH_CHECK(parallel_k >= 1, "parallel_k must be >= 1");

    int num_fp4_tiles = (N_fp4 + 127) / 128;
    int num_fp8_tiles = (N_fp8 + 127) / 128;
    int num_tiles = num_fp4_tiles + num_fp8_tiles;
    int num_k_tiles = K / 128;
    TORCH_CHECK(parallel_k <= num_k_tiles,
                "parallel_k must be <= K/128 for the split-K prototype");
    TORCH_CHECK(workspace.numel() == N_fp4 + N_fp8,
                "workspace must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(tile_counters.numel() == num_tiles,
                "tile_counters must have one counter per output tile");

    auto y = torch::empty({1, N_fp4 + N_fp8},
        torch::dtype(torch::kFloat16).device(x.device()));

    fused_mixed_gemv_marlin_weights_splitk(
        x.data_ptr(),
        w_fp4_q.data_ptr<int32_t>(),
        w_fp4_scales.data_ptr<uint8_t>(),
        static_cast<float>(w_fp4_global_scale),
        w_fp8_q.data_ptr<int32_t>(),
        w_fp8_scales.data_ptr<float>(),
        fp4_word_offsets.data_ptr<int32_t>(),
        fp4_slot_map.data_ptr<int32_t>(),
        fp8_word_offsets.data_ptr<int32_t>(),
        inv_perm.data_ptr<int32_t>(),
        workspace.data_ptr<float>(),
        tile_counters.data_ptr<int32_t>(),
        y.data_ptr(),
        N_fp4, N_fp8, K, parallel_k
    );

    return y;
}

torch::Tensor fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4_torch(
    const torch::Tensor &x,              // [1, K] float16
    const torch::Tensor &w_fp4_q,        // [K/16, N_fp4*2] int32 Marlin qweight
    const torch::Tensor &w_fp4_scales,   // [N_fp4, K/16] uint8 FP8 E4M3
    double w_fp4_global_scale,           // scalar
    const torch::Tensor &w_fp8_q,        // [K/16, N_fp8*4] int32 Marlin qweight
    const torch::Tensor &w_fp8_scales,   // [N_fp8] float32 per-channel
    const torch::Tensor &fp4_word_offsets, // [64, 4] int32
    const torch::Tensor &fp4_slot_map,     // [64, 4, 4] int32
    const torch::Tensor &fp8_word_offsets, // [64, 4] int32
    const torch::Tensor &inv_perm,       // [N_total] int32
    const torch::Tensor &workspace,      // [N_total] float32
    const torch::Tensor &tile_counters,  // [num_tiles] int32
    int N_fp4, int N_fp8, int K, int parallel_k
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_fp4_q.is_cuda() && w_fp4_scales.is_cuda() && w_fp8_q.is_cuda() &&
                    w_fp8_scales.is_cuda() && fp4_word_offsets.is_cuda() &&
                    fp4_slot_map.is_cuda() && fp8_word_offsets.is_cuda() &&
                    inv_perm.is_cuda(),
                "all fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4 inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4 currently only supports M=1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N_fp4 % 64 == 0 && N_fp8 % 64 == 0,
                "N_fp4 and N_fp8 must be multiples of 64 for the Marlin-packed prototype");
    TORCH_CHECK(w_fp4_q.dtype() == torch::kInt32 && w_fp8_q.dtype() == torch::kInt32,
                "Marlin qweights must be int32");
    TORCH_CHECK(fp4_word_offsets.dtype() == torch::kInt32 &&
                    fp4_slot_map.dtype() == torch::kInt32 &&
                    fp8_word_offsets.dtype() == torch::kInt32,
                "compact Marlin maps must be int32");
    TORCH_CHECK(w_fp4_q.size(0) * 16 == K && w_fp4_q.size(1) == N_fp4 * 2,
                "w_fp4_q must have shape [K/16, N_fp4*2]");
    TORCH_CHECK(w_fp4_scales.size(0) == N_fp4 && w_fp4_scales.size(1) * 16 == K,
                "w_fp4_scales must have shape [N_fp4, K/16]");
    TORCH_CHECK(w_fp8_q.size(0) * 16 == K && w_fp8_q.size(1) == N_fp8 * 4,
                "w_fp8_q must have shape [K/16, N_fp8*4]");
    TORCH_CHECK(w_fp8_scales.numel() == N_fp8, "w_fp8_scales must have N_fp8 elements");
    TORCH_CHECK(fp4_word_offsets.dim() == 2 &&
                    fp4_word_offsets.size(0) == 64 &&
                    fp4_word_offsets.size(1) == 4,
                "fp4_word_offsets must have shape [64, 4]");
    TORCH_CHECK(fp4_slot_map.dim() == 3 &&
                    fp4_slot_map.size(0) == 64 &&
                    fp4_slot_map.size(1) == 4 &&
                    fp4_slot_map.size(2) == 4,
                "fp4_slot_map must have shape [64, 4, 4]");
    TORCH_CHECK(fp8_word_offsets.dim() == 2 &&
                    fp8_word_offsets.size(0) == 64 &&
                    fp8_word_offsets.size(1) == 4,
                "fp8_word_offsets must have shape [64, 4]");
    TORCH_CHECK(inv_perm.numel() == N_fp4 + N_fp8, "inv_perm must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(workspace.is_cuda() && tile_counters.is_cuda(),
                "workspace and tile_counters must be CUDA tensors");
    TORCH_CHECK(workspace.dtype() == torch::kFloat32, "workspace must be float32");
    TORCH_CHECK(tile_counters.dtype() == torch::kInt32, "tile_counters must be int32");
    TORCH_CHECK(parallel_k >= 1, "parallel_k must be >= 1");

    int num_fp4_tiles = (N_fp4 + 127) / 128;
    int num_fp8_tiles = (N_fp8 + 127) / 128;
    int num_tiles = num_fp4_tiles + num_fp8_tiles;
    int num_k_tiles = K / 128;
    TORCH_CHECK(parallel_k <= num_k_tiles,
                "parallel_k must be <= K/128 for the split-K prototype");
    TORCH_CHECK(workspace.numel() == N_fp4 + N_fp8,
                "workspace must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(tile_counters.numel() == num_tiles,
                "tile_counters must have one counter per output tile");

    auto y = torch::empty({1, N_fp4 + N_fp8},
        torch::dtype(torch::kFloat16).device(x.device()));

    fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4(
        x.data_ptr(),
        w_fp4_q.data_ptr<int32_t>(),
        w_fp4_scales.data_ptr<uint8_t>(),
        static_cast<float>(w_fp4_global_scale),
        w_fp8_q.data_ptr<int32_t>(),
        w_fp8_scales.data_ptr<float>(),
        fp4_word_offsets.data_ptr<int32_t>(),
        fp4_slot_map.data_ptr<int32_t>(),
        fp8_word_offsets.data_ptr<int32_t>(),
        inv_perm.data_ptr<int32_t>(),
        workspace.data_ptr<float>(),
        tile_counters.data_ptr<int32_t>(),
        y.data_ptr(),
        N_fp4, N_fp8, K, parallel_k
    );

    return y;
}

torch::Tensor fused_mixed_gemv_marlin_weights_splitk_auto_torch(
    const torch::Tensor &x,
    const torch::Tensor &w_fp4_q,
    const torch::Tensor &w_fp4_scales,
    double w_fp4_global_scale,
    const torch::Tensor &w_fp8_q,
    const torch::Tensor &w_fp8_scales,
    const torch::Tensor &fp4_word_offsets,
    const torch::Tensor &fp4_slot_map,
    const torch::Tensor &fp8_word_offsets,
    const torch::Tensor &inv_perm,
    const torch::Tensor &workspace,
    const torch::Tensor &tile_counters,
    int N_fp4, int N_fp8, int K, int parallel_k
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_fp4_q.is_cuda() && w_fp4_scales.is_cuda() && w_fp8_q.is_cuda() &&
                    w_fp8_scales.is_cuda() && fp4_word_offsets.is_cuda() &&
                    fp4_slot_map.is_cuda() && fp8_word_offsets.is_cuda() &&
                    inv_perm.is_cuda(),
                "all fused_mixed_gemv_marlin_weights_splitk_auto inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused_mixed_gemv_marlin_weights_splitk_auto currently only supports M=1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N_fp4 % 64 == 0 && N_fp8 % 64 == 0,
                "N_fp4 and N_fp8 must be multiples of 64 for the Marlin-packed prototype");
    TORCH_CHECK(w_fp4_q.dtype() == torch::kInt32 && w_fp8_q.dtype() == torch::kInt32,
                "Marlin qweights must be int32");
    TORCH_CHECK(fp4_word_offsets.dtype() == torch::kInt32 &&
                    fp4_slot_map.dtype() == torch::kInt32 &&
                    fp8_word_offsets.dtype() == torch::kInt32,
                "compact Marlin maps must be int32");
    TORCH_CHECK(w_fp4_q.size(0) * 16 == K && w_fp4_q.size(1) == N_fp4 * 2,
                "w_fp4_q must have shape [K/16, N_fp4*2]");
    TORCH_CHECK(w_fp4_scales.size(0) == N_fp4 && w_fp4_scales.size(1) * 16 == K,
                "w_fp4_scales must have shape [N_fp4, K/16]");
    TORCH_CHECK(w_fp8_q.size(0) * 16 == K && w_fp8_q.size(1) == N_fp8 * 4,
                "w_fp8_q must have shape [K/16, N_fp8*4]");
    TORCH_CHECK(w_fp8_scales.numel() == N_fp8, "w_fp8_scales must have N_fp8 elements");
    TORCH_CHECK(fp4_word_offsets.dim() == 2 &&
                    fp4_word_offsets.size(0) == 64 &&
                    fp4_word_offsets.size(1) == 4,
                "fp4_word_offsets must have shape [64, 4]");
    TORCH_CHECK(fp4_slot_map.dim() == 3 &&
                    fp4_slot_map.size(0) == 64 &&
                    fp4_slot_map.size(1) == 4 &&
                    fp4_slot_map.size(2) == 4,
                "fp4_slot_map must have shape [64, 4, 4]");
    TORCH_CHECK(fp8_word_offsets.dim() == 2 &&
                    fp8_word_offsets.size(0) == 64 &&
                    fp8_word_offsets.size(1) == 4,
                "fp8_word_offsets must have shape [64, 4]");
    TORCH_CHECK(inv_perm.numel() == N_fp4 + N_fp8, "inv_perm must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(workspace.is_cuda() && tile_counters.is_cuda(),
                "workspace and tile_counters must be CUDA tensors");
    TORCH_CHECK(workspace.dtype() == torch::kFloat32, "workspace must be float32");
    TORCH_CHECK(tile_counters.dtype() == torch::kInt32, "tile_counters must be int32");
    TORCH_CHECK(parallel_k >= 1, "parallel_k must be >= 1");

    int num_fp4_tiles = (N_fp4 + 127) / 128;
    int num_fp8_tiles = (N_fp8 + 127) / 128;
    int num_tiles = num_fp4_tiles + num_fp8_tiles;
    int num_k_tiles = K / 128;
    TORCH_CHECK(parallel_k <= num_k_tiles,
                "parallel_k must be <= K/128 for the split-K prototype");
    TORCH_CHECK(workspace.numel() == N_fp4 + N_fp8,
                "workspace must have N_fp4 + N_fp8 elements");
    TORCH_CHECK(tile_counters.numel() == num_tiles,
                "tile_counters must have one counter per output tile");

    auto y = torch::empty({1, N_fp4 + N_fp8},
        torch::dtype(torch::kFloat16).device(x.device()));

    fused_mixed_gemv_marlin_weights_splitk_auto(
        x.data_ptr(),
        w_fp4_q.data_ptr<int32_t>(),
        w_fp4_scales.data_ptr<uint8_t>(),
        static_cast<float>(w_fp4_global_scale),
        w_fp8_q.data_ptr<int32_t>(),
        w_fp8_scales.data_ptr<float>(),
        fp4_word_offsets.data_ptr<int32_t>(),
        fp4_slot_map.data_ptr<int32_t>(),
        fp8_word_offsets.data_ptr<int32_t>(),
        inv_perm.data_ptr<int32_t>(),
        workspace.data_ptr<float>(),
        tile_counters.data_ptr<int32_t>(),
        y.data_ptr(),
        N_fp4, N_fp8, K, parallel_k
    );

    return y;
}

// ============================================================================
// PyBind11 module definition
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mx_gemm_w4a8", &mx_gemm_w4a8_torch,
          "MXFP8 activation x MXFP4 weight GEMM -> BF16 output\n"
          "Args: x_fp8[M,K], x_sf, w_fp4, w_sf, M, N, K\n"
          "Returns: D[M,N] BF16",
          py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w_fp4"), py::arg("w_sf"),
          py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("mx_gemm_w6a8", &mx_gemm_w6a8_torch,
          "MXFP8 activation x MXFP6 weight GEMM -> BF16 output\n"
          "Args: x_fp8[M,K], x_sf, w_fp6, w_sf, M, N, K\n"
          "Returns: D[M,N] BF16",
          py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w_fp6"), py::arg("w_sf"),
          py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("mx_gemm_w8a8", &mx_gemm_w8a8_torch,
          "MXFP8 activation x MXFP8 weight GEMM -> BF16 output\n"
          "Args: x_fp8[M,K], x_sf, w_fp8, w_sf, M, N, K\n"
          "Returns: D[M,N] BF16",
          py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w_fp8"), py::arg("w_sf"),
          py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("reorder_sf", &reorder_sf_torch,
          "Reorder scale factors from row-major to CUTLASS interleaved layout\n"
          "Args: sf_rowmajor[dim0, K/32], dim0, K\n"
          "Returns: sf_reordered (1D uint8)",
          py::arg("sf_rowmajor"), py::arg("dim0"), py::arg("K"));

    m.def("quantize_act_mxfp8", &quantize_act_mxfp8_torch,
          "Online MXFP8 activation quantization from BF16\n"
          "Args: x_bf16[M, K]\n"
          "Returns: (x_fp8[M,K], x_sf[M,K/32]) as uint8 tensors",
          py::arg("x_bf16"));

    m.def("cublas_mxfp8_gemm", &cublas_mxfp8_gemm_torch,
          "cuBLASLt MXFP8×MXFP8 GEMM", py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w"), py::arg("w_sf"), py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("cublas_mx_w4a8", &cublas_mx_w4a8_torch,
          "cuBLASLt MXFP8×MXFP4 GEMM", py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w"), py::arg("w_sf"), py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("cublas_mx_w6a8", &cublas_mx_w6a8_torch,
          "cuBLASLt MXFP8×MXFP6 GEMM", py::arg("x_fp8"), py::arg("x_sf"),
          py::arg("w"), py::arg("w_sf"), py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("fused_mixed_gemv", &fused_mixed_gemv_torch,
          "Fused NVFP4+FP8 mixed-precision GEMV for M=1 decode\n"
          "Args: x[1,K] fp16, w_fp4[N_fp4,K/2] uint8, w_fp4_scales[N_fp4,K/16] uint8,\n"
          "      w_fp4_global_scale (float), w_fp8[N_fp8,K] uint8,\n"
          "      w_fp8_scales[N_fp8] float32, inv_perm[N_total] int32,\n"
          "      N_fp4, N_fp8, K\n"
          "Returns: y[1, N_total] fp16",
          py::arg("x"), py::arg("w_fp4"), py::arg("w_fp4_scales"),
          py::arg("w_fp4_global_scale"),
          py::arg("w_fp8"), py::arg("w_fp8_scales"),
          py::arg("inv_perm"),
          py::arg("N_fp4"), py::arg("N_fp8"), py::arg("K"));

    m.def("fused_mixed_gemv_marlin_weights", &fused_mixed_gemv_marlin_weights_torch,
          "Fused mixed GEMV using Marlin-repacked qweights and plain scales\n"
          "Args: x[1,K] fp16, w_fp4_q[K/16,N_fp4*2] int32,\n"
          "      w_fp4_scales[N_fp4,K/16] uint8, w_fp4_global_scale,\n"
          "      w_fp8_q[K/16,N_fp8*4] int32, w_fp8_scales[N_fp8] float32,\n"
          "      fp4_word_offsets[64,4] int32, fp4_slot_map[64,4,4] int32,\n"
          "      fp8_word_offsets[64,4] int32,\n"
          "      inv_perm[N_total] int32, N_fp4, N_fp8, K\n"
          "Returns: y[1, N_total] fp16",
          py::arg("x"), py::arg("w_fp4_q"), py::arg("w_fp4_scales"),
          py::arg("w_fp4_global_scale"),
          py::arg("w_fp8_q"), py::arg("w_fp8_scales"),
          py::arg("fp4_word_offsets"), py::arg("fp4_slot_map"),
          py::arg("fp8_word_offsets"),
          py::arg("inv_perm"),
          py::arg("N_fp4"), py::arg("N_fp8"), py::arg("K"));

    m.def("fused_mixed_gemv_marlin_weights_splitk",
          &fused_mixed_gemv_marlin_weights_splitk_torch,
          "Fused mixed GEMV using Marlin-repacked qweights with split-K tile scheduling\n"
          "Args: x[1,K] fp16, w_fp4_q[K/16,N_fp4*2] int32,\n"
          "      w_fp4_scales[N_fp4,K/16] uint8, w_fp4_global_scale,\n"
          "      w_fp8_q[K/16,N_fp8*4] int32, w_fp8_scales[N_fp8] float32,\n"
          "      fp4_word_offsets[64,4] int32, fp4_slot_map[64,4,4] int32,\n"
          "      fp8_word_offsets[64,4] int32, inv_perm[N_total] int32,\n"
          "      workspace[N_total] float32, tile_counters[num_tiles] int32,\n"
          "      N_fp4, N_fp8, K, parallel_k\n"
          "Returns: y[1, N_total] fp16",
          py::arg("x"), py::arg("w_fp4_q"), py::arg("w_fp4_scales"),
          py::arg("w_fp4_global_scale"),
          py::arg("w_fp8_q"), py::arg("w_fp8_scales"),
          py::arg("fp4_word_offsets"), py::arg("fp4_slot_map"),
          py::arg("fp8_word_offsets"),
          py::arg("inv_perm"), py::arg("workspace"), py::arg("tile_counters"),
          py::arg("N_fp4"), py::arg("N_fp8"), py::arg("K"),
          py::arg("parallel_k"));

    m.def("fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4",
          &fused_mixed_gemv_marlin_weights_splitk_staged_nvfp4_torch,
          "Alternate split-K fused mixed GEMV where the NVFP4 path also stages qweights/scales\n"
          "Args: x[1,K] fp16, w_fp4_q[K/16,N_fp4*2] int32,\n"
          "      w_fp4_scales[N_fp4,K/16] uint8, w_fp4_global_scale,\n"
          "      w_fp8_q[K/16,N_fp8*4] int32, w_fp8_scales[N_fp8] float32,\n"
          "      fp4_word_offsets[64,4] int32, fp4_slot_map[64,4,4] int32,\n"
          "      fp8_word_offsets[64,4] int32, inv_perm[N_total] int32,\n"
          "      workspace[N_total] float32, tile_counters[num_tiles] int32,\n"
          "      N_fp4, N_fp8, K, parallel_k\n"
          "Returns: y[1, N_total] fp16",
          py::arg("x"), py::arg("w_fp4_q"), py::arg("w_fp4_scales"),
          py::arg("w_fp4_global_scale"),
          py::arg("w_fp8_q"), py::arg("w_fp8_scales"),
          py::arg("fp4_word_offsets"), py::arg("fp4_slot_map"),
          py::arg("fp8_word_offsets"),
          py::arg("inv_perm"), py::arg("workspace"), py::arg("tile_counters"),
          py::arg("N_fp4"), py::arg("N_fp8"), py::arg("K"),
          py::arg("parallel_k"));

    m.def("fused_mixed_gemv_marlin_weights_splitk_auto",
          &fused_mixed_gemv_marlin_weights_splitk_auto_torch,
          "Heuristic wrapper that chooses the current best split-K mixed GEMV variant\n"
          "Args: x[1,K] fp16, w_fp4_q[K/16,N_fp4*2] int32,\n"
          "      w_fp4_scales[N_fp4,K/16] uint8, w_fp4_global_scale,\n"
          "      w_fp8_q[K/16,N_fp8*4] int32, w_fp8_scales[N_fp8] float32,\n"
          "      fp4_word_offsets[64,4] int32, fp4_slot_map[64,4,4] int32,\n"
          "      fp8_word_offsets[64,4] int32, inv_perm[N_total] int32,\n"
          "      workspace[N_total] float32, tile_counters[num_tiles] int32,\n"
          "      N_fp4, N_fp8, K, parallel_k\n"
          "Returns: y[1, N_total] fp16",
          py::arg("x"), py::arg("w_fp4_q"), py::arg("w_fp4_scales"),
          py::arg("w_fp4_global_scale"),
          py::arg("w_fp8_q"), py::arg("w_fp8_scales"),
          py::arg("fp4_word_offsets"), py::arg("fp4_slot_map"),
          py::arg("fp8_word_offsets"),
          py::arg("inv_perm"), py::arg("workspace"), py::arg("tile_counters"),
          py::arg("N_fp4"), py::arg("N_fp8"), py::arg("K"),
          py::arg("parallel_k"));
}
