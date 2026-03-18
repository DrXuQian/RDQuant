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
}
