#include "rdquant_kernels.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations: MXFP8 activation x MXFP6 weight -> BF16 output
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration (activation — always MXFP8)
using         ElementA_w6a8    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using         LayoutATag_w6a8  = cutlass::layout::RowMajor;
constexpr int AlignmentA_w6a8  = 16;

// B matrix configuration (weight — MXFP6)
using         ElementB_w6a8    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using         LayoutBTag_w6a8  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB_w6a8  = 96 * 8 / cutlass::sizeof_bits<typename ElementB_w6a8::DataType>::value;

// C/D matrix configuration (output)
using         ElementD_w6a8    = cutlass::bfloat16_t;
using         ElementC_w6a8    = cutlass::bfloat16_t;
using         LayoutCTag_w6a8  = cutlass::layout::RowMajor;
using         LayoutDTag_w6a8  = cutlass::layout::RowMajor;
constexpr int AlignmentD_w6a8  = 128 / cutlass::sizeof_bits<ElementD_w6a8>::value;
constexpr int AlignmentC_w6a8  = 128 / cutlass::sizeof_bits<ElementC_w6a8>::value;

// Kernel functional config
using ElementAccumulator_w6a8  = float;
using ArchTag_w6a8             = cutlass::arch::Sm120;
using OperatorClass_w6a8       = cutlass::arch::OpClassBlockScaledTensorOp;

// Kernel perf config
using ThreadBlockShape_w6a8    = Shape<_128,_128,_128>;
using ClusterShape_w6a8        = Shape<_1,_1,_1>;

void mx_gemm_w6a8(
    const void *A_ptr,
    const void *SFA_ptr,
    const void *B_ptr,
    const void *SFB_ptr,
    int M, int N, int K,
    void *D_ptr
)
{
    auto A   = reinterpret_cast<const cutlass::float_e4m3_t*>(A_ptr);
    auto SFA = reinterpret_cast<const cutlass::float_ue8m0_t*>(SFA_ptr);
    auto B   = reinterpret_cast<const cutlass::float_e3m2_t*>(B_ptr);
    auto SFB = reinterpret_cast<const cutlass::float_ue8m0_t*>(SFB_ptr);
    auto D   = reinterpret_cast<cutlass::bfloat16_t*>(D_ptr);

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag_w6a8, OperatorClass_w6a8,
        ThreadBlockShape_w6a8, ClusterShape_w6a8,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator_w6a8, ElementAccumulator_w6a8,
        ElementC_w6a8, LayoutCTag_w6a8, AlignmentC_w6a8,
        ElementD_w6a8, LayoutDTag_w6a8, AlignmentD_w6a8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag_w6a8, OperatorClass_w6a8,
        ElementA_w6a8, LayoutATag_w6a8, AlignmentA_w6a8,
        ElementB_w6a8, LayoutBTag_w6a8, AlignmentB_w6a8,
        ElementAccumulator_w6a8,
        ThreadBlockShape_w6a8, ClusterShape_w6a8,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    Gemm gemmOp;

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            A, stride_A,
            B, stride_B,
            SFA, layout_SFA,
            SFB, layout_SFB
        },
        { // Epilogue arguments
            {1.0f, 0.0f},
            nullptr, stride_C,
            D, stride_D
        }
    };

    auto status = gemmOp(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM w6a8 failed: "
                  << cutlass::cutlassGetStatusString(status)
                  << " (code: " << static_cast<int>(status) << ")"
                  << std::endl;
    }
    assert(status == cutlass::Status::kSuccess);
}
