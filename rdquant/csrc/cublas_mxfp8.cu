/*
 * cuBLASLt MX block-scaled GEMM wrappers.
 *
 * Uses CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 for per-32-element
 * block scaling — the same format as OCP MX.
 *
 * Supports: MXFP8×MXFP8, MXFP8×MXFP4, MXFP8×MXFP6
 *
 * Key advantage over CUTLASS: cuBLASLt auto-selects the best kernel
 * via heuristics, which gives lower latency for small M.
 */

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <library_types.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                    \
                    __FILE__, __LINE__, (int)status);                          \
        }                                                                     \
    } while (0)

// Persistent handle
static cublasLtHandle_t g_ltHandle = nullptr;
static void* g_workspace = nullptr;
static size_t g_workspaceSize = 32 * 1024 * 1024;

static void ensure_handle() {
    if (g_ltHandle == nullptr) {
        CHECK_CUBLAS(cublasLtCreate(&g_ltHandle));
        cudaMalloc(&g_workspace, g_workspaceSize);
    }
}

// Generic MX GEMM: computes Y[M,N] = X[M,K] @ W[N,K]^T  (row-major)
// A_type: activation element type (always CUDA_R_8F_E4M3)
// B_type: weight element type (CUDA_R_8F_E4M3, CUDA_R_4F_E2M1, or CUDA_R_6F_E3M2)
static void cublas_mx_gemm_impl(
    const void *X_ptr,     // activation [M, K] row-major
    const void *X_sf_ptr,  // activation scales [M * K/32] UE8M0
    const void *W_ptr,     // weight [N, K] row-major
    const void *W_sf_ptr,  // weight scales [N * K/32] UE8M0
    int M, int N, int K,
    void *Y_ptr,           // output [M, N] row-major, BF16
    cudaDataType_t B_type  // weight data type
)
{
    ensure_handle();

    // Row-major → col-major trick:
    // Y_row[M,N] = X_row[M,K] @ W_row[N,K]^T
    // Reinterpret as col-major:
    //   Y_col[N,M] = W_col[N,K] @ X_col[K,M]  (no transpose needed!)
    //   where X_row[M,K] reinterpreted as col gives [K,M] with ld=K
    //   and W_row[N,K] reinterpreted as col gives [K,N] with ld=K
    //
    // But cuBLASLt expects: D = op(A) * op(B)
    //   A_cublas = W, transa=N (W is already [N,K] in col-major layout sense → [K,N] col)
    //   Actually: W_row[N,K] = contiguous N*K floats, col-major sees this as [K,N] matrix.
    //   We want W acting as [N,K] → need transa=T on the [K,N] col-major view.
    //   X_row[M,K] = col-major [K,M], transb=N.
    //
    // D_col[N,M] = W^T[N,K] * X[K,M], transa=T, transb=N
    //   A=W_ptr, lda=K (physical [K,N] col-major)
    //   B=X_ptr, ldb=K (physical [K,M] col-major)
    //   D=Y_ptr, ldd=N (physical [N,M] col-major = [M,N] row-major)

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Block scaling for A (weight) and B (activation)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Scale pointers: A_cublas=W, B_cublas=X
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &W_sf_ptr, sizeof(W_sf_ptr)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &X_sf_ptr, sizeof(X_sf_ptr)));

    // Matrix layouts: A_cublas=W[K,N] col-major, B_cublas=X[K,M] col-major
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, B_type, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, N, M, N));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspaceSize, sizeof(g_workspaceSize)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heurResult = {};
    cublasStatus_t heurStatus = cublasLtMatmulAlgoGetHeuristic(
        g_ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &heurResult, &returnedResults);

    if (heurStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        fprintf(stderr, "cuBLASLt MX GEMM: no algorithm (M=%d,N=%d,K=%d,Btype=%d,status=%d)\n",
                M, N, K, (int)B_type, (int)heurStatus);
        goto cleanup;
    }

    CHECK_CUBLAS(cublasLtMatmul(
        g_ltHandle, opDesc, &alpha,
        W_ptr, Adesc,   // A_cublas = W
        X_ptr, Bdesc,   // B_cublas = X
        &beta,
        Y_ptr, Cdesc,
        Y_ptr, Ddesc,
        &heurResult.algo,
        g_workspace, g_workspaceSize, 0));

cleanup:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
}

// Public API: MXFP8 activation × MXFP8 weight
void cublas_mxfp8_gemm(
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C
) {
    cublas_mx_gemm_impl(A, A_sf, B, B_sf, M, N, K, C, CUDA_R_8F_E4M3);
}

// Public API: MXFP8 activation × MXFP4 weight
void cublas_mx_gemm_w4a8(
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C
) {
    cublas_mx_gemm_impl(A, A_sf, B, B_sf, M, N, K, C, CUDA_R_4F_E2M1);
}

// Public API: MXFP8 activation × MXFP6 weight
void cublas_mx_gemm_w6a8(
    const void *A, const void *A_sf, const void *B, const void *B_sf,
    int M, int N, int K, void *C
) {
    cublas_mx_gemm_impl(A, A_sf, B, B_sf, M, N, K, C, CUDA_R_6F_E3M2);
}
