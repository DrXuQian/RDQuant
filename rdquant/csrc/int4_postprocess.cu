/**
 * Fused pre/post-processing kernels for INT4/INT8 single-Marlin inference.
 *
 * v2: Vectorized half2 loads/stores, shared memory post-kernel,
 *     int32 perm, in-place AWQ scaling, precomputed 1/α.
 *
 * Pre-kernel:  x *= 1/α (in-place) + sum_x reduction     →  1 launch
 * Post-kernel: INT8 correction + inv_perm reorder (shmem)  →  1 launch
 */

#include <torch/extension.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Pre-kernel v2: in-place AWQ scaling + sum_x, vectorized half2
// ---------------------------------------------------------------------------
// Processes x in-place: x[k] *= inv_awq[k], accumulates sum.
// inv_awq = 1/α is precomputed on host to avoid per-element division.
// Uses half2 vectorized loads/stores for 2x throughput.

__global__ void fused_awq_sum_v2_kernel(
    half* __restrict__ x,                  // [M, K] — modified in-place
    const half* __restrict__ inv_awq,      // [K] = 1/α
    float* __restrict__ sum_x,             // [M]
    int K)
{
    const int m = blockIdx.x;
    half* row = x + m * K;

    const int K2 = K >> 1;           // number of half2 elements
    half2* row2 = reinterpret_cast<half2*>(row);
    const half2* inv2 = reinterpret_cast<const half2*>(inv_awq);

    float local_sum = 0.0f;

    for (int k = threadIdx.x; k < K2; k += blockDim.x) {
        half2 v = row2[k];
        half2 a = inv2[k];
        half2 r = __hmul2(v, a);      // x * (1/α) = x / α
        row2[k] = r;
        float2 rf = __half22float2(r);
        local_sum += rf.x + rf.y;
    }
    // Handle odd K (unlikely for model dims, but safe)
    if ((K & 1) && threadIdx.x == 0) {
        int k = K - 1;
        float v = __half2float(row[k]);
        float a = __half2float(inv_awq[k]);
        float r = v * a;
        row[k] = __float2half(r);
        local_sum += r;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float val = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) sum_x[m] = val;
    }
}


// ---------------------------------------------------------------------------
// Pre-kernel v2 (no AWQ): just sum_x, vectorized
// ---------------------------------------------------------------------------
__global__ void fused_sum_only_v2_kernel(
    const half* __restrict__ x,
    float* __restrict__ sum_x,
    int K)
{
    const int m = blockIdx.x;
    const half2* row2 = reinterpret_cast<const half2*>(x + m * K);
    const int K2 = K >> 1;

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < K2; k += blockDim.x) {
        float2 v = __half22float2(row2[k]);
        local_sum += v.x + v.y;
    }
    if ((K & 1) && threadIdx.x == 0)
        local_sum += __half2float(x[m * K + K - 1]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float val = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) sum_x[m] = val;
    }
}


// ---------------------------------------------------------------------------
// Post-kernel v2: shared-memory + int32 perm + vectorized
// ---------------------------------------------------------------------------
// One block per token-row.  All threads cooperate to:
//   1. Load y_combined into shared memory (coalesced)
//   2. Each thread reads from shmem via inv_perm (fast random access)
//   3. Write y_out (coalesced)
//
// Max shared memory needed: N_combined * sizeof(half) = 13696*2 = 27 KB (fits)

__global__ void fused_post_v2_kernel(
    const half* __restrict__ y_combined,   // [M, N_combined]
    const half* __restrict__ corr,         // [N_int8]
    const float* __restrict__ sum_x,       // [M]
    const int* __restrict__ inv_perm,      // [N_total] int32
    half* __restrict__ y_out,              // [M, N_total]
    int N_total, int N_int4, int N_int8, int N_combined)
{
    extern __shared__ half shmem[];        // [N_combined]

    const int m = blockIdx.x;              // token index
    const half* y_row = y_combined + m * N_combined;
    half* out_row = y_out + m * N_total;
    const float sx = sum_x[m];

    // --- Phase 1: coalesced load of y_combined into shmem ---
    for (int i = threadIdx.x; i < N_combined; i += blockDim.x) {
        shmem[i] = y_row[i];
    }
    __syncthreads();

    // --- Phase 2: scatter-read from shmem + write coalesced ---
    for (int n = threadIdx.x; n < N_total; n += blockDim.x) {
        int src = inv_perm[n];
        float result;

        if (src < N_int4) {
            result = __half2float(shmem[src]);
        } else {
            int i8 = src - N_int4;
            float y_h = __half2float(shmem[N_int4 + i8]);
            float y_l = __half2float(shmem[N_int4 + N_int8 + i8]);
            float c   = __half2float(corr[i8]);
            result = y_h + y_l + c * sx;
        }

        out_row[n] = __float2half(result);
    }
}


// ---------------------------------------------------------------------------
// Post-kernel for M > 1: no shared memory (would exceed shmem for large N*M)
// Uses int32 perm + __ldg for cache-friendly reads
// ---------------------------------------------------------------------------
__global__ void fused_post_v2_batched_kernel(
    const half* __restrict__ y_combined,
    const half* __restrict__ corr,
    const float* __restrict__ sum_x,
    const int* __restrict__ inv_perm,
    half* __restrict__ y_out,
    int M, int N_total, int N_int4, int N_int8, int N_combined)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N_total) return;

    int m = idx / N_total;
    int n = idx % N_total;

    int src = __ldg(&inv_perm[n]);
    const half* y_row = y_combined + m * N_combined;
    float result;

    if (src < N_int4) {
        result = __half2float(__ldg(&y_row[src]));
    } else {
        int i8 = src - N_int4;
        float y_h = __half2float(__ldg(&y_row[N_int4 + i8]));
        float y_l = __half2float(__ldg(&y_row[N_int4 + N_int8 + i8]));
        float c   = __half2float(__ldg(&corr[i8]));
        result = y_h + y_l + c * sum_x[m];
    }

    y_out[m * N_total + n] = __float2half(result);
}


// ===================================================================
// Python bindings
// ===================================================================

// Pre-process v2: in-place AWQ (x *= 1/α) + sum_x
// inv_awq [K] half must be precomputed as 1.0/awq_scales
std::tuple<torch::Tensor, torch::Tensor> fused_awq_sum_v2(
    torch::Tensor x,            // [M, K] half — MODIFIED IN-PLACE
    torch::Tensor inv_awq)      // [K] half = 1/α
{
    int M = x.size(0);
    int K = x.size(1);

    auto sum_x = torch::empty({M}, x.options().dtype(torch::kFloat32));

    // Threads: enough to cover K/2 half2 elements, power of 2
    int threads = 256;
    if (K / 2 > 512) threads = 512;
    if (K / 2 > 1024) threads = 1024;

    fused_awq_sum_v2_kernel<<<M, threads>>>(
        reinterpret_cast<half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(inv_awq.data_ptr<at::Half>()),
        sum_x.data_ptr<float>(),
        K
    );

    return {x, sum_x};   // x modified in-place
}


// Pre-process v2: sum only (no AWQ)
torch::Tensor fused_sum_only_v2(torch::Tensor x)
{
    int M = x.size(0);
    int K = x.size(1);

    auto sum_x = torch::empty({M}, x.options().dtype(torch::kFloat32));

    int threads = 256;
    if (K / 2 > 512) threads = 512;
    if (K / 2 > 1024) threads = 1024;

    fused_sum_only_v2_kernel<<<M, threads>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        sum_x.data_ptr<float>(),
        K
    );

    return sum_x;
}


// Post-process v2: correction + reorder
// inv_perm is int32 (not int64) to halve bandwidth
torch::Tensor fused_post_v2(
    torch::Tensor y_combined,   // [M, N_combined] half
    torch::Tensor corr,         // [N_int8] half
    torch::Tensor sum_x,        // [M] float32
    torch::Tensor inv_perm,     // [N_total] int32
    int N_int4,
    int N_int8)
{
    int M = y_combined.size(0);
    int N_combined = y_combined.size(1);
    int N_total = N_int4 + N_int8;

    auto y_out = torch::empty({M, N_total}, y_combined.options());

    if (M <= 4) {
        // Small M: use shared memory version (one block per token)
        int shmem_bytes = N_combined * sizeof(half);
        int threads = 256;
        if (N_total > 512) threads = 512;
        if (N_total > 1024) threads = 1024;

        fused_post_v2_kernel<<<M, threads, shmem_bytes>>>(
            reinterpret_cast<const half*>(y_combined.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(corr.data_ptr<at::Half>()),
            sum_x.data_ptr<float>(),
            inv_perm.data_ptr<int>(),
            reinterpret_cast<half*>(y_out.data_ptr<at::Half>()),
            N_total, N_int4, N_int8, N_combined
        );
    } else {
        // Large M: batched version without shared memory
        int total = M * N_total;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        fused_post_v2_batched_kernel<<<blocks, threads>>>(
            reinterpret_cast<const half*>(y_combined.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(corr.data_ptr<at::Half>()),
            sum_x.data_ptr<float>(),
            inv_perm.data_ptr<int>(),
            reinterpret_cast<half*>(y_out.data_ptr<at::Half>()),
            M, N_total, N_int4, N_int8, N_combined
        );
    }

    return y_out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // v1 (kept for backwards compat)
    m.def("fused_awq_sum", &fused_awq_sum_v2, "Fused AWQ scaling + sum_x (v2, in-place, half2)");
    m.def("fused_sum_only", &fused_sum_only_v2, "Fused sum_x (v2, half2)");
    m.def("fused_post", &fused_post_v2, "Fused correction + reorder (v2, shmem, int32)");
}
