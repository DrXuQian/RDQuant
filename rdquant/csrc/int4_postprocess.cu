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

namespace {

constexpr int kUint4GroupSize = 128;
constexpr int kUint4Threads = 128;
constexpr int kUint4Pack = 8;

__device__ __forceinline__ float unpack_uint4_dot8(
    int32_t packed_word,
    const half* __restrict__ x_chunk,
    float scale)
{
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kUint4Pack; ++i) {
        int q = (packed_word >> (4 * i)) & 0xF;
        acc += (static_cast<float>(q) - 8.0f) *
               __half2float(x_chunk[i]) * scale;
    }
    return acc;
}

__global__ void fused_uint4_groupwise_gemv_kernel(
    const half* __restrict__ x,              // [M, K]
    const int32_t* __restrict__ w_packed,    // [N, K/8]
    const half* __restrict__ w_scales,       // [N, K/group_size]
    half* __restrict__ y,                    // [M, N]
    int M,
    int N,
    int K,
    int n_groups)
{
    __shared__ half sh_x[kUint4GroupSize];

    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int words_per_group = kUint4GroupSize / kUint4Pack;
    const int words_per_row = K / kUint4Pack;

    float acc = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const int k0 = g * kUint4GroupSize;
        sh_x[threadIdx.x] = x[m * K + k0 + threadIdx.x];
        __syncthreads();

        if (n < N) {
            const float scale = __half2float(w_scales[n * n_groups + g]);
            const int32_t* row_words =
                w_packed + n * words_per_row + g * words_per_group;
#pragma unroll
            for (int word_idx = 0; word_idx < words_per_group; ++word_idx) {
                acc += unpack_uint4_dot8(
                    row_words[word_idx], sh_x + word_idx * kUint4Pack, scale);
            }
        }
        __syncthreads();
    }

    if (n < N) {
        y[m * N + n] = __float2half(acc);
    }
}

__global__ void fused_uint4_groupwise_gemv_splitk_kernel(
    const half* __restrict__ x,              // [M, K]
    const int32_t* __restrict__ w_packed,    // [N, K/8]
    const half* __restrict__ w_scales,       // [N, K/group_size]
    float* __restrict__ workspace,           // [M, N]
    int M,
    int N,
    int K,
    int n_groups,
    int parallel_k)
{
    __shared__ half sh_x[kUint4GroupSize];

    const int m = blockIdx.y;
    const int slice = blockIdx.z;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int words_per_group = kUint4GroupSize / kUint4Pack;
    const int words_per_row = K / kUint4Pack;
    const int groups_per_slice = (n_groups + parallel_k - 1) / parallel_k;
    const int g_begin = slice * groups_per_slice;
    const int g_end = min(n_groups, g_begin + groups_per_slice);

    if (g_begin >= g_end) {
        return;
    }

    float acc = 0.0f;
    for (int g = g_begin; g < g_end; ++g) {
        const int k0 = g * kUint4GroupSize;
        sh_x[threadIdx.x] = x[m * K + k0 + threadIdx.x];
        __syncthreads();

        if (n < N) {
            const float scale = __half2float(w_scales[n * n_groups + g]);
            const int32_t* row_words =
                w_packed + n * words_per_row + g * words_per_group;
#pragma unroll
            for (int word_idx = 0; word_idx < words_per_group; ++word_idx) {
                acc += unpack_uint4_dot8(
                    row_words[word_idx], sh_x + word_idx * kUint4Pack, scale);
            }
        }
        __syncthreads();
    }

    if (n < N) {
        atomicAdd(&workspace[m * N + n], acc);
    }
}

__global__ void fused_uint4_finalize_kernel(
    const float* __restrict__ workspace,
    half* __restrict__ y,
    int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        y[idx] = __float2half(workspace[idx]);
    }
}

__global__ void fused_uint4_decode_splitk_kernel(
    const half* __restrict__ x,              // [M, K]
    const half* __restrict__ inv_awq,        // [K] or nullptr
    const int32_t* __restrict__ w_packed,    // [N, K/8]
    const half* __restrict__ w_scales,       // [N, K/group_size]
    float* __restrict__ workspace,           // [M, N]
    float* __restrict__ sum_workspace,       // [M]
    int M,
    int N,
    int K,
    int n_groups,
    int parallel_k,
    bool apply_awq)
{
    __shared__ half sh_x[kUint4GroupSize];

    const int m = blockIdx.y;
    const int tile = blockIdx.x;
    const int slice = blockIdx.z;
    const int n = tile * blockDim.x + threadIdx.x;
    const int words_per_group = kUint4GroupSize / kUint4Pack;
    const int words_per_row = K / kUint4Pack;
    const int groups_per_slice = (n_groups + parallel_k - 1) / parallel_k;
    const int g_begin = slice * groups_per_slice;
    const int g_end = min(n_groups, g_begin + groups_per_slice);

    if (g_begin >= g_end) {
        return;
    }

    float acc = 0.0f;
    float sum_local = 0.0f;

    for (int g = g_begin; g < g_end; ++g) {
        const int k0 = g * kUint4GroupSize;
        half x_val = x[m * K + k0 + threadIdx.x];
        if (apply_awq) {
            x_val = __hmul(x_val, inv_awq[k0 + threadIdx.x]);
        }
        sh_x[threadIdx.x] = x_val;
        if (tile == 0) {
            sum_local += __half2float(x_val);
        }
        __syncthreads();

        if (n < N) {
            const float scale = __half2float(w_scales[n * n_groups + g]);
            const int32_t* row_words =
                w_packed + n * words_per_row + g * words_per_group;
#pragma unroll
            for (int word_idx = 0; word_idx < words_per_group; ++word_idx) {
                acc += unpack_uint4_dot8(
                    row_words[word_idx], sh_x + word_idx * kUint4Pack, scale);
            }
        }
        __syncthreads();
    }

    if (tile == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_local += __shfl_down_sync(0xFFFFFFFF, sum_local, offset);
        }
        __shared__ float warp_sums[4];
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        if (lane == 0) {
            warp_sums[warp_id] = sum_local;
        }
        __syncthreads();
        if (warp_id == 0) {
            float block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
            }
            if (lane == 0) {
                atomicAdd(&sum_workspace[m], block_sum);
            }
        }
    }

    if (n < N) {
        atomicAdd(&workspace[m * N + n], acc);
    }
}

__global__ void fused_uint4_decode_post_kernel(
    const float* __restrict__ workspace,     // [M, N_combined]
    const half* __restrict__ corr,           // [N_int8]
    const float* __restrict__ sum_x,         // [M]
    const int* __restrict__ inv_perm,        // [N_total]
    half* __restrict__ y_out,                // [M, N_total]
    int N_total,
    int N_int4,
    int N_int8,
    int N_combined)
{
    extern __shared__ float shmem[];         // [N_combined]

    const int m = blockIdx.x;
    const float* ws_row = workspace + m * N_combined;
    half* out_row = y_out + m * N_total;
    const float sx = sum_x[m];

    for (int i = threadIdx.x; i < N_combined; i += blockDim.x) {
        shmem[i] = ws_row[i];
    }
    __syncthreads();

    for (int n = threadIdx.x; n < N_total; n += blockDim.x) {
        const int src = inv_perm[n];
        float result;
        if (src < N_int4) {
            result = shmem[src];
        } else {
            const int i8 = src - N_int4;
            result = shmem[N_int4 + i8] + shmem[N_int4 + N_int8 + i8] +
                     __half2float(corr[i8]) * sx;
        }
        out_row[n] = __float2half(result);
    }
}

__global__ void fused_uint4_decode_post_batched_kernel(
    const float* __restrict__ workspace,     // [M, N_combined]
    const half* __restrict__ corr,           // [N_int8]
    const float* __restrict__ sum_x,         // [M]
    const int* __restrict__ inv_perm,        // [N_total]
    half* __restrict__ y_out,                // [M, N_total]
    int M,
    int N_total,
    int N_int4,
    int N_int8,
    int N_combined)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N_total) {
        return;
    }

    const int m = idx / N_total;
    const int n = idx % N_total;
    const int src = inv_perm[n];
    const float* ws_row = workspace + m * N_combined;
    float result;

    if (src < N_int4) {
        result = ws_row[src];
    } else {
        const int i8 = src - N_int4;
        result = ws_row[N_int4 + i8] + ws_row[N_int4 + N_int8 + i8] +
                 __half2float(corr[i8]) * sum_x[m];
    }
    y_out[m * N_total + n] = __float2half(result);
}

}  // namespace

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


torch::Tensor fused_uint4_groupwise_gemv(
    torch::Tensor x,           // [M, K] half
    torch::Tensor w_packed,    // [N, K/8] int32
    torch::Tensor w_scales,    // [N, K/group_size] half
    int group_size,
    int parallel_k)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be CUDA");
    TORCH_CHECK(w_scales.is_cuda(), "w_scales must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(w_packed.dtype() == torch::kInt32, "w_packed must be int32");
    TORCH_CHECK(w_scales.dtype() == torch::kFloat16, "w_scales must be float16");
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be [N, K/8]");
    TORCH_CHECK(w_scales.dim() == 2, "w_scales must be [N, K/group_size]");
    TORCH_CHECK(group_size == kUint4GroupSize,
                "fused_uint4_groupwise_gemv currently requires group_size=128");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(0);
    const int n_groups = K / group_size;

    TORCH_CHECK(K % kUint4Pack == 0, "K must be divisible by 8");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(w_packed.size(1) == K / kUint4Pack,
                "w_packed must have shape [N, K/8]");
    TORCH_CHECK(w_scales.size(0) == N && w_scales.size(1) == n_groups,
                "w_scales must have shape [N, K/group_size]");

    auto y = torch::empty({M, N}, x.options());
    const dim3 block(kUint4Threads);
    const dim3 grid((N + kUint4Threads - 1) / kUint4Threads, M);

    if (parallel_k <= 1) {
        fused_uint4_groupwise_gemv_kernel<<<grid, block>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            w_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(w_scales.data_ptr<at::Half>()),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()),
            M, N, K, n_groups);
    } else {
        auto workspace = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));
        const dim3 splitk_grid((N + kUint4Threads - 1) / kUint4Threads, M,
                               parallel_k);
        fused_uint4_groupwise_gemv_splitk_kernel<<<splitk_grid, block>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            w_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(w_scales.data_ptr<at::Half>()),
            workspace.data_ptr<float>(),
            M, N, K, n_groups, parallel_k);

        const int total = M * N;
        const int finalize_threads = 256;
        const int finalize_blocks = (total + finalize_threads - 1) /
                                    finalize_threads;
        fused_uint4_finalize_kernel<<<finalize_blocks, finalize_threads>>>(
            workspace.data_ptr<float>(),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()),
            total);
    }

    return y;
}


torch::Tensor fused_uint4_decode(
    torch::Tensor x,           // [M, K] half
    torch::Tensor inv_awq,     // [K] half or empty
    torch::Tensor w_packed,    // [N_combined, K/8] int32
    torch::Tensor w_scales,    // [N_combined, K/group_size] half
    torch::Tensor corr,        // [N_int8] half
    torch::Tensor inv_perm,    // [N_total] int32
    int N_int4,
    int N_int8,
    int group_size,
    int parallel_k)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && w_scales.is_cuda() &&
                    corr.is_cuda() && inv_perm.is_cuda(),
                "all fused_uint4_decode inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(w_packed.dtype() == torch::kInt32, "w_packed must be int32");
    TORCH_CHECK(w_scales.dtype() == torch::kFloat16, "w_scales must be float16");
    TORCH_CHECK(corr.dtype() == torch::kFloat16, "corr must be float16");
    TORCH_CHECK(inv_perm.dtype() == torch::kInt32, "inv_perm must be int32");
    TORCH_CHECK(group_size == kUint4GroupSize,
                "fused_uint4_decode currently requires group_size=128");

    const bool apply_awq = inv_awq.defined() && inv_awq.numel() > 0;
    if (apply_awq) {
        TORCH_CHECK(inv_awq.is_cuda() && inv_awq.dtype() == torch::kFloat16,
                    "inv_awq must be CUDA float16");
    }

    const int M = x.size(0);
    const int K = x.size(1);
    const int N_combined = w_packed.size(0);
    const int N_total = N_int4 + N_int8;
    const int n_groups = K / group_size;

    TORCH_CHECK(K % kUint4Pack == 0, "K must be divisible by 8");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(w_packed.size(1) == K / kUint4Pack,
                "w_packed must have shape [N_combined, K/8]");
    TORCH_CHECK(w_scales.size(0) == N_combined && w_scales.size(1) == n_groups,
                "w_scales must have shape [N_combined, K/group_size]");
    TORCH_CHECK(corr.numel() == N_int8, "corr must have N_int8 elements");
    TORCH_CHECK(inv_perm.numel() == N_total, "inv_perm must have N_total elements");

    auto workspace = torch::zeros({M, N_combined}, x.options().dtype(torch::kFloat32));
    auto sum_workspace = torch::zeros({M}, x.options().dtype(torch::kFloat32));
    auto y_out = torch::empty({M, N_total}, x.options());

    const dim3 block(kUint4Threads);
    const dim3 grid((N_combined + kUint4Threads - 1) / kUint4Threads, M,
                    max(parallel_k, 1));
    const half* inv_awq_ptr = apply_awq ?
        reinterpret_cast<const half*>(inv_awq.data_ptr<at::Half>()) : nullptr;

    fused_uint4_decode_splitk_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        inv_awq_ptr,
        w_packed.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(w_scales.data_ptr<at::Half>()),
        workspace.data_ptr<float>(),
        sum_workspace.data_ptr<float>(),
        M, N_combined, K, n_groups, max(parallel_k, 1), apply_awq);

    int post_threads = 256;
    if (N_total > 512) post_threads = 512;
    if (N_total > 1024) post_threads = 1024;
    const int shmem_bytes = N_combined * sizeof(float);
    if (M <= 4 && shmem_bytes <= 48 * 1024) {
        fused_uint4_decode_post_kernel<<<M, post_threads, shmem_bytes>>>(
            workspace.data_ptr<float>(),
            reinterpret_cast<const half*>(corr.data_ptr<at::Half>()),
            sum_workspace.data_ptr<float>(),
            inv_perm.data_ptr<int>(),
            reinterpret_cast<half*>(y_out.data_ptr<at::Half>()),
            N_total, N_int4, N_int8, N_combined);
    } else {
        const int total = M * N_total;
        const int blocks = (total + post_threads - 1) / post_threads;
        fused_uint4_decode_post_batched_kernel<<<blocks, post_threads>>>(
            workspace.data_ptr<float>(),
            reinterpret_cast<const half*>(corr.data_ptr<at::Half>()),
            sum_workspace.data_ptr<float>(),
            inv_perm.data_ptr<int>(),
            reinterpret_cast<half*>(y_out.data_ptr<at::Half>()),
            M, N_total, N_int4, N_int8, N_combined);
    }

    return y_out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // v1 (kept for backwards compat)
    m.def("fused_awq_sum", &fused_awq_sum_v2, "Fused AWQ scaling + sum_x (v2, in-place, half2)");
    m.def("fused_sum_only", &fused_sum_only_v2, "Fused sum_x (v2, half2)");
    m.def("fused_post", &fused_post_v2, "Fused correction + reorder (v2, shmem, int32)");
    m.def("fused_uint4_gemv", &fused_uint4_groupwise_gemv,
          "Non-persistent UINT4 groupwise GEMV (row-major packed)");
    m.def("fused_uint4_decode", &fused_uint4_decode,
          "Decode-oriented UINT4 path with fused AWQ, correction, and reorder");
}
