/* rms_norm.cu — RMSNorm CUDA kernels
 *
 * Standard RMSNorm:  out = x * (1 / sqrt(mean(x^2) + eps)) * weight
 * Per-head RMSNorm:  each [head_dim] slice normalized independently, in-place
 */

#include <cuda_runtime.h>

/* Standard RMSNorm: x[seq, hidden] -> out[seq, hidden]
 * weight[hidden], eps scalar. Each row normalized independently.
 * 1D block layout: one block per row, all threads in block cooperate on reduction.
 * shared_sum[0] = sum of squares, shared_sum[1] = scale */
__global__ void rms_norm_kernel(float *__restrict__ out,
                                   const float *__restrict__ x,
                                   const float *__restrict__ weight,
                                   int seq_len,
                                   int hidden,
                                   float eps) {
    int row = blockIdx.x;
    if (row >= seq_len) return;

    const float * row_x = x + row * hidden;
    float * row_out = out + row * hidden;

    /* Each thread computes partial sum of squares */
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum_sq += row_x[i] * row_x[i];
    }

    /* Shared memory: index 0 = sum, index 1 = scale */
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = sum_sq;
    __syncthreads();

    /* Tree-based reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    /* Thread 0 computes scale and stores in shared_mem[1] so all threads can read it */
    float scale = 0.0f;
    if (threadIdx.x == 0) {
        float rms = sqrtf(shared_mem[0] / (float)hidden + eps);
        scale = 1.0f / rms;
        shared_mem[1] = scale;
    }
    __syncthreads();
    scale = shared_mem[1];

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        row_out[i] = row_x[i] * scale * weight[i];
    }
}

/* Per-head RMSNorm: x[seq, n_heads * head_dim] in-place
 * weight[head_dim]. Each head slice normalized independently.
 * 1D block per slice: one block per (row, head) pair.
 * shared_mem[0] = sum, shared_mem[1] = scale */
__global__ void rms_norm_per_head_kernel(float *__restrict__ x,
                                            const float *__restrict__ weight,
                                            int seq_len,
                                            int n_heads,
                                            int head_dim,
                                            float eps) {
    int slice_idx = blockIdx.x;
    int total_slices = seq_len * n_heads;
    if (slice_idx >= total_slices) return;

    int row = slice_idx / n_heads;
    int head = slice_idx % n_heads;
    float * slice = x + row * (n_heads * head_dim) + head * head_dim;

    /* Partial sum of squares */
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sum_sq += slice[i] * slice[i];
    }

    /* Shared memory: index 0 = sum, index 1 = scale */
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float scale = 0.0f;
    if (threadIdx.x == 0) {
        float rms = sqrtf(shared_mem[0] / (float)head_dim + eps);
        scale = 1.0f / rms;
        shared_mem[1] = scale;
    }
    __syncthreads();
    scale = shared_mem[1];

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        slice[i] = slice[i] * scale * weight[i];
    }
}

/* Host-side dispatchers (extern "C" for linkage from .cc) */
extern "C" {

void launch_rms_norm(cudaStream_t stream,
                        float * out,
                        const float * x,
                        const float * weight,
                        int seq_len,
                        int hidden,
                        float eps) {
    int block_size = 256;
    if (block_size > hidden) {
        /* Find largest power of 2 <= hidden */
        block_size = 1;
        while (block_size * 2 <= hidden) block_size *= 2;
    }
    int shmem_bytes = (block_size + 2) * sizeof(float);  /* +2 for sum and scale */
    rms_norm_kernel<<<seq_len, block_size, shmem_bytes, stream>>>(
        out, x, weight, seq_len, hidden, eps);
}

void launch_rms_norm_per_head(cudaStream_t stream,
                                 float * x,
                                 const float * weight,
                                 int seq_len,
                                 int n_heads,
                                 int head_dim,
                                 float eps) {
    int total = seq_len * n_heads;
    int block_size = 256;
    if (block_size > head_dim) {
        block_size = 1;
        while (block_size * 2 <= head_dim) block_size *= 2;
    }
    int shmem_bytes = (block_size + 2) * sizeof(float);  /* +2 for sum and scale */
    rms_norm_per_head_kernel<<<total, block_size, shmem_bytes, stream>>>(
        x, weight, seq_len, n_heads, head_dim, eps);
}

}  // extern "C"
