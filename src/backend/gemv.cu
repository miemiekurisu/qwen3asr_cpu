/* gemv.cu — Vector-matrix multiplication for DecodeStep (seq_len=1)
 *
 * y[out_dim] = x[in_dim] @ W_T[in_dim, out_dim]
 * W_T is fp32 pre-transposed [in_dim, out_dim] row-major.
 *
 * Optimized version: uses shared memory to tile x, reducing redundant
 * global memory reads from (threads × in_dim) to (block_size × in_dim).
 */

#include <cuda_runtime.h>

/* Shared-memory tiled gemv:
 * Each block loads x[block] into shared memory once, then all threads
 * in the block reuse it for their W_T columns. */
__global__ void gemv_tiled_kernel(float * __restrict__ out,
                                    const float * __restrict__ x,
                                    const float * __restrict__ W_T,
                                    int in_dim, int out_dim) {
    extern __shared__ float shared_x[];

    /* Load x into shared memory (cooperative within block) */
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();

    /* Each thread computes one or more output elements */
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < out_dim; j += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += shared_x[i] * W_T[i * out_dim + j];
        }
        out[j] = sum;
    }
}

/* Original gemv kernel (fallback) */
__global__ void gemv_kernel(float * __restrict__ out,
                               const float * __restrict__ x,
                               const float * __restrict__ W_T,
                               int in_dim, int out_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        sum += x[i] * W_T[i * out_dim + j];
    }
    out[j] = sum;
}

extern "C" {

void launch_gemv(cudaStream_t stream,
                   float * out,
                   const float * x,
                   const float * W_T,
                   int in_dim, int out_dim) {
    int threads = 256;

    /* Use tiled kernel for larger matrices where shared memory helps */
    if (in_dim >= 256 && out_dim >= 512) {
        int blocks = min((out_dim + threads - 1) / threads, 16);
        int shmem = in_dim * sizeof(float);
        gemv_tiled_kernel<<<blocks, threads, shmem, stream>>>(out, x, W_T, in_dim, out_dim);
    } else {
        int blocks = (out_dim + threads - 1) / threads;
        gemv_kernel<<<blocks, threads, 0, stream>>>(out, x, W_T, in_dim, out_dim);
    }
}

}  // extern "C"
