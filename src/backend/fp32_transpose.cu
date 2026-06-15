/* fp32_transpose.cu — fp32 matrix transpose for encoder weights
 *
 * Encoder weights are loaded as fp32 on GPU (via load_bf16_as_f32).
 * This kernel transposes W[out_dim, in_dim] → W_T[in_dim, out_dim]
 * for use with cuBLAS GEMM.
 */

#include <cuda_runtime.h>

__global__ void fp32_transpose_kernel(float * __restrict__ W_T,
                                         const float * __restrict__ W,
                                         int out_dim, int in_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim || j >= in_dim) return;
    W_T[j * out_dim + i] = W[i * in_dim + j];
}

extern "C" {

void launch_fp32_transpose(cudaStream_t stream,
                             float * W_T, const float * W,
                             int out_dim, int in_dim) {
    dim3 block(16, 16);
    dim3 grid((in_dim + block.x - 1) / block.x,
              (out_dim + block.y - 1) / block.y);
    fp32_transpose_kernel<<<grid, block, 0, stream>>>(W_T, W, out_dim, in_dim);
}

}  // extern "C"
