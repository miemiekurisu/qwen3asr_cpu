/* elementwise.cu — Element-wise CUDA kernels
 *
 * Simple element-wise operations: add, scale, etc.
 */

#include <cuda_runtime.h>

/* Element-wise add: out[i] = a[i] + b[i] */
__global__ void add_kernel(float * out,
                            const float * a,
                            const float * b,
                            int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = a[idx] + b[idx];
}

extern "C" {

void launch_add(cudaStream_t stream,
                 float * out,
                 const float * a,
                 const float * b,
                 int total_elements) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size, 0, stream>>>(
        out, a, b, total_elements);
}

}  // extern "C"
