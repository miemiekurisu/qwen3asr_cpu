/* gelu.cu — GELU activation CUDA kernel
 *
 * GELU approximation matching PyTorch/CPU:
 *   x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */

#include <cuda_runtime.h>

#ifndef M_2_PI
#define M_2_PI 0.63661977236758134307553505349005f
#endif

__global__ void gelu_kernel(float * __restrict__ out,
                              const float * __restrict__ x,
                              int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    float val = x[i];
    /* GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
    out[i] = val * cdf;
}

extern "C" {

void launch_gelu(cudaStream_t stream,
                  float * out,
                  const float * x,
                  int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(out, x, total_elements);
}

}  // extern "C"
