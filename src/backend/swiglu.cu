/* swiglu.cu — SwiGLU activation CUDA kernel
 *
 * SiLU(g) * u where SiLU(g) = g / (1 + exp(-g))
 *
 * Gate and up are produced by two separate GEMMs:
 *   gate: gate_up[0 : seq_len * intermediate]    (shape [seq_len, intermediate])
 *   up:   gate_up[seq_len*intermediate : 2*seq_len*intermediate]
 *
 * Output: out[seq_len, intermediate] row-major
 *
 * Each thread handles one (row, intermediate_dim) pair.
 */

#include <cuda_runtime.h>

__global__ void swiglu_kernel(float *__restrict__ out,
                                  const float *__restrict__ gate_up,
                                  int seq_len,
                                  int intermediate) {
    int row = blockIdx.y;
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= intermediate || row >= seq_len) return;

    /* Correct layout: gate and up are separate [seq_len, intermediate] blocks */
    const float * gate = gate_up + row * intermediate;
    const float * up = gate_up + seq_len * intermediate + row * intermediate;
    float * row_out = out + row * intermediate;

    float g = gate[si];
    float u = up[si];
    float silu = g / (1.0f + expf(-g));
    row_out[si] = silu * u;
}

extern "C" {

void launch_swiglu(cudaStream_t stream,
                    float * out,
                    const float * gate_up,
                    int seq_len,
                    int intermediate) {
    int threads_per_block = 512;
    dim3 block(threads_per_block);
    dim3 grid((intermediate + block.x - 1) / block.x, seq_len);
    swiglu_kernel<<<grid, block, 0, stream>>>(out, gate_up, seq_len, intermediate);
}

}  // extern "C"
