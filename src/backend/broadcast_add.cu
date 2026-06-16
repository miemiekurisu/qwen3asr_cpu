/* broadcast_add.cu — Add a bias vector to each row of a matrix
 *
 * out[seq, hidden] = matrix[seq, hidden] + bias[hidden]
 * bias is broadcast across all seq positions.
 */

#include <cuda_runtime.h>

__global__ void broadcast_add_kernel(float * __restrict__ out,
                                        const float * __restrict__ matrix,
                                        const float * __restrict__ bias,
                                        int seq_len, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden;
    if (i >= total) return;

    int row = i / hidden;
    int col = i % hidden;
    out[i] = matrix[i] + bias[col];
}

extern "C" {

void launch_broadcast_add(cudaStream_t stream,
                            float * out,
                            const float * matrix,
                            const float * bias,
                            int seq_len, int hidden) {
    int total = seq_len * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_add_kernel<<<blocks, threads, 0, stream>>>(
        out, matrix, bias, seq_len, hidden);
}

}  // extern "C"
