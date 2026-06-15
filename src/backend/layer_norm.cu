/* layer_norm.cu — LayerNorm CUDA kernel (encoder uses LayerNorm, not RMSNorm)
 *
 * Row-major interleaved layout:
 *   x[seq, hidden], weight[hidden], bias[hidden]
 *   out[i * hidden + j] = (x[i * hidden + j] - mean[i]) / std[i] * weight[j] + bias[j]
 *
 * Two-pass LayerNorm:
 *   Pass 1: compute mean and std per row
 *   Pass 2: normalize and apply weight/bias
 */

#include <cuda_runtime.h>

__global__ void layer_norm_forward_kernel(float * __restrict__ out,
                                            const float * __restrict__ x,
                                            const float * __restrict__ weight,
                                            const float * __restrict__ bias,
                                            int seq_len, int hidden, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    const float * x_row = x + row * hidden;
    float * o_row = out + row * hidden;

    /* Pass 1: compute mean */
    float mean = 0.0f;
    for (int j = 0; j < hidden; j++) {
        mean += x_row[j];
    }
    mean /= (float)hidden;

    /* Pass 2: compute variance and normalize */
    float var = 0.0f;
    for (int j = 0; j < hidden; j++) {
        float diff = x_row[j] - mean;
        var += diff * diff;
    }
    var /= (float)hidden;

    float inv_std = rsqrtf(var + eps);

    for (int j = 0; j < hidden; j++) {
        o_row[j] = (x_row[j] - mean) * inv_std * weight[j] + bias[j];
    }
}

extern "C" {

void launch_layer_norm(cudaStream_t stream,
                        float * out,
                        const float * x,
                        const float * weight,
                        const float * bias,
                        int seq_len, int hidden, float eps) {
    int threads = 256;
    int blocks = (seq_len + threads - 1) / threads;
    layer_norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        out, x, weight, bias, seq_len, hidden, eps);
}

}  // extern "C"
