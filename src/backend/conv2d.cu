/* conv2d.cu — 2D convolution kernel for encoder conv2D stem
 *
 * Each thread computes one output element: out[oc * h_out * w_out + oh * w_out + ow]
 * Input: [c_in, h_in, w_in]
 * Weight: [c_out, c_in, kh, kw] (row-major, same as CPU)
 * Bias: [c_out]
 * Stride=2, padding=1, kh=3, kw=3 (Qwen3-ASR encoder conv2D)
 */

#include <cuda_runtime.h>

/* Extract mel chunk: from [mel_bins, mel_frames] extract [mel_bins, chunk_w]
 * at column offset 'start' */
__global__ void extract_mel_chunk_kernel(const float * __restrict__ mel,
                                           float * __restrict__ chunk,
                                           int mel_bins, int mel_frames,
                                           int chunk_w, int start) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= mel_bins || c >= chunk_w) return;
    chunk[m * chunk_w + c] = mel[m * mel_frames + start + c];
}

/* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
__device__ __forceinline__ float gelu_approx(float x) {
    float t = tanhf((0.79788456f) * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

__global__ void conv2d_kernel(const float * __restrict__ in,
                                const float * __restrict__ weight,
                                const float * __restrict__ bias,
                                float * __restrict__ out,
                                int c_in, int c_out,
                                int h_in, int w_in,
                                int h_out, int w_out,
                                int kh, int kw,
                                int stride, int padding,
                                int fused_gelu) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (ow >= w_out || oh >= h_out || oc >= c_out) return;

    float sum = 0.0f;
    for (int ki = 0; ki < kh; ki++) {
        int ih = oh * stride - padding + ki;
        if (ih < 0 || ih >= h_in) continue;
        for (int kj = 0; kj < kw; kj++) {
            int iw = ow * stride - padding + kj;
            if (iw < 0 || iw >= w_in) continue;
            for (int ic = 0; ic < c_in; ic++) {
                sum += weight[oc * c_in * kh * kw + ic * kh * kw + ki * kw + kj] *
                       in[ic * h_in * w_in + ih * w_in + iw];
            }
        }
    }

    if (bias) sum += bias[oc];
    if (fused_gelu) sum = gelu_approx(sum);
    out[oc * h_out * w_out + oh * w_out + ow] = sum;
}

extern "C" {

/* Launch conv2D kernel
 * in:    [c_in, h_in, w_in]
 * weight:[c_out, c_in, kh, kw]
 * bias:  [c_out]
 * out:   [c_out, h_out, w_out]
 * fused_gelu: 1 to apply GELU activation in-place
 */
void launch_conv2d(cudaStream_t stream,
                    const float * in,
                    const float * weight,
                    const float * bias,
                    float * out,
                    int c_in, int c_out,
                    int h_in, int w_in,
                    int kh, int kw,
                    int stride, int padding,
                    int fused_gelu) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;

    dim3 block(8, 8, 1);
    dim3 grid((w_out + block.x - 1) / block.x,
              (h_out + block.y - 1) / block.y,
              c_out);
    conv2d_kernel<<<grid, block, 0, stream>>>(
        in, weight, bias, out,
        c_in, c_out, h_in, w_in, h_out, w_out,
        kh, kw, stride, padding, fused_gelu);
}

/* Launch mel chunk extraction: mel[mel_bins, mel_frames] → chunk[mel_bins, chunk_w] */
void launch_extract_mel_chunk(cudaStream_t stream,
                                float * chunk,
                                const float * mel,
                                int mel_bins, int mel_frames,
                                int chunk_w, int start) {
    dim3 block(16, 8);
    dim3 grid((chunk_w + block.x - 1) / block.x,
              (mel_bins + block.y - 1) / block.y);
    extract_mel_chunk_kernel<<<grid, block, 0, stream>>>(
        mel, chunk, mel_bins, mel_frames, chunk_w, start);
}

}  // extern "C"
