/* sinusoidal_pe.cu — Sinusoidal positional encoding kernel
 *
 * Matches CPU qwen_asr_kernels.c: qwen_sinusoidal_pe(PE, seq_len, d_model)
 * Layout: first half = sin, second half = cos
 * Formula: inv_timescale = exp(-d * log(10000) / (half-1)), angle = pos * inv_timescale
 */

#include <cuda_runtime.h>

__global__ void sinusoidal_pe_kernel(float * __restrict__ pe,
                                       int seq_len, int d_model) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);
    float *row = pe + pos * d_model;

    for (int d = 0; d < half; d++) {
        float inv_timescale = expf(-(float)d * log_timescale);
        float angle = (float)pos * inv_timescale;
        row[d]          = sinf(angle);   /* first half: sin */
        row[half + d]   = cosf(angle);   /* second half: cos */
    }
}

extern "C" {

/* Launch sinusoidal PE generation
 * pe: [seq_len, d_model] (output)
 */
void launch_sinusoidal_pe(cudaStream_t stream,
                           float * pe,
                           int seq_len, int d_model) {
    int block_size = 256;
    int grid_size = (seq_len + block_size - 1) / block_size;
    sinusoidal_pe_kernel<<<grid_size, block_size, 0, stream>>>(pe, seq_len, d_model);
}

}  // extern "C"
