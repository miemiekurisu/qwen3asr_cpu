/* rope.cu — RoPE (Rotary Position Embedding) CUDA kernel
 *
 * NeoX split-half RoPE: first half and second half of each head are rotated.
 * cos_vals / sin_vals are per-sequence-position, per-head-dimension.
 *
 * Algorithm derived from qwen_asr_kernels.c:qwen_apply_rope_neox
 */

#include <cuda_runtime.h>
#include "qasr/backend/cuda_decode_params.h"

/* NeoX split-half RoPE: x[seq, n_heads * head_dim] in-place
 * cos_vals[seq, head_dim], sin_vals[seq, head_dim]
 *
 * For each position p, each head h:
 *   x1 = x[h*head_dim : h*head_dim + half]
 *   x2 = x[h*head_dim + half : h*head_dim + head_dim]
 *   cos_p = cos_vals[p, :]
 *   sin_p = sin_vals[p, :]
 *   out[d]        = x1[d] * cos_p[d] - x2[d] * sin_p[d]
 *   out[half+d]   = x2[d] * cos_p[d] + x1[d] * sin_p[d]
 *   (cos/sin repeated for full head_dim) */
__global__ void rope_neox_kernel(float *__restrict__ x,
                                   const float *__restrict__ cos_vals,
                                   const float *__restrict__ sin_vals,
                                   int seq,
                                   int n_heads,
                                   int head_dim) {
    int p = blockIdx.x;
    if (p >= seq) return;

    int tid = threadIdx.x;
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float * head_ptr = x + (p * n_heads + h) * head_dim;
        const float * cos_p = cos_vals + p * head_dim;
        const float * sin_p = sin_vals + p * head_dim;

        for (int d = tid; d < half; d += blockDim.x) {
            float x1 = head_ptr[d];
            float x2 = head_ptr[half + d];
            float c = cos_p[d];
            float s = sin_p[d];
            head_ptr[d]       = x1 * c - x2 * s;
            head_ptr[half + d] = x2 * c + x1 * s;
        }
    }
}

/* Graph-compatible RoPE kernel: reads seq_pos from d_params struct.
 * rope_cos_base / rope_sin_base are the base pointers (seq 0).
 * The kernel computes the offset: cos_vals + seq_pos * head_dim */
__global__ void rope_neox_graph_kernel(float *__restrict__ x,
                                         const float *__restrict__ rope_cos_base,
                                         const float *__restrict__ rope_sin_base,
                                         CudaDecodeParams *params,
                                         int n_heads,
                                         int head_dim) {
    /* seq=1, position = seq_pos from d_params */
    int tid = threadIdx.x;
    int half = head_dim / 2;

    int seq_pos = params->seq_pos;
    const float * cos_p = rope_cos_base + seq_pos * head_dim;
    const float * sin_p = rope_sin_base + seq_pos * head_dim;

    for (int h = tid; h < n_heads; h += blockDim.x) {
        float * head_ptr = x + h * head_dim;

        for (int d = 0; d < half; d++) {
            float x1 = head_ptr[d];
            float x2 = head_ptr[half + d];
            float c = cos_p[d];
            float s = sin_p[d];
            head_ptr[d]       = x1 * c - x2 * s;
            head_ptr[half + d] = x2 * c + x1 * s;
        }
    }
}

/* Build RoPE cos/sin cache on GPU:
 * rope_inv_freq[head_dim/2] -> cos_vals[seq, head_dim], sin_vals[seq, head_dim]
 * cos[d] = cos(pos * inv_freq[d]), repeated for [d] and [half+d] */
__global__ void build_rope_cache_kernel(const float *__restrict__ inv_freq,
                                         float *__restrict__ cos_vals,
                                         float *__restrict__ sin_vals,
                                         int seq,
                                         int head_dim) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= seq) return;

    int half = head_dim / 2;
    for (int d = 0; d < half; d++) {
        float val = p * inv_freq[d];
        float c = cosf(val);
        float s = sinf(val);
        cos_vals[p * head_dim + d]       = c;
        cos_vals[p * head_dim + half + d] = c;
        sin_vals[p * head_dim + d]       = s;
        sin_vals[p * head_dim + half + d] = s;
    }
}

extern "C" {

void launch_rope_neox(cudaStream_t stream,
                        float * x,
                        const float * cos_vals,
                        const float * sin_vals,
                        int seq,
                        int n_heads,
                        int head_dim) {
    /* One block per sequence position */
    dim3 block(256);
    dim3 grid(seq);
    rope_neox_kernel<<<grid, block, 0, stream>>>(
        x, cos_vals, sin_vals, seq, n_heads, head_dim);
}

void launch_rope_neox_graph(cudaStream_t stream,
                              float * x,
                              const float * rope_cos_base,
                              const float * rope_sin_base,
                              struct CudaDecodeParams *params,
                              int n_heads,
                              int head_dim) {
    /* One block, threads = n_heads (20 for 0.6B, 14 for 1.7B) */
    rope_neox_graph_kernel<<<1, 256, 0, stream>>>(
        x, rope_cos_base, rope_sin_base, params, n_heads, head_dim);
}

void launch_build_rope_cache(cudaStream_t stream,
                              const float * inv_freq,
                              float * cos_vals,
                              float * sin_vals,
                              int seq,
                              int head_dim) {
    dim3 block(256);
    dim3 grid((seq + block.x - 1) / block.x);
    build_rope_cache_kernel<<<grid, block, 0, stream>>>(
        inv_freq, cos_vals, sin_vals, seq, head_dim);
}

}  // extern "C"
