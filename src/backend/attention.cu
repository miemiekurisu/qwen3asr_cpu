/* attention.cu — Causal attention CUDA kernel
 *
 * Row-major interleaved layout (same as CPU reference):
 *   Q[pos, head, d] = Q[pos * q_hidden + head * head_dim + d]
 *   K[j, kv_head, d] = K[j * kv_hidden + kv_head * head_dim + d]
 *   V[j, kv_head, d] = V[j * kv_hidden + kv_head * head_dim + d]
 *   out[pos, head, d] = out[pos * q_hidden + head * head_dim + d]
 *
 * GQA: kv_head = head / (n_heads / n_kv_heads)
 */

#include <cuda_runtime.h>
#include "qasr/backend/cuda_decode_params.h"

__global__ void causal_attention_kernel(float * __restrict__ out,
                                            const float * __restrict__ Q,
                                            const float * __restrict__ K,
                                            const float * __restrict__ V,
                                            int seq_q,
                                            int seq_k,
                                            int n_heads,
                                            int n_kv_heads,
                                            int head_dim,
                                            float scale,
                                            int q_offset) {
    int head = blockIdx.x;
    int pos = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= n_heads || pos >= seq_q || tid >= head_dim) return;

    int heads_per_kv = n_heads / n_kv_heads;
    int kv_head = head / heads_per_kv;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    const float * q_row = Q + pos * q_hidden + head * head_dim;
    float * o_row = out + pos * q_hidden + head * head_dim;

    int global_pos = q_offset + pos;
    int k_end = global_pos + 1;
    if (k_end > seq_k) k_end = seq_k;

    extern __shared__ float shared_scratch[];
    int n_warps = (blockDim.x + 31) / 32;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float local_out = 0.0f;

    int warp_id = tid >> 5;
    int lane = tid & 0x1f;
    unsigned full_mask = 0xffffffff;

    for (int j = 0; j < k_end; j++) {
        const float * k_row = K + j * kv_hidden + kv_head * head_dim;
        const float * v_row = V + j * kv_hidden + kv_head * head_dim;

        /* Dot product: each thread contributes one element */
        float score = q_row[tid] * k_row[tid];
        for (int mask = 16; mask > 0; mask >>= 1)
            score += __shfl_xor_sync(full_mask, score, mask);
        /* lane 0 of each warp has partial sum; cross-warp reduction */
        if (lane == 0) shared_scratch[warp_id] = score;
        __syncthreads();
        if (warp_id == 0) {
            float t = (lane < n_warps) ? shared_scratch[lane] : 0.0f;
            for (int mask = 16; mask > 0; mask >>= 1)
                t += __shfl_xor_sync(full_mask, t, mask);
            if (lane == 0) shared_scratch[0] = t * scale;
        }
        __syncthreads();
        score = shared_scratch[0];

        /* Online softmax — all threads have same score */
        float new_max = fmaxf(score, max_score);
        float old_exp = expf(max_score - new_max);
        float score_exp = expf(score - new_max);
        sum_exp = sum_exp * old_exp + score_exp;
        max_score = new_max;

        /* Each thread accumulates its element of output */
        local_out = local_out * old_exp + score_exp * v_row[tid];
    }

    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    o_row[tid] = local_out * inv_sum;
}

/* Graph-compatible causal attention: reads seq_pos from d_params.
 * For seq_q=1 (DecodeStep), n_heads blocks, 1 thread each.
 * global_pos = seq_pos, k_end = seq_pos + 1 */
__global__ void causal_attention_graph_kernel(float * __restrict__ out,
                                                const float * __restrict__ Q,
                                                const float * __restrict__ K,
                                                const float * __restrict__ V,
                                                CudaDecodeParams *params,
                                                int n_heads,
                                                int n_kv_heads,
                                                int head_dim,
                                                float scale) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int heads_per_kv = n_heads / n_kv_heads;
    int kv_head = head / heads_per_kv;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    int seq_pos = params->seq_pos;
    int k_end = seq_pos + 1;

    const float * q_row = Q + head * head_dim;  /* seq_q=1, pos=0 */
    float * o_row = out + head * head_dim;

    /* Online softmax */
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    float local_out[128];
    for (int d = 0; d < head_dim; d++) local_out[d] = 0.0f;

    for (int j = 0; j < k_end; j++) {
        const float * k_row = K + j * kv_hidden + kv_head * head_dim;
        const float * v_row = V + j * kv_hidden + kv_head * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;

        float new_max = fmaxf(score, max_score);
        float old_exp = expf(max_score - new_max);
        float score_exp = expf(score - new_max);
        sum_exp = sum_exp * old_exp + score_exp;
        max_score = new_max;

        for (int d = 0; d < head_dim; d++) {
            local_out[d] = local_out[d] * old_exp + score_exp * v_row[d];
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (int d = 0; d < head_dim; d++) {
        o_row[d] = local_out[d] * inv_sum;
    }
}

extern "C" {

void launch_causal_attention(cudaStream_t stream,
                                float * out,
                                const float * Q,
                                const float * K,
                                const float * V,
                                int seq_q,
                                int seq_k,
                                int n_heads,
                                int n_kv_heads,
                                int head_dim,
                                float scale,
                                int q_offset) {
    /* head_dim threads per block (parallel dot product), one block per (head, pos) */
    int n_warps = (head_dim + 31) / 32;
    int shmem = n_warps * sizeof(float);  /* cross-warp reduction */
    dim3 block(head_dim);
    dim3 grid(n_heads, seq_q);
    causal_attention_kernel<<<grid, block, shmem, stream>>>(
        out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, q_offset);
}

void launch_causal_attention_graph(cudaStream_t stream,
                                     float * out,
                                     const float * Q,
                                     const float * K,
                                     const float * V,
                                     struct CudaDecodeParams *params,
                                     int n_heads,
                                     int n_kv_heads,
                                     int head_dim,
                                     float scale) {
    /* One thread per head; seq_q=1, pos=0 */
    causal_attention_graph_kernel<<<n_heads, 1, 0, stream>>>(
        out, Q, K, V, params, n_heads, n_kv_heads, head_dim, scale);
}

}  // extern "C"
