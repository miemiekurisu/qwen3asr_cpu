/* bidir_attention.cu — Bidirectional windowed attention CUDA kernel
 *
 * Encoder attention: each position attends to ALL positions within its window.
 * Windows partition the sequence: window_starts[w] to window_starts[w+1].
 *
 * Row-major interleaved layout (same as CPU reference):
 *   Q[pos, head, d] = Q[pos * hidden + head * head_dim + d]
 *   K[j, head, d] = K[j * hidden + head * head_dim + d]
 *   V[j, head, d] = V[j * hidden + head * head_dim + d]
 *   out[pos, head, d] = out[pos * hidden + head * head_dim + d]
 *
 * No GQA: encoder uses full attention (n_heads = n_kv_heads).
 * No masking: all positions within a window attend to each other.
 */

#include <cuda_runtime.h>

/* Each thread handles one (head, pos) pair.
 * Iterates over windows, computing attention for positions in each window. */
__global__ void bidir_attention_kernel(float * __restrict__ out,
                                         const float * __restrict__ Q,
                                         const float * __restrict__ K,
                                         const float * __restrict__ V,
                                         int seq_len,
                                         int n_heads,
                                         int head_dim,
                                         float scale,
                                         const int * __restrict__ window_starts,
                                         int n_windows,
                                         int hidden) {
    int head = blockIdx.x;
    int pos = blockIdx.y * blockDim.x + threadIdx.x;

    if (head >= n_heads || pos >= seq_len) return;

    /* Find which window this position belongs to */
    int my_window = -1;
    for (int w = 0; w < n_windows; w++) {
        if (pos >= window_starts[w] && pos < window_starts[w + 1]) {
            my_window = w;
            break;
        }
    }

    if (my_window < 0) return;  /* Position not in any window (shouldn't happen) */

    int ws = window_starts[my_window];
    int we = window_starts[my_window + 1];

    const float * q_row = Q + pos * hidden + head * head_dim;
    float * o_row = out + pos * hidden + head * head_dim;

    /* Online softmax */
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    /* Per-thread partials — head_dim=64 for encoder (stack-allocated) */
    float local_out[64];
    for (int d = 0; d < head_dim; d++) local_out[d] = 0.0f;

    for (int j = ws; j < we; j++) {
        const float * k_row = K + j * hidden + head * head_dim;
        const float * v_row = V + j * hidden + head * head_dim;

        /* Dot product */
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;

        /* Online softmax (numerically stable) */
        float new_max = fmaxf(score, max_score);
        float old_exp = expf(max_score - new_max);
        float score_exp = expf(score - new_max);
        sum_exp = sum_exp * old_exp + score_exp;
        max_score = new_max;

        for (int d = 0; d < head_dim; d++) {
            local_out[d] = local_out[d] * old_exp + score_exp * v_row[d];
        }
    }

    /* Normalize and write output */
    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (int d = 0; d < head_dim; d++) {
        o_row[d] = local_out[d] * inv_sum;
    }
}

extern "C" {

/* out must be zeroed before calling.
 * window_starts: [n_windows + 1] on device or pinned host memory. */
void launch_bidir_attention(cudaStream_t stream,
                              float * out,
                              const float * Q,
                              const float * K,
                              const float * V,
                              int seq_len,
                              int n_heads,
                              int head_dim,
                              float scale,
                              const int * window_starts,
                              int n_windows) {
    int threads_per_block = 1;
    dim3 block(threads_per_block);
    dim3 grid(n_heads, seq_len);
    bidir_attention_kernel<<<grid, block, 0, stream>>>(
        out, Q, K, V, seq_len, n_heads, head_dim, scale,
        window_starts, n_windows, n_heads * head_dim);
}

}  // extern "C"
