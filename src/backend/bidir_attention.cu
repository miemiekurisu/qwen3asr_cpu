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
#include <cstdlib>

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
 * window_starts: [n_windows + 1] on device or pinned host memory.
 *
 * ============================================================================
 * THREAD BLOCK CONFIGURATION - NVIDIA CUDA Best Practices
 * ============================================================================
 *
 * According to CUDA C++ Best Practices Guide (Section 11.3, Thread and Block Heuristics):
 *
 *   "Between 128 and 256 threads per block is a good initial range for 
 *    experimentation with different block sizes."
 *
 *   "The number of threads per block should be a multiple of 32 threads, 
 *    because this provides optimal computing efficiency and facilitates coalescing."
 *
 *   "A minimum of 64 threads per block should be used, and only if there are
 *    multiple concurrent blocks per multiprocessor."
 *
 * Configuration:
 *   - Default: 256 threads per block (within NVIDIA's recommended 128-256 range)
 *   - Must be multiple of 32 (warp size) for optimal efficiency
 *   - Environment variable QASR_ATTENTION_THREADS can override (must be 32-1024)
 *
 * The kernel layout (blockIdx.y * blockDim.x + threadIdx.x) already supports
 * multiple threads per block. Each thread handles one (head, pos) pair independently.
 * Stack allocation (local_out[64]) is safe for head_dim<=64 (encoder uses 64).
 *
 * This configuration follows NVIDIA's official best practices for all platforms
 * (DGX Spark, RTX 3070/4090, etc.). No platform-specific exceptions needed. */
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
    /* Default: 256 threads per block (NVIDIA recommended range: 128-256) */
    int threads_per_block = 256;
    
    /* Allow runtime override via environment variable for benchmarking */
    const char * env_threads = std::getenv("QASR_ATTENTION_THREADS");
    if (env_threads) {
        int override = std::atoi(env_threads);
        /* Validate: must be multiple of 32 (warp size), range 32-1024 */
        if (override >= 32 && override <= 1024 && (override % 32) == 0) {
            threads_per_block = override;
        }
    }

    /* Final validation: must be multiple of 32, range 32-1024 */
    if (threads_per_block < 32 || threads_per_block > 1024 ||
        (threads_per_block % 32) != 0) {
        threads_per_block = 256;  /* Fallback to NVIDIA-recommended default */
    }

    dim3 block(threads_per_block);
    dim3 grid(n_heads, seq_len);
    bidir_attention_kernel<<<grid, block, 0, stream>>>(
        out, Q, K, V, seq_len, n_heads, head_dim, scale,
        window_starts, n_windows, n_heads * head_dim);
}

}  // extern "C"
