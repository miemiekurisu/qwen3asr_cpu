/* embedding.cu — Embedding lookup CUDA kernel
 *
 * Performs token embedding lookup:
 *   embeddings[token_id] -> output[seq, hidden]
 *
 * This is needed for decoder prefill where we need to look up
 * embeddings for input tokens.
 */

#include <cuda_runtime.h>
#include "qasr/backend/cuda_decode_params.h"

/* Embedding lookup kernel
 * tokens[seq_len] -> embeddings[seq_len, hidden]
 * W[vocab_size, hidden] */
__global__ void embedding_lookup_kernel(float * embeddings,
                                          const int * tokens,
                                          const float * W,
                                          int seq_len,
                                          int hidden) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    int token_id = tokens[pos];
    const float * emb_row = W + token_id * hidden;
    float * out_row = embeddings + pos * hidden;

    for (int d = 0; d < hidden; d++) {
        out_row[d] = emb_row[d];
    }
}

extern "C" {

void launch_embedding_lookup(cudaStream_t stream,
                               float * embeddings,
                               const int * tokens,
                               const float * W,
                               int seq_len,
                               int hidden) {
    int block_size = 256;
    int grid_size = (seq_len + block_size - 1) / block_size;
    embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
        embeddings, tokens, W, seq_len, hidden);
}

}  // extern "C"
