/* cuda_graph_decode.cu — CUDA Graph support for DecodeStep replay
 *
 * Two helper kernels to replace cudaMemcpyAsync in the graph:
 *   1. embed_lookup_from_token: reads prev_token from device variable,
 *      then copies the corresponding embedding row to workspace
 *   2. update_seq_pos: increments d_seq_pos by 1 (for rope offset)
 *
 * Usage:
 *   - Allocate device variables d_prev_token (int) and d_seq_pos (int)
 *   - Before recording: cudaMemcpy them from host
 *   - Record DecodeStep as CUDA graph
 *   - Between replays: cudaMemcpyAsync new token + increment seq_pos
 *   - Replay graph
 */

#include <cuda_runtime.h>

/* Embedding lookup: read token from device variable, copy embedding row to output.
 * d_token points to a single int32 on device.
 * W is tok_embeddings_fp32: [vocab_size, hidden], row-major.
 * out[hidden] = W[token * hidden : (token+1)*hidden] */
__global__ void embed_lookup_from_token_kernel(
    float * __restrict__ out,
    const float * __restrict__ W,
    const int * __restrict__ d_token,
    int hidden)
{
    int token = *d_token;
    const float * row = W + (size_t)token * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out[i] = row[i];
    }
}

/* Increment device-side sequence position counter (for rope offset).
 * Atomically adds delta to *d_pos. */
__global__ void increment_seq_pos_kernel(int * __restrict__ d_pos, int delta)
{
    atomicAdd(d_pos, delta);
}

extern "C" {

void launch_embed_lookup_from_token(cudaStream_t stream,
                                      float * out,
                                      const float * W,
                                      const int * d_token,
                                      int hidden)
{
    embed_lookup_from_token_kernel<<<1, 256, 0, stream>>>(out, W, d_token, hidden);
}

void launch_increment_seq_pos(cudaStream_t stream,
                                int * d_pos,
                                int delta)
{
    increment_seq_pos_kernel<<<1, 1, 0, stream>>>(d_pos, delta);
}

}  // extern "C"
