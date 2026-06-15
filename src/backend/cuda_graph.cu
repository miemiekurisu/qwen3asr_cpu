/* cuda_graph.cu — CUDA Graph infrastructure for DecodeStep replay
 *
 * Captures the full DecodeStep compute graph once, then replays each
 * decode step with updated device-side parameters (prev_token, seq_pos).
 *
 * Design: all dynamic kernel arguments are read from a shared d_params
 * structure on device, so the graph can be captured once and never updated.
 */

#include <cuda_runtime.h>

/* Device-side dynamic parameters — all kernels read from here */
struct CudaDecodeParams {
    int prev_token;   /* previous token ID for embedding lookup */
    int seq_pos;      /* current sequence position (0-based) */
};

/* ------------------------------------------------------------------ */
/* Write prev_token to d_params */
/* ------------------------------------------------------------------ */
__global__ void write_prev_token_kernel(CudaDecodeParams *params, int prev_token) {
    params->prev_token = prev_token;
}

/* ------------------------------------------------------------------ */
/* Write seq_pos to d_params */
/* ------------------------------------------------------------------ */
__global__ void write_seq_pos_kernel(CudaDecodeParams *params, int seq_pos) {
    params->seq_pos = seq_pos;
}

/* ------------------------------------------------------------------ */
/* Embedding lookup from prev_token (replaces cudaMemcpyAsync)
 *
 * Reads prev_token from d_params, copies tok_embeddings[prev_token*hidden:]
 * to output.
 * ------------------------------------------------------------------ */
__global__ void embed_lookup_from_token_kernel(float *out,
                                                 const float *tok_embeddings,
                                                 CudaDecodeParams *params,
                                                 int hidden) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden) return;

    int token = params->prev_token;
    const float *emb_row = tok_embeddings + (size_t)token * hidden;
    out[tid] = emb_row[tid];
}

/* ------------------------------------------------------------------ */
/* KV cache store kernel (replaces cudaMemcpyAsync)
 *
 * Reads seq_pos from d_params, stores K and V at the correct offset
 * within the layer's KV cache slice.
 * ------------------------------------------------------------------ */
__global__ void kv_cache_store_kernel(const float *k_src,
                                        const float *v_src,
                                        float *kv_cache_k,
                                        float *kv_cache_v,
                                        CudaDecodeParams *params,
                                        int kv_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_dim) return;

    int pos = params->seq_pos;
    kv_cache_k[pos * kv_dim + tid] = k_src[tid];
    kv_cache_v[pos * kv_dim + tid] = v_src[tid];
}

/* ------------------------------------------------------------------ */
/* Host-side launchers (extern "C" for linkage from .cc) */
/* ------------------------------------------------------------------ */
extern "C" {

void launch_write_prev_token(cudaStream_t stream,
                               CudaDecodeParams *params,
                               int prev_token) {
    write_prev_token_kernel<<<1, 1, 0, stream>>>(params, prev_token);
}

void launch_write_seq_pos(cudaStream_t stream,
                            CudaDecodeParams *params,
                            int seq_pos) {
    write_seq_pos_kernel<<<1, 1, 0, stream>>>(params, seq_pos);
}

void launch_embed_lookup_from_token(cudaStream_t stream,
                                      float *out,
                                      const float *tok_embeddings,
                                      CudaDecodeParams *params,
                                      int hidden) {
    int block = 256;
    int grid = (hidden + block - 1) / block;
    embed_lookup_from_token_kernel<<<grid, block, 0, stream>>>(
        out, tok_embeddings, params, hidden);
}

void launch_kv_cache_store(cudaStream_t stream,
                             const float *k_src,
                             const float *v_src,
                             float *kv_cache_k,
                             float *kv_cache_v,
                             CudaDecodeParams *params,
                             int kv_dim) {
    int block = 256;
    int grid = (kv_dim + block - 1) / block;
    kv_cache_store_kernel<<<grid, block, 0, stream>>>(
        k_src, v_src, kv_cache_k, kv_cache_v, params, kv_dim);
}

}  // extern "C"
