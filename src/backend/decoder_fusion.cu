/* decoder_fusion.cu — Fused kernels for decoder DecodeStep (seq_len=1)
 *
 * Reduces per-layer kernel launches from 19 to ~4:
 *   1. rmsnorm_qkv_rope: RMSNorm + QKV projection + per-head RMSNorm + RoPE
 *   2. attention: causal attention
 *   3. wo_residual: WO projection + residual add
 *   4. ffn: post-attn RMSNorm + gate/up + SwiGLU + down + residual
 *
 * Row-major layout, fp32 throughout.
 */

#include <cuda_runtime.h>
#include "qasr/backend/cuda_decode_params.h"

/* ======================================================================
 * Kernel 1: RMSNorm + QKV projection + per-head RMSNorm + RoPE
 *
 * Grid: n_heads blocks (20 for 0.6B, 14 for 1.7B)
 * Block: 256 threads
 *
 * Shared memory layout (per block):
 *   offset 0          : x_norm[hidden]         (RMSNorm output)
 *   offset hidden     : q_head[head_dim]       (Q per-head values)
 *   offset hidden+HD  : k_head[head_dim]       (K per-head values)
 *   offset hidden+2*HD: v_head[head_dim]       (V per-head values)
 *   offset hidden+3*HD: red[258]               (reduction temp)
 *   offset hidden+3*HD+258: q_red[258]         (Q per-head norm)
 *   offset hidden+3*HD+516: k_red[258]         (K per-head norm)
 *
 * Total: (hidden + 3*head_dim + 774) * sizeof(float)
 *        ≈ (1024 + 156 + 774) * 4 = 7.7KB per block
 * ====================================================================== */

__global__ void rmsnorm_qkv_rope_kernel(
    float * __restrict__ Q,
    float * __restrict__ K,
    float * __restrict__ V,
    const float * __restrict__ x,
    const float * __restrict__ wq_T,
    const float * __restrict__ wk_T,
    const float * __restrict__ wv_T,
    const float * __restrict__ input_norm_w,
    const float * __restrict__ q_norm_w,
    const float * __restrict__ k_norm_w,
    const float * __restrict__ rope_cos,
    const float * __restrict__ rope_sin,
    CudaDecodeParams *params,
    int hidden,
    int q_dim,
    int kv_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float eps) {

    extern __shared__ float shared_mem[];

    float *xn  = shared_mem;
    float *qh  = shared_mem + hidden;
    float *kh  = shared_mem + hidden + head_dim;
    float *vh  = shared_mem + hidden + 2 * head_dim;
    float *red = shared_mem + hidden + 3 * head_dim;
    float *qrd = shared_mem + hidden + 3 * head_dim + 258;
    float *krd = shared_mem + hidden + 3 * head_dim + 516;

    /* ---- Step 1: RMSNorm of x → xn ---- */
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }
    red[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    float scale;
    if (threadIdx.x == 0) red[0] = 1.0f / sqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    scale = red[0];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        xn[i] = x[i] * scale * input_norm_w[i];
    }
    __syncthreads();

    /* ---- Step 2: QKV projection for this head ---- */
    int head = blockIdx.x;
    int heads_per_kv = n_heads / n_kv_heads;
    int kv_head = head / heads_per_kv;
    int q_off = head * head_dim;
    int kv_off = kv_head * head_dim;

    if (threadIdx.x < head_dim) {
        float qv = 0.0f, kv = 0.0f, vv = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float xn_i = xn[i];
            qv += xn_i * wq_T[i * q_dim + q_off + threadIdx.x];
            kv += xn_i * wk_T[i * kv_dim + kv_off + threadIdx.x];
            vv += xn_i * wv_T[i * kv_dim + kv_off + threadIdx.x];
        }
        qh[threadIdx.x] = qv;
        kh[threadIdx.x] = kv;
        vh[threadIdx.x] = vv;
    }
    __syncthreads();

    /* ---- Step 3: Per-head RMSNorm for Q ---- */
    sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum_sq += qh[i] * qh[i];
    qrd[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) qrd[threadIdx.x] += qrd[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) qrd[0] = 1.0f / sqrtf(qrd[0] / (float)head_dim + eps);
    __syncthreads();
    scale = qrd[0];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        qh[i] *= scale * q_norm_w[i];
    __syncthreads();

    /* ---- Step 4: Per-head RMSNorm for K ---- */
    sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum_sq += kh[i] * kh[i];
    krd[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) krd[threadIdx.x] += krd[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) krd[0] = 1.0f / sqrtf(krd[0] / (float)head_dim + eps);
    __syncthreads();
    scale = krd[0];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        kh[i] *= scale * k_norm_w[i];
    __syncthreads();

    /* ---- Step 5: RoPE + write output ---- */
    int half = head_dim / 2;
    if (threadIdx.x < head_dim) {
        int d = threadIdx.x;
        float c = rope_cos[d];
        float s = rope_sin[d];

        /* Q with RoPE (NeoX split-half, only d<half threads apply rotation) */
        if (d < half) {
            float x1 = qh[d], x2 = qh[half + d];
            Q[q_off + d] = x1 * c - x2 * s;
            Q[q_off + half + d] = x2 * c + x1 * s;
        }

        /* K with RoPE */
        if (d < half) {
            float x1 = kh[d], x2 = kh[half + d];
            K[kv_off + d] = x1 * c - x2 * s;
            K[kv_off + half + d] = x2 * c + x1 * s;
        }

        /* V (no RoPE) */
        V[kv_off + d] = vh[d];
    }
}

/* ======================================================================
 * Kernel 2: WO projection + residual add
 *
 * x[hidden] += attn_out[q_dim] @ wo_T[q_dim, hidden]
 *
 * Grid: (hidden + 255) / 256 blocks
 * Shared memory: attn_out[q_dim] loaded once, reused by all threads
 * ====================================================================== */
__global__ void wo_residual_kernel(
    float * __restrict__ x,
    const float * __restrict__ attn_out,
    const float * __restrict__ wo_T,
    int q_dim,
    int hidden) {

    extern __shared__ float shared_attn[];

    /* Load attn_out into shared memory */
    for (int i = threadIdx.x; i < q_dim; i += blockDim.x)
        shared_attn[i] = attn_out[i];
    __syncthreads();

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < q_dim; i++)
            sum += shared_attn[i] * wo_T[i * hidden + j];
        x[j] += sum;
    }
}

/* ======================================================================
 * Kernel 3: Post-attn RMSNorm + gate/up + SwiGLU + down + residual
 *
 * Grid: 16 blocks × 256 threads = 4096 (covers intermediate=4096)
 *
 * Shared memory:
 *   s_post[hidden]       (4KB)
 *   s_ffn[intermediate]  (16KB)
 *   s_red[260]           (1KB)
 * Total: 21KB per block
 *
 * Phase 1: RMSNorm post_norm → s_post (all blocks cooperate via block 0)
 * Phase 2: gate/up projection → SwiGLU → s_ffn
 * Phase 3: down projection + residual add
 * ====================================================================== */
__global__ void ffn_kernel(
    float * __restrict__ x,
    const float * __restrict__ post_norm,
    const float * __restrict__ gate_T,
    const float * __restrict__ up_T,
    const float * __restrict__ down_T,
    const float * __restrict__ post_attn_norm_w,
    int hidden,
    int intermediate,
    float eps) {

    extern __shared__ float shared_mem[];
    float *s_post = shared_mem;
    float *s_ffn = s_post + hidden;
    float *s_red = s_ffn + intermediate;

    /* ---- Phase 1: Post-attn RMSNorm ---- */
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        sum_sq += post_norm[i] * post_norm[i];
    s_red[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = 0.0f;
    if (threadIdx.x == 0) scale = 1.0f / sqrtf(s_red[0] / (float)hidden + eps);
    s_red[0] = scale;
    __syncthreads();
    scale = s_red[0];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        s_post[i] = post_norm[i] * scale * post_attn_norm_w[i];
    __syncthreads();

    /* ---- Phase 2: gate/up + SwiGLU (single block, 256 threads cover all) ---- */
    for (int si = threadIdx.x; si < intermediate; si += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        for (int i = 0; i < hidden; i++) {
            g += s_post[i] * gate_T[i * intermediate + si];
            u += s_post[i] * up_T[i * intermediate + si];
        }
        s_ffn[si] = (g / (1.0f + expf(-g))) * u;
    }
    __syncthreads();

    /* ---- Phase 3: down projection + residual (single block) ---- */
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float down_val = 0.0f;
        for (int si = 0; si < intermediate; si++)
            down_val += s_ffn[si] * down_T[si * hidden + j];
        x[j] += down_val;
    }
}

extern "C" {

void launch_rmsnorm_qkv_rope(cudaStream_t stream,
                               float * Q,
                               float * K,
                               float * V,
                               const float * x,
                               const float * wq_T,
                               const float * wk_T,
                               const float * wv_T,
                               const float * input_norm_w,
                               const float * q_norm_w,
                               const float * k_norm_w,
                               const float * rope_cos,
                               const float * rope_sin,
                               CudaDecodeParams *params,
                               int hidden,
                               int q_dim,
                               int kv_dim,
                               int n_heads,
                               int n_kv_heads,
                               int head_dim,
                               float eps) {
    int shmem = (hidden + 3 * head_dim + 774) * sizeof(float);
    rmsnorm_qkv_rope_kernel<<<n_heads, 256, shmem, stream>>>(
        Q, K, V, x, wq_T, wk_T, wv_T,
        input_norm_w, q_norm_w, k_norm_w,
        rope_cos, rope_sin, params,
        hidden, q_dim, kv_dim, n_heads, n_kv_heads, head_dim, eps);
}

void launch_wo_residual(cudaStream_t stream,
                          float * x,
                          const float * attn_out,
                          const float * wo_T,
                          int q_dim,
                          int hidden) {
    int shmem = q_dim * sizeof(float);
    int blocks = (hidden + 255) / 256;
    wo_residual_kernel<<<blocks, 256, shmem, stream>>>(
        x, attn_out, wo_T, q_dim, hidden);
}

void launch_ffn(cudaStream_t stream,
                  float * x,
                  const float * post_norm,
                  const float * gate_T,
                  const float * up_T,
                  const float * down_T,
                  const float * post_attn_norm_w,
                  int hidden,
                  int intermediate,
                  float eps) {
    int shmem = (hidden + intermediate + 260) * sizeof(float);
    ffn_kernel<<<1, 256, shmem, stream>>>(
        x, post_norm, gate_T, up_T, down_T, post_attn_norm_w,
        hidden, intermediate, eps);
}

}  // extern "C"
