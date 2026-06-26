#include "qasr/backend/cuda_backend.h"
#include <cstdio>
#include <cmath>
#include <vector>

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
}

#ifdef QASR_CUDA_BACKEND_ENABLED
/* CUDA kernel launchers (extern "C" from .cu files) */
extern "C" {
void launch_rms_norm(cudaStream_t stream,
                      float * out, const float * x, const float * weight,
                      int seq_len, int hidden, float eps);
void launch_rms_norm_per_head(cudaStream_t stream,
                               float * x, const float * weight,
                               int seq_len, int n_heads, int head_dim, float eps);
void launch_rope_neox(cudaStream_t stream,
                       float * x, const float * cos_vals, const float * sin_vals,
                       int seq, int n_heads, int head_dim);
void launch_swiglu(cudaStream_t stream,
                    float * out, const float * gate_up,
                    int seq_len, int intermediate);
void launch_argmax(cudaStream_t stream,
                    const float * logits, int vocab_size,
                    float * out_val, int * out_idx);
void launch_bf16_to_fp32(cudaStream_t stream,
                          float * out, const uint16_t * in,
                          int total_elements);
void launch_fp32_to_bf16(cudaStream_t stream,
                          uint16_t * out, const float * in,
                          int total_elements);
void launch_add(cudaStream_t stream,
                 float * out, const float * a, const float * b,
                 int total_elements);
void launch_causal_attention(cudaStream_t stream,
                               float * out,
                               const float * Q, const float * K, const float * V,
                               int seq_q, int seq_k,
                               int n_heads, int n_kv_heads, int head_dim,
                               float scale, int q_offset);
void launch_embedding_lookup(cudaStream_t stream,
                               float * embeddings,
                               const int * tokens,
                               const float * W,
                               int seq_len,
                               int hidden);
void launch_build_rope_cache(cudaStream_t stream,
                                const float * inv_freq,
                                float * cos_vals, float * sin_vals,
                                int seq, int head_dim);
void launch_bf16_transpose(cudaStream_t stream,
                              float * W_T, const uint16_t * W,
                              int out_dim, int in_dim);
void launch_layer_norm(cudaStream_t stream,
                        float * out, const float * x,
                        const float * weight, const float * bias,
                        int seq_len, int hidden, float eps);
void launch_bidir_attention(cudaStream_t stream,
                              float * out,
                              const float * Q, const float * K, const float * V,
                              int seq_len, int n_heads, int head_dim,
                              float scale, const int * window_starts, int n_windows);
void launch_gelu(cudaStream_t stream,
                  float * out, const float * x, int total_elements);
void launch_fp32_transpose(cudaStream_t stream,
                             float * W_T, const float * W,
                             int out_dim, int in_dim);
void launch_broadcast_add(cudaStream_t stream,
                             float * out,
                             const float * matrix,
                             const float * bias,
                             int seq_len, int hidden);
void launch_gemv(cudaStream_t stream,
                    float * out,
                    const float * x,
                    const float * W_T,
                    int in_dim, int out_dim);
void launch_extract_mel_chunk(cudaStream_t stream,
                                float * chunk,
                                const float * mel,
                                int mel_bins, int mel_frames,
                                int chunk_w, int start);
void launch_conv2d(cudaStream_t stream,
                     const float * in,
                     const float * weight,
                     const float * bias,
                     float * out,
                     int c_in, int c_out,
                     int h_in, int w_in,
                     int kh, int kw,
                     int stride, int padding,
                     int fused_gelu);
void launch_reshape(cudaStream_t stream,
                      float * out,
                      const float * in,
                      int c, int h, int w);
void launch_sinusoidal_pe(cudaStream_t stream,
                              float * pe,
                              int seq_len, int d_model);

/* Graph-compatible launchers (read dynamic params from d_params) */
#include "qasr/backend/cuda_decode_params.h"
void launch_write_prev_token(cudaStream_t stream,
                               CudaDecodeParams *params,
                               int prev_token);
void launch_write_seq_pos(cudaStream_t stream,
                            CudaDecodeParams *params,
                            int seq_pos);
void launch_embed_lookup_from_token(cudaStream_t stream,
                                      float *out,
                                      const float *tok_embeddings,
                                      CudaDecodeParams *params,
                                      int hidden);
void launch_kv_cache_store(cudaStream_t stream,
                             const float *k_src,
                             const float *v_src,
                             float *kv_cache_k,
                             float *kv_cache_v,
                             CudaDecodeParams *params,
                             int kv_dim);
void launch_rope_neox_graph(cudaStream_t stream,
                              float * x,
                              const float * rope_cos_base,
                              const float * rope_sin_base,
                              CudaDecodeParams *params,
                              int n_heads,
                              int head_dim);
void launch_causal_attention_graph(cudaStream_t stream,
                                     float * out,
                                     const float * Q,
                                     const float * K,
                                     const float * V,
                                     CudaDecodeParams *params,
                                     int n_heads,
                                     int n_kv_heads,
                                     int head_dim,
                                     float scale);

/* Fusion kernels (decoder_fusion.cu) — reduce 19 launches/layer to ~4 */
void launch_rmsnorm_qkv_rope(cudaStream_t stream,
                               float *Q, float *K, float *V, const float *x,
                               const float *wq_T, const float *wk_T, const float *wv_T,
                               const float *input_norm_w, const float *q_norm_w, const float *k_norm_w,
                               const float *rope_cos, const float *rope_sin,
                               CudaDecodeParams *params,
                               int hidden, int q_dim, int kv_dim,
                               int n_heads, int n_kv_heads, int hd, float eps);
void launch_wo_residual(cudaStream_t stream,
                          float * x,
                          const float * attn_out,
                          const float * wo_T,
                          int q_dim, int hidden);
void launch_ffn(cudaStream_t stream,
                  float * x,
                  const float * post_norm,
                  const float * gate_T,
                  const float * up_T,
                  const float * down_T,
                  const float * post_attn_norm_w,
                  int hidden, int intermediate, float eps);
}
#endif

namespace qasr {

#ifdef QASR_CUDA_BACKEND_ENABLED
/* Y[seq, out] = X[seq, in] @ W[out, in]^T  — W is fp32 [out, in] */
static Status CublasGemmF32(int seq_len, int in_dim, int out_dim,
                              const float * X, const float * W_fp32,
                              float * Y, float * W_T_workspace,
                              cudaStream_t stream, cublasHandle_t cublas) {
    launch_fp32_transpose(stream, W_T_workspace, W_fp32, out_dim, in_dim);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         out_dim, seq_len, in_dim,
                                         &alpha, W_T_workspace, out_dim,
                                         X, in_dim, &beta, Y, out_dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return Status(StatusCode::kInternal,
                      std::string("cublasSgemm encoder failed: ") + std::to_string(status));
    }
    return OkStatus();
}
#endif

CudaBuffer::~CudaBuffer() {
    Reset();
}

Status CudaBuffer::Allocate(size_t bytes) {
    if (ptr_) {
#ifdef QASR_CUDA_BACKEND_ENABLED
        cudaFree(ptr_);
#else
        std::free(ptr_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }
    if (bytes == 0) return OkStatus();
#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
        return Status(StatusCode::kResourceExhausted,
                      "cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
#else
    ptr_ = std::malloc(bytes);
    if (!ptr_) {
        return Status(StatusCode::kResourceExhausted, "malloc failed");
    }
#endif
    size_ = bytes;
    return OkStatus();
}

#ifdef QASR_CUDA_BACKEND_ENABLED
Status CudaBuffer::AllocateAsync(cudaStream_t stream, size_t bytes) {
    (void)stream;
    return Allocate(bytes);
}
#endif

void CudaBuffer::Reset() {
    if (ptr_) {
#ifdef QASR_CUDA_BACKEND_ENABLED
        cudaFree(ptr_);
#else
        std::free(ptr_);
#endif
        ptr_ = nullptr;
    }
    size_ = 0;
}

CudaBuffer::CudaBuffer(CudaBuffer && other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

CudaBuffer & CudaBuffer::operator=(CudaBuffer && other) noexcept {
    if (this != &other) {
        Reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

CudaStreamHandle::~CudaStreamHandle() {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
#endif
}

#ifdef QASR_CUDA_BACKEND_ENABLED
Status CudaStreamHandle::Create() {
    if (stream_) return OkStatus();
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        return Status(StatusCode::kInternal,
                      "cudaStreamCreate failed: " + std::string(cudaGetErrorString(err)));
    }
    return OkStatus();
}
#endif

CublasHandle::~CublasHandle() {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
#endif
}

#ifdef QASR_CUDA_BACKEND_ENABLED
Status CublasHandle::Create() {
    if (handle_) return OkStatus();
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return Status(StatusCode::kInternal,
                      "cublasCreate failed: " + std::to_string(status));
    }
    /* Disable TF32 — force precise fp32 GEMM.
     * TF32 truncates 23-bit mantissa to 10-bit, causing accumulation drift
     * across 18 encoder transformer layers. whisper.cpp uses precise fp32. */
    status = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return Status(StatusCode::kInternal,
                      "cublasSetMathMode(CUBLAS_DEFAULT_MATH) failed: " + std::to_string(status));
    }
    return OkStatus();
}

Status CublasHandle::SetStream(cudaStream_t stream) {
    if (!handle_) {
        return Status(StatusCode::kFailedPrecondition, "cuBLAS handle not created");
    }
    cublasStatus_t status = cublasSetStream(handle_, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return Status(StatusCode::kInternal,
                      "cublasSetStream failed: " + std::to_string(status));
    }
    return OkStatus();
}
#endif

CudaSessionState::~CudaSessionState() {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (graph_exec) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
    }
    if (graph) {
        cudaGraphDestroy(graph);
        graph = nullptr;
    }
    if (graph_sync_event) {
        cudaEventDestroy(graph_sync_event);
        graph_sync_event = nullptr;
    }
#endif
}

CudaBackend::~CudaBackend() {
    Shutdown();
}

Status CudaBackend::Initialize() {
    if (initialized_) return OkStatus();

#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        return Status(StatusCode::kInternal,
                      "cudaSetDevice(" + std::to_string(device_id_) + ") failed: " +
                      std::string(cudaGetErrorString(err)));
    }

    err = cudaGetDeviceProperties(&device_prop_, device_id_);
    if (err != cudaSuccess) {
        return Status(StatusCode::kInternal,
                      "cudaGetDeviceProperties failed: " + std::string(cudaGetErrorString(err)));
    }

    if (auto status = compute_stream_.Create(); !status.ok()) {
        return status;
    }
    if (auto status = cublas_.Create(); !status.ok()) {
        return status;
    }
    if (auto status = cublas_.SetStream(compute_stream_.stream()); !status.ok()) {
        return status;
    }

    cuda_weights_ = std::make_shared<CudaWeights>();
    initialized_ = true;

    fprintf(stderr, "CUDA: device=%d name=%s sm=%d.%d\n",
            device_id_, device_prop_.name,
            device_prop_.major, device_prop_.minor);
#else
    fprintf(stderr, "CUDA: stub backend (not compiled with CUDA)\n");
    initialized_ = true;
#endif
    return OkStatus();
}

Status CudaBackend::Shutdown() {
    if (!initialized_) return OkStatus();
    cuda_weights_.reset();
    initialized_ = false;
    return OkStatus();
}

/* Allocate session workspace and KV cache */
Status CudaBackend::AllocateSession(CudaSessionState * session,
                                      int max_seq_len) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    int hidden = cuda_weights_->dec_hidden;
    int intermediate = cuda_weights_->dec_intermediate;
    int n_heads = cuda_weights_->dec_heads;
    int n_kv_heads = cuda_weights_->dec_kv_heads;
    int head_dim = cuda_weights_->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Encoder config */
    int enc_d_model = cuda_weights_->enc_d_model;
    int enc_ffn_dim = cuda_weights_->enc_ffn_dim;
    int enc_heads = cuda_weights_->enc_heads;
    int enc_head_dim = cuda_weights_->enc_head_dim;
    int enc_output_dim = cuda_weights_->enc_output_dim;

    /* KV cache: [dec_layers, max_seq_len, kv_dim] each for K and V
     * Match CPU layout: kv_cache_k[layer * kv_cache_max + pos] * kv_dim */
    size_t kv_size = (size_t)cuda_weights_->dec_layers * max_seq_len * kv_dim * sizeof(float);
    session->kv_cache_k.Allocate(kv_size);
    session->kv_cache_v.Allocate(kv_size);

    /* RoPE cache: [max_seq_len, head_dim] each for cos and sin */
    size_t rope_size = max_seq_len * head_dim * sizeof(float);
    session->rope_cos.Allocate(rope_size);
    session->rope_sin.Allocate(rope_size);

    /* Workspace buffer for decoder forward pass */
    size_t workspace_size = 0;
    workspace_size += max_seq_len * hidden * sizeof(float);        /* x_norm */
    workspace_size += max_seq_len * q_dim * sizeof(float);         /* q */
    workspace_size += max_seq_len * kv_dim * sizeof(float);        /* k */
    workspace_size += max_seq_len * kv_dim * sizeof(float);        /* v */
    workspace_size += max_seq_len * max_seq_len * sizeof(float);   /* attn_score */
    workspace_size += max_seq_len * q_dim * sizeof(float);         /* attn_out */
    workspace_size += max_seq_len * hidden * sizeof(float);        /* proj_out */
    workspace_size += max_seq_len * hidden * sizeof(float);        /* post_norm */
    workspace_size += max_seq_len * (2 * intermediate) * sizeof(float); /* gate_up */
    workspace_size += max_seq_len * hidden * sizeof(float);        /* ffn_out */

    /* DecodeStep lm_head: only logits[vocab] needed (W_T is pre-transposed in lm_head_T_fp32) */
    workspace_size += cuda_weights_->vocab_size * sizeof(float);   /* lm_head logits */
    session->workspace.Allocate(workspace_size);

    /* Initialize workspace to zero to avoid NaN from uninitialized memory */
    cudaMemset(session->workspace.data(), 0, workspace_size);

    /* Encoder output buffer: max enc tokens for ~1min audio */
    int max_enc_tokens = max_seq_len;
    size_t enc_output_size = (size_t)max_enc_tokens * enc_output_dim * sizeof(float);
    session->enc_output.Allocate(enc_output_size);

    /* Encoder workspace: transformer forward scratch */
    size_t enc_ws_size = 0;
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* x (input copy) */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* x_norm */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* q */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* k */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* v */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* attn_out */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* proj_out */
    enc_ws_size += (size_t)max_enc_tokens * enc_ffn_dim * sizeof(float);   /* ffn_mid */
    enc_ws_size += (size_t)max_enc_tokens * enc_d_model * sizeof(float);   /* ffn_out */
    enc_ws_size += (size_t)enc_d_model * enc_ffn_dim * sizeof(float);     /* W_fp32 (largest: fc1) */
    enc_ws_size += (max_enc_tokens + 2) * sizeof(int);                     /* window_starts */
    session->enc_workspace.Allocate(enc_ws_size);

    /* Encoder conv2D stem workspace (pre-allocated, no hot-path malloc) */
    {
        int full_w = cuda_weights_->enc_chunk_size;    /* 100 */
        int mel_bins = 128;
        int full_h1 = (mel_bins + 2 - 3) / 2 + 1;   /* 64 */
        int full_w1 = (full_w + 2 - 3) / 2 + 1;     /* 50 */
        int full_h2 = (full_h1 + 2 - 3) / 2 + 1;    /* 32 */
        int full_w2 = (full_w1 + 2 - 3) / 2 + 1;    /* 25 */
        int full_h3 = (full_h2 + 2 - 3) / 2 + 1;    /* 16 */
        int full_w3 = (full_w2 + 2 - 3) / 2 + 1;    /* 12 */
        int conv_proj_dim = QWEN_CONV_HIDDEN * full_h3; /* 7680 */

        session->enc_d_mel.Allocate(mel_bins * max_seq_len * sizeof(float));  /* full mel */
        session->enc_d_mel_chunk.Allocate(mel_bins * full_w * sizeof(float)); /* chunk mel */
        session->enc_d_c1.Allocate(QWEN_CONV_HIDDEN * full_h1 * full_w1 * sizeof(float));
        session->enc_d_c2.Allocate(QWEN_CONV_HIDDEN * full_h2 * full_w2 * sizeof(float));
        session->enc_d_c3.Allocate(QWEN_CONV_HIDDEN * full_h3 * full_w3 * sizeof(float));
        session->enc_d_reshape.Allocate(full_w3 * conv_proj_dim * sizeof(float));
        session->enc_d_proj.Allocate(max_enc_tokens * enc_d_model * sizeof(float));
        session->enc_d_pe.Allocate(full_w3 * enc_d_model * sizeof(float));
    }

    /* Decoder prefill: pre-allocated token buffer */
    session->d_tokens.Allocate(max_seq_len * sizeof(int));

  /* Build RoPE cache once (invariant — no need to rebuild per segment) */
    {
        float * rope_cos = static_cast<float *>(session->rope_cos.data());
        float * rope_sin = static_cast<float *>(session->rope_sin.data());
        launch_build_rope_cache(compute_stream_.stream(),
                                  static_cast<float *>(cuda_weights_->inv_freq.data()),
                                  rope_cos, rope_sin,
                                  max_seq_len, head_dim);
        cudaStreamSynchronize(compute_stream_.stream());
        session->rope_cache_built = true;
    }

    /* Capture DecodeStep compute graph for replay */
    if (auto gs = CaptureDecodeGraph(session); !gs.ok()) {
        fprintf(stderr, "CUDA: graph capture failed: %s (falling back to non-graph)\n",
                gs.ToString().c_str());
    }

    return OkStatus();
#else
    (void)session; (void)max_seq_len;
    return Status(StatusCode::kUnimplemented,
                  "AllocateSession requires CUDA backend");
#endif
}

Status CudaBackend::PrepareWeights(const std::string & model_dir) {
    if (!initialized_) {
        return Initialize();
    }

#ifdef QASR_CUDA_BACKEND_ENABLED
    /* Load model configuration from C backend to extract architecture params */
    auto * ctx = qwen_load(model_dir.c_str());
    if (!ctx) {
        return Status(StatusCode::kInternal,
                      "qwen_load failed for " + model_dir + " (CUDA-2 weight transfer requires CPU load first)");
    }

    auto * qwen_ctx = static_cast<qwen_ctx_t *>(ctx);
    auto & cfg = qwen_ctx->config;

    if (!cuda_weights_) {
        cuda_weights_ = std::make_shared<CudaWeights>();
    }

    /* Store model config in weights */
    cuda_weights_->model_dir = model_dir;
    cuda_weights_->dec_layers = cfg.dec_layers;
    cuda_weights_->dec_hidden = cfg.dec_hidden;
    cuda_weights_->dec_intermediate = cfg.dec_intermediate;
    cuda_weights_->vocab_size = cfg.vocab_size;
    cuda_weights_->dec_heads = cfg.dec_heads;
    cuda_weights_->dec_kv_heads = cfg.dec_kv_heads;
    cuda_weights_->dec_head_dim = cfg.dec_head_dim;
    cuda_weights_->dec_rms_norm_eps = cfg.dec_rms_norm_eps;

   /* Resize per-layer weight vectors */
    cuda_weights_->wq_T.resize(cfg.dec_layers);
    cuda_weights_->wk_T.resize(cfg.dec_layers);
    cuda_weights_->wv_T.resize(cfg.dec_layers);
    cuda_weights_->wo_T.resize(cfg.dec_layers);
    cuda_weights_->gate_T.resize(cfg.dec_layers);
    cuda_weights_->up_T.resize(cfg.dec_layers);
    cuda_weights_->down_T.resize(cfg.dec_layers);
    cuda_weights_->input_norm.resize(cfg.dec_layers);
    cuda_weights_->post_attn_norm.resize(cfg.dec_layers);
    cuda_weights_->q_norm.resize(cfg.dec_layers);
    cuda_weights_->k_norm.resize(cfg.dec_layers);

    /* Standard CUDA SDK pattern: bf16→fp32+transpose ONCE at PrepareWeights.
     * Uses a single temporary GPU buffer reused across all layers.
     * Original bf16 freed immediately — no double memory cost. */
    {
        size_t max_bf16_size = (size_t)cfg.dec_intermediate * cfg.dec_hidden * sizeof(uint16_t);
        void * tmp_bf16 = nullptr;
        cudaMalloc(&tmp_bf16, max_bf16_size);

        for (int l = 0; l < cfg.dec_layers; l++) {
            auto & layer = qwen_ctx->decoder.layers[l];

            /* Q: [q_dim, hidden] bf16 → [hidden, q_dim] fp32 */
            {
                size_t bf16_size = cfg.dec_heads * cfg.dec_head_dim * cfg.dec_hidden * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->wq_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.wq_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->wq_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_heads * cfg.dec_head_dim, cfg.dec_hidden);
            }
            /* K: [kv_dim, hidden] bf16 → [hidden, kv_dim] fp32 */
            {
                size_t bf16_size = cfg.dec_kv_heads * cfg.dec_head_dim * cfg.dec_hidden * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->wk_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.wk_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->wk_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_kv_heads * cfg.dec_head_dim, cfg.dec_hidden);
            }
            /* V: [kv_dim, hidden] bf16 → [hidden, kv_dim] fp32 */
            {
                size_t bf16_size = cfg.dec_kv_heads * cfg.dec_head_dim * cfg.dec_hidden * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->wv_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.wv_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->wv_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_kv_heads * cfg.dec_head_dim, cfg.dec_hidden);
            }
            /* WO: [hidden, q_dim] bf16 → [q_dim, hidden] fp32 */
            {
                size_t bf16_size = cfg.dec_hidden * cfg.dec_heads * cfg.dec_head_dim * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->wo_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.wo_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->wo_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_hidden, cfg.dec_heads * cfg.dec_head_dim);
            }
            /* Gate: [intermediate, hidden] bf16 → [hidden, intermediate] fp32 */
            {
                size_t bf16_size = cfg.dec_intermediate * cfg.dec_hidden * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->gate_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.gate_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->gate_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_intermediate, cfg.dec_hidden);
            }
            /* Up: [intermediate, hidden] bf16 → [hidden, intermediate] fp32 */
            {
                size_t bf16_size = cfg.dec_intermediate * cfg.dec_hidden * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->up_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.up_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->up_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_intermediate, cfg.dec_hidden);
            }
            /* Down: [hidden, intermediate] bf16 → [intermediate, hidden] fp32 */
            {
                size_t bf16_size = cfg.dec_hidden * cfg.dec_intermediate * sizeof(uint16_t);
                size_t fp32_size = bf16_size / sizeof(uint16_t) * sizeof(float);
                cuda_weights_->down_T[l].Allocate(fp32_size);
                cudaMemcpy(tmp_bf16, layer.down_weight_bf16, bf16_size, cudaMemcpyHostToDevice);
                launch_bf16_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->down_T[l].data()),
                                        static_cast<uint16_t *>(tmp_bf16),
                                        cfg.dec_hidden, cfg.dec_intermediate);
            }
            /* Norm weights (fp32) */
            {
                size_t norm_size = cfg.dec_hidden * sizeof(float);
                cuda_weights_->input_norm[l].Allocate(norm_size);
                cudaMemcpyAsync(cuda_weights_->input_norm[l].data(),
                                layer.input_norm, norm_size,
                                cudaMemcpyHostToDevice, compute_stream_.stream());
                cuda_weights_->post_attn_norm[l].Allocate(norm_size);
                cudaMemcpyAsync(cuda_weights_->post_attn_norm[l].data(),
                                layer.post_attn_norm, norm_size,
                                cudaMemcpyHostToDevice, compute_stream_.stream());
            }
            /* Per-head Q/K norm (fp32) */
            {
                size_t hn_size = cfg.dec_head_dim * sizeof(float);
                cuda_weights_->q_norm[l].Allocate(hn_size);
                cudaMemcpyAsync(cuda_weights_->q_norm[l].data(),
                                layer.q_norm_weight, hn_size,
                                cudaMemcpyHostToDevice, compute_stream_.stream());
                cuda_weights_->k_norm[l].Allocate(hn_size);
                cudaMemcpyAsync(cuda_weights_->k_norm[l].data(),
                                layer.k_norm_weight, hn_size,
                                cudaMemcpyHostToDevice, compute_stream_.stream());
            }
        }
        cudaFree(tmp_bf16);
    }

    /* tok_embeddings: bf16→fp32 for embedding lookup.
     * tok_embeddings_bf16/lm_head_bf16 are host pointers — CUDA kernels
     * cannot access CPU memory on discrete GPUs (e.g. RTX 3070).
     * Copy to a temp GPU buffer first, then launch. */
    {
        size_t emb_bf16_size = (size_t)cfg.vocab_size * cfg.dec_hidden * sizeof(uint16_t);
        size_t emb_fp32_size = (size_t)cfg.vocab_size * cfg.dec_hidden * sizeof(float);
        void * tmp_emb_bf16 = nullptr;
        cudaMalloc(&tmp_emb_bf16, emb_bf16_size);
        cudaMemcpy(tmp_emb_bf16, qwen_ctx->decoder.tok_embeddings_bf16, emb_bf16_size,
                    cudaMemcpyHostToDevice);
        cuda_weights_->tok_embeddings_fp32.Allocate(emb_fp32_size);
        launch_bf16_to_fp32(compute_stream_.stream(),
                              static_cast<float *>(cuda_weights_->tok_embeddings_fp32.data()),
                              static_cast<const uint16_t *>(tmp_emb_bf16),
                              cfg.vocab_size * cfg.dec_hidden);

        /* lm_head: tied with tok_embeddings for ASR, separate for Aligner.
         * If lm_head_bf16 exists, overwrite tmp_emb_bf16 with its data;
         * otherwise reuse tok_embeddings data already on GPU. */
        if (qwen_ctx->decoder.lm_head_bf16) {
            cuda_weights_->lm_head_ready = true;
            cudaMemcpy(tmp_emb_bf16, qwen_ctx->decoder.lm_head_bf16, emb_bf16_size,
                        cudaMemcpyHostToDevice);
        } else {
            cuda_weights_->lm_head_ready = false;
        }
        size_t lm_T_size = (size_t)cfg.dec_hidden * cfg.vocab_size * sizeof(float);
        cuda_weights_->lm_head_T_fp32.Allocate(lm_T_size);
        launch_bf16_transpose(compute_stream_.stream(),
                                static_cast<float *>(cuda_weights_->lm_head_T_fp32.data()),
                                static_cast<const uint16_t *>(tmp_emb_bf16),
                                cfg.vocab_size, cfg.dec_hidden);
        cudaFree(tmp_emb_bf16);
    }

    /* final_norm (fp32) */
    {
        size_t norm_size = cfg.dec_hidden * sizeof(float);
        cuda_weights_->final_norm.Allocate(norm_size);
        cudaMemcpyAsync(cuda_weights_->final_norm.data(),
                        qwen_ctx->decoder.norm, norm_size,
                        cudaMemcpyHostToDevice, compute_stream_.stream());
    }

    /* Pre-compute inv_freq for RoPE (invariant across all inference) */
    {
        int head_dim = cuda_weights_->dec_head_dim;
        size_t inv_freq_size = (head_dim / 2) * sizeof(float);
        cuda_weights_->inv_freq.Allocate(inv_freq_size);
        float * h_inv_freq = new float[head_dim / 2];
        for (int d = 0; d < head_dim / 2; d++) {
            h_inv_freq[d] = 1.0f / powf(1e6f, (float)(2 * d) / (float)head_dim);
        }
        cudaMemcpyAsync(static_cast<float *>(cuda_weights_->inv_freq.data()),
                         h_inv_freq, inv_freq_size,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
        delete[] h_inv_freq;
    }

    /* Synchronize and mark decoder ready */
    cudaStreamSynchronize(compute_stream_.stream());
    cuda_weights_->decoder_ready = true;

    /* === Load encoder weights === */
    cuda_weights_->enc_layers = cfg.enc_layers;
    cuda_weights_->enc_d_model = cfg.enc_d_model;
    cuda_weights_->enc_heads = cfg.enc_heads;
    cuda_weights_->enc_head_dim = cfg.enc_head_dim;
    cuda_weights_->enc_ffn_dim = cfg.enc_ffn_dim;
    cuda_weights_->enc_output_dim = cfg.enc_output_dim;
    cuda_weights_->enc_chunk_size = cfg.enc_chunk_size;
    cuda_weights_->enc_n_window_infer = cfg.enc_n_window_infer;

    /* Conv2D stem weights (fp32) */
    {
        auto & enc = qwen_ctx->encoder;
        size_t conv1_w = (size_t)QWEN_CONV_HIDDEN * 1 * 3 * 3 * sizeof(float);
        cuda_weights_->enc_conv1_w.Allocate(conv1_w);
        cudaMemcpyAsync(cuda_weights_->enc_conv1_w.data(), enc.conv1_weight, conv1_w,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
        size_t conv1_b = QWEN_CONV_HIDDEN * sizeof(float);
        cuda_weights_->enc_conv1_b.Allocate(conv1_b);
        cudaMemcpyAsync(cuda_weights_->enc_conv1_b.data(), enc.conv1_bias, conv1_b,
                         cudaMemcpyHostToDevice, compute_stream_.stream());

        size_t conv2_w = (size_t)QWEN_CONV_HIDDEN * QWEN_CONV_HIDDEN * 3 * 3 * sizeof(float);
        cuda_weights_->enc_conv2_w.Allocate(conv2_w);
        cudaMemcpyAsync(cuda_weights_->enc_conv2_w.data(), enc.conv2_weight, conv2_w,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
        size_t conv2_b = QWEN_CONV_HIDDEN * sizeof(float);
        cuda_weights_->enc_conv2_b.Allocate(conv2_b);
        cudaMemcpyAsync(cuda_weights_->enc_conv2_b.data(), enc.conv2_bias, conv2_b,
                         cudaMemcpyHostToDevice, compute_stream_.stream());

        size_t conv3_w = (size_t)QWEN_CONV_HIDDEN * QWEN_CONV_HIDDEN * 3 * 3 * sizeof(float);
        cuda_weights_->enc_conv3_w.Allocate(conv3_w);
        cudaMemcpyAsync(cuda_weights_->enc_conv3_w.data(), enc.conv3_weight, conv3_w,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
        size_t conv3_b = QWEN_CONV_HIDDEN * sizeof(float);
        cuda_weights_->enc_conv3_b.Allocate(conv3_b);
        cudaMemcpyAsync(cuda_weights_->enc_conv3_b.data(), enc.conv3_bias, conv3_b,
                         cudaMemcpyHostToDevice, compute_stream_.stream());

        /* conv_out: bf16 → fp32 transpose [d_model, conv_proj_dim] → [conv_proj_dim, d_model] */
        size_t conv_proj_dim = QWEN_CONV_HIDDEN * 16; /* 480 * 16 = 7680 */
        size_t conv_out_size = (size_t)cfg.enc_d_model * conv_proj_dim * sizeof(float);
        cuda_weights_->enc_conv_out_T_fp32.Allocate(conv_out_size);
        void * tmp_conv_out = nullptr;
        cudaMalloc(&tmp_conv_out, conv_out_size);
        cudaMemcpy(tmp_conv_out, enc.conv_out_weight, conv_out_size, cudaMemcpyHostToDevice);
        launch_fp32_transpose(compute_stream_.stream(),
                                static_cast<float *>(cuda_weights_->enc_conv_out_T_fp32.data()),
                                static_cast<float *>(tmp_conv_out),
                                cfg.enc_d_model, conv_proj_dim);
        cudaStreamSynchronize(compute_stream_.stream());
        cudaFree(tmp_conv_out);
    }

    /* Resize per-layer encoder weight vectors */
    cuda_weights_->enc_wq.resize(cfg.enc_layers);
    cuda_weights_->enc_wk.resize(cfg.enc_layers);
    cuda_weights_->enc_wv.resize(cfg.enc_layers);
    cuda_weights_->enc_wo.resize(cfg.enc_layers);
    cuda_weights_->enc_fc1.resize(cfg.enc_layers);
    cuda_weights_->enc_fc2.resize(cfg.enc_layers);
    cuda_weights_->enc_wq_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_wk_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_wv_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_wo_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_fc1_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_fc2_bias.resize(cfg.enc_layers);
    cuda_weights_->enc_attn_norm_w.resize(cfg.enc_layers);
    cuda_weights_->enc_attn_norm_b.resize(cfg.enc_layers);
    cuda_weights_->enc_ffn_norm_w.resize(cfg.enc_layers);
    cuda_weights_->enc_ffn_norm_b.resize(cfg.enc_layers);

/* Encoder weights: fp32 (CPU load_bf16_as_f32). Transpose to [in_dim, out_dim] for cuBLAS. */
    {
        void * tmp_enc_fp32 = nullptr;
        size_t max_enc_size = (size_t)cfg.enc_ffn_dim * cfg.enc_d_model * sizeof(float);
        cudaMalloc(&tmp_enc_fp32, max_enc_size);

        for (int l = 0; l < cfg.enc_layers; l++) {
            auto * layer = &qwen_ctx->encoder.layers[l];

            /* QKV: [d_model, d_model] → [d_model, d_model] (square, transpose in-place) */
            {
                size_t qkv_size = cfg.enc_d_model * cfg.enc_d_model * sizeof(float);
                cuda_weights_->enc_wq[l].Allocate(qkv_size);
                cudaMemcpy(tmp_enc_fp32, layer->wq_weight, qkv_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_wq[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_d_model, cfg.enc_d_model);

                cuda_weights_->enc_wk[l].Allocate(qkv_size);
                cudaMemcpy(tmp_enc_fp32, layer->wk_weight, qkv_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_wk[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_d_model, cfg.enc_d_model);

                cuda_weights_->enc_wv[l].Allocate(qkv_size);
                cudaMemcpy(tmp_enc_fp32, layer->wv_weight, qkv_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_wv[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_d_model, cfg.enc_d_model);
            }
            /* WO: [d_model, d_model] → [d_model, d_model] */
            {
                size_t qkv_size = cfg.enc_d_model * cfg.enc_d_model * sizeof(float);
                cuda_weights_->enc_wo[l].Allocate(qkv_size);
                cudaMemcpy(tmp_enc_fp32, layer->wo_weight, qkv_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_wo[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_d_model, cfg.enc_d_model);
            }
            /* FC1: [ffn_dim, d_model] → [d_model, ffn_dim] */
            {
                size_t fc1_size = cfg.enc_ffn_dim * cfg.enc_d_model * sizeof(float);
                cuda_weights_->enc_fc1[l].Allocate(fc1_size);
                cudaMemcpy(tmp_enc_fp32, layer->fc1_weight, fc1_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_fc1[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_ffn_dim, cfg.enc_d_model);
            }
            /* FC2: [d_model, ffn_dim] → [ffn_dim, d_model] */
            {
                size_t fc2_size = cfg.enc_d_model * cfg.enc_ffn_dim * sizeof(float);
                cuda_weights_->enc_fc2[l].Allocate(fc2_size);
                cudaMemcpy(tmp_enc_fp32, layer->fc2_weight, fc2_size, cudaMemcpyHostToDevice);
                launch_fp32_transpose(compute_stream_.stream(),
                                        static_cast<float *>(cuda_weights_->enc_fc2[l].data()),
                                        static_cast<float *>(tmp_enc_fp32),
                                        cfg.enc_d_model, cfg.enc_ffn_dim);
            }

            /* Biases (fp32) */
            size_t bias_size = cfg.enc_d_model * sizeof(float);
            cuda_weights_->enc_wq_bias[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_wq_bias[l].data(), layer->wq_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_wk_bias[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_wk_bias[l].data(), layer->wk_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_wv_bias[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_wv_bias[l].data(), layer->wv_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_wo_bias[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_wo_bias[l].data(), layer->wo_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_fc2_bias[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_fc2_bias[l].data(), layer->fc2_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());

            /* FC1 bias (ffn_dim) */
            size_t fc1_bias_size = cfg.enc_ffn_dim * sizeof(float);
            cuda_weights_->enc_fc1_bias[l].Allocate(fc1_bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_fc1_bias[l].data(), layer->fc1_bias, fc1_bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());

            /* LayerNorm weights and biases (fp32) */
            cuda_weights_->enc_attn_norm_w[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_attn_norm_w[l].data(), layer->attn_norm_weight, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_attn_norm_b[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_attn_norm_b[l].data(), layer->attn_norm_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_ffn_norm_w[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_ffn_norm_w[l].data(), layer->ffn_norm_weight, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
            cuda_weights_->enc_ffn_norm_b[l].Allocate(bias_size);
            cudaMemcpyAsync(cuda_weights_->enc_ffn_norm_b[l].data(), layer->ffn_norm_bias, bias_size,
                             cudaMemcpyHostToDevice, compute_stream_.stream());
        }
        cudaStreamSynchronize(compute_stream_.stream());
        cudaFree(tmp_enc_fp32);
    }

    /* Encoder post-LayerNorm (fp32) */
    {
        size_t norm_size = cfg.enc_d_model * sizeof(float);
        cuda_weights_->enc_ln_post_w.Allocate(norm_size);
        cudaMemcpyAsync(cuda_weights_->enc_ln_post_w.data(), qwen_ctx->encoder.ln_post_weight, norm_size,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
        cuda_weights_->enc_ln_post_b.Allocate(norm_size);
        cudaMemcpyAsync(cuda_weights_->enc_ln_post_b.data(), qwen_ctx->encoder.ln_post_bias, norm_size,
                         cudaMemcpyHostToDevice, compute_stream_.stream());
    }

    /* Encoder projection layers: fp32 → transpose for cuBLAS */
    {
        void * tmp_proj = nullptr;
        size_t max_proj_size = (size_t)cfg.enc_d_model * (size_t)cfg.enc_d_model * sizeof(float);
        size_t proj2_size_check = (size_t)cfg.enc_output_dim * (size_t)cfg.enc_d_model * sizeof(float);
        if (proj2_size_check > max_proj_size) max_proj_size = proj2_size_check;
        cudaMalloc(&tmp_proj, max_proj_size);

        size_t proj1_size = cfg.enc_d_model * cfg.enc_d_model * sizeof(float);
        cuda_weights_->enc_proj1.Allocate(proj1_size);
        cudaMemcpy(tmp_proj, qwen_ctx->encoder.proj1_weight, proj1_size, cudaMemcpyHostToDevice);
        launch_fp32_transpose(compute_stream_.stream(),
                                static_cast<float *>(cuda_weights_->enc_proj1.data()),
                                static_cast<float *>(tmp_proj),
                                cfg.enc_d_model, cfg.enc_d_model);
        size_t proj1_bias_size = cfg.enc_d_model * sizeof(float);
        cuda_weights_->enc_proj1_bias.Allocate(proj1_bias_size);
        cudaMemcpyAsync(cuda_weights_->enc_proj1_bias.data(), qwen_ctx->encoder.proj1_bias, proj1_bias_size,
                         cudaMemcpyHostToDevice, compute_stream_.stream());

        size_t proj2_size = (size_t)cfg.enc_output_dim * cfg.enc_d_model * sizeof(float);
        cuda_weights_->enc_proj2.Allocate(proj2_size);
        cudaMemcpy(tmp_proj, qwen_ctx->encoder.proj2_weight, proj2_size, cudaMemcpyHostToDevice);
        launch_fp32_transpose(compute_stream_.stream(),
                                static_cast<float *>(cuda_weights_->enc_proj2.data()),
                                static_cast<float *>(tmp_proj),
                                cfg.enc_output_dim, cfg.enc_d_model);
        size_t proj2_bias_size = cfg.enc_output_dim * sizeof(float);
        cuda_weights_->enc_proj2_bias.Allocate(proj2_bias_size);
        cudaMemcpyAsync(cuda_weights_->enc_proj2_bias.data(), qwen_ctx->encoder.proj2_bias, proj2_bias_size,
                         cudaMemcpyHostToDevice, compute_stream_.stream());

        cudaStreamSynchronize(compute_stream_.stream());
        cudaFree(tmp_proj);
    }

    cudaStreamSynchronize(compute_stream_.stream());
    cuda_weights_->encoder_ready = true;

    fprintf(stderr, "CUDA-8: encoder weights loaded (layers=%d, d_model=%d, heads=%d, ffn=%d)\n",
            cfg.enc_layers, cfg.enc_d_model, cfg.enc_heads, cfg.enc_ffn_dim);
    fprintf(stderr, "CUDA-2: decoder weights loaded (layers=%d, hidden=%d, vocab=%d)\n",
            cfg.dec_layers, cfg.dec_hidden, cfg.vocab_size);

    /* Free CPU context (weights are now on GPU) */
    qwen_free(ctx);

    return OkStatus();
#else
    (void)model_dir;
    return Status(StatusCode::kUnimplemented,
                  "CUDA decoder residency not yet implemented (CUDA-2 stage)");
#endif
}

Status CudaBackend::EncodeMel(void * workspace_ptr,
                                 const float * mel_features,
                                 int mel_frames,
                                 float * output,
                                 int & out_tokens) {
    /* Delegate to EncoderForward, then copy GPU result to CPU output buffer */
    auto * session = static_cast<CudaSessionState *>(workspace_ptr);
    int enc_tokens = 0;
    auto status = EncoderForward(session, mel_features, mel_frames, &enc_tokens);
    if (!status.ok()) return status;
    out_tokens = enc_tokens;

    /* Copy GPU encoder output to CPU output buffer */
#ifdef QASR_CUDA_BACKEND_ENABLED
    int enc_out_dim = cuda_weights_->enc_output_dim;
    cudaMemcpy(output, session->enc_output.data(),
                enc_tokens * enc_out_dim * sizeof(float),
                cudaMemcpyDeviceToHost);
#endif
    return OkStatus();
}

Status CudaBackend::EncoderForward(void * session_ptr,
                                       const float * mel_features,
                                       int mel_frames,
                                       int * out_tokens) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    auto * session = static_cast<CudaSessionState *>(session_ptr);
    if (!session || !cuda_weights_ || !cuda_weights_->encoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                       "CUDA encoder not prepared or invalid session");
    }

    int enc_d_model = cuda_weights_->enc_d_model;
    int enc_ffn_dim = cuda_weights_->enc_ffn_dim;
    int enc_heads = cuda_weights_->enc_heads;
    int enc_head_dim = cuda_weights_->enc_head_dim;
    int enc_output_dim = cuda_weights_->enc_output_dim;
    int enc_layers = cuda_weights_->enc_layers;
    int chunk_size = cuda_weights_->enc_chunk_size;
    int n_window_infer = cuda_weights_->enc_n_window_infer;
    int mel_bins = 128;

    /* GPU conv2D stem + PE → GPU transformer (no CPU, no hot-path malloc) */

    /* Step 1: Upload mel features to GPU */
    float * d_mel = static_cast<float *>(session->enc_d_mel.data());
    cudaMemcpyAsync(d_mel, mel_features, mel_bins * mel_frames * sizeof(float),
                     cudaMemcpyHostToDevice, compute_stream_.stream());

    /* Step 2: Compute chunk dimensions */
    int tokens_per_chunk = 0;
    {
        int w = chunk_size;
        int w1 = (w + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        tokens_per_chunk = w3;
    }
    int n_chunks = (mel_frames + chunk_size - 1) / chunk_size;
    int total_tokens = 0;
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = start + chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = end - start;
        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        total_tokens += w3;
    }

    /* Conv2D output dimensions (full chunk) */
    int full_h1 = (mel_bins + 2 - 3) / 2 + 1;   /* 64 */
    int full_w1 = (chunk_size + 2 - 3) / 2 + 1;  /* 50 */
    int full_h2 = (full_h1 + 2 - 3) / 2 + 1;    /* 32 */
    int full_w2 = (full_w1 + 2 - 3) / 2 + 1;    /* 25 */
    int full_h3 = (full_h2 + 2 - 3) / 2 + 1;    /* 16 */
    int full_w3 = (full_w2 + 2 - 3) / 2 + 1;    /* 12 */
    int conv_proj_dim = QWEN_CONV_HIDDEN * full_h3; /* 7680 */

    /* GPU conv2D buffers (pre-allocated in session) */
    float * d_c1 = static_cast<float *>(session->enc_d_c1.data());
    float * d_c2 = static_cast<float *>(session->enc_d_c2.data());
    float * d_c3 = static_cast<float *>(session->enc_d_c3.data());
    float * d_reshape = static_cast<float *>(session->enc_d_reshape.data());
    float * d_proj = static_cast<float *>(session->enc_d_proj.data());
    float * d_pe = static_cast<float *>(session->enc_d_pe.data());

    /* Zero d_proj for accumulation */
    cudaMemsetAsync(d_proj, 0, total_tokens * enc_d_model * sizeof(float),
                     compute_stream_.stream());

    /* Step 3: GPU conv2D stem per chunk */
    int token_offset = 0;
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = start + chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = end - start;

/* Extract chunk mel on GPU */
        float * d_mel_chunk = static_cast<float *>(session->enc_d_mel_chunk.data());
        launch_extract_mel_chunk(compute_stream_.stream(),
                                    d_mel_chunk, d_mel,
                                    mel_bins, mel_frames,
                                    chunk_w, start);

        /* conv1 + GELU */
        int h1 = (mel_bins + 2 - 3) / 2 + 1;
        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        launch_conv2d(compute_stream_.stream(),
                         d_mel_chunk,
                         static_cast<float *>(cuda_weights_->enc_conv1_w.data()),
                         static_cast<float *>(cuda_weights_->enc_conv1_b.data()),
                         d_c1,
                         1, QWEN_CONV_HIDDEN, mel_bins, chunk_w,
                         3, 3, 2, 1, 1);  /* fused_gelu=1 */

        /* conv2 + GELU */
        int h2 = (h1 + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        launch_conv2d(compute_stream_.stream(),
                         d_c1,
                         static_cast<float *>(cuda_weights_->enc_conv2_w.data()),
                         static_cast<float *>(cuda_weights_->enc_conv2_b.data()),
                         d_c2,
                         QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h1, w1,
                         3, 3, 2, 1, 1);

        /* conv3 + GELU */
        int h3 = (h2 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        launch_conv2d(compute_stream_.stream(),
                         d_c2,
                         static_cast<float *>(cuda_weights_->enc_conv3_w.data()),
                         static_cast<float *>(cuda_weights_->enc_conv3_b.data()),
                         d_c3,
                         QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h2, w2,
                         3, 3, 2, 1, 1);

        /* Reshape: [c, h, w] → [w, c*h] */
        int cur_conv_proj_dim = QWEN_CONV_HIDDEN * h3;
        launch_reshape(compute_stream_.stream(),
                          d_reshape, d_c3,
                          QWEN_CONV_HIDDEN, h3, w3);

        /* Linear projection */
        float * d_proj_chunk = d_proj + (size_t)token_offset * enc_d_model;
        CublasGemm(w3, cur_conv_proj_dim, enc_d_model,
                      d_reshape,
                      static_cast<float *>(cuda_weights_->enc_conv_out_T_fp32.data()),
                      d_proj_chunk);

        /* Sinusoidal PE */
        launch_sinusoidal_pe(compute_stream_.stream(), d_pe, w3, enc_d_model);
        launch_add(compute_stream_.stream(), d_proj_chunk, d_proj_chunk, d_pe,
                     w3 * enc_d_model);

        token_offset += w3;
    }

    /* Step 4: Build window_starts on GPU */
    int window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
    int n_windows = (total_tokens + window_token_size - 1) / window_token_size;

    /* Use enc_workspace for transformer buffers and window_starts */
    float * enc_ws = static_cast<float *>(session->enc_workspace.data());
    size_t off = 0;
    float * d_x = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_x_norm = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_q = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_k = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_v = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_attn_out = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_proj_out = enc_ws + off; off += total_tokens * enc_d_model;
    float * d_ffn_mid = enc_ws + off; off += total_tokens * enc_ffn_dim;
    float * d_ffn_out = enc_ws + off; off += total_tokens * enc_d_model;

    /* Copy d_proj (conv2D stem output) to d_x */
    cudaMemcpyAsync(d_x, d_proj, total_tokens * enc_d_model * sizeof(float),
                     cudaMemcpyDeviceToDevice, compute_stream_.stream());

    /* window_starts at end of workspace (after float buffers, int-aligned) */
    size_t int_off = (off + 3) / 4 * 4; /* 4-byte align */
    int * d_window_starts = reinterpret_cast<int *>(enc_ws + int_off);
    {
        std::vector<int> h_ws(n_windows + 1);
        for (int w = 0; w < n_windows; w++) h_ws[w] = w * window_token_size;
        h_ws[n_windows] = total_tokens;
        cudaMemcpyAsync(d_window_starts, h_ws.data(), (n_windows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice, compute_stream_.stream());
    }

    float scale = 1.0f / sqrtf((float)enc_head_dim);

    /* Step 5: GPU transformer layers */
    for (int layer = 0; layer < enc_layers; layer++) {
        launch_layer_norm(compute_stream_.stream(),
                           d_x_norm, d_x,
                           static_cast<float *>(cuda_weights_->enc_attn_norm_w[layer].data()),
                           static_cast<float *>(cuda_weights_->enc_attn_norm_b[layer].data()),
                           total_tokens, enc_d_model, 1e-5f);

        CublasGemm(total_tokens, enc_d_model, enc_d_model,
                     d_x_norm,
                     static_cast<float *>(cuda_weights_->enc_wq[layer].data()),
                     d_q);
        launch_broadcast_add(compute_stream_.stream(), d_q, d_q,
                               static_cast<float *>(cuda_weights_->enc_wq_bias[layer].data()),
                               total_tokens, enc_d_model);

        CublasGemm(total_tokens, enc_d_model, enc_d_model,
                     d_x_norm,
                     static_cast<float *>(cuda_weights_->enc_wk[layer].data()),
                     d_k);
        launch_broadcast_add(compute_stream_.stream(), d_k, d_k,
                               static_cast<float *>(cuda_weights_->enc_wk_bias[layer].data()),
                               total_tokens, enc_d_model);

        CublasGemm(total_tokens, enc_d_model, enc_d_model,
                     d_x_norm,
                     static_cast<float *>(cuda_weights_->enc_wv[layer].data()),
                     d_v);
        launch_broadcast_add(compute_stream_.stream(), d_v, d_v,
                               static_cast<float *>(cuda_weights_->enc_wv_bias[layer].data()),
                               total_tokens, enc_d_model);

        cudaMemsetAsync(d_attn_out, 0, total_tokens * enc_d_model * sizeof(float),
                         compute_stream_.stream());
        launch_bidir_attention(compute_stream_.stream(),
                                  d_attn_out, d_q, d_k, d_v,
                                  total_tokens, enc_heads, enc_head_dim, scale,
                                  d_window_starts, n_windows);

        CublasGemm(total_tokens, enc_d_model, enc_d_model,
                     d_attn_out,
                     static_cast<float *>(cuda_weights_->enc_wo[layer].data()),
                     d_proj_out);
        launch_broadcast_add(compute_stream_.stream(), d_proj_out, d_proj_out,
                               static_cast<float *>(cuda_weights_->enc_wo_bias[layer].data()),
                               total_tokens, enc_d_model);
        launch_add(compute_stream_.stream(), d_x, d_x, d_proj_out,
                     total_tokens * enc_d_model);

        launch_layer_norm(compute_stream_.stream(),
                           d_x_norm, d_x,
                           static_cast<float *>(cuda_weights_->enc_ffn_norm_w[layer].data()),
                           static_cast<float *>(cuda_weights_->enc_ffn_norm_b[layer].data()),
                           total_tokens, enc_d_model, 1e-5f);

        CublasGemm(total_tokens, enc_d_model, enc_ffn_dim,
                     d_x_norm,
                     static_cast<float *>(cuda_weights_->enc_fc1[layer].data()),
                     d_ffn_mid);
        launch_broadcast_add(compute_stream_.stream(), d_ffn_mid, d_ffn_mid,
                               static_cast<float *>(cuda_weights_->enc_fc1_bias[layer].data()),
                               total_tokens, enc_ffn_dim);
        launch_gelu(compute_stream_.stream(), d_ffn_mid, d_ffn_mid,
                     total_tokens * enc_ffn_dim);

        CublasGemm(total_tokens, enc_ffn_dim, enc_d_model,
                     d_ffn_mid,
                     static_cast<float *>(cuda_weights_->enc_fc2[layer].data()),
                     d_ffn_out);
        launch_broadcast_add(compute_stream_.stream(), d_ffn_out, d_ffn_out,
                               static_cast<float *>(cuda_weights_->enc_fc2_bias[layer].data()),
                               total_tokens, enc_d_model);
        launch_add(compute_stream_.stream(), d_x, d_x, d_ffn_out,
                     total_tokens * enc_d_model);
    }

    /* Step 6: Final LayerNorm + proj1(GELU) + proj2 */
    launch_layer_norm(compute_stream_.stream(),
                        d_x, d_x,
                        static_cast<float *>(cuda_weights_->enc_ln_post_w.data()),
                        static_cast<float *>(cuda_weights_->enc_ln_post_b.data()),
                        total_tokens, enc_d_model, 1e-5f);

    CublasGemm(total_tokens, enc_d_model, enc_d_model,
                 d_x,
                 static_cast<float *>(cuda_weights_->enc_proj1.data()),
                 d_proj_out);
    launch_broadcast_add(compute_stream_.stream(), d_proj_out, d_proj_out,
                           static_cast<float *>(cuda_weights_->enc_proj1_bias.data()),
                           total_tokens, enc_d_model);
    launch_gelu(compute_stream_.stream(), d_proj_out, d_proj_out,
                 total_tokens * enc_d_model);

    CublasGemm(total_tokens, enc_d_model, enc_output_dim,
                 d_proj_out,
                 static_cast<float *>(cuda_weights_->enc_proj2.data()),
                 static_cast<float *>(session->enc_output.data()));
    launch_broadcast_add(compute_stream_.stream(),
                           static_cast<float *>(session->enc_output.data()),
                           static_cast<float *>(session->enc_output.data()),
                           static_cast<float *>(cuda_weights_->enc_proj2_bias.data()),
                           total_tokens, enc_output_dim);

    cudaStreamSynchronize(compute_stream_.stream());

    session->enc_output_tokens = total_tokens;
    *out_tokens = total_tokens;
    return OkStatus();
#else
    (void)session_ptr; (void)mel_features; (void)mel_frames; (void)out_tokens;
    return Status(StatusCode::kUnimplemented, "EncoderForward requires CUDA backend");
#endif
}

Status CudaBackend::DecoderPrefill(void * workspace_ptr,
                                     const float * encoder_output,
                                     int encoder_tokens,
                                     std::int32_t * input_tokens,
                                     int n_tokens) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    /* workspace_ptr is a CudaSessionState pointer */
    auto * session = static_cast<CudaSessionState *>(workspace_ptr);
    if (!session) {
        return Status(StatusCode::kFailedPrecondition,
                      "DecoderPrefill requires valid session workspace");
    }

    float * workspace = static_cast<float *>(session->workspace.data());
    float * kv_cache_k = static_cast<float *>(session->kv_cache_k.data());
    float * kv_cache_v = static_cast<float *>(session->kv_cache_v.data());

    int hidden = cuda_weights_->dec_hidden;
    int intermediate = cuda_weights_->dec_intermediate;
    int n_heads = cuda_weights_->dec_heads;
    int n_kv_heads = cuda_weights_->dec_kv_heads;
    int head_dim = cuda_weights_->dec_head_dim;
    int dec_layers = cuda_weights_->dec_layers;
    float eps = cuda_weights_->dec_rms_norm_eps;

    /* Step 1: Build RoPE cache on GPU */
  float * rope_cos = static_cast<float *>(session->rope_cos.data());
    float * rope_sin = static_cast<float *>(session->rope_sin.data());

/* Step 2: Embedding lookup for input tokens
     * Match CPU layout: embeddings for all tokens, with AUDIO_PAD positions
     * replaced by encoder output. */
    int * d_tokens = static_cast<int *>(session->d_tokens.data());
    cudaMemcpyAsync(d_tokens, input_tokens, (size_t)n_tokens * sizeof(int),
                      cudaMemcpyHostToDevice, compute_stream_.stream());

    /* Use pre-converted fp32 tok_embeddings (no per-prefill bf16→fp32) */
    float * token_embeddings = workspace;
    launch_embedding_lookup(compute_stream_.stream(),
                               token_embeddings, d_tokens,
                               static_cast<float *>(cuda_weights_->tok_embeddings_fp32.data()),
                               n_tokens, hidden);

    /* Step 3: Replace AUDIO_PAD positions with encoder output (match CPU) */
    /* Find AUDIO_PAD (151676) range in input_tokens */
    static const int AUDIO_PAD_TOKEN = 151676;
    int audio_pad_start = -1;
    for (int i = 0; i < n_tokens; i++) {
        if (input_tokens[i] == AUDIO_PAD_TOKEN) {
            audio_pad_start = i;
            break;
        }
    }

   if (audio_pad_start >= 0 && encoder_tokens > 0) {
        /* encoder_output is already on GPU (from EncoderForward) — device-to-device copy */
        cudaMemcpyAsync(token_embeddings + audio_pad_start * hidden,
                          encoder_output,
                          encoder_tokens * hidden * sizeof(float),
                          cudaMemcpyDeviceToDevice, compute_stream_.stream());
    }

   /* Total sequence length = n_tokens - 1 (match CPU: prefill all but last token)
     * The last token is reserved for the first autoregressive decode step. */
    int seq_len = n_tokens - 1;
    float * x = token_embeddings;

    /* Step 4: Decoder layers */
    for (int l = 0; l < dec_layers; l++) {
        DecoderLayerForward(l, seq_len, hidden, intermediate,
                               n_heads, n_kv_heads, head_dim, eps,
                               x, workspace,
                               rope_cos, rope_sin,
                               kv_cache_k, kv_cache_v,
                               session->current_seq_len);
    }

    session->current_seq_len = seq_len;
    session->prev_token = input_tokens[n_tokens - 1];  /* Last prompt token for autoregressive decode */
    return OkStatus();
#else
    (void)workspace_ptr; (void)encoder_output; (void)encoder_tokens;
    (void)input_tokens; (void)n_tokens;
    return Status(StatusCode::kUnimplemented,
                  "CUDA prefill not yet implemented (CUDA-3 stage)");
#endif
}

Status CudaBackend::DecodeStep(void * workspace_ptr, std::int32_t & out_token) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    auto * session = static_cast<CudaSessionState *>(workspace_ptr);
    if (!session) {
        return Status(StatusCode::kFailedPrecondition,
                      "DecodeStep requires valid session workspace");
    }

    int hidden = cuda_weights_->dec_hidden;
    int intermediate = cuda_weights_->dec_intermediate;
    int n_heads = cuda_weights_->dec_heads;
    int n_kv_heads = cuda_weights_->dec_kv_heads;
    int head_dim = cuda_weights_->dec_head_dim;
    int dec_layers = cuda_weights_->dec_layers;
    float eps = cuda_weights_->dec_rms_norm_eps;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int max_seq_len = 4096;  /* match AllocateSession */

    float * workspace = static_cast<float *>(session->workspace.data());

    /* --- CUDA Graph path --- */
    /* CUDA graph path disabled — unreliable on sm12.1 GB10, use fusion kernels instead */
    if (false && session->graph_ready) {
        CudaDecodeParams *d_params = static_cast<CudaDecodeParams *>(session->d_params.data());

        /* Write params directly on compute stream, then launch graph.
         * No cross-stream sync needed: same stream guarantees ordering. */
        launch_write_prev_token(compute_stream_.stream(), d_params, session->prev_token);
        launch_write_seq_pos(compute_stream_.stream(), d_params, session->current_seq_len);

        cudaGraphLaunch(session->graph_exec, compute_stream_.stream());
        cudaStreamSynchronize(compute_stream_.stream());

        /* Read logits for argmax (outside graph) */
        {
            size_t off = 0;
            off += max_seq_len * hidden;
            off += max_seq_len * q_dim;
            off += max_seq_len * kv_dim;
            off += max_seq_len * kv_dim;
            off += max_seq_len * max_seq_len;
            off += max_seq_len * q_dim;
            off += max_seq_len * hidden;
            off += max_seq_len * hidden;
            off += max_seq_len * (2 * intermediate);
            off += max_seq_len * hidden;

            float * logits = workspace + off;
            int best_idx = -1;
            float best_val = 0.0f;
            ArgMax(logits, cuda_weights_->vocab_size, &best_idx, &best_val);
            out_token = static_cast<std::int32_t>(best_idx);
        }

        session->current_seq_len++;
        session->prev_token = out_token;
        return OkStatus();
    }

    /* --- Regular (non-graph) path --- */
    float * kv_cache_k = static_cast<float *>(session->kv_cache_k.data());
    float * kv_cache_v = static_cast<float *>(session->kv_cache_v.data());
    float * rope_cos = static_cast<float *>(session->rope_cos.data());
    float * rope_sin = static_cast<float *>(session->rope_sin.data());

  /* Step 1: Lookup embedding for prev_token from pre-converted fp32 tok_embeddings */
    int prev_tok = session->prev_token;
    if (prev_tok < 0 || prev_tok >= cuda_weights_->vocab_size) {
        return Status(StatusCode::kInvalidArgument,
                      "DecodeStep: prev_token out of range: " + std::to_string(prev_tok));
    }
    size_t emb_offset = static_cast<size_t>(prev_tok) * hidden;

    /* Direct copy from pre-converted fp32 embeddings (no per-step bf16→fp32) */
    cudaMemcpyAsync(workspace,
                      static_cast<float *>(cuda_weights_->tok_embeddings_fp32.data()) + emb_offset,
                      hidden * sizeof(float),
                      cudaMemcpyDeviceToDevice, compute_stream_.stream());

    float * x = workspace;

/* Run decoder layers (offset RoPE by current_seq_len for correct position) */
    float * rope_cos_offset = rope_cos + session->current_seq_len * head_dim;
    float * rope_sin_offset = rope_sin + session->current_seq_len * head_dim;
    for (int l = 0; l < dec_layers; l++) {
        DecoderLayerForward(l, 1, hidden, intermediate,
                              n_heads, n_kv_heads, head_dim, eps,
                              x, workspace,
                              rope_cos_offset, rope_sin_offset,
                              kv_cache_k, kv_cache_v,
                              session->current_seq_len);
   }

 /* Final RMSNorm */
    launch_rms_norm(compute_stream_.stream(),
                       x, x,
                       static_cast<float *>(cuda_weights_->final_norm.data()),
                       1, hidden, eps);

/* logits[vocab] = x[hidden] @ W^T[hidden, vocab] — gemv for seq_len=1 */
    {
        size_t off = 0;
        off += max_seq_len * hidden;
        off += max_seq_len * q_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * max_seq_len;
        off += max_seq_len * q_dim;
        off += max_seq_len * hidden;
        off += max_seq_len * hidden;
        off += max_seq_len * (2 * intermediate);
        off += max_seq_len * hidden;
        off += cuda_weights_->vocab_size * hidden;
        off += max_seq_len * hidden;
        float * logits = workspace + off;
        launch_gemv(compute_stream_.stream(),
                      logits, x,
                      static_cast<float *>(cuda_weights_->lm_head_T_fp32.data()),
                      hidden, cuda_weights_->vocab_size);
    }
  cudaStreamSynchronize(compute_stream_.stream());

    /* Read logits for argmax */
    {
        int best_idx = -1;
        float best_val = 0.0f;
        size_t off = 0;
        off += max_seq_len * hidden;
        off += max_seq_len * q_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * max_seq_len;
        off += max_seq_len * q_dim;
        off += max_seq_len * hidden;
        off += max_seq_len * hidden;
        off += max_seq_len * (2 * intermediate);
        off += max_seq_len * hidden;
        off += cuda_weights_->vocab_size * hidden;
        off += max_seq_len * hidden;
        float * logits = workspace + off;
        ArgMax(logits, cuda_weights_->vocab_size, &best_idx, &best_val);
        session->current_seq_len++;
        out_token = static_cast<std::int32_t>(best_idx);
    }

  /* Store for next step */
    session->prev_token = out_token;

    return OkStatus();
#else
    (void)workspace_ptr; (void)out_token;
  return Status(StatusCode::kUnimplemented,
                   "CUDA decode step not yet implemented (CUDA-5 stage)");
#endif
}

/* Decoder layer forward for CUDA graph capture.
 * Same logic as DecoderLayerForward but uses graph-compatible kernels
 * that read dynamic params (seq_pos) from d_params struct. */
Status CudaBackend::DecoderLayerForwardGraph(int layer_idx,
                                               int hidden,
                                               int intermediate,
                                               int n_heads,
                                               int n_kv_heads,
                                               int head_dim,
                                               float eps,
                                               float * x,
                                               float * workspace,
                                               const float * rope_cos_base,
                                               const float * rope_sin_base,
                                               float * layer_k,
                                               float * layer_v,
                                               CudaDecodeParams *d_params) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Workspace offsets (same as DecoderLayerForward for seq_len=1) */
    float * x_norm = workspace;
    float * q = workspace + hidden;
    float * k = workspace + hidden + q_dim;
    float * v = workspace + hidden + q_dim + kv_dim;
    float * attn_out = workspace + hidden + q_dim + kv_dim + kv_dim;
    float * proj_out = workspace + hidden + q_dim + kv_dim + kv_dim + q_dim;
    float * post_norm = workspace + hidden + q_dim + kv_dim + kv_dim + q_dim + hidden;
    float * gate_up = workspace + hidden + q_dim + kv_dim + kv_dim + q_dim + hidden + hidden;
    float * ffn_out = workspace + hidden + q_dim + kv_dim + kv_dim + q_dim + hidden + hidden + 2 * intermediate;

    cudaStream_t stream = compute_stream_.stream();

    /* Step 1: Input RMSNorm */
    launch_rms_norm(stream,
                      x_norm, x,
                      static_cast<float *>(cuda_weights_->input_norm[layer_idx].data()),
                      1, hidden, eps);

    /* Step 2: QKV projections (gemv for seq_len=1) */
    launch_gemv(stream, q, x_norm,
                  static_cast<float *>(cuda_weights_->wq_T[layer_idx].data()),
                  hidden, q_dim);
    launch_gemv(stream, k, x_norm,
                  static_cast<float *>(cuda_weights_->wk_T[layer_idx].data()),
                  hidden, kv_dim);
    launch_gemv(stream, v, x_norm,
                  static_cast<float *>(cuda_weights_->wv_T[layer_idx].data()),
                  hidden, kv_dim);

    /* Step 3: Per-head Q/K RMSNorm */
    launch_rms_norm_per_head(stream, q,
                                static_cast<float *>(cuda_weights_->q_norm[layer_idx].data()),
                                1, n_heads, head_dim, eps);
    launch_rms_norm_per_head(stream, k,
                                static_cast<float *>(cuda_weights_->k_norm[layer_idx].data()),
                                1, n_kv_heads, head_dim, eps);

    /* Step 4: RoPE (graph-compatible: reads seq_pos from d_params) */
    launch_rope_neox_graph(stream, q, rope_cos_base, rope_sin_base, d_params, n_heads, head_dim);
    launch_rope_neox_graph(stream, k, rope_cos_base, rope_sin_base, d_params, n_kv_heads, head_dim);

    /* Step 5: KV cache store (graph-compatible: reads seq_pos from d_params) */
    launch_kv_cache_store(stream, k, v, layer_k, layer_v, d_params, kv_dim);

    /* Step 6: Causal attention (graph-compatible: reads seq_pos from d_params) */
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    launch_causal_attention_graph(stream, attn_out, q, layer_k, layer_v,
                                     d_params, n_heads, n_kv_heads, head_dim, attn_scale);

    /* Step 7: WO projection (gemv) */
    launch_gemv(stream, proj_out, attn_out,
                  static_cast<float *>(cuda_weights_->wo_T[layer_idx].data()),
                  q_dim, hidden);

    /* Step 8: Residual add */
    launch_add(stream, x, x, proj_out, hidden);

    /* Step 9: Post-attention RMSNorm */
    launch_rms_norm(stream, post_norm, x,
                      static_cast<float *>(cuda_weights_->post_attn_norm[layer_idx].data()),
                      1, hidden, eps);

    /* Step 10: SwiGLU MLP (gemv for seq_len=1) */
    launch_gemv(stream, gate_up, post_norm,
                  static_cast<float *>(cuda_weights_->gate_T[layer_idx].data()),
                  hidden, intermediate);
    launch_gemv(stream, gate_up + intermediate, post_norm,
                  static_cast<float *>(cuda_weights_->up_T[layer_idx].data()),
                  hidden, intermediate);

    /* SwiGLU */
    launch_swiglu(stream, ffn_out, gate_up, 1, intermediate);

    /* Step 11: Down projection (gemv) */
    launch_gemv(stream, proj_out, ffn_out,
                  static_cast<float *>(cuda_weights_->down_T[layer_idx].data()),
                  intermediate, hidden);

    /* Step 12: Residual add */
    launch_add(stream, x, x, proj_out, hidden);

    return OkStatus();
#else
    (void)layer_idx; (void)hidden; (void)intermediate;
    (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)eps;
    (void)x; (void)workspace; (void)rope_cos_base; (void)rope_sin_base;
    (void)layer_k; (void)layer_v; (void)d_params;
    return Status(StatusCode::kUnimplemented,
                  "DecoderLayerForwardGraph requires CUDA backend");
#endif
}

/* Capture DecodeStep compute graph for replay.
 *
 * Key design: d_params is written on the compute stream directly
 * BEFORE graph launch (NOT inside the graph). The captured graph
 * only contains the compute kernels that read d_params — no cross-stream
 * event sync needed.
 */
Status CudaBackend::CaptureDecodeGraph(CudaSessionState * session) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    cudaStream_t stream = compute_stream_.stream();
    int hidden = cuda_weights_->dec_hidden;
    int intermediate = cuda_weights_->dec_intermediate;
    int n_heads = cuda_weights_->dec_heads;
    int n_kv_heads = cuda_weights_->dec_kv_heads;
    int head_dim = cuda_weights_->dec_head_dim;
    int dec_layers = cuda_weights_->dec_layers;
    float eps = cuda_weights_->dec_rms_norm_eps;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int max_seq_len = 4096;

    /* Allocate d_params on device */
    session->d_params.Allocate(sizeof(CudaDecodeParams));
    CudaDecodeParams *d_params = static_cast<CudaDecodeParams *>(session->d_params.data());

    /* Write placeholder params on compute stream, THEN capture.
     * During replay, we write new params before each graph launch.
     * Both happen on the same stream, so no cross-stream sync needed. */
    CudaDecodeParams h_params = {0, 0};
    cudaMemcpy(d_params, &h_params, sizeof(CudaDecodeParams), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);

    float * workspace = static_cast<float *>(session->workspace.data());
    float * rope_cos = static_cast<float *>(session->rope_cos.data());
    float * rope_sin = static_cast<float *>(session->rope_sin.data());
    float * kv_cache_k = static_cast<float *>(session->kv_cache_k.data());
    float * kv_cache_v = static_cast<float *>(session->kv_cache_v.data());

    /* Begin graph capture (Relaxed mode allows non-captured work before) */
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        return Status(StatusCode::kInternal,
                      "cudaStreamBeginCapture failed: " + std::string(cudaGetErrorString(err)));
    }

    /* Embedding lookup from prev_token (reads from d_params) */
    {
        float * x = workspace;
        launch_embed_lookup_from_token(stream,
                                         x,
                                         static_cast<float *>(cuda_weights_->tok_embeddings_fp32.data()),
                                         d_params,
                                         hidden);
    }

    /* Run 28 decoder layers (x is modified in-place, workspace reused sequentially) */
    {
        float * x = workspace;
        for (int l = 0; l < dec_layers; l++) {
            float * layer_k = kv_cache_k + (size_t)l * max_seq_len * kv_dim;
            float * layer_v = kv_cache_v + (size_t)l * max_seq_len * kv_dim;
            DecoderLayerForwardGraph(l, hidden, intermediate,
                                       n_heads, n_kv_heads, head_dim, eps,
                                       x, workspace,
                                       rope_cos, rope_sin,
                                       layer_k, layer_v,
                                       d_params);
        }
    }

    /* Final RMSNorm + lm_head gemv */
    {
        size_t off = 0;
        off += max_seq_len * hidden;
        off += max_seq_len * q_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * kv_dim;
        off += max_seq_len * max_seq_len;
        off += max_seq_len * q_dim;
        off += max_seq_len * hidden;
        off += max_seq_len * hidden;
        off += max_seq_len * (2 * intermediate);
        off += max_seq_len * hidden;
        off += cuda_weights_->vocab_size * hidden;
        off += max_seq_len * hidden;
        float * logits = workspace + off;

        launch_rms_norm(stream,
                          workspace, workspace,
                          static_cast<float *>(cuda_weights_->final_norm.data()),
                          1, hidden, eps);

        launch_gemv(stream,
                      logits, workspace,
                      static_cast<float *>(cuda_weights_->lm_head_T_fp32.data()),
                      hidden, cuda_weights_->vocab_size);
    }

    /* End graph capture */
    cudaGraph_t graph;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        return Status(StatusCode::kInternal,
                      "cudaStreamEndCapture failed: " + std::string(cudaGetErrorString(err)));
    }

    /* Instantiate graph */
    cudaGraphExec_t instance;
    err = cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return Status(StatusCode::kInternal,
                      "cudaGraphInstantiate failed: " + std::string(cudaGetErrorString(err)));
    }

    session->graph = graph;
    session->graph_exec = instance;
    session->graph_ready = true;

    return OkStatus();
#else
    (void)session;
    return Status(StatusCode::kUnimplemented,
                  "CaptureDecodeGraph requires CUDA backend");
#endif
}

Status CudaBackend::ResetDecoder(void *) {
    return OkStatus();
}

/* Decoder layer forward pass (prefill mode)
 * Processes one transformer layer for a sequence of tokens.
 *
 * Layout:
 *   x_norm[seq, hidden] - RMSNorm output
 *   q[seq, q_dim]       - Q projection
 *   k[seq, kv_dim]      - K projection
 *   v[seq, kv_dim]      - V projection
 *   attn_out[seq, q_dim] - attention output
 *   proj_out[seq, hidden] - WO projection output
 *   gate_up[seq, 2*inter] - gate+up fused output
 *   ffn_out[seq, hidden]  - down projection output
 *
 * Workspace layout (sequential in workspace buffer):
 *   [0]       x_norm:         [seq, hidden]
 *   [1]       q:              [seq, q_dim]
 *   [2]       k:              [seq, kv_dim]
 *   [3]       v:              [seq, kv_dim]
 *   [4]       attn_score:     [seq, seq] (for attention)
 *   [5]       attn_out:       [seq, q_dim]
 *   [6]       proj_out:       [seq, hidden]
 *   [7]       post_norm:      [seq, hidden]
 *   [8]       gate_up:        [seq, 2*inter]
 *   [9]       ffn_out:        [seq, hidden]
 *   [10+]     W_fp32 buffers: temporary bf16->fp32 conversion
 */
Status CudaBackend::DecoderLayerForward(int layer_idx,
                                         int seq_len,
                                         int hidden,
                                         int intermediate,
                                         int n_heads,
                                         int n_kv_heads,
                                         int head_dim,
                                         float eps,
                                         float * x,                    /* [seq, hidden] input + residual */
                                         float * workspace,            /* workspace buffer */
                                         const float * rope_cos,       /* [seq, head_dim] */
                                         const float * rope_sin,       /* [seq, head_dim] */
                                         float * kv_cache_k,           /* [dec_layers, kv_cache_max, kv_dim] */
                                          float * kv_cache_v,           /* [dec_layers, kv_cache_max, kv_dim] */
                                          int kv_cache_offset) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_weights_ || !cuda_weights_->decoder_ready) {
        return Status(StatusCode::kFailedPrecondition,
                      "CUDA decoder weights not prepared");
    }

    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int max_seq_len = 4096;  /* match AllocateSession */

    /* Workspace offsets (row-major layout, no W_T_fp32 — weights pre-converted) */
    size_t off_x_norm = seq_len * hidden;
    size_t off_q = off_x_norm + seq_len * hidden;
    size_t off_k = off_q + seq_len * q_dim;
    size_t off_v = off_k + seq_len * kv_dim;
    size_t off_attn_score = off_v + seq_len * kv_dim;
    size_t off_attn_out = off_attn_score + seq_len * seq_len;
    size_t off_proj_out = off_attn_out + seq_len * q_dim;
    size_t off_post_norm = off_proj_out + seq_len * hidden;
    size_t off_gate_up = off_post_norm + seq_len * hidden;
    size_t off_ffn_out = off_gate_up + seq_len * (2 * intermediate);

    float * x_norm = workspace + off_x_norm;
    float * q = workspace + off_q;
    float * k = workspace + off_k;
    float * v = workspace + off_v;
    float * attn_score = workspace + off_attn_score;
    float * attn_out = workspace + off_attn_out;
    float * proj_out = workspace + off_proj_out;
    float * post_norm = workspace + off_post_norm;
    float * gate_up = workspace + off_gate_up;
    float * ffn_out = workspace + off_ffn_out;

/* ---- seq_len==1 path (DecodeStep): individual kernels (efficient) ---- */
    if (seq_len == 1) {
        cudaStream_t stream = compute_stream_.stream();

        /* Step 1: Input RMSNorm */
        launch_rms_norm(stream,
                          x_norm, x,
                          static_cast<float *>(cuda_weights_->input_norm[layer_idx].data()),
                          1, hidden, eps);

        /* Step 2: QKV projections (gemv_tiled — efficient, reads weights once) */
        launch_gemv(stream, q, x_norm,
                      static_cast<float *>(cuda_weights_->wq_T[layer_idx].data()),
                      hidden, q_dim);
        launch_gemv(stream, k, x_norm,
                      static_cast<float *>(cuda_weights_->wk_T[layer_idx].data()),
                      hidden, kv_dim);
        launch_gemv(stream, v, x_norm,
                      static_cast<float *>(cuda_weights_->wv_T[layer_idx].data()),
                      hidden, kv_dim);

        /* Step 3: Per-head Q/K RMSNorm */
        launch_rms_norm_per_head(stream, q,
                                    static_cast<float *>(cuda_weights_->q_norm[layer_idx].data()),
                                    1, n_heads, head_dim, eps);
        launch_rms_norm_per_head(stream, k,
                                    static_cast<float *>(cuda_weights_->k_norm[layer_idx].data()),
                                    1, n_kv_heads, head_dim, eps);

        /* Step 4: RoPE */
        launch_rope_neox(stream, q, rope_cos, rope_sin, 1, n_heads, head_dim);
        launch_rope_neox(stream, k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        /* KV cache store */
        float *layer_k = kv_cache_k + (size_t)layer_idx * max_seq_len * kv_dim;
        float *layer_v = kv_cache_v + (size_t)layer_idx * max_seq_len * kv_dim;
        cudaMemcpyAsync(layer_k + kv_cache_offset * kv_dim, k, kv_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(layer_v + kv_cache_offset * kv_dim, v, kv_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream);

        /* Causal attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        launch_causal_attention(stream, attn_out, q, layer_k, layer_v,
                                   1, kv_cache_offset + 1, n_heads, n_kv_heads, head_dim,
                                   attn_scale, kv_cache_offset);

        /* Step 7: WO projection */
        launch_gemv(stream, proj_out, attn_out,
                      static_cast<float *>(cuda_weights_->wo_T[layer_idx].data()),
                      q_dim, hidden);

        /* Step 8: Residual add */
        launch_add(stream, x, x, proj_out, hidden);

        /* Step 9: Post-attention RMSNorm */
        launch_rms_norm(stream, post_norm, x,
                          static_cast<float *>(cuda_weights_->post_attn_norm[layer_idx].data()),
                          1, hidden, eps);

        /* Step 10: SwiGLU MLP (gate/up gemv) */
        launch_gemv(stream, gate_up, post_norm,
                      static_cast<float *>(cuda_weights_->gate_T[layer_idx].data()),
                      hidden, intermediate);
        launch_gemv(stream, gate_up + intermediate, post_norm,
                      static_cast<float *>(cuda_weights_->up_T[layer_idx].data()),
                      hidden, intermediate);

        /* SwiGLU */
        launch_swiglu(stream, ffn_out, gate_up, 1, intermediate);

        /* Step 11: Down projection */
        launch_gemv(stream, proj_out, ffn_out,
                      static_cast<float *>(cuda_weights_->down_T[layer_idx].data()),
                      intermediate, hidden);

        /* Step 12: Residual add */
        launch_add(stream, x, x, proj_out, hidden);

        return OkStatus();
    }

    /* ---- Prefill path (seq_len > 1, uses cuBLAS) ---- */
/* Step 1: Input RMSNorm */
    launch_rms_norm(compute_stream_.stream(),
                      x_norm, x,
                      static_cast<float *>(cuda_weights_->input_norm[layer_idx].data()),
                      seq_len, hidden, eps);

    CublasGemm(seq_len, hidden, q_dim,
                 x_norm, static_cast<float *>(cuda_weights_->wq_T[layer_idx].data()), q);
    CublasGemm(seq_len, hidden, kv_dim,
                 x_norm, static_cast<float *>(cuda_weights_->wk_T[layer_idx].data()), k);
    CublasGemm(seq_len, hidden, kv_dim,
                 x_norm, static_cast<float *>(cuda_weights_->wv_T[layer_idx].data()), v);

    /* Step 3: Per-head Q/K RMSNorm */
    launch_rms_norm_per_head(compute_stream_.stream(),
                                q,
                                static_cast<float *>(cuda_weights_->q_norm[layer_idx].data()),
                                seq_len, n_heads, head_dim, eps);
    launch_rms_norm_per_head(compute_stream_.stream(),
                                k,
                                static_cast<float *>(cuda_weights_->k_norm[layer_idx].data()),
                                seq_len, n_kv_heads, head_dim, eps);

    /* Step 4: RoPE */
    launch_rope_neox(compute_stream_.stream(),
                       q, rope_cos, rope_sin,
                       seq_len, n_heads, head_dim);
    launch_rope_neox(compute_stream_.stream(),
                       k, rope_cos, rope_sin,
                       seq_len, n_kv_heads, head_dim);

    /* Step 5: KV cache store */
    float *layer_k_prefill = kv_cache_k + (size_t)layer_idx * max_seq_len * kv_dim;
    float *layer_v_prefill = kv_cache_v + (size_t)layer_idx * max_seq_len * kv_dim;
    size_t kv_size_prefill = seq_len * kv_dim * sizeof(float);
    cudaMemcpyAsync(layer_k_prefill + kv_cache_offset * kv_dim,
                      k, kv_size_prefill, cudaMemcpyDeviceToDevice, compute_stream_.stream());
    cudaMemcpyAsync(layer_v_prefill + kv_cache_offset * kv_dim,
                      v, kv_size_prefill, cudaMemcpyDeviceToDevice, compute_stream_.stream());

/* Step 6: Causal attention */
    float attn_scale_prefill = 1.0f / sqrtf((float)head_dim);
    launch_causal_attention(compute_stream_.stream(),
                                attn_out, q,
                                layer_k_prefill, layer_v_prefill,
                                seq_len, kv_cache_offset + seq_len,
                                n_heads, n_kv_heads, head_dim,
                                attn_scale_prefill, kv_cache_offset);

    CublasGemm(seq_len, q_dim, hidden,
                 attn_out, static_cast<float *>(cuda_weights_->wo_T[layer_idx].data()), proj_out);

    /* Step 8: Residual add */
    launch_add(compute_stream_.stream(),
                x, x, proj_out,
                seq_len * hidden);

    /* Step 9: Post-attention RMSNorm */
    launch_rms_norm(compute_stream_.stream(),
                     post_norm, x,
                     static_cast<float *>(cuda_weights_->post_attn_norm[layer_idx].data()),
                     seq_len, hidden, eps);

    CublasGemm(seq_len, hidden, intermediate,
                 post_norm, static_cast<float *>(cuda_weights_->gate_T[layer_idx].data()), gate_up);
    CublasGemm(seq_len, hidden, intermediate,
                 post_norm, static_cast<float *>(cuda_weights_->up_T[layer_idx].data()),
                 gate_up + seq_len * intermediate);

/* SwiGLU */
    launch_swiglu(compute_stream_.stream(),
                    ffn_out, gate_up,
                    seq_len, intermediate);

    CublasGemm(seq_len, intermediate, hidden,
                 ffn_out, static_cast<float *>(cuda_weights_->down_T[layer_idx].data()), proj_out);

   /* Step 12: Residual add */
    launch_add(compute_stream_.stream(),
                x, x, proj_out,
                seq_len * hidden);

    return OkStatus();
#else
    (void)layer_idx; (void)seq_len; (void)hidden; (void)intermediate;
    (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)eps;
    (void)x; (void)workspace; (void)rope_cos; (void)rope_sin;
    (void)kv_cache_k; (void)kv_cache_v; (void)kv_cache_offset;
    return Status(StatusCode::kUnimplemented,
                  "DecoderLayerForward requires CUDA backend");
#endif
}

 /* cuBLAS GEMM wrapper: C = alpha * op(A) * op(B) + beta * C
     * For decoder: Y = X @ W^T, where:
     *   X is [seq_len, in_dim] (fp32)
     *   W is [out_dim, in_dim] (bf16)
     *   Y is [seq_len, out_dim] (fp32)
     *
     * Strategy (verified by gemm_verify2.cu V7):
     * 1. Convert W from bf16 [out_dim, in_dim] to fp32 W_T [in_dim, out_dim] (transposed)
     * 2. cublasSgemm(N,N,out_dim,seq_len,in_dim, W_T, out_dim, X, in_dim, Y, out_dim)
     */
 Status CudaBackend::CublasGemm(int seq_len,
                                   int in_dim,
                                   int out_dim,
                                   const float * X,
                                   const float * W_T_fp32,
                                   float * Y,
                                   float alpha,
                                   float beta) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    /* Y[seq,out] = X[seq,in] @ W_T[in,out] — pre-transposed fp32, direct cublasSgemm */
    cublasStatus_t status = cublasSgemm(cublas_.handle(),
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            out_dim, seq_len, in_dim,
                                            &alpha,
                                            W_T_fp32, out_dim,
                                            X, in_dim,
                                            &beta,
                                            Y, out_dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return Status(StatusCode::kInternal,
                      "cublasSgemm failed: " + std::to_string(status));
    }
    return OkStatus();
#else
    (void)seq_len; (void)in_dim; (void)out_dim;
    (void)X; (void)W_T_fp32; (void)Y; (void)alpha; (void)beta;
    return Status(StatusCode::kUnimplemented,
                  "CublasGemm requires CUDA backend");
#endif
}

size_t CudaBackend::WorkspaceBytes(const V2EngineConfig & config) const {
    (void)config;
    /* Per-session workspace (lm_head W_T moved to PrepareWeights):
     * - KV cache: layers * seq_len * 2 * kv_heads * head_dim * sizeof(float)
     * - Decoder buffers: ~12 * hidden * sizeof(float) per layer
     * - Prefill buffers: seq_len * hidden * sizeof(float)
     * - DecodeStep lm_head: x[hidden] + logits[vocab] (W_T is pre-loaded)
     * For 1.7B: 28 layers, 2048 hidden, 128 head_dim, 8 kv_heads, 4096 max_seq
     *   vocab=151936, hidden=2048
     * KV: 28 * 4096 * 2 * 8 * 128 * 4 ≈ 92 MB
     * Buffers: ~2 MB
     * lm_head: (2048 + 151936) * 4 ≈ 616 KB
     * Total per session: ~100 MB */
    return 100ULL * 1024 * 1024;
}

/* --- Operator-level primitives --- */

Status CudaBackend::RmsNorm(float * out,
                              const float * x,
                              const float * weight,
                              int seq_len,
                              int hidden,
                              float eps) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    launch_rms_norm(compute_stream_.stream(),
                     out, x, weight, seq_len, hidden, eps);
    return OkStatus();
#else
    (void)out; (void)x; (void)weight; (void)seq_len; (void)hidden; (void)eps;
    return Status(StatusCode::kUnimplemented,
                  "RmsNorm requires CUDA backend (compile with -DQASR_ENABLE_CUDA_BACKEND=ON)");
#endif
}

Status CudaBackend::RmsNormPerHead(float * x,
                                     const float * weight,
                                     int seq_len,
                                     int n_heads,
                                     int head_dim,
                                     float eps) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    launch_rms_norm_per_head(compute_stream_.stream(),
                               x, weight, seq_len, n_heads, head_dim, eps);
    return OkStatus();
#else
    (void)x; (void)weight; (void)seq_len; (void)n_heads; (void)head_dim; (void)eps;
    return Status(StatusCode::kUnimplemented,
                  "RmsNormPerHead requires CUDA backend");
#endif
}

Status CudaBackend::ApplyRoPE(float * x,
                                const float * cos_vals,
                                const float * sin_vals,
                                int seq,
                                int n_heads,
                                int head_dim) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    launch_rope_neox(compute_stream_.stream(),
                       x, cos_vals, sin_vals, seq, n_heads, head_dim);
    return OkStatus();
#else
    (void)x; (void)cos_vals; (void)sin_vals; (void)seq; (void)n_heads; (void)head_dim;
    return Status(StatusCode::kUnimplemented,
                  "ApplyRoPE requires CUDA backend");
#endif
}

Status CudaBackend::SwiGLU(float * out,
                             const float * gate_up,
                             int seq_len,
                             int intermediate) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    launch_swiglu(compute_stream_.stream(),
                    out, gate_up, seq_len, intermediate);
    return OkStatus();
#else
    (void)out; (void)gate_up; (void)seq_len; (void)intermediate;
    return Status(StatusCode::kUnimplemented,
                  "SwiGLU requires CUDA backend");
#endif
}

Status CudaBackend::ArgMax(const float * logits,
                             int vocab_size,
                             int * out_idx,
                             float * out_val) {
#ifdef QASR_CUDA_BACKEND_ENABLED
    float h_val;
    int h_idx;

    /* Launch argmax kernel with device-side output buffers */
    float * d_val = nullptr;
    int * d_idx = nullptr;
    cudaError_t e = cudaMalloc(&d_val, sizeof(float));
    if (e != cudaSuccess) {
        return Status(StatusCode::kResourceExhausted, "cudaMalloc for argmax val failed");
    }
    e = cudaMalloc(&d_idx, sizeof(int));
    if (e != cudaSuccess) {
        cudaFree(d_val);
        return Status(StatusCode::kResourceExhausted, "cudaMalloc for argmax idx failed");
    }

    launch_argmax(compute_stream_.stream(),
                    logits, vocab_size, d_val, d_idx);

    e = cudaStreamSynchronize(compute_stream_.stream());
    if (e != cudaSuccess) {
        cudaFree(d_val);
        cudaFree(d_idx);
        return Status(StatusCode::kInternal, "cudaStreamSynchronize failed");
    }

    e = cudaMemcpy(&h_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaFree(d_val);
        cudaFree(d_idx);
        return Status(StatusCode::kInternal, "cudaMemcpy failed for argmax idx");
    }
    *out_idx = h_idx;

    if (out_val) {
        e = cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            cudaFree(d_val);
            cudaFree(d_idx);
            return Status(StatusCode::kInternal, "cudaMemcpy failed for argmax val");
        }
        *out_val = h_val;
    }

    cudaFree(d_val);
    cudaFree(d_idx);
    return OkStatus();
#else
    (void)logits; (void)vocab_size; (void)out_idx; (void)out_val;
    return Status(StatusCode::kUnimplemented,
                  "ArgMax requires CUDA backend");
#endif
}

}  // namespace qasr
