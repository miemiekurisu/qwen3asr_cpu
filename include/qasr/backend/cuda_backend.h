#pragma once

#include "qasr/backend/device_backend.h"
#include "qasr/backend/cuda_decode_params.h"
#include "qasr/engine/config.h"
#include "qasr/core/status.h"
#include <memory>
#include <string>
#include <mutex>
#include <vector>
#include <cstdint>

#ifdef QASR_CUDA_BACKEND_ENABLED
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace qasr {

class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer();

#ifdef QASR_CUDA_BACKEND_ENABLED
    Status Allocate(size_t bytes);
    Status AllocateAsync(cudaStream_t stream, size_t bytes);
#else
    Status Allocate(size_t bytes);
    Status AllocateAsync(void *, size_t bytes) { return Allocate(bytes); }
#endif
    void Reset();

    void* data() const { return ptr_; }
    size_t size() const { return size_; }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer & operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer && other) noexcept;
    CudaBuffer & operator=(CudaBuffer && other) noexcept;

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

class CudaStreamHandle {
public:
    CudaStreamHandle() = default;
    ~CudaStreamHandle();

#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaStream_t stream() const { return stream_; }
    Status Create();
#else
    void *stream() const { return nullptr; }
    Status Create() { return OkStatus(); }
#endif

    CudaStreamHandle(const CudaStreamHandle &) = delete;
    CudaStreamHandle & operator=(const CudaStreamHandle &) = delete;

private:
#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaStream_t stream_ = nullptr;
#endif
};

class CublasHandle {
public:
    CublasHandle() = default;
    ~CublasHandle();

#ifdef QASR_CUDA_BACKEND_ENABLED
    cublasHandle_t handle() const { return handle_; }
    Status Create();
    Status SetStream(cudaStream_t stream);
#else
    void *handle() const { return nullptr; }
    Status Create() { return OkStatus(); }
    Status SetStream(void *) { return OkStatus(); }
#endif

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle & operator=(const CublasHandle &) = delete;

private:
#ifdef QASR_CUDA_BACKEND_ENABLED
    cublasHandle_t handle_ = nullptr;
#endif
};

class CudaWeights {
public:
    std::string model_dir;
    bool decoder_ready = false;
    bool lm_head_ready = false;
    bool encoder_ready = false;

    /* Decoder layer weights: fp32 pre-transposed [in_dim, out_dim] for cuBLAS.
     * Standard CUDA SDK pattern (llama.cpp, vLLM, TensorRT): convert bf16→fp32+transpose
     * ONCE at PrepareWeights, free bf16, use cublasSgemm directly at runtime.
     * This avoids per-inference bf16_transpose kernel launches (168× per forward). */
    std::vector<CudaBuffer> wq_T;     /* [dec_layers] fp32 [hidden, q_dim] */
    std::vector<CudaBuffer> wk_T;     /* [dec_layers] fp32 [hidden, kv_dim] */
    std::vector<CudaBuffer> wv_T;     /* [dec_layers] fp32 [hidden, kv_dim] */
    std::vector<CudaBuffer> wo_T;     /* [dec_layers] fp32 [q_dim, hidden] */
    std::vector<CudaBuffer> gate_T;   /* [dec_layers] fp32 [hidden, intermediate] */
    std::vector<CudaBuffer> up_T;     /* [dec_layers] fp32 [hidden, intermediate] */
    std::vector<CudaBuffer> down_T;   /* [dec_layers] fp32 [intermediate, hidden] */
    std::vector<CudaBuffer> input_norm;     /* [dec_layers] fp32 [hidden] */
    std::vector<CudaBuffer> post_attn_norm; /* [dec_layers] fp32 [hidden] */
    std::vector<CudaBuffer> q_norm;         /* [dec_layers] fp32 [head_dim] */
    std::vector<CudaBuffer> k_norm;         /* [dec_layers] fp32 [head_dim] */

    /* Global weights */
    CudaBuffer tok_embeddings_fp32; /* [vocab_size, hidden] fp32 */
    CudaBuffer final_norm;          /* [hidden] fp32 */
    CudaBuffer lm_head_T_fp32;      /* [hidden, vocab_size] fp32 */
    CudaBuffer inv_freq;            /* [head_dim/2] fp32 — RoPE frequency cache, built once */

    /* Encoder config */
    int enc_layers = 0;
    int enc_d_model = 0;
    int enc_heads = 0;
    int enc_head_dim = 0;
    int enc_ffn_dim = 0;
    int enc_output_dim = 0;
    int enc_chunk_size = 100;
    int enc_n_window_infer = 800;

    /* Encoder per-layer weights (bf16 weights, fp32 biases/norms) */
    std::vector<CudaBuffer> enc_wq;    /* [enc_layers] bf16 [d_model, d_model] */
    std::vector<CudaBuffer> enc_wk;    /* [enc_layers] bf16 [d_model, d_model] */
    std::vector<CudaBuffer> enc_wv;    /* [enc_layers] bf16 [d_model, d_model] */
    std::vector<CudaBuffer> enc_wo;    /* [enc_layers] bf16 [d_model, d_model] */
    std::vector<CudaBuffer> enc_fc1;   /* [enc_layers] bf16 [ffn_dim, d_model] */
    std::vector<CudaBuffer> enc_fc2;   /* [enc_layers] bf16 [d_model, ffn_dim] */
    std::vector<CudaBuffer> enc_wq_bias;   /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_wk_bias;   /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_wv_bias;   /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_wo_bias;   /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_fc1_bias;  /* [enc_layers] fp32 [ffn_dim] */
    std::vector<CudaBuffer> enc_fc2_bias;  /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_attn_norm_w; /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_attn_norm_b; /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_ffn_norm_w;  /* [enc_layers] fp32 [d_model] */
    std::vector<CudaBuffer> enc_ffn_norm_b;  /* [enc_layers] fp32 [d_model] */

    /* Conv2D stem weights (fp32) */
    CudaBuffer enc_conv1_w;  /* fp32 [480, 1, 3, 3] */
    CudaBuffer enc_conv1_b;  /* fp32 [480] */
    CudaBuffer enc_conv2_w;  /* fp32 [480, 480, 3, 3] */
    CudaBuffer enc_conv2_b;  /* fp32 [480] */
    CudaBuffer enc_conv3_w;  /* fp32 [480, 480, 3, 3] */
    CudaBuffer enc_conv3_b;  /* fp32 [480] */
    CudaBuffer enc_conv_out_T_fp32; /* fp32 pre-transposed [conv_proj_dim, d_model] */

    /* Encoder post-LayerNorm and projection weights */
    CudaBuffer enc_ln_post_w;   /* fp32 [d_model] */
    CudaBuffer enc_ln_post_b;   /* fp32 [d_model] */
    CudaBuffer enc_proj1;       /* bf16 [d_model, d_model] */
    CudaBuffer enc_proj1_bias;  /* fp32 [d_model] */
    CudaBuffer enc_proj2;       /* bf16 [output_dim, d_model] */
    CudaBuffer enc_proj2_bias;  /* fp32 [output_dim] */

    /* Model config */
    int dec_layers = 0;
    int dec_hidden = 0;
    int dec_intermediate = 0;
    int vocab_size = 0;
    int dec_heads = 0;
    int dec_kv_heads = 0;
    int dec_head_dim = 0;
    float dec_rms_norm_eps = 1e-6f;
};

class CudaSessionState : public BackendSessionState {
public:
    CudaSessionState() = default;
    ~CudaSessionState();

    CudaBuffer kv_cache_k;
    CudaBuffer kv_cache_v;
    CudaBuffer workspace;
    CudaBuffer rope_cos;  /* [max_seq, head_dim] */
    CudaBuffer rope_sin;  /* [max_seq, head_dim] */
    int current_seq_len = 0;
    int stream_index = 0;

    /* Decode state: previous token for autoregressive loop */
    int prev_token = 0;

    /* Encoder output buffer: stays on GPU for decoder prefill */
    CudaBuffer enc_output;       /* fp32 [total_tokens, enc_output_dim] */
    CudaBuffer enc_workspace;    /* encoder transformer scratch workspace */
    int enc_output_tokens = 0;

    /* Encoder conv2D stem workspace (pre-allocated, no hot-path malloc) */
    CudaBuffer enc_d_mel;        /* fp32 [mel_bins, mel_frames] */
    CudaBuffer enc_d_mel_chunk;  /* fp32 [mel_bins, max_chunk] */
    CudaBuffer enc_d_c1;         /* fp32 [480, max_h1, max_w1] */
    CudaBuffer enc_d_c2;         /* fp32 [480, max_h2, max_w2] */
    CudaBuffer enc_d_c3;         /* fp32 [480, max_h3, max_w3] */
    CudaBuffer enc_d_reshape;    /* fp32 [max_w3, conv_proj_dim] */
    CudaBuffer enc_d_proj;       /* fp32 [max_tokens, d_model] */
    CudaBuffer enc_d_pe;         /* fp32 [max_w3, d_model] */

    /* Decoder prefill: pre-allocated token buffer (avoids hot-path malloc) */
    CudaBuffer d_tokens;        /* int32 [max_seq_len] */
    bool rope_cache_built;      /* true once RoPE cache is initialized */

    /* CUDA Graph state for DecodeStep replay */
#ifdef QASR_CUDA_BACKEND_ENABLED
    CudaBuffer d_params;         /* CudaDecodeParams on device */
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t graph_sync_event = nullptr;  /* external event for param sync */
    CudaStreamHandle param_stream;           /* separate stream for param writes */
    bool graph_ready = false;
#endif
};

class CudaBackend final : public DeviceBackend {
public:
    CudaBackend() = default;
    ~CudaBackend() override;

    BackendKind kind() const override { return BackendKind::kCuda; }
    Status Initialize() override;
    Status Shutdown() override;
    Status PrepareWeights(const std::string & model_dir) override;

    Status EncodeMel(void * workspace,
                      const float * mel_features,
                      int mel_frames,
                      float * output,
                      int & out_tokens) override;

    /* Encoder forward: CPU conv2D + transformer, GPU output buffer.
     * Stores result in session->enc_output (GPU) and session->enc_output_tokens. */
    Status EncoderForward(void * session_ptr,
                           const float * mel_features,
                           int mel_frames,
                           int * out_tokens);

    Status DecoderPrefill(void * workspace,
                           const float * encoder_output,
                           int encoder_tokens,
                           std::int32_t * input_tokens,
                           int n_tokens) override;

    Status DecodeStep(void * workspace,
                       std::int32_t & out_token) override;

    Status ResetDecoder(void * workspace) override;
    size_t WorkspaceBytes(const V2EngineConfig & config) const override;

    Status RmsNorm(float * out, const float * x, const float * weight,
                    int seq_len, int hidden, float eps) override;
    Status RmsNormPerHead(float * x, const float * weight,
                           int seq_len, int n_heads, int head_dim, float eps) override;
    Status ApplyRoPE(float * x, const float * cos_vals, const float * sin_vals,
                      int seq, int n_heads, int head_dim) override;
    Status SwiGLU(float * out, const float * gate_up,
                   int seq_len, int intermediate) override;
    Status ArgMax(const float * logits, int vocab_size,
                   int * out_idx, float * out_val = nullptr) override;

/* cuBLAS GEMM wrapper: Y[seq_len, out_dim] = X[seq_len, in_dim] @ W_T[in_dim, out_dim]
     * W_T is fp32 pre-transposed [in_dim, out_dim] (prepared once in PrepareWeights).
     * Direct cublasSgemm — no runtime conversion overhead. */
    Status CublasGemm(int seq_len, int in_dim, int out_dim,
                        const float * X,
                        const float * W_T_fp32,       /* fp32 [in_dim, out_dim] pre-transposed */
                        float * Y,                    /* fp32 [seq_len, out_dim] */
                        float alpha = 1.0f, float beta = 0.0f);

    /* Decoder layer forward pass (prefill mode) */
    Status DecoderLayerForward(int layer_idx, int seq_len, int hidden, int intermediate,
                                int n_heads, int n_kv_heads, int head_dim, float eps,
                                float * x, float * workspace,
                                const float * rope_cos, const float * rope_sin,
                                float * kv_cache_k, float * kv_cache_v, int kv_cache_offset);

    /* Allocate session workspace and KV cache */
    Status AllocateSession(CudaSessionState * session, int max_seq_len);

    /* Capture DecodeStep compute graph for replay (CUDA graphs optimization) */
    Status CaptureDecodeGraph(CudaSessionState * session);

    /* Decoder layer forward for CUDA graph (reads seq_pos from d_params) */
    Status DecoderLayerForwardGraph(int layer_idx, int hidden, int intermediate,
                                     int n_heads, int n_kv_heads, int head_dim, float eps,
                                     float * x, float * workspace,
                                     const float * rope_cos_base, const float * rope_sin_base,
                                     float * layer_k, float * layer_v,
                                     CudaDecodeParams *d_params);

    int device_id() const { return device_id_; }
    const CudaWeights * cuda_weights() const { return cuda_weights_.get(); }
#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaDeviceProp & device_prop() { return device_prop_; }
#endif

private:
    int device_id_ = 0;
#ifdef QASR_CUDA_BACKEND_ENABLED
    cudaDeviceProp device_prop_;
#endif
    CudaStreamHandle compute_stream_;
    CublasHandle cublas_;

    std::vector<CudaStreamHandle> stream_pool_;
    std::vector<CublasHandle> cublas_pool_;

    std::shared_ptr<CudaWeights> cuda_weights_;
    std::mutex mu_;
    bool initialized_ = false;
};

}  // namespace qasr
