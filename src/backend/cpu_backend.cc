#include "qasr/backend/cpu_backend.h"
#include "qasr/core/status.h"

extern "C" {
#include "qwen_cpu/qwen_asr.h"
#include "qwen_cpu/qwen_asr_kernels.h"
}

namespace qasr {

CpuBackend::~CpuBackend() {
    Shutdown();
}

Status CpuBackend::Initialize() {
    return OkStatus();
}

Status CpuBackend::Shutdown() {
    if (weights_) {
        if (weights_->ctx) {
            qwen_free(reinterpret_cast<qwen_ctx_t *>(weights_->ctx));
            weights_->ctx = nullptr;
        }
        weights_.reset();
    }
    return OkStatus();
}

Status CpuBackend::PrepareWeights(const std::string & model_dir) {
    std::lock_guard<std::mutex> lock(mu_);
    weights_ = std::make_shared<CpuWeights>();
    weights_->model_dir = model_dir;
    weights_->ctx = qwen_load(model_dir.c_str());
    if (!weights_->ctx) {
        weights_.reset();
        return Status(StatusCode::kInternal, "qwen_load failed for " + model_dir);
    }
    return OkStatus();
}

Status CpuBackend::EncodeMel(void *,
                               const float * mel_features,
                               int mel_frames,
                               float * output,
                               int & out_tokens) {
    if (!weights_ || !weights_->ctx) {
        return Status(StatusCode::kFailedPrecondition,
                      "EncodeMel: weights not loaded");
    }
    auto * ctx = static_cast<qwen_ctx_t *>(weights_->ctx);
    float * enc_out = qwen_encoder_forward(ctx, mel_features, mel_frames, &out_tokens);
    if (!enc_out || out_tokens <= 0) {
        return Status(StatusCode::kInternal, "qwen_encoder_forward failed");
    }
    if (output) {
        std::memcpy(output, enc_out,
                    static_cast<size_t>(out_tokens) * ctx->config.enc_output_dim * sizeof(float));
    }
    return OkStatus();
}

Status CpuBackend::DecoderPrefill(void *,
                                    const float * encoder_output,
                                    int encoder_tokens,
                                    std::int32_t * input_tokens,
                                    int n_tokens) {
    (void)encoder_output;
    (void)encoder_tokens;
    (void)input_tokens;
    (void)n_tokens;
    return Status(StatusCode::kUnimplemented,
                  "DecoderPrefill not exposed via CPU backend; use TranscribeSegment directly");
}

Status CpuBackend::DecodeStep(void *, std::int32_t & out_token) {
    (void)out_token;
    return Status(StatusCode::kUnimplemented,
                  "DecodeStep not exposed via CPU backend; use TranscribeSegment directly");
}

Status CpuBackend::ResetDecoder(void *) {
    return OkStatus();
}

size_t CpuBackend::WorkspaceBytes(const V2EngineConfig & config) const {
    (void)config;
    return 0;
}

/* --- Operator-level primitives --- */

Status CpuBackend::RmsNorm(float * out,
                             const float * x,
                             const float * weight,
                             int seq_len,
                             int hidden,
                             float eps) {
    qwen_rms_norm(out, x, weight, seq_len, hidden, eps);
    return OkStatus();
}

Status CpuBackend::RmsNormPerHead(float * x,
                                    const float * weight,
                                    int seq_len,
                                    int n_heads,
                                    int head_dim,
                                    float eps) {
    qwen_rms_norm_per_head(x, weight, seq_len, n_heads, head_dim, eps);
    return OkStatus();
}

Status CpuBackend::ApplyRoPE(float * x,
                               const float * cos_vals,
                               const float * sin_vals,
                               int seq,
                               int n_heads,
                               int head_dim) {
    qwen_apply_rope_neox(x, cos_vals, sin_vals, seq, n_heads, head_dim);
    return OkStatus();
}

Status CpuBackend::SwiGLU(float * out,
                            const float * gate_up,
                            int seq_len,
                            int intermediate) {
    qwen_swiglu_multiply(out, gate_up, seq_len, intermediate);
    return OkStatus();
}

Status CpuBackend::ArgMax(const float * logits,
                            int vocab_size,
                            int * out_idx,
                            float * out_val) {
    /* Simple CPU argmax */
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    *out_idx = best;
    if (out_val) *out_val = best_val;
    return OkStatus();
}

void *CpuBackend::vadHandle() const {
    if (!weights_ || !weights_->ctx) return nullptr;
    auto * ctx = reinterpret_cast<qwen_ctx_t *>(weights_->ctx);
    return ctx->vad ? ctx->vad : nullptr;
}

}  // namespace qasr
