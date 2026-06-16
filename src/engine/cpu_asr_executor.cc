#include "qasr/engine/cpu_asr_engine.h"
#include <cstring>

extern "C" {
#include "qwen_asr.h"
}

namespace qasr {

Status QwenExecutor::Initialize(void * base_ctx, const V2EngineConfig & config) {
    (void)config;
    if (!base_ctx) {
        return Status(StatusCode::kFailedPrecondition,
                      "QwenExecutor::Initialize base_ctx is null");
    }
    base_ctx_ = base_ctx;
    return OkStatus();
}

Status QwenExecutor::EncodeMel(void * workspace,
                                 const float * mel,
                                 int mel_frames,
                                 float * output,
                                 int & out_tokens) {
    if (!base_ctx_) {
        return Status(StatusCode::kFailedPrecondition,
                      "QwenExecutor not initialized");
    }
    (void)workspace;
    auto * ctx = static_cast<qwen_ctx_t *>(base_ctx_);
    float * enc_out = qwen_encoder_forward(ctx, mel, mel_frames, &out_tokens);
    if (!enc_out || out_tokens <= 0) {
        return Status(StatusCode::kInternal, "qwen_encoder_forward failed");
    }
    if (output) {
        std::memcpy(output, enc_out,
                    static_cast<size_t>(out_tokens) *
                    static_cast<size_t>(ctx->config.enc_output_dim) * sizeof(float));
    }
    return OkStatus();
}

Status QwenExecutor::Prefill(void * workspace,
                               const float * enc_out,
                               int enc_tokens,
                               const std::vector<std::int32_t> & input_tokens) {
    if (!base_ctx_) {
        return Status(StatusCode::kFailedPrecondition,
                      "QwenExecutor not initialized");
    }
    (void)workspace;
    (void)enc_out;
    auto * ctx = static_cast<qwen_ctx_t *>(base_ctx_);

    if (enc_tokens <= 0 || input_tokens.empty()) {
        return Status(StatusCode::kFailedPrecondition,
                      "Prefill requires encoder output and input tokens");
    }

    int total_seq = enc_tokens + static_cast<int>(input_tokens.size());
    if (total_seq > ctx->kv_cache_max) {
        return Status(StatusCode::kResourceExhausted,
                      "prefill sequence length exceeds KV cache capacity");
    }

    return Status(StatusCode::kUnimplemented,
                  "QwenExecutor::Prefill — embedding lookup not exposed by C API "
                  "(will be replaced by CUDA backend prefill)");
}

Status QwenExecutor::DecodeStep(void * workspace, std::int32_t & out_token) {
    if (!base_ctx_) {
        return Status(StatusCode::kFailedPrecondition,
                      "QwenExecutor not initialized");
    }
    (void)workspace;
    auto * ctx = static_cast<qwen_ctx_t *>(base_ctx_);

    /* qwen_decoder_forward expects a single input embedding pointer and
     * returns the next token ID.  For the first decode step, the caller
     * must have already run prefill.  Subsequent steps use the last
     * generated token's embedding. */
    if (ctx->kv_cache_len <= 0) {
        return Status(StatusCode::kFailedPrecondition,
                      "DecodeStep requires prefill to be called first");
    }

    out_token = qwen_decoder_forward(ctx, nullptr);
    return OkStatus();
}

Status QwenExecutor::ResetDecoder(void * workspace) {
    if (!base_ctx_) {
        return Status(StatusCode::kFailedPrecondition,
                      "QwenExecutor not initialized");
    }
    (void)workspace;
    auto * ctx = static_cast<qwen_ctx_t *>(base_ctx_);
    ctx->kv_cache_len = 0;
    return OkStatus();
}

}  // namespace qasr
