#include "qasr/core/inference_backend.h"

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
}
#endif

namespace qasr {

#ifdef QASR_CPU_BACKEND_ENABLED

class CpuBackend final : public InferenceBackend {
public:
    ~CpuBackend() override {
        if (ctx_) {
            qwen_free(ctx_);
        }
    }

    Status Load(const std::string & model_dir, int threads) override {
        if (ctx_) {
            qwen_free(ctx_);
            ctx_ = nullptr;
        }
        const int n = threads > 0 ? threads : qwen_get_num_cpus();
        qwen_set_threads(n);
        ctx_ = qwen_load(model_dir.c_str());
        if (!ctx_) {
            return Status(StatusCode::kInternal, "qwen_load failed for: " + model_dir);
        }
        return OkStatus();
    }

    Status Encode(const float * mel, std::int32_t mel_frames,
                  std::vector<float> * output,
                  std::int32_t * out_seq_len) override {
        if (!ctx_) {
            return Status(StatusCode::kFailedPrecondition, "model not loaded");
        }
        if (!mel || !output || !out_seq_len) {
            return Status(StatusCode::kInvalidArgument, "null pointer argument");
        }
        int seq_len = 0;
        float * enc = qwen_encoder_forward(ctx_, mel, static_cast<int>(mel_frames), &seq_len);
        if (!enc || seq_len <= 0) {
            return Status(StatusCode::kInternal, "encoder forward failed");
        }
        const auto dim = static_cast<std::size_t>(ctx_->config.enc_output_dim);
        const auto total = static_cast<std::size_t>(seq_len) * dim;
        output->assign(enc, enc + total);
        *out_seq_len = static_cast<std::int32_t>(seq_len);
        return OkStatus();
    }

    Status Prefill(const float * embeddings, std::int32_t seq_len) override {
        if (!ctx_) {
            return Status(StatusCode::kFailedPrecondition, "model not loaded");
        }
        if (!embeddings || seq_len <= 0) {
            return Status(StatusCode::kInvalidArgument, "invalid prefill arguments");
        }
        qwen_decoder_prefill(ctx_, embeddings, static_cast<int>(seq_len));
        return OkStatus();
    }

    Status DecodeStep(const float * embed,
                      std::int32_t * out_token_id) override {
        if (!ctx_) {
            return Status(StatusCode::kFailedPrecondition, "model not loaded");
        }
        if (!embed || !out_token_id) {
            return Status(StatusCode::kInvalidArgument, "null pointer argument");
        }
        *out_token_id = static_cast<std::int32_t>(qwen_decoder_forward(ctx_, embed));
        return OkStatus();
    }

    void ResetDecoder() override {
        if (ctx_) {
            ctx_->kv_cache_len = 0;
        }
    }

    bool IsLoaded() const noexcept override {
        return ctx_ != nullptr;
    }

    std::int32_t EncoderOutputDim() const noexcept override {
        return ctx_ ? static_cast<std::int32_t>(ctx_->config.enc_output_dim) : 0;
    }

    std::int32_t DecoderHiddenDim() const noexcept override {
        return ctx_ ? static_cast<std::int32_t>(ctx_->config.dec_hidden) : 0;
    }

private:
    qwen_ctx_t * ctx_ = nullptr;
};

std::unique_ptr<InferenceBackend> CreateCpuBackend() {
    return std::make_unique<CpuBackend>();
}

#else  // !QASR_CPU_BACKEND_ENABLED

std::unique_ptr<InferenceBackend> CreateCpuBackend() {
    return nullptr;
}

#endif

}  // namespace qasr
