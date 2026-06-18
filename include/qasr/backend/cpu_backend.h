#pragma once

#include "qasr/backend/device_backend.h"
#include "qasr/engine/config.h"
#include <memory>
#include <string>
#include <mutex>
#include <vector>

namespace qasr {

class CpuWeights {
public:
    std::string model_dir;
    void * ctx = nullptr;
};

class CpuSessionState : public BackendSessionState {
public:
    void * ctx_clone = nullptr;
    std::vector<float> workspace;
    int workspace_size = 0;
};

class CpuBackend final : public DeviceBackend {
public:
    CpuBackend() = default;
    ~CpuBackend() override;

    BackendKind kind() const override { return BackendKind::kCpu; }
    Status Initialize() override;
    Status Shutdown() override;
    Status PrepareWeights(const std::string & model_dir) override;

    Status EncodeMel(void * workspace,
                      const float * mel_features,
                      int mel_frames,
                      float * output,
                      int & out_tokens) override;

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

    void * base_ctx() const { return weights_ ? weights_->ctx : nullptr; }
    // Return the Silero VAD handle owned by the loaded model context.
    // CPU only; returns nullptr if model not loaded or VAD unavailable.
    void * vadHandle() const;

private:
    std::shared_ptr<CpuWeights> weights_;
    std::mutex mu_;
};

}  // namespace qasr
