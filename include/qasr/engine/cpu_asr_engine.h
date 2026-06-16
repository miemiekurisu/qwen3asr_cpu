#pragma once

#include "qasr/engine/asr_engine.h"
#include "qasr/backend/cpu_backend.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <functional>

namespace qasr {

class QwenExecutor {
public:
    QwenExecutor() = default;
    ~QwenExecutor() = default;

    Status Initialize(void * base_ctx, const V2EngineConfig & config);
    Status EncodeMel(void * workspace,
                      const float * mel, int mel_frames,
                      float * output, int & out_tokens);
    Status Prefill(void * workspace,
                    const float * enc_out, int enc_tokens,
                    const std::vector<std::int32_t> & input_tokens);
    Status DecodeStep(void * workspace, std::int32_t & out_token);
    Status ResetDecoder(void * workspace);

private:
    void * base_ctx_ = nullptr;
};

class CpuSessionHandle final : public SessionHandle {
public:
    CpuSessionHandle(void * ctx, AsrEngine * engine, std::uint64_t id)
        : ctx_(ctx), engine_(engine), id_(id) {}
    void *nativeCtx() const override { return ctx_; }
    AsrEngine *engine() const override { return engine_; }
    std::uint64_t sessionId() const override { return id_; }

private:
    void * ctx_;
    AsrEngine * engine_;
    std::uint64_t id_;
};

class CpuAsrEngine final : public AsrEngine {
public:
    CpuAsrEngine() = default;
    ~CpuAsrEngine() override = default;

    Status LoadModel(const V2EngineConfig & config) override;
    Status CreateSession(const SessionOptions & opts, std::uint64_t & out_id) override;
    Status CloseSession(std::uint64_t session_id) override;
    AsrSegmentResult TranscribeSegment(std::uint64_t session_id,
                                        const std::vector<float> & samples,
                                        int64_t sample_rate = 16000,
                                        TokenCallback on_token = {},
                                        CancelCallback on_cancel = {}) override;
    int ActiveSessionCount() const override;
    const V2EngineConfig & config() const override { return config_; }
    std::unique_ptr<SessionHandle> CreateRealtimeSession(
        const SessionOptions & opts) override;
    void CloseSessionHandle(std::unique_ptr<SessionHandle>) override;
    void *getVadHandle() const override;

    // Expose the underlying C context for server-side C bridge calls.
    // The facade knows C API; the engine owns the context lifetime.
    void *base_ctx() const { return backend_ ? backend_->base_ctx() : nullptr; }

private:
    struct CpuSessionEntry {
        SessionOptions opts;
        bool active = true;
    };

    V2EngineConfig config_;
    std::shared_ptr<CpuBackend> backend_;
    QwenExecutor executor_;
    std::unordered_map<std::uint64_t, CpuSessionEntry> sessions_;
    uint64_t next_session_id_ = 1;
    mutable std::mutex mu_;
};

}  // namespace qasr
