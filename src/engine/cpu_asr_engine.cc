#include "qasr/engine/cpu_asr_engine.h"
#include "qasr/backend/cpu_backend.h"
#include <cstdio>

extern "C" {
#include "qwen_asr.h"
}

namespace qasr {

Status CpuAsrEngine::LoadModel(const V2EngineConfig & config) {
    config_ = config;
    backend_ = std::make_shared<CpuBackend>();
    auto status = backend_->PrepareWeights(config_.model_dir);
    if (!status.ok()) {
        return status;
    }
    return OkStatus();
}

Status CpuAsrEngine::CreateSession(const SessionOptions & opts, std::uint64_t & out_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int>(sessions_.size()) >= config_.max_sessions) {
        return Status(StatusCode::kResourceExhausted,
                      "max sessions reached: " + std::to_string(config_.max_sessions));
    }
    auto * base = reinterpret_cast<qwen_ctx_t *>(backend_->base_ctx());
    if (!base) {
        return Status(StatusCode::kFailedPrecondition, "model not loaded");
    }
    auto * clone = qwen_clone_shared(base);
    if (!clone) {
        return Status(StatusCode::kResourceExhausted, "qwen_clone_shared failed");
    }
    out_id = next_session_id_++;
    CpuSessionEntry entry;
    entry.opts = opts;
    entry.active = true;
    entry.ctx_clone = clone;
    sessions_[out_id] = std::move(entry);
    return OkStatus();
}

Status CpuAsrEngine::CloseSession(std::uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        qwen_free(static_cast<qwen_ctx_t *>(it->second.ctx_clone));
        it->second.active = false;
        sessions_.erase(it);
    }
    return OkStatus();
}

AsrSegmentResult CpuAsrEngine::TranscribeSegment(std::uint64_t session_id,
                                                    const std::vector<float> & samples,
                                                    int64_t sample_rate,
                                                    TokenCallback on_token,
                                                    CancelCallback on_cancel) {
    (void)sample_rate;
    (void)on_token;
    (void)on_cancel;
    AsrSegmentResult result;

    CpuSessionEntry * session = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end() || !it->second.active) {
            result.status = Status(StatusCode::kFailedPrecondition,
                                   "session not found: " + std::to_string(session_id));
            return result;
        }
        session = &it->second;
    }

    auto * base = static_cast<qwen_ctx_t *>(session->ctx_clone);
    if (!base) {
        result.status = Status(StatusCode::kFailedPrecondition, "session ctx not available");
        return result;
    }

    const char * lang = session->opts.language.empty()
        ? (config_.language.empty() ? nullptr : config_.language.c_str())
        : session->opts.language.c_str();
    const char * prompt = session->opts.prompt.empty()
        ? (config_.prompt.empty() ? nullptr : config_.prompt.c_str())
        : session->opts.prompt.c_str();

    qwen_set_force_language(base, lang);
    qwen_set_prompt(base, prompt);

    float temp = session->opts.temperature >= 0.0f
        ? session->opts.temperature
        : config_.temperature;
    if (temp >= 0.0f) {
        base->decode_temperature = temp;
    }

    char * raw = qwen_transcribe_audio(base, samples.data(), static_cast<int>(samples.size()));
    if (!raw) {
        result.status = Status(StatusCode::kInternal, "transcription failed");
        return result;
    }

    result.text = raw;
    std::free(raw);
    result.total_ms = base->perf_total_ms;
    result.audio_ms = base->perf_audio_ms;
    result.text_tokens = base->perf_text_tokens;
    result.encode_ms = base->perf_encode_ms;
    result.decode_ms = base->perf_decode_ms;
    result.status = OkStatus();
    return result;
}

int CpuAsrEngine::ActiveSessionCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(sessions_.size());
}

std::unique_ptr<SessionHandle> CpuAsrEngine::CreateRealtimeSession(
    const SessionOptions & opts) {
    (void)opts;
    auto * base = reinterpret_cast<qwen_ctx_t *>(backend_->base_ctx());
    if (!base) {
        return nullptr;
    }
    auto * clone = qwen_clone_shared(base);
    if (!clone) {
        return nullptr;
    }
    std::uint64_t id = next_session_id_++;
    return std::make_unique<CpuSessionHandle>(clone, this, id);
}

void CpuAsrEngine::CloseSessionHandle(std::unique_ptr<SessionHandle> handle) {
    if (!handle) return;
    auto * ctx = static_cast<qwen_ctx_t *>(handle->nativeCtx());
    if (ctx) {
        qwen_free(ctx);
    }
}

void *CpuAsrEngine::getVadHandle() const {
    if (!backend_) return nullptr;
    return backend_->vadHandle();
}

}  // namespace qasr
