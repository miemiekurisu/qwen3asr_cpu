#pragma once

#include "qasr/engine/types.h"
#include "qasr/engine/config.h"
#include "qasr/core/status.h"
#include "qasr/core/timestamp.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace qasr {

struct SessionOptions {
    std::string language;
    std::string prompt;
    std::int32_t stream_max_new_tokens = 32;
    float temperature = -1.0f;
    bool realtime = false;
};

/* Engine v2 segment result — mirrors model_bridge::TimedSegment. */
struct V2TimedSegment {
    std::string text;
    TimestampRange range;
};

struct AsrSegmentResult {
    Status status;
    std::string text;
    std::vector<V2TimedSegment> segments;
    double audio_ms = 0.0;
    double encode_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    int text_tokens = 0;
};

using TokenCallback = std::function<void(std::string_view)>;
using CancelCallback = std::function<bool()>;

class AsrEngine;

/// Lightweight handle to a per-session inference context.
/// CPU: nativeCtx() returns qwen_ctx_t* for C bridge interop.
/// CUDA: nativeCtx() returns nullptr; use engine()->TranscribeSegment().
class SessionHandle {
public:
    virtual ~SessionHandle() = default;
    virtual void *nativeCtx() const = 0;
    virtual AsrEngine *engine() const = 0;
    virtual std::uint64_t sessionId() const = 0;
};

class AsrEngine {
public:
    AsrEngine() = default;
    virtual ~AsrEngine() = default;

    virtual Status LoadModel(const V2EngineConfig & config) = 0;
    virtual Status CreateSession(const SessionOptions & opts, std::uint64_t & out_id) = 0;
    virtual Status CloseSession(std::uint64_t session_id) = 0;
    virtual AsrSegmentResult TranscribeSegment(std::uint64_t session_id,
                                                const std::vector<float> & samples,
                                                int64_t sample_rate = 16000,
                                                TokenCallback on_token = {},
                                                CancelCallback on_cancel = {}) = 0;
    virtual int ActiveSessionCount() const = 0;
    virtual const V2EngineConfig & config() const = 0;

    // Create a realtime clone for a VAD/ASR worker thread.
    // CPU returns a handle with nativeCtx() pointing to a qwen_clone_shared().
    // CUDA returns a handle with nativeCtx() == nullptr, using engine() instead.
    virtual std::unique_ptr<SessionHandle> CreateRealtimeSession(
        const SessionOptions & opts) = 0;

    // Release a realtime session handle.
    virtual void CloseSessionHandle(std::unique_ptr<SessionHandle>) = 0;

    // Return shared VAD handle (CPU only, for Silero VAD).
    // Returns nullptr on CUDA backend (VAD is always CPU).
    virtual void *getVadHandle() const = 0;

    // Return underlying C context (for C bridge calls in ServerAsrFacade).
    // CPU: returns qwen_ctx_t* from CpuBackend.
    // CUDA: returns nullptr (or CPU fallback ctx when applicable).
    virtual void *base_ctx() const { return nullptr; }
};

std::unique_ptr<AsrEngine> CreateEngine(BackendKind backend);

}  // namespace qasr
