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
};

std::unique_ptr<AsrEngine> CreateEngine(BackendKind backend);

}  // namespace qasr
