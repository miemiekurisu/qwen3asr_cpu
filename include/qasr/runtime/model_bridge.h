#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/core/timestamp.h"

namespace qasr {

inline constexpr std::int32_t kDefaultStreamMaxNewTokens = 32;
inline constexpr std::int32_t kMaxStreamMaxNewTokens = 128;

/// A single transcript segment with timing.
struct TimedSegment {
    std::string text;
    TimestampRange range;  // begin_ms / end_ms
};

struct AsrRunOptions {
    std::string model_dir;
    std::string audio_path;
    std::int32_t threads = 0;
    std::int32_t stream_max_new_tokens = kDefaultStreamMaxNewTokens;
    std::int32_t segment_max_codepoints = 48;
    std::int32_t verbosity = 0;
    bool stream = false;
    bool emit_tokens = false;
    bool emit_segments = false;
    bool decoder_int8 = false;
    bool encoder_int8 = false;
    float temperature = -1.0f;
    std::string prompt;
    std::string language;
};

struct AsrRunResult {
    Status status;
    std::string text;
    std::vector<TimedSegment> segments;  // populated when segmented transcription used
    double total_ms = 0.0;
    std::int32_t text_tokens = 0;
    double audio_ms = 0.0;
    double encode_ms = 0.0;
    double decode_ms = 0.0;
};

bool CpuBackendAvailable() noexcept;
Status ValidateModelDirectory(const std::string & model_dir);
Status ValidateAsrRunOptions(const AsrRunOptions & options);
bool ShouldFlushAsrSegment(std::string_view text, std::int32_t max_codepoints) noexcept;
AsrRunResult RunAsr(const AsrRunOptions & options);

/// Transcribe audio and return timed segments (for subtitle generation).
/// Pre: options.model_dir valid, options.audio_path exists.
/// Post: result.segments populated with per-segment timestamps.
/// Thread-safe: yes (creates own context).
AsrRunResult RunAsrSegmented(const AsrRunOptions & options);

/// Callback invoked after each segment is transcribed.
/// Parameters: 0-based index, segment.
using SegmentCallback = std::function<void(int index, const TimedSegment & segment)>;

/// Like RunAsrSegmented but invokes callback after each segment finishes,
/// enabling incremental subtitle output. Existing behaviour unchanged when
/// callback is empty.
/// Pre: same as RunAsrSegmented.
/// Post: callback called for each segment in order, then result returned.
AsrRunResult RunAsrSegmentedStreaming(const AsrRunOptions & options,
                                     SegmentCallback on_segment);

}  // namespace qasr
