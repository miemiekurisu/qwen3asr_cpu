#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "qasr/core/status.h"

namespace qasr {

inline constexpr std::int32_t kDefaultStreamMaxNewTokens = 32;
inline constexpr std::int32_t kMaxStreamMaxNewTokens = 128;

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
    std::string prompt;
    std::string language;
};

struct AsrRunResult {
    Status status;
    std::string text;
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

}  // namespace qasr
