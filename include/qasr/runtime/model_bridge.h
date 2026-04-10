#pragma once

#include <cstdint>
#include <string>

#include "qasr/core/status.h"

namespace qasr {

struct AsrRunOptions {
    std::string model_dir;
    std::string audio_path;
    std::int32_t threads = 0;
    std::int32_t stream_max_new_tokens = 32;
    std::int32_t verbosity = 0;
    bool stream = false;
    bool emit_tokens = false;
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
AsrRunResult RunAsr(const AsrRunOptions & options);

}  // namespace qasr
