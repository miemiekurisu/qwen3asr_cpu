#pragma once

#include <cstdint>

#include "qasr/core/status.h"

namespace qasr {

struct AudioSpan {
    const float * samples = nullptr;
    std::int64_t sample_count = 0;
    std::int32_t sample_rate_hz = 0;
    std::int32_t channels = 0;
};

Status ValidateAudioSpan(const AudioSpan & audio);
bool IsMono16kAudio(const AudioSpan & audio) noexcept;
std::int64_t AudioDurationMs(const AudioSpan & audio) noexcept;

}  // namespace qasr
