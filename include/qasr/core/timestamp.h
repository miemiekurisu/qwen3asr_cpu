#pragma once

#include <cstdint>
#include <string>

#include "qasr/core/status.h"

namespace qasr {

struct TimestampRange {
    std::int64_t begin_ms = 0;
    std::int64_t end_ms = 0;
};

Status ValidateTimestampRange(const TimestampRange & range);
Status SamplesToMilliseconds(std::int64_t sample_count, std::int32_t sample_rate_hz, std::int64_t * out_ms) noexcept;
Status FormatSrtTimestamp(std::int64_t timestamp_ms, std::string * out);
Status FormatJsonTimestamp(std::int64_t timestamp_ms, std::string * out);

}  // namespace qasr
