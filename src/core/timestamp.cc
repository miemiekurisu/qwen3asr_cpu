#include "qasr/core/timestamp.h"

#include <cstdio>

namespace qasr {

Status ValidateTimestampRange(const TimestampRange & range) {
    if (range.begin_ms < 0 || range.end_ms < 0) {
        return Status(StatusCode::kInvalidArgument, "timestamps must be >= 0");
    }
    if (range.end_ms < range.begin_ms) {
        return Status(StatusCode::kOutOfRange, "end_ms must be >= begin_ms");
    }
    return OkStatus();
}

Status SamplesToMilliseconds(std::int64_t sample_count, std::int32_t sample_rate_hz, std::int64_t * out_ms) noexcept {
    if (out_ms == nullptr) {
        return Status(StatusCode::kInvalidArgument, "out_ms must not be null");
    }
    if (sample_count < 0) {
        return Status(StatusCode::kInvalidArgument, "sample_count must be >= 0");
    }
    if (sample_rate_hz <= 0) {
        return Status(StatusCode::kInvalidArgument, "sample_rate_hz must be > 0");
    }
    *out_ms = (sample_count * 1000) / sample_rate_hz;
    return OkStatus();
}

namespace {

Status FormatTimestamp(std::int64_t timestamp_ms, char decimal_separator, std::string * out) {
    if (out == nullptr) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    if (timestamp_ms < 0) {
        return Status(StatusCode::kInvalidArgument, "timestamp_ms must be >= 0");
    }
    const std::int64_t hours = timestamp_ms / 3600000;
    const std::int64_t minutes = (timestamp_ms / 60000) % 60;
    const std::int64_t seconds = (timestamp_ms / 1000) % 60;
    const std::int64_t millis = timestamp_ms % 1000;

    char buffer[32];
    const int written = std::snprintf(
        buffer,
        sizeof(buffer),
        "%02lld:%02lld:%02lld%c%03lld",
        static_cast<long long>(hours),
        static_cast<long long>(minutes),
        static_cast<long long>(seconds),
        decimal_separator,
        static_cast<long long>(millis));
    if (written <= 0) {
        return Status(StatusCode::kInternal, "snprintf failed");
    }
    *out = buffer;
    return OkStatus();
}

}  // namespace

Status FormatSrtTimestamp(std::int64_t timestamp_ms, std::string * out) {
    return FormatTimestamp(timestamp_ms, ',', out);
}

Status FormatJsonTimestamp(std::int64_t timestamp_ms, std::string * out) {
    return FormatTimestamp(timestamp_ms, '.', out);
}

}  // namespace qasr
