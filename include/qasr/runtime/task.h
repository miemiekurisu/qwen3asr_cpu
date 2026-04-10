#pragma once

#include <cstdint>
#include <string>

#include "qasr/core/status.h"

namespace qasr {

enum class TaskMode {
    kOffline = 0,
    kStreaming,
};

enum class TimestampMode {
    kNone = 0,
    kSegment,
    kWord,
};

struct DecodeRequestOptions {
    TaskMode task_mode = TaskMode::kOffline;
    TimestampMode timestamp_mode = TimestampMode::kNone;
    bool want_partial_results = false;
    bool want_async = false;
    std::int32_t max_new_tokens = 256;
    std::int32_t chunk_ms = 2000;
    std::int32_t rollback_tokens = 5;
};

Status ValidateDecodeRequestOptions(const DecodeRequestOptions & options);
bool TimestampModeSupported(TaskMode task_mode, TimestampMode timestamp_mode) noexcept;
std::string MakeDeterministicRequestId(std::uint64_t high_bits, std::uint64_t low_bits);

}  // namespace qasr
