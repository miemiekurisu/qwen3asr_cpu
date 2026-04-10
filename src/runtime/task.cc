#include "qasr/runtime/task.h"

#include <iomanip>
#include <sstream>

namespace qasr {

Status ValidateDecodeRequestOptions(const DecodeRequestOptions & options) {
    if (options.max_new_tokens <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_new_tokens must be > 0");
    }
    if (options.chunk_ms <= 0) {
        return Status(StatusCode::kInvalidArgument, "chunk_ms must be > 0");
    }
    if (options.rollback_tokens < 0) {
        return Status(StatusCode::kInvalidArgument, "rollback_tokens must be >= 0");
    }
    if (!TimestampModeSupported(options.task_mode, options.timestamp_mode)) {
        return Status(StatusCode::kFailedPrecondition, "timestamp mode is not supported for the task mode");
    }
    if (options.task_mode == TaskMode::kOffline && options.want_partial_results) {
        return Status(StatusCode::kFailedPrecondition, "offline mode must not request partial results");
    }
    return OkStatus();
}

bool TimestampModeSupported(TaskMode task_mode, TimestampMode timestamp_mode) noexcept {
    if (timestamp_mode == TimestampMode::kNone) {
        return true;
    }
    if (task_mode == TaskMode::kStreaming) {
        return false;
    }
    return true;
}

std::string MakeDeterministicRequestId(std::uint64_t high_bits, std::uint64_t low_bits) {
    std::ostringstream builder;
    builder << "req-"
            << std::hex << std::setfill('0')
            << std::setw(16) << high_bits
            << std::setw(16) << low_bits;
    return builder.str();
}

}  // namespace qasr
