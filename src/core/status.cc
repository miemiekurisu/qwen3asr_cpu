#include "qasr/core/status.h"

#include <utility>

namespace qasr {

Status::Status() noexcept : code_(StatusCode::kOk) {}

Status::Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

bool Status::ok() const noexcept {
    return code_ == StatusCode::kOk;
}

StatusCode Status::code() const noexcept {
    return code_;
}

const std::string & Status::message() const noexcept {
    return message_;
}

std::string Status::ToString() const {
    if (ok()) {
        return "OK";
    }
    if (message_.empty()) {
        return std::string(StatusCodeName(code_));
    }
    return std::string(StatusCodeName(code_)) + ": " + message_;
}

Status OkStatus() noexcept {
    return Status();
}

std::string_view StatusCodeName(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::kOk:
            return "OK";
        case StatusCode::kInvalidArgument:
            return "INVALID_ARGUMENT";
        case StatusCode::kOutOfRange:
            return "OUT_OF_RANGE";
        case StatusCode::kFailedPrecondition:
            return "FAILED_PRECONDITION";
        case StatusCode::kNotFound:
            return "NOT_FOUND";
        case StatusCode::kInternal:
            return "INTERNAL";
        case StatusCode::kUnimplemented:
            return "UNIMPLEMENTED";
    }
    return "UNKNOWN";
}

}  // namespace qasr
