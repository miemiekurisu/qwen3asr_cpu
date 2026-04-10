#pragma once

#include <string>
#include <string_view>

namespace qasr {

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kOutOfRange,
    kFailedPrecondition,
    kNotFound,
    kInternal,
    kUnimplemented,
};

class Status {
public:
    Status() noexcept;
    Status(StatusCode code, std::string message);

    bool ok() const noexcept;
    StatusCode code() const noexcept;
    const std::string & message() const noexcept;
    std::string ToString() const;

private:
    StatusCode code_;
    std::string message_;
};

Status OkStatus() noexcept;
std::string_view StatusCodeName(StatusCode code) noexcept;

}  // namespace qasr
