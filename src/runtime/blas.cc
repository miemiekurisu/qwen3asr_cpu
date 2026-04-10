#include "qasr/runtime/blas.h"

#include <string>

namespace qasr {

BlasBackend CompiledBlasBackend() noexcept {
#if defined(QASR_BLAS_ACCELERATE)
    return BlasBackend::kAccelerate;
#elif defined(QASR_BLAS_OPENBLAS)
    return BlasBackend::kOpenBlas;
#else
    return BlasBackend::kUnknown;
#endif
}

std::string_view BlasBackendName(BlasBackend backend) noexcept {
    switch (backend) {
        case BlasBackend::kUnknown:
            return "unknown";
        case BlasBackend::kAccelerate:
            return "accelerate";
        case BlasBackend::kOpenBlas:
            return "openblas";
    }
    return "unknown";
}

Status ValidateBlasPolicy(std::string_view target_os, BlasBackend backend) {
    if (target_os.empty()) {
        return Status(StatusCode::kInvalidArgument, "target_os must not be empty");
    }
    const std::string os(target_os);
    if (os == "macos") {
        if (backend != BlasBackend::kAccelerate) {
            return Status(StatusCode::kFailedPrecondition, "macos must use Accelerate");
        }
        return OkStatus();
    }
    if (os == "linux" || os == "windows") {
        if (backend != BlasBackend::kOpenBlas) {
            return Status(StatusCode::kFailedPrecondition, "linux/windows must use OpenBLAS");
        }
        return OkStatus();
    }
    return Status(StatusCode::kInvalidArgument, "target_os must be one of macos/linux/windows");
}

}  // namespace qasr
