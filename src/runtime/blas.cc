#include "qasr/runtime/blas.h"

#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

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

Status CheckBlasAvailable() {
#if defined(_WIN32) && defined(QASR_BLAS_OPENBLAS)
    HMODULE h = LoadLibraryA("libopenblas.dll");
    if (!h) {
        return Status(StatusCode::kFailedPrecondition,
            "libopenblas.dll not found. "
            "Either add the OpenBLAS bin/ directory to your PATH, "
            "or copy libopenblas.dll next to this executable.");
    }
    // Verify the DLL exports the functions we actually need.
    // A mismatched DLL version would pass the LoadLibrary check but crash
    // at the first delay-loaded call.
    if (!GetProcAddress(h, "cblas_sgemm")) {
        FreeLibrary(h);
        return Status(StatusCode::kFailedPrecondition,
            "libopenblas.dll was found but does not export cblas_sgemm. "
            "The DLL may be the wrong version or architecture. "
            "Please use the OpenBLAS 0.3.x release matching this build.");
    }
    FreeLibrary(h);
#endif
    return OkStatus();
}

}  // namespace qasr
