#pragma once

#include <string_view>

#include "qasr/core/status.h"

namespace qasr {

enum class BlasBackend {
    kUnknown = 0,
    kAccelerate,
    kOpenBlas,
};

BlasBackend CompiledBlasBackend() noexcept;
std::string_view BlasBackendName(BlasBackend backend) noexcept;
Status ValidateBlasPolicy(std::string_view target_os, BlasBackend backend);

/// Check that the BLAS shared library is loadable at runtime.
/// On Windows+OpenBLAS this probes for libopenblas.dll; returns an error
/// with a human-readable hint if the DLL cannot be found.
Status CheckBlasAvailable();

}  // namespace qasr
