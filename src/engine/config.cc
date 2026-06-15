#include "qasr/engine/config.h"
#include <cstdlib>

namespace qasr {

BackendKind ParseBackendKind(std::string_view s) {
    if (s == "cuda" || s == "CUDA") return BackendKind::kCuda;
    if (s == "mlx" || s == "MLX") return BackendKind::kMlx;
    return BackendKind::kCpu;
}

PlatformProfile AutoDetectPlatform() {
#if defined(__linux__) && defined(__aarch64__)
    if (std::getenv("QASR_PLATFORM_DGX_SPARK")) {
        return PlatformProfile::kDgxSparkCuda13Sm121;
    }
    return PlatformProfile::kLinuxAarch64Cpu;
#elif defined(__linux__) && defined(__x86_64__)
    return PlatformProfile::kLinuxX86_64Cpu;
#elif defined(_WIN32) && defined(__x86_64__)
    if (std::getenv("QASR_PLATFORM_WINDOWS_CUDA")) {
        return PlatformProfile::kWindowsCuda12Sm89;
    }
    return PlatformProfile::kGenericCpu;
#elif defined(__APPLE__) && defined(__aarch64__)
    return PlatformProfile::kMacosArm64Mlx;
#else
    return PlatformProfile::kGenericCpu;
#endif
}

}  // namespace qasr
