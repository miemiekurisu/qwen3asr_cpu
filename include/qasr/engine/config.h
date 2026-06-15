#pragma once

#include "qasr/engine/types.h"
#include <string>

namespace qasr {

struct V2EngineConfig {
    std::string model_dir;

    BackendKind backend = BackendKind::kCpu;
    PlatformProfile platform = PlatformProfile::kGenericCpu;

    int device_id = 0;
    int threads = 0;

    bool allow_backend_fallback = false;

    enum class Residency {
        kCpuOnly,
        kDecoderOnly,
        kFullModel
    };

    Residency residency = Residency::kCpuOnly;

    enum class Precision {
        kFp32,
        kBf16,
        kFp16
    };

    Precision precision = Precision::kFp32;

    int max_sessions = 1;
    int max_realtime_sessions = 1;
    int max_batch_jobs = 1;

    int max_active_gpu_jobs = 1;
    int cuda_stream_pool_size = 1;
    bool enable_decode_microbatch = false;

    int verbosity = 0;
    int32_t stream_max_new_tokens = 32;
    float temperature = -1.0f;
    std::string language;
    std::string prompt;
};

BackendKind ParseBackendKind(std::string_view s);
PlatformProfile AutoDetectPlatform();

}  // namespace qasr
