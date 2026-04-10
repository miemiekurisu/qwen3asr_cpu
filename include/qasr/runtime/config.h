#pragma once

#include <cstdint>
#include <string>

#include "qasr/core/status.h"

namespace qasr {

struct EngineConfig {
    std::int32_t intra_threads = 0;
    std::int32_t inter_threads = 0;
    std::int32_t max_sessions = 256;
    std::int32_t max_queue_size = 1024;
    bool enable_streaming = true;
    bool enable_timestamps = true;
    bool enable_openai_compat = true;
    bool enable_vllm_compat = true;
    std::string instance_name = "qasr";
};

Status ValidateEngineConfig(const EngineConfig & config);
bool HasAnyProtocolSurface(const EngineConfig & config) noexcept;

}  // namespace qasr
