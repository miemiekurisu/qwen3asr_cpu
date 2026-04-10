#include "qasr/runtime/config.h"

namespace qasr {

Status ValidateEngineConfig(const EngineConfig & config) {
    if (config.intra_threads < 0) {
        return Status(StatusCode::kInvalidArgument, "intra_threads must be >= 0");
    }
    if (config.inter_threads < 0) {
        return Status(StatusCode::kInvalidArgument, "inter_threads must be >= 0");
    }
    if (config.max_sessions <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_sessions must be > 0");
    }
    if (config.max_queue_size <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_queue_size must be > 0");
    }
    if (config.instance_name.empty()) {
        return Status(StatusCode::kInvalidArgument, "instance_name must not be empty");
    }
    if (!HasAnyProtocolSurface(config)) {
        return Status(StatusCode::kFailedPrecondition, "at least one protocol surface must be enabled");
    }
    return OkStatus();
}

bool HasAnyProtocolSurface(const EngineConfig & config) noexcept {
    return config.enable_openai_compat || config.enable_vllm_compat;
}

}  // namespace qasr
