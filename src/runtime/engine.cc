#include "qasr/runtime/engine.h"

namespace qasr {

Status ValidateBootstrapInputs(const EngineConfig & config, const DecodeRequestOptions & options) {
    Status status = ValidateEngineConfig(config);
    if (!status.ok()) {
        return status;
    }
    status = ValidateDecodeRequestOptions(options);
    if (!status.ok()) {
        return status;
    }
    if (options.task_mode == TaskMode::kStreaming && !config.enable_streaming) {
        return Status(StatusCode::kFailedPrecondition, "streaming request received while streaming is disabled");
    }
    if (options.timestamp_mode != TimestampMode::kNone && !config.enable_timestamps) {
        return Status(StatusCode::kFailedPrecondition, "timestamp request received while timestamps are disabled");
    }
    return OkStatus();
}

BootstrapPlan BuildBootstrapPlan(const EngineConfig & config, const DecodeRequestOptions & options) noexcept {
    BootstrapPlan plan;
    plan.blas_backend = CompiledBlasBackend();
    plan.start_async_executor = config.inter_threads != 0 || options.want_async;
    plan.start_stream_worker = config.enable_streaming && options.task_mode == TaskMode::kStreaming;
    plan.start_openai_chat_surface = config.enable_openai_compat;
    plan.start_openai_audio_surface = config.enable_openai_compat;
    plan.start_vllm_surface = config.enable_vllm_compat;
    plan.start_forced_aligner = config.enable_timestamps && options.timestamp_mode == TimestampMode::kWord;
    return plan;
}

}  // namespace qasr
