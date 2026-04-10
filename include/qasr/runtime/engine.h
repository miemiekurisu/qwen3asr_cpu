#pragma once

#include "qasr/runtime/blas.h"
#include "qasr/runtime/config.h"
#include "qasr/runtime/task.h"

namespace qasr {

struct BootstrapPlan {
    BlasBackend blas_backend = BlasBackend::kUnknown;
    bool start_async_executor = false;
    bool start_stream_worker = false;
    bool start_openai_chat_surface = false;
    bool start_openai_audio_surface = false;
    bool start_vllm_surface = false;
    bool start_forced_aligner = false;
};

Status ValidateBootstrapInputs(const EngineConfig & config, const DecodeRequestOptions & options);
BootstrapPlan BuildBootstrapPlan(const EngineConfig & config, const DecodeRequestOptions & options) noexcept;

}  // namespace qasr
