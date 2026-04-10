#include "qasr/protocol/vllm.h"

namespace qasr {

std::string_view VllmChatCompletionsPath() noexcept {
    return "/v1/chat/completions";
}

Status ValidateVllmRequest(const DecodeRequestOptions & options, bool want_stream, bool is_batch_request) {
    Status status = ValidateDecodeRequestOptions(options);
    if (!status.ok()) {
        return status;
    }

    if (want_stream) {
        if (options.task_mode != TaskMode::kStreaming) {
            return Status(StatusCode::kFailedPrecondition, "vLLM stream requires streaming task mode");
        }
        if (is_batch_request) {
            return Status(StatusCode::kFailedPrecondition, "vLLM stream does not support batch inference");
        }
        if (options.timestamp_mode != TimestampMode::kNone) {
            return Status(StatusCode::kFailedPrecondition, "vLLM stream does not support timestamps");
        }
    }

    return OkStatus();
}

}  // namespace qasr
