#include "qasr/protocol/openai.h"

namespace qasr {

std::string_view OpenAiEndpointPath(OpenAiEndpoint endpoint) noexcept {
    switch (endpoint) {
        case OpenAiEndpoint::kChatCompletions:
            return "/v1/chat/completions";
        case OpenAiEndpoint::kAudioTranscriptions:
            return "/v1/audio/transcriptions";
        case OpenAiEndpoint::kRealtimeSessions:
            return "/v1/realtime";
    }
    return "";
}

bool IsOpenAiPathSupported(std::string_view path) noexcept {
    return path == OpenAiEndpointPath(OpenAiEndpoint::kChatCompletions) ||
           path == OpenAiEndpointPath(OpenAiEndpoint::kAudioTranscriptions) ||
           path == OpenAiEndpointPath(OpenAiEndpoint::kRealtimeSessions);
}

Status ValidateOpenAiRequest(OpenAiEndpoint endpoint, const DecodeRequestOptions & options, bool want_stream) {
    Status status = ValidateDecodeRequestOptions(options);
    if (!status.ok()) {
        return status;
    }

    switch (endpoint) {
        case OpenAiEndpoint::kChatCompletions:
            if (options.timestamp_mode != TimestampMode::kNone) {
                return Status(StatusCode::kFailedPrecondition, "chat completions path does not carry timestamp output in the first milestone");
            }
            return OkStatus();
        case OpenAiEndpoint::kAudioTranscriptions:
            if (options.task_mode == TaskMode::kStreaming || want_stream) {
                return Status(StatusCode::kFailedPrecondition, "audio transcription path is offline-only in the first milestone");
            }
            return OkStatus();
        case OpenAiEndpoint::kRealtimeSessions:
            if (options.task_mode != TaskMode::kStreaming || !want_stream) {
                return Status(StatusCode::kFailedPrecondition, "realtime path requires streaming mode");
            }
            if (options.timestamp_mode != TimestampMode::kNone) {
                return Status(StatusCode::kFailedPrecondition, "realtime path does not support timestamps in the first milestone");
            }
            return OkStatus();
    }
    return Status(StatusCode::kInvalidArgument, "unknown OpenAI endpoint");
}

}  // namespace qasr
