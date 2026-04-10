#pragma once

#include <string_view>

#include "qasr/core/status.h"
#include "qasr/runtime/task.h"

namespace qasr {

enum class OpenAiEndpoint {
    kChatCompletions = 0,
    kAudioTranscriptions,
    kRealtimeSessions,
};

std::string_view OpenAiEndpointPath(OpenAiEndpoint endpoint) noexcept;
bool IsOpenAiPathSupported(std::string_view path) noexcept;
Status ValidateOpenAiRequest(OpenAiEndpoint endpoint, const DecodeRequestOptions & options, bool want_stream);

}  // namespace qasr
