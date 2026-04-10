#pragma once

#include <string_view>

#include "qasr/core/status.h"
#include "qasr/runtime/task.h"

namespace qasr {

std::string_view VllmChatCompletionsPath() noexcept;
Status ValidateVllmRequest(const DecodeRequestOptions & options, bool want_stream, bool is_batch_request);

}  // namespace qasr
