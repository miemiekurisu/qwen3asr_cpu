#include "tests/test_registry.h"

#include "qasr/protocol/vllm.h"

QASR_TEST(VllmPathIsStable) {
    QASR_EXPECT_EQ(qasr::VllmChatCompletionsPath(), std::string_view("/v1/chat/completions"));
}

QASR_TEST(ValidateVllmRequestMatchesOfficialStreamingConstraint) {
    qasr::DecodeRequestOptions options;
    options.task_mode = qasr::TaskMode::kStreaming;
    QASR_EXPECT(qasr::ValidateVllmRequest(options, true, false).ok());

    options = qasr::DecodeRequestOptions{};
    options.task_mode = qasr::TaskMode::kStreaming;
    options.timestamp_mode = qasr::TimestampMode::kWord;
    QASR_EXPECT_EQ(qasr::ValidateVllmRequest(options, true, false).code(), qasr::StatusCode::kFailedPrecondition);

    options = qasr::DecodeRequestOptions{};
    options.task_mode = qasr::TaskMode::kStreaming;
    QASR_EXPECT_EQ(qasr::ValidateVllmRequest(options, true, true).code(), qasr::StatusCode::kFailedPrecondition);
}
