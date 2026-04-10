#include "tests/test_registry.h"

#include "qasr/protocol/openai.h"

QASR_TEST(OpenAiEndpointPathsAreStable) {
    QASR_EXPECT_EQ(qasr::OpenAiEndpointPath(qasr::OpenAiEndpoint::kChatCompletions), std::string_view("/v1/chat/completions"));
    QASR_EXPECT_EQ(qasr::OpenAiEndpointPath(qasr::OpenAiEndpoint::kAudioTranscriptions), std::string_view("/v1/audio/transcriptions"));
    QASR_EXPECT_EQ(qasr::OpenAiEndpointPath(qasr::OpenAiEndpoint::kRealtimeSessions), std::string_view("/v1/realtime"));
    QASR_EXPECT(qasr::IsOpenAiPathSupported("/v1/chat/completions"));
    QASR_EXPECT(!qasr::IsOpenAiPathSupported("/unknown"));
}

QASR_TEST(ValidateOpenAiRequestMatchesMilestoneRules) {
    qasr::DecodeRequestOptions options;
    QASR_EXPECT(qasr::ValidateOpenAiRequest(qasr::OpenAiEndpoint::kChatCompletions, options, false).ok());

    options.timestamp_mode = qasr::TimestampMode::kWord;
    QASR_EXPECT_EQ(qasr::ValidateOpenAiRequest(qasr::OpenAiEndpoint::kChatCompletions, options, false).code(), qasr::StatusCode::kFailedPrecondition);

    options = qasr::DecodeRequestOptions{};
    options.task_mode = qasr::TaskMode::kStreaming;
    QASR_EXPECT_EQ(qasr::ValidateOpenAiRequest(qasr::OpenAiEndpoint::kAudioTranscriptions, options, true).code(), qasr::StatusCode::kFailedPrecondition);

    QASR_EXPECT(qasr::ValidateOpenAiRequest(qasr::OpenAiEndpoint::kRealtimeSessions, options, true).ok());
}
