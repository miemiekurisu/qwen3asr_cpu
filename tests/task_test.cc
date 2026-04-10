#include "tests/test_registry.h"

#include <string>

#include "qasr/runtime/task.h"

QASR_TEST(TimestampModeSupportMatchesCurrentMilestone) {
    QASR_EXPECT(qasr::TimestampModeSupported(qasr::TaskMode::kOffline, qasr::TimestampMode::kNone));
    QASR_EXPECT(qasr::TimestampModeSupported(qasr::TaskMode::kOffline, qasr::TimestampMode::kSegment));
    QASR_EXPECT(qasr::TimestampModeSupported(qasr::TaskMode::kOffline, qasr::TimestampMode::kWord));
    QASR_EXPECT(!qasr::TimestampModeSupported(qasr::TaskMode::kStreaming, qasr::TimestampMode::kSegment));
}

QASR_TEST(ValidateDecodeRequestOptionsRejectsErrors) {
    qasr::DecodeRequestOptions options;
    options.max_new_tokens = 0;
    QASR_EXPECT_EQ(qasr::ValidateDecodeRequestOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options = qasr::DecodeRequestOptions{};
    options.chunk_ms = 0;
    QASR_EXPECT_EQ(qasr::ValidateDecodeRequestOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options = qasr::DecodeRequestOptions{};
    options.rollback_tokens = -1;
    QASR_EXPECT_EQ(qasr::ValidateDecodeRequestOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options = qasr::DecodeRequestOptions{};
    options.task_mode = qasr::TaskMode::kOffline;
    options.want_partial_results = true;
    QASR_EXPECT_EQ(qasr::ValidateDecodeRequestOptions(options).code(), qasr::StatusCode::kFailedPrecondition);

    options = qasr::DecodeRequestOptions{};
    options.task_mode = qasr::TaskMode::kStreaming;
    options.timestamp_mode = qasr::TimestampMode::kWord;
    QASR_EXPECT_EQ(qasr::ValidateDecodeRequestOptions(options).code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(MakeDeterministicRequestIdIsStable) {
    const std::string request_id = qasr::MakeDeterministicRequestId(0x1234ULL, 0xabcdefULL);
    QASR_EXPECT_EQ(request_id, std::string("req-00000000000012340000000000abcdef"));
}
