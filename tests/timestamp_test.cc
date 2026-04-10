#include "tests/test_registry.h"

#include <string>

#include "qasr/core/timestamp.h"

QASR_TEST(ValidateTimestampRangeChecksBounds) {
    QASR_EXPECT(qasr::ValidateTimestampRange(qasr::TimestampRange{0, 0}).ok());
    QASR_EXPECT(qasr::ValidateTimestampRange(qasr::TimestampRange{1, 2}).ok());
    QASR_EXPECT_EQ(qasr::ValidateTimestampRange(qasr::TimestampRange{-1, 2}).code(), qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(qasr::ValidateTimestampRange(qasr::TimestampRange{10, 9}).code(), qasr::StatusCode::kOutOfRange);
}

QASR_TEST(SamplesToMillisecondsHandlesErrorsAndRandomValues) {
    std::int64_t out_ms = -1;
    QASR_EXPECT_EQ(qasr::SamplesToMilliseconds(-1, 16000, &out_ms).code(), qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(qasr::SamplesToMilliseconds(1, 0, &out_ms).code(), qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(qasr::SamplesToMilliseconds(1, 16000, nullptr).code(), qasr::StatusCode::kInvalidArgument);

    QASR_EXPECT(qasr::SamplesToMilliseconds(16000, 16000, &out_ms).ok());
    QASR_EXPECT_EQ(out_ms, 1000);

    for (std::int64_t sample_count = 0; sample_count <= 123456; sample_count += 7919) {
        QASR_EXPECT(qasr::SamplesToMilliseconds(sample_count, 16000, &out_ms).ok());
        QASR_EXPECT_EQ(out_ms, (sample_count * 1000) / 16000);
    }
}

QASR_TEST(FormatTimestampVariantsWork) {
    std::string text;
    QASR_EXPECT_EQ(qasr::FormatSrtTimestamp(-1, &text).code(), qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(qasr::FormatSrtTimestamp(3723004, &text).code(), qasr::StatusCode::kOk);
    QASR_EXPECT_EQ(text, std::string("01:02:03,004"));
    QASR_EXPECT_EQ(qasr::FormatJsonTimestamp(3723004, &text).code(), qasr::StatusCode::kOk);
    QASR_EXPECT_EQ(text, std::string("01:02:03.004"));
}
