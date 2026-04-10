#include "tests/test_registry.h"

#include <vector>

#include "qasr/service/realtime.h"

QASR_TEST(ValidateRealtimePolicyConfigRejectsBadValues) {
    qasr::RealtimePolicyConfig config;
    config.sample_rate_hz = 0;
    QASR_EXPECT_EQ(
        qasr::ValidateRealtimePolicyConfig(config).code(),
        qasr::StatusCode::kInvalidArgument);

    config = qasr::RealtimePolicyConfig{};
    config.max_decode_window_ms = 0;
    QASR_EXPECT_EQ(
        qasr::ValidateRealtimePolicyConfig(config).code(),
        qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(RealtimeDecodeWindowComputesAndTrimsRetainedSamples) {
    const qasr::RealtimePolicyConfig config;
    QASR_EXPECT_EQ(qasr::RealtimeMaxDecodeSamples(config), 512000U);

    std::vector<float> samples = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    QASR_EXPECT_EQ(qasr::TrimRealtimeSamples(&samples, 3U), 2U);
    QASR_EXPECT_EQ(samples.size(), 3U);
    QASR_EXPECT_EQ(samples[0], 2.0F);
    QASR_EXPECT_EQ(samples[2], 4.0F);

    QASR_EXPECT_EQ(qasr::TrimRealtimeSamples(&samples, 0U), 3U);
    QASR_EXPECT(samples.empty());
}

QASR_TEST(RealtimeShouldDecodeUsesCadenceThreshold) {
    const qasr::RealtimePolicyConfig config;
    QASR_EXPECT(!qasr::RealtimeShouldDecode(config, 1000U, 0U, false));
    QASR_EXPECT(qasr::RealtimeShouldDecode(config, 12800U, 0U, false));
    QASR_EXPECT(qasr::RealtimeShouldDecode(config, 1025U, 1024U, true));
}

QASR_TEST(AdvanceRealtimeTextStateCommitsStableEnglishPrefix) {
    const qasr::RealtimePolicyConfig config;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello wor", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("hello wor"));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "hello world ", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));
    QASR_EXPECT_EQ(update.partial_text, std::string("world "));
    QASR_EXPECT_EQ(update.text, std::string("hello world "));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "hello world again", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello world "));
    QASR_EXPECT_EQ(update.partial_text, std::string("again"));
}

QASR_TEST(AdvanceRealtimeTextStateCommitsUtf8Prefix) {
    const qasr::RealtimePolicyConfig config;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你好世", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("你好世"));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好世界", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好世"));
    QASR_EXPECT_EQ(update.partial_text, std::string("界"));
}

QASR_TEST(AdvanceRealtimeTextStateForceFinalizeFlushesTail) {
    const qasr::RealtimePolicyConfig config;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello world", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello world", true, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello world"));
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.text, std::string("hello world"));
}

QASR_TEST(AdvanceRealtimeTextStateForceFreezesAgedTail) {
    qasr::RealtimePolicyConfig config;
    config.max_unstable_ms = 1000;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "supercalifragilistic", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "supercalifragilistic", false, &state, &update).ok());
    QASR_EXPECT(!update.stable_text.empty());
    QASR_EXPECT(update.text.size() >= update.stable_text.size());
}

QASR_TEST(AdvanceRealtimeTextStateRejectsNullOutputs) {
    const qasr::RealtimePolicyConfig config;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;
    QASR_EXPECT_EQ(
        qasr::AdvanceRealtimeTextState(config, 0U, "", false, nullptr, &update).code(),
        qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(
        qasr::AdvanceRealtimeTextState(config, 0U, "", false, &state, nullptr).code(),
        qasr::StatusCode::kInvalidArgument);
}
