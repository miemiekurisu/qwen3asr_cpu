#include "tests/test_registry.h"

#include <array>

#include "qasr/core/audio_types.h"

QASR_TEST(ValidateAudioSpanAcceptsValidMono16k) {
    std::array<float, 160> samples{};
    const qasr::AudioSpan audio{samples.data(), 160, 16000, 1};
    QASR_EXPECT(qasr::ValidateAudioSpan(audio).ok());
    QASR_EXPECT(qasr::IsMono16kAudio(audio));
    QASR_EXPECT_EQ(qasr::AudioDurationMs(audio), 10);
}

QASR_TEST(ValidateAudioSpanRejectsInvalidArguments) {
    const qasr::AudioSpan null_audio{nullptr, 1, 16000, 1};
    QASR_EXPECT_EQ(qasr::ValidateAudioSpan(null_audio).code(), qasr::StatusCode::kInvalidArgument);

    const qasr::AudioSpan negative_count{nullptr, -1, 16000, 1};
    QASR_EXPECT_EQ(qasr::ValidateAudioSpan(negative_count).code(), qasr::StatusCode::kInvalidArgument);

    const qasr::AudioSpan bad_rate{nullptr, 0, 0, 1};
    QASR_EXPECT_EQ(qasr::ValidateAudioSpan(bad_rate).code(), qasr::StatusCode::kInvalidArgument);

    const qasr::AudioSpan bad_channels{nullptr, 0, 16000, 0};
    QASR_EXPECT_EQ(qasr::ValidateAudioSpan(bad_channels).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(AudioDurationHandlesExtremeAndRandomLikeValues) {
    std::array<float, 64000> samples{};
    const qasr::AudioSpan stereo{samples.data(), 64000, 32000, 2};
    QASR_EXPECT_EQ(qasr::AudioDurationMs(stereo), 1000);
    QASR_EXPECT(!qasr::IsMono16kAudio(stereo));

    for (std::int64_t frames = 1; frames <= 1000; frames += 137) {
        const qasr::AudioSpan audio{samples.data(), frames, 1000, 1};
        QASR_EXPECT_EQ(qasr::AudioDurationMs(audio), frames);
    }
}
