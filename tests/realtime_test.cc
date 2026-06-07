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

QASR_TEST(AdvanceRealtimeTextStateDivergentClearsStable) {
    const qasr::RealtimePolicyConfig config;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello wor", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "hello world ", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "hola mundo", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("hola mundo"));
    QASR_EXPECT_EQ(update.text, std::string("hola mundo"));
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

QASR_TEST(AdvanceRealtimeDisplayStateBuildsLiveTail) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.partial_text = "hello wor";
    update.text = "hello wor";
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT(snapshot.recent_segments.empty());
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string());
    QASR_EXPECT_EQ(snapshot.live_partial_text, std::string("hello wor"));
    QASR_EXPECT_EQ(snapshot.display_text, std::string("hello wor"));

    update.stable_text = "hello ";
    update.partial_text = "world";
    update.text = "hello world";
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("hello "));
    QASR_EXPECT_EQ(snapshot.live_partial_text, std::string("world"));
    QASR_EXPECT_EQ(snapshot.live_text, std::string("hello world"));
}

QASR_TEST(AdvanceRealtimeDisplayStateFinalizesPunctuatedSegment) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "hello world. ";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.recent_segments.size(), std::size_t(1));
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("hello world."));
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string());
    QASR_EXPECT_EQ(snapshot.display_text, std::string("hello world."));
}

QASR_TEST(AdvanceRealtimeDisplayStateSplitsCommittedSentenceBeforeLiveTail) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "第一句。第二句还在继续";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.recent_segments.size(), std::size_t(1));
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("第一句。"));
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("第二句还在继续"));
    QASR_EXPECT_EQ(snapshot.display_text, std::string("第一句。\n第二句还在继续"));
}

QASR_TEST(AdvanceRealtimeDisplayStateSplitsStableChineseClauseOnComma) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "请马上前往医院，后面立刻安排救护处理";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.recent_segments.size(), std::size_t(1));
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("请马上前往医院，"));
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("后面立刻安排救护处理"));
}

QASR_TEST(AdvanceRealtimeDisplayStateKeepsOnlyRecentSegments) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "one. ";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());

    update.stable_text = "one. two. ";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());

    update.stable_text = "one. two. three. ";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());

    /* kRecentSegmentLimit was raised to 1000 (Web Speech API style: archive
     * retains the full session, not just the last few).  All three
     * segments stay in recent_segments so the user can scroll back. */
    QASR_EXPECT_EQ(snapshot.total_finalized_segments, std::size_t(3));
    QASR_EXPECT_EQ(snapshot.recent_segments.size(), std::size_t(3));
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("one."));
    QASR_EXPECT_EQ(snapshot.recent_segments[1], std::string("two."));
    QASR_EXPECT_EQ(snapshot.recent_segments[2], std::string("three."));
}

QASR_TEST(AdvanceRealtimeDisplayStateRevisionReplacesLiveStable) {
    /* When the C layer revises its committed prefix (e.g. the model
     * initially decoded "试一哈" then re-decoded the prefix and
     * settled on "一次"), the new stable_text from the C layer does
     * NOT extend the previous one.  We must take the new text as the
     * source of truth so the UI shows the latest (correct)
     * interpretation rather than the stale rejected version.  Without
     * this, the user sees a stable line stuck on the model's first
     * guess even after the model has corrected itself. */
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    /* First decode: model said "好的，我再试一哈" */
    update.stable_text = "好的，我再试一哈";
    update.partial_text = "： 是不是有問題？";
    update.text = update.stable_text + update.partial_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("好的，我再试一哈"));
    QASR_EXPECT_EQ(snapshot.live_partial_text, std::string("： 是不是有問題？"));

    /* Second decode: model re-decoded the prefix and revised to
     * "好的，我再是一次".  Note the prefix "好的，我再" is shared but
     * the rest is different — the C layer's LCP-based revision.
     * After the revision, DrainStableSegments will split the new
     * stable on its soft-clause punctuation (the "，"), so the
     * first part goes to recent_segments and only the tail stays
     * in live_stable. */
    update.stable_text = "好的，我再是一次，看看是不是有问题";
    update.partial_text = "";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());

    /* The new stable_text fully replaces the old one.  The most
     * important assertion is that "好的，我再试一哈" is GONE — the
     * model revised it to "好的，我再是一次" and the new text took
     * effect.  After draining, "好的，我再是一次，" is committed to
     * recent_segments and the live tail is "看看是不是有问题". */
    QASR_EXPECT(snapshot.live_stable_text.find("试一哈") == std::string::npos);
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("看看是不是有问题"));
    QASR_EXPECT(snapshot.live_partial_text.empty());
    /* The revised first clause "好的，我再是一次，" is in the
     * archive (it contains "一次", confirming the new text took
     * effect).  The old "试一哈" should be nowhere. */
    QASR_EXPECT_EQ(snapshot.recent_segments.size(), std::size_t(1));
    QASR_EXPECT(snapshot.recent_segments[0].find("一次") != std::string::npos);
    QASR_EXPECT(snapshot.recent_segments[0].find("试一哈") == std::string::npos);
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("好的，我再是一次，"));
}

QASR_TEST(AdvanceRealtimeDisplayStateRevisionEmptyIsIgnored) {
    /* Edge case: if the C layer sends an empty stable_text as a
     * "revision" (e.g. recovery reset cleared state), we must NOT
     * wipe live_stable.  Treat empty as "no change" so the user's
     * accumulated text isn't lost during a recovery / reset. */
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "已经积累了一些文本";
    update.partial_text = "";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("已经积累了一些文本"));

    /* C layer sends empty stable (e.g. recovery reset).  Don't wipe. */
    update.stable_text = "";
    update.partial_text = "新 partial";
    update.text = update.partial_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, false, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.live_stable_text, std::string("已经积累了一些文本"));
    QASR_EXPECT_EQ(snapshot.live_partial_text, std::string("新 partial"));
}

QASR_TEST(AdvanceRealtimeDisplayStateForceFinalizeFlushesTail) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    update.stable_text = "hello world again";
    update.text = update.stable_text;
    QASR_EXPECT(qasr::AdvanceRealtimeDisplayState(update, true, &state, &snapshot).ok());
    QASR_EXPECT_EQ(snapshot.total_finalized_segments, std::size_t(1));
    QASR_EXPECT_EQ(snapshot.recent_segments[0], std::string("hello world again"));
    QASR_EXPECT(snapshot.live_text.empty());
}

QASR_TEST(AdvanceRealtimeDisplayStateRejectsNullOutputs) {
    qasr::RealtimeDisplayState state;
    qasr::RealtimeDisplaySnapshot snapshot;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT_EQ(
        qasr::AdvanceRealtimeDisplayState(update, false, nullptr, &snapshot).code(),
        qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(
        qasr::AdvanceRealtimeDisplayState(update, false, &state, nullptr).code(),
        qasr::StatusCode::kInvalidArgument);
}

// ===========================================================================
// CommitStrategy::kEager: backward-compatibility lock
// ===========================================================================
//
// The default strategy is kEager, which preserves the historical behavior:
// partial_text = latest - stable.  This test guards against accidental
// default-flip that would silently change every existing caller's display.

QASR_TEST(RealtimePolicyConfigDefaultCommitStrategyIsEager) {
    qasr::RealtimePolicyConfig config;
    QASR_EXPECT_EQ(
        static_cast<int>(config.commit_strategy),
        static_cast<int>(qasr::CommitStrategy::kEager));
}

QASR_TEST(AdvanceRealtimeTextStateEagerPartialIsLatestMinusStable) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kEager;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你好", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.partial_text, std::string("你好"));
    QASR_EXPECT_EQ(update.tentative_text, std::string());

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好的", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好"));
    QASR_EXPECT_EQ(update.partial_text, std::string("的"));
    QASR_EXPECT_EQ(update.tentative_text, std::string());

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "你好", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好"));
    QASR_EXPECT_EQ(update.partial_text, std::string(""));
    QASR_EXPECT_EQ(update.tentative_text, std::string());
}

// ===========================================================================
// CommitStrategy::kConsensus2x: 2-decode agreement gating
// ===========================================================================
//
// On each chunk, only the portion that AGREES between last_text and
// latest_text (and is past stable_text) goes into partial_text.  The
// unagreed tail goes into tentative_text (which a renderer may show
// dimmed or hide).  This eliminates the "前段不稳定" flicker where a
// model backtracks in chunk 3 and the user's "的" disappears.

QASR_TEST(AdvanceRealtimeTextStateConsensus2xFixesEarlyFlicker) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你好", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("你好"));
    QASR_EXPECT_EQ(update.tentative_text, std::string());
    const std::string step1_display = update.stable_text + update.partial_text;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好的", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好"));
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.tentative_text, std::string("的"));
    const std::string step2_display = update.stable_text + update.partial_text;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "你好", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好"));
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.tentative_text, std::string());
    const std::string step3_display = update.stable_text + update.partial_text;

    QASR_EXPECT_EQ(step2_display, std::string("你好"));
    QASR_EXPECT_EQ(step3_display, std::string("你好"));
    QASR_EXPECT(step2_display.size() >= step1_display.size());
    QASR_EXPECT(step3_display.size() >= step2_display.size());
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xGrowsPartialOnAgreement) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.partial_text, std::string("hello"));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "hello world", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("hello"));
    QASR_EXPECT_EQ(update.tentative_text, std::string(" world"));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "hello world again", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));
    QASR_EXPECT_EQ(update.partial_text, std::string("world"));
    QASR_EXPECT_EQ(update.tentative_text, std::string(" again"));
    QASR_EXPECT(update.committed);
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xHidesPartialOnLcpShrink) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello world", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "hello world", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));
    QASR_EXPECT_EQ(update.partial_text, std::string("world"));
    QASR_EXPECT_EQ(update.tentative_text, std::string());

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "hello", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.tentative_text, std::string("hello"));
    const std::string display = update.stable_text + update.partial_text;
    QASR_EXPECT_EQ(display, std::string("hello "));
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xPopulatesTentativeOnLcp) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好世界", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.tentative_text, std::string("好世界"));
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xForceFreezeStillFires) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    config.max_unstable_ms = 1000;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你好世", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好世", false, &state, &update).ok());
    QASR_EXPECT(!update.stable_text.empty());
    QASR_EXPECT_EQ(update.tentative_text, std::string());
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xForceFinalizeClearsTentative) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "你", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好世界", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.tentative_text, std::string("好世界"));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "你好世界", true, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("你好世界"));
    QASR_EXPECT_EQ(update.partial_text, std::string());
    QASR_EXPECT_EQ(update.tentative_text, std::string());
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xDivergentClearsStable) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 16000U, "hello wor", false, &state, &update).ok());
    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 32000U, "hello world ", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string("hello "));

    QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, 48000U, "hola mundo", false, &state, &update).ok());
    QASR_EXPECT_EQ(update.stable_text, std::string());
    QASR_EXPECT_EQ(update.partial_text, std::string("hola mundo"));
    QASR_EXPECT_EQ(update.tentative_text, std::string());
}

QASR_TEST(AdvanceRealtimeTextStateConsensus2xNoDisplayShrink) {
    qasr::RealtimePolicyConfig config;
    config.commit_strategy = qasr::CommitStrategy::kConsensus2x;
    qasr::RealtimeTextState state;
    qasr::RealtimeTextUpdate update;

    std::string prev_display;
    auto step = [&](std::size_t samples, const char * txt) {
        QASR_EXPECT(qasr::AdvanceRealtimeTextState(config, samples, txt, false, &state, &update).ok());
        const std::string cur_display = update.stable_text + update.partial_text;
        QASR_EXPECT(cur_display.size() >= prev_display.size());
        prev_display = cur_display;
    };

    step(16000U, "你好");
    step(32000U, "你好的");
    step(48000U, "你好");
    step(64000U, "你好世");
    step(80000U, "你好世界");
}
