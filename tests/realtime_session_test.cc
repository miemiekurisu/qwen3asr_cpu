#include "tests/test_registry.h"
#include "qasr/runtime/realtime_session.h"

#include <vector>

namespace {

qasr::RealtimePolicyConfig DefaultPolicy() {
    qasr::RealtimePolicyConfig p;
    p.sample_rate_hz = 16000;
    p.min_decode_interval_ms = 800;
    p.max_unstable_ms = 12000;
    p.max_decode_window_ms = 32000;
    return p;
}

qasr::StreamPolicyConfig DefaultStreamPolicy() {
    qasr::StreamPolicyConfig s;
    s.chunk_sec = 2.0f;
    s.window_sec = 8.0f;
    s.history_sec = 32.0f;
    s.rollback_tokens = 5;
    s.unfixed_chunks = 2;
    s.prefix_cap_tokens = 150;
    s.max_new_tokens = 32;
    s.force_freeze_age_ms = 12000;
    return s;
}

}  // namespace

// --- Normal ---

QASR_TEST(RealtimeSessionInitialState) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    QASR_EXPECT(session.stable_text().empty());
    QASR_EXPECT_EQ(session.total_samples(), std::size_t(0));
    QASR_EXPECT(session.current_lane() == qasr::RealtimeTextLane::kUnseen);
}

QASR_TEST(RealtimeSessionAppendAudio) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(8000, 0.1f);  // 0.5 seconds at 16kHz
    qasr::Status s = session.AppendAudio(audio.data(), audio.size());
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(session.total_samples(), std::size_t(8000));
    QASR_EXPECT(session.current_lane() == qasr::RealtimeTextLane::kPartial);
}

QASR_TEST(RealtimeSessionAppendMultiple) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(4000, 0.1f);
    session.AppendAudio(audio.data(), audio.size());
    session.AppendAudio(audio.data(), audio.size());
    QASR_EXPECT_EQ(session.total_samples(), std::size_t(8000));
}

// --- Error: invalid inputs ---

QASR_TEST(RealtimeSessionAppendNull) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.AppendAudio(nullptr, 100);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(RealtimeSessionAppendZero) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(100, 0.0f);
    qasr::Status s = session.AppendAudio(audio.data(), 0);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(RealtimeSessionTickDecodeNull) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.TickDecode(false, nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(RealtimeSessionFlushTailNull) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.FlushTail(nullptr);
    QASR_EXPECT(!s.ok());
}

// --- TickDecode (no force, not enough audio) ---

QASR_TEST(RealtimeSessionTickDecodeNoForce) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(8000, 0.1f);  // Only 0.5s, need 2s
    session.AppendAudio(audio.data(), audio.size());

    qasr::RealtimeTextUpdate update;
    qasr::Status s = session.TickDecode(false, &update);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(!update.committed);
}

// --- CommitStableText ---

QASR_TEST(RealtimeSessionCommitStable) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.CommitStableText("hello world");
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(session.stable_text(), std::string("hello world"));
    QASR_EXPECT(session.current_lane() == qasr::RealtimeTextLane::kStable);
}

QASR_TEST(RealtimeSessionCommitEmpty) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.CommitStableText("");
    QASR_EXPECT(s.ok());
    QASR_EXPECT(session.current_lane() == qasr::RealtimeTextLane::kUnseen);
}

// --- BuildPartialDelta ---

QASR_TEST(RealtimeSessionBuildPartialDeltaNull) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::Status s = session.BuildPartialDelta(nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(RealtimeSessionBuildPartialDeltaEmpty) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::string partial;
    qasr::Status s = session.BuildPartialDelta(&partial);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(partial.empty());
}

// --- SnapshotMetrics ---

QASR_TEST(RealtimeSessionMetricsInitial) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    qasr::RealtimeSessionMetrics m = session.SnapshotMetrics();
    QASR_EXPECT(m.first_partial_ms == 0.0);
    QASR_EXPECT(m.first_stable_ms == 0.0);
    QASR_EXPECT(m.rss_bytes == 0);
}

QASR_TEST(RealtimeSessionMetricsAfterAppend) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(16000, 0.1f);
    session.AppendAudio(audio.data(), audio.size());
    qasr::RealtimeSessionMetrics m = session.SnapshotMetrics();
    QASR_EXPECT(m.rss_bytes > 0);
}

// --- FlushTail ---

QASR_TEST(RealtimeSessionFlushTailSetsLane) {
    qasr::RealtimeSession session(DefaultPolicy(), DefaultStreamPolicy());
    std::vector<float> audio(32000, 0.1f);  // 2 seconds
    session.AppendAudio(audio.data(), audio.size());

    qasr::RealtimeTextUpdate update;
    qasr::Status s = session.FlushTail(&update);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(session.current_lane() == qasr::RealtimeTextLane::kFinal);
}
