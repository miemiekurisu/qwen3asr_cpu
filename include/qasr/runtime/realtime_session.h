#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qasr/core/state_machine.h"
#include "qasr/core/status.h"
#include "qasr/inference/streaming_policy.h"
#include "qasr/service/realtime.h"

namespace qasr {

struct RealtimeSessionMetrics {
    double first_partial_ms = 0.0;
    double first_stable_ms = 0.0;
    double commit_lag_ms = 0.0;
    double unstable_tail_ms = 0.0;
    double realtime_factor = 0.0;
    std::size_t rss_bytes = 0;
};

/// Per-session streaming state for realtime ASR.
/// Pre: must be initialized with a valid policy config.
/// Post: manages audio buffer, decode cadence, and text commitment.
/// Thread-safe: NOT thread-safe; one owner per session.
class RealtimeSession {
public:
    explicit RealtimeSession(const RealtimePolicyConfig & policy,
                             const StreamPolicyConfig & stream_policy);

    Status AppendAudio(const float * data, std::size_t n_samples);
    Status TickDecode(bool force, RealtimeTextUpdate * update);
    Status BuildPartialDelta(std::string * partial) const;
    Status CommitStableText(std::string_view new_stable);
    Status FlushTail(RealtimeTextUpdate * update);
    RealtimeSessionMetrics SnapshotMetrics() const;

    const std::string & stable_text() const noexcept { return text_state_.stable_text; }
    std::size_t total_samples() const noexcept { return total_samples_; }
    RealtimeTextLane current_lane() const noexcept { return lane_; }

private:
    RealtimePolicyConfig policy_;
    StreamPolicyConfig stream_policy_;
    StreamChunkPlanner planner_;
    RealtimeTextState text_state_;
    RealtimeTextLane lane_ = RealtimeTextLane::kUnseen;
    std::vector<float> audio_buffer_;
    std::size_t total_samples_ = 0;
    std::size_t retained_offset_ = 0;
    double first_partial_ms_ = -1.0;
    double first_stable_ms_ = -1.0;
};

}  // namespace qasr
