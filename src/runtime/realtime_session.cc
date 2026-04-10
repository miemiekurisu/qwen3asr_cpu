#include "qasr/runtime/realtime_session.h"

#include <algorithm>

namespace qasr {

RealtimeSession::RealtimeSession(const RealtimePolicyConfig & policy,
                                 const StreamPolicyConfig & stream_policy)
    : policy_(policy),
      stream_policy_(stream_policy),
      planner_(stream_policy, policy.sample_rate_hz) {}

Status RealtimeSession::AppendAudio(const float * data, std::size_t n_samples) {
    if (!data || n_samples == 0) {
        return Status(StatusCode::kInvalidArgument, "audio data required");
    }
    audio_buffer_.insert(audio_buffer_.end(), data, data + n_samples);
    total_samples_ += n_samples;

    if (lane_ == RealtimeTextLane::kUnseen) {
        lane_ = RealtimeTextLane::kPartial;
    }
    return OkStatus();
}

Status RealtimeSession::TickDecode(bool force, RealtimeTextUpdate * update) {
    if (!update) {
        return Status(StatusCode::kInvalidArgument, "update must not be null");
    }

    if (!force && !planner_.ShouldDecode(total_samples_)) {
        update->committed = false;
        update->partial_text.clear();
        update->stable_text = text_state_.stable_text;
        update->text.clear();
        return OkStatus();
    }

    // Run partial decode on available audio
    std::string candidate;
    const float * audio_ptr = audio_buffer_.data() + retained_offset_;
    const std::size_t audio_len = audio_buffer_.size() - retained_offset_;

    if (audio_len > 0) {
        Status s = RunPartialDecode(
            text_state_.stable_text,
            audio_ptr, audio_len,
            stream_policy_.max_new_tokens,
            &candidate);
        if (!s.ok()) return s;
    } else {
        candidate = text_state_.stable_text;
    }

    // Advance realtime text state using the service layer
    Status s = AdvanceRealtimeTextState(
        policy_, total_samples_, candidate, force,
        &text_state_, update);
    if (!s.ok()) return s;

    // Track metrics
    if (first_partial_ms_ < 0.0 && !update->partial_text.empty()) {
        first_partial_ms_ = static_cast<double>(total_samples_) /
                            static_cast<double>(policy_.sample_rate_hz) * 1000.0;
    }
    if (first_stable_ms_ < 0.0 && update->committed) {
        first_stable_ms_ = static_cast<double>(total_samples_) /
                           static_cast<double>(policy_.sample_rate_hz) * 1000.0;
    }

    if (update->committed) {
        lane_ = RealtimeTextLane::kStable;
    }

    planner_.MarkDecoded(total_samples_);
    return OkStatus();
}

Status RealtimeSession::BuildPartialDelta(std::string * partial) const {
    if (!partial) {
        return Status(StatusCode::kInvalidArgument, "partial must not be null");
    }
    // Delta is the text beyond the stable prefix
    if (text_state_.last_text.size() > text_state_.stable_text.size()) {
        *partial = text_state_.last_text.substr(text_state_.stable_text.size());
    } else {
        partial->clear();
    }
    return OkStatus();
}

Status RealtimeSession::CommitStableText(std::string_view new_stable) {
    text_state_.stable_text = std::string(new_stable);
    if (!new_stable.empty()) {
        lane_ = RealtimeTextLane::kStable;
    }
    return OkStatus();
}

Status RealtimeSession::FlushTail(RealtimeTextUpdate * update) {
    if (!update) {
        return Status(StatusCode::kInvalidArgument, "update must not be null");
    }

    // Force finalize: commit everything
    Status s = TickDecode(/*force=*/true, update);
    if (!s.ok()) return s;

    lane_ = RealtimeTextLane::kFinal;
    return OkStatus();
}

RealtimeSessionMetrics RealtimeSession::SnapshotMetrics() const {
    RealtimeSessionMetrics m;
    m.first_partial_ms = first_partial_ms_ >= 0.0 ? first_partial_ms_ : 0.0;
    m.first_stable_ms = first_stable_ms_ >= 0.0 ? first_stable_ms_ : 0.0;

    // Compute commit lag: difference between audio position and last decode
    const double audio_ms = static_cast<double>(total_samples_) /
                            static_cast<double>(policy_.sample_rate_hz) * 1000.0;
    const double decode_ms = static_cast<double>(text_state_.last_decode_samples) /
                             static_cast<double>(policy_.sample_rate_hz) * 1000.0;
    m.commit_lag_ms = audio_ms - decode_ms;

    // Realtime factor: decoded audio time / wall time (approximation)
    m.realtime_factor = audio_ms > 0.0 ? audio_ms / (audio_ms + m.commit_lag_ms) : 0.0;

    m.rss_bytes = audio_buffer_.size() * sizeof(float);
    return m;
}

}  // namespace qasr
