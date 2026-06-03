#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Display-commit policy for the realtime streaming text pipeline.
///
/// The streaming text machine produces a sequence of partial decodes.
/// When the model backtracks (e.g. chunk N=1 says "你好", chunk N=2
/// says "你好的", chunk N=3 reverts to "你好") the naive
/// "partial = latest - stable" policy shrinks the live display,
/// which the user sees as a flicker.  This enum selects how
/// aggressively the renderer should commit characters to the
/// "agreed" region shown to the user.
///
/// Industrial context (searched Jun 2026):
///   - WhisperLive (open-source): eager display, no consensus gate.
///   - faster-whisper: end-of-segment only, no incremental UI.
///   - NVIDIA Riva: server-side "Word Boosting + Token Voting"
///     explicitly waits for 2-decode agreement on each token before
///     emitting it as committed text.  Same shape as `kConsensus2x`.
///   - Azure Speech: emits `Recognizing` (tentative) and `Recognized`
///     (committed) events separately; the UI is expected to render
///     tentative dimmed and never let it overwrite a committed span.
///
/// Pre/Post: callers must not assume `latest` text will be shown
/// verbatim.  The chosen strategy decides what subset of `latest` is
/// routed into `RealtimeTextUpdate::partial_text` vs
/// `tentative_text`.  Existing text in `stable_text` is never
/// shrunk in `kConsensus2x` mode unless the latest decode diverges
/// from it entirely (no common prefix).
enum class CommitStrategy {
    kEager,        // partial = latest - stable, tentative = ""  (legacy default)
    kConsensus2x,  // partial = LCP(last, latest) - stable, tentative = latest - LCP
};

struct RealtimePolicyConfig {
    int sample_rate_hz = 16000;
    int min_decode_interval_ms = 800;
    int max_unstable_ms = 6000;
    int max_decode_window_ms = 32000;
    CommitStrategy commit_strategy = CommitStrategy::kEager;
};

struct RealtimeTextState {
    std::string stable_text;
    std::string last_text;
    std::size_t last_decode_samples = 0;
    std::size_t unstable_since_samples = 0;
};

struct RealtimeTextUpdate {
    bool committed = false;
    std::string stable_text;
    std::string partial_text;
    std::string tentative_text;
    std::string text;
    // `text` = stable + partial (the canonical "display" string).
    // `tentative_text` is only populated when
    // `RealtimePolicyConfig::commit_strategy == kConsensus2x`.  A
    // renderer may show it dimmed or hide it; it MUST NOT be
    // combined with `partial_text` when computing visible text.
};

struct RealtimeDisplayState {
    std::string last_stable_text;
    std::vector<std::string> recent_segments;
    std::string live_stable_text;
    std::string live_partial_text;
    std::size_t total_finalized_segments = 0;
};

struct RealtimeDisplaySnapshot {
    std::vector<std::string> recent_segments;
    std::string live_stable_text;
    std::string live_partial_text;
    std::string live_text;
    std::string display_text;
    std::size_t total_finalized_segments = 0;
};

Status ValidateRealtimePolicyConfig(const RealtimePolicyConfig & config);
std::size_t RealtimeMaxDecodeSamples(const RealtimePolicyConfig & config);
std::size_t TrimRealtimeSamples(std::vector<float> * samples, std::size_t max_samples);
bool RealtimeShouldDecode(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::size_t last_decode_samples,
    bool force);
Status AdvanceRealtimeTextState(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::string_view latest_text,
    bool force_finalize,
    RealtimeTextState * state,
    RealtimeTextUpdate * update);
Status AdvanceRealtimeDisplayState(
    const RealtimeTextUpdate & text_update,
    bool force_finalize,
    RealtimeDisplayState * state,
    RealtimeDisplaySnapshot * snapshot);

}  // namespace qasr
