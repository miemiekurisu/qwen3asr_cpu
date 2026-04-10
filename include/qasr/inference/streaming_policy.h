#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

struct StreamPolicyConfig {
    float chunk_sec = 2.0f;
    float window_sec = 8.0f;
    float history_sec = 32.0f;
    std::int32_t rollback_tokens = 5;
    std::int32_t unfixed_chunks = 2;
    std::int32_t prefix_cap_tokens = 150;
    std::int32_t max_new_tokens = 32;
    std::int32_t force_freeze_age_ms = 12000;
};

/// Pre: config fields > 0.
/// Post: returns Ok if all values are within valid range.
/// Thread-safe: yes.
Status ValidateStreamPolicyConfig(const StreamPolicyConfig & config);

/// Chunk timing planner for streaming inference.
/// Pre: sample_rate > 0.
/// Post: tracks accumulated audio and determines when to decode.
/// Thread-safe: NOT thread-safe; owned per session.
class StreamChunkPlanner {
public:
    explicit StreamChunkPlanner(const StreamPolicyConfig & config,
                                std::int32_t sample_rate_hz = 16000);

    bool ShouldDecode(std::size_t total_samples) const noexcept;
    std::int32_t chunk_samples() const noexcept { return chunk_samples_; }
    std::int32_t window_samples() const noexcept { return window_samples_; }

    void MarkDecoded(std::size_t at_samples) noexcept;
    std::size_t last_decode_samples() const noexcept { return last_decode_samples_; }

private:
    std::int32_t chunk_samples_;
    std::int32_t window_samples_;
    std::size_t last_decode_samples_ = 0;
};

/// Cache of already-encoded audio windows.
/// Pre: none. Post: stores completed encoder output per window.
/// Thread-safe: NOT thread-safe; owned per session.
class EncoderCache {
public:
    void Store(std::int32_t window_index, std::vector<float> data, std::int32_t seq_len);
    bool Has(std::int32_t window_index) const;
    void Evict(std::int32_t older_than);
    std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry {
        std::int32_t window_index;
        std::vector<float> data;
        std::int32_t seq_len;
    };
    std::vector<Entry> entries_;
};

/// Run a single-round partial decode, producing candidate text.
/// Pre: audio is available, weights loaded.
/// Post: candidate_text is the raw model output for this round.
/// Thread-safe: no.
Status RunPartialDecode(std::string_view context_text,
                        const float * audio, std::size_t n_samples,
                        std::int32_t max_new_tokens,
                        std::string * candidate_text);

/// Compute the longest common stable prefix between two strings (UTF-8 safe).
/// Pre: valid UTF-8.
/// Post: returns byte offset of the longest common prefix.
/// Thread-safe: yes.
std::size_t LongestCommonStablePrefix(std::string_view a, std::string_view b) noexcept;

/// Commit the stable frontier: advance the stable prefix.
/// Pre: stable_prefix is current committed text.
/// Post: stable_prefix extended if new candidate confirms prior text.
/// Thread-safe: no.
Status CommitFrontier(std::string_view candidate_text,
                      std::string * stable_prefix,
                      std::string * unstable_suffix,
                      std::int32_t rollback_guard_tokens);

/// Detect degenerate (repeating) tail in candidate text.
/// Pre: text.size() > 0.
/// Post: returns true if the tail is repetitive.
/// Thread-safe: yes.
bool DetectDegenerateTail(std::string_view text, std::int32_t min_repeat_chars) noexcept;

/// Force-freeze the unsettled suffix after age exceeds threshold.
/// Pre: unstable_suffix not empty, age exceeds limit.
/// Post: moves portion of unstable into stable, returns frozen text.
/// Thread-safe: no.
Status ForceFreezeAgedSuffix(std::string * stable_prefix,
                             std::string * unstable_suffix,
                             std::string * frozen_text);

/// Re-anchor context after degenerate detection or error recovery.
/// Pre: session is in degraded state.
/// Post: resets decoder context to stable prefix only.
/// Thread-safe: no.
Status ReanchorContext(std::string_view stable_prefix,
                       std::string * context_text);

/// Evict old encoder/decoder history beyond the retention window.
/// Pre: history_sec > 0.
/// Post: trims entries older than the window.
/// Thread-safe: no.
Status EvictOldHistory(EncoderCache * cache, std::size_t total_samples,
                       std::int32_t history_samples);

}  // namespace qasr
