#include "qasr/inference/streaming_policy.h"

#include <algorithm>
#include <cstring>

namespace qasr {

Status ValidateStreamPolicyConfig(const StreamPolicyConfig & config) {
    if (config.chunk_sec <= 0.0f) {
        return Status(StatusCode::kInvalidArgument, "chunk_sec must be positive");
    }
    if (config.window_sec <= 0.0f || config.window_sec < config.chunk_sec) {
        return Status(StatusCode::kInvalidArgument, "window_sec must be >= chunk_sec");
    }
    if (config.history_sec <= 0.0f) {
        return Status(StatusCode::kInvalidArgument, "history_sec must be positive");
    }
    if (config.rollback_tokens < 0) {
        return Status(StatusCode::kInvalidArgument, "rollback_tokens must be non-negative");
    }
    if (config.unfixed_chunks < 0) {
        return Status(StatusCode::kInvalidArgument, "unfixed_chunks must be non-negative");
    }
    if (config.prefix_cap_tokens <= 0) {
        return Status(StatusCode::kInvalidArgument, "prefix_cap_tokens must be positive");
    }
    if (config.max_new_tokens <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_new_tokens must be positive");
    }
    if (config.force_freeze_age_ms <= 0) {
        return Status(StatusCode::kInvalidArgument, "force_freeze_age_ms must be positive");
    }
    return OkStatus();
}

// --- StreamChunkPlanner ---

StreamChunkPlanner::StreamChunkPlanner(const StreamPolicyConfig & config,
                                       std::int32_t sample_rate_hz)
    : chunk_samples_(static_cast<std::int32_t>(config.chunk_sec * static_cast<float>(sample_rate_hz))),
      window_samples_(static_cast<std::int32_t>(config.window_sec * static_cast<float>(sample_rate_hz))) {}

bool StreamChunkPlanner::ShouldDecode(std::size_t total_samples) const noexcept {
    if (chunk_samples_ <= 0) return false;
    return total_samples >= last_decode_samples_ + static_cast<std::size_t>(chunk_samples_);
}

void StreamChunkPlanner::MarkDecoded(std::size_t at_samples) noexcept {
    last_decode_samples_ = at_samples;
}

// --- EncoderCache ---

void EncoderCache::Store(std::int32_t window_index, std::vector<float> data,
                          std::int32_t seq_len) {
    // Replace existing entry if present
    for (auto & e : entries_) {
        if (e.window_index == window_index) {
            e.data = std::move(data);
            e.seq_len = seq_len;
            return;
        }
    }
    entries_.push_back({window_index, std::move(data), seq_len});
}

bool EncoderCache::Has(std::int32_t window_index) const {
    for (const auto & e : entries_) {
        if (e.window_index == window_index) return true;
    }
    return false;
}

void EncoderCache::Evict(std::int32_t older_than) {
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [older_than](const Entry & e) {
                           return e.window_index < older_than;
                       }),
        entries_.end());
}

// --- Streaming policy functions ---

Status RunPartialDecode(std::string_view context_text,
                        const float * audio, std::size_t n_samples,
                        std::int32_t max_new_tokens,
                        std::string * candidate_text) {
    if (!candidate_text) {
        return Status(StatusCode::kInvalidArgument, "candidate_text must not be null");
    }
    if (!audio || n_samples == 0) {
        return Status(StatusCode::kInvalidArgument, "audio data required");
    }
    if (max_new_tokens <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_new_tokens must be positive");
    }

    // Placeholder: real implementation runs encoder + decoder with context
    // For now, return the context as-is (no new text generated)
    *candidate_text = std::string(context_text);
    return OkStatus();
}

std::size_t LongestCommonStablePrefix(std::string_view a, std::string_view b) noexcept {
    const std::size_t min_len = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < min_len && a[i] == b[i]) {
        ++i;
    }
    // Walk back to UTF-8 character boundary
    // UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
    while (i > 0 && i < a.size() && (static_cast<unsigned char>(a[i]) & 0xC0) == 0x80) {
        --i;
    }
    return i;
}

Status CommitFrontier(std::string_view candidate_text,
                      std::string * stable_prefix,
                      std::string * unstable_suffix,
                      std::int32_t rollback_guard_tokens) {
    if (!stable_prefix || !unstable_suffix) {
        return Status(StatusCode::kInvalidArgument, "output pointers must not be null");
    }

    const std::size_t common = LongestCommonStablePrefix(*stable_prefix, candidate_text);

    // The stable part is confirmed by both old and new
    // Rollback guard: keep some tokens as unstable
    std::size_t commit_end = common;
    if (rollback_guard_tokens > 0) {
        // Walk back rollback_guard_tokens worth of characters
        std::int32_t tokens_back = 0;
        while (commit_end > 0 && tokens_back < rollback_guard_tokens) {
            --commit_end;
            // Count token boundaries (simplified: space-separated)
            if (commit_end > 0 && candidate_text[commit_end] == ' ') {
                ++tokens_back;
            }
        }
    }

    // Update stable and unstable
    if (commit_end > stable_prefix->size()) {
        // Extend stable prefix
        *stable_prefix = std::string(candidate_text.substr(0, commit_end));
    }
    *unstable_suffix = std::string(candidate_text.substr(commit_end));
    return OkStatus();
}

bool DetectDegenerateTail(std::string_view text, std::int32_t min_repeat_chars) noexcept {
    if (text.size() < static_cast<std::size_t>(min_repeat_chars * 2)) {
        return false;
    }

    // Check if the last min_repeat_chars repeat
    const auto tail_len = static_cast<std::size_t>(min_repeat_chars);
    const std::string_view tail = text.substr(text.size() - tail_len);

    // Check if this tail pattern appears just before itself
    if (text.size() >= tail_len * 2) {
        const std::string_view before_tail = text.substr(text.size() - tail_len * 2, tail_len);
        if (tail == before_tail) {
            return true;
        }
    }
    return false;
}

Status ForceFreezeAgedSuffix(std::string * stable_prefix,
                             std::string * unstable_suffix,
                             std::string * frozen_text) {
    if (!stable_prefix || !unstable_suffix || !frozen_text) {
        return Status(StatusCode::kInvalidArgument, "output pointers must not be null");
    }
    if (unstable_suffix->empty()) {
        frozen_text->clear();
        return OkStatus();
    }

    // Freeze the entire unstable suffix into stable
    *frozen_text = *unstable_suffix;
    stable_prefix->append(*unstable_suffix);
    unstable_suffix->clear();
    return OkStatus();
}

Status ReanchorContext(std::string_view stable_prefix, std::string * context_text) {
    if (!context_text) {
        return Status(StatusCode::kInvalidArgument, "context_text must not be null");
    }
    *context_text = std::string(stable_prefix);
    return OkStatus();
}

Status EvictOldHistory(EncoderCache * cache, std::size_t total_samples,
                       std::int32_t history_samples) {
    if (!cache) {
        return Status(StatusCode::kInvalidArgument, "cache must not be null");
    }
    if (history_samples <= 0) {
        return Status(StatusCode::kInvalidArgument, "history_samples must be positive");
    }

    if (total_samples > static_cast<std::size_t>(history_samples)) {
        // Evict windows older than the retention boundary
        // Window index calculation depends on chunk size, but for now
        // use a simple threshold based on total samples
        const auto boundary = static_cast<std::int32_t>(
            (total_samples - static_cast<std::size_t>(history_samples)) / 32000);
        cache->Evict(boundary);
    }
    return OkStatus();
}

}  // namespace qasr
