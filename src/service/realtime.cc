#include "qasr/service/realtime.h"

#include <algorithm>
#include <cctype>
#include <limits>

namespace qasr {
namespace {

constexpr std::size_t kNoPendingUnstable = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kForcedTailGuardCodepoints = 4;
constexpr std::size_t kForcedTailMinCodepoints = 8;

bool IsUtf8Continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

std::size_t SnapUtf8Boundary(std::string_view text, std::size_t size) {
    if (size >= text.size()) {
        return text.size();
    }
    while (size > 0U && IsUtf8Continuation(static_cast<unsigned char>(text[size]))) {
        --size;
    }
    return size;
}

std::string LongestCommonUtf8Prefix(std::string_view lhs, std::string_view rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    std::size_t matched = 0;
    while (matched < limit && lhs[matched] == rhs[matched]) {
        ++matched;
    }
    matched = SnapUtf8Boundary(lhs, matched);
    return std::string(lhs.substr(0, matched));
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

std::string TrimAsciiWordTail(std::string_view text) {
    if (text.empty()) {
        return {};
    }

    std::size_t end = text.size();
    while (end > 0U) {
        const unsigned char byte = static_cast<unsigned char>(text[end - 1U]);
        if (byte >= 0x80U || !std::isalnum(byte)) {
            break;
        }
        --end;
    }

    if (end == text.size()) {
        return std::string(text);
    }
    if (end == 0U) {
        return {};
    }
    return std::string(text.substr(0, end));
}

std::size_t CountUtf8Codepoints(std::string_view text) {
    std::size_t count = 0;
    for (const char ch : text) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (!IsUtf8Continuation(byte)) {
            ++count;
        }
    }
    return count;
}

std::string DropLastUtf8Codepoints(std::string_view text, std::size_t count) {
    if (text.empty() || count == 0U) {
        return std::string(text);
    }

    std::size_t end = text.size();
    std::size_t dropped = 0;
    while (end > 0U && dropped < count) {
        --end;
        while (end > 0U && IsUtf8Continuation(static_cast<unsigned char>(text[end]))) {
            --end;
        }
        ++dropped;
    }
    return std::string(text.substr(0, end));
}

std::size_t MillisecondsToSamples(int sample_rate_hz, int milliseconds) {
    return static_cast<std::size_t>((static_cast<long long>(sample_rate_hz) * milliseconds) / 1000LL);
}

std::string ForceFreezePrefix(std::string_view text) {
    std::string committed = TrimAsciiWordTail(text);
    if (!committed.empty()) {
        return committed;
    }

    const std::size_t codepoints = CountUtf8Codepoints(text);
    if (codepoints <= kForcedTailMinCodepoints) {
        return {};
    }
    return DropLastUtf8Codepoints(text, kForcedTailGuardCodepoints);
}

RealtimeTextUpdate BuildRealtimeTextUpdate(
    const RealtimeTextState & state,
    std::string_view latest_text,
    bool committed) {
    RealtimeTextUpdate update;
    update.committed = committed;
    update.stable_text = state.stable_text;
    if (StartsWith(latest_text, state.stable_text)) {
        update.partial_text = std::string(latest_text.substr(state.stable_text.size()));
        update.text = state.stable_text + update.partial_text;
    } else {
        update.stable_text.clear();
        update.partial_text = std::string(latest_text);
        update.text = std::string(latest_text);
    }
    return update;
}

}  // namespace

Status ValidateRealtimePolicyConfig(const RealtimePolicyConfig & config) {
    if (config.sample_rate_hz <= 0) {
        return Status(StatusCode::kInvalidArgument, "sample_rate_hz must be > 0");
    }
    if (config.min_decode_interval_ms <= 0) {
        return Status(StatusCode::kInvalidArgument, "min_decode_interval_ms must be > 0");
    }
    if (config.max_unstable_ms <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_unstable_ms must be > 0");
    }
    if (config.max_decode_window_ms <= 0) {
        return Status(StatusCode::kInvalidArgument, "max_decode_window_ms must be > 0");
    }
    return OkStatus();
}

std::size_t RealtimeMaxDecodeSamples(const RealtimePolicyConfig & config) {
    if (!ValidateRealtimePolicyConfig(config).ok()) {
        return 0U;
    }
    return MillisecondsToSamples(config.sample_rate_hz, config.max_decode_window_ms);
}

std::size_t TrimRealtimeSamples(std::vector<float> * samples, std::size_t max_samples) {
    if (samples == nullptr || samples->size() <= max_samples) {
        return 0U;
    }

    const std::size_t dropped = samples->size() - max_samples;
    if (max_samples == 0U) {
        samples->clear();
        return dropped;
    }
    std::move(samples->begin() + static_cast<std::ptrdiff_t>(dropped), samples->end(), samples->begin());
    samples->resize(max_samples);
    return dropped;
}

bool RealtimeShouldDecode(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::size_t last_decode_samples,
    bool force) {
    if (force) {
        return total_samples > last_decode_samples;
    }
    if (!ValidateRealtimePolicyConfig(config).ok()) {
        return false;
    }
    const std::size_t min_decode_samples =
        MillisecondsToSamples(config.sample_rate_hz, config.min_decode_interval_ms);
    return total_samples >= last_decode_samples + min_decode_samples;
}

Status AdvanceRealtimeTextState(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::string_view latest_text,
    bool force_finalize,
    RealtimeTextState * state,
    RealtimeTextUpdate * update) {
    if (state == nullptr || update == nullptr) {
        return Status(StatusCode::kInvalidArgument, "state and update must not be null");
    }

    Status status = ValidateRealtimePolicyConfig(config);
    if (!status.ok()) {
        return status;
    }

    if (force_finalize) {
        const bool committed = state->stable_text != latest_text;
        state->stable_text = std::string(latest_text);
        state->last_text = state->stable_text;
        state->last_decode_samples = total_samples;
        state->unstable_since_samples = kNoPendingUnstable;
        *update = BuildRealtimeTextUpdate(*state, state->stable_text, committed);
        update->partial_text.clear();
        update->text = update->stable_text;
        return OkStatus();
    }

    bool committed = false;
    if (state->last_text.empty()) {
        state->last_text = std::string(latest_text);
        state->last_decode_samples = total_samples;
        state->unstable_since_samples = latest_text.empty() ? kNoPendingUnstable : total_samples;
        *update = BuildRealtimeTextUpdate(*state, latest_text, false);
        return OkStatus();
    }

    const std::string common_prefix = LongestCommonUtf8Prefix(state->last_text, latest_text);
    if (StartsWith(common_prefix, state->stable_text)) {
        std::string candidate_commit = std::string(common_prefix.substr(state->stable_text.size()));
        candidate_commit = TrimAsciiWordTail(candidate_commit);
        if (!candidate_commit.empty()) {
            state->stable_text += candidate_commit;
            committed = true;
        }
    }

    if (StartsWith(latest_text, state->stable_text) && state->stable_text.size() < latest_text.size()) {
        if (state->unstable_since_samples == kNoPendingUnstable) {
            state->unstable_since_samples = state->last_decode_samples;
        }
        const std::size_t max_unstable_samples =
            MillisecondsToSamples(config.sample_rate_hz, config.max_unstable_ms);
        if (total_samples >= state->unstable_since_samples + max_unstable_samples) {
            std::string force_piece =
                ForceFreezePrefix(std::string_view(latest_text).substr(state->stable_text.size()));
            if (!force_piece.empty()) {
                state->stable_text += force_piece;
                committed = true;
            }
        }
    }

    state->last_text = std::string(latest_text);
    state->last_decode_samples = total_samples;
    if (StartsWith(latest_text, state->stable_text) && state->stable_text.size() < latest_text.size()) {
        if (committed || state->unstable_since_samples == kNoPendingUnstable) {
            state->unstable_since_samples = total_samples;
        }
    } else {
        state->unstable_since_samples = kNoPendingUnstable;
    }

    *update = BuildRealtimeTextUpdate(*state, latest_text, committed);
    return OkStatus();
}

}  // namespace qasr
