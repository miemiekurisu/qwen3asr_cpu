#include "qasr/service/realtime.h"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <limits>

namespace qasr {
namespace {

constexpr std::size_t kNoPendingUnstable = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kForcedTailGuardCodepoints = 4;
constexpr std::size_t kForcedTailMinCodepoints = 8;
constexpr std::size_t kRecentSegmentLimit = 2;
constexpr std::size_t kSoftClauseMinCodepoints = 8;
constexpr std::size_t kSoftClauseTailCodepoints = 6;
constexpr std::size_t kSoftSegmentCodepoints = 32;
constexpr std::size_t kHardSegmentCodepoints = 64;

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

bool EndsWith(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

bool MatchesAnyTokenAt(
    std::string_view text,
    std::size_t offset,
    std::initializer_list<std::string_view> tokens,
    std::size_t * matched_size) {
    for (std::string_view token : tokens) {
        if (offset + token.size() <= text.size() &&
            text.substr(offset, token.size()) == token) {
            if (matched_size != nullptr) {
                *matched_size = token.size();
            }
            return true;
        }
    }
    return false;
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

std::string_view TrimAsciiWhitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const unsigned char byte = static_cast<unsigned char>(text[begin]);
        if (!std::isspace(byte)) {
            break;
        }
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin) {
        const unsigned char byte = static_cast<unsigned char>(text[end - 1U]);
        if (!std::isspace(byte)) {
            break;
        }
        --end;
    }
    return text.substr(begin, end - begin);
}

bool EndsWithAsciiWhitespace(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    return std::isspace(static_cast<unsigned char>(text.back())) != 0;
}

bool EndsWithTerminalPunctuation(std::string_view text) {
    const std::string_view trimmed = TrimAsciiWhitespace(text);
    if (trimmed.empty()) {
        return false;
    }
    return EndsWith(trimmed, ".") ||
        EndsWith(trimmed, "!") ||
        EndsWith(trimmed, "?") ||
        EndsWith(trimmed, "。") ||
        EndsWith(trimmed, "！") ||
        EndsWith(trimmed, "？") ||
        EndsWith(trimmed, "\n");
}

bool MatchesTerminalPunctuationAt(
    std::string_view text,
    std::size_t offset,
    std::size_t * matched_size) {
    return MatchesAnyTokenAt(
        text,
        offset,
        {".", "!", "?", "。", "！", "？", "\n"},
        matched_size);
}

bool MatchesSoftClausePunctuationAt(
    std::string_view text,
    std::size_t offset,
    std::size_t * matched_size) {
    return MatchesAnyTokenAt(
        text,
        offset,
        {",", ";", ":", "，", "、", "；", "："},
        matched_size);
}

std::size_t ConsumeTrailingAsciiWhitespace(std::string_view text, std::size_t offset) {
    while (offset < text.size()) {
        const unsigned char byte = static_cast<unsigned char>(text[offset]);
        if (!std::isspace(byte)) {
            break;
        }
        ++offset;
    }
    return offset;
}

bool ShouldFinalizeStableSegment(std::string_view text, bool force_finalize) {
    const std::string_view trimmed = TrimAsciiWhitespace(text);
    if (trimmed.empty()) {
        return false;
    }
    if (force_finalize || EndsWithTerminalPunctuation(trimmed)) {
        return true;
    }
    const std::size_t codepoints = CountUtf8Codepoints(trimmed);
    if (codepoints >= kHardSegmentCodepoints) {
        return true;
    }
    return codepoints >= kSoftSegmentCodepoints && EndsWithAsciiWhitespace(text);
}

void PushRecentSegment(std::string segment, RealtimeDisplayState * state);

std::size_t FindStableSegmentBoundary(std::string_view text, bool force_finalize) {
    if (text.empty()) {
        return 0U;
    }
    if (force_finalize) {
        return text.size();
    }

    for (std::size_t offset = 0; offset < text.size(); ++offset) {
        if (IsUtf8Continuation(static_cast<unsigned char>(text[offset]))) {
            continue;
        }

        std::size_t matched_size = 0;
        if (MatchesTerminalPunctuationAt(text, offset, &matched_size)) {
            return ConsumeTrailingAsciiWhitespace(text, offset + matched_size);
        }

        if (MatchesSoftClausePunctuationAt(text, offset, &matched_size)) {
            const std::size_t boundary = ConsumeTrailingAsciiWhitespace(text, offset + matched_size);
            const std::size_t prefix_codepoints = CountUtf8Codepoints(text.substr(0, boundary));
            const std::size_t tail_codepoints =
                CountUtf8Codepoints(TrimAsciiWhitespace(text.substr(boundary)));
            if (prefix_codepoints >= kSoftClauseMinCodepoints &&
                tail_codepoints >= kSoftClauseTailCodepoints) {
                return boundary;
            }
        }
    }

    if (ShouldFinalizeStableSegment(text, false)) {
        return text.size();
    }
    return 0U;
}

void DrainStableSegments(bool force_finalize, RealtimeDisplayState * state) {
    if (state == nullptr) {
        return;
    }

    while (!state->live_stable_text.empty()) {
        const std::size_t boundary = FindStableSegmentBoundary(state->live_stable_text, force_finalize);
        if (boundary == 0U) {
            break;
        }
        PushRecentSegment(state->live_stable_text.substr(0, boundary), state);
        state->live_stable_text.erase(0, boundary);
        force_finalize = false;
    }
}

void PushRecentSegment(std::string segment, RealtimeDisplayState * state) {
    if (state == nullptr) {
        return;
    }
    const std::string_view trimmed = TrimAsciiWhitespace(segment);
    if (trimmed.empty()) {
        return;
    }
    state->recent_segments.push_back(std::string(trimmed));
    while (state->recent_segments.size() > kRecentSegmentLimit) {
        state->recent_segments.erase(state->recent_segments.begin());
    }
    ++state->total_finalized_segments;
}

std::string BuildDisplayText(const RealtimeDisplayState & state) {
    std::string text;
    for (const std::string & segment : state.recent_segments) {
        if (!text.empty()) {
            text.push_back('\n');
        }
        text += segment;
    }

    const std::string live_text = state.live_stable_text + state.live_partial_text;
    if (!live_text.empty()) {
        if (!text.empty()) {
            text.push_back('\n');
        }
        text += live_text;
    }
    return text;
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

Status AdvanceRealtimeDisplayState(
    const RealtimeTextUpdate & text_update,
    bool force_finalize,
    RealtimeDisplayState * state,
    RealtimeDisplaySnapshot * snapshot) {
    if (state == nullptr || snapshot == nullptr) {
        return Status(StatusCode::kInvalidArgument, "state and snapshot must not be null");
    }

    if (StartsWith(text_update.stable_text, state->last_stable_text)) {
        state->live_stable_text += std::string(text_update.stable_text.substr(state->last_stable_text.size()));
        state->last_stable_text = text_update.stable_text;
    } else if (state->last_stable_text.empty()) {
        state->last_stable_text = text_update.stable_text;
        state->live_stable_text = text_update.stable_text;
    }

    state->live_partial_text = text_update.partial_text;
    if (force_finalize && !state->live_partial_text.empty()) {
        state->live_stable_text += state->live_partial_text;
        state->live_partial_text.clear();
    }

    DrainStableSegments(force_finalize, state);

    snapshot->recent_segments = state->recent_segments;
    snapshot->live_stable_text = state->live_stable_text;
    snapshot->live_partial_text = state->live_partial_text;
    snapshot->live_text = state->live_stable_text + state->live_partial_text;
    snapshot->display_text = BuildDisplayText(*state);
    snapshot->total_finalized_segments = state->total_finalized_segments;
    return OkStatus();
}

}  // namespace qasr
