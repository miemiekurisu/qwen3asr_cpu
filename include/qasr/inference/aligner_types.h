#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/core/timestamp.h"
#include "qasr/runtime/model_bridge.h"

namespace qasr {

/// A single word/character-level alignment item from ForcedAligner.
struct AlignedWord {
    std::string text;
    double start_sec = 0.0;   // start time in seconds
    double end_sec = 0.0;     // end time in seconds
};

/// Result of forced alignment for one audio chunk.
struct AlignResult {
    std::vector<AlignedWord> words;
};

/// Maximum audio duration (seconds) that ForcedAligner supports per chunk.
/// classify_num=5000 × timestamp_segment_time=80ms = 400s theoretical;
/// official docs state 300s practical limit.
inline constexpr double kAlignerMaxChunkSec = 300.0;

/// Languages supported by Qwen3-ForcedAligner-0.6B.
/// If source language is not in this list, fall back to segment-level timestamps.
inline constexpr const char * kAlignerSupportedLanguages[] = {
    "chinese", "cantonese", "english", "german", "spanish",
    "french", "italian", "portuguese", "russian", "korean", "japanese",
};
inline constexpr int kAlignerSupportedLanguageCount = 11;

/// Check if a language (lowercase) is supported by the ForcedAligner.
/// Thread-safe: yes.
bool IsAlignerLanguageSupported(const std::string & language) noexcept;

/// Validate an AlignResult: words non-empty, times non-negative, monotonic.
/// Thread-safe: yes.
Status ValidateAlignResult(const AlignResult & result);

/// Convert AlignedWord spans into TimedSegment list for subtitle pipeline.
/// Groups adjacent words into segments by punctuation/pause/max-width rules.
/// Pre: words sorted by start_sec ascending.
/// Post: returns segments suitable for LayoutSubtitles().
/// Thread-safe: yes.
std::vector<TimedSegment> WordsToSegments(
    const std::vector<AlignedWord> & words,
    std::int32_t max_segment_chars = 42,
    double max_gap_sec = 1.0);

}  // namespace qasr
