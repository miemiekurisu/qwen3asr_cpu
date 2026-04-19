#include "qasr/inference/aligner_types.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

namespace qasr {
namespace {

bool IsCjkCodepoint(std::uint32_t cp) noexcept {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x3000 && cp <= 0x303F);  // CJK symbols
}

bool IsSentenceEndPunct(std::uint32_t cp) noexcept {
    // Only true sentence-ending punctuation.  Commas (,/，) must NOT
    // break segments — they produce fragments too short for translation.
    return cp == '.' || cp == '!' || cp == '?' ||
           cp == 0x3002 || // 。
           cp == 0xFF01 || // ！
           cp == 0xFF1F;   // ？
}

std::int32_t DisplayWidthUtf8(const std::string & text) noexcept {
    std::int32_t width = 0;
    const auto * p = reinterpret_cast<const unsigned char *>(text.data());
    const auto * end = p + text.size();
    while (p < end) {
        std::uint32_t cp = 0;
        int bytes = 1;
        if (*p < 0x80) { cp = *p; }
        else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F; bytes = 2; }
        else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F; bytes = 3; }
        else if ((*p & 0xF8) == 0xF0) { cp = *p & 0x07; bytes = 4; }
        for (int i = 1; i < bytes && p + i < end; ++i) {
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        width += IsCjkCodepoint(cp) ? 2 : 1;
        p += bytes;
    }
    return width;
}

/// Get the first codepoint from a UTF-8 string.
std::uint32_t FirstCodepoint(const std::string & text) noexcept {
    if (text.empty()) return 0;
    const auto * p = reinterpret_cast<const unsigned char *>(text.data());
    const auto * end = p + text.size();
    std::uint32_t cp = 0;
    int bytes = 1;
    if (*p < 0x80) { cp = *p; }
    else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F; bytes = 2; }
    else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F; bytes = 3; }
    else if ((*p & 0xF8) == 0xF0) { cp = *p & 0x07; bytes = 4; }
    for (int i = 1; i < bytes && p + i < end; ++i) {
        cp = (cp << 6) | (p[i] & 0x3F);
    }
    return cp;
}

/// Get the last codepoint from a UTF-8 string.
std::uint32_t LastCodepoint(const std::string & text) noexcept {
    if (text.empty()) return 0;
    const auto * p = reinterpret_cast<const unsigned char *>(text.data());
    const auto * end = p + text.size();
    // Walk back from end to find start of last char
    const auto * last = end - 1;
    while (last > p && (*last & 0xC0) == 0x80) --last;
    std::uint32_t cp = 0;
    int bytes = 1;
    if (*last < 0x80) { cp = *last; }
    else if ((*last & 0xE0) == 0xC0) { cp = *last & 0x1F; bytes = 2; }
    else if ((*last & 0xF0) == 0xE0) { cp = *last & 0x0F; bytes = 3; }
    else if ((*last & 0xF8) == 0xF0) { cp = *last & 0x07; bytes = 4; }
    for (int i = 1; i < bytes && last + i < end; ++i) {
        cp = (cp << 6) | (last[i] & 0x3F);
    }
    return cp;
}

}  // namespace

bool IsAlignerLanguageSupported(const std::string & language) noexcept {
    std::string lower;
    lower.reserve(language.size());
    for (char c : language) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (int i = 0; i < kAlignerSupportedLanguageCount; ++i) {
        if (lower == kAlignerSupportedLanguages[i]) return true;
    }
    return false;
}

Status ValidateAlignResult(const AlignResult & result) {
    if (result.words.empty()) {
        return Status(StatusCode::kInvalidArgument, "align result has no words");
    }
    for (std::size_t i = 0; i < result.words.size(); ++i) {
        const auto & w = result.words[i];
        if (w.start_sec < 0.0 || w.end_sec < 0.0) {
            return Status(StatusCode::kOutOfRange, "negative timestamp in align result");
        }
        if (w.end_sec < w.start_sec) {
            return Status(StatusCode::kOutOfRange, "end < start in align result");
        }
    }
    return OkStatus();
}

std::vector<TimedSegment> WordsToSegments(
    const std::vector<AlignedWord> & words,
    std::int32_t max_segment_chars,
    double max_gap_sec) {

    std::vector<TimedSegment> segments;
    if (words.empty()) return segments;

    TimedSegment current;
    current.range.begin_ms = static_cast<std::int64_t>(words[0].start_sec * 1000.0);
    std::int32_t current_width = 0;

    for (std::size_t i = 0; i < words.size(); ++i) {
        const auto & w = words[i];
        const std::int32_t word_width = DisplayWidthUtf8(w.text);

        // Check if we should start a new segment
        bool force_new = false;
        if (!current.text.empty()) {
            // Gap between this word and previous
            const double prev_end = words[i - 1].end_sec;
            if (w.start_sec - prev_end > max_gap_sec) {
                force_new = true;
            }
            // Width overflow
            if (current_width + word_width + 1 > max_segment_chars) {
                force_new = true;
            }
            // Sentence-end punctuation in previous word
            if (IsSentenceEndPunct(LastCodepoint(words[i - 1].text))) {
                force_new = true;
            }
        }

        if (force_new && !current.text.empty()) {
            current.range.end_ms = static_cast<std::int64_t>(words[i - 1].end_sec * 1000.0);
            segments.push_back(std::move(current));
            current = TimedSegment{};
            current.range.begin_ms = static_cast<std::int64_t>(w.start_sec * 1000.0);
            current_width = 0;
        }

        // Append word to current segment.
        // Insert space only between non-CJK tokens; CJK tokens join directly.
        if (!current.text.empty()) {
            const bool prev_cjk = IsCjkCodepoint(LastCodepoint(current.text));
            const bool curr_cjk = IsCjkCodepoint(FirstCodepoint(w.text));
            if (!prev_cjk && !curr_cjk) {
                current.text += ' ';
                current_width += 1;
            }
        }
        current.text += w.text;
        current_width += word_width;
    }

    // Flush last segment
    if (!current.text.empty()) {
        current.range.end_ms = static_cast<std::int64_t>(words.back().end_sec * 1000.0);
        segments.push_back(std::move(current));
    }

    return segments;
}

}  // namespace qasr
