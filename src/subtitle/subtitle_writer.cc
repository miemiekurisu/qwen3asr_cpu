#include "qasr/subtitle/subtitle_writer.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace qasr {
namespace {

std::string ToLowerAscii(const std::string & s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

/// Count display width: ASCII = 1, CJK (U+2E80..U+9FFF, U+F900..U+FAFF) = 2,
/// other multibyte = 1.
std::int32_t DisplayWidth(std::string_view text) noexcept {
    std::int32_t width = 0;
    const auto * p = reinterpret_cast<const unsigned char *>(text.data());
    const auto * end = p + text.size();
    while (p < end) {
        uint32_t cp = 0;
        int bytes = 1;
        if (*p < 0x80) {
            cp = *p;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = *p & 0x1F;
            bytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = *p & 0x0F;
            bytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = *p & 0x07;
            bytes = 4;
        }
        for (int i = 1; i < bytes && p + i < end; ++i) {
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        // CJK Unified Ideographs and extensions
        if ((cp >= 0x2E80 && cp <= 0x9FFF) ||
            (cp >= 0xF900 && cp <= 0xFAFF) ||
            (cp >= 0xFF01 && cp <= 0xFF60) ||   // fullwidth forms
            (cp >= 0x20000 && cp <= 0x2FA1F)) {
            width += 2;
        } else {
            width += 1;
        }
        p += bytes;
    }
    return width;
}

/// Replace "-->" in subtitle text with "->" (SRT/VTT reserved).
std::string SanitizeSubtitleText(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (i + 2 < text.size() && text[i] == '-' && text[i + 1] == '-' && text[i + 2] == '>') {
            result += "->";
            i += 2;
        } else {
            result += text[i];
        }
    }
    return result;
}

std::string TrimWhitespace(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.remove_suffix(1);
    return std::string(text);
}

}  // namespace

Status ParseOutputFormat(const std::string & format, OutputFormat * out) {
    if (out == nullptr) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    const std::string lower = ToLowerAscii(format);
    if (lower == "text" || lower == "txt") {
        *out = OutputFormat::kText;
    } else if (lower == "srt") {
        *out = OutputFormat::kSrt;
    } else if (lower == "vtt") {
        *out = OutputFormat::kVtt;
    } else if (lower == "json") {
        *out = OutputFormat::kJson;
    } else {
        return Status(StatusCode::kInvalidArgument, "unknown output format: " + format);
    }
    return OkStatus();
}

std::string OutputFormatExtension(OutputFormat format) noexcept {
    switch (format) {
        case OutputFormat::kText: return "txt";
        case OutputFormat::kSrt:  return "srt";
        case OutputFormat::kVtt:  return "vtt";
        case OutputFormat::kJson: return "json";
    }
    return "txt";
}

std::vector<SubtitleCue> LayoutSubtitles(
    const std::vector<TimedSegment> & segments,
    const SubtitlePolicy & policy) {
    std::vector<SubtitleCue> cues;
    if (segments.empty()) return cues;

    for (const auto & seg : segments) {
        const std::string trimmed = TrimWhitespace(seg.text);
        if (trimmed.empty()) continue;

        const std::int32_t seg_width = DisplayWidth(trimmed);
        const std::int32_t effective_max = policy.max_line_width * policy.max_line_count;

        if (seg_width <= effective_max) {
            // Fits in one cue
            SubtitleCue cue;
            cue.range = seg.range;
            cue.text = SanitizeSubtitleText(trimmed);
            cues.push_back(std::move(cue));
        } else {
            // Split long segment into multiple cues by approximate width
            // Distribute time proportionally
            const double total_ms = static_cast<double>(seg.range.end_ms - seg.range.begin_ms);

            // Split by bytes proportionally (approximate)
            const std::size_t text_len = trimmed.size();
            std::size_t pos = 0;
            std::int64_t cue_begin = seg.range.begin_ms;

            while (pos < text_len) {
                // Find split point respecting effective_max display width
                std::size_t chunk_end = pos;
                std::int32_t chunk_width = 0;
                while (chunk_end < text_len && chunk_width < effective_max) {
                    const unsigned char byte = static_cast<unsigned char>(trimmed[chunk_end]);
                    std::size_t char_bytes = 1;
                    uint32_t cp = byte;
                    if (byte >= 0xF0) { char_bytes = 4; cp = byte & 0x07; }
                    else if (byte >= 0xE0) { char_bytes = 3; cp = byte & 0x0F; }
                    else if (byte >= 0xC0) { char_bytes = 2; cp = byte & 0x1F; }
                    for (std::size_t i = 1; i < char_bytes && chunk_end + i < text_len; ++i)
                        cp = (cp << 6) | (static_cast<unsigned char>(trimmed[chunk_end + i]) & 0x3F);

                    int w = 1;
                    if ((cp >= 0x2E80 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
                        (cp >= 0xFF01 && cp <= 0xFF60) || (cp >= 0x20000 && cp <= 0x2FA1F))
                        w = 2;

                    if (chunk_width + w > effective_max) break;
                    chunk_width += w;
                    chunk_end += char_bytes;
                }
                if (chunk_end == pos) {
                    // Safety: advance at least one char to avoid infinite loop
                    chunk_end = pos + 1;
                    while (chunk_end < text_len &&
                           (static_cast<unsigned char>(trimmed[chunk_end]) & 0xC0) == 0x80)
                        ++chunk_end;
                }

                // Proportional time for this chunk
                const double fraction = static_cast<double>(chunk_end - pos) / static_cast<double>(text_len);
                const auto chunk_duration_ms = static_cast<std::int64_t>(fraction * total_ms);
                const std::int64_t cue_end = (chunk_end >= text_len)
                    ? seg.range.end_ms
                    : cue_begin + (std::max)(chunk_duration_ms, static_cast<std::int64_t>(1));

                SubtitleCue cue;
                cue.range.begin_ms = cue_begin;
                cue.range.end_ms = cue_end;
                cue.text = SanitizeSubtitleText(trimmed.substr(pos, chunk_end - pos));
                cues.push_back(std::move(cue));

                pos = chunk_end;
                cue_begin = cue_end;
            }
        }
    }

    // Apply max_gap_sec: if gap between consecutive cues > max_gap, leave as-is
    // (each cue is independent); if gap is small, cues may overlap (clamp end to next start)
    for (std::size_t i = 0; i + 1 < cues.size(); ++i) {
        const std::int64_t gap_ms = cues[i + 1].range.begin_ms - cues[i].range.end_ms;
        if (gap_ms < 0) {
            // Overlap: clamp
            cues[i].range.end_ms = cues[i + 1].range.begin_ms;
        }
    }

    return cues;
}

Status WriteSrt(const std::vector<SubtitleCue> & cues, std::ostream & out) {
    for (std::size_t i = 0; i < cues.size(); ++i) {
        Status status = WriteSrtCue(cues[i], static_cast<int>(i + 1), out);
        if (!status.ok()) return status;
    }
    return OkStatus();
}

Status WriteSrtCue(const SubtitleCue & cue, int index, std::ostream & out) {
    std::string start_ts, end_ts;
    Status status = FormatSrtTimestamp(cue.range.begin_ms, &start_ts);
    if (!status.ok()) return status;
    status = FormatSrtTimestamp(cue.range.end_ms, &end_ts);
    if (!status.ok()) return status;

    out << index << "\n"
        << start_ts << " --> " << end_ts << "\n"
        << cue.text << "\n\n";
    return OkStatus();
}

Status WriteVtt(const std::vector<SubtitleCue> & cues, std::ostream & out) {
    WriteVttHeader(out);
    for (const auto & cue : cues) {
        Status status = WriteVttCue(cue, out);
        if (!status.ok()) return status;
    }
    return OkStatus();
}

void WriteVttHeader(std::ostream & out) {
    out << "WEBVTT\n\n";
}

Status WriteVttCue(const SubtitleCue & cue, std::ostream & out) {
    std::string start_ts, end_ts;
    Status status = FormatJsonTimestamp(cue.range.begin_ms, &start_ts);
    if (!status.ok()) return status;
    status = FormatJsonTimestamp(cue.range.end_ms, &end_ts);
    if (!status.ok()) return status;

    out << start_ts << " --> " << end_ts << "\n"
        << cue.text << "\n\n";
    return OkStatus();
}

Status WriteSegmentJson(const std::vector<TimedSegment> & segments,
                        double audio_duration_sec, std::ostream & out) {
    out << "{\n  \"duration\": " << audio_duration_sec << ",\n  \"segments\": [\n";
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto & seg = segments[i];
        const double start_sec = static_cast<double>(seg.range.begin_ms) / 1000.0;
        const double end_sec = static_cast<double>(seg.range.end_ms) / 1000.0;

        // Escape text for JSON
        std::string escaped;
        for (char c : seg.text) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }

        out << "    {\"id\": " << i
            << ", \"start\": " << start_sec
            << ", \"end\": " << end_sec
            << ", \"text\": \"" << escaped << "\"}";
        if (i + 1 < segments.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
    return OkStatus();
}

Status WriteText(const std::vector<TimedSegment> & segments, std::ostream & out) {
    for (const auto & seg : segments) {
        const std::string trimmed = TrimWhitespace(seg.text);
        if (!trimmed.empty()) {
            out << trimmed << "\n";
        }
    }
    return OkStatus();
}

}  // namespace qasr
