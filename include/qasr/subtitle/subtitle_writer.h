#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/core/timestamp.h"
#include "qasr/runtime/model_bridge.h"

namespace qasr {

/// Output format for subtitle/transcription export.
enum class OutputFormat {
    kText = 0,   // plain text
    kSrt,        // SubRip (.srt)
    kVtt,        // WebVTT (.vtt)
    kJson,       // JSON with segments
};

/// Policy for laying out subtitle lines.
struct SubtitlePolicy {
    std::int32_t max_line_width = 42;  // max chars per line (CJK = 2)
    std::int32_t max_line_count = 2;   // max lines per subtitle screen
    float max_gap_sec = 3.0f;          // force new subtitle on long pause
};

/// A single subtitle cue ready for output.
struct SubtitleCue {
    TimestampRange range;
    std::string text;
};

/// Parse an output format string. Returns kInvalidArgument on unknown format.
/// Pre: format is one of "text", "srt", "vtt", "json" (case-insensitive).
/// Post: *out set to parsed format.
/// Thread-safe: yes.
Status ParseOutputFormat(const std::string & format, OutputFormat * out);

/// Get the canonical file extension for an output format (without dot).
/// Thread-safe: yes.
std::string OutputFormatExtension(OutputFormat format) noexcept;

/// Lay out timed segments into subtitle cues according to policy.
/// Pre: segments must be sorted by begin_ms ascending.
/// Post: returns cues suitable for SRT/VTT output.
/// Thread-safe: yes.
std::vector<SubtitleCue> LayoutSubtitles(
    const std::vector<TimedSegment> & segments,
    const SubtitlePolicy & policy);

/// Write subtitle cues in SRT format.
/// Pre: cues must be ordered by timestamp.
/// Post: output written to stream.
/// Thread-safe: yes (different streams).
Status WriteSrt(const std::vector<SubtitleCue> & cues, std::ostream & out);

/// Write a single SRT cue (for incremental output).
/// 'index' is the 1-based cue number.
/// Thread-safe: yes (different streams).
Status WriteSrtCue(const SubtitleCue & cue, int index, std::ostream & out);

/// Write subtitle cues in WebVTT format.
/// Pre: cues must be ordered by timestamp.
/// Post: output written to stream.
/// Thread-safe: yes (different streams).
Status WriteVtt(const std::vector<SubtitleCue> & cues, std::ostream & out);

/// Write VTT header (call once before WritVttCue).
void WriteVttHeader(std::ostream & out);

/// Write a single VTT cue (for incremental output).
/// Thread-safe: yes (different streams).
Status WriteVttCue(const SubtitleCue & cue, std::ostream & out);

/// Write timed segments as JSON.
/// Pre: segments must be valid.
/// Post: output written to stream.
/// Thread-safe: yes (different streams).
Status WriteSegmentJson(const std::vector<TimedSegment> & segments,
                        double audio_duration_sec, std::ostream & out);

/// Write plain text (no timestamps).
/// Thread-safe: yes.
Status WriteText(const std::vector<TimedSegment> & segments, std::ostream & out);

}  // namespace qasr
