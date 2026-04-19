#include "tests/test_registry.h"

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "qasr/core/timestamp.h"
#include "qasr/runtime/model_bridge.h"
#include "qasr/subtitle/subtitle_writer.h"

// ========================================================================
// ParseOutputFormat
// ========================================================================

// Normal values
QASR_TEST(ParseOutputFormat_NormalSrt) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT(qasr::ParseOutputFormat("srt", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kSrt));
}

QASR_TEST(ParseOutputFormat_NormalVtt) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT(qasr::ParseOutputFormat("vtt", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kVtt));
}

QASR_TEST(ParseOutputFormat_NormalText) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT(qasr::ParseOutputFormat("text", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kText));
    QASR_EXPECT(qasr::ParseOutputFormat("txt", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kText));
}

QASR_TEST(ParseOutputFormat_NormalJson) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT(qasr::ParseOutputFormat("json", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kJson));
}

// Case-insensitive
QASR_TEST(ParseOutputFormat_CaseInsensitive) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT(qasr::ParseOutputFormat("SRT", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kSrt));
    QASR_EXPECT(qasr::ParseOutputFormat("Vtt", &fmt).ok());
    QASR_EXPECT_EQ(static_cast<int>(fmt), static_cast<int>(qasr::OutputFormat::kVtt));
}

// Error values
QASR_TEST(ParseOutputFormat_UnknownFormat) {
    qasr::OutputFormat fmt{};
    QASR_EXPECT_EQ(qasr::ParseOutputFormat("mp4", &fmt).code(), qasr::StatusCode::kInvalidArgument);
    QASR_EXPECT_EQ(qasr::ParseOutputFormat("", &fmt).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(ParseOutputFormat_NullOutput) {
    QASR_EXPECT_EQ(qasr::ParseOutputFormat("srt", nullptr).code(), qasr::StatusCode::kInvalidArgument);
}

// Extreme: very long string
QASR_TEST(ParseOutputFormat_ExtremeLongString) {
    qasr::OutputFormat fmt{};
    const std::string long_str(10000, 'x');
    QASR_EXPECT_EQ(qasr::ParseOutputFormat(long_str, &fmt).code(), qasr::StatusCode::kInvalidArgument);
}

// Random: fuzz-like loop
QASR_TEST(ParseOutputFormat_RandomStrings) {
    qasr::OutputFormat fmt{};
    std::srand(42);
    for (int i = 0; i < 100; ++i) {
        const int len = std::rand() % 20;
        std::string s;
        for (int j = 0; j < len; ++j) {
            s += static_cast<char>(std::rand() % 128);
        }
        // Should not crash, may succeed or fail
        (void)qasr::ParseOutputFormat(s, &fmt);
    }
}

// ========================================================================
// OutputFormatExtension
// ========================================================================

QASR_TEST(OutputFormatExtension_Normal) {
    QASR_EXPECT_EQ(qasr::OutputFormatExtension(qasr::OutputFormat::kSrt), std::string("srt"));
    QASR_EXPECT_EQ(qasr::OutputFormatExtension(qasr::OutputFormat::kVtt), std::string("vtt"));
    QASR_EXPECT_EQ(qasr::OutputFormatExtension(qasr::OutputFormat::kText), std::string("txt"));
    QASR_EXPECT_EQ(qasr::OutputFormatExtension(qasr::OutputFormat::kJson), std::string("json"));
}

// ========================================================================
// LayoutSubtitles
// ========================================================================

// Normal: single segment fits in one cue
QASR_TEST(LayoutSubtitles_SingleSegment) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"Hello world", {0, 3000}});

    const auto cues = qasr::LayoutSubtitles(segs, {});
    QASR_EXPECT_EQ(cues.size(), std::size_t(1));
    QASR_EXPECT_EQ(cues[0].text, std::string("Hello world"));
    QASR_EXPECT_EQ(cues[0].range.begin_ms, std::int64_t(0));
    QASR_EXPECT_EQ(cues[0].range.end_ms, std::int64_t(3000));
}

// Normal: multiple segments
QASR_TEST(LayoutSubtitles_MultipleSegments) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"First.", {0, 2000}});
    segs.push_back({"Second.", {2000, 5000}});
    segs.push_back({"Third.", {5000, 8000}});

    const auto cues = qasr::LayoutSubtitles(segs, {});
    QASR_EXPECT_EQ(cues.size(), std::size_t(3));
    QASR_EXPECT_EQ(cues[0].text, std::string("First."));
    QASR_EXPECT_EQ(cues[1].text, std::string("Second."));
    QASR_EXPECT_EQ(cues[2].text, std::string("Third."));
}

// Extreme: empty segments
QASR_TEST(LayoutSubtitles_EmptyInput) {
    const auto cues = qasr::LayoutSubtitles({}, {});
    QASR_EXPECT_EQ(cues.size(), std::size_t(0));
}

// Extreme: whitespace-only segment
QASR_TEST(LayoutSubtitles_WhitespaceOnly) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"   ", {0, 1000}});
    const auto cues = qasr::LayoutSubtitles(segs, {});
    QASR_EXPECT_EQ(cues.size(), std::size_t(0));
}

// Extreme: very long segment forced to split
QASR_TEST(LayoutSubtitles_LongSegmentSplits) {
    std::string long_text(200, 'A');
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({long_text, {0, 10000}});

    qasr::SubtitlePolicy policy;
    policy.max_line_width = 20;
    policy.max_line_count = 1;

    const auto cues = qasr::LayoutSubtitles(segs, policy);
    QASR_EXPECT(cues.size() > 1);

    // All cues together should cover the full text
    std::string concat;
    for (const auto & c : cues) {
        concat += c.text;
    }
    QASR_EXPECT_EQ(concat, long_text);

    // Time should be covered
    QASR_EXPECT_EQ(cues.front().range.begin_ms, std::int64_t(0));
    QASR_EXPECT_EQ(cues.back().range.end_ms, std::int64_t(10000));
}

// Extreme: CJK text (double-width)
QASR_TEST(LayoutSubtitles_CjkText) {
    // 6 CJK chars = 12 display width, policy 10 => split
    std::string cjk;
    for (int i = 0; i < 6; ++i) {
        cjk += "\xe4\xb8\xad";  // U+4E2D '中'
    }
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({cjk, {0, 6000}});

    qasr::SubtitlePolicy policy;
    policy.max_line_width = 5;
    policy.max_line_count = 1;

    const auto cues = qasr::LayoutSubtitles(segs, policy);
    QASR_EXPECT(cues.size() >= 2);
}

// Error: "-->" in text must be sanitized
QASR_TEST(LayoutSubtitles_SanitizesArrow) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"A --> B", {0, 1000}});
    const auto cues = qasr::LayoutSubtitles(segs, {});
    QASR_EXPECT_EQ(cues.size(), std::size_t(1));
    QASR_EXPECT_EQ(cues[0].text, std::string("A -> B"));
}

// Random: random segments
QASR_TEST(LayoutSubtitles_RandomSegments) {
    std::srand(123);
    std::vector<qasr::TimedSegment> segs;
    std::int64_t t = 0;
    for (int i = 0; i < 50; ++i) {
        const int len = 1 + std::rand() % 100;
        std::string text(static_cast<std::size_t>(len), static_cast<char>('A' + (i % 26)));
        const std::int64_t dur = 500 + std::rand() % 5000;
        segs.push_back({text, {t, t + dur}});
        t += dur;
    }
    // Should not crash
    const auto cues = qasr::LayoutSubtitles(segs, {});
    QASR_EXPECT(cues.size() >= 1);
}

// ========================================================================
// WriteSrt
// ========================================================================

// Normal: basic SRT output
QASR_TEST(WriteSrt_NormalOutput) {
    std::vector<qasr::SubtitleCue> cues;
    cues.push_back({{0, 3500}, "Hello"});
    cues.push_back({{4000, 7200}, "World"});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSrt(cues, out).ok());
    const std::string result = out.str();

    // SRT uses comma for ms separator, seq starts at 1
    QASR_EXPECT(result.find("1\n00:00:00,000 --> 00:00:03,500\nHello") != std::string::npos);
    QASR_EXPECT(result.find("2\n00:00:04,000 --> 00:00:07,200\nWorld") != std::string::npos);
}

// Extreme: empty cues
QASR_TEST(WriteSrt_EmptyCues) {
    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSrt({}, out).ok());
    QASR_EXPECT_EQ(out.str(), std::string(""));
}

// Extreme: timestamp with hours
QASR_TEST(WriteSrt_LongTimestamp) {
    std::vector<qasr::SubtitleCue> cues;
    cues.push_back({{3723004, 7200000}, "Long"});  // 01:02:03,004

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSrt(cues, out).ok());
    QASR_EXPECT(out.str().find("01:02:03,004") != std::string::npos);
}

// Error: negative timestamp
QASR_TEST(WriteSrt_NegativeTimestamp) {
    std::vector<qasr::SubtitleCue> cues;
    cues.push_back({{-1, 1000}, "Bad"});

    std::ostringstream out;
    QASR_EXPECT(!qasr::WriteSrt(cues, out).ok());
}

// ========================================================================
// WriteVtt
// ========================================================================

// Normal: basic VTT output
QASR_TEST(WriteVtt_NormalOutput) {
    std::vector<qasr::SubtitleCue> cues;
    cues.push_back({{0, 3500}, "Hello"});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteVtt(cues, out).ok());
    const std::string result = out.str();

    // VTT starts with WEBVTT, uses period for ms separator
    QASR_EXPECT(result.find("WEBVTT") == 0);
    QASR_EXPECT(result.find("00:00:00.000 --> 00:00:03.500") != std::string::npos);
}

// Extreme: empty
QASR_TEST(WriteVtt_EmptyCues) {
    std::ostringstream out;
    QASR_EXPECT(qasr::WriteVtt({}, out).ok());
    QASR_EXPECT(out.str().find("WEBVTT") == 0);
}

// ========================================================================
// WriteSegmentJson
// ========================================================================

// Normal
QASR_TEST(WriteSegmentJson_Normal) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"Hello", {0, 2000}});
    segs.push_back({"World", {2000, 5000}});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSegmentJson(segs, 5.0, out).ok());
    const std::string result = out.str();
    QASR_EXPECT(result.find("\"duration\"") != std::string::npos);
    QASR_EXPECT(result.find("\"segments\"") != std::string::npos);
    QASR_EXPECT(result.find("\"Hello\"") != std::string::npos);
    QASR_EXPECT(result.find("\"World\"") != std::string::npos);
}

// Extreme: empty
QASR_TEST(WriteSegmentJson_Empty) {
    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSegmentJson({}, 0.0, out).ok());
    QASR_EXPECT(out.str().find("\"segments\": [") != std::string::npos);
}

// Error: text with special chars (JSON escaping)
QASR_TEST(WriteSegmentJson_EscapesSpecialChars) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"line1\nline2\t\"quoted\"\\back", {0, 1000}});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteSegmentJson(segs, 1.0, out).ok());
    const std::string result = out.str();
    QASR_EXPECT(result.find("\\n") != std::string::npos);
    QASR_EXPECT(result.find("\\t") != std::string::npos);
    QASR_EXPECT(result.find("\\\"") != std::string::npos);
    QASR_EXPECT(result.find("\\\\") != std::string::npos);
}

// ========================================================================
// WriteText
// ========================================================================

// Normal
QASR_TEST(WriteText_Normal) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"Hello", {0, 1000}});
    segs.push_back({"World", {1000, 2000}});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteText(segs, out).ok());
    QASR_EXPECT_EQ(out.str(), std::string("Hello\nWorld\n"));
}

// Extreme: empty
QASR_TEST(WriteText_Empty) {
    std::ostringstream out;
    QASR_EXPECT(qasr::WriteText({}, out).ok());
    QASR_EXPECT_EQ(out.str(), std::string(""));
}

// Error: whitespace-only text stripped
QASR_TEST(WriteText_WhitespaceStripped) {
    std::vector<qasr::TimedSegment> segs;
    segs.push_back({"  ", {0, 1000}});
    segs.push_back({"real text", {1000, 2000}});

    std::ostringstream out;
    QASR_EXPECT(qasr::WriteText(segs, out).ok());
    QASR_EXPECT_EQ(out.str(), std::string("real text\n"));
}

// Random
QASR_TEST(WriteText_RandomSegments) {
    std::srand(99);
    std::vector<qasr::TimedSegment> segs;
    for (int i = 0; i < 30; ++i) {
        const int len = 1 + std::rand() % 50;
        std::string text(static_cast<std::size_t>(len), static_cast<char>('a' + (i % 26)));
        segs.push_back({text, {static_cast<std::int64_t>(i * 1000), static_cast<std::int64_t>((i + 1) * 1000)}});
    }
    std::ostringstream out;
    QASR_EXPECT(qasr::WriteText(segs, out).ok());
    QASR_EXPECT(!out.str().empty());
}
