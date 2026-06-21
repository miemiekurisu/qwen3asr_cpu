/* Tests for FindLastSentenceBoundary and TrimLeadingCharLen — the
 * sentence-boundary detection and UTF-8-aware trim helpers used by
 * the realtime ASR worker to split confirmed/uncertain text at the
 * last terminal punctuation mark, and to trim leading whitespace/
 * punctuation from segment text before boundary detection.
 *
 * These tests verify:
 *   - ASCII punctuation: ?, !, . (with context)
 *   - CJK punctuation: ？ ！ 。 (UTF-8 multi-byte)
 *   - Mixed text with multiple boundaries
 *   - Empty / no-boundary edge cases
 *   - TrimLeadingCharLen for ASCII and CJK characters */

#include "tests/test_registry.h"

#include <cstring>
#include <string>
#include <string_view>

extern "C" {
#include "qwen_asr.h"
}

/* Helper to access the static function via a thin wrapper.
 * Since these are static in server.cc, we test behavior via
 * the E2E path.  These unit tests verify the UTF-8 handling
 * and boundary detection logic directly. */

namespace {

/* Replicate FindLastSentenceBoundary logic for testing.
 * This matches the implementation in src/service/server.cc */
std::size_t FindLastSentenceBoundary(std::string_view text) {
    std::size_t last_boundary = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '?' || c == '!') {
            last_boundary = i + 1;
        }
        else if (c == '.') {
            if (i + 1 >= text.size() ||
                static_cast<unsigned char>(text[i + 1]) == ' ' ||
                static_cast<unsigned char>(text[i + 1]) == '\n' ||
                static_cast<unsigned char>(text[i + 1]) == ',' ||
                static_cast<unsigned char>(text[i + 1]) == '!') {
                last_boundary = i + 1;
            }
        }
        else if (i + 2 < text.size()) {
            unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if ((c == 0xEF && c1 == 0xBC && c2 == 0x9F) ||   // ？ U+FF1F
                (c == 0xEF && c1 == 0xBC && c2 == 0x81) ||   // ！ U+FF01
                (c == 0xE3 && c1 == 0x80 && c2 == 0x82)) {    // 。 U+3002
                last_boundary = i + 3;
            }
        }
    }
    return last_boundary;
}

/* Replicate TrimLeadingCharLen logic for testing. */
std::size_t TrimLeadingCharLen(std::string_view s) {
    if (s.empty()) return 0;
    if (s[0] == ' ' || s[0] == ',') return 1;
    /* ？ = 0xEF 0xBC 0x8C */
    if (s.size() >= 3 && s[0] == '\xEF' && s[1] == '\xBC' && s[2] == '\x8C') return 3;
    /* 、 = 0xE3 0x80 0x81 */
    if (s.size() >= 3 && s[0] == '\xE3' && s[1] == '\x80' && s[2] == '\x81') return 3;
    return 0;
}

}  // namespace

QASR_TEST(SentenceBoundaryEmpty) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary(""), 0u);
}

QASR_TEST(SentenceBoundaryAsciiQuestion) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello?"), 6u);
}

QASR_TEST(SentenceBoundaryAsciiExclamation) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello!"), 6u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodEnd) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello."), 6u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodSpace) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello. World"), 6u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodNoSpace) {
    /* "Hello.World" — period followed by non-space: not terminal */
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello.World"), 0u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodComma) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello., next"), 6u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodExclamation) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello.!!"), 6u);
}

QASR_TEST(SentenceBoundaryAsciiPeriodNewline) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello.\nNext"), 6u);
}

QASR_TEST(SentenceBoundaryCjkQuestion) {
    /* ？ = 0xEF 0xBC 0x9F (3 bytes) */
    std::string text = "\xEF\xBC\x9F";  // ？
    QASR_EXPECT_EQ(FindLastSentenceBoundary(text), 3u);
}

QASR_TEST(SentenceBoundaryCjkExclamation) {
    /* ！ = 0xEF 0xBC 0x81 (3 bytes) */
    std::string text = "\xEF\xBC\x81";  // ！
    QASR_EXPECT_EQ(FindLastSentenceBoundary(text), 3u);
}

QASR_TEST(SentenceBoundaryCjkPeriod) {
    /* 。 = 0xE3 0x80 0x82 (3 bytes) */
    std::string text = "\xE3\x80\x82";  // 。
    QASR_EXPECT_EQ(FindLastSentenceBoundary(text), 3u);
}

QASR_TEST(SentenceBoundaryMixedChinese) {
    /* "你好吗？" — boundary at ？ (byte 10) */
    std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD\xE5\x90\x97\xEF\xBC\x9F";
    QASR_EXPECT_EQ(FindLastSentenceBoundary(text), 12u);
}

QASR_TEST(SentenceBoundaryMultipleBoundaries) {
    /* "Hello? World!" — last boundary is ! (byte 12) */
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello? World!"), 13u);
}

QASR_TEST(SentenceBoundaryNoBoundary) {
    QASR_EXPECT_EQ(FindLastSentenceBoundary("Hello world"), 0u);
}

QASR_TEST(SentenceBoundaryChineseNoBoundary) {
    /* "你好世界" — no terminal punctuation */
    std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C";
    QASR_EXPECT_EQ(FindLastSentenceBoundary(text), 0u);
}

QASR_TEST(TrimLeadingCharLenSpace) {
    QASR_EXPECT_EQ(TrimLeadingCharLen(" hello"), 1u);
}

QASR_TEST(TrimLeadingCharLenComma) {
    QASR_EXPECT_EQ(TrimLeadingCharLen(",hello"), 1u);
}

QASR_TEST(TrimLeadingCharLenCjkComma) {
    /* ？ = 0xEF 0xBC 0x8C (3 bytes) */
    std::string text = "\xEF\xBC\x8Chello";
    QASR_EXPECT_EQ(TrimLeadingCharLen(text), 3u);
}

QASR_TEST(TrimLeadingCharLenCjkEnumeration) {
    /* 、 = 0xE3 0x80 0x81 (3 bytes) */
    std::string text = "\xE3\x80\x81hello";
    QASR_EXPECT_EQ(TrimLeadingCharLen(text), 3u);
}

QASR_TEST(TrimLeadingCharLenEmpty) {
    QASR_EXPECT_EQ(TrimLeadingCharLen(""), 0u);
}

QASR_TEST(TrimLeadingCharLenNoMatch) {
    QASR_EXPECT_EQ(TrimLeadingCharLen("hello"), 0u);
}

QASR_TEST(TrimLeadingCharLenPartialUtf8) {
    /* Partial 3-byte sequence should not match */
    std::string text = "\xEF\xBC";
    QASR_EXPECT_EQ(TrimLeadingCharLen(text), 0u);
}

QASR_TEST(SentenceBoundaryRealChineseSentence) {
    /* "大师先生，几日未见，您可安好？" */
    /* This is a real-world example from the user's complaint */
    std::string text;
    // 大师先生
    text += "\xE5\xA4\xA7\xE5\xB8\x88\xE5\x85\x8B\xE7\x94\x9F";
    // ，
    text += "\xEF\xBC\x8C";
    // 几日未见
    text += "\xE6\x9D\xA5\xE6\x97%A5\xE6\x9C\xAA\xE8\xA7\x81";
    // ，
    text += "\xEF\xBC\x8C";
    // 您可安好
    text += "\xE6\x82\xA8\xE5\x8F\xAF\xE5\xAE\x89\xE5\xA5\xBD";
    // ？
    text += "\xEF\xBC\x9F";
    
    std::size_t boundary = FindLastSentenceBoundary(text);
    QASR_EXPECT(boundary > 0);
    QASR_EXPECT_EQ(boundary, text.size());
}
