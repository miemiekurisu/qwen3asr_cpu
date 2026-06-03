/* Tests for qwen_utf8_truncate — the streaming decoder's safety net
 * against partial UTF-8 character tails when the decoder cuts
 * mid-token.  Qwen3's BPE token boundaries don't align with UTF-8
 * character boundaries, so a cut at a token boundary can leave
 * partial multi-byte sequences at the tail.  These tests pin down
 * the truncation behavior:
 *   - ASCII-only: no-op (no truncation)
 *   - Aligned 2/3/4-byte UTF-8: no-op
 *   - Partial 2/3/4-byte UTF-8: trim to last complete code point
 *   - Empty / NULL: no crash, returns 0
 *   - Continuation-byte-only tail: trims back to the last lead
 *
 * The streaming decoder applies this in three places: per-chunk
 * stable/tentative piece building, and the final result string. */
extern "C" {
#include "qwen_asr.h"
}

#include <cstring>
#include <string>

#include "tests/test_registry.h"

QASR_TEST(Utf8TruncateAsciiIsNoop) {
    char buf[] = "hello world";
    size_t n = qwen_utf8_truncate(buf, strlen(buf));
    QASR_EXPECT_EQ(n, std::strlen("hello world"));
    QASR_EXPECT_EQ(std::string(buf), std::string("hello world"));
}

QASR_TEST(Utf8TruncateAlignedChineseIsNoop) {
    /* "你好世界" is 4 three-byte UTF-8 chars = 12 bytes total.
     * Ending on a complete code point: no truncation. */
    char buf[32];
    std::memcpy(buf, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c", 12);
    buf[12] = '\0';
    size_t n = qwen_utf8_truncate(buf, 12);
    QASR_EXPECT_EQ(n, std::size_t(12));
    QASR_EXPECT_EQ(std::string(buf), std::string("\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"));
}

QASR_TEST(Utf8TruncatePartialTwoByteTrimsOneByte) {
    /* "好" (3 bytes: E5 A5 BD) with one trailing byte cut: 2 bytes
     * left (E5 A5).  Expected trim: 0 bytes (need at least one
     * lead byte to anchor).  Wait — E5 is a lead (1110xxxx → 3 bytes
     * promised), so 2 bytes (E5 A5) is partial, trim back to 0. */
    char buf[8];
    std::memcpy(buf, "\xe5\xa5", 2);
    buf[2] = '\0';
    size_t n = qwen_utf8_truncate(buf, 2);
    QASR_EXPECT_EQ(n, std::size_t(0));
    QASR_EXPECT_EQ(std::string(buf), std::string(""));
}

QASR_TEST(Utf8TruncatePartialThreeByteKeepsPrior) {
    /* "你好" (6 bytes) followed by partial 2 bytes of a third char:
     * 8 bytes total.  Expected: trim back to "你好" (6 bytes). */
    char buf[16];
    std::memcpy(buf, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xbd", 8);
    buf[8] = '\0';
    size_t n = qwen_utf8_truncate(buf, 8);
    QASR_EXPECT_EQ(n, std::size_t(6));
    QASR_EXPECT_EQ(std::string(buf),
                   std::string("\xe4\xbd\xa0\xe5\xa5\xbd"));
}

QASR_TEST(Utf8TruncateMultipleCompletePlusPartial) {
    /* "ABC你" + partial 2 bytes of next char (4 bytes total at tail):
     *   bytes = 0x41 0x42 0x43 0xE4 0xBD 0xA0 0xE4 0xBD
     * Expected trim: back to "ABC你" (6 bytes), since the trailing
     * 0xE4 0xBD is partial (0xE4 promises 3 bytes). */
    char buf[16];
    std::memcpy(buf, "\x41\x42\x43\xe4\xbd\xa0\xe4\xbd", 8);
    buf[8] = '\0';
    size_t n = qwen_utf8_truncate(buf, 8);
    QASR_EXPECT_EQ(n, std::size_t(6));
    QASR_EXPECT_EQ(std::string(buf), std::string("\x41\x42\x43\xe4\xbd\xa0"));
}

QASR_TEST(Utf8TruncateFourByteEmojiPartial) {
    /* U+1F600 (😀) is 4 bytes: F0 9F 98 80.  With 2 trailing bytes
     * left: F0 9F → partial, trim to before F0. */
    char buf[16];
    std::memcpy(buf, "Hi\xf0\x9f", 4);
    buf[4] = '\0';
    size_t n = qwen_utf8_truncate(buf, 4);
    QASR_EXPECT_EQ(n, std::size_t(2));
    QASR_EXPECT_EQ(std::string(buf), std::string("Hi"));
}

QASR_TEST(Utf8TruncateEmptyInput) {
    char buf[1] = {'\0'};
    size_t n = qwen_utf8_truncate(buf, 0);
    QASR_EXPECT_EQ(n, std::size_t(0));
    QASR_EXPECT_EQ(std::string(buf), std::string(""));
}

QASR_TEST(Utf8TruncateSingleByteAscii) {
    char buf[2] = {'A', '\0'};
    size_t n = qwen_utf8_truncate(buf, 1);
    QASR_EXPECT_EQ(n, std::size_t(1));
    QASR_EXPECT_EQ(std::string(buf), std::string("A"));
}

QASR_TEST(Utf8TruncateFourByteComplete) {
    /* "A" + 😀 complete (4 bytes) = 5 bytes total.  Aligned: no-op. */
    char buf[8];
    std::memcpy(buf, "A\xf0\x9f\x98\x80", 5);
    buf[5] = '\0';
    size_t n = qwen_utf8_truncate(buf, 5);
    QASR_EXPECT_EQ(n, std::size_t(5));
    QASR_EXPECT_EQ(std::string(buf), std::string("A\xf0\x9f\x98\x80"));
}

QASR_TEST(Utf8TruncateContinuationByteOnly) {
    /* Only continuation bytes at the tail (no lead): 0x80 0x80.
     * The scan walks back and finds no lead within 4 bytes (assuming
     * the rest of the string is well-formed); it should trim back
     * before these orphan continuation bytes.  We use a 6-byte input
     * "AB" + 0x80 0x80 to make sure the leading "AB" is preserved. */
    char buf[8];
    std::memcpy(buf, "AB\x80\x80", 4);
    buf[4] = '\0';
    size_t n = qwen_utf8_truncate(buf, 4);
    QASR_EXPECT_EQ(n, std::size_t(2));
    QASR_EXPECT_EQ(std::string(buf), std::string("AB"));
}
