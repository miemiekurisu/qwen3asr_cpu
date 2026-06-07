/*
 * utf8_test.cc - Unit tests for the shared UTF-8 helpers extracted to
 * `qasr::base::IsUtf8Continuation` and `qasr::base::CountUtf8Codepoints`.
 *
 * These tests pin down the contract that the two prior call sites
 * (`runtime/model_bridge.cc` and `service/realtime.cc`) relied on:
 *   - ASCII strings: codepoint count == byte count
 *   - 2-byte UTF-8 sequences: each pair counts as one codepoint
 *   - 3-byte UTF-8 sequences: each triple counts as one codepoint
 *   - 4-byte UTF-8 sequences: each quadruple counts as one codepoint
 *   - The continuation-byte predicate matches 0b10xxxxxx.
 */
#include "tests/test_registry.h"

#include <string>
#include <string_view>

#include "qasr/base/utf8.h"

QASR_TEST(Utf8IsContinuationAscii) {
    for (unsigned int b = 0; b < 0x80; ++b) {
        QASR_EXPECT(!qasr::base::IsUtf8Continuation(static_cast<unsigned char>(b)));
    }
}

QASR_TEST(Utf8IsContinuationContinuationBytes) {
    for (unsigned int b = 0x80; b < 0xC0; ++b) {
        QASR_EXPECT(qasr::base::IsUtf8Continuation(static_cast<unsigned char>(b)));
    }
}

QASR_TEST(Utf8IsContinuationLeadingBytes) {
    // All leading bytes (0b11xxxxxx) and the 0xC0/0xC1 invalid leads.
    for (unsigned int b = 0xC0; b <= 0xFF; ++b) {
        QASR_EXPECT(!qasr::base::IsUtf8Continuation(static_cast<unsigned char>(b)));
    }
}

QASR_TEST(Utf8CountEmpty) {
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(std::string_view()), 0u);
}

QASR_TEST(Utf8CountAscii) {
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(std::string_view("")), 0u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(std::string_view("a")), 1u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(std::string_view("hello world")), 11u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(std::string_view("0123456789")), 10u);
}

QASR_TEST(Utf8CountTwoByteSequences) {
    // "héllo" = h e(0xC3 0xA9) l l o  -> 5 codepoints, 6 bytes
    const std::string text = "h\xC3\xA9llo";
    QASR_EXPECT_EQ(text.size(), 6u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(text), 5u);
}

QASR_TEST(Utf8CountThreeByteSequence) {
    // "中" = 0xE4 0xB8 0xAD  -> 1 codepoint, 3 bytes
    const std::string text = "\xE4\xB8\xAD";
    QASR_EXPECT_EQ(text.size(), 3u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(text), 1u);
}

QASR_TEST(Utf8CountFourByteSequence) {
    // U+1F600 = 0xF0 0x9F 0x98 0x80  -> 1 codepoint, 4 bytes
    const std::string text = "\xF0\x9F\x98\x80";
    QASR_EXPECT_EQ(text.size(), 4u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(text), 1u);
}

QASR_TEST(Utf8CountMixed) {
    // "中a😀é" = 0xE4 0xB8 0xAD 'a' 0xF0 0x9F 0x98 0x80 0xC3 0xA9
    //           = 3 byte     1     4 byte        2 byte     -> 4 codepoints, 10 bytes
    const std::string text = "\xE4\xB8\xAD\x61\xF0\x9F\x98\x80\xC3\xA9";
    QASR_EXPECT_EQ(text.size(), 10u);
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(text), 4u);
}

QASR_TEST(Utf8CountIsByteForByteForPureAscii) {
    // The shared helper must agree with `text.size()` for ASCII inputs
    // (this is the property that model_bridge and realtime rely on for
    //  segment boundary decisions).
    const std::string text = "the quick brown fox jumps over the lazy dog";
    QASR_EXPECT_EQ(qasr::base::CountUtf8Codepoints(text), text.size());
}
