#include "tests/test_registry.h"

#include <cstdint>
#include <string>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/inference/aligner_types.h"

// ========================================================================
// IsAlignerLanguageSupported
// ========================================================================

QASR_TEST(AlignerLanguage_Chinese) {
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("chinese"));
}

QASR_TEST(AlignerLanguage_English) {
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("english"));
}

QASR_TEST(AlignerLanguage_Japanese) {
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("japanese"));
}

QASR_TEST(AlignerLanguage_Korean) {
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("korean"));
}

QASR_TEST(AlignerLanguage_CaseInsensitive) {
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("Chinese"));
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("ENGLISH"));
    QASR_EXPECT(qasr::IsAlignerLanguageSupported("Japanese"));
}

QASR_TEST(AlignerLanguage_Unsupported) {
    QASR_EXPECT(!qasr::IsAlignerLanguageSupported("hindi"));
    QASR_EXPECT(!qasr::IsAlignerLanguageSupported("arabic"));
    QASR_EXPECT(!qasr::IsAlignerLanguageSupported(""));
}

QASR_TEST(AlignerLanguage_AllSupported) {
    const char * langs[] = {
        "chinese", "cantonese", "english", "german", "spanish",
        "french", "italian", "portuguese", "russian", "korean", "japanese",
    };
    for (const char * lang : langs) {
        QASR_EXPECT(qasr::IsAlignerLanguageSupported(lang));
    }
}

// ========================================================================
// ValidateAlignResult
// ========================================================================

QASR_TEST(ValidateAlignResult_Empty) {
    qasr::AlignResult result;
    QASR_EXPECT_EQ(qasr::ValidateAlignResult(result).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(ValidateAlignResult_Valid) {
    qasr::AlignResult result;
    result.words.push_back({"hello", 0.0, 0.5});
    result.words.push_back({"world", 0.6, 1.0});
    QASR_EXPECT(qasr::ValidateAlignResult(result).ok());
}

QASR_TEST(ValidateAlignResult_NegativeStart) {
    qasr::AlignResult result;
    result.words.push_back({"hello", -0.1, 0.5});
    QASR_EXPECT_EQ(qasr::ValidateAlignResult(result).code(), qasr::StatusCode::kOutOfRange);
}

QASR_TEST(ValidateAlignResult_NegativeEnd) {
    qasr::AlignResult result;
    result.words.push_back({"hello", 0.0, -0.1});
    QASR_EXPECT_EQ(qasr::ValidateAlignResult(result).code(), qasr::StatusCode::kOutOfRange);
}

QASR_TEST(ValidateAlignResult_EndBeforeStart) {
    qasr::AlignResult result;
    result.words.push_back({"hello", 1.0, 0.5});
    QASR_EXPECT_EQ(qasr::ValidateAlignResult(result).code(), qasr::StatusCode::kOutOfRange);
}

QASR_TEST(ValidateAlignResult_ZeroDuration) {
    qasr::AlignResult result;
    result.words.push_back({"hello", 1.0, 1.0});
    QASR_EXPECT(qasr::ValidateAlignResult(result).ok());
}

// ========================================================================
// WordsToSegments
// ========================================================================

QASR_TEST(WordsToSegments_Empty) {
    std::vector<qasr::AlignedWord> words;
    auto segs = qasr::WordsToSegments(words);
    QASR_EXPECT(segs.empty());
}

QASR_TEST(WordsToSegments_SingleWord) {
    std::vector<qasr::AlignedWord> words = {{"hello", 0.0, 0.5}};
    auto segs = qasr::WordsToSegments(words);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    QASR_EXPECT_EQ(segs[0].text, std::string("hello"));
    QASR_EXPECT_EQ(segs[0].range.begin_ms, std::int64_t(0));
    QASR_EXPECT_EQ(segs[0].range.end_ms, std::int64_t(500));
}

QASR_TEST(WordsToSegments_MultipleWords) {
    std::vector<qasr::AlignedWord> words = {
        {"hello", 0.0, 0.3},
        {"world", 0.35, 0.7},
    };
    auto segs = qasr::WordsToSegments(words, 42, 1.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    QASR_EXPECT_EQ(segs[0].text, std::string("hello world"));
    QASR_EXPECT_EQ(segs[0].range.begin_ms, std::int64_t(0));
    QASR_EXPECT_EQ(segs[0].range.end_ms, std::int64_t(700));
}

QASR_TEST(WordsToSegments_SplitOnGap) {
    std::vector<qasr::AlignedWord> words = {
        {"hello", 0.0, 0.3},
        {"world", 2.0, 2.5},  // gap > 1.0
    };
    auto segs = qasr::WordsToSegments(words, 42, 1.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(2));
    QASR_EXPECT_EQ(segs[0].text, std::string("hello"));
    QASR_EXPECT_EQ(segs[1].text, std::string("world"));
}

QASR_TEST(WordsToSegments_SplitOnWidth) {
    // With max_segment_chars=10, "hello world" (11 chars) should split
    std::vector<qasr::AlignedWord> words = {
        {"hello", 0.0, 0.3},
        {"world", 0.35, 0.7},
    };
    auto segs = qasr::WordsToSegments(words, 10, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(2));
}

QASR_TEST(WordsToSegments_SplitOnPunctuation) {
    std::vector<qasr::AlignedWord> words = {
        {"hello.", 0.0, 0.3},
        {"world", 0.35, 0.7},
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(2));
    QASR_EXPECT_EQ(segs[0].text, std::string("hello."));
    QASR_EXPECT_EQ(segs[1].text, std::string("world"));
}

QASR_TEST(WordsToSegments_CjkNoSpace) {
    // CJK characters should not get spaces between them
    // 你 = U+4F60, 好 = U+597D
    std::vector<qasr::AlignedWord> words = {
        {"\xe4\xbd\xa0", 0.0, 0.3},   // 你
        {"\xe5\xa5\xbd", 0.35, 0.7},   // 好
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    // Should be "你好" without space
    QASR_EXPECT_EQ(segs[0].text, std::string("\xe4\xbd\xa0\xe5\xa5\xbd"));
}

QASR_TEST(WordsToSegments_Timestamps) {
    std::vector<qasr::AlignedWord> words = {
        {"a", 1.234, 1.567},
        {"b", 1.6, 2.0},
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    QASR_EXPECT_EQ(segs[0].range.begin_ms, std::int64_t(1234));
    QASR_EXPECT_EQ(segs[0].range.end_ms, std::int64_t(2000));
}

// Commas must NOT cause segment splits (fragments too short for translation).
QASR_TEST(WordsToSegments_CommaNoSplit) {
    std::vector<qasr::AlignedWord> words = {
        {"hello,", 0.0, 0.3},
        {"world", 0.35, 0.7},
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    QASR_EXPECT_EQ(segs[0].text, std::string("hello, world"));
}

// Full-width comma also must not split.
QASR_TEST(WordsToSegments_FullWidthCommaNoSplit) {
    std::vector<qasr::AlignedWord> words = {
        {"\xe4\xbd\xa0\xef\xbc\x8c", 0.0, 0.3},  // 你，
        {"\xe5\xa5\xbd", 0.35, 0.7},               // 好
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
}

// Mixed CJK + English: space between English-English, no space CJK-CJK.
QASR_TEST(WordsToSegments_MixedCjkEnglish) {
    std::vector<qasr::AlignedWord> words = {
        {"\xe4\xbd\xa0", 0.0, 0.2},   // 你
        {"hello", 0.25, 0.5},          // English after CJK
        {"world", 0.55, 0.8},          // English after English
    };
    auto segs = qasr::WordsToSegments(words, 42, 10.0);
    QASR_EXPECT_EQ(segs.size(), std::size_t(1));
    // CJK followed by English: no space (CJK last char is CJK → no space added)
    // But English followed by English: space.
    // The actual text depends on whether we consider CJK→English transition.
    // Current logic: prev_cjk=true, curr_cjk=false → no space.
    // Then "hello"→"world": prev_cjk=false, curr_cjk=false → space.
    QASR_EXPECT_EQ(segs[0].text, std::string("\xe4\xbd\xa0hello world"));
}

// ========================================================================
// kAlignerMaxChunkSec
// ========================================================================

QASR_TEST(AlignerMaxChunkSec_Value) {
    const double v = qasr::kAlignerMaxChunkSec;
    QASR_EXPECT(v > 0.0);
    QASR_EXPECT(v <= 400.0);
}
