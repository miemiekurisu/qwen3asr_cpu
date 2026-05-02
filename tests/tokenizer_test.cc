#include "tests/test_registry.h"
#include "tests/test_paths.h"
#include "qasr/model/tokenizer.h"

// --- Normal ---

QASR_TEST(TokenizerDefaultNotLoaded) {
    qasr::Tokenizer tok;
    QASR_EXPECT(!tok.is_loaded());
    QASR_EXPECT_EQ(tok.vocab_size(), std::int32_t(0));
}

// --- Error: operations on unloaded tokenizer ---

QASR_TEST(TokenizerEncodeUnloaded) {
    qasr::Tokenizer tok;
    std::vector<std::int32_t> ids;
    qasr::Status s = tok.Encode("hello", &ids);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(TokenizerDecodeUnloaded) {
    qasr::Tokenizer tok;
    std::string text;
    qasr::Status s = tok.Decode({1, 2, 3}, &text);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(TokenizerDecodeSingleUnloaded) {
    qasr::Tokenizer tok;
    std::string piece;
    qasr::Status s = tok.DecodeSingle(0, &piece);
    QASR_EXPECT(!s.ok());
}

// --- Error: null output pointers ---

QASR_TEST(TokenizerEncodeNullOutput) {
    qasr::Tokenizer tok;
    qasr::Status s = tok.Encode("hello", nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(TokenizerDecodeNullOutput) {
    qasr::Tokenizer tok;
    qasr::Status s = tok.Decode({1}, nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(TokenizerDecodeSingleNullOutput) {
    qasr::Tokenizer tok;
    qasr::Status s = tok.DecodeSingle(0, nullptr);
    QASR_EXPECT(!s.ok());
}

// --- Error: Load from nonexistent files ---

QASR_TEST(TokenizerLoadMissingVocab) {
    qasr::Tokenizer tok;
    qasr::Status s = qasr::Tokenizer::Load(
        qasr_test::MissingTempPath(__FILE__, "qasr_no_vocab.json").string(),
        qasr_test::MissingTempPath(__FILE__, "qasr_no_merges.txt").string(),
        &tok);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(TokenizerLoadNullOut) {
    qasr::Status s = qasr::Tokenizer::Load(
        qasr_test::MissingTempPath(__FILE__, "a.json").string(),
        qasr_test::MissingTempPath(__FILE__, "b.txt").string(),
        nullptr);
    QASR_EXPECT(!s.ok());
}

// --- LoadVocabJson / LoadMergesTxt ---

QASR_TEST(LoadVocabJsonNullOutput) {
    qasr::Status s = qasr::LoadVocabJson(
        qasr_test::MissingTempPath(__FILE__, "fake.json").string(),
        nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(LoadVocabJsonMissingFile) {
    std::vector<std::string> result;
    qasr::Status s = qasr::LoadVocabJson(
        qasr_test::MissingTempPath(__FILE__, "qasr_nonexistent.json").string(),
        &result);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(LoadMergesTxtNullOutput) {
    qasr::Status s = qasr::LoadMergesTxt(
        qasr_test::MissingTempPath(__FILE__, "fake.txt").string(),
        nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(LoadMergesTxtMissingFile) {
    std::vector<std::pair<std::string, std::string>> result;
    qasr::Status s = qasr::LoadMergesTxt(
        qasr_test::MissingTempPath(__FILE__, "qasr_nonexistent.txt").string(),
        &result);
    QASR_EXPECT(!s.ok());
}

// --- EncodeUtf8 / DecodeIds with unloaded tokenizer ---

QASR_TEST(EncodeUtf8Unloaded) {
    qasr::Tokenizer tok;
    std::vector<std::int32_t> ids;
    qasr::Status s = qasr::EncodeUtf8(tok, "test", &ids);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(DecodeIdsUnloaded) {
    qasr::Tokenizer tok;
    std::string text;
    qasr::Status s = qasr::DecodeIds(tok, {1, 2}, &text);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(DecodeIdsNullOutput) {
    qasr::Tokenizer tok;
    qasr::Status s = qasr::DecodeIds(tok, {1}, nullptr);
    QASR_EXPECT(!s.ok());
}
