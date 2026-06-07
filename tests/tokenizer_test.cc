#include "tests/test_registry.h"
#include "tests/test_paths.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qasr/model/tokenizer.h"

namespace {

std::string WriteVocabFile(const std::string & name, const std::string & body) {
    const std::filesystem::path temp = qasr_test::TempPath(__FILE__, "vocab_" + name);
    std::ofstream output(temp);
    output << body;
    output.close();
    return temp.string();
}

}  // namespace

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

QASR_TEST(LoadVocabJsonBasic) {
    const std::string body = R"({"hello":0,"world":1,"<|endoftext|>":2})";
    const std::string path = WriteVocabFile("basic.json", body);
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(result.size(), std::size_t{3});
    QASR_EXPECT_EQ(result[0], std::string{"hello"});
    QASR_EXPECT_EQ(result[1], std::string{"world"});
    QASR_EXPECT_EQ(result[2], std::string{"<|endoftext|>"});
}

QASR_TEST(LoadVocabJsonSparseIds) {
    // Non-contiguous ids (gaps) — the parser must size the result to
    // max_id+1 and leave empty entries for the gaps.
    const std::string body = R"({"a":0,"b":5,"c":100})";
    const std::string path = WriteVocabFile("sparse.json", body);
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(result.size(), std::size_t{101});
    QASR_EXPECT_EQ(result[0], std::string{"a"});
    QASR_EXPECT_EQ(result[5], std::string{"b"});
    QASR_EXPECT_EQ(result[100], std::string{"c"});
}

QASR_TEST(LoadVocabJsonWithWhitespaceVariations) {
    // Tabs and multiple spaces around the colon must parse the same
    // way as a single space.
    const std::string body = "{\"a\"\t:\t1, \"b\"   :   2}";
    const std::string path = WriteVocabFile("ws.json", body);
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(result.size(), std::size_t{3});
    QASR_EXPECT_EQ(result[1], std::string{"a"});
    QASR_EXPECT_EQ(result[2], std::string{"b"});
}

QASR_TEST(LoadVocabJsonWithEscapedQuote) {
    // A token containing an escaped double-quote must be parsed
    // correctly.  The token text in JSON is `a"b` (literal quote in
    // the middle).
    const std::string body = R"({"a\"b":7})";
    const std::string path = WriteVocabFile("escape.json", body);
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(result.size(), std::size_t{8});
    QASR_EXPECT_EQ(result[7], std::string{"a\\\"b"});
}

QASR_TEST(LoadVocabJsonEmptyFile) {
    const std::string path = WriteVocabFile("empty.json", "");
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(LoadVocabJsonMalformedNoId) {
    // Token with no numeric id following the colon — must skip the
    // entry, not crash.
    const std::string body = R"({"hello":})";
    const std::string path = WriteVocabFile("noid.json", body);
    std::vector<std::string> result;
    const qasr::Status s = qasr::LoadVocabJson(path, &result);
    // Empty result → max_id < 0 → kInvalidArgument.
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
