#include "tests/test_registry.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "qasr/runtime/model_bridge.h"
#include "src/backend/qwen_cpu/qwen_asr_stream.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeTestDirectory(const std::string & name) {
    const fs::path dir = fs::temp_directory_path() / name;
    fs::create_directories(dir);
    return dir;
}

void WriteFile(const fs::path & path, const std::string & body) {
    std::ofstream output(path);
    output << body;
}

}  // namespace

QASR_TEST(ValidateModelDirectoryRejectsMissingShards) {
    const fs::path dir = MakeTestDirectory("qasr_model_missing_shard");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00002-of-00002.safetensors", "");
    WriteFile(dir / "model.safetensors.index.json",
        "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}");

    const qasr::Status status = qasr::ValidateModelDirectory(dir.string());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kNotFound);
    QASR_EXPECT(status.message().find("model-00001-of-00002.safetensors") != std::string::npos);
}

QASR_TEST(ValidateModelDirectoryAcceptsMinimalCompleteLayout) {
    const fs::path dir = MakeTestDirectory("qasr_model_ok");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00001-of-00002.safetensors", "");
    WriteFile(dir / "model-00002-of-00002.safetensors", "");
    WriteFile(dir / "model.safetensors.index.json",
        "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}");

    QASR_EXPECT(qasr::ValidateModelDirectory(dir.string()).ok());
}

QASR_TEST(ValidateAsrRunOptionsRejectsBrokenValues) {
    const fs::path dir = MakeTestDirectory("qasr_options");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00001-of-00002.safetensors", "");

    qasr::AsrRunOptions options;
    options.model_dir = dir.string();
    options.audio_path = (dir / "missing.wav").string();
    options.stream_max_new_tokens = 0;

    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kNotFound);

    WriteFile(dir / "sample.wav", "wav");
    options.audio_path = (dir / "sample.wav").string();
    options.stream_max_new_tokens = 1;
    options.threads = -1;
    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options.threads = 0;
    options.stream_max_new_tokens = qasr::kMaxStreamMaxNewTokens + 1;
    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options.stream_max_new_tokens = qasr::kMaxStreamMaxNewTokens;
    options.threads = 0;
    options.segment_max_codepoints = 0;
    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kInvalidArgument);

    options.segment_max_codepoints = 10;
    options.emit_tokens = true;
    options.emit_segments = true;
    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(RunAsrRejectsMissingModelDir) {
    qasr::AsrRunOptions options;
    options.model_dir = "/tmp/does-not-exist-qwen3asr-cpu";
    options.audio_path = "/tmp/does-not-exist.wav";
    options.stream_max_new_tokens = 1;
    options.segment_max_codepoints = 32;
    options.threads = 1;

    const qasr::AsrRunResult result = qasr::RunAsr(options);
    QASR_EXPECT_EQ(result.status.code(), qasr::StatusCode::kNotFound);
}

QASR_TEST(CpuBackendAvailabilityIsConsistent) {
#ifdef QASR_CPU_BACKEND_ENABLED
    QASR_EXPECT(qasr::CpuBackendAvailable());
#else
    QASR_EXPECT(!qasr::CpuBackendAvailable());
#endif
}

QASR_TEST(ShouldFlushAsrSegmentUsesPunctuationAndLength) {
    QASR_EXPECT(!qasr::ShouldFlushAsrSegment("", 4));
    QASR_EXPECT(!qasr::ShouldFlushAsrSegment("abc", 4));
    QASR_EXPECT(qasr::ShouldFlushAsrSegment("abc.", 40));
    QASR_EXPECT(qasr::ShouldFlushAsrSegment("\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82", 40));
    QASR_EXPECT(qasr::ShouldFlushAsrSegment("abcd", 4));
    QASR_EXPECT(!qasr::ShouldFlushAsrSegment("abcd", 0));
}

QASR_TEST(StreamDuplicatePrefixSkipsRecentCommittedText) {
    const std::array<int, 8> emitted{11, 12, 13, 21, 22, 23, 24, 31};
    const std::array<int, 7> candidate{90, 91, 21, 22, 23, 24, 40};

    const int next = qwen_stream_skip_recent_duplicate_prefix(
        emitted.data(),
        static_cast<int>(emitted.size()),
        candidate.data(),
        2,
        static_cast<int>(candidate.size()),
        3,
        8,
        16);

    QASR_EXPECT_EQ(next, 6);
}

QASR_TEST(StreamDuplicatePrefixHandlesInvalidAndShortValues) {
    const std::array<int, 4> emitted{1, 2, 3, 4};
    const std::array<int, 4> candidate{2, 3, 9, 10};

    QASR_EXPECT_EQ(qwen_stream_skip_recent_duplicate_prefix(
                       nullptr, 4, candidate.data(), 0, 4, 2, 4, 16),
                   0);
    QASR_EXPECT_EQ(qwen_stream_skip_recent_duplicate_prefix(
                       emitted.data(), 4, nullptr, 0, 4, 2, 4, 16),
                   0);
    QASR_EXPECT_EQ(qwen_stream_skip_recent_duplicate_prefix(
                       emitted.data(), 4, candidate.data(), -1, 4, 2, 4, 16),
                   -1);
    QASR_EXPECT_EQ(qwen_stream_skip_recent_duplicate_prefix(
                       emitted.data(), 4, candidate.data(), 0, 4, 3, 2, 16),
                   0);
    QASR_EXPECT_EQ(qwen_stream_skip_recent_duplicate_prefix(
                       emitted.data(), 4, candidate.data(), 0, 1, 2, 4, 16),
                   0);
}

QASR_TEST(StreamDuplicatePrefixSkipsRandomEmbeddedSpan) {
    std::mt19937 rng(20260410U);
    std::uniform_int_distribution<int> value_dist(1000, 9000);
    std::vector<int> emitted(160);
    for (int & value : emitted) {
        value = value_dist(rng);
    }

    const int span_start = 91;
    const int span_len = 17;
    std::vector<int> candidate{7, 8, 9};
    candidate.insert(
        candidate.end(),
        emitted.begin() + span_start,
        emitted.begin() + span_start + span_len);
    candidate.push_back(42);

    const int next = qwen_stream_skip_recent_duplicate_prefix(
        emitted.data(),
        static_cast<int>(emitted.size()),
        candidate.data(),
        3,
        static_cast<int>(candidate.size()),
        4,
        32,
        96);

    QASR_EXPECT_EQ(next, 3 + span_len);
}
