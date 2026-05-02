#include "tests/test_registry.h"
#include "tests/test_paths.h"

#include <filesystem>
#include <fstream>

#include "qasr/cli/options.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeCliFixtureDirectory() {
    const fs::path dir = qasr_test::FreshTempDir(__FILE__, "qasr_cli_fixture");
    std::ofstream(dir / "config.json") << "{}";
    std::ofstream(dir / "vocab.json") << "{}";
    std::ofstream(dir / "merges.txt") << "";
    std::ofstream(dir / "model-00001-of-00002.safetensors") << "";
    std::ofstream(dir / "sample.wav") << "wav";
    return dir;
}

}  // namespace

QASR_TEST(ParseCliArgumentsAcceptsMinimalInvocation) {
    const fs::path dir = MakeCliFixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string audio_path = (dir / "sample.wav").string();
    const char * argv[] = {
        "qasr_cli",
        "--model-dir", model_dir.c_str(),
        "--audio", audio_path.c_str(),
        "--threads", "2",
        "--stream-max-new-tokens", "16",
        "--verbosity", "1",
        "--stream",
        "--emit-segments",
        "--segment-max-codepoints", "24",
    };

    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &options);
    QASR_EXPECT(status.ok());
    QASR_EXPECT_EQ(options.asr.threads, 2);
    QASR_EXPECT_EQ(options.asr.stream_max_new_tokens, 16);
    QASR_EXPECT_EQ(options.asr.verbosity, 1);
    QASR_EXPECT_EQ(options.asr.segment_max_codepoints, 24);
    QASR_EXPECT(options.asr.stream);
    QASR_EXPECT(options.asr.emit_segments);
}

QASR_TEST(ParseCliArgumentsRejectsUnknownFlags) {
    const char * argv[] = {"qasr_cli", "--bad"};
    qasr::CliOptions options;
    QASR_EXPECT_EQ(qasr::ParseCliArguments(2, argv, &options).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(ParseCliArgumentsSupportsHelpWithoutPaths) {
    const char * argv[] = {"qasr_cli", "--help"};
    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(2, argv, &options);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(options.show_help);
}

QASR_TEST(BuildCliUsageMentionsRequiredFlags) {
    const std::string usage = qasr::BuildCliUsage("qasr_cli");
    QASR_EXPECT(usage.find("--model-dir") != std::string::npos);
    QASR_EXPECT(usage.find("--audio") != std::string::npos);
    QASR_EXPECT(usage.find("--emit-segments") != std::string::npos);
    QASR_EXPECT(usage.find("max 128") != std::string::npos);
}

QASR_TEST(ParseCliArgumentsAcceptsAlignFlags) {
    const fs::path dir = MakeCliFixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string audio_path = (dir / "sample.wav").string();
    const char * argv[] = {
        "qasr_cli",
        "--model-dir", model_dir.c_str(),
        "--audio", audio_path.c_str(),
        "--align",
        "--aligner-model-dir", "/some/aligner/path",
    };

    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &options);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(options.align);
    QASR_EXPECT_EQ(options.aligner_model_dir, std::string("/some/aligner/path"));
}

QASR_TEST(ParseCliArgumentsAlignRequiresModelDir) {
    const fs::path dir = MakeCliFixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string audio_path = (dir / "sample.wav").string();
    const char * argv[] = {
        "qasr_cli",
        "--model-dir", model_dir.c_str(),
        "--audio", audio_path.c_str(),
        "--align",
    };

    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &options);
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(BuildCliUsageMentionsAlignFlags) {
    const std::string usage = qasr::BuildCliUsage("qasr_cli");
    QASR_EXPECT(usage.find("--align") != std::string::npos);
    QASR_EXPECT(usage.find("--aligner-model-dir") != std::string::npos);
}
