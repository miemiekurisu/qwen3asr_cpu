#include "tests/test_registry.h"

#include <filesystem>
#include <fstream>

#include "qasr/service/server.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeServerFixture() {
    const fs::path dir = fs::temp_directory_path() / "qasr_server_fixture";
    fs::create_directories(dir / "ui");
    std::ofstream(dir / "ui" / "index.html") << "ok";
    std::ofstream(dir / "ui" / "app.js") << "ok";
    std::ofstream(dir / "ui" / "style.css") << "ok";
    std::ofstream(dir / "config.json") << "{}";
    std::ofstream(dir / "vocab.json") << "{}";
    std::ofstream(dir / "merges.txt") << "";
    std::ofstream(dir / "model-00001-of-00002.safetensors") << "";
    return dir;
}

}  // namespace

QASR_TEST(ValidateServerConfigAcceptsFixture) {
    const fs::path dir = MakeServerFixture();
    qasr::ServerConfig config;
    config.model_dir = dir.string();
    config.ui_dir = (dir / "ui").string();
    QASR_EXPECT(qasr::ValidateServerConfig(config).ok());
}

QASR_TEST(ValidateServerConfigRejectsBadPort) {
    qasr::ServerConfig config;
    config.port = 0;
    QASR_EXPECT_EQ(qasr::ValidateServerConfig(config).code(), qasr::StatusCode::kOutOfRange);
}

QASR_TEST(ParseServerArgumentsSupportsHelp) {
    const char * argv[] = {"qasr_server", "--help"};
    qasr::ServerConfig config;
    bool show_help = false;
    const qasr::Status status = qasr::ParseServerArguments(2, argv, &config, &show_help);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(show_help);
}

QASR_TEST(ParseServerArgumentsAcceptsExplicitConfig) {
    const fs::path dir = MakeServerFixture();
    const std::string model_dir = dir.string();
    const std::string ui_dir = (dir / "ui").string();
    const char * argv[] = {
        "qasr_server",
        "--model-dir", model_dir.c_str(),
        "--ui-dir", ui_dir.c_str(),
        "--host", "0.0.0.0",
        "--port", "9090",
        "--threads", "4",
        "--verbosity", "1",
    };

    qasr::ServerConfig config;
    bool show_help = false;
    const qasr::Status status = qasr::ParseServerArguments(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &config, &show_help);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(!show_help);
    QASR_EXPECT_EQ(config.port, 9090);
    QASR_EXPECT_EQ(config.threads, 4);
    QASR_EXPECT_EQ(config.verbosity, 1);
    QASR_EXPECT_EQ(config.host, std::string("0.0.0.0"));
}

QASR_TEST(ParseBooleanTextAcceptsCommonValues) {
    bool value = false;
    QASR_EXPECT(qasr::ParseBooleanText("stream", "true", &value).ok());
    QASR_EXPECT(value);
    QASR_EXPECT(qasr::ParseBooleanText("stream", "0", &value).ok());
    QASR_EXPECT(!value);
}

QASR_TEST(ParseBooleanTextRejectsBadValue) {
    bool value = false;
    QASR_EXPECT_EQ(
        qasr::ParseBooleanText("stream", "maybe", &value).code(),
        qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(ParseTranscriptionResponseFormatSupportsVerboseJson) {
    qasr::TranscriptionResponseFormat format = qasr::TranscriptionResponseFormat::kJson;
    QASR_EXPECT(qasr::ParseTranscriptionResponseFormat("verbose_json", &format).ok());
    QASR_EXPECT_EQ(format, qasr::TranscriptionResponseFormat::kVerboseJson);
}

QASR_TEST(ValidateTimestampGranularitiesRejectsWordMode) {
    QASR_EXPECT_EQ(
        qasr::ValidateTimestampGranularities(false, true).code(),
        qasr::StatusCode::kUnimplemented);
}

QASR_TEST(ResolveServedModelIdNormalizesModelScopeName) {
    QASR_EXPECT_EQ(
        qasr::ResolveServedModelId("/tmp/Qwen3-ASR-1___7B"),
        std::string("Qwen/Qwen3-ASR-1.7B"));
}

QASR_TEST(BuildServerUsageIncludesProgramName) {
    const std::string usage = qasr::BuildServerUsage("qasr_server");
    QASR_EXPECT(usage.find("qasr_server") != std::string::npos);
    QASR_EXPECT(usage.find("--model-dir") != std::string::npos);
}

QASR_TEST(RunServerRejectsMissingModelDir) {
    qasr::ServerConfig config;
    config.port = 8080;
    QASR_EXPECT_EQ(qasr::RunServer(config), 1);
}
