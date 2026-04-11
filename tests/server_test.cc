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

QASR_TEST(IsTerminalJobStateRecognizesTerminalStates) {
    QASR_EXPECT(qasr::IsTerminalJobState("completed"));
    QASR_EXPECT(qasr::IsTerminalJobState("failed"));
    QASR_EXPECT(qasr::IsTerminalJobState("cancelled"));
    QASR_EXPECT(!qasr::IsTerminalJobState("queued"));
    QASR_EXPECT(!qasr::IsTerminalJobState("running"));
    QASR_EXPECT(!qasr::IsTerminalJobState("cancelling"));
}

QASR_TEST(ShouldEvictCompletedJobMatchesTtlRules) {
    QASR_EXPECT(qasr::ShouldEvictCompletedJob("completed", 100, 3700, 3600));
    QASR_EXPECT(qasr::ShouldEvictCompletedJob("failed", 100, 3700, 3600));
    QASR_EXPECT(qasr::ShouldEvictCompletedJob("cancelled", 100, 3700, 3600));
    QASR_EXPECT(!qasr::ShouldEvictCompletedJob("running", 100, 3700, 3600));
    QASR_EXPECT(!qasr::ShouldEvictCompletedJob("completed", 100, 3699, 3600));
    QASR_EXPECT(!qasr::ShouldEvictCompletedJob("completed", 500, 400, 3600));
    QASR_EXPECT(!qasr::ShouldEvictCompletedJob("completed", 100, 3700, 0));
}

QASR_TEST(ParseOpenAiRealtimeRequestDefaultsToSessionCreate) {
    qasr::OpenAiRealtimeRequest request;
    QASR_EXPECT(qasr::ParseOpenAiRealtimeRequest("{}", &request).ok());
    QASR_EXPECT_EQ(request.action, qasr::OpenAiRealtimeAction::kSessionCreate);
    QASR_EXPECT(request.stream);
    QASR_EXPECT_EQ(request.input_audio_format, std::string("pcm16le"));
}

QASR_TEST(ParseOpenAiRealtimeRequestAcceptsNestedSessionFields) {
    qasr::OpenAiRealtimeRequest request;
    const char * body =
        "{\"type\":\"input_audio_buffer.append\",\"session\":{\"id\":\"sess-1\",\"model\":\"Qwen/Qwen3-ASR-1.7B\",\"language\":\"zh\",\"input_audio_format\":\"pcm16\"},\"audio\":\"AIAAAP9/\"}";
    QASR_EXPECT(qasr::ParseOpenAiRealtimeRequest(body, &request).ok());
    QASR_EXPECT_EQ(request.action, qasr::OpenAiRealtimeAction::kInputAudioBufferAppend);
    QASR_EXPECT_EQ(request.session_id, std::string("sess-1"));
    QASR_EXPECT_EQ(request.model, std::string("Qwen/Qwen3-ASR-1.7B"));
    QASR_EXPECT_EQ(request.language, std::string("zh"));
    QASR_EXPECT_EQ(request.input_audio_format, std::string("pcm16le"));
    QASR_EXPECT_EQ(request.audio, std::string("AIAAAP9/"));
}

QASR_TEST(ParseOpenAiRealtimeRequestRejectsMissingAppendAudio) {
    qasr::OpenAiRealtimeRequest request;
    QASR_EXPECT_EQ(
        qasr::ParseOpenAiRealtimeRequest(
            "{\"type\":\"input_audio_buffer.append\",\"session_id\":\"sess-1\"}",
            &request).code(),
        qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(DecodeBase64Pcm16LeConvertsSamples) {
    std::vector<float> samples;
    QASR_EXPECT(qasr::DecodeBase64Pcm16Le("AIAAAP9/", &samples).ok());
    QASR_EXPECT_EQ(samples.size(), std::size_t{3});
    QASR_EXPECT(samples[0] < -0.99f);
    QASR_EXPECT(samples[1] == 0.0f);
    QASR_EXPECT(samples[2] > 0.99f);
}

QASR_TEST(DecodeBase64Pcm16LeRejectsOddByteLength) {
    std::vector<float> samples;
    QASR_EXPECT_EQ(
        qasr::DecodeBase64Pcm16Le("AA==", &samples).code(),
        qasr::StatusCode::kInvalidArgument);
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

QASR_TEST(RealtimeStreamChunkSecondsClampsToReasonableRange) {
    qasr::RealtimePolicyConfig policy;
    policy.min_decode_interval_ms = 200;
    QASR_EXPECT_EQ(qasr::RealtimeStreamChunkSeconds(policy), 0.4f);

    policy.min_decode_interval_ms = 800;
    QASR_EXPECT_EQ(qasr::RealtimeStreamChunkSeconds(policy), 0.8f);

    policy.min_decode_interval_ms = 1600;
    QASR_EXPECT_EQ(qasr::RealtimeStreamChunkSeconds(policy), 1.0f);
}

QASR_TEST(RealtimeStreamMaxNewTokensTracksChunkCadence) {
    qasr::RealtimePolicyConfig policy;
    policy.min_decode_interval_ms = 600;
    QASR_EXPECT_EQ(qasr::RealtimeStreamMaxNewTokens(policy), 24);

    policy.min_decode_interval_ms = 950;
    QASR_EXPECT_EQ(qasr::RealtimeStreamMaxNewTokens(policy), 32);
}
