#include "tests/test_registry.h"
#include "tests/test_paths.h"

#include <filesystem>
#include <fstream>

#include "qasr/base/json.h"
#include "qasr/service/server.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeServerFixture() {
    const fs::path dir = qasr_test::FreshTempDir(__FILE__, "qasr_server_fixture");
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
        qasr::ResolveServedModelId(qasr_test::TempPath(__FILE__, "Qwen3-ASR-1___7B").string()),
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

    policy.min_decode_interval_ms = 0;
    QASR_EXPECT_EQ(qasr::RealtimeStreamMaxNewTokens(policy), 24);
}

/* ────── HTTP handler helpers (extracted from RunServer) ────── */

QASR_TEST(BuildHealthJsonReturnsOkStatus) {
    /* The current contract: /health and /api/health both return
     * {"status":"ok"} regardless of model state.  This is the
     * "liveness" probe, not the "readiness" probe.  A future
     * readiness endpoint would query the SharedAsrModel and the
     * realtime worker.  Until then, this is the contract. */
    const std::string body = qasr::BuildHealthJson();
    QASR_EXPECT_EQ(body, std::string("{\"status\":\"ok\"}"));
}

QASR_TEST(BuildHealthJsonIsValidJson) {
    /* Just parsing it to catch accidental syntax breakage in a
     * future "add more fields" commit. */
    const std::string body = qasr::BuildHealthJson();
    qasr::Json parsed = qasr::Json::parse(body);
    QASR_EXPECT_EQ(parsed["status"].get<std::string>(), std::string("ok"));
}

QASR_TEST(ServeStaticTextFileLoadsExistingFile) {
    /* Write a temp file, point the helper at it, verify the
     * response body matches and the content type is set. */
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "qasr_static_test.txt";
    {
        std::ofstream out(tmp);
        out << "hello world\n";
    }
    qasr::HttpResponse response;
    qasr::ServeStaticTextFile(response, tmp, "text/plain; charset=utf-8", "test.txt");
    QASR_EXPECT_EQ(response.status, 200);
    QASR_EXPECT_EQ(response.body_, std::string("hello world\n"));
    QASR_EXPECT_EQ(response.content_type_, std::string("text/plain; charset=utf-8"));
    fs::remove(tmp);
}

QASR_TEST(ServeStaticTextFileReportsMissingFileAsError) {
    /* When the file is missing or empty, the helper must set a
     * 500 with an error message that names the file.  This is
     * what the operator sees in the browser when ui/index.html
     * is missing. */
    qasr::HttpResponse response;
    qasr::ServeStaticTextFile(
        response,
        std::filesystem::path("/nonexistent/path/missing.html"),
        "text/html; charset=utf-8",
        "missing.html");
    QASR_EXPECT_EQ(response.status, 500);
    QASR_EXPECT(response.body_.find("missing.html") != std::string::npos);
}

QASR_TEST(ServeStaticTextFileReportsEmptyFileAsError) {
    /* Zero-byte file == LoadTextFile returns empty == error path.
     * Same 500 + named label. */
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "qasr_empty_test.txt";
    {
        std::ofstream out(tmp);
        /* write nothing */
    }
    qasr::HttpResponse response;
    qasr::ServeStaticTextFile(response, tmp, "text/plain", "empty.txt");
    QASR_EXPECT_EQ(response.status, 500);
    QASR_EXPECT(response.body_.find("empty.txt") != std::string::npos);
    fs::remove(tmp);
}

QASR_TEST(ServeStaticTextFileHandlesAllUiAssets) {
    /* Walk the real ui/ directory and verify each static asset
     * loads with its expected content type.  This catches the
     * case where a script is renamed but the route still points
     * at the old name (the bug class that produced the
     * "404 on /app.js" 90s-after-deploy outage). */
    namespace fs = std::filesystem;
    const fs::path ui = fs::path(QASR_TEST_SOURCE_DIR) / "ui";
    struct Asset { const char * path; const char * mime; };
    const Asset assets[] = {
        {"index.html", "text/html; charset=utf-8"},
        {"app.js", "application/javascript; charset=utf-8"},
        {"live_monitor.js", "application/javascript; charset=utf-8"},
        {"state_pure.js", "application/javascript; charset=utf-8"},
        {"style.css", "text/css; charset=utf-8"},
    };
    for (const Asset & a : assets) {
        qasr::HttpResponse response;
        qasr::ServeStaticTextFile(response, ui / a.path, a.mime, a.path);
        QASR_EXPECT_EQ(response.status, 200);
        QASR_EXPECT(!response.body_.empty());
        QASR_EXPECT_EQ(response.content_type_, std::string(a.mime));
    }
}

QASR_TEST(BuildServerUsageMentionsRealtimeModelDir) {
    /* Sanity: the usage string documents the new --realtime-model-dir
     * flag.  If a future refactor drops the line, the user is
     * surprised when their config no longer works. */
    const std::string usage = qasr::BuildServerUsage("qasr_server");
    QASR_EXPECT(usage.find("--realtime-model-dir") != std::string::npos);
}
