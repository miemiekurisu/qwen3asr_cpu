#pragma once

#include <filesystem>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/base/http_server.h"
#include "qasr/core/status.h"
#include "qasr/service/realtime.h"

namespace qasr {

enum class TranscriptionResponseFormat {
    kJson = 0,
    kText,
    kVerboseJson,
};

enum class OpenAiRealtimeAction {
    kSessionCreate = 0,
    kInputAudioBufferAppend,
    kInputAudioBufferCommit,
};

struct OpenAiRealtimeRequest {
    OpenAiRealtimeAction action = OpenAiRealtimeAction::kSessionCreate;
    std::string session_id;
    std::string model;
    std::string language;
    std::string input_audio_format = "pcm16le";
    std::string audio;
    bool stream = true;
};

struct ServerConfig {
    std::string model_dir;
    /* Model used by the realtime / host-capture worker.  Empty means
     * "same as model_dir" — the SharedAsrModel instance is then shared
     * between batch and realtime paths, so loading the same model
     * twice is avoided.  When set to a different path, a second
     * SharedAsrModel is loaded for realtime (typical use: 0.6B for
     * realtime, 1.7B for batch).  This is the per-feature model
     * override the operator asked for: batch and realtime can use
     * the same model (default) or independent models for quality
     * / memory / latency trade-offs. */
    std::string realtime_model_dir;
    std::string host = "127.0.0.1";
    std::string ui_dir = "ui";
    std::int32_t port = 8080;
    std::int32_t threads = 0;
    std::int32_t verbosity = 0;
    float temperature = -1.0f;
    bool decoder_int8 = false;
    bool encoder_int8 = false;
    /* Realtime/host-capture sessions are cloned via qwen_clone_shared() and
     * INT8 must be re-applied on the clone. Decoder INT8 noticeably degrades
     * the autoregressive Qwen3 LM (language consistency, code-switch leakage,
     * hallucinations on low-confidence audio), so by default the realtime
     * clone keeps the decoder on BF16 even when --decoder-int8 is set for
     * batch transcription. Set this flag to opt back in. Encoder INT8 is
     * always propagated to the realtime clone since its quality impact is
     * minimal. */
    bool realtime_decoder_int8 = false;
};

Status ParseBooleanText(std::string_view field_name, std::string_view text, bool * value);
Status ParseTranscriptionResponseFormat(
    std::string_view text,
    TranscriptionResponseFormat * format);
Status ValidateTimestampGranularities(bool want_segment_timestamps, bool want_word_timestamps);
std::string ResolveServedModelId(std::string_view model_dir);
bool IsTerminalJobState(std::string_view state) noexcept;
bool ShouldEvictCompletedJob(
    std::string_view state,
    std::int64_t updated_at_seconds,
    std::int64_t now_seconds,
    std::int64_t ttl_seconds) noexcept;
Status ParseOpenAiRealtimeRequest(std::string_view body, OpenAiRealtimeRequest * request);
Status DecodeBase64Pcm16Le(std::string_view encoded, std::vector<float> * samples);
float RealtimeStreamChunkSeconds(const RealtimePolicyConfig & policy) noexcept;
int RealtimeStreamMaxNewTokens(const RealtimePolicyConfig & policy) noexcept;

Status ValidateServerConfig(const ServerConfig & config);
Status ParseServerArguments(int argc, const char * const argv[], ServerConfig * config, bool * show_help);
std::string BuildServerUsage(std::string_view program_name);
int RunServer(const ServerConfig & config);

/* ────── HTTP handler helpers (extracted for testability) ──────
 *
 * These were previously inline lambdas inside RunServer.  They
 * were pulled out so unit tests can drive them without spinning
 * up the full server (which requires a real model on disk).  The
 * signatures are designed to be mockable: ServeStaticTextFile
 * takes the file path and content type as parameters, never
 * reaches into module-level state. */

namespace fs = std::filesystem;

/// Serve a static text file from \p path.  On success sets the
/// response body and the given MIME content type.  On failure
/// (empty file, unreadable) sends a 500 with an error message
/// identifying \p label for operator diagnostics.
void ServeStaticTextFile(
    HttpResponse & response,
    const fs::path & path,
    const std::string & content_type,
    const std::string & label);

/// Build the JSON body returned by /health and /api/health.  The
/// current contract is a constant {"status": "ok"}, but pulling
/// this out as a free function lets a future test (or a future
/// "deep health" endpoint) replace it without touching the
/// route-registration code.
std::string BuildHealthJson();

}  // namespace qasr
