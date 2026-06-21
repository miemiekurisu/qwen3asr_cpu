#include "qasr/service/server.h"
#include "qasr/service/realtime.h"
#include "qasr/audio/audio_convert.h"

#include <atomic>
#include <cmath>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qasr/base/json.h"
#include "qasr/protocol/openai.h"

#if !defined(_WIN32)
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef QASR_CURL_AVAILABLE
#include <curl/curl.h>
#endif

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_audio.h"
#include "qwen_silero_vad.h"
}
#include "qasr/base/http_server.h"
#include "qasr/base/process_spawn.h"
#endif

#include "qasr/runtime/model_bridge.h"
#include "qasr/engine/asr_engine.h"
#include "qasr/engine/cpu_asr_engine.h"
#include "qasr/scheduler/scheduler.h"

namespace qasr {

Status ParseBooleanText(std::string_view field_name, std::string_view text, bool * value) {
    if (value == nullptr) {
        return Status(StatusCode::kInvalidArgument, "value output must not be null");
    }

    std::string normalized;
    normalized.reserve(text.size());
    for (const char ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            normalized.push_back(ch);
        }
    }

    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        *value = true;
        return OkStatus();
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        *value = false;
        return OkStatus();
    }
    return Status(StatusCode::kInvalidArgument, std::string(field_name) + " must be a boolean");
}

Status ParseTranscriptionResponseFormat(
    std::string_view text,
    TranscriptionResponseFormat * format) {
    if (format == nullptr) {
        return Status(StatusCode::kInvalidArgument, "format output must not be null");
    }

    std::string normalized;
    normalized.reserve(text.size());
    for (const char ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            normalized.push_back(ch);
        }
    }

    if (normalized.empty() || normalized == "json") {
        *format = TranscriptionResponseFormat::kJson;
        return OkStatus();
    }
    if (normalized == "text") {
        *format = TranscriptionResponseFormat::kText;
        return OkStatus();
    }
    if (normalized == "verbose_json") {
        *format = TranscriptionResponseFormat::kVerboseJson;
        return OkStatus();
    }
    return Status(StatusCode::kInvalidArgument, "unsupported response_format: " + std::string(text));
}

Status ValidateTimestampGranularities(bool want_segment_timestamps, bool want_word_timestamps) {
    if (!want_segment_timestamps && !want_word_timestamps) {
        return OkStatus();
    }
    if (want_word_timestamps) {
        return Status(
            StatusCode::kUnimplemented,
            "word timestamps require the forced aligner model and are not available in the current CPU bridge");
    }
    return OkStatus();
}

std::string ResolveServedModelId(std::string_view model_dir) {
    const std::filesystem::path path(model_dir);
    std::string base = path.filename().string();
    if (base.empty()) {
        return "Qwen/Qwen3-ASR";
    }

    std::string normalized;
    normalized.reserve(base.size());
    for (std::size_t index = 0; index < base.size();) {
        if (index + 2 < base.size() &&
            base[index] == '_' &&
            base[index + 1] == '_' &&
            base[index + 2] == '_') {
            normalized.push_back('.');
            index += 3;
            continue;
        }
        normalized.push_back(base[index]);
        ++index;
    }

    if (normalized.rfind("Qwen/", 0) == 0) {
        return normalized;
    }
    if (normalized.rfind("Qwen3-", 0) == 0) {
        return "Qwen/" + normalized;
    }
    return normalized;
}

bool IsTerminalJobState(std::string_view state) noexcept {
    return state == "completed" || state == "failed" || state == "cancelled";
}

bool ShouldEvictCompletedJob(
    std::string_view state,
    std::int64_t updated_at_seconds,
    std::int64_t now_seconds,
    std::int64_t ttl_seconds) noexcept {
    if (!IsTerminalJobState(state) || ttl_seconds <= 0 || now_seconds < updated_at_seconds) {
        return false;
    }
    return now_seconds - updated_at_seconds >= ttl_seconds;
}

namespace {

std::string NormalizeAsciiLower(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (const char ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            normalized.push_back(ch);
        }
    }
    return normalized;
}

bool IsAsciiWhitespace(char ch) noexcept {
    return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
}

int DecodeBase64Value(char ch) noexcept {
    if (ch >= 'A' && ch <= 'Z') {
        return ch - 'A';
    }
    if (ch >= 'a' && ch <= 'z') {
        return ch - 'a' + 26;
    }
    if (ch >= '0' && ch <= '9') {
        return ch - '0' + 52;
    }
    if (ch == '+') {
        return 62;
    }
    if (ch == '/') {
        return 63;
    }
    return -1;
}

}  // namespace

Status ParseOpenAiRealtimeRequest(std::string_view body, OpenAiRealtimeRequest * request) {
    if (request == nullptr) {
        return Status(StatusCode::kInvalidArgument, "request output must not be null");
    }

    Json json_body = body.empty() ? Json::object() : Json::parse(std::string(body));
    if (json_body.is_discarded() || !json_body.is_object()) {
        return Status(StatusCode::kInvalidArgument, "request body must be a JSON object");
    }

    *request = OpenAiRealtimeRequest{};

    const std::string type = NormalizeAsciiLower(json_body.value("type", std::string("session.create")));
    if (type == "session.create") {
        request->action = OpenAiRealtimeAction::kSessionCreate;
    } else if (type == "input_audio_buffer.append") {
        request->action = OpenAiRealtimeAction::kInputAudioBufferAppend;
    } else if (type == "input_audio_buffer.commit") {
        request->action = OpenAiRealtimeAction::kInputAudioBufferCommit;
    } else {
        return Status(StatusCode::kInvalidArgument, "unsupported realtime request type: " + type);
    }

    request->stream = json_body.value("stream", true);

    const Json * session = nullptr;
    if (json_body.contains("session") && json_body["session"].is_object()) {
        session = &json_body["session"];
    }

    request->session_id = json_body.value("session_id", std::string());
    if (request->session_id.empty() && session != nullptr) {
        request->session_id = session->value("id", std::string());
    }

    request->model = json_body.value("model", std::string());
    if (request->model.empty() && session != nullptr) {
        request->model = session->value("model", std::string());
    }

    request->language = json_body.value("language", std::string());
    if (request->language.empty() && session != nullptr) {
        request->language = session->value("language", std::string());
    }

    request->input_audio_format = json_body.value("input_audio_format", std::string());
    if (request->input_audio_format.empty() && session != nullptr) {
        request->input_audio_format = session->value("input_audio_format", std::string());
    }
    if (request->input_audio_format.empty()) {
        request->input_audio_format = "pcm16le";
    }
    request->input_audio_format = NormalizeAsciiLower(request->input_audio_format);
    if (request->input_audio_format == "pcm16") {
        request->input_audio_format = "pcm16le";
    }
    if (request->input_audio_format != "pcm16le") {
        return Status(StatusCode::kFailedPrecondition, "realtime path currently supports only input_audio_format=pcm16le");
    }

    request->audio = json_body.value("audio", std::string());
    if (request->audio.empty() &&
        json_body.contains("input_audio_buffer") &&
        json_body["input_audio_buffer"].is_object()) {
        request->audio = json_body["input_audio_buffer"].value("audio", std::string());
    }

    if (request->action != OpenAiRealtimeAction::kSessionCreate && request->session_id.empty()) {
        return Status(StatusCode::kInvalidArgument, "session_id is required");
    }
    if (request->action == OpenAiRealtimeAction::kInputAudioBufferAppend && request->audio.empty()) {
        return Status(StatusCode::kInvalidArgument, "audio is required for input_audio_buffer.append");
    }
    return OkStatus();
}

Status DecodeBase64Pcm16Le(std::string_view encoded, std::vector<float> * samples) {
    if (samples == nullptr) {
        return Status(StatusCode::kInvalidArgument, "samples output must not be null");
    }

    std::string compact;
    compact.reserve(encoded.size());
    for (const char ch : encoded) {
        if (!IsAsciiWhitespace(ch)) {
            compact.push_back(ch);
        }
    }

    if (compact.empty()) {
        return Status(StatusCode::kInvalidArgument, "audio must be a base64-encoded pcm16le payload");
    }
    if (compact.size() % 4U != 0U) {
        return Status(StatusCode::kInvalidArgument, "audio base64 payload must be padded to 4-byte groups");
    }

    std::string bytes;
    bytes.reserve((compact.size() / 4U) * 3U);
    for (std::size_t index = 0; index < compact.size(); index += 4U) {
        const char c0 = compact[index + 0U];
        const char c1 = compact[index + 1U];
        const char c2 = compact[index + 2U];
        const char c3 = compact[index + 3U];
        const int v0 = DecodeBase64Value(c0);
        const int v1 = DecodeBase64Value(c1);
        const int v2 = (c2 == '=') ? 0 : DecodeBase64Value(c2);
        const int v3 = (c3 == '=') ? 0 : DecodeBase64Value(c3);
        if (v0 < 0 || v1 < 0 || (c2 != '=' && v2 < 0) || (c3 != '=' && v3 < 0)) {
            return Status(StatusCode::kInvalidArgument, "audio contains invalid base64 characters");
        }
        if (c2 == '=' && c3 != '=') {
            return Status(StatusCode::kInvalidArgument, "audio base64 padding is malformed");
        }

        bytes.push_back(static_cast<char>((v0 << 2) | (v1 >> 4)));
        if (c2 != '=') {
            bytes.push_back(static_cast<char>(((v1 & 0x0F) << 4) | (v2 >> 2)));
        }
        if (c3 != '=') {
            bytes.push_back(static_cast<char>(((v2 & 0x03) << 6) | v3));
        }
    }

    if (bytes.size() % 2U != 0U) {
        return Status(StatusCode::kInvalidArgument, "pcm16le audio must contain an even number of bytes");
    }

    samples->clear();
    samples->reserve(bytes.size() / 2U);
    for (std::size_t index = 0; index < bytes.size(); index += 2U) {
        const std::uint8_t lo = static_cast<std::uint8_t>(bytes[index + 0U]);
        const std::uint8_t hi = static_cast<std::uint8_t>(bytes[index + 1U]);
        const std::int16_t sample = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(lo) |
            (static_cast<std::uint16_t>(hi) << 8U));
        samples->push_back(static_cast<float>(sample) / 32768.0f);
    }
    return OkStatus();
}

float RealtimeStreamChunkSeconds(const RealtimePolicyConfig & policy) noexcept {
    float seconds = static_cast<float>(policy.min_decode_interval_ms) / 1000.0f;
    if (seconds < 0.4f) {
        seconds = 0.4f;
    }
    if (seconds > 1.0f) {
        seconds = 1.0f;
    }
    return seconds;
}

int RealtimeStreamMaxNewTokens(const RealtimePolicyConfig & policy) noexcept {
    return RealtimeStreamChunkSeconds(policy) <= 0.8f ? 24 : 32;
}

namespace {

namespace fs = std::filesystem;

#ifdef QASR_CPU_BACKEND_ENABLED
using Json = qasr::Json;
#endif

constexpr std::size_t kHttpWorkerQueueLimit = 64;
constexpr std::size_t kMaxRealtimeSessions = 64;
constexpr std::int64_t kAsyncJobCleanupIntervalSeconds = 60;
constexpr std::int64_t kCompletedAsyncJobTtlSeconds = 3600;

Status ParseInt32Argument(std::string_view text, const char * field_name, std::int32_t * value) {
    if (value == nullptr) {
        return Status(StatusCode::kInvalidArgument, "value output must not be null");
    }
    if (text.empty()) {
        return Status(StatusCode::kInvalidArgument, std::string(field_name) + " must not be empty");
    }
    std::int32_t parsed = 0;
    const char * begin = text.data();
    const char * end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) {
        return Status(StatusCode::kInvalidArgument, std::string(field_name) + " must be a valid int32");
    }
    *value = parsed;
    return OkStatus();
}

Status RequireValue(int argc, const char * const argv[], int index, const char * flag_name, const char ** value) {
    if (value == nullptr) {
        return Status(StatusCode::kInvalidArgument, "value output must not be null");
    }
    if (index + 1 >= argc) {
        return Status(StatusCode::kInvalidArgument, std::string(flag_name) + " requires a value");
    }
    *value = argv[index + 1];
    return OkStatus();
}

std::string JsonErrorBody(const Status & status) {
#ifdef QASR_CPU_BACKEND_ENABLED
    Json body;
    body["error"] = Json::object({
        {"code", StatusCodeName(status.code())},
        {"message", status.message()},
    });
    return body.dump();
#else
    return "{\"error\":{\"code\":\"internal\",\"message\":\"cpu backend disabled\"}}";
#endif
}

std::int64_t CurrentUnixSeconds() {
    const auto now = std::chrono::system_clock::now();
    return static_cast<std::int64_t>(std::chrono::system_clock::to_time_t(now));
}

int StatusToHttpCode(const Status & status) {
    switch (status.code()) {
        case StatusCode::kInvalidArgument:
        case StatusCode::kOutOfRange:
            return 400;
        case StatusCode::kNotFound:
            return 404;
        case StatusCode::kFailedPrecondition:
            return 412;
        case StatusCode::kUnimplemented:
            return 501;
        default:
            return 500;
    }
}

bool HasWavExtension(const fs::path & path) {
    std::string extension = path.extension().string();
    for (char & ch : extension) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return extension == ".wav" || extension == ".wave";
}

bool IsHttpUrl(std::string_view value) {
    return value.rfind("http://", 0) == 0 || value.rfind("https://", 0) == 0;
}

std::string NormalizeAudioLocator(std::string_view locator) {
    if (locator.rfind("file://", 0) == 0) {
        return std::string(locator.substr(7));
    }
    return std::string(locator);
}

bool CommandExists(const char * name) {
    /* We must NOT use `command -v` here.  `command` is a bash shell
     * builtin and is not a real executable at /usr/bin/command, so
     * posix_spawnp("command", ...) fails (rc=127).  This was a long-
     * standing bug that made every FfmpegAvailable() check return
     * false on Linux, which in turn made every non-WAV /api/transcriptions*
     * upload fail with "ffmpeg is required" even though /usr/bin/ffmpeg
     * was clearly in PATH.
     *
     * Prefer `which` (a real binary on virtually every Linux distro and
     * in /usr/bin on macOS).  Fall back to the explicit /usr/bin/which
     * path on Linux if the PATH-only lookup fails.  On Windows the
     * `where.exe` builtin is a real binary.  We still avoid `system()`
     * so the executable name is never interpreted as shell syntax. */
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
#ifdef _WIN32
    return qasr::SpawnAndWait({"where", name}) == 0;
#else
    if (qasr::SpawnAndWait({"which", name}) == 0) {
        return true;
    }
    /* Hard-coded fallback for the most common Linux location. */
    if (qasr::SpawnAndWait({"/usr/bin/which", name}) == 0) {
        return true;
    }
    return false;
#endif
}

bool FfmpegAvailable() {
    static const bool available = CommandExists("ffmpeg");
    return available;
}

fs::path MakeTempPath(std::string_view prefix, std::string_view suffix) {
    static std::atomic<std::uint64_t> counter{1};
    const std::uint64_t id = counter.fetch_add(1);
    return fs::temp_directory_path() /
        (std::string(prefix) + "-" + std::to_string(CurrentUnixSeconds()) + "-" + std::to_string(id) + std::string(suffix));
}

Status WriteBinaryFile(const fs::path & path, const std::string & data) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return Status(StatusCode::kInternal, "failed to open temp file: " + path.string());
    }
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!output.good()) {
        return Status(StatusCode::kInternal, "failed to write temp file: " + path.string());
    }
    return OkStatus();
}

Status NormalizeAudioToWav16kMono(std::string_view locator, const fs::path & output_path) {
    if (!FfmpegAvailable()) {
        return Status(StatusCode::kFailedPrecondition, "ffmpeg is required for non-wav audio normalization");
    }

    // Build the ffmpeg argv directly; do NOT pass through a shell,
    // so the locator and output paths are preserved verbatim even
    // if they contain shell metacharacters.
    const std::vector<std::string> args = {
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", std::string(locator),
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path.string(),
    };
    if (qasr::SpawnAndWait(args) != 0) {
        return Status(StatusCode::kInternal, "ffmpeg normalization failed");
    }
    return OkStatus();
}

struct PreparedAudioInput {
    fs::path original_path;
    fs::path wav_path;
    bool cleanup_original = false;
    bool cleanup_wav = false;
};

void CleanupPreparedAudio(PreparedAudioInput * prepared) {
    if (prepared == nullptr) {
        return;
    }
    std::error_code ec;
    if (prepared->cleanup_wav && !prepared->wav_path.empty()) {
        fs::remove(prepared->wav_path, ec);
        ec.clear();
    }
    if (prepared->cleanup_original && !prepared->original_path.empty()) {
        if (prepared->original_path != prepared->wav_path) {
            fs::remove(prepared->original_path, ec);
        }
    }
}

#ifdef QASR_CPU_BACKEND_ENABLED
const MultipartFormData * FindUploadedAudio(const HttpRequest & request) {
    auto it = request.files.find("file");
    if (it != request.files.end()) {
        return &it->second;
    }
    it = request.files.find("audio");
    if (it != request.files.end()) {
        return &it->second;
    }
    return nullptr;
}

bool TryGetFormField(const HttpRequest & request, const std::string & name, std::string * value) {
    if (value == nullptr) {
        return false;
    }
    if (request.has_param(name)) {
        *value = request.get_param_value(name);
        return true;
    }
    const auto it = request.files.find(name);
    if (it != request.files.end() && it->second.filename.empty()) {
        *value = it->second.content;
        return true;
    }
    return false;
}
#endif

Status PrepareUploadedAudio(
#ifdef QASR_CPU_BACKEND_ENABLED
    const MultipartFormData & file,
#else
    const std::string &,
#endif
    PreparedAudioInput * prepared) {
    if (prepared == nullptr) {
        return Status(StatusCode::kInvalidArgument, "prepared output must not be null");
    }
#ifndef QASR_CPU_BACKEND_ENABLED
    return Status(StatusCode::kUnimplemented, "cpu backend disabled");
#else
    if (file.content.empty()) {
        return Status(StatusCode::kInvalidArgument, "uploaded audio must not be empty");
    }

    const std::string suffix = file.filename.empty() ? ".bin" : fs::path(file.filename).extension().string();
    prepared->original_path = MakeTempPath("qasr-upload", suffix.empty() ? ".bin" : suffix);
    Status status = WriteBinaryFile(prepared->original_path, file.content);
    if (!status.ok()) {
        return status;
    }
    prepared->cleanup_original = true;

    if (HasWavExtension(prepared->original_path)) {
        prepared->wav_path = prepared->original_path;
        return OkStatus();
    }

    prepared->wav_path = MakeTempPath("qasr-normalized", ".wav");
    prepared->cleanup_wav = true;
    status = NormalizeAudioToWav16kMono(prepared->original_path.string(), prepared->wav_path);
    if (!status.ok()) {
        CleanupPreparedAudio(prepared);
        return status;
    }
    return OkStatus();
#endif
}

Status PrepareAudioLocator(std::string_view locator, PreparedAudioInput * prepared) {
    if (prepared == nullptr) {
        return Status(StatusCode::kInvalidArgument, "prepared output must not be null");
    }

    const std::string normalized_locator = NormalizeAudioLocator(locator);
    if (normalized_locator.empty()) {
        return Status(StatusCode::kInvalidArgument, "audio locator must not be empty");
    }

    if (!IsHttpUrl(normalized_locator)) {
        const fs::path path(normalized_locator);
        if (!fs::exists(path)) {
            return Status(StatusCode::kNotFound, "audio source does not exist: " + normalized_locator);
        }
        if (!fs::is_regular_file(path)) {
            return Status(StatusCode::kInvalidArgument, "audio source must be a file: " + normalized_locator);
        }
        if (HasWavExtension(path)) {
            prepared->wav_path = path;
            return OkStatus();
        }
    }

    prepared->wav_path = MakeTempPath("qasr-source", ".wav");
    prepared->cleanup_wav = true;
    Status status = NormalizeAudioToWav16kMono(normalized_locator, prepared->wav_path);
    if (!status.ok()) {
        CleanupPreparedAudio(prepared);
        return status;
    }
    return OkStatus();
}

Status DecodePcm16Le(const std::string & body, std::vector<float> * samples) {
     if (!samples) {
         return Status(StatusCode::kInvalidArgument, "samples output must not be null");
     }
     if ((body.size() % 2U) != 0U) {
         return Status(StatusCode::kInvalidArgument,
             "pcm16le audio must contain an even number of bytes (got " +
             std::to_string(body.size()) + ")");
     }
     samples->resize(body.size() / 2U);
     for (std::size_t index = 0; index < samples->size(); ++index) {
         const unsigned char low = static_cast<unsigned char>(body[index * 2U]);
         const unsigned char high = static_cast<unsigned char>(body[index * 2U + 1U]);
         const std::int16_t value = static_cast<std::int16_t>(static_cast<std::uint16_t>(low) |
             (static_cast<std::uint16_t>(high) << 8U));
         (*samples)[index] = static_cast<float>(value) / 32768.0f;
     }
     return OkStatus();
 }

#if defined(QASR_CPU_BACKEND_ENABLED)
 Status DecodePcm16Le(const char * data, std::size_t size, std::vector<float> * samples) {
     return DecodePcm16Le(std::string(data, size), samples);
 }
#endif

std::string LoadTextFile(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

#ifdef QASR_CPU_BACKEND_ENABLED
struct TranscriptionApiOptions {
    std::string model;
    std::string prompt;
    std::string language;
    TranscriptionResponseFormat response_format = TranscriptionResponseFormat::kJson;
    bool stream = false;
    bool want_segment_timestamps = false;
    bool want_word_timestamps = false;
};

Status ParseTranscriptionApiOptions(const HttpRequest & request, TranscriptionApiOptions * options) {
    if (options == nullptr) {
        return Status(StatusCode::kInvalidArgument, "options output must not be null");
    }

    *options = TranscriptionApiOptions{};

    std::string field_value;
    if (TryGetFormField(request, "model", &field_value)) {
        options->model = field_value;
    }
    if (TryGetFormField(request, "prompt", &field_value)) {
        options->prompt = field_value;
    }
    if (TryGetFormField(request, "language", &field_value)) {
        options->language = field_value;
    }
    if (TryGetFormField(request, "response_format", &field_value)) {
        Status status = ParseTranscriptionResponseFormat(field_value, &options->response_format);
        if (!status.ok()) {
            return status;
        }
    }
    if (TryGetFormField(request, "stream", &field_value)) {
        Status status = ParseBooleanText("stream", field_value, &options->stream);
        if (!status.ok()) {
            return status;
        }
    }

    const auto apply_granularity = [&](std::string value) -> Status {
        for (char & ch : value) {
            if (ch >= 'A' && ch <= 'Z') {
                ch = static_cast<char>(ch - 'A' + 'a');
            }
        }
        if (value == "segment") {
            options->want_segment_timestamps = true;
            return OkStatus();
        }
        if (value == "word") {
            options->want_word_timestamps = true;
            return OkStatus();
        }
        return Status(StatusCode::kInvalidArgument, "unsupported timestamp granularity: " + value);
    };

    for (const char * key : {"timestamp_granularities[]", "timestamp_granularities"}) {
        const std::size_t count = request.get_param_value_count(key);
        for (std::size_t index = 0; index < count; ++index) {
            Status status = apply_granularity(request.get_param_value(key, index));
            if (!status.ok()) {
                return status;
            }
        }
        if (TryGetFormField(request, key, &field_value)) {
            Status status = apply_granularity(field_value);
            if (!status.ok()) {
                return status;
            }
        }
    }

    Status status = ValidateTimestampGranularities(options->want_segment_timestamps, options->want_word_timestamps);
    if (!status.ok()) {
        return status;
    }
    if ((options->want_segment_timestamps || options->want_word_timestamps) &&
        options->response_format != TranscriptionResponseFormat::kVerboseJson) {
        return Status(StatusCode::kInvalidArgument, "timestamp_granularities require response_format=verbose_json");
    }
    return OkStatus();
}

struct ChatCompletionRequestOptions {
    std::string model;
    std::string prompt;
    std::string language;
    std::string audio_locator;
    bool stream = false;
};

Status ParseChatCompletionRequest(const HttpRequest & request, ChatCompletionRequestOptions * options) {
    if (options == nullptr) {
        return Status(StatusCode::kInvalidArgument, "options output must not be null");
    }

    Json body = Json::parse(request.body);
    if (body.is_discarded()) {
        return Status(StatusCode::kInvalidArgument, "request body must be valid JSON");
    }

    *options = ChatCompletionRequestOptions{};
    options->model = body.value("model", std::string());
    options->stream = body.value("stream", false);
    options->language = body.value("language", std::string());
    if (body.contains("extra_body") && body["extra_body"].is_object()) {
        options->language = body["extra_body"].value("language", options->language);
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        return Status(StatusCode::kInvalidArgument, "messages must be a non-empty array");
    }

    for (const Json & message : body["messages"]) {
        if (!message.is_object() || message.value("role", std::string()) != "user") {
            continue;
        }

        if (!message.contains("content")) {
            continue;
        }

        const Json & content = message["content"];
        if (content.is_string()) {
            if (!options->prompt.empty()) {
                options->prompt.push_back(' ');
            }
            options->prompt += content.get<std::string>();
            continue;
        }

        if (!content.is_array()) {
            continue;
        }

        for (const Json & item : content) {
            if (!item.is_object()) {
                continue;
            }
            const std::string type = item.value("type", std::string());
            if (type == "text") {
                if (!options->prompt.empty()) {
                    options->prompt.push_back(' ');
                }
                options->prompt += item.value("text", std::string());
                continue;
            }
            if (type == "audio_url" && item.contains("audio_url") && item["audio_url"].is_object()) {
                options->audio_locator = item["audio_url"].value("url", std::string());
            }
        }
    }

    if (options->audio_locator.empty()) {
        return Status(StatusCode::kInvalidArgument, "chat completion request must contain one audio_url");
    }
    return OkStatus();
}

struct ModelDecodeOptions {
    std::string prompt;
    std::string language;
    int stream_max_new_tokens = 32;
    float stream_chunk_sec = 0.0f;
    float temperature = -1.0f;
    bool use_stream_path = false;
    std::function<void(std::string_view)> token_callback;
    std::function<bool()> cancel_callback;
};

void ForwardTokenPiece(const char * piece, void * userdata) {
    if (piece == nullptr || userdata == nullptr) {
        return;
    }
    auto * callback = static_cast<std::function<void(std::string_view)> *>(userdata);
    (*callback)(piece);
}

void ForwardStreamChunk(const qwen_stream_chunk_t * chunk, void * userdata) {
    if (chunk == nullptr || userdata == nullptr) {
        return;
    }
    auto * callback = static_cast<std::function<void(const qwen_stream_chunk_t *)> *>(userdata);
    (*callback)(chunk);
}

int ForwardCancelRequest(void * userdata) {
    if (userdata == nullptr) {
        return 0;
    }
    auto * callback = static_cast<std::function<bool()> *>(userdata);
    return (*callback)() ? 1 : 0;
}


// InferHandle: opaque handle returned by createInferHandle().
// Wraps a SessionHandle or a raw qwen_ctx_t* depending on backend.
struct InferHandle {
    std::unique_ptr<SessionHandle> engineHandle;
    qwen_ctx_t * nativeCtx = nullptr;  // CPU: ctx; CUDA: nullptr
};

/* ServerAsrFacade: facade over AsrEngine (CPU/CUDA) + C bridge.
 *
 * Provides the 6 call modes that RunServer uses:
 *   ① TranscribeFile  (sync, batch)
 *   ② TranscribeRealtime  (sync, chat)
 *   ③ createInferHandle  (VAD-segmented batch, async)
 *   ④ CreateRealtimeClone → via engine CreateRealtimeSession
 *   ⑤ vad()
 *   ⑥ temperature() / verbosity()
 *
 * CPU path: CpuAsrEngine owns qwen_ctx_t; facade calls C bridge on base_ctx().
 * CUDA path: CudaAsrEngine handles inference via engine methods. */
class ServerAsrFacade {
public:
    Status Initialize(const ServerConfig & config) {
        config_ = config;
        backendKind_ = config.backend;
        backendFallback_ = false;

        V2EngineConfig engCfg;
        engCfg.model_dir = config.model_dir;
        engCfg.threads = config.threads;
        engCfg.temperature = config.temperature;
        engCfg.max_sessions = 8;
        engCfg.verbosity = config.verbosity;

        /* Try requested backend */
        auto engine = CreateEngine(config.backend);
        if (engine && engine->LoadModel(engCfg).ok()) {
            engine_ = std::move(engine);
        } else {
            /* Fallback to CPU */
            if (config.allow_backend_fallback && config.backend != BackendKind::kCpu) {
                backendKind_ = BackendKind::kCpu;
                backendFallback_ = true;
                engine = CreateEngine(BackendKind::kCpu);
                if (engine && engine->LoadModel(engCfg).ok()) {
                    engine_ = std::move(engine);
                }
            }
        }

        if (!engine_) {
            return Status(StatusCode::kInternal,
                          "engine init failed");
        }

        /* Initialize GPU scheduler for non-CPU backends (§7).
         * CPU backend uses C bridge directly; scheduler is for GPU
         * segment queuing and concurrency control. */
        if (backendKind_ != BackendKind::kCpu) {
            scheduler_ = std::make_unique<GpuScheduler>();
            scheduler_->SetEngine(engine_.get());
            scheduler_->SetCallback([](const SegmentResult & res) {
                (void)res;
            });
            scheduler_->SetWorker(engCfg.max_active_gpu_jobs);
            scheduler_->Start();
        }

        return OkStatus();
    }

  AsrRunResult TranscribeFile(const fs::path & audio_path,
                                   const ModelDecodeOptions & decode) {
        if (!engine_) {
            AsrRunResult r;
            r.status = Status(StatusCode::kFailedPrecondition, "engine not loaded");
            return r;
        }
        auto * base = static_cast<qwen_ctx_t *>(engine_->base_ctx());
        if (base) {
            return doTranscribeFile(base, decode, audio_path);
        }
        /* GPU path: TranscribeSegment handles the full pipeline. */
        return doTranscribeViaEngine(decode, [this, &decode, &audio_path]() {
            std::vector<float> samples;
            std::int64_t dur_ms = 0;
            Status st = LoadAudioFile(audio_path.string(), &samples, &dur_ms);
            if (!st.ok()) {
                AsrRunResult r;
                r.status = Status(StatusCode::kInvalidArgument, "failed to load audio: " + audio_path.string());
                return r;
            }
            return doTranscribeSamples(decode, samples);
        });
    }

    AsrRunResult TranscribeRealtime(const std::vector<float> & samples,
                                       const ModelDecodeOptions & decode) {
        if (!engine_) {
            AsrRunResult r;
            r.status = Status(StatusCode::kFailedPrecondition, "engine not loaded");
            return r;
        }
        auto * base = static_cast<qwen_ctx_t *>(engine_->base_ctx());
        if (base) {
            return doTranscribeRealtime(base, decode, samples);
        }
        /* GPU path: TranscribeSegment handles the full pipeline. */
        return doTranscribeViaEngine(decode, [&]() {
            return doTranscribeSamples(decode, samples);
        });
    }

    InferHandle createInferHandle() {
        InferHandle h;
        if (engine_) {
            auto handle = engine_->CreateRealtimeSession({});
            if (handle) {
                h.engineHandle = std::move(handle);
                h.nativeCtx = static_cast<qwen_ctx_t *>(h.engineHandle->nativeCtx());
            }
        }
        return h;
    }

    void releaseInferHandle(InferHandle & h) {
        if (h.engineHandle && engine_) {
            engine_->CloseSessionHandle(std::move(h.engineHandle));
            h.nativeCtx = nullptr;
        } else if (h.nativeCtx) {
            qwen_free(h.nativeCtx);
            h.nativeCtx = nullptr;
        }
    }

    qwen_silero_vad_t *vad() const noexcept {
        if (engine_) {
            void * vh = engine_->getVadHandle();
            return static_cast<qwen_silero_vad_t *>(vh);
        }
        return nullptr;
    }

    float temperature() const noexcept { return config_.temperature; }
    int verbosity() const noexcept { return config_.verbosity; }
    BackendKind backendKind() const noexcept { return backendKind_; }
    bool backendFallback() const noexcept { return backendFallback_; }
    AsrEngine *engine() const noexcept { return engine_.get(); }
    GpuScheduler *scheduler() const noexcept { return scheduler_.get(); }

    /* GPU pipeline: transcribe samples via engine->TranscribeSegment.
     * Public so AsrWorkerLoop (defined outside the class) can call it. */
    AsrRunResult TranscribeSamplesViaEngine(
        const ModelDecodeOptions & decode,
        const std::vector<float> & samples) {
        return doTranscribeSamples(decode, samples);
    }

private:
    /* Engine-backed helpers. */
    AsrRunResult doTranscribeFile(qwen_ctx_t * ctx,
                                    const ModelDecodeOptions & decode,
                                    const fs::path & audio_path) {
        AsrRunResult result;
        qwen_verbose = config_.verbosity;
        ctx->stream_max_new_tokens = decode.stream_max_new_tokens;
        if (decode.stream_chunk_sec > 0.0f) ctx->stream_chunk_sec = decode.stream_chunk_sec;
        float temp = decode.temperature >= 0.0f
            ? decode.temperature
            : (config_.temperature >= 0.0f ? config_.temperature : -1.0f);
        if (temp >= 0.0f) ctx->decode_temperature = temp;
        if (qwen_set_prompt(ctx, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument,
                                   "unsupported language: " + decode.language);
            return result;
        }
        std::function<void(std::string_view)> token_cb = decode.token_callback;
        std::function<bool()> cancel_cb = decode.cancel_callback;
        qwen_set_token_callback(ctx, token_cb ? ForwardTokenPiece : nullptr,
                                token_cb ? &token_cb : nullptr);
        qwen_set_cancel_callback(ctx, cancel_cb ? ForwardCancelRequest : nullptr,
                                 cancel_cb ? &cancel_cb : nullptr);
        char * raw = qwen_transcribe(ctx, audio_path.string().c_str());
        bool was_cancelled = qwen_was_cancelled(ctx) != 0;
        qwen_set_cancel_callback(ctx, nullptr, nullptr);
        qwen_set_token_callback(ctx, nullptr, nullptr);
        if (!raw) {
            result.status = was_cancelled
                ? Status(StatusCode::kFailedPrecondition, "transcription cancelled")
                : Status(StatusCode::kInternal, "transcription failed");
            return result;
        }
        result.text = raw;
        std::free(raw);
        result.total_ms = ctx->perf_total_ms;
        result.audio_ms = ctx->perf_audio_ms;
        result.text_tokens = ctx->perf_text_tokens;
        result.encode_ms = ctx->perf_encode_ms;
        result.decode_ms = ctx->perf_decode_ms;
        result.status = was_cancelled
            ? Status(StatusCode::kFailedPrecondition, "transcription cancelled")
            : OkStatus();
        return result;
    }

      AsrRunResult doTranscribeRealtime(qwen_ctx_t * ctx,
                                        const ModelDecodeOptions & decode,
                                        const std::vector<float> & samples) {
        AsrRunResult result;
        qwen_verbose = config_.verbosity;
        ctx->stream_max_new_tokens = decode.stream_max_new_tokens;
        float temp = decode.temperature >= 0.0f
            ? decode.temperature
            : (config_.temperature >= 0.0f ? config_.temperature : -1.0f);
        if (temp >= 0.0f) ctx->decode_temperature = temp;
        if (qwen_set_prompt(ctx, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument,
                                   "unsupported language: " + decode.language);
            return result;
        }
        std::function<void(std::string_view)> token_cb = decode.token_callback;
        std::function<bool()> cancel_cb = decode.cancel_callback;
        qwen_set_token_callback(ctx, token_cb ? ForwardTokenPiece : nullptr,
                                token_cb ? &token_cb : nullptr);
        qwen_set_cancel_callback(ctx, cancel_cb ? ForwardCancelRequest : nullptr,
                                 cancel_cb ? &cancel_cb : nullptr);
        char * raw = decode.use_stream_path
            ? qwen_transcribe_stream(ctx, samples.data(), static_cast<int>(samples.size()))
            : qwen_transcribe_audio(ctx, samples.data(), static_cast<int>(samples.size()));
        bool was_cancelled = qwen_was_cancelled(ctx) != 0;
        qwen_set_cancel_callback(ctx, nullptr, nullptr);
        qwen_set_token_callback(ctx, nullptr, nullptr);
        if (!raw) {
            result.status = was_cancelled
                ? Status(StatusCode::kFailedPrecondition, decode.use_stream_path
                    ? "stream transcription cancelled" : "audio transcription cancelled")
                : Status(StatusCode::kInternal, decode.use_stream_path
                    ? "stream transcription failed" : "audio transcription failed");
            return result;
        }
        result.text = raw;
        std::free(raw);
        result.total_ms = ctx->perf_total_ms;
        result.audio_ms = ctx->perf_audio_ms;
        result.text_tokens = ctx->perf_text_tokens;
        result.encode_ms = ctx->perf_encode_ms;
        result.decode_ms = ctx->perf_decode_ms;
        result.status = was_cancelled
            ? Status(StatusCode::kFailedPrecondition, decode.use_stream_path
                ? "stream transcription cancelled" : "audio transcription cancelled")
            : OkStatus();
        return result;
    }

    /* Generic engine-backed transcription: creates a session, calls
     * TranscribeSegment, then closes the session.  Used by both batch
     * and realtime paths when base_ctx() is null (CUDA GPU pipeline). */
    AsrRunResult doTranscribeViaEngine(
        const ModelDecodeOptions & decode,
        std::function<AsrRunResult()> transcribe_fn) {
        auto * base = static_cast<qwen_ctx_t *>(engine_->base_ctx());
        if (!base) {
            std::uint64_t sid = 0;
            SessionOptions opts;
            opts.language = decode.language;
            opts.prompt = decode.prompt;
            if (decode.temperature >= 0.0f) {
                opts.temperature = decode.temperature;
            } else if (config_.temperature >= 0.0f) {
                opts.temperature = config_.temperature;
            }
            Status st = engine_->CreateSession(opts, sid);
            if (!st.ok()) {
                AsrRunResult r;
                r.status = st;
                return r;
            }
            AsrRunResult result = transcribe_fn();
            if (result.status.ok()) {
                /* TranscribeSegment was called by the lambda — convert result. */
            }
            engine_->CloseSession(sid);
            return result;
        }
        return transcribe_fn();
    }

    /* Transcribe samples via TranscribeSegment. */
    AsrRunResult doTranscribeSamples(const ModelDecodeOptions & decode,
                                       const std::vector<float> & samples) {
        if (samples.empty()) {
            AsrRunResult r;
            r.status = Status(StatusCode::kInvalidArgument, "empty audio");
            return r;
        }
        std::uint64_t sid = 0;
        SessionOptions opts;
        opts.language = decode.language;
        opts.prompt = decode.prompt;
        if (decode.temperature >= 0.0f) {
            opts.temperature = decode.temperature;
        } else if (config_.temperature >= 0.0f) {
            opts.temperature = config_.temperature;
        }
        Status st = engine_->CreateSession(opts, sid);
        if (!st.ok()) {
            AsrRunResult r;
            r.status = st;
            return r;
        }
        AsrSegmentResult seg = engine_->TranscribeSegment(sid, samples);
        AsrRunResult result;
        result.status = seg.status;
        result.text = seg.text;
        result.total_ms = seg.total_ms;
        result.audio_ms = seg.audio_ms;
        result.text_tokens = seg.text_tokens;
        result.encode_ms = seg.encode_ms;
        result.decode_ms = seg.decode_ms;
        engine_->CloseSession(sid);
        return result;
    }

    ServerConfig config_;
    std::unique_ptr<AsrEngine> engine_;
    BackendKind backendKind_ = BackendKind::kCpu;
    bool backendFallback_ = false;

    /* ── Session pool (§6 ServerAsrFacade, §7 Scheduler) ── */

    struct RealtimeSessionData {
        std::string sessionId;
        std::string language;
        std::string prompt;
        float temperature = -1.0f;
        std::uint64_t engineSid = 0;
        bool active = true;
        std::chrono::steady_clock::time_point createdAt = std::chrono::steady_clock::now();
    };

    mutable std::mutex poolMu_;
    std::unordered_map<std::string, std::unique_ptr<RealtimeSessionData>> sessions_;
    uint64_t nextSessionId_ = 1;

    /* ── GPU Scheduler (§7) ── */
    std::unique_ptr<GpuScheduler> scheduler_;

    /* Mode ④: realtime session management */
    Status createRealtimeSession(const std::string & sid) {
        if (!engine_) {
            return Status(StatusCode::kFailedPrecondition, "engine not loaded");
        }
        std::lock_guard<std::mutex> lock(poolMu_);
        if (sessions_.count(sid) > 0) {
            return Status(StatusCode::kFailedPrecondition,
                          "realtime session exists: " + sid);
        }
        SessionOptions opts;
        std::uint64_t esid = 0;
        Status st = engine_->CreateSession(opts, esid);
        if (!st.ok()) return st;
        auto data = std::make_unique<RealtimeSessionData>();
        data->sessionId = sid;
        data->engineSid = esid;
        sessions_[sid] = std::move(data);
        return OkStatus();
    }

    void destroyRealtimeSession(const std::string & sid) {
        std::lock_guard<std::mutex> lock(poolMu_);
        auto it = sessions_.find(sid);
        if (it != sessions_.end()) {
            auto * d = it->second.get();
            if (d && engine_ && d->engineSid) {
                engine_->CloseSession(d->engineSid);
            }
            sessions_.erase(it);
        }
    }

    /* Get the engine session id for a realtime session. */
    std::uint64_t getSessionEngineSid(const std::string & sid) const {
        std::lock_guard<std::mutex> lock(poolMu_);
        auto it = sessions_.find(sid);
        if (it != sessions_.end() && it->second) {
            return it->second->engineSid;
        }
        return 0;
    }
};

void SetJsonResponse(HttpResponse & response, const Json & body) {
    response.set_content(body.dump(), "application/json");
}

void SetErrorResponse(HttpResponse & response, const Status & status, int http_code) {
    response.status = http_code;
    response.set_content(JsonErrorBody(status), "application/json");
}

std::string DetectLanguageLabel(std::string_view requested_language) {
    return requested_language.empty() ? "unknown" : std::string(requested_language);
}

Json BuildBasicTranscriptionJson(
    const AsrRunResult & result,
    const TranscriptionApiOptions & options) {
    Json body;
    body["text"] = result.text;
    body["language"] = DetectLanguageLabel(options.language);
    body["inference_ms"] = result.total_ms;
    body["audio_ms"] = result.audio_ms;
    body["tokens"] = result.text_tokens;
    return body;
}

Json BuildVerboseTranscriptionJson(
    const AsrRunResult & result,
    const TranscriptionApiOptions & options) {
    Json body;
    body["task"] = "transcribe";
    body["language"] = DetectLanguageLabel(options.language);
    body["duration"] = result.audio_ms / 1000.0;
    body["text"] = result.text;

    Json segments = Json::array();
    Json segment;
    segment["id"] = 0;
    segment["seek"] = 0;
    segment["start"] = 0.0;
    segment["end"] = result.audio_ms / 1000.0;
    segment["text"] = result.text;
    segment["tokens"] = Json::array();
    if (options.want_segment_timestamps) {
        segment["words"] = Json::array();
    }
    segments.push_back(segment);
    body["segments"] = segments;
    return body;
}

struct OfflineJob {
    std::string id;
    std::string state = "queued";
    std::string text;
    std::string error;
    std::string language = "unknown";
    double inference_ms = 0.0;
    double audio_ms = 0.0;
    std::int32_t tokens = 0;
    std::int32_t token_count = 0;
    bool cancel_requested = false;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::int64_t created_at = 0;
    std::int64_t updated_at = 0;
};

Json BuildJobJson(const OfflineJob & job) {
    Json body;
    body["id"] = job.id;
    body["state"] = job.state;
    body["text"] = job.text;
    body["error"] = job.error;
    body["language"] = job.language;
    body["inference_ms"] = job.inference_ms;
    body["audio_ms"] = job.audio_ms;
    body["tokens"] = job.tokens;
    body["token_count"] = job.token_count;
    body["cancel_requested"] = job.cancel_requested;
    body["created_at"] = job.created_at;
    body["updated_at"] = job.updated_at;
    return body;
}

std::size_t CleanupExpiredJobs(
    std::unordered_map<std::string, OfflineJob> * jobs,
    std::int64_t now_seconds,
    std::int64_t ttl_seconds) {
    if (jobs == nullptr) {
        return 0U;
    }

    std::size_t removed = 0U;
    for (auto it = jobs->begin(); it != jobs->end();) {
        if (ShouldEvictCompletedJob(it->second.state, it->second.updated_at, now_seconds, ttl_seconds)) {
            it = jobs->erase(it);
            ++removed;
            continue;
        }
        ++it;
    }
    return removed;
}

class SseStreamState {
public:
    void Push(std::string event) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            events_.push_back(std::move(event));
        }
        cv_.notify_one();
    }

    void Finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            finished_ = true;
        }
        cv_.notify_all();
    }

    bool WriteNext(std::string & output) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return !events_.empty() || finished_; });
        if (!events_.empty()) {
            output = std::move(events_.front());
            events_.pop_front();
            return true;
        }
        return false;
    }

    void Join() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::thread worker_;

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> events_;
    bool finished_ = false;
};

std::string BuildSseData(const std::string & payload) {
    return "data: " + payload + "\n\n";
}

std::string BuildChatChunk(
    std::string_view id,
    std::string_view model,
    std::string_view content,
    bool include_role,
    bool is_final) {
    Json chunk;
    chunk["id"] = id;
    chunk["object"] = "chat.completion.chunk";
    chunk["created"] = CurrentUnixSeconds();
    chunk["model"] = model;

    Json choice;
    choice["index"] = 0;
    choice["finish_reason"] = is_final ? Json("stop") : Json(nullptr);
    choice["delta"] = Json::object();
    if (include_role) {
        choice["delta"]["role"] = "assistant";
    }
    if (!content.empty()) {
        choice["delta"]["content"] = content;
    }

    chunk["choices"] = Json::array({choice});
    return chunk.dump();
}

Json BuildChatCompletionResponse(
    std::string_view id,
    std::string_view model,
    const AsrRunResult & result) {
    Json response;
    response["id"] = id;
    response["object"] = "chat.completion";
    response["created"] = CurrentUnixSeconds();
    response["model"] = model;

    Json choice;
    choice["index"] = 0;
    choice["finish_reason"] = "stop";
    choice["message"] = Json::object({
        {"role", "assistant"},
        {"content", result.text},
    });
    response["choices"] = Json::array({choice});
    response["usage"] = Json::object({
        {"prompt_tokens", 0},
        {"completion_tokens", result.text_tokens},
        {"total_tokens", result.text_tokens},
    });
    return response;
}

/* Helper: check if the string starts with the given UTF-8 encoded
 * string literal prefix.  len is the byte length of the prefix. */
static bool StartsWithUtf8(std::string_view s, const char * prefix, std::size_t len) {
    if (s.size() < len) return false;
    return s.compare(0, len, prefix, len) == 0;
}

/* Helper: check if the first UTF-8 character of s matches one of the
 * leading whitespace/punctuation characters we want to trim.
 * Returns the byte length to erase if a match, 0 otherwise. */
static std::size_t TrimLeadingCharLen(std::string_view s) {
    if (s.empty()) return 0;
    /* ASCII space or comma. */
    if (s[0] == ' ' || s[0] == ',') return 1;
    /* CJK full-width comma '，' = 0xEF 0xBC 0x8C (3 bytes). */
    if (StartsWithUtf8(s, "\xEF\xBC\x8C", 3)) return 3;
    /* CJK enumeration comma '、' = 0xE3 0x80 0x81 (3 bytes). */
    if (StartsWithUtf8(s, "\xE3\x80\x81", 3)) return 3;
    return 0;
}

/* §7.2 #1: Find the last terminal sentence boundary that lies
 * strictly BEFORE the text end.  Returns the byte offset of the
 * character AFTER the punctuation (i.e., the start of the tail).
 * Returns 0 if no mid-text boundary exists (only boundary-at-end
 * or no boundary at all — entire text is uncertain). */
static std::size_t FindLastMidTextBoundary(std::string_view text) {
    std::size_t last_boundary = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        std::size_t boundary_after = 0;
        /* ASCII punctuation. */
        if (c == '?' || c == '!') {
            boundary_after = i + 1;
        }
        /* English period: only terminal if followed by space, newline, or end. */
        else if (c == '.') {
            if (i + 1 >= text.size() ||
                static_cast<unsigned char>(text[i + 1]) == ' ' ||
                static_cast<unsigned char>(text[i + 1]) == '\n' ||
                static_cast<unsigned char>(text[i + 1]) == ',' ||
                static_cast<unsigned char>(text[i + 1]) == '!') {
                boundary_after = i + 1;
            }
        }
        /* CJK full-width punctuation: ？ ！ 。 (U+FF1F / U+FF01 / U+3002) */
        else if (i + 2 < text.size()) {
            unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if ((c == 0xEF && c1 == 0xBC && c2 == 0x9F) ||   // ？ U+FF1F
                (c == 0xEF && c1 == 0xBC && c2 == 0x81) ||   // ！ U+FF01
                (c == 0xE3 && c1 == 0x80 && c2 == 0x82)) {    // 。 U+3002
                boundary_after = i + 3;
            }
        }
        /* Only accept boundaries that leave at least 1 byte of tail. */
        if (boundary_after > 0 && boundary_after < text.size()) {
            last_boundary = boundary_after;
        }
    }
    return last_boundary;
}

/* Check if text ends with sentence-ending punctuation.
 * Returns true if the last character(s) are a terminal marker
 * (. ? !  。 ？ ！).  Used to distinguish "boundary at end"
 * (text is complete — no tail needed) from "no boundary at all"
 * (text is incomplete — carry full audio as tail). */
static bool EndsWithBoundary(std::string_view text) {
    if (text.empty()) return false;
    /* ASCII terminal punctuation at end. */
    char last = text.back();
    if (last == '.' || last == '?' || last == '!') {
        return true;
    }
    /* CJK full-width punctuation:  。(U+3002) ？(U+FF1F) ！(U+FF01) */
    if (text.size() >= 3) {
        unsigned char c2 = static_cast<unsigned char>(text[text.size() - 1]);
        unsigned char c1 = static_cast<unsigned char>(text[text.size() - 2]);
        unsigned char c0 = static_cast<unsigned char>(text[text.size() - 3]);
        if ((c0 == 0xE3 && c1 == 0x80 && c2 == 0x82) ||    // 。
            (c0 == 0xEF && c1 == 0xBC && c2 == 0x9F) ||    // ？
            (c0 == 0xEF && c1 == 0xBC && c2 == 0x81)) {    // ！
            return true;
        }
    }
    return false;
}

struct RealtimeSession {
    std::mutex mu;
    std::string id;
    std::string model;
    std::string language;
    std::vector<float> samples;
    std::vector<float> full_audio;       /* untrimmed — for post-stop reconciliation */
    std::size_t total_samples = 0;
    std::size_t decoded_samples = 0;
    std::size_t retained_sample_offset = 0;
    RealtimeTextState text_state;
    RealtimeDisplayState display_state;
    RealtimeDisplaySnapshot display_snapshot;
    std::string text;
    std::string stable_text;
    std::string partial_text;
    /* VAD-segmented mode: committed sentences, append-only.  Each
     * completed segment's text is pushed here.  The UI renders this
     * as a running transcript; live_stable_text holds the in-flight
     * segment text (or the most recent committed one after the
     * segment is finalized).  This replaces the rolling-decoder
     * partial/revision model with a sentence-bounded one. */
    std::vector<std::string> segments_text;
    /* Per-segment cumulative sample position (monotonic, 16kHz).
     * segments_sample_positions[i] is the end position of segments_text[i].
     * Populated in AsrWorkerLoop at commit time. */
    std::vector<std::size_t> segments_sample_positions;
    std::size_t segment_cumulative_samples = 0;
    /* Uncommitted tail text from the last ASR segment (mid-text boundary
     * case).  When the final segment's tail audio is lost (no next segment
     * to re-decode), this text is force-finalized into segments_text. */
    std::string tail_text;
/* VAD candidate buffer — tentative text shown to SSE client. */
     std::vector<std::string> candidates;
     std::size_t sse_last_candidate_count = 0;
     /* Pipeline config stored on session. */
     bool gpuPipeline = false;
     std::uint64_t gpuSessionId = 0;
     ServerAsrFacade * facade = nullptr;
    double current_segment_audio_sec = 0.0;
    /* Audio ingress diagnostic.  Set on every chunk POST, surfaced via
     * /api/realtime/audio_diag so the UI can show "did the server
     * actually receive my audio". */
    float last_ingress_peak = 0.0f;
    float last_ingress_rms = 0.0f;
    float max_ingress_peak = 0.0f;
    uint64_t ingress_chunks = 0;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
    bool worker_done = false;
    bool finalized = false;
    std::string error;
    std::unique_ptr<struct RealtimeLiveWorker> live_worker;
    /* SSE notification.  The ASR worker calls sse_cv.notify_all()
     * after committing a new segment and on finalize.  The SSE
     * endpoint (/api/realtime/stream) waits on this CV instead of
     * polling, eliminating the per-client /status poll.  Paired with
     * mu (always lock mu before waiting/poking this CV). */
    std::condition_variable sse_cv;
    /* Number of segments already pushed to the SSE client.  The
     * SSE stream uses this to diff and only emit incremental
     * segments.  Updated by the SSE writer; read by it too. */
    std::size_t sse_last_segment_count = 0;
    /* Post-stop full-audio retranscription result.  The background
     * retranscription thread sets these after re-decoding the full
     * audio, then notifies sse_cv.  The SSE loop (if still active)
     * delivers this to the client before sending [DONE].
     *
     * reconcile_revised is true when the retranscribed text differs
     * from the original VAD-based segments text. */
    std::string reconcile_text;
    bool reconcile_ready = false;
    bool reconcile_revised = false;
    /* Monotonic version counter incremented on every partial/live text
     * update.  The SSE handler uses this to detect quickly-changing
     * live text and push it to the client without waiting for a full
     * VAD segment commit.  See `ApplyChunkRealtimeCommit`. */
    std::uint64_t partial_version = 0;
    /* Tail carry-over: audio that corresponds to text AFTER the last
     * sentence boundary in a committed segment.  This audio is
     * prepended to the next segment so the ASR model has full context
     * for the incomplete sentence. */
    std::vector<float> tail_audio;
};

struct RealtimeSessionSnapshot {
    std::string id;
    std::string model;
    std::string language;
    std::size_t total_samples = 0;
    std::size_t decoded_samples = 0;
    std::size_t retained_sample_count = 0;
    std::size_t retained_sample_offset = 0;
    RealtimeDisplaySnapshot display_snapshot;
    std::string text;
    std::string stable_text;
    std::string partial_text;
    std::vector<std::string> segments_text;
    std::vector<std::size_t> segments_sample_positions;
    std::size_t segment_cumulative_samples = 0;
    std::string tail_text;
    std::vector<std::string> candidates;
    double current_segment_audio_sec = 0.0;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
    bool finalized = false;
    /* Audio ingress diagnostic, mirrored from RealtimeSession so the
     * /status endpoint can emit them without going back to the
     * session.  The snapshot is built under session->mu so the read
     * is consistent with the rest of the fields. */
    float last_ingress_peak = 0.0f;
    float last_ingress_rms = 0.0f;
    float max_ingress_peak = 0.0f;
    std::uint64_t ingress_chunks = 0;
    std::string error;
};

/* ========================================================================
 * Producer-Consumer primitives for VAD → ASR pipeline
 * ========================================================================
 * 
 * Design: the VAD facade is a PRODUCER of AudioSegment, the ASR worker
 * is a CONSUMER.  They communicate through a bounded SegmentQueue with
 * proper condition-variable signalling and a poison-pill EOF marker.
 *
 *   [VAD facade thread]                 [ASR worker thread]
 *        |                                    |
 *   queue_.Push(seg)  ───blocks if full──>    |
 *   (backpressure: producer slows down if     |
 *    consumer is slow, never loses audio)      |
 *        |                                    |
 *        |       queue_.Pop(&seg)  ───blocks if empty───
 *        |                                    |
 *        |                              qwen_transcribe_audio()
 *        |                                    |
 *        |       poison-pill eof_segment      |
 *        |       (is_eof_terminator=true)     |
 *        |       consumer sees it, exits      |
 *        v                                    v
 *
 * Lifecycle invariants:
 *   1. Each session has its OWN SegmentQueue (per-session isolation).
 *   2. Push() blocks if queue is full → producer naturally backpressures.
 *   3. Pop() blocks if queue is empty → consumer sleeps, no busy wait.
 *   4. Close() marks the queue as closed; Pop() after that drains
 *      remaining items, then returns false → consumer exits.
 *   5. The poison-pill AudioSegment{is_eof_terminator=true} is a
 *      belt-and-suspenders EOF marker in case Close() races with
 *      a still-buffered Push.
 *   6. atomic stop_requested on the worker allows prompt cancellation
 *      even if both threads are blocked in cv.wait.
 */

struct AudioSegment {
    std::vector<float> samples;     /* raw float PCM, 16 kHz mono */
    std::string commit_reason;      /* "vad_silence" | "10s_soft_cap" | "eof" */
    bool is_eof_terminator = false; /* poison pill: consumer should exit */

    /* Phase 2A: observability fields — help diagnose missing text. */
    std::uint64_t seq = 0;          /* per-session monotonic sequence number */
    bool first_segment = false;     /* true for the first segment of this session */
    std::string endpoint_mode;      /* "legacy" | "cap_only" */
    int64_t total_samples_at_push = 0; /* cumulative samples consumed at push time */
    double queued_audio_sec = 0.0;  /* audio duration of this segment */

    /* Phase 2A: boundary context fields — reserved for Phase 3. */
    int left_context_samples = 0;
    int right_context_samples = 0;
    int boundary_overlap_samples = 0;
    bool truncated_left = false;
    bool truncated_right = false;

    /* Phase 2A: ASR performance fields — filled by ASR worker after decode. */
    double asr_total_ms = 0.0;
    double asr_encode_ms = 0.0;
    double asr_decode_ms = 0.0;
    int asr_tokens = 0;
    bool asr_empty = false;
    std::string asr_text;
};

class SegmentQueue {
 public:
     /* Producer: enqueue a segment.  Blocks if the queue is at capacity
      * (backpressure).  Returns false if the queue was closed before
      * the segment could be pushed — the caller should drop the segment
      * and treat the producer as terminated. */
     bool Push(AudioSegment seg) {
         const auto wait_start = std::chrono::steady_clock::now();
         {
             std::unique_lock<std::mutex> lock(mu_);
             cv_not_full_.wait(lock, [this] {
                 return closed_ || queue_.size() < kMaxDepth;
             });
             const auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - wait_start).count();
             if (wait_ms > 0) {
                 stats_.wait_full_count.fetch_add(1, std::memory_order_relaxed);
                 stats_.wait_full_ms_total.fetch_add(wait_ms, std::memory_order_relaxed);
                 /* Atomic max update. */
                 uint64_t cur = stats_.wait_full_ms_max.load(std::memory_order_relaxed);
                 while (static_cast<uint64_t>(wait_ms) > cur &&
                        !stats_.wait_full_ms_max.compare_exchange_weak(cur,
                                                                       wait_ms,
                                                                       std::memory_order_relaxed)) {}
             }
             if (closed_) {
                 stats_.push_failed_closed.fetch_add(1, std::memory_order_relaxed);
                 return false;
             }
             const std::size_t new_depth = queue_.size() + 1;
             uint64_t cur = stats_.max_depth_seen.load(std::memory_order_relaxed);
             while (new_depth > cur &&
                    !stats_.max_depth_seen.compare_exchange_weak(cur,
                                                                  static_cast<uint64_t>(new_depth),
                                                                  std::memory_order_relaxed)) {}
             queue_.push_back(std::move(seg));
             stats_.push_total.fetch_add(1, std::memory_order_relaxed);
         }
         cv_not_empty_.notify_one();
         return true;
     }

     /* Consumer: dequeue a segment.  Blocks while the queue is empty.
      * Returns true if a segment was popped.  Returns false ONLY when
      * the queue is closed AND empty (i.e., the consumer should exit). */
     bool Pop(AudioSegment * out) {
         std::unique_lock<std::mutex> lock(mu_);
         cv_not_empty_.wait(lock, [this] {
             return closed_ || !queue_.empty();
         });
         if (queue_.empty()) {
             /* closed_ must be true here (predicate guarantees it) */
             return false;
         }
         *out = std::move(queue_.front());
         queue_.pop_front();
         stats_.pop_total.fetch_add(1, std::memory_order_relaxed);
         lock.unlock();
         cv_not_full_.notify_one();
         return true;
     }

     /* Phase 2.1 §3.6: log queue stats for diagnostic output. */
     void LogStats(const char * sid) const {
         std::fprintf(stderr,
             "SEGQ-stats sid=%s push_total=%lu pop_total=%lu push_failed_closed=%lu "
             "wait_full_count=%lu wait_full_max=%lums max_depth_seen=%lu\n",
             sid,
             static_cast<unsigned long>(stats_.push_total.load(std::memory_order_relaxed)),
             static_cast<unsigned long>(stats_.pop_total.load(std::memory_order_relaxed)),
             static_cast<unsigned long>(stats_.push_failed_closed.load(std::memory_order_relaxed)),
             static_cast<unsigned long>(stats_.wait_full_count.load(std::memory_order_relaxed)),
             static_cast<unsigned long>(stats_.wait_full_ms_max.load(std::memory_order_relaxed)),
             static_cast<unsigned long>(stats_.max_depth_seen.load(std::memory_order_relaxed)));
     }

    /* Mark the queue as closed.  After this call:
     *   - Push() returns false immediately
     *   - Pop() drains remaining items, then returns false
     *   - All blocked cv.wait() calls wake up
     * Safe to call multiple times (idempotent). */
    void Close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    std::size_t Size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

    /* Bound on the in-flight queue.  8 segments × ~3 s each ≈ 24 s of
     * buffered audio, which is enough to keep the ASR worker busy
     * during natural pauses without unbounded growth.  Backpressure
     * kicks in beyond this. */
    static constexpr std::size_t kMaxDepth = 8;

  private:
    struct SegmentQueueStats {
        std::atomic<uint64_t> push_total{0};
        std::atomic<uint64_t> push_failed_closed{0};
        std::atomic<uint64_t> pop_total{0};
        std::atomic<uint64_t> wait_full_count{0};
        std::atomic<uint64_t> wait_full_ms_total{0};
        std::atomic<uint64_t> wait_full_ms_max{0};
        std::atomic<uint64_t> max_depth_seen{0};
    };

    mutable std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<AudioSegment> queue_;
    bool closed_ = false;
    SegmentQueueStats stats_;
};

/* ========================================================================
 * PreRollBuffer — fixed-size ring buffer of the last N samples of audio.
 *
 * Used by the VAD facade to keep a small "look-back" of audio BEFORE the
 * first VAD-detected speech frame, so the emitted AudioSegment can
 * include the leading consonants of the utterance (e.g., the "n" in
 * "你好" or the "w" in "world") that would otherwise be clipped at the
 * speech-start boundary.
 *
 * Thread-safety: NOT thread-safe.  Owned and accessed only by the
 * VadFacadeLoop thread.  Live audio is drained under live->mu but the
 * PreRollBuffer is updated after the mutex is released, so the buffer
 * itself never participates in cross-thread synchronization.
 *
 * Capacity: capacity_samples (e.g., 4000 = 250 ms at 16 kHz mono).
 * When full, oldest samples are overwritten in-place.
 * ========================================================================
 */
class PreRollBuffer {
public:
    explicit PreRollBuffer(int capacity_samples)
        : buffer_(static_cast<std::size_t>(capacity_samples), 0.0f),
          capacity_(static_cast<std::size_t>(capacity_samples)),
          write_idx_(0),
          filled_(0) {}

    /* Append `n` samples.  Wraps around if capacity is reached. */
    void Push(const float * samples, int n) {
        if (n <= 0 || capacity_ == 0) return;
        for (int i = 0; i < n; ++i) {
            buffer_[write_idx_] = samples[i];
            write_idx_ = (write_idx_ + 1) % capacity_;
        }
        filled_ = std::min(capacity_, filled_ + static_cast<std::size_t>(n));
    }

    /* Copy the buffered samples in chronological order into `out`.
     * `out` is cleared first.  No-op if buffer is empty. */
    void Snapshot(std::vector<float> * out) {
        out->clear();
        if (filled_ == 0) return;
        out->reserve(filled_);
        const std::size_t start = (filled_ < capacity_) ? 0 : write_idx_;
        for (std::size_t i = 0; i < filled_; ++i) {
            out->push_back(buffer_[(start + i) % capacity_]);
        }
    }

    std::size_t Size() const { return filled_; }
    std::size_t Capacity() const { return capacity_; }

private:
    std::vector<float> buffer_;
    std::size_t capacity_;
    std::size_t write_idx_;
    std::size_t filled_;
};

/* ========================================================================
 * RAII wrapper for per-session Silero VAD instance.
 * Each realtime session owns its own VAD so that LSTM hidden state is
 * not shared across sessions — no cross-session pollution.
 * ======================================================================== */
struct QwenVadDeleter {
    void operator()(qwen_silero_vad_t * vad) const noexcept {
        if (vad) {
            qwen_silero_vad_destroy(vad);
        }
    }
};

using QwenVadPtr = std::unique_ptr<qwen_silero_vad_t, QwenVadDeleter>;

/* Server-level VAD instance for ASR worker audio cutting.
 * This is a separate instance from the facade VADs so that cutting
 * operations don't pollute the facade's LSTM state.  Protected by
 * its own mutex to prevent cross-session state corruption. */
static QwenVadPtr g_cut_vad;
static std::mutex g_cut_vad_mutex;

/* Find first speech onset in [start_offset, end_offset) using the
 * shared cut VAD.  Returns the sample offset of the first speech
 * frame, or start_offset if no speech is found.
 *
 * Thread-safe: caller must NOT hold g_cut_vad_mutex.
 * Each call resets the VAD state so sessions don't interfere. */
static int VadFindSpeechOnset(const float * samples,
                              int start_offset, int end_offset) {
    std::lock_guard<std::mutex> lock(g_cut_vad_mutex);
    qwen_silero_vad_t * v = g_cut_vad.get();
    if (!v || !qwen_silero_vad_is_active(v)) return start_offset;

    /* Reset to clean state — each cut is independent. */
    qwen_silero_vad_reset(v);
    const int VAD_FRAME = 512;
    int result = start_offset;
    for (int off = (start_offset / VAD_FRAME) * VAD_FRAME;
         off + VAD_FRAME <= end_offset;
         off += VAD_FRAME) {
        float prob = 0;
        if (qwen_silero_vad_process(v, samples + off, VAD_FRAME, &prob) == 0
            && prob > 0.5f) {
            result = off;
            break;
        }
    }
    return result;
}

/* The realtime session's worker is now a 2-thread producer-consumer
 * pair instead of the old single-thread "VAD+ASR tight loop".  Each
 * session has its own pair, fully isolated from other sessions.
 *
 * NOTE: the host-capture path (StartHostCaptureLiveWorker) still uses
 * the old single-thread "stream_live" C API, so we keep a `thread`
 * field for it.  Once host-capture is also migrated to producer-
 * consumer, the `thread` field can be removed. */
struct RealtimeLiveWorker {
    qwen_live_audio_t live{};      /* HTTP writes here, VAD reads here */
    SegmentQueue segment_queue;     /* VAD pushes, ASR pops */
    std::thread thread;             /* host-capture single-thread path */
    std::thread vad_thread;         /* producer: VAD facade */
    std::thread asr_thread;         /* consumer: ASR worker */
    std::atomic<bool> stop_requested{false};  /* set by /api/realtime/stop */

    /* Per-session VAD instance.  Owned by the worker; only the VAD
     * facade thread accesses it.  If session_vad is null, the facade
     * falls back to the shared model VAD (protected by vad_mu). */
    QwenVadPtr session_vad;
    bool session_vad_active = false;
    bool session_vad_fallback_shared = false;

    /* Phase 2A: segment sequencing and dump observability. */
    std::atomic<uint64_t> next_segment_seq{0};
    std::atomic<bool> first_segment_queued{false};
    std::atomic<bool> first_segment_emitted{false};
    std::atomic<uint64_t> dumped_segment_count{0};
    std::atomic<uint64_t> dumped_sample_count{0};
    /* Cumulative count of samples the VAD has consumed since session
     * start.  Updated by the VAD on every consume (both the main
     * poll loop and the stop_drain path).  The snapshot reads this
     * (NOT live->decoded_cursor, which is reset to 0 by
     * TrimConsumedLiveAudio on every memmove) to populate
     * session->decoded_samples.  Atomic so the snapshot can read it
     * without taking live->mu. */
    std::atomic<int64_t> cumulative_decoded_samples{0};
    bool live_ready = false;
    bool vad_thread_joined = false;
    bool asr_thread_joined = false;
};

struct ServerMetrics {
    std::atomic<std::uint64_t> offline_requests{0};
    std::atomic<std::uint64_t> async_jobs_submitted{0};
    std::atomic<std::uint64_t> async_job_cleanup_runs{0};
    std::atomic<std::uint64_t> async_jobs_evicted{0};
    std::atomic<std::uint64_t> chat_requests{0};
    std::atomic<std::uint64_t> realtime_sessions_started{0};
    std::atomic<std::uint64_t> realtime_decode_runs{0};
    std::atomic<std::uint64_t> realtime_finalizations{0};
    std::atomic<std::uint64_t> host_capture_sessions_started{0};
};

/* ========================================================================
 * Phase 2A: Realtime segment dump configuration.
 * Debug-only; disabled by default in production.
 * ======================================================================== */
struct DumpConfig {
    bool enabled = false;
    std::string dir;
    uint64_t max_segments = 200;
    uint64_t max_seconds = 600;

    static DumpConfig FromEnv() {
        DumpConfig cfg;
        cfg.enabled = getenv("QASR_DUMP_REALTIME_SEGMENTS") != nullptr &&
                      atoi(getenv("QASR_DUMP_REALTIME_SEGMENTS")) != 0;
        if (cfg.enabled) {
            const char * dir = getenv("QASR_DUMP_REALTIME_DIR");
            cfg.dir = dir ? dir : "./realtime_dumps";
            const char * ms = getenv("QASR_DUMP_REALTIME_MAX_SEGMENTS");
            if (ms) cfg.max_segments = std::max(1u, std::min(10000u, (unsigned int)std::stoul(ms)));
            const char * mcs = getenv("QASR_DUMP_REALTIME_MAX_SECONDS");
            if (mcs) cfg.max_seconds = std::max(1u, std::min(3600u, (unsigned int)std::stoul(mcs)));
        }
        return cfg;
    }
};

/* Write float PCM to a 16-bit mono WAV file at 16 kHz.
 * Returns Status::ok() on success, or an error status. */
Status WriteFloatMono16kWav(const fs::path & path,
                            const float * samples, int n_samples) {
    if (!samples || n_samples <= 0) return OkStatus();

    /* WAV header: 44 bytes RIFF/WAVE, PCM 16-bit mono 16000 Hz */
    const int data_bytes = n_samples * 2;
    const int file_size = 44 + data_bytes;
    const int byte_rate = 16000 * 2;  /* sample_rate * channels * bits/8 */
    const int block_align = 2;  /* channels * bits/8 */

    std::vector<char> header(44);
    /* RIFF header */
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    std::memcpy(&header[4], &file_size, 4);  /* file_size - 8, little-endian */
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    /* fmt chunk */
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    const int fmt_size = 16;
    std::memcpy(&header[16], &fmt_size, 4);
    std::int16_t audio_format = 1;  /* PCM */
    std::int16_t num_channels = 1;
    std::int32_t sample_rate = 16000;
    std::memcpy(&header[20], &audio_format, 2);
    std::memcpy(&header[22], &num_channels, 2);
    std::memcpy(&header[24], &sample_rate, 4);
    std::memcpy(&header[28], &byte_rate, 4);
    std::memcpy(&header[32], &block_align, 2);
    std::int16_t bits_per_sample = 16;
    std::memcpy(&header[34], &bits_per_sample, 2);
    /* data chunk */
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    std::memcpy(&header[40], &data_bytes, 4);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return Status(StatusCode::kInternal, "cannot open WAV file: " + path.string());

    ofs.write(header.data(), header.size());

    /* Write PCM16 samples */
    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float v = samples[i];
        if (v < -1.0f) v = -1.0f;
        if (v > 1.0f) v = 1.0f;
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }
    ofs.write(reinterpret_cast<const char *>(pcm.data()), data_bytes);

    if (!ofs) return Status(StatusCode::kInternal, "WAV write failed: " + path.string());
    return OkStatus();
}

/* ========================================================================
 * Phase 2B: Cap-only endpoint mode + latency guardrail policy.
 * ======================================================================== */
enum class EndpointMode {
    kLegacy,
    kCapOnly,
};

enum class SegmentEndpointKind {
    kFirst,
    kNormal,
};

struct CapOnlyEndpointPolicy {
    /* Existing parameters — MUST remain unchanged in Phase 2. */
    int stable_silence_ms = 1500;
    int min_emit_ms = 1800;
    int pending_merge_ms = 1200;

    /* New cap parameters.  First cap protects against long first-segment
     * wait during continuous speech.  Normal cap protects subsequent
     * segments.  Grace + valley avoid mid-speech cutting. */
    int first_latency_cap_ms = 20000;
    int normal_latency_cap_ms = 15000;
    int cap_grace_ms = 800;
    int cap_valley_silence_ms = 256;

    static CapOnlyEndpointPolicy FromEnv() {
        CapOnlyEndpointPolicy p;
        const char * fl = getenv("QASR_FIRST_LATENCY_CAP_MS");
        if (fl) p.first_latency_cap_ms = std::min(60000, std::max(1000, std::stoi(fl)));
        const char * nl = getenv("QASR_NORMAL_LATENCY_CAP_MS");
        if (nl) p.normal_latency_cap_ms = std::min(120000, std::max(3000, std::stoi(nl)));
        const char * gr = getenv("QASR_LATENCY_CAP_GRACE_MS");
        if (gr) p.cap_grace_ms = std::max(100, std::stoi(gr));
        const char * vl = getenv("QASR_LATENCY_CAP_VALLEY_MS");
        if (vl) p.cap_valley_silence_ms = std::max(64, std::stoi(vl));
        return p;
    }
};

static EndpointMode ParseEndpointMode() {
    const char * env = getenv("QASR_ENDPOINT_MODE");
    if (env && std::strcmp(env, "cap_only") == 0) {
        return EndpointMode::kCapOnly;
    }
    return EndpointMode::kLegacy;
}

struct HostCaptureSession {
    std::string id;
    std::string backend;
    std::string device;
    std::vector<float> samples;
    std::vector<float> full_audio;
    std::size_t total_samples = 0;
    std::size_t decoded_samples = 0;
    std::size_t retained_sample_offset = 0;
    RealtimeTextState text_state;
    RealtimeDisplayState display_state;
    RealtimeDisplaySnapshot display_snapshot;
    std::string text;
    std::string stable_text;
    std::string partial_text;
    std::string error;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
    bool finalized = false;
    bool active = true;
    bool stop_requested = false;
    bool worker_done = false;
    std::unique_ptr<RealtimeLiveWorker> live_worker;
    std::uint64_t partial_version = 0;
    std::condition_variable sse_cv;
#if defined(_WIN32)
    HANDLE child_process = INVALID_HANDLE_VALUE;
    HANDLE read_handle = INVALID_HANDLE_VALUE;
#else
    pid_t child_pid = -1;
    int read_fd = -1;
#endif
    std::thread reader;
    std::mutex mu;
};

#if defined(_WIN32)
Status SpawnCaptureProcess(
    const std::vector<std::string> & argv,
    HANDLE * child_process,
    HANDLE * read_handle) {
    if (child_process == nullptr || read_handle == nullptr) {
        return Status(StatusCode::kInvalidArgument, "capture outputs must not be null");
    }

    SECURITY_ATTRIBUTES sa = {};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE pipe_read = INVALID_HANDLE_VALUE;
    HANDLE pipe_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&pipe_read, &pipe_write, &sa, 0)) {
        return Status(StatusCode::kInternal, "CreatePipe failed");
    }
    /* Prevent the read end from being inherited by the child. */
    SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0);

    /* Build a single command line string. */
    std::string cmdline;
    for (std::size_t i = 0; i < argv.size(); i++) {
        if (i > 0) cmdline.push_back(' ');
        /* Simple quoting: wrap each arg in double quotes and escape embedded
           double-quotes.  Sufficient for ffmpeg argument values. */
        cmdline.push_back('"');
        for (const char ch : argv[i]) {
            if (ch == '"') cmdline += "\\\"";
            else cmdline.push_back(ch);
        }
        cmdline.push_back('"');
    }

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe_write;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessA(
        nullptr,
        &cmdline[0],
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi);
    CloseHandle(pipe_write);

    if (!ok) {
        CloseHandle(pipe_read);
        return Status(StatusCode::kInternal, "CreateProcess failed for: " + cmdline);
    }

    CloseHandle(pi.hThread);
    *child_process = pi.hProcess;
    *read_handle = pipe_read;
    return OkStatus();
}
#else
Status SpawnCaptureProcess(
    const std::vector<std::string> & argv,
    pid_t * child_pid,
    int * read_fd) {
    if (child_pid == nullptr || read_fd == nullptr) {
        return Status(StatusCode::kInvalidArgument, "capture outputs must not be null");
    }
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        return Status(StatusCode::kInternal, "pipe() failed");
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return Status(StatusCode::kInternal, "fork() failed");
    }

    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);

        std::vector<char *> raw_argv;
        raw_argv.reserve(argv.size() + 1);
        for (const std::string & value : argv) {
            raw_argv.push_back(const_cast<char *>(value.c_str()));
        }
        raw_argv.push_back(nullptr);
        execvp(raw_argv[0], raw_argv.data());
        _exit(127);
    }

    close(fds[1]);
    *child_pid = pid;
    *read_fd = fds[0];
    return OkStatus();
}
#endif

Status BuildCaptureCommand(
    std::string backend,
    const std::string & device,
    std::vector<std::string> * argv,
    std::string * selected_backend) {
    if (argv == nullptr || selected_backend == nullptr) {
        return Status(StatusCode::kInvalidArgument, "capture outputs must not be null");
    }

    for (char & ch : backend) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    if (backend.empty()) {
        backend = "auto";
    }

    const bool have_arecord = CommandExists("arecord");
    const bool have_parec = CommandExists("parec");
    const bool have_ffmpeg = FfmpegAvailable();
    if (backend == "auto") {
#if defined(__linux__)
        if (have_arecord) {
            backend = "arecord";
        } else if (have_parec) {
            backend = "parec";
        } else if (have_ffmpeg) {
            backend = "ffmpeg";
        }
#else
        if (have_ffmpeg) {
            backend = "ffmpeg";
        }
#endif
        if (backend == "auto") {
            return Status(StatusCode::kFailedPrecondition,
                "no capture backend available (install ffmpeg, arecord, or parec)");
        }
    }

    if (backend == "arecord") {
        if (!have_arecord) {
            return Status(StatusCode::kFailedPrecondition, "arecord is not available");
        }
        *argv = {"arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"};
        if (!device.empty()) {
            argv->push_back("-D");
            argv->push_back(device);
        }
        *selected_backend = "arecord";
        return OkStatus();
    }

    if (backend == "parec") {
        if (!have_parec) {
            return Status(StatusCode::kFailedPrecondition, "parec is not available");
        }
        *argv = {"parec", "--raw", "--rate=16000", "--channels=1", "--format=s16le"};
        if (!device.empty()) {
            argv->push_back("--device=" + device);
        }
        *selected_backend = "parec";
        return OkStatus();
    }

    if (backend == "ffmpeg") {
        if (!have_ffmpeg) {
            return Status(StatusCode::kFailedPrecondition, "ffmpeg is not available");
        }
        *argv = {"ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"};
#if defined(__APPLE__)
        argv->insert(argv->end(), {"-f", "avfoundation", "-i",
            device.empty() ? ":default" : (":" + device)});
#elif defined(__linux__)
        if (CommandExists("pulseaudio") || CommandExists("pipewire-pulse")) {
            argv->insert(argv->end(), {"-f", "pulse", "-i",
                device.empty() ? "default" : device});
        } else {
            argv->insert(argv->end(), {"-f", "alsa", "-i",
                device.empty() ? "default" : device});
        }
#elif defined(_WIN32)
        argv->insert(argv->end(), {"-f", "dshow", "-i",
            device.empty() ? "audio=virtual-audio-capturer" : ("audio=" + device)});
#else
        argv->insert(argv->end(), {"-f", "alsa", "-i",
            device.empty() ? "default" : device});
#endif
        argv->insert(argv->end(), {"-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"});
        *selected_backend = "ffmpeg";
        return OkStatus();
    }

    return Status(StatusCode::kInvalidArgument, "unsupported capture backend: " + backend);
}

void JoinRealtimeLiveWorker(RealtimeLiveWorker * worker);

void StopHostCaptureSession(const std::shared_ptr<HostCaptureSession> & capture) {
    if (!capture) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(capture->mu);
        capture->stop_requested = true;
    }

#if defined(_WIN32)
    /* Terminate the child process; this closes the pipe write-end from the
       child side, which unblocks the reader thread's ReadFile(). */
    if (capture->child_process != INVALID_HANDLE_VALUE) {
        TerminateProcess(capture->child_process, 1);
    }
    if (capture->reader.joinable()) {
        capture->reader.join();
    }
    if (capture->read_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(capture->read_handle);
        capture->read_handle = INVALID_HANDLE_VALUE;
    }
    if (capture->child_process != INVALID_HANDLE_VALUE) {
        WaitForSingleObject(capture->child_process, 2000);
        CloseHandle(capture->child_process);
        capture->child_process = INVALID_HANDLE_VALUE;
    }
#else
    // Send SIGTERM first, then SIGKILL as forceful fallback.
    // On macOS, ffmpeg -f avfoundation may not respond to SIGTERM if stuck
    // acquiring the audio device.  SIGKILL guarantees the child dies, which
    // closes the pipe write-end and unblocks the reader thread's read().
    if (capture->child_pid > 0) {
        kill(capture->child_pid, SIGTERM);
        kill(capture->child_pid, SIGKILL);
    }
    if (capture->reader.joinable()) {
        capture->reader.join();
    }
    // Close the read-end only after the reader thread has exited to avoid
    // closing an fd that another thread is actively read()ing from.
    if (capture->read_fd >= 0) {
        close(capture->read_fd);
        capture->read_fd = -1;
    }
    if (capture->child_pid > 0) {
        int status = 0;
        waitpid(capture->child_pid, &status, 0);
        capture->child_pid = -1;
    }
#endif

    RealtimeLiveWorker * worker = nullptr;
    {
        std::lock_guard<std::mutex> lock(capture->mu);
        capture->active = false;
        capture->finalized = true;
        worker = capture->live_worker.get();
    }
    if (worker) {
        JoinRealtimeLiveWorker(worker);
        std::lock_guard<std::mutex> lock(capture->mu);
        capture->live_worker.reset();
    }
}

void LockLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return;
    }
#if defined(_WIN32)
    EnterCriticalSection(&live->mutex);
#else
    pthread_mutex_lock(&live->mutex);
#endif
}

void UnlockLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return;
    }
#if defined(_WIN32)
    LeaveCriticalSection(&live->mutex);
#else
    pthread_mutex_unlock(&live->mutex);
#endif
}

void SignalLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return;
    }
#if defined(_WIN32)
    WakeConditionVariable(&live->cond);
#else
    pthread_cond_signal(&live->cond);
#endif
}

Status InitializeManualLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return Status(StatusCode::kInvalidArgument, "live audio must not be null");
    }

    std::memset(live, 0, sizeof(*live));
#if defined(_WIN32)
    InitializeCriticalSection(&live->mutex);
    InitializeConditionVariable(&live->cond);
    live->thread = nullptr;
#else
    if (pthread_mutex_init(&live->mutex, nullptr) != 0) {
        return Status(StatusCode::kInternal, "pthread_mutex_init failed");
    }
    if (pthread_cond_init(&live->cond, nullptr) != 0) {
        pthread_mutex_destroy(&live->mutex);
        return Status(StatusCode::kInternal, "pthread_cond_init failed");
    }
#endif
    return OkStatus();
}

void DestroyManualLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return;
    }
    std::free(live->samples);
    live->samples = nullptr;
    live->n_samples = 0;
    live->capacity = 0;
    live->sample_offset = 0;
    live->eof = 0;
#if defined(_WIN32)
    DeleteCriticalSection(&live->mutex);
    live->thread = nullptr;
#else
    pthread_cond_destroy(&live->cond);
    pthread_mutex_destroy(&live->mutex);
#endif
}

Status AppendManualLiveAudio(qwen_live_audio_t * live, const float * samples, std::size_t n_samples) {
    if (live == nullptr || samples == nullptr || n_samples == 0U) {
        return Status(StatusCode::kInvalidArgument, "live audio samples are required");
    }
    if (n_samples > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        return Status(StatusCode::kOutOfRange, "live audio chunk is too large");
    }

    LockLiveAudio(live);
    const int64_t add = static_cast<int64_t>(n_samples);
    if (live->n_samples > std::numeric_limits<int64_t>::max() - add) {
        UnlockLiveAudio(live);
        return Status(StatusCode::kOutOfRange, "live audio buffer would overflow");
    }
    const int64_t required = live->n_samples + add;
    if (required > live->capacity) {
        int64_t new_capacity = live->capacity > 0 ? live->capacity : 32000;
        while (new_capacity < required) {
            if (new_capacity > std::numeric_limits<int64_t>::max() / 2) {
                new_capacity = required;
                break;
            }
            new_capacity *= 2;
        }
        if (new_capacity <= 0 ||
            static_cast<std::uint64_t>(new_capacity) >
                static_cast<std::uint64_t>(SIZE_MAX / sizeof(float))) {
            UnlockLiveAudio(live);
            return Status(StatusCode::kOutOfRange, "live audio buffer is too large");
        }
        float * grown = static_cast<float *>(std::realloc(live->samples, static_cast<std::size_t>(new_capacity) * sizeof(float)));
        if (grown == nullptr) {
            UnlockLiveAudio(live);
            return Status(StatusCode::kInternal, "failed to grow live audio buffer");
        }
        live->samples = grown;
        live->capacity = new_capacity;
    }

    std::memcpy(live->samples + static_cast<std::size_t>(live->n_samples), samples, n_samples * sizeof(float));
    live->n_samples = required;
    SignalLiveAudio(live);
    UnlockLiveAudio(live);
    return OkStatus();
    /* TODO(oom-audit-C1): live->samples growth is now bounded by
     * TrimConsumedLiveAudio() called from VadFacadeLoop after each
     * consume.  With kLiveTrimThresholdSamples = 1.6M (100 s @ 16 kHz
     * = 6.4 MB), worst-case RSS per session is the trim threshold plus
     * the VAD's segment_buffer / pending_buffer caps (10 s soft cap =
     * 640 KB each), i.e. < 8 MB / session, down from 230 MB / h.  See
     * docs/AUDIT_C1.md §4.1. */
}

/* Periodically reclaim the consumed prefix of live->samples so long
 * sessions do not OOM.  The VAD facade's local consumed_samples is
 * the read cursor: everything in [0, consumed_samples) has already
 * been copied into the segment_buffer and processed.  Keeping that
 * prefix around only wastes RSS; the bytes are never read again.
 *
 * Implementation: hold live->mu, memmove the unconsumed tail to the
 * front, decrement n_samples by consumed_samples, reset the caller's
 * local cursor to 0, and try realloc() to shrink the allocation.
 *
 * Threshold: 1.6M samples (100 s @ 16 kHz mono = 6.4 MB).  Trim cost
 * is one memmove of (n_samples - consumed_samples) * 4 bytes.  With
 * the threshold at 6.4 MB and worst case trim of ~13 MB, that is
 * single-digit ms on x86 — well below the 50 ms poll cadence, so
 * it never blocks the VAD facade perceptibly. */
static constexpr int64_t kLiveTrimThresholdSamples = 1600000;  /* 100 s @ 16 kHz */

void TrimConsumedLiveAudio(qwen_live_audio_t * live, int64_t & consumed_samples) {
    if (live == nullptr || consumed_samples < kLiveTrimThresholdSamples) {
        return;
    }
    LockLiveAudio(live);
    const int64_t remaining = live->n_samples - consumed_samples;
    if (remaining > 0) {
        std::memmove(live->samples, live->samples + consumed_samples,
                     static_cast<std::size_t>(remaining) * sizeof(float));
    } else {
        /* All consumed — leave capacity in place; the next Append will
         * either reuse it (if within capacity) or grow.  Either way
         * we do not free here because the next push is imminent. */
    }
    live->n_samples = remaining;
    /* Try to shrink the underlying allocation.  realloc with a smaller
     * size usually just frees the tail of an oversized mmap, so the
     * cost is a couple of syscalls, not a full copy.  Failure to
     * shrink is harmless: the freed memory will be returned to the
     * OS at the next alloc or process exit. */
    if (live->capacity > 2 * remaining && remaining > 0) {
        float * shrunk = static_cast<float *>(
            std::realloc(live->samples,
                         static_cast<std::size_t>(remaining) * sizeof(float)));
        if (shrunk != nullptr) {
            live->samples = shrunk;
            live->capacity = remaining;
        }
    }
    consumed_samples = 0;
    UnlockLiveAudio(live);
    if (qwen_verbose >= 1) {
        std::fprintf(stderr,
                     "TrimConsumedLiveAudio trim applied, remaining=%lld samples (%.2fs)\n",
                     (long long)remaining, (double)remaining / 16000.0);
    }
}

void FinishManualLiveAudio(qwen_live_audio_t * live) {
    if (live == nullptr) {
        return;
    }
    LockLiveAudio(live);
    live->eof = 1;
    SignalLiveAudio(live);
    UnlockLiveAudio(live);
}

void JoinRealtimeLiveWorker(RealtimeLiveWorker * worker) {
    if (worker == nullptr) {
        return;
    }
    if (qwen_verbose >= 1) {
        std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker enter worker=%p\n",
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                     (void*)worker);
    }
    /* Step 1: signal the VAD producer to stop.  The VAD will:
     *   - wake from cv.wait (live->eof or stop_requested)
     *   - FLUSH any in-flight pending_buffer / segment_buffer as a
     *     final segment so audio captured before /stop is not lost
     *   - push a poison-pill AudioSegment (is_eof_terminator=true)
     *   - return
     * The segment_queue MUST still be open at this point — the VAD
     * needs to push the flush + poison pill.  If we Close() the
     * queue now, both Push() calls fail and the user's last few
     * seconds of speech are silently dropped (the "second session
     * outputs no text" bug).  We close it AFTER vad_thread.join(). */
    worker->stop_requested.store(true, std::memory_order_release);
    FinishManualLiveAudio(&worker->live);

    /* Step 2: join the VAD producer thread.  By the time join()
     * returns, the VAD has flushed its buffers and pushed the poison
     * pill into the queue (or exited without flushing if the queue
     * was already closed by some other path). */
    if (!worker->vad_thread_joined && worker->vad_thread.joinable()) {
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker about to join VAD thread worker=%p\n",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                         (void*)worker);
        }
        worker->vad_thread.join();
        worker->vad_thread_joined = true;
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker VAD thread joined worker=%p\n",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                         (void*)worker);
        }
    }

    /* Step 3: now that the VAD is done pushing, close the segment
     * queue.  This wakes the ASR consumer if it is blocked in Pop()
     * and ensures the closed-empty predicate is reachable. */
    worker->segment_queue.Close();

    /* Step 4: join the ASR consumer thread.  It will:
     *   - drain any remaining segments in the queue (including the
     *     poison pill pushed by the VAD)
     *   - see the queue closed-and-empty predicate
     *   - return
     */
    if (!worker->asr_thread_joined && worker->asr_thread.joinable()) {
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker about to join ASR thread worker=%p\n",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                         (void*)worker);
        }
        worker->asr_thread.join();
        worker->asr_thread_joined = true;
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker ASR thread joined worker=%p\n",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch())
                         .count(),
                         (void*)worker);
        }
    }

     /* Step 5: destroy per-session VAD now that both threads have
     * exited.  The VAD is only accessed by the VAD facade thread;
     * after join, it's safe to destroy.  unique_ptr will call
     * qwen_silero_vad_destroy. */
    if (worker->session_vad) {
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "[rt t=%.0fms] JoinRealtimeLiveWorker destroy session_vad worker=%p\n",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count(),
                         (void*)worker);
        }
        worker->session_vad.reset();
    }

    /* Step 6: free the live audio buffer.  Both threads have exited
     * so no one is reading from it anymore. */
    if (worker->live_ready) {
        DestroyManualLiveAudio(&worker->live);
        worker->live_ready = false;
    }
}

RealtimeSessionSnapshot SnapshotRealtimeSession(const RealtimeSession & session) {
    RealtimeSessionSnapshot snapshot;
    snapshot.id = session.id;
    snapshot.model = session.model;
    snapshot.language = session.language;
    snapshot.total_samples = session.total_samples;
    snapshot.decoded_samples = session.decoded_samples;
    snapshot.retained_sample_count = session.samples.size();
    snapshot.retained_sample_offset = session.retained_sample_offset;
    snapshot.display_snapshot = session.display_snapshot;
    snapshot.text = session.text;
    snapshot.stable_text = session.stable_text;
    snapshot.partial_text = session.partial_text;
    snapshot.segments_text = session.segments_text;
    snapshot.segments_sample_positions = session.segments_sample_positions;
    snapshot.segment_cumulative_samples = session.segment_cumulative_samples;
    snapshot.tail_text = session.tail_text;
    snapshot.current_segment_audio_sec = session.current_segment_audio_sec;
    snapshot.last_inference_ms = session.last_inference_ms;
    snapshot.last_decode_ran = session.last_decode_ran;
    snapshot.finalized = session.finalized;
    snapshot.last_ingress_peak = session.last_ingress_peak;
    snapshot.last_ingress_rms = session.last_ingress_rms;
    snapshot.max_ingress_peak = session.max_ingress_peak;
    snapshot.ingress_chunks = session.ingress_chunks;
    snapshot.error = session.error;
    return snapshot;
}

template <typename SessionLike>
void ApplyStableRealtimeCommit(
    std::size_t total_samples,
    std::string_view stable_text,
    double inference_ms,
    bool finalized,
    SessionLike * session) {
    if (session == nullptr) {
        return;
    }

    RealtimeTextUpdate update;
    update.committed = session->stable_text != stable_text || finalized;
    update.stable_text = std::string(stable_text);
    update.partial_text.clear();
    update.text = update.stable_text;
    session->text_state.stable_text = update.stable_text;
    session->text_state.last_text = update.text;
    session->text_state.last_decode_samples = total_samples;
    session->text_state.unstable_since_samples = total_samples;
    ApplyRealtimeUpdate(update, inference_ms, true, finalized, session);
}

/* Apply one per-chunk snapshot from the C engine.  The chunk carries:
 *  - stable_piece: NEWLY committed text in this chunk (delta, append-only)
 *  - tentative_piece: the still-tentative tail (replaces the previous one)
 *  - is_final: true on the very last chunk
 * The receiver is expected to track its own displayed state; we just merge
 * the delta into session->stable_text and replace partial_text in place. */
template <typename SessionLike>
void ApplyChunkRealtimeCommit(
    const qwen_stream_chunk_t * chunk,
    std::size_t total_samples,
    SessionLike * session) {
    if (session == nullptr || chunk == nullptr) {
        return;
    }

    RealtimeTextUpdate update;
    /* Update stable_text first, then read it back so the display state
     * sees the new cumulative value. */
    session->stable_text.append(chunk->stable_piece);
    session->partial_text = std::string(chunk->tentative_piece);
    update.committed = chunk->stable_token_count > 0 || chunk->is_final;
    update.stable_text = session->stable_text;
    update.partial_text = session->partial_text;
    update.text = session->stable_text;
    session->text_state.stable_text = update.stable_text;
    session->text_state.last_text = update.text;
    session->text_state.last_decode_samples = total_samples;
    session->text_state.unstable_since_samples = total_samples;
    /* Accumulate the per-chunk decode_ms into the running total so the
     * UI can show the wall-clock cost of incremental decoding.  Idle /
     * silent-skip chunks contribute 0 ms which is correct. */
    const double new_inference_ms = session->last_inference_ms + chunk->decode_ms;
    ApplyRealtimeUpdate(update, new_inference_ms, true, chunk->is_final, session);
    /* Notify SSE so the client gets partial/live text pushes.  This
     * enables fast first-word display and smooth live partial updates.
     * A version counter lets the SSE handler efficiently detect change
     * without string comparison of partial_text. */
    session->partial_version++;
    session->sse_cv.notify_all();
}

template <typename SessionLike>
std::size_t RetainedSampleCount(const SessionLike & session) {
    return session.samples.size();
}

std::size_t RetainedSampleCount(const RealtimeSessionSnapshot & session) {
    return session.retained_sample_count;
}

template <typename SessionLike>
void AppendRealtimeSamples(
    const RealtimePolicyConfig & policy,
    const std::vector<float> & chunk,
    SessionLike * session) {
    if (session == nullptr || chunk.empty()) {
        return;
    }
    session->samples.insert(session->samples.end(), chunk.begin(), chunk.end());
    session->total_samples += chunk.size();
    session->retained_sample_offset += TrimRealtimeSamples(&session->samples, RealtimeMaxDecodeSamples(policy));
    /* Always keep a copy for post-stop reconciliation.  A 60s session
     * at 16kHz float32 = 2.4 MB, well within memory budget. */
    session->full_audio.insert(session->full_audio.end(), chunk.begin(), chunk.end());
}

template <typename SessionLike>
void ApplyRealtimeUpdate(
    const RealtimeTextUpdate & update,
    double inference_ms,
    bool decoded,
    bool finalized,
    SessionLike * session) {
    if (session == nullptr) {
        return;
    }
    session->stable_text = update.stable_text;
    session->partial_text = update.partial_text;
    session->text = update.text;
    (void)AdvanceRealtimeDisplayState(update, finalized, &session->display_state, &session->display_snapshot);
    session->last_inference_ms = inference_ms;
    session->last_decode_ran = decoded;
}

template <typename SessionLike>
Json BuildRealtimeJson(
    const SessionLike & session,
    bool finalized,
    bool supported) {
    Json body;
    Json recent_segments = Json::array();
    for (const std::string & segment : session.display_snapshot.recent_segments) {
        recent_segments.push_back(segment);
    }
    body["session_id"] = session.id;
    body["sample_count"] = session.total_samples;
    body["decoded_samples"] = session.decoded_samples;
    body["retained_sample_count"] = RetainedSampleCount(session);
    body["retained_sample_offset"] = session.retained_sample_offset;
    body["decoded"] = session.last_decode_ran;
    body["finalized"] = finalized || session.finalized;
    body["supported"] = supported;
    body["stable_text"] = session.stable_text;
    body["partial_text"] = session.partial_text;
    body["text"] = session.text;
    body["recent_segments"] = std::move(recent_segments);
    /* VAD-segmented: the VAD-segmented worker populates a flat
     * list of committed sentences on the session itself.  Mirror
     * it into the JSON so the UI can render it as a transcript
     * without depending on display_snapshot internals.  Older
     * session types (HostCaptureSession) don't carry this field;
     * for those we emit an empty array. */
    Json segments_arr = Json::array();
    const std::vector<std::string> * segments_ptr = nullptr;
    if constexpr (std::is_same<SessionLike, RealtimeSessionSnapshot>::value) {
        segments_ptr = &session.segments_text;
    } else if constexpr (std::is_same<SessionLike, RealtimeSession>::value) {
        segments_ptr = &session.segments_text;
    }
    if (segments_ptr) {
        for (const std::string & s : *segments_ptr) {
            segments_arr.push_back(s);
        }
    }
    body["segments"] = std::move(segments_arr);
    /* Per-segment cumulative sample positions for timeline display. */
    Json segpos_arr = Json::array();
    if constexpr (std::is_same<SessionLike, RealtimeSessionSnapshot>::value ||
                  std::is_same<SessionLike, RealtimeSession>::value) {
        for (const auto & p : session.segments_sample_positions) {
            segpos_arr.push_back(p);
        }
    }
    body["segmentSamples"] = std::move(segpos_arr);
    /* P1: VAD candidates — tentative, not yet confirmed by finalizer.
     * Sent separately from segments so the UI can distinguish tentative
     * from confirmed text. */
    Json candidates_arr = Json::array();
    if constexpr (std::is_same<SessionLike, RealtimeSessionSnapshot>::value) {
        for (const std::string & s : session.candidates) {
            candidates_arr.push_back(s);
        }
    } else if constexpr (std::is_same<SessionLike, RealtimeSession>::value) {
        for (const std::string & s : session.candidates) {
            candidates_arr.push_back(s);
        }
    }
    body["candidates"] = std::move(candidates_arr);
    double current_sec = 0.0;
    if constexpr (std::is_same<SessionLike, RealtimeSessionSnapshot>::value) {
        current_sec = session.current_segment_audio_sec;
    } else if constexpr (std::is_same<SessionLike, RealtimeSession>::value) {
        current_sec = session.current_segment_audio_sec;
    }
    body["current_segment_audio_sec"] = current_sec;
    body["finalized_segment_count"] = session.display_snapshot.total_finalized_segments;
    body["live_stable_text"] = session.display_snapshot.live_stable_text;
    body["live_partial_text"] = session.display_snapshot.live_partial_text;
    body["live_text"] = session.display_snapshot.live_text;
    body["display_text"] = session.display_snapshot.display_text;
    body["inference_ms"] = session.last_inference_ms;
    /* Audio ingress diagnostic piggybacked onto /status so the UI
     * doesn't need a separate /audio_diag poll.  These are the same
     * fields /audio_diag returned; see /api/realtime/audio_diag below
     * for the canonical producer. */
    if constexpr (std::is_same<SessionLike, RealtimeSession>::value ||
                  std::is_same<SessionLike, RealtimeSessionSnapshot>::value) {
        body["ingress_peak"] = session.last_ingress_peak;
        body["ingress_rms"] = session.last_ingress_rms;
        body["max_ingress_peak"] = session.max_ingress_peak;
        body["ingress_chunks"] = session.ingress_chunks;
    }
    if (!session.error.empty()) {
        body["error"] = session.error;
    }
    return body;
}

template <typename SessionLike>
Json BuildOpenAiRealtimeSessionJson(
    const SessionLike & session,
    std::string_view model_id,
    const RealtimePolicyConfig & realtime_policy) {
    Json body = Json::object({
        {"id", session.id},
        {"object", "realtime.session"},
        {"model", std::string(model_id)},
        {"language", session.language},
        {"input_audio_format", "pcm16le"},
        {"max_decode_window_ms", realtime_policy.max_decode_window_ms},
        {"supported", true},
    });
    return body;
}

template <typename SessionLike>
Json BuildOpenAiRealtimeEventJson(
    const SessionLike & session,
    std::string_view type,
    bool finalized,
    std::string_view model_id,
    const RealtimePolicyConfig & realtime_policy) {
    Json body = Json::object({
        {"object", "realtime.response"},
        {"type", std::string(type)},
        {"session_id", session.id},
        {"session", BuildOpenAiRealtimeSessionJson(session, model_id, realtime_policy)},
        {"state", BuildRealtimeJson(session, finalized, true)},
    });
    return body;
}

}  // namespace
#else
}  // namespace
#endif

Status ValidateServerConfig(const ServerConfig & config) {
    if (config.host.empty()) {
        return Status(StatusCode::kInvalidArgument, "host must not be empty");
    }
    if (config.port <= 0 || config.port > 65535) {
        return Status(StatusCode::kOutOfRange, "port must be in 1..65535");
    }
    if (config.threads < 0) {
        return Status(StatusCode::kInvalidArgument, "threads must be >= 0");
    }
    if (config.verbosity < 0) {
        return Status(StatusCode::kInvalidArgument, "verbosity must be >= 0");
    }
    if (config.temperature > 2.0f) {
        return Status(StatusCode::kOutOfRange, "temperature must be <= 2.0");
    }
    if (config.ui_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "ui_dir must not be empty");
    }
    if (!fs::exists(config.ui_dir) || !fs::is_directory(config.ui_dir)) {
        return Status(StatusCode::kNotFound, "ui_dir does not exist: " + config.ui_dir);
    }
    if (!fs::exists(fs::path(config.ui_dir) / "index.html")) {
        return Status(StatusCode::kNotFound, "ui_dir is missing index.html");
    }
    if (!fs::exists(fs::path(config.ui_dir) / "app.js")) {
        return Status(StatusCode::kNotFound, "ui_dir is missing app.js");
    }
    if (!fs::exists(fs::path(config.ui_dir) / "style.css")) {
        return Status(StatusCode::kNotFound, "ui_dir is missing style.css");
    }
    if (Status status = ValidateModelDirectory(config.model_dir); !status.ok()) {
        return status;
    }
    /* realtime_model_dir is optional; if set, it must point to a valid
     * Qwen3-ASR model directory too.  An empty string means "use the
     * same model as batch". */
    if (!config.realtime_model_dir.empty() &&
        config.realtime_model_dir != config.model_dir) {
        if (Status status = ValidateModelDirectory(config.realtime_model_dir);
            !status.ok()) {
            return Status(
                status.code(),
                "realtime_model_dir invalid: " + status.message());
        }
    }
    return OkStatus();
}

Status ParseServerArguments(int argc, const char * const argv[], ServerConfig * config, bool * show_help) {
    if (config == nullptr || show_help == nullptr) {
        return Status(StatusCode::kInvalidArgument, "outputs must not be null");
    }
    if (argc <= 0 || argv == nullptr || argv[0] == nullptr) {
        return Status(StatusCode::kInvalidArgument, "argv must contain program name");
    }

    *config = ServerConfig{};
    *show_help = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "-h" || arg == "--help") {
            *show_help = true;
            continue;
        }
        if (arg == "--model-dir") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--model-dir", &value);
            if (!status.ok()) {
                return status;
            }
            config->model_dir = value;
            ++index;
            continue;
        }
        if (arg == "--realtime-model-dir") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--realtime-model-dir", &value);
            if (!status.ok()) {
                return status;
            }
            config->realtime_model_dir = value;
            ++index;
            continue;
        }
        if (arg == "--host") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--host", &value);
            if (!status.ok()) {
                return status;
            }
            config->host = value;
            ++index;
            continue;
        }
        if (arg == "--ui-dir") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--ui-dir", &value);
            if (!status.ok()) {
                return status;
            }
            config->ui_dir = value;
            ++index;
            continue;
        }
        if (arg == "--port") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--port", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Argument(value, "port", &config->port);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--threads") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--threads", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Argument(value, "threads", &config->threads);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--verbosity") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--verbosity", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Argument(value, "verbosity", &config->verbosity);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--quiet" || arg == "-q") {
            /* Alias for --verbosity 0.  Production / supervised runs use
             * this to silence the per-poll VAD / ingress fprintf spam
             * while keeping fatal errors and the [ERROR]/[WARN] lines
             * (those are unconditional stderr writes). */
            config->verbosity = 0;
            continue;
        }
        if (arg == "--encoder-int8") {
            config->encoder_int8 = true;
            std::fprintf(stderr,
                "warning: --encoder-int8 is temporarily disabled (code retained, no-op); "
                "see docs/INCIDENTS.md 2026-06-05 encoder INT8 disabled\n");
            continue;
        }
        if (arg == "--backend") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--backend", &value);
            if (!status.ok()) return status;
            config->backend = ParseBackendKind(value);
            ++index;
            continue;
        }
        if (arg == "--no-fallback") {
            config->allow_backend_fallback = false;
            continue;
        }
        if (arg == "--temperature") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--temperature", &value);
            if (!status.ok()) {
                return status;
            }
            char * endp = nullptr;
            float t = std::strtof(value, &endp);
            if (endp == value || *endp != '\0') {
                return Status(StatusCode::kInvalidArgument, "temperature must be a valid float");
            }
            config->temperature = t;
            ++index;
            continue;
        }
        return Status(StatusCode::kInvalidArgument, "unknown argument: " + std::string(arg));
    }

    if (*show_help) {
        return OkStatus();
    }
    if (config->model_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "--model-dir is required");
    }
    return ValidateServerConfig(*config);
}

std::string BuildServerUsage(std::string_view program_name) {
    std::string usage;
    usage += std::string(program_name);
    usage += " --model-dir <dir> [options]\n";
    usage += "  --model-dir <dir>          (batch model, required)\n";
    usage += "  --realtime-model-dir <dir> (realtime/host-capture model; default = same as --model-dir.\n";
    usage += "                              If set to a different path, a second model is loaded.\n";
    usage += "                              Typical use: 0.6B for realtime, 1.7B for batch.)\n";
    usage += "  --host <ip>\n";
    usage += "  --port <n>\n";
    usage += "  --ui-dir <dir>\n";
    usage += "  --threads <n>\n";
    usage += "  --verbosity <n>          0=silent (recommended for production),\n";
    usage += "                           1=commit/summary, 2=per-poll, 3=raw\n";
    usage += "  --quiet, -q              alias for --verbosity 0\n";
    usage += "  --temperature <float>  (default: auto, 0=greedy, >0=sampling)\n";
    usage += "  --backend cpu|cuda     (default: cpu)\n";
    usage += "  --no-fallback          fail if requested backend unavailable\n";
    usage += "  -h, --help\n";
    return usage;
}

/* ────── HTTP handler helpers (testable wrappers) ────── */

void ServeStaticTextFile(
    HttpResponse & response,
    const fs::path & path,
    const std::string & content_type,
    const std::string & label) {
    const std::string body = LoadTextFile(path);
    if (body.empty()) {
        SetErrorResponse(response, Status(StatusCode::kInternal, "failed to load " + label), 500);
        return;
    }
    response.set_content(body, content_type);
}

std::string BuildHealthJson() {
    /* Constant for now.  If we ever add a "deep health" endpoint
     * (model loaded, last successful inference, etc.), this is
     * where the per-field logic goes. */
    return "{\"status\":\"ok\"}";
}

/* ============================================================================
 * VAD-segmented batch transcription (file → VAD segments → per-segment decode)
 * ============================================================================
 *
 * Why this exists:  Qwen3-ASR has a 5×8s = 40s encoder context window.
 * Feeding a 28-minute audio file directly to qwen_transcribe() makes the
 * encoder run way past its designed window; token generation crawls
 * (RTF > 1, the model never converges in any reasonable wall time), and
 * the UI times out at 5 min.
 *
 * The fix is the same sentence-bounded VAD-decode pattern used by the
 * realtime worker (see RunVadSegmentedDecode), but operating on a file:
 *
 *   1. Read the WAV file as 16 kHz mono float (qwen_load_wav handles
 *      arbitrary source sample rate via sinc resampling).
 *   2. Stream the samples through a 512-sample frame loop.
 *   3. For each frame, run Silero VAD; track segment_active and
 *      silence_run (consecutive non-speech frames after speech).
 *   4. Commit the current segment when EITHER:
 *        (a) silence_run reaches kBatchVadSilenceFrames (500ms) — the
 *            speaker has paused between sentences, OR
 *        (b) the segment buffer reaches 40 s — safety cap so a long
 *            monologue never overflows the encoder window.
 *   5. Run qwen_transcribe_audio() on the committed segment (full
 *      offline decode, no rollback, no coalesce, no partial).  Append
 *      the result, fire the per-segment callback (which the async
 *      handler uses to update the job text under jobs_mu), and reset
 *      the VAD state for the next segment.
 *   6. After the loop, flush any trailing audio as a final EOF commit.
 *
 * Per-segment wall time is ~1.2-1.8 s (VAD silence 500 ms + decode
 * 700-1300 ms for 2-5 s of speech).  A 28 min file with 200-400
 * segments takes ~10-15 min total wall time, which is acceptable
 * because the async API streams progress and the UI shows growing
 * text per segment.
 */
struct VadSegmentedBatchResult {
    Status status;
    std::string text;
    int64_t segments = 0;
    double audio_ms = 0.0;
    double inference_ms = 0.0;
    int64_t total_samples = 0;
    int64_t text_chars = 0;
};

/* Per-segment callback.  Called from the worker thread for each
 * committed segment, before the next segment is decoded.  Must not
 * throw (exceptions are caught by the caller and treated as cancel).
 *
 * Return true to continue decoding, false to cancel the whole job. */
using VadSegmentCallback = std::function<bool(int /*seg_idx*/,
                                               std::string_view /*seg_text*/,
                                               int /*seg_samples*/,
                                               int64_t /*total_samples*/)>;

constexpr int kBatchVadSilenceFrames = 16;            /* 16 × 32ms = ~500ms */
constexpr int kBatchVadMaxSamples = 40 * 16000;        /* 40 s safety cap */
constexpr int kBatchVadFrameSamples = 512;            /* Silero VAD chunk */
constexpr float kBatchVadSpeechProbThreshold = 0.35f;

VadSegmentedBatchResult TranscribeFileVadSegmentedImpl(
    qwen_ctx_t *ctx,
    qwen_silero_vad_t *vad,
    const char *wav_path,
    const char *forced_language,
    int max_new_tokens,
    int verbosity,
    const VadSegmentCallback &on_segment,
    const std::function<bool()> &cancel_cb,
    std::mutex *vad_mu) {

    VadSegmentedBatchResult out;

    if (ctx == nullptr || wav_path == nullptr) {
        out.status = Status(StatusCode::kInvalidArgument, "ctx and wav_path required");
        return out;
    }

    /* Load WAV as 16 kHz mono float (handles any source rate via sinc
     * resample inside qwen_load_wav). */
    int n_samples = 0;
    float *samples = qwen_load_wav(wav_path, &n_samples);
    if (samples == nullptr) {
        out.status = Status(StatusCode::kInvalidArgument,
                            std::string("failed to load WAV file: ") + wav_path);
        return out;
    }
    out.total_samples = n_samples;
    out.audio_ms = (double)n_samples * 1000.0 / 16000.0;

    if (verbosity >= 1) {
        std::fprintf(stderr,
                     "VAD-segmented batch: %.2fs of audio, lang=%s, max_new_tokens=%d\n",
                     (double)n_samples / 16000.0,
                     (forced_language != nullptr && forced_language[0] != '\0') ? forced_language : "<auto>",
                     max_new_tokens);
    }

    /* Configure ctx for offline-style per-segment decode.  This matches
     * the realtime VAD-segmented worker: each segment is decoded
     * independently, with no streaming chunk cadence and no rolling
     * rollback. */
    ctx->segment_sec = 0.0f;
    ctx->search_sec = 0.0f;
    ctx->past_text_conditioning = 0;
    ctx->stream_chunk_sec = 0.0f;
    ctx->stream_max_new_tokens = 0;
    if (qwen_set_force_language(ctx, forced_language) != 0) {
        std::free(samples);
        out.status = Status(StatusCode::kInvalidArgument,
                            std::string("unsupported language: ") +
                            (forced_language ? forced_language : ""));
        return out;
    }

    /* Wire the cancel callback into the C decoder so qwen_transcribe_audio
     * checks cancel mid-decode (not just between segments).  Uses the
     * same ForwardCancelRequest trampoline as TranscribeFile. */
    std::function<bool()> cancel_trampoline = cancel_cb;
    if (cancel_trampoline) {
        qwen_set_cancel_callback(ctx, ForwardCancelRequest, &cancel_trampoline);
    } else {
        qwen_set_cancel_callback(ctx, nullptr, nullptr);
    }

    /* Use the server-scope shared VAD (loaded once at server start).
     * NULL-safe: if the VAD could not be loaded (model missing, ONNX
     * runtime absent) we fall back to a pure 40s timer — segments are
     * bigger but the result is still correct, just coarser.  We do
     * NOT destroy the VAD here; the server owns its lifetime. */
    const bool vad_active = qwen_silero_vad_is_active(vad);
    if (verbosity >= 1) {
        std::fprintf(stderr, "VAD-segmented batch: Silero VAD %s (shared instance)\n",
                     vad_active ? "active" : "inactive (40s timer fallback)");
    }
    /* Reset the VAD's LSTM state at the start of every batch so a
      * previous run's context doesn't leak into the first segment of
      * this file.  Protect with vad_mu since a realtime session may
      * be using the same VAD concurrently. */
    if (vad_active) {
        if (vad_mu) {
            std::lock_guard<std::mutex> lock(*vad_mu);
        }
        qwen_silero_vad_reset(vad);
    }

    std::vector<float> segment_buffer;
    segment_buffer.reserve(kBatchVadMaxSamples * 2);
    int64_t processed_frames = 0;   /* frames in segment_buffer already seen by VAD */
    int silence_run = 0;            /* consecutive non-speech frames after segment_active */
    bool segment_active = false;    /* speech is in progress in current segment */
    int seg_idx = 0;
    int64_t total_consumed = 0;     /* samples read from source into segment_buffer */

    auto commit_segment = [&](const char *reason) -> bool {
        if (segment_buffer.empty()) {
            silence_run = 0;
            return true;
        }
        const int seg_n = static_cast<int>(segment_buffer.size());
        const double seg_sec = (double)seg_n / 16000.0;
        if (verbosity >= 1) {
            std::fprintf(stderr,
                         "VAD-segmented batch: committing segment %d (%.2fs, reason=%s)\n",
                         seg_idx, seg_sec, reason);
        }
        char *raw = qwen_transcribe_audio(ctx, segment_buffer.data(), seg_n);
        std::string text;
        if (raw != nullptr) {
            text.assign(raw);
            std::free(raw);
            /* Trim trailing whitespace the model may emit. */
            while (!text.empty() &&
                   (text.back() == ' ' || text.back() == '\n' || text.back() == '\t')) {
                text.pop_back();
            }
        } else if (verbosity >= 1) {
            std::fprintf(stderr,
                         "VAD-segmented batch: qwen_transcribe_audio returned null for seg %d\n",
                         seg_idx);
        }
        const int this_seg_samples = seg_n;
        out.text += text;
        out.text_chars += static_cast<int64_t>(text.size());
        /* qwen_transcribe_audio() resets ctx->perf_total_ms = 0 at the
         * start of each call, so by the time we get here it's only
         * the LAST segment's time.  Accumulate across segments so
         * the final out.inference_ms is the cumulative wall time
         * spent in qwen_transcribe_audio() across all VAD
         * segments.  We also reset the perf counter right after
         * reading so the next call's increment is uncontaminated
         * (in case the C code didn't reset). */
        out.inference_ms += ctx->perf_total_ms;
        ctx->perf_total_ms = 0.0;
        segment_buffer.clear();
        processed_frames = 0;
        silence_run = 0;
        segment_active = false;
        if (vad_active) {
            if (vad_mu) {
                std::lock_guard<std::mutex> lock(*vad_mu);
            }
            qwen_silero_vad_reset(vad);
        }

        /* Fire the per-segment callback (async handler uses this to
         * publish partial text to the job).  Returning false aborts. */
        const bool keep_going = !on_segment
            ? true
            : on_segment(seg_idx, text, this_seg_samples, total_consumed);
        seg_idx++;
        out.segments = seg_idx;
        return keep_going;
    };

    /* Main loop.  For each iteration:
     *   1. Copy at most one VAD frame (512 samples) from the source
     *      into segment_buffer (capped at 40 s).
     *   2. Run VAD on any newly-copied frames, updating segment_active
     *      and silence_run.
     *   3. Decide whether to commit (silence ≥ 16 frames or 40 s cap).
     *   4. Check cancel.  Exit on EOF + empty buffer. */
    const int frame = kBatchVadFrameSamples;
    while (true) {
        /* Step 1: copy one frame.  Always resize() BEFORE memcpy() —
         * std::vector value-initializes new elements to 0 on grow, so
         * memcpy-first would be silently overwritten. */
        if (segment_buffer.size() < static_cast<std::size_t>(kBatchVadMaxSamples) &&
            total_consumed < n_samples) {
            int64_t want = n_samples - total_consumed;
            if (want > frame) want = frame;
            const int64_t room = kBatchVadMaxSamples - static_cast<int64_t>(segment_buffer.size());
            if (want > room) want = room;
            if (want > 0) {
                const std::size_t old = segment_buffer.size();
                segment_buffer.resize(old + static_cast<std::size_t>(want));
                std::memcpy(segment_buffer.data() + old,
                            samples + total_consumed,
                            static_cast<std::size_t>(want) * sizeof(float));
                total_consumed += want;
            }
        }

        /* Step 2: VAD sweep on any new frames.  Hold vad_mu across the
         * entire sweep so a concurrent realtime session's VAD doesn't
         * race on the shared LSTM / context buffers. */
        if (vad_active) {
            const int64_t total_buf = static_cast<int64_t>(segment_buffer.size());
            const int64_t total_frames = total_buf / frame;
            if (vad_mu) {
                std::lock_guard<std::mutex> lock(*vad_mu);
                for (int64_t fi = processed_frames; fi < total_frames; ++fi) {
                    float prob = 0.0f;
                    qwen_silero_vad_process(vad,
                                            segment_buffer.data() + fi * frame,
                                            frame, &prob);
                    if (prob >= kBatchVadSpeechProbThreshold) {
                        segment_active = true;
                        silence_run = 0;
                    } else {
                        if (segment_active) {
                            silence_run++;
                        }
                    }
                }
            } else {
                for (int64_t fi = processed_frames; fi < total_frames; ++fi) {
                    float prob = 0.0f;
                    qwen_silero_vad_process(vad,
                                            segment_buffer.data() + fi * frame,
                                            frame, &prob);
                    if (prob >= kBatchVadSpeechProbThreshold) {
                        segment_active = true;
                        silence_run = 0;
                    } else {
                        if (segment_active) {
                            silence_run++;
                        }
                    }
                }
            }
            processed_frames = total_frames;
        } else {
            /* No VAD: activate after 0.5 s of buffered audio (treat the
             * whole buffer as one continuous segment).  We do NOT
             * advance silence_run here, so the only commit trigger is
             * the 40 s safety cap or EOF. */
            if (segment_buffer.size() >= 8000) {
                segment_active = true;
            }
        }

        /* Step 3: commit decision.
         *
         * Triggers, in priority order:
         *   1. VAD active + segment active + 16 silent frames (~500ms)
         *      → speaker paused, commit.
         *   2. Buffer ≥ 40 s → safety cap (prevents encoder overflow).
         *   3. Source exhausted and buffer has anything → "eof_soak"
         *      safety: even if VAD never reported silence, the audio
         *      is fully consumed, so commit what we have.  Prevents
         *      an infinite loop when VAD is wrong and the audio ends
         *      mid-speech.  The post-loop flush handles the case
         *      where this branch is skipped because the source is
         *      drained but the buffer still has data. */
        bool should_commit = false;
        const char *commit_reason = nullptr;
        if (vad_active && segment_active && silence_run >= kBatchVadSilenceFrames) {
            should_commit = true;
            commit_reason = "vad_silence";
        } else if (static_cast<int>(segment_buffer.size()) >= kBatchVadMaxSamples) {
            should_commit = true;
            commit_reason = "40s_safety_cap";
        } else if (total_consumed >= n_samples && !segment_buffer.empty() &&
                   segment_active) {
            should_commit = true;
            commit_reason = "eof_with_active_segment";
        }
        if (should_commit) {
            if (!commit_segment(commit_reason)) {
                std::free(samples);
                /* Do NOT destroy vad — it is the server-scope shared
                 * instance, owned by RunServer. */
                qwen_set_cancel_callback(ctx, nullptr, nullptr);
                out.status = Status(StatusCode::kFailedPrecondition,
                                    "transcription cancelled by callback");
                return out;
            }
        }

        /* Step 4: exit conditions.
         *
         * (a) Source fully consumed AND no in-flight segment to flush
         *     → all done, break immediately.
         * (b) Source fully consumed AND in-flight segment present →
         *     flush it now.  Without this branch, the loop would
         *     spin forever on the trailing audio: step 1 copies 0
         *     samples (EOF), step 2 processes 0 new VAD frames,
         *     step 3 doesn't fire a commit (silence_run=0 if VAD
         *     keeps reporting speech right up to the end), and
         *     step 4's "empty buffer" check is false.  So we break
         *     here and rely on the post-loop EOF flush to commit. */
        if (total_consumed >= n_samples) {
            if (segment_buffer.empty()) {
                break;
            }
            if (verbosity >= 1) {
                std::fprintf(stderr,
                             "VAD-segmented batch: EOF + %zu samples trailing, "
                             "breaking to post-loop EOF flush (segments=%lld)\n",
                             segment_buffer.size(),
                             static_cast<long long>(out.segments));
            }
            break;
        }

        /* Step 5: check cancel.  Cheap atomic load, no allocation. */
        if (cancel_cb && cancel_cb()) {
            std::free(samples);
            /* Do NOT destroy vad — it is the server-scope shared
             * instance, owned by RunServer. */
            qwen_set_cancel_callback(ctx, nullptr, nullptr);
            out.status = Status(StatusCode::kFailedPrecondition,
                                "transcription cancelled");
            return out;
        }
    }

    /* Flush any trailing audio (e.g. last segment cut by EOF). */
    if (!segment_buffer.empty()) {
        commit_segment("eof");
    }

    std::free(samples);
    /* Do NOT destroy vad — it is the server-scope shared instance,
     * owned by RunServer. */
    qwen_set_cancel_callback(ctx, nullptr, nullptr);
    return out;
}

/* GPU engine version of VAD-segmented batch transcription.
 * Same VAD frame loop logic, but uses AsrEngine::TranscribeSegment
 * instead of qwen_transcribe_audio.  This provides per-segment
 * progress streaming for GPU backend (previously GPU path called
 * TranscribeFile which processed the entire file in one shot). */
VadSegmentedBatchResult TranscribeFileVadSegmentedEngineImpl(
    AsrEngine * engine,
    qwen_silero_vad_t * vad,
    const char * wav_path,
    const std::string & forced_language,
    int verbosity,
    const VadSegmentCallback & on_segment,
    const std::function<bool()> & cancel_cb,
    std::mutex * vad_mu) {

    VadSegmentedBatchResult out;

    if (engine == nullptr || wav_path == nullptr) {
        out.status = Status(StatusCode::kInvalidArgument,
                            "engine and wav_path required");
        return out;
    }

    /* Load WAV as 16 kHz mono float. */
    int n_samples = 0;
    float * samples = qwen_load_wav(wav_path, &n_samples);
    if (samples == nullptr) {
        out.status = Status(StatusCode::kInvalidArgument,
                            std::string("failed to load WAV file: ") + wav_path);
        return out;
    }
    out.total_samples = n_samples;
    out.audio_ms = (double)n_samples * 1000.0 / 16000.0;

    if (verbosity >= 1) {
        std::fprintf(stderr,
                     "VAD-segmented batch (engine): %.2fs of audio, "
                     "lang=%s\n",
                     (double)n_samples / 16000.0,
                     forced_language.empty() ? "<auto>" : forced_language.c_str());
    }

    /* Create persistent engine session for all segments. */
    std::uint64_t sid = 0;
    SessionOptions opts;
    opts.language = forced_language;
    opts.prompt = "";
    Status sess_st = engine->CreateSession(opts, sid);
    if (!sess_st.ok()) {
        std::free(samples);
        out.status = sess_st;
        return out;
    }

    const bool vad_active = qwen_silero_vad_is_active(vad);
    if (verbosity >= 1) {
        std::fprintf(stderr,
                     "VAD-segmented batch (engine): Silero VAD %s\n",
                     vad_active ? "active" : "inactive (40s timer fallback)");
    }
    if (vad_active) {
        if (vad_mu) {
            std::lock_guard<std::mutex> lock(*vad_mu);
        }
        qwen_silero_vad_reset(vad);
    }

    std::vector<float> segment_buffer;
    segment_buffer.reserve(kBatchVadMaxSamples * 2);
    int64_t processed_frames = 0;
    int silence_run = 0;
    bool segment_active = false;
    int seg_idx = 0;
    int64_t total_consumed = 0;

    auto commit_segment = [&](const char * reason) -> bool {
        if (segment_buffer.empty()) {
            silence_run = 0;
            return true;
        }
        const int seg_n = static_cast<int>(segment_buffer.size());
        const double seg_sec = (double)seg_n / 16000.0;
        if (verbosity >= 1) {
            std::fprintf(stderr,
                         "VAD-segmented batch (engine): committing "
                         "segment %d (%.2fs, reason=%s)\n",
                         seg_idx, seg_sec, reason);
        }

        /* GPU transcription via engine. */
        AsrSegmentResult segRes = engine->TranscribeSegment(
            sid, segment_buffer, 16000);
        std::string text;
        if (segRes.status.ok()) {
            text = segRes.text;
        } else if (verbosity >= 1) {
            std::fprintf(stderr,
                         "VAD-segmented batch (engine): TranscribeSegment "
                         "failed for seg %d: %s\n",
                         seg_idx, segRes.status.message().c_str());
        }

        out.text += text;
        out.text_chars += static_cast<int64_t>(text.size());
        out.inference_ms += segRes.total_ms;
        segment_buffer.clear();
        processed_frames = 0;
        silence_run = 0;
        segment_active = false;
        if (vad_active) {
            if (vad_mu) {
                std::lock_guard<std::mutex> lock(*vad_mu);
            }
            qwen_silero_vad_reset(vad);
        }

        const bool keep_going = !on_segment
            ? true
            : on_segment(seg_idx, text, seg_n, total_consumed);
        seg_idx++;
        out.segments = seg_idx;
        return keep_going;
    };

    const int frame = kBatchVadFrameSamples;
    while (true) {
        /* Step 1: copy one frame. */
        if (segment_buffer.size() < static_cast<std::size_t>(kBatchVadMaxSamples) &&
            total_consumed < n_samples) {
            int64_t want = n_samples - total_consumed;
            if (want > frame) want = frame;
            const int64_t room = kBatchVadMaxSamples -
                                 static_cast<int64_t>(segment_buffer.size());
            if (want > room) want = room;
            if (want > 0) {
                const std::size_t old = segment_buffer.size();
                segment_buffer.resize(old + static_cast<std::size_t>(want));
                std::memcpy(segment_buffer.data() + old,
                            samples + total_consumed,
                            static_cast<std::size_t>(want) * sizeof(float));
                total_consumed += want;
            }
        }

        /* Step 2: VAD sweep on any new frames. */
        if (vad_active) {
            const int64_t total_buf = static_cast<int64_t>(segment_buffer.size());
            const int64_t total_frames = total_buf / frame;
            if (vad_mu) {
                std::lock_guard<std::mutex> lock(*vad_mu);
                for (int64_t fi = processed_frames; fi < total_frames; ++fi) {
                    float prob = 0.0f;
                    qwen_silero_vad_process(vad,
                                            segment_buffer.data() + fi * frame,
                                            frame, &prob);
                    if (prob >= kBatchVadSpeechProbThreshold) {
                        segment_active = true;
                        silence_run = 0;
                    } else {
                        if (segment_active) silence_run++;
                    }
                }
            } else {
                for (int64_t fi = processed_frames; fi < total_frames; ++fi) {
                    float prob = 0.0f;
                    qwen_silero_vad_process(vad,
                                            segment_buffer.data() + fi * frame,
                                            frame, &prob);
                    if (prob >= kBatchVadSpeechProbThreshold) {
                        segment_active = true;
                        silence_run = 0;
                    } else {
                        if (segment_active) silence_run++;
                    }
                }
            }
            processed_frames = total_frames;
        } else {
            if (segment_buffer.size() >= 8000) {
                segment_active = true;
            }
        }

        /* Step 3: commit decision. */
        bool should_commit = false;
        const char * commit_reason = nullptr;
        if (vad_active && segment_active &&
            silence_run >= kBatchVadSilenceFrames) {
            should_commit = true;
            commit_reason = "vad_silence";
        } else if (static_cast<int>(segment_buffer.size()) >=
                   kBatchVadMaxSamples) {
            should_commit = true;
            commit_reason = "40s_safety_cap";
        } else if (total_consumed >= n_samples && !segment_buffer.empty() &&
                   segment_active) {
            should_commit = true;
            commit_reason = "eof_with_active_segment";
        }
        if (should_commit) {
            if (!commit_segment(commit_reason)) {
                std::free(samples);
                engine->CloseSession(sid);
                out.status = Status(StatusCode::kFailedPrecondition,
                                    "transcription cancelled by callback");
                return out;
            }
        }

        /* Step 4: exit conditions. */
        if (total_consumed >= n_samples) {
            if (segment_buffer.empty()) {
                break;
            }
            break;
        }

        /* Step 5: check cancel. */
        if (cancel_cb && cancel_cb()) {
            std::free(samples);
            engine->CloseSession(sid);
            out.status = Status(StatusCode::kFailedPrecondition,
                                "transcription cancelled");
            return out;
        }
    }

    /* Flush any trailing audio. */
    if (!segment_buffer.empty()) {
        commit_segment("eof");
    }

    std::free(samples);
    engine->CloseSession(sid);
    return out;
}

int RunServer(const ServerConfig & config) {
#ifndef QASR_CPU_BACKEND_ENABLED
    (void)config;
    std::fprintf(stderr, "error: CPU backend is not enabled in this build\n");
    return 1;
#else
    const Status config_status = ValidateServerConfig(config);
    if (!config_status.ok()) {
        std::fprintf(stderr, "server config invalid: %s\n", config_status.message().c_str());
        return 1;
    }

    /* Load model(s) via ServerAsrFacade.
     *
     * The facade wraps SharedAsrModel (CPU) or AsrEngine (CUDA).
     * Currently both batch and realtime share the same facade.
     * Future: separate facades for batch vs realtime when different
     * models are needed (--realtime-model-dir != --model-dir). */
    std::unique_ptr<ServerAsrFacade> facade_ = std::make_unique<ServerAsrFacade>();
    const Status facade_status = facade_->Initialize(config);
    if (!facade_status.ok()) {
        std::fprintf(stderr, "model load failed: %s\n",
                     facade_status.message().c_str());
        return 1;
    }
    std::string backend_label = std::string(facade_->backendKind() == BackendKind::kCpu ? "CPU" : "CUDA");
    if (facade_->backendFallback()) {
        std::fprintf(stderr, "backend: %s (fallback from requested %s)\n",
                     backend_label.c_str(),
                     config.backend == BackendKind::kCuda ? "CUDA" : "CPU");
    } else {
        std::fprintf(stderr, "backend: %s\n", backend_label.c_str());
    }

    /* Aliases for backward compat with existing call sites.
     * batch_model and realtime_model now point to the same facade.
     * Future: separate facades when --realtime-model-dir differs. */
    ServerAsrFacade *const batch_model = facade_.get();
    ServerAsrFacade *const realtime_model = facade_.get();

    /* Warmup: run a dummy inference on each loaded model to trigger
     * oneDNN JIT compilation and warm CPU caches.  Uses a clone so
     * we don't hold the SharedAsrModel mutex while inferring.
     * Runs synchronously before the HTTP server starts listening, so
     * the first real request is not penalised. */
    {
        auto do_warmup = [&](ServerAsrFacade * model, const char * label) {
            InferHandle wHandle = model->createInferHandle();
            qwen_ctx_t * ctx = wHandle.nativeCtx;
            if (!ctx) {
                if (qwen_verbose >= 1) {
                    std::fprintf(stderr, "warmup: %s clone failed, skipping\n", label);
                }
                return;
            }
            /* 1 s of silence at 16 kHz — minimal but enough to trigger
             * full encoder → decoder pipeline and oneDNN JIT compile. */
            const int n_samples = 16000;
            std::vector<float> silence(n_samples, 0.0f);
            ctx->stream_max_new_tokens = 0;
            ctx->stream_chunk_sec = 0.0f;
            ctx->segment_sec = 0.0f;
            ctx->search_sec = 0.0f;
            ctx->past_text_conditioning = 0;

            const auto t0 = std::chrono::steady_clock::now();
            char * raw = qwen_transcribe_audio(ctx, silence.data(), n_samples);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::free(raw);
            model->releaseInferHandle(wHandle);
            if (qwen_verbose >= 1) {
                std::fprintf(stderr, "warmup: %s completed in %.0f ms\n", label, ms);
            }
        };

        do_warmup(batch_model, "batch");
        // share_model removed: single facade handles both paths
    }

    /* Initialize server-level cut VAD for ASR worker audio boundary
     * detection.  This is a separate instance from the facade/session
     * VADs so that cutting operations don't interfere with VAD facade
     * state.  The cut VAD is protected by g_cut_vad_mutex. */
    g_cut_vad.reset(qwen_silero_vad_create(nullptr));
    if (g_cut_vad && qwen_silero_vad_is_active(g_cut_vad.get())) {
        qwen_silero_vad_reset(g_cut_vad.get());
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "cut-vad: server-level VAD created for audio cutting\n");
        }
    } else {
        g_cut_vad.reset();
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "cut-vad: unavailable, falling back to fixed-tail cutting\n");
        }
    }

    const std::string served_model_id = ResolveServedModelId(config.model_dir);
    const fs::path ui_dir(config.ui_dir);
    const RealtimePolicyConfig realtime_policy;
    /* Phase 2A: segment dump config — parsed once at startup. */
    const DumpConfig dump_config = DumpConfig::FromEnv();
    if (dump_config.enabled && qwen_verbose >= 1) {
        std::fprintf(stderr, "Phase 2A: segment dump ENABLED dir=%s max_seg=%lu max_sec=%lu\n",
                     dump_config.dir.c_str(),
                     static_cast<unsigned long>(dump_config.max_segments),
                     static_cast<unsigned long>(dump_config.max_seconds));
    }
    /* Phase 2B: endpoint mode + cap policy — parsed once at startup. */
    const EndpointMode endpoint_mode = ParseEndpointMode();
    const CapOnlyEndpointPolicy cap_policy = CapOnlyEndpointPolicy::FromEnv();
    if (qwen_verbose >= 1) {
        std::fprintf(stderr, "Phase 2B: endpoint_mode=%s first_cap=%dms normal_cap=%dms grace=%dms valley=%dms\n",
                     endpoint_mode == EndpointMode::kCapOnly ? "cap_only" : "legacy",
                     cap_policy.first_latency_cap_ms,
                     cap_policy.normal_latency_cap_ms,
                     cap_policy.cap_grace_ms,
                     cap_policy.cap_valley_silence_ms);
    }
    const auto server_start = std::chrono::steady_clock::now();
    ServerMetrics metrics;
    std::atomic<std::uint64_t> session_counter{1};
    std::unordered_map<std::string, std::shared_ptr<RealtimeSession>> realtime_sessions;
    std::mutex realtime_mu;
    std::unordered_map<std::string, OfflineJob> jobs;
    std::mutex jobs_mu;

    /* VAD mutex: serializes access to the Silero VAD instance so
     * concurrent realtime sessions (created by rapid Stop/Start) do
     * not race on the shared LSTM / context buffers.  The VAD itself
     * lives inside each Qwen context (ctx->vad, created by
     * qwen_load).  VAD-segmented paths use ctx->vad directly — no
     * separate VAD instance is needed.
     *
     * Locking strategy: hold the lock across the entire VAD sweep
     * loop (typically 10-26 frames, ~30-800 ms of audio) rather than
     * per-frame, to amortize lock overhead.  This serializes VAD
     * processing across sessions but VAD inference is cheap (~1 ms
     * per 32 ms frame) so the serialized time is small.
     *
     * See docs/INCIDENTS.md 2026-06-05 VAD shared + mutex. */
    std::mutex vad_mu;
    std::shared_ptr<HostCaptureSession> host_capture;
    std::mutex host_capture_mu;
    std::mutex maintenance_mu;
    std::condition_variable maintenance_cv;
    bool stop_maintenance = false;

    HttpServer server;
    {
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const std::size_t workers = hardware_threads == 0U ? 4U : static_cast<std::size_t>(hardware_threads);
        server.set_thread_pool_size(workers, kHttpWorkerQueueLimit);
    }
    server.set_keep_alive_max_count(100);
    server.set_keep_alive_timeout(5);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(30, 0);
    server.set_idle_interval(1, 0);
    server.set_payload_max_length(64ULL * 1024ULL * 1024ULL);

    /* Debug logging for Stop→Start realtime lifecycle.  Gated on
     * qwen_verbose (== config.verbosity) so the user can enable
     * with --verbosity 1.  Each call stamps the session id, the
     * realtime_mu / session->mu / live-audio lock sequence, and a
     * monotonic timestamp so we can reconstruct the timeline after
     * the fact. */
    #define RT_LOG(fmt, ...)                                                  \
        do {                                                                  \
            if (qwen_verbose >= 1) {                                          \
                const double _rt_now_ms_ = std::chrono::duration<double,      \
                    std::milli>(std::chrono::steady_clock::now()              \
                        .time_since_epoch()).count();                         \
                std::fprintf(stderr, "[rt t=%.0fms] " fmt "\n",               \
                             _rt_now_ms_, ##__VA_ARGS__);                      \
            }                                                                 \
        } while (0)

    std::thread job_cleanup_thread([&]() {
        std::unique_lock<std::mutex> lock(maintenance_mu);
        while (!stop_maintenance) {
            const bool stopping = maintenance_cv.wait_for(
                lock,
                std::chrono::seconds(kAsyncJobCleanupIntervalSeconds),
                [&]() { return stop_maintenance; });
            if (stopping) {
                break;
            }

            lock.unlock();
            const std::int64_t now_seconds = CurrentUnixSeconds();
            std::size_t removed = 0U;
            {
                std::lock_guard<std::mutex> jobs_lock(jobs_mu);
                removed = CleanupExpiredJobs(&jobs, now_seconds, kCompletedAsyncJobTtlSeconds);
            }
            metrics.async_job_cleanup_runs.fetch_add(1);
            metrics.async_jobs_evicted.fetch_add(static_cast<std::uint64_t>(removed));
            lock.lock();
        }
    });

   auto SnapshotRealtimeSessionState = [&](const std::shared_ptr<RealtimeSession> & session,
                                             bool consume_decode_flag,
                                             RealtimeSessionSnapshot * snapshot) -> Status {
        if (session == nullptr || snapshot == nullptr) {
            return Status(StatusCode::kInvalidArgument, "session snapshot output must not be null");
        }
        RT_LOG("SnapshotRealtimeSessionState sid=%s consume=%d enter", session->id.c_str(), consume_decode_flag ? 1 : 0);

        /* Read the VAD's cumulative_decoded_samples BEFORE acquiring
         * session->mu to avoid AB-BA deadlock.  The worker thread locks
         * in the order: LockLiveAudio → session->mu.  If we lock
         * session->mu first (as before) and then try LockLiveAudio, a
         * concurrent worker holding LockLiveAudio and waiting for
         * session->mu would deadlock.
         *
         * We use cumulative_decoded_samples (a per-worker atomic the
         * VAD increments on every consume), NOT live->decoded_cursor
         * (a live-buffer-relative cursor that TrimConsumedLiveAudio
         * resets to 0 on every memmove).  Reading the cursor would
         * cause session->decoded_samples to flip to 0 after every
         * trim, breaking the UI's "已解码 Xs" lag indicator.  See
         * RealtimeLiveWorker::cumulative_decoded_samples for the
         * full rationale. */
        int64_t dc = 0;
        {
            RealtimeLiveWorker * worker = nullptr;
            {
                std::lock_guard<std::mutex> lock(session->mu);
                worker = session->live_worker.get();
            }
            if (worker) {
                dc = worker->cumulative_decoded_samples.load(std::memory_order_relaxed);
            }
        }
        {
            if (qwen_verbose >= 2) RT_LOG("Snapshot sid=%s about to lock(session->mu)", session->id.c_str());
            std::lock_guard<std::mutex> lock(session->mu);
            if (qwen_verbose >= 2) RT_LOG("Snapshot sid=%s session->mu ACQUIRED", session->id.c_str());
            if (dc > 0) {
                session->decoded_samples = static_cast<std::size_t>(dc);
            }
            *snapshot = SnapshotRealtimeSession(*session);
            if (consume_decode_flag) {
                session->last_decode_ran = false;
            }
        }
        RT_LOG("SnapshotRealtimeSessionState sid=%s exit", session->id.c_str());
        return OkStatus();
    };

    /* VAD-segmented decode loop.  Replaces the legacy
     * qwen_transcribe_stream_live's rolling-decode path with a
     * sentence-bounded one:
     *
     *   1. Maintain a per-session "current segment" audio buffer.
     *   2. As the user speaks, copy new audio from the live buffer
     *      into the segment buffer.  Run Silero VAD on the new audio
     *      to track silence_run (number of consecutive silent frames,
     *      1 frame = 32 ms = 512 samples at 16 kHz).
     *   3. Commit the segment when EITHER:
     *        (a) silence_run reaches 16 frames (≈ 500 ms) — the
     *            speaker has paused between sentences, OR
     *        (b) the segment audio reaches 40 s — safety cap for
     *            continuous monologues so we never overflow the
     *            encoder's 8s×5 = 40s context window.
     *   4. Run qwen_transcribe_audio on the committed segment
     *      (full offline decode, no rollback, no coalesce, no
     *      partial).  Push the result to session->segments_text
     *      and update the live_stable_text / recent_segments
     *      display state.  Then reset for the next segment.
     *
     * This eliminates all rolling-decoder artifacts (mid-character
     * cuts, model revisions, partial flickering, "1s pause not
     * refreshing") because each segment is decoded exactly once
     * with the full audio visible.  Cost is per-sentence latency
     * = VAD silence + decode time ≈ 500ms + 700-1300ms = 1.2-1.8s,
     * which the user has explicitly accepted. */
    constexpr int kVadSegmentSilenceMs = 1500;            /* silence gap (ms) that triggers a
                                                                 commit.  1.5s is the statistical
                                                                 boundary between intra-sentence
                                                                 pause and inter-sentence pause. */
    /* DISABLED: fast-silence path that used to commit at 500 ms
     * silence for segments > 3 s.  User reported it broke sentences
     * at unnatural mid-phrase pauses.  Keeping the constant defined
     * for reference but the commit-decision code now always uses
     * kVadSegmentSilenceMs.  Re-enable by:
     *   1. uncomment the two constants below
     *   2. uncomment the `silence_threshold` ternary in the commit
     *      decision block
     *   3. decide what the new trade-off is */
    /* constexpr int kVadSegmentFastSilenceMs = 500; */   /* DISABLED */
    /* constexpr int kVadSegmentMinForFastMs = 3000;  */   /* DISABLED */
    constexpr int kVadSegmentMinValidSamples = 4800;      /* 0.30 s minimum valid speech.
                                                                Below this, treat as noise /
                                                                accidental trigger and discard.
                                                                Pure safety floor — doesn't
                                                                decide whether to emit, only
                                                                whether to keep. */
    constexpr int kVadSegmentMinEmitSamples = 28800;      /* 1.80 s minimum emit length.
                                                                Below this, the segment is held
                                                                in "pending" state for up to
                                                                kVadShortMergeGapMs in case
                                                                more speech arrives to merge
                                                                with.  Above this, emit
                                                                immediately on silence. */
    constexpr int kVadShortMergeGapMs = 1200;            /* 1.2 s grace after a pending
                                                                short segment: if new speech
                                                                arrives within this window,
                                                                merge it into the pending
                                                                segment; otherwise emit the
                                                                pending as-is. */
    constexpr int kVadPreRollMs = 500;                   /* pre-roll: include the last
                                                                 500 ms of audio BEFORE the
                                                                 first VAD-said-speech frame,
                                                                 so initial consonants (e.g.
                                                                 "n" in "你好", "w" in "world")
                                                                 are not clipped.  500ms is
                                                                 chosen to also catch word-
                                                                 internal unvoiced onsets at
                                                                 segment boundaries (e.g.
                                                                 "ich" in "ichthyosaurus")
                                                                 that the VAD often misses. */
    constexpr int kVadPostRollMs = 500;                  /* post-roll: include up to
                                                                  300 ms of audio AFTER the
                                                                  last VAD-said-speech frame,
                                                                 so trailing phoneme decay is
                                                                 preserved.  ASR gets
                                                                 [speech_start - 250 ms,
                                                                  last_speech + 300 ms]. */
   /* Soft-cap with grace: when buffer reaches the soft deadline, enter
     * a grace period (up to 800ms) to look for a short acoustic valley.
     * If a valley of >= 256ms silence appears, emit at the valley.
     * If grace expires, emit forced.  If hard cap reached, emit hard.
     * This avoids cutting directly inside active speech.
     *
     * (adjusted to 5s per user request for CUDA low-latency) */
    constexpr int kVadSegmentSoftDeadlineSamples = 5 * 16000;
    constexpr int kVadSegmentSoftGraceMs = 800;
    constexpr int kVadSegmentHardCapSamples =
        kVadSegmentSoftDeadlineSamples +
        (kVadSegmentSoftGraceMs * 16000 / 1000);  /* 15.8s */
    /* Short acoustic valley after soft deadline.
     * 256ms = 8 VAD frames at 32ms.
     * Intentionally shorter than kVadSegmentSilenceMs=1500ms.
     * This is not a full pause — just enough to avoid cutting mid-speech. */
    constexpr int kVadSoftcapValleySilenceMs = 256;
    /* NOTE: merge_grace / overlap prepending was REMOVED after
    /* NOTE: merge_grace / overlap prepending was REMOVED after
     * testing showed the ASR re-transcribes the prepended tail with
     * slightly different text than the original (same audio, different
     * recognition context), causing duplicate text at segment
     * boundaries: "...花拳绣腿。" → "画拳修腿。...".  We now rely on
     * (a) the VAD being continuous across boundaries (no reset on
     * commit, so context is preserved) and (b) the 1.5s minimum
     * segment length to give the model enough head context.
     * If "first char cut off" reappears, fix it at the ASR level
     * (e.g., emit a forced-prepended prefix anchor), not by audio
     * prepending. */
    constexpr int kVadSegmentPollMs = 50;                 /* poll live buffer every 50 ms */
    constexpr int kVadVadFrameMs = 32;                    /* Silero VAD frame: 512 samples / 16 kHz */
    constexpr float kVadSpeechProbThreshold = 0.3f;       /* LOWERED from 0.5 to 0.3 to catch
                                                               low-energy onsets like "ich" in
                                                               "ichthyosaurus" that the model
                                                               can decode correctly but VAD
                                                               was missing at 0.5.  Risk:
                                                               more false positives (noise
                                                               counted as speech).  Mitigation:
                                                               kVadSegmentMinValidSamples
                                                               (0.3s) + silence 1500ms still
                                                               require sustained speech before
                                                               committing.  Tune up if too
                                                               many spurious segments appear. */
    /* (kVadMinRms / kVadMinPeak energy gates intentionally
     * removed — Silero VAD prob threshold + kVadSegmentMinValidSamples
     * give cleaner segment boundaries than an energy-only gate.  The
     * VAD is the only authority on whether audio is speech.  Kept
     * values in the comment for future tuning if needed: rms 0.01f,
     * peak 0.02f.) */

    /* ==================================================================
     * VAD facade (PRODUCER): reads audio from the live buffer, runs
     * Silero VAD, decides when to commit a segment.  On commit, it
     * builds an AudioSegment and Push()es it onto the shared
     * SegmentQueue.  The ASR worker (consumer) picks it up on the
     * other side.
     *
     * The VAD facade NEVER calls qwen_transcribe_audio() — that
     * is the consumer's job.  This decoupling is the whole point
     * of the producer-consumer rewrite.
     * ================================================================== */
    auto VadFacadeLoop = [&](qwen_live_audio_t * live,
                              const std::shared_ptr<RealtimeSession> & session,
                              RealtimeLiveWorker * worker,
                              std::mutex * vad_mu_ptr,
                              std::atomic<bool> * stop_requested,
                              SegmentQueue * queue,
                              std::atomic<int64_t> * cumulative_decoded_samples,
                              const DumpConfig & dump_cfg,
                              EndpointMode ep_mode,
                              const CapOnlyEndpointPolicy & cap_pol) {
        /* Section 6.1: Select VAD pointer inside VadFacadeLoop.
         * Per-session VAD: no mutex needed (only this thread accesses it).
         * Shared fallback VAD: protected by vad_mu. */
        qwen_silero_vad_t * vad = nullptr;
        std::mutex * vad_mutex = nullptr;
        if (!worker->session_vad_fallback_shared && worker->session_vad_active &&
            worker->session_vad) {
            vad = worker->session_vad.get();
        } else {
            vad = realtime_model->vad();
            vad_mutex = vad_mu_ptr;
        }

        (void)dump_cfg; /* Used in ASR worker, not VAD loop. */
        RT_LOG("VadFacadeLoop sid=%s enter vad=%p mutex_guarded=%d",
               session->id.c_str(), (void*)vad, vad_mutex ? 1 : 0);
        const bool vad_active = qwen_silero_vad_is_active(vad);
        if (qwen_verbose >= 2) {
            std::fprintf(stderr, "VAD-facade: Silero VAD %s (%s)\n",
                         vad_active ? "active" : "inactive (will use timer only)",
                         vad_mutex ? "shared (mutex-guarded)" : "per-session (no mutex)");
        }
        /* NOTE: do NOT reset the VAD here.  For per-session VAD:
         * StartRealtimeLiveWorker already called qwen_silero_vad_reset()
         * at creation time.  For shared fallback VAD: qwen_load()
         * (qwen_asr.c:373-374) already reset it.  Calling reset() again
         * in VadFacadeLoop clears the LSTM hidden state, which means the
         * first few frames of the first segment get classified as
         * non-speech regardless of signal energy. */

        /* Convert ms-level config to sample counts. */
        const int preroll_samples = (kVadPreRollMs * 16000) / 1000;
        const int postroll_samples = (kVadPostRollMs * 16000) / 1000;

        /* State: current accumulating segment. */
        std::vector<float> segment_buffer;
        segment_buffer.reserve(kVadSegmentHardCapSamples * 2);
        PreRollBuffer preroll(preroll_samples);
        int64_t consumed_samples = 0;
        int silence_run_ms = 0;
        bool segment_active = false;
        int speech_start_offset = -1;   /* sample offset in segment_buffer of first VAD-said-speech frame; -1 = none yet */
        int last_speech_offset = -1;    /* sample offset in segment_buffer of last VAD-said-speech frame; -1 = none yet */

      /* State: softcap grace — entered when buffer exceeds soft deadline.
           * Waits for a short acoustic valley before emitting, to avoid
           * cutting mid-speech.  Max grace = kVadSegmentSoftGraceMs. */
         bool softcap_grace_active = false;
         std::int64_t softcap_grace_start_ms = 0;
         int softcap_grace_enter_silence_ms = 0;

         /* Phase 2B: cap-only latency guardrail state. */
         bool cap_grace_active = false;
         std::int64_t cap_grace_start_ms = 0;
        SegmentEndpointKind active_endpoint_kind = SegmentEndpointKind::kNormal;
          bool active_endpoint_locked = false; /* Guard flag — set at speech start, cleared after cap emit. */

        auto steady_ms = []() -> std::int64_t {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };

         /* Phase 2.1 §3.4: fill Phase 2A observability fields on any
          * AudioSegment before push.  Shared by emit_buffer and
          * emit_segment_buffer to ensure all segments have consistent
          * seq/first_segment/endpoint_mode metadata. */
        auto FillSegmentObservability = [&](AudioSegment * seg) {
            seg->seq = worker->next_segment_seq.fetch_add(1, std::memory_order_relaxed);
            seg->first_segment = !worker->first_segment_queued.load(std::memory_order_relaxed);
            seg->endpoint_mode = ep_mode == EndpointMode::kCapOnly ? "cap_only" : "legacy";
            seg->total_samples_at_push = worker->cumulative_decoded_samples.load(std::memory_order_relaxed);
            seg->queued_audio_sec = static_cast<double>(seg->samples.size()) / 16000.0;
        };

                    /* State: pending short segment (a sub-min_emit segment held in
                      * case more speech arrives within kVadShortMergeGapMs). */
                    std::vector<float> pending_buffer;
                    int pending_last_speech_offset = -1;
                    int64_t pending_expire_at_ms = 0;

                    /* Track first emit for per-session VAD observability. */
                    bool first_emit_logged = false;

                    /* Track total VAD frames processed for first-20 logging
                     * and first-speech detection.  Reset per VadFacadeLoop. */
                    int total_vad_frames = 0;

        /* Helper: build & push an AudioSegment from a vector of samples.
         * Trims trailing audio to last_speech_off + post_roll (if last_speech_off >= 0). */
        auto emit_buffer = [&](std::vector<float> * src, int last_speech_off,
                               const char * reason) -> bool {
            if (src->empty()) return true;
            int end_offset = (last_speech_off >= 0)
                ? std::min(static_cast<int>(src->size()), last_speech_off + postroll_samples)
                : static_cast<int>(src->size());
            if (end_offset <= 0) return true;
            AudioSegment seg;
            seg.commit_reason = reason;
            seg.is_eof_terminator = false;
            seg.samples.assign(src->begin(), src->begin() + end_offset);
            FillSegmentObservability(&seg);
            const bool was_first = seg.first_segment;
            const uint64_t seq = seg.seq;
            if (!first_emit_logged) {
                RT_LOG("VAD-first-emit sid=%s reason=%s audio=%.2fs silence=%dms",
                       session->id.c_str(), reason,
                       (double)seg.samples.size() / 16000.0,
                       silence_run_ms);
                first_emit_logged = true;
            }
            if (qwen_verbose >= 1) {
                std::fprintf(stderr, "VAD-facade: emit seq=%lu %.2fs reason=%s\n",
                             static_cast<unsigned long>(seq),
                             (double)seg.samples.size() / 16000.0, reason);
            }
            RT_LOG("VadFacadeLoop sid=%s emit seq=%lu reason=%s n=%zu",
                   session->id.c_str(), static_cast<unsigned long>(seq), reason, seg.samples.size());
            const bool pushed = queue->Push(std::move(seg));
            if (!pushed) {
                RT_LOG("VAD-push-failed sid=%s seq=%lu reason=%s queue_closed=1",
                       session->id.c_str(), static_cast<unsigned long>(seq), reason);
                return false;
            }
            if (was_first) {
                worker->first_segment_queued.store(true, std::memory_order_release);
            }
            src->clear();
            return true;
        };

        /* Helper: check if reason is a forced-emit (soft-cap or eof) where
          * we should emit even if VAD never saw speech. */
        auto is_forced_emit_reason = [](const char * reason) -> bool {
            return std::strcmp(reason, "softcap_valley") == 0 ||
                   std::strcmp(reason, "softcap_forced") == 0 ||
                   std::strcmp(reason, "softcap_hard") == 0 ||
                   std::strcmp(reason, "eof") == 0 ||
                   std::strcmp(reason, "eof_stop") == 0;
        };

       /* Helper: build & push an AudioSegment from segment_buffer,
          * with tail_audio (from previous segment) + pre-roll prepended
          * and trailing silence trimmed to last_speech + post_roll.
          *
          * §7.2 #7: tail_audio from the previous segment's uncertain
          * tail is prepended to the current segment's audio so the ASR
          * model has full context for the incomplete sentence. */
         auto emit_segment_buffer = [&](const char * reason) -> bool {
             if (segment_buffer.empty()) {
                 silence_run_ms = 0;
                 segment_active = false;
                 speech_start_offset = -1;
                 last_speech_offset = -1;

                 softcap_grace_active = false;
                 softcap_grace_start_ms = 0;
                 softcap_grace_enter_silence_ms = 0;

                 return true;
             }
             std::vector<float> emit_samples;

             /* §7.2 #7: Prepend tail_audio from previous segment's
              * uncertain tail.  This gives the ASR model context for
              * the incomplete sentence. */
             {
                 std::lock_guard<std::mutex> tlock(session->mu);
                 if (!session->tail_audio.empty()) {
                     emit_samples.insert(emit_samples.end(),
                                         session->tail_audio.begin(),
                                         session->tail_audio.end());
                     RT_LOG("VadFacadeLoop sid=%s prepend tail=%d samples (%.2fs)",
                            session->id.c_str(),
                            static_cast<int>(session->tail_audio.size()),
                            session->tail_audio.size() / 16000.0);
                     session->tail_audio.clear();
                 }
             }

             if (last_speech_offset >= 0) {
                 int end_offset = std::min(static_cast<int>(segment_buffer.size()),
                                           last_speech_offset + postroll_samples);
                 int start_offset = (speech_start_offset >= 0)
                     ? std::max(0, speech_start_offset - preroll_samples)
                     : 0;
                 if (end_offset > start_offset) {
                     /* Prepend pre-roll (last `start_offset` samples of
                      * the preroll ring buffer) if non-empty.  This
                      * gives ASR the audio BEFORE the first VAD-said-
                      * speech frame so leading consonants are preserved.
                      */
                     if (start_offset > 0) {
                         std::vector<float> pr;
                         preroll.Snapshot(&pr);
                         if (static_cast<int>(pr.size()) > start_offset) {
                             emit_samples.insert(emit_samples.end(),
                                                 pr.end() - start_offset, pr.end());
                        } else {
                            emit_samples.insert(emit_samples.end(),
                                                pr.begin(), pr.end());
                        }
                    }
                    emit_samples.insert(emit_samples.end(),
                                        segment_buffer.begin() + start_offset,
                                        segment_buffer.begin() + end_offset);
                }
            } else if (is_forced_emit_reason(reason)) {
                /* Soft-cap or stop: even if VAD never saw speech, emit
                 * whatever audio we have.  The ASR may still produce
                 * text (e.g. low-volume speech the VAD missed, or the
                 * user just wants to see *something*).
                 * Preserve any prepended tail_audio from previous segment
                 * (appended above at line 4427-4443) — do NOT overwrite. */
                emit_samples.insert(emit_samples.end(),
                                    segment_buffer.begin(), segment_buffer.end());
            } else {
                /* No VAD-said-speech at all — emit empty (will discard). */
                emit_samples.clear();
            }
            if (emit_samples.empty()) {
                segment_buffer.clear();
                silence_run_ms = 0;
                segment_active = false;
                speech_start_offset = -1;
                last_speech_offset = -1;

                softcap_grace_active = false;
                softcap_grace_start_ms = 0;
                softcap_grace_enter_silence_ms = 0;

                return true;
            }
            if (qwen_verbose >= 1) {
                std::fprintf(stderr,
                    "VAD-facade: emit %.2fs reason=%s (pre_roll=%dms post_roll=%dms)\n",
                    (double)emit_samples.size() / 16000.0, reason,
                    kVadPreRollMs, kVadPostRollMs);
            }
          if (!first_emit_logged) {
                RT_LOG("VAD-first-emit sid=%s reason=%s audio=%.2fs silence=%dms",
                       session->id.c_str(), reason,
                       (double)emit_samples.size() / 16000.0,
                       silence_run_ms);
                first_emit_logged = true;
            }
            AudioSegment seg;
            seg.samples = std::move(emit_samples);
            seg.commit_reason = reason;
            seg.is_eof_terminator = false;
            FillSegmentObservability(&seg);
            const bool was_first = seg.first_segment;
            const uint64_t seg_seq = seg.seq;
            RT_LOG("VadFacadeLoop sid=%s emit seq=%lu reason=%s n=%zu",
                   session->id.c_str(), static_cast<unsigned long>(seg_seq), reason, seg.samples.size());
            const bool pushed = queue->Push(std::move(seg));
            if (!pushed) {
                RT_LOG("VAD-push-failed sid=%s seq=%lu reason=%s queue_closed=1",
                       session->id.c_str(), static_cast<unsigned long>(seg_seq), reason);
                return false;
            }
            if (was_first) {
                worker->first_segment_queued.store(true, std::memory_order_release);
            }
            segment_buffer.clear();
            segment_buffer.reserve(kVadSegmentHardCapSamples * 2);
            silence_run_ms = 0;
            segment_active = false;
            speech_start_offset = -1;
            last_speech_offset = -1;

            softcap_grace_active = false;
            softcap_grace_start_ms = 0;
            softcap_grace_enter_silence_ms = 0;

            return true;
        };

        int idle_polls = 0;
        int poll_count = 0;
        while (true) {
            /* Check stop_requested FIRST so a pending /stop is
             * responsive even if the queue is full.
             *
             * IMPORTANT: do NOT just `break` here.  If the user clicked
             * Stop within the VAD silence window (1.5 s), the natural
             * commit path never fired and the segment_buffer still
             * holds the just-pushed audio.  Dropping it on break loses
             * the user's speech — symptom: "second session outputs no
             * text".  Instead, FLUSH both pending_buffer and
             * segment_buffer, push a poison pill, and only then exit.
             * The poison pill is what makes the producer/consumer
             * shutdown protocol safe: the ASR worker drains remaining
             * items, sees the pill, and exits cleanly.
             *
             * ALSO IMPORTANT: if the user pushed audio and immediately
             * called /stop, the VAD's main-loop drain path may never
             * have read live->samples (e.g., a short push + stop that
             * happened between two 50 ms polls).  We must DRAIN any
             * remaining live audio into segment_buffer (and run VAD on
             * it) BEFORE flushing — otherwise the user's last words are
             * silently dropped, producing the "no text comes out"
             * symptom.  Symptom: 3 s push + immediate stop → consumed=0
             * → no segment → empty text.  Fix: drain live->samples on
             * the stop path. */
            if (stop_requested->load(std::memory_order_acquire)) {
                RT_LOG("VadFacadeLoop sid=%s stop_requested, draining live", session->id.c_str());
                /* Drain any remaining audio in live->samples into
                 * segment_buffer (and run VAD on the new tail) so
                 * audio pushed between the last VAD poll and /stop is
                 * not lost.  We deliberately do NOT merge pending into
                 * the drained audio here — the flush below will emit
                 * pending separately to preserve order. */
                {
                    int64_t live_n_drain = 0;
                    {
                        LockLiveAudio(live);
                        live_n_drain = live->n_samples;
                        UnlockLiveAudio(live);
                    }
                    if (live_n_drain > consumed_samples) {
                        int64_t take_drain = live_n_drain - consumed_samples;
                        std::vector<float> tail(static_cast<std::size_t>(take_drain));
                        {
                            LockLiveAudio(live);
                            std::memcpy(tail.data(),
                                        live->samples + consumed_samples,
                                        static_cast<std::size_t>(take_drain) * sizeof(float));
                            UnlockLiveAudio(live);
                        }
                        preroll.Push(tail.data(), static_cast<int>(tail.size()));
                        /* No merge: stop is final, so just append. */
                        segment_buffer.insert(segment_buffer.end(),
                                              tail.begin(), tail.end());
                        consumed_samples += take_drain;
                        /* Accumulate the cumulative count for the
                         * snapshot (see comment in main-loop path). */
                        if (cumulative_decoded_samples) {
                            cumulative_decoded_samples->fetch_add(take_drain, std::memory_order_relaxed);
                        }
                        TrimConsumedLiveAudio(live, consumed_samples);
                        /* Run VAD on the drained tail to update
                         * last_speech_offset / speech_start_offset /
                         * silence_run_ms so emit_segment_buffer
                         * correctly trims trailing silence. */
                        const int frame_drain = 512;
                        int64_t tail_start = static_cast<int64_t>(segment_buffer.size()) - take_drain;
                        if (tail_start < 0) tail_start = 0;
                        int64_t aligned_start = (tail_start / frame_drain) * frame_drain;
                        int64_t total = static_cast<int64_t>(segment_buffer.size());
                        for (int64_t off = aligned_start + frame_drain; off <= total; off += frame_drain) {
                            float prob = 1.0f;  /* fail-open: treat as speech on error */
                            if (vad == nullptr || !qwen_silero_vad_is_active(vad)) {
                                /* VAD not active — fail-open */
                            } else if (vad_mutex) {
                                std::lock_guard<std::mutex> lock(*vad_mutex);
                                if (qwen_silero_vad_process(vad,
                                    segment_buffer.data() + (off - frame_drain),
                                    frame_drain, &prob) != 0) {
                                    prob = 1.0f;  /* fail-open */
                                    if (qwen_verbose >= 1) {
                                        RT_LOG("VAD-process-failed sid=%s rc=-1 drain",
                                               session->id.c_str());
                                    }
                                }
                            } else {
                                if (qwen_silero_vad_process(vad,
                                    segment_buffer.data() + (off - frame_drain),
                                    frame_drain, &prob) != 0) {
                                    prob = 1.0f;  /* fail-open */
                                    if (qwen_verbose >= 1) {
                                        RT_LOG("VAD-process-failed sid=%s rc=-1 drain",
                                               session->id.c_str());
                                    }
                                }
                            }
                            total_vad_frames++;
                            if (prob >= kVadSpeechProbThreshold) {
                                segment_active = true;
                                silence_run_ms = 0;
                                if (speech_start_offset < 0) {
                                    speech_start_offset = static_cast<int>(off - frame_drain);
                                    /* Phase 2B §6.5: lock endpoint kind at speech start. */
                                    const bool is_first =
                                        !worker->first_segment_queued.load(std::memory_order_relaxed);
                                    active_endpoint_kind = is_first
                                        ? SegmentEndpointKind::kFirst
                                        : SegmentEndpointKind::kNormal;
                                    active_endpoint_locked = true;
                                }
                                last_speech_offset = static_cast<int>(off);
                            } else {
                                if (segment_active) {
                                    silence_run_ms += kVadVadFrameMs;
                                }
                            }
                        }
                        RT_LOG("VadFacadeLoop sid=%s stop_drain took %lld samples, buf=%.2fs last_speech=%d",
                               session->id.c_str(), (long long)take_drain,
                               (double)segment_buffer.size() / 16000.0, last_speech_offset);
                    }
                }
                RT_LOG("VadFacadeLoop sid=%s stop_requested, flushing", session->id.c_str());
                if (!pending_buffer.empty()) {
                    RT_LOG("VadFacadeLoop sid=%s flush pending (%.2fs)",
                           session->id.c_str(),
                           (double)pending_buffer.size() / 16000.0);
                    if (!emit_buffer(&pending_buffer,
                                     pending_last_speech_offset,
                                     "eof_stop")) {
                        RT_LOG("VadFacadeLoop sid=%s flush pending push failed (queue closed)",
                               session->id.c_str());
                        break;
                    }
                    pending_last_speech_offset = -1;
                    pending_expire_at_ms = 0;
                }
                if (!segment_buffer.empty()) {
                    RT_LOG("VadFacadeLoop sid=%s flush segment (%.2fs)",
                           session->id.c_str(),
                           (double)segment_buffer.size() / 16000.0);
                    if (!emit_segment_buffer("eof_stop")) {
                        RT_LOG("VadFacadeLoop sid=%s flush segment push failed (queue closed)",
                               session->id.c_str());
                        break;
                    }
                }
                /* Poison pill: signals the ASR worker to exit.  The
                 * segment_queue must NOT be closed yet (JoinRealtimeLiveWorker
                 * orders Close() AFTER vad_thread.join()).  If it is
                 * closed (e.g., a future refactor breaks the order),
                 * Push returns false and the ASR will see the close
                 * predicate on Pop and exit naturally. */
                AudioSegment eof;
                eof.is_eof_terminator = true;
                const bool pushed = queue->Push(std::move(eof));
                if (!pushed) {
                    RT_LOG("VadFacadeLoop sid=%s poison pill push failed (queue closed)",
                           session->id.c_str());
                }
                RT_LOG("VadFacadeLoop sid=%s stop_requested, exiting", session->id.c_str());
                break;
            }

            /* Pending expiry check: if pending exists and merge window
             * elapsed without new speech, emit pending as-is. */
            if (!pending_buffer.empty()) {
                const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                if (now_ms >= pending_expire_at_ms) {
                    if (qwen_verbose >= 1) {
                        std::fprintf(stderr,
                            "VAD-facade: pending short segment (%.2fs) expired, emitting\n",
                            (double)pending_buffer.size() / 16000.0);
                    }
                    if (!emit_buffer(&pending_buffer, pending_last_speech_offset, "pending_expired")) {
                        break;
                    }
                    pending_last_speech_offset = -1;
                    pending_expire_at_ms = 0;
                }
            }

            poll_count++;
            if (qwen_verbose >= 1 && (poll_count % 50) == 0) {
                std::fprintf(stderr,
                    "[rt t=%.0fms] VadFacadeLoop heartbeat sid=%s poll=%d consumed=%lld "
                    "segment_buf=%zu pending=%zu queue=%zu\n",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count(),
                    session->id.c_str(), poll_count, (long long)consumed_samples,
                    segment_buffer.size(), pending_buffer.size(), queue->Size());
            }

            int64_t live_n = 0;
            int live_eof = 0;
            {
                LockLiveAudio(live);
                live_n = live->n_samples;
                live_eof = live->eof;
                UnlockLiveAudio(live);
            }

            if (live_n > consumed_samples) {
                int64_t take = live_n - consumed_samples;
                if (take > 0) {
                    /* Copy new audio out from under live->mu once. */
                    std::vector<float> new_samples(static_cast<std::size_t>(take));
                    {
                        LockLiveAudio(live);
                        std::memcpy(new_samples.data(),
                                    live->samples + consumed_samples,
                                    static_cast<std::size_t>(take) * sizeof(float));
                        UnlockLiveAudio(live);
                    }

                    /* Update pre-roll ring buffer. */
                    preroll.Push(new_samples.data(),
                                 static_cast<int>(new_samples.size()));

                    /* If we have a pending short segment, MERGE: prepend
                     * it to the new audio so the combined audio forms
                     * one segment.  After merge, clear pending. */
                    if (!pending_buffer.empty()) {
                        const int shift = static_cast<int>(pending_buffer.size());
                        std::vector<float> merged;
                        merged.reserve(pending_buffer.size() + new_samples.size());
                        merged.insert(merged.end(),
                                      pending_buffer.begin(), pending_buffer.end());
                        merged.insert(merged.end(),
                                      new_samples.begin(), new_samples.end());
                        segment_buffer = std::move(merged);
                        /* Inherit last_speech from pending if no new
                         * speech on this poll's audio yet. */
                        if (last_speech_offset < 0 && pending_last_speech_offset >= 0) {
                            last_speech_offset = shift + pending_last_speech_offset;
                        }
                        if (speech_start_offset < 0 && pending_last_speech_offset >= 0) {
                            speech_start_offset = shift;  /* pending started at 0 */
                        }
                        if (qwen_verbose >= 1) {
                            std::fprintf(stderr,
                                "VAD-facade: merged pending (%.2fs) into new audio → buf=%.2fs\n",
                                (double)shift / 16000.0,
                                (double)segment_buffer.size() / 16000.0);
                        }
                        pending_buffer.clear();
                        pending_last_speech_offset = -1;
                        pending_expire_at_ms = 0;
                    } else {
                        /* No merge: just append new audio. */
                        segment_buffer.insert(segment_buffer.end(),
                                              new_samples.begin(), new_samples.end());
                    }
                    consumed_samples += take;
                    new_samples.clear();
                    /* Accumulate the cumulative count BEFORE trim so
                     * the snapshot can report the true total even
                     * after TrimConsumedLiveAudio resets the live
                     * cursor to 0.  See RealtimeLiveWorker::
                     * cumulative_decoded_samples for the rationale. */
                    if (cumulative_decoded_samples) {
                        cumulative_decoded_samples->fetch_add(take, std::memory_order_relaxed);
                    }

                    /* Periodically reclaim the consumed prefix of the
                     * live audio buffer so a 1-hour session does not
                     * OOM at ~230 MB / session.  The trim is a single
                     * memmove + realloc under live->mu; with the
                     * threshold at 1.6M samples (6.4 MB) it runs at
                     * most a few times per minute.  See
                     * TrimConsumedLiveAudio for cost analysis. */
                    TrimConsumedLiveAudio(live, consumed_samples);

                    /* Run VAD on the new tail of segment_buffer in
                     * 512-sample frames.  NO energy gate — VAD decides.
                     */
                    const int frame = 512;
                    int64_t tail_start = static_cast<int64_t>(segment_buffer.size()) - take;
                    if (tail_start < 0) tail_start = 0;
                    int64_t aligned_start = (tail_start / frame) * frame;
                    int64_t total = static_cast<int64_t>(segment_buffer.size());
                    int frames_processed = 0;
                    float last_prob = 0.0f;
                    /* Track first-speech frame for per-session VAD
                     * observability.  Only emitted once per VadFacadeLoop. */
                    int first_speech_frame = -1;
                    int64_t first_speech_abs_sample = -1;
                    for (int64_t off = aligned_start + frame; off <= total; off += frame) {
                        float prob = 1.0f;  /* fail-open: treat as speech on error */
                        if (vad == nullptr || !qwen_silero_vad_is_active(vad)) {
                            /* VAD not active — fail-open */
                        } else if (vad_mutex) {
                            std::lock_guard<std::mutex> lock(*vad_mutex);
                            if (qwen_silero_vad_process(vad,
                                segment_buffer.data() + (off - frame),
                                frame, &prob) != 0) {
                                prob = 1.0f;  /* fail-open */
                                if (qwen_verbose >= 1) {
                                    RT_LOG("VAD-process-failed sid=%s rc=-1",
                                           session->id.c_str());
                                }
                            }
                        } else {
                            if (qwen_silero_vad_process(vad,
                                segment_buffer.data() + (off - frame),
                                frame, &prob) != 0) {
                                prob = 1.0f;  /* fail-open */
                                if (qwen_verbose >= 1) {
                                    RT_LOG("VAD-process-failed sid=%s rc=-1",
                                           session->id.c_str());
                                }
                            }
                        }
                        last_prob = prob;
                        frames_processed++;
                        total_vad_frames++;
                        /* First 20 frames VAD prob logging (document 10.2).
                         * Only in verbose mode to avoid log spam. */
                        if (total_vad_frames <= 20 && qwen_verbose >= 2) {
                            std::fprintf(stderr,
                                "VAD-frame sid=%s idx=%d prob=%.3f speech=%s\n",
                                session->id.c_str(),
                                total_vad_frames - 1,
                                prob,
                                prob >= kVadSpeechProbThreshold ? "true" : "false");
                        }
                        if (prob >= kVadSpeechProbThreshold) {
                            segment_active = true;
                            silence_run_ms = 0;
                            if (speech_start_offset < 0) {
                                speech_start_offset = static_cast<int>(off - frame);
                                /* First speech frame detected in this
                                 * VadFacadeLoop — log it for debugging. */
                                first_speech_frame = frames_processed;
                                first_speech_abs_sample = consumed_samples + (off - frame);
                                /* Phase 2B §6.5: lock endpoint kind at speech
                                 * start so cap logic uses correct threshold. */
                                const bool is_first =
                                    !worker->first_segment_queued.load(std::memory_order_relaxed);
                                active_endpoint_kind = is_first
                                    ? SegmentEndpointKind::kFirst
                                    : SegmentEndpointKind::kNormal;
                                active_endpoint_locked = true;
                            }
                            last_speech_offset = static_cast<int>(off);
                        } else {
                            if (segment_active) {
                                silence_run_ms += kVadVadFrameMs;
                            }
                        }
                    }
                    if (first_speech_frame >= 0 && qwen_verbose >= 1) {
                        RT_LOG("VAD-first-speech sid=%s frame=%d abs_sample=%lld prob=%.3f",
                               session->id.c_str(),
                               first_speech_frame,
                               (long long)first_speech_abs_sample,
                               last_prob);
                    }
                    if (qwen_verbose >= 2 && frames_processed > 0) {
                        std::fprintf(stderr,
                            "VAD-facade: take=%lld frames=%d "
                            "active=%d silence_run=%dms last_prob=%.3f "
                            "buf=%.2fs speech=[%d,%d]\n",
                            static_cast<long long>(take),
                            frames_processed,
                            segment_active ? 1 : 0,
                            silence_run_ms,
                            last_prob,
                            (double)segment_buffer.size() / 16000.0,
                            speech_start_offset, last_speech_offset);
                    }
                    idle_polls = 0;
                }
            } else {
                idle_polls++;
                /* Silence grows during idle polls ONLY if we already
                 * had speech and VAD hasn't said otherwise.  This
                 * matches the C-level VAD fix in qwen_asr.c. */
                if (segment_active) {
                    silence_run_ms += kVadSegmentPollMs;
                }
                if (qwen_verbose >= 2 && idle_polls % 5 == 0) {
                    std::fprintf(stderr,
                        "VAD-facade: idle_polls=%d silence_run=%dms active=%d "
                        "buf=%.2fs pending=%.2fs queue=%zu\n",
                        idle_polls, silence_run_ms, segment_active ? 1 : 0,
                        (double)segment_buffer.size() / 16000.0,
                        (double)pending_buffer.size() / 16000.0,
                        queue->Size());
                }
            }

            {
                std::lock_guard<std::mutex> lock(session->mu);
                session->current_segment_audio_sec =
                    static_cast<double>(segment_buffer.size()) / 16000.0;
            }

            const int buf_samples = static_cast<int>(segment_buffer.size());
            const bool silence_elapsed = segment_active && silence_run_ms >= kVadSegmentSilenceMs;
            const bool buf_long_enough = buf_samples >= kVadSegmentMinEmitSamples;
            const bool buf_min_valid = buf_samples >= kVadSegmentMinValidSamples;
            const bool buf_soft_deadline = buf_samples >= kVadSegmentSoftDeadlineSamples;

            /* Commit / save-as-pending decision. */
            if (silence_elapsed && buf_long_enough && pending_buffer.empty()) {
                /* Normal commit: long enough segment, no pending. */
                if (qwen_verbose >= 2) {
                    std::fprintf(stderr,
                        "VAD-facade: commit (silence=%dms >= %dms, buf=%.2fs)\n",
                        silence_run_ms, kVadSegmentSilenceMs,
                        (double)buf_samples / 16000.0);
                }
                if (!emit_segment_buffer("vad_silence")) {
                    break;
                }
           } else if (silence_elapsed && buf_min_valid && pending_buffer.empty()) {
                /* Short segment (>= min_valid, < min_emit), no pending
                 * yet: save as pending for possible merge. */
                pending_buffer = std::move(segment_buffer);
                pending_last_speech_offset = last_speech_offset;
                const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                pending_expire_at_ms = now_ms + kVadShortMergeGapMs;
                if (qwen_verbose >= 1) {
                    std::fprintf(stderr,
                        "VAD-facade: short segment (%.2fs) saved as pending, expires in %dms\n",
                        (double)pending_buffer.size() / 16000.0, kVadShortMergeGapMs);
                }
                segment_buffer.clear();
                segment_buffer.reserve(kVadSegmentHardCapSamples * 2);
                silence_run_ms = 0;
                segment_active = false;
                speech_start_offset = -1;
                last_speech_offset = -1;

                softcap_grace_active = false;
                softcap_grace_start_ms = 0;
                softcap_grace_enter_silence_ms = 0;

        } else if (ep_mode == EndpointMode::kCapOnly) {
                 /* Phase 2B: cap-only latency guardrail.
                  * Only active when QASR_ENDPOINT_MODE=cap_only.
                  * Does not change vad_silence / min_emit / pending behavior. */
                 const int cap_ms = active_endpoint_kind == SegmentEndpointKind::kFirst
                     ? cap_pol.first_latency_cap_ms
                     : cap_pol.normal_latency_cap_ms;
                 const int cap_samples = cap_ms * 16000 / 1000;
                 const bool latency_cap_reached =
                     segment_active && buf_samples >= cap_samples;

                 if (latency_cap_reached && !cap_grace_active) {
                     cap_grace_active = true;
                     cap_grace_start_ms = steady_ms();
                     if (qwen_verbose >= 1) {
                         std::fprintf(stderr,
                             "VAD-facade: cap_only grace enter buf=%.2fs cap=%dms kind=%s\n",
                             (double)buf_samples / 16000.0,
                             cap_ms,
                             active_endpoint_kind == SegmentEndpointKind::kFirst ? "first" : "normal");
                     }
                 }

                 if (cap_grace_active) {
                      const int64_t now_ms_val = steady_ms();
                      const bool cap_valley =
                          segment_active && silence_run_ms >= cap_pol.cap_valley_silence_ms;
                      const bool cap_grace_expired =
                          now_ms_val - cap_grace_start_ms >= cap_pol.cap_grace_ms;

                      if (cap_valley || cap_grace_expired) {
                          const char * cap_reason = cap_valley
                              ? "latency_cap_valley"
                              : "latency_cap_forced";
                          /* Phase 2.1 §3.5: flush pending before cap emit to
                           * maintain segment order consistency, matching softcap. */
                          if (!pending_buffer.empty()) {
                              if (!emit_buffer(&pending_buffer,
                                               pending_last_speech_offset,
                                               "pending_pre_latency_cap")) {
                                  break;
                              }
                              pending_last_speech_offset = -1;
                              pending_expire_at_ms = 0;
                          }
                          if (qwen_verbose >= 1) {
                              std::fprintf(stderr,
                                  "VAD-facade: cap_only emit reason=%s buf=%.2fs\n",
                                  cap_reason, (double)buf_samples / 16000.0);
                          }
                          if (!emit_segment_buffer(cap_reason)) {
                              break;
                          }
                          cap_grace_active = false;
                          cap_grace_start_ms = 0;
                          active_endpoint_locked = false;
                          continue;
                      }
                 }
             } else if (buf_soft_deadline) {
                 /* Soft-cap grace: buffer reached soft deadline.
                  * Enter grace period — wait for a short acoustic valley
                  * (>= kVadSoftcapValleySilenceMs) or grace expiry or hard cap. */
                {
                    const int64_t now_ms = steady_ms();
                    const bool buf_hard_cap = buf_samples >= kVadSegmentHardCapSamples;

                    if (!softcap_grace_active) {
                        softcap_grace_active = true;
                        softcap_grace_start_ms = now_ms;
                        softcap_grace_enter_silence_ms = silence_run_ms;

                        if (qwen_verbose >= 1) {
                            std::fprintf(stderr,
                                "VAD-facade: softcap grace enter buf=%.2fs "
                                "silence_run=%dms grace=%dms hard=%.2fs\n",
                                (double)buf_samples / 16000.0,
                                silence_run_ms,
                                kVadSegmentSoftGraceMs,
                                (double)kVadSegmentHardCapSamples / 16000.0);
                        }

                        RT_LOG("VadFacadeLoop sid=%s softcap_grace_enter buf=%.2fs silence=%d",
                               session->id.c_str(),
                               (double)buf_samples / 16000.0,
                               silence_run_ms);
                    }

                    const int64_t grace_wait_ms = now_ms - softcap_grace_start_ms;

                    /* Low-risk cut: after soft deadline, a short non-speech
                     * valley is enough to avoid cutting directly inside active
                     * speech.  We do NOT wait for the full 1500ms vad_silence. */
                    const bool valley_cut =
                        segment_active && silence_run_ms >= kVadSoftcapValleySilenceMs;

                    const bool grace_expired =
                        grace_wait_ms >= kVadSegmentSoftGraceMs;

                    const bool should_emit_softcap =
                         valley_cut || grace_expired || buf_hard_cap;

                    if (should_emit_softcap) {
                        const char * reason =
                            valley_cut ? "softcap_valley"
                                : (buf_hard_cap ? "softcap_hard" : "softcap_forced");

                        if (qwen_verbose >= 1) {
                            std::fprintf(stderr,
                                "VAD-facade: softcap emit reason=%s buf=%.2fs "
                                "grace_wait=%lldms silence_run=%dms\n",
                                reason,
                                (double)buf_samples / 16000.0,
                                (long long)grace_wait_ms,
                                silence_run_ms);
                        }

                        RT_LOG("VadFacadeLoop sid=%s softcap_emit reason=%s buf=%.2fs "
                               "grace_wait=%lld silence=%d queue=%zu",
                               session->id.c_str(),
                               reason,
                               (double)buf_samples / 16000.0,
                               (long long)grace_wait_ms,
                               silence_run_ms,
                               queue->Size());

                        /* If we have a pending, emit it first to preserve order. */
                        if (!pending_buffer.empty()) {
                            if (!emit_buffer(&pending_buffer, pending_last_speech_offset, "pending_pre_softcap")) {
                                break;
                            }
                            pending_last_speech_offset = -1;
                            pending_expire_at_ms = 0;
                        }

                        if (!emit_segment_buffer(reason)) {
                            break;
                        }
                    }
                }
            }

            /* §7.2 #9: Idle silence fallback — after ~0.5s of idle
             * (no speech, no active segment), force-commit pending
             * candidates.  This ensures short utterances like "你好。"
             * are finalized even if the user stops speaking. */
            if (idle_polls > 10 && !session->finalized) {
                std::lock_guard<std::mutex> lock(session->mu);
                if (!session->candidates.empty() && !session->finalized) {
                    for (auto & c : session->candidates) {
                        session->segments_text.push_back(std::move(c));
                    }
                    session->candidates.clear();
                    session->sse_cv.notify_all();
                    RT_LOG("VadFacadeLoop sid=%s idle_silence_commit "
                           "segments=%zu",
                           session->id.c_str(),
                           session->segments_text.size());
                }
            }

            /* EOF from the live buffer (set by /stop or eof endpoint).
             * Flush any in-flight segment and pending, then exit. */
            if (live_eof && consumed_samples > 0) {
                if (!pending_buffer.empty()) {
                    if (!emit_buffer(&pending_buffer, pending_last_speech_offset, "eof")) {
                        break;
                    }
                    pending_last_speech_offset = -1;
                    pending_expire_at_ms = 0;
                }
                if (!segment_buffer.empty()) {
                    if (!emit_segment_buffer("eof")) {
                        break;
                    }
                }
                RT_LOG("VadFacadeLoop sid=%s BREAK live_eof=1", session->id.c_str());
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(kVadSegmentPollMs));
        }
        RT_LOG("VadFacadeLoop sid=%s loop EXIT consumed=%lld", session->id.c_str(), (long long)consumed_samples);
    };

    /* ==================================================================
     * ASR worker (CONSUMER): pops AudioSegments from the queue, runs
     * qwen_transcribe_audio() on each, and updates the session state
     * with the resulting text.  Exits when it sees the EOF terminator
     * OR when Pop() returns false (queue closed and empty).
     *
     * On exit, marks the session as finalized and frees the live_ctx
     * clone.  The HTTP /stop handler waits on this thread via join.
     * ================================================================== */
  auto AsrWorkerLoop = [&](qwen_ctx_t * live_ctx,
                                const std::shared_ptr<RealtimeSession> & session,
                                const std::string & forced_language,
                                SegmentQueue * queue,
                                const DumpConfig & dump_cfg,
                                ServerAsrFacade * facade) {
        RT_LOG("AsrWorkerLoop sid=%s enter", session->id.c_str());
        qwen_verbose = realtime_model->verbosity();
        bool gpuPipeline = live_ctx == nullptr;
        std::uint64_t gpuSessionId = 0;

        if (!gpuPipeline) {
            /* Configure ctx for offline-style per-segment decode.  No
             * streaming chunk cadence, no coalesce step, no partial. */
            live_ctx->segment_sec = 0.0f;
            live_ctx->search_sec = 0.0f;
            live_ctx->past_text_conditioning = 0;
            live_ctx->stream_chunk_sec = 0.0f;
            live_ctx->stream_max_new_tokens = 0;
            if (qwen_set_force_language(live_ctx,
                                        forced_language.empty() ? nullptr
                                                                : forced_language.c_str()) != 0) {
                std::lock_guard<std::mutex> lock(session->mu);
                session->error = "unsupported language: " + forced_language;
                session->worker_done = true;
                qwen_free(live_ctx);
                RT_LOG("AsrWorkerLoop sid=%s END (set_force_language failed)", session->id.c_str());
                return;
            }
            if (qwen_set_prompt(live_ctx, nullptr) != 0) {
                std::lock_guard<std::mutex> lock(session->mu);
                session->error = "failed to set realtime prompt";
                session->worker_done = true;
                qwen_free(live_ctx);
                RT_LOG("AsrWorkerLoop sid=%s END (set_prompt failed)", session->id.c_str());
                return;
            }
        } else if (facade) {
            /* GPU path: acquire persistent engine session for this realtime session. */
            AsrEngine * eng = facade->engine();
            if (eng) {
                qasr::SessionOptions gpuOpts;
                gpuOpts.language = forced_language;
                gpuOpts.prompt = "";
                gpuOpts.temperature = facade->temperature();
                gpuOpts.realtime = true;
                if (eng->CreateSession(gpuOpts, gpuSessionId).ok()) {
                    RT_LOG("AsrWorkerLoop sid=%s GPU session created esid=%lu",
                           session->id.c_str(),
                           static_cast<unsigned long>(gpuSessionId));
                } else {
                    RT_LOG("AsrWorkerLoop sid=%s GPU session creation failed",
                           session->id.c_str());
                }
            }
        }

/* Store pipeline config for session state. */
        {
            std::lock_guard<std::mutex> lock(session->mu);
            session->gpuPipeline = gpuPipeline;
            session->facade = facade;
            session->gpuSessionId = gpuSessionId;
        }

        while (true) {
            AudioSegment seg;
            if (!queue->Pop(&seg)) {
                /* Queue closed and empty — natural exit. */
                break;
            }
            if (seg.is_eof_terminator) {
                /* Poison pill: producer says no more. */
                break;
            }
            const int n_samples = static_cast<int>(seg.samples.size());
            /* Phase 2A: ASR worker logging with seq/reason/forced_language. */
            RT_LOG("ASR-worker sid=%s seq=%lu reason=%s audio=%.2fs lang=%s forced=%s first=%d",
                   session->id.c_str(),
                   static_cast<unsigned long>(seg.seq),
                   seg.commit_reason.c_str(),
                   (double)n_samples / 16000.0,
                   session->language.c_str(),
                   forced_language.c_str(),
                   seg.first_segment ? 1 : 0);

            if (qwen_verbose >= 1) {
                std::fprintf(stderr,
                              "ASR-worker: decoding candidate #%d (%.2fs, reason=%s, total_buf=%.2fs)\n",
                              (int)session->candidates.size() + 1,
                             (double)n_samples / 16000.0, seg.commit_reason.c_str(),
                             (double)session->total_samples / 16000.0);
            }
     RT_LOG("AsrWorkerLoop sid=%s decoding n=%d reason=%s",
                    session->id.c_str(), n_samples, seg.commit_reason.c_str());

            /* CPU: C bridge on live_ctx clone.
             * GPU:  scheduler.SubmitAndAwait or direct TranscribeSegment. */
            std::string asr_text;
            double seg_total_ms = 0, seg_encode_ms = 0, seg_decode_ms = 0;
            int seg_tokens = 0;
            if (gpuPipeline && facade) {
                GpuScheduler * sched = facade->scheduler();
                if (sched) {
                    /* Use scheduler for multi-session concurrency.
                     * SubmitAndAwait preserves per-session ordering while
                     * allowing other sessions' segments to process in parallel. */
                    SegmentJob job;
                    job.session_id = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(session.get()));
                    job.segment_id = seg.seq;
                    job.samples = seg.samples;
                    job.sample_rate = 16000;
                    job.realtime = true;
                    job.language = forced_language;
                    job.prompt = "";
                    job.engine_session_id = gpuSessionId;
                    job.enqueue_time = std::chrono::steady_clock::now();
                    SegmentResult res = sched->SubmitAndAwait(job);
                    if (res.status.ok()) {
                        asr_text = res.text;
                        seg_total_ms = res.total_ms;
                        seg_encode_ms = res.encode_ms;
                        seg_decode_ms = res.decode_ms;
                        seg_tokens = res.tokens;
                    }
                } else if (gpuSessionId) {
                    /* Fallback: direct engine call (no scheduler). */
                    AsrEngine * eng = facade->engine();
                    AsrSegmentResult segRes = eng->TranscribeSegment(
                        gpuSessionId, seg.samples, 16000);
                    if (segRes.status.ok()) {
                        asr_text = segRes.text;
                        seg_total_ms = segRes.total_ms;
                        seg_encode_ms = segRes.encode_ms;
                        seg_decode_ms = segRes.decode_ms;
                        seg_tokens = segRes.text_tokens;
                    }
                }
            } else if (!gpuPipeline) {
                char * raw = qwen_transcribe_audio(live_ctx, seg.samples.data(), n_samples);
                RT_LOG("AsrWorkerLoop sid=%s qwen_transcribe_audio RETURNED", session->id.c_str());
                if (raw != nullptr) {
                    asr_text.assign(raw);
                    std::free(raw);
                    while (!asr_text.empty() &&
                           (asr_text.back() == ' ' || asr_text.back() == '\n' || asr_text.back() == '\t')) {
                        asr_text.pop_back();
                    }
                    seg_total_ms = live_ctx->perf_total_ms;
                    seg_encode_ms = live_ctx->perf_encode_ms;
                    seg_decode_ms = live_ctx->perf_decode_ms;
                    seg_tokens = live_ctx->perf_text_tokens;
                }
            }
            const bool text_empty = asr_text.empty();
            const bool raw_null = asr_text.empty() && seg_total_ms == 0;

            RT_LOG("ASR-worker-result sid=%s seq=%lu text_empty=%d total=%.0fms enc=%.0fms dec=%.0fms tokens=%d",
                   session->id.c_str(),
                   static_cast<unsigned long>(seg.seq),
                   text_empty ? 1 : 0,
                   seg_total_ms, seg_encode_ms, seg_decode_ms, seg_tokens);

            /* Phase 2.1 §3.1: dump ALL segments, including empty-text ones.
             * This eliminates the blind spot where ASR empty segments have
             * no WAV/JSON evidence, making missing-sentence diagnosis impossible. */
            if (dump_cfg.enabled) {
                RealtimeLiveWorker * wrk = session->live_worker.get();
                const uint64_t dumped = wrk->dumped_segment_count.fetch_add(1, std::memory_order_relaxed);
                const uint64_t dumped_samples = wrk->dumped_sample_count.fetch_add(
                    static_cast<uint64_t>(n_samples), std::memory_order_relaxed);
                const double total_sec = static_cast<double>(dumped_samples) / 16000.0;
                if (dumped < dump_cfg.max_segments && total_sec < static_cast<double>(dump_cfg.max_seconds)) {
                    const fs::path dump_dir(dump_cfg.dir);
                    fs::create_directories(dump_dir);
                    char name_buf[256];
                    std::snprintf(name_buf, sizeof(name_buf),
                                  "sid_%s_seq_%06lu_%s.wav",
                                  session->id.c_str(),
                                  static_cast<unsigned long>(seg.seq),
                                  seg.commit_reason.c_str());
                    const std::string wav_name(name_buf);
                    std::snprintf(name_buf, sizeof(name_buf),
                                  "sid_%s_seq_%06lu_%s.json",
                                  session->id.c_str(),
                                  static_cast<unsigned long>(seg.seq),
                                  seg.commit_reason.c_str());
                    const std::string json_name(name_buf);
                    WriteFloatMono16kWav(dump_dir / wav_name,
                                         seg.samples.data(), n_samples);
                    Json meta = Json::object();
                    meta["sid"] = session->id;
                    meta["seq"] = static_cast<std::int64_t>(seg.seq);
                    meta["reason"] = seg.commit_reason;
                    meta["endpoint_mode"] = seg.endpoint_mode;
                    meta["first_segment"] = seg.first_segment;
                    meta["audio_sec"] = seg.queued_audio_sec;
                    meta["sample_count"] = static_cast<std::int64_t>(n_samples);
                    meta["language"] = session->language;
                    meta["forced_language"] = forced_language;
                    meta["raw_null"] = raw_null;
                    meta["text_empty"] = text_empty;
                    meta["asr_empty"] = text_empty;
                    meta["asr_text"] = asr_text;
                    meta["total_samples_at_push"] = seg.total_samples_at_push;
                    meta["left_context_ms"] = 0;
                    meta["right_context_ms"] = 0;
                    meta["boundary_overlap_ms"] = 0;
                    meta["truncated_left"] = false;
                    meta["truncated_right"] = false;
                    meta["asr_total_ms"] = seg_total_ms;
                    meta["asr_encode_ms"] = seg_encode_ms;
                    meta["asr_decode_ms"] = seg_decode_ms;
                    meta["asr_tokens"] = seg_tokens;
                    std::ofstream jf(dump_dir / json_name);
                    if (jf) {
                        jf << meta.dump() << std::endl;
                    }
                }
            }

 /* §7.2-7.3: Sentence-boundary split with tail carry-over.
           *
           * The ASR model may return accumulated full text.  Strip the
           * prefix that was already in the previous segment.  Then find
           * the last sentence boundary (terminal punctuation).  Text
           * before the boundary is confirmed; text after is uncertain.
           * The audio after the boundary is estimated by proportion and
           * carried over as tail_audio for the next segment.
           *
           * This prevents mid-sentence cuts: "大师先生，几日未见，您可安好？"
           * is confirmed immediately.  The tail "哦，yes, I'm fine. Aside from"
           * is carried to the next segment for re-decode with more context. */
          if (!asr_text.empty()) {
               std::lock_guard<std::mutex> lock(session->mu);
               if (!asr_text.empty()) {
                   RT_LOG("AsrWorkerLoop sid=%s seq=%lu asr_text=%s "
                          "segments_text_count=%zu",
                          session->id.c_str(),
                          static_cast<unsigned long>(seg.seq),
                          asr_text.c_str(),
                          session->segments_text.size());
                    /* Build combined prefix: confirmed segments + last candidate
                       * text (if any).  When tail audio is prepended, the model
                       * re-outputs everything it heard, including the portion
                       * that was a candidate in the previous segment.  We need
                       * to strip that portion to get only the new text. */
                     std::string segment_text = asr_text;
                     /* When tail audio is prepended, the model re-outputs
                      * everything it heard.  The re-outputed text may differ
                      * slightly from candidates due to model reinterpretation
                      * (e.g. "了，最近。" → "最近，").  Therefore, we search
                      * for the last candidate text WITHIN the asr_text and
                      * strip everything up to and including its end.  This
                      * gives us only the new portion that corresponds to the
                      * fresh audio beyond the tail. */
                     if (!session->candidates.empty()) {
                         const std::string & last_cand =
                             session->candidates.back();
                         std::size_t pos = asr_text.find(last_cand);
                         if (pos != std::string::npos) {
                             /* Found last candidate within asr_text — strip
                              * everything up to and including the match. */
                             segment_text = asr_text.substr(
                                 pos + last_cand.size());
                             while (!segment_text.empty()) {
                                 std::size_t trim_len =
                                     TrimLeadingCharLen(segment_text);
                                 if (trim_len == 0) break;
                                 segment_text.erase(0, trim_len);
                             }
                             RT_LOG("AsrWorkerLoop sid=%s seq=%lu candidate-strip "
                                    "found at pos=%zu len=%zu → remaining=%s",
                                    session->id.c_str(),
                                    static_cast<unsigned long>(seg.seq),
                                    pos, last_cand.size(),
                                    segment_text.c_str());
                         }
                     }

                    if (segment_text.empty()) {
                       RT_LOG("AsrWorkerLoop sid=%s seq=%lu empty segment after trim, skipping",
                              session->id.c_str(),
                              static_cast<unsigned long>(seg.seq));
                    } else {
                        /* §7.2 #1: Find last mid-text sentence boundary
                         * (terminal punctuation NOT at the text end).
                         * Returns 0 if only boundary-at-end or no boundary. */
                        std::size_t boundary = FindLastMidTextBoundary(segment_text);
                        int n_samples = static_cast<int>(seg.samples.size());

                    if (boundary > 0) {
                        /* Extract confirmed text before boundary. */
                        std::string confirmed_text = segment_text.substr(0, boundary);
                        /* Strip trailing whitespace. */
                        while (!confirmed_text.empty() &&
                               (confirmed_text.back() == ' ' ||
                                confirmed_text.back() == '\n' ||
                                confirmed_text.back() == '\t')) {
                            confirmed_text.pop_back();
                        }
                        /* Strip leading terminal punctuation from tail residue.
                         * When tail audio is prepended, the model may output the
                         * previous segment's ending punctuation at the start. */
                        /* Guard: if confirmed text is too short (< 3 CJK chars
                         * or < 9 raw bytes), it's likely a false boundary from
                         * tail residue.  Don't commit — fall through to
                         * candidates path for merge-redecode. */
                        if (confirmed_text.size() < 9) {
                            RT_LOG("AsrWorkerLoop sid=%s seq=%lu skip-commit=%s "
                                   "(too short, %zu bytes, likely tail residue)",
                                   session->id.c_str(),
                                   static_cast<unsigned long>(seg.seq),
                                   confirmed_text.c_str(), confirmed_text.size());
                            session->candidates.push_back(segment_text);
                            session->tail_audio.assign(
                                seg.samples.begin(), seg.samples.end());
                        } else {
                          /* Cut at text ratio — full tail for stop drain.
                              * candidate-strip handles dedup on next decode. */
                             double ratio = static_cast<double>(boundary) /
                                            static_cast<double>(segment_text.size());
                             ratio = std::max(0.15, std::min(0.90, ratio));
                             int cut_samples = static_cast<int>(n_samples * ratio);

                            session->segment_cumulative_samples += cut_samples;
                            session->segments_sample_positions.push_back(
                                session->segment_cumulative_samples);
                            session->segments_text.push_back(confirmed_text);
                            /* Clear superseded candidates — their audio has been
                             * covered by this confirmed segment, so they would
                             * create duplicates if pushed to the UI. */
                            session->candidates.clear();
                            RT_LOG("AsrWorkerLoop sid=%s seq=%lu confirmed=%s (boundary=%zu cut=%d/%d tail=%.2fs)",
                                   session->id.c_str(),
                                   static_cast<unsigned long>(seg.seq),
                                   confirmed_text.c_str(),
                                   boundary, cut_samples, n_samples,
                                     (n_samples - cut_samples) / 16000.0);

                            if (cut_samples < n_samples) {
                                session->tail_audio.assign(
                                    seg.samples.begin() + cut_samples,
                                    seg.samples.end());
                                RT_LOG("AsrWorkerLoop sid=%s tail_carry=%d samples (%.2fs)",
                                       session->id.c_str(),
                                       n_samples - cut_samples,
                                       (n_samples - cut_samples) / 16000.0);
                            }
                            {
                                session->tail_text = segment_text.substr(boundary);
                                while (!session->tail_text.empty() &&
                                       (session->tail_text.front() == ' ' ||
                                        session->tail_text.front() == '\n' ||
                                        session->tail_text.front() == '\t')) {
                                    session->tail_text.erase(session->tail_text.begin());
                                }
                            }
                        }
                    } else {
                        /* No mid-text boundary.  Two sub-cases:
                         *
                         * ① Boundary at end (text ends with . ? !  。 ？ ！)
                         *    → candidates, cut at end (100%), NO tail.
                         *      Text appears complete; the audio after the last
                         *      punctuation is VAD silence so no acoustic context
                         *      is lost by dropping the tail.
                         *
                         * ② No boundary at all
                         *    → candidates, cut at 0%, carry FULL audio as tail.
                         *      Text is uncertain; next segment needs full
                         *      acoustic context for re-decode. */
                       /* No mid-text boundary — push to candidates and carry
                          * FULL audio as tail for next-segment merge-redecode.
                          * This applies whether text ends with punctuation or not:
                          * the next VAD segment needs the full acoustic context so
                          * the model can re-organize across segment boundaries.
                          * NOTE: do NOT increment segment_cumulative_samples here
                          * since this audio is unconfirmed — it will be counted
                          * when the merged segment eventually produces a confirmed
                          * boundary. */
                        session->candidates.push_back(segment_text);
                       session->tail_audio.assign(
                           seg.samples.begin(), seg.samples.end());
                       const char *tag = EndsWithBoundary(segment_text)
                                            ? "candidate+full-tail"
                                            : "candidate+full-tail";
                       RT_LOG("AsrWorkerLoop sid=%s seq=%lu %s=%s tail=%.2fs",
                              session->id.c_str(),
                              static_cast<unsigned long>(seg.seq),
                              tag, segment_text.c_str(),
                              seg.samples.size() / 16000.0);
                    }
                    } /* end: segment_text not empty */

                   /* Update session text state for display. */
                  RealtimeTextUpdate update;
                  update.committed = true;
                  update.stable_text = asr_text;
                  update.partial_text.clear();
                  update.text = asr_text;
                  session->stable_text = asr_text;
                  session->text_state.stable_text = asr_text;
                  session->text_state.last_text = asr_text;
                  session->text_state.last_decode_samples = session->total_samples;
                  session->text_state.unstable_since_samples = session->total_samples;
                  session->last_inference_ms = seg_total_ms;
                  ApplyRealtimeUpdate(update, session->last_inference_ms,
                                      true, false, session.get());
              }
              /* Wake the SSE stream. */
              session->sse_cv.notify_all();
          } else if (qwen_verbose >= 1) {
             std::fprintf(stderr,
                          "ASR-worker: qwen_transcribe_audio returned null\n");
          }
      }

    /* Drain remaining tail audio after stop: decode once and push the
         * entire result as a single segment.  No boundary splitting, no
         * loop — just finalize the last portion of audio. */
       {
           std::vector<float> drain_audio;
           {
               std::lock_guard<std::mutex> lock(session->mu);
               drain_audio = session->tail_audio;
               session->tail_audio.clear();
           }
           if (!drain_audio.empty()) {
               std::string drain_text;
               int n_drain = static_cast<int>(drain_audio.size());

               if (gpuPipeline && facade) {
                   GpuScheduler * sched = facade->scheduler();
                   if (sched) {
                       SegmentJob job;
                       job.session_id = static_cast<std::uint64_t>(
                           reinterpret_cast<std::uintptr_t>(session.get()));
                       job.segment_id = 0;
                       job.samples = drain_audio;
                       job.sample_rate = 16000;
                       job.realtime = true;
                       job.language = forced_language;
                       job.prompt = "";
                       job.engine_session_id = gpuSessionId;
                       job.enqueue_time = std::chrono::steady_clock::now();
                       SegmentResult res = sched->SubmitAndAwait(job);
                       if (res.status.ok()) drain_text = res.text;
                   } else if (gpuSessionId) {
                       AsrEngine * eng = facade->engine();
                       AsrSegmentResult segRes = eng->TranscribeSegment(
                           gpuSessionId, drain_audio, 16000);
                       if (segRes.status.ok()) drain_text = segRes.text;
                   }
               } else if (live_ctx) {
                   char * raw = qwen_transcribe_audio(
                       live_ctx, drain_audio.data(), n_drain);
                   if (raw != nullptr) {
                       drain_text.assign(raw);
                       std::free(raw);
                       while (!drain_text.empty() &&
                              (drain_text.back() == ' ' ||
                               drain_text.back() == '\n' ||
                               drain_text.back() == '\t')) {
                           drain_text.pop_back();
                       }
                   }
               }

               if (!drain_text.empty()) {
                   std::lock_guard<std::mutex> lock(session->mu);
                   /* Drain covers everything remaining — clear tail_text
                    * to prevent the finalize block from duplicating it. */
                   session->tail_text.clear();
                   std::string seg_text = drain_text;

                   /* Strip confirmed prefix. */
                   std::string cp;
                   for (const auto & s : session->segments_text) cp += s;
                   if (!cp.empty() && seg_text.size() > cp.size()) {
                       seg_text = seg_text.substr(cp.size());
                       while (!seg_text.empty()) {
                           std::size_t tl = TrimLeadingCharLen(seg_text);
                           if (tl == 0) break;
                           seg_text.erase(0, tl);
                       }
                   }

                   if (!seg_text.empty()) {
                       session->segment_cumulative_samples += n_drain;
                       session->segments_sample_positions.push_back(
                           session->segment_cumulative_samples);
                       session->segments_text.push_back(seg_text);
                       RT_LOG("AsrWorkerLoop sid=%s drain-final=%s (audio=%.2fs)",
                               session->id.c_str(), seg_text.c_str(),
                               n_drain / 16000.0);
                       session->candidates.clear(); /* drain covered everything */
                   }
               } else {
                   /* Even with no drain_text, clear stale tail_text. */
                   std::lock_guard<std::mutex> lock(session->mu);
                   session->tail_text.clear();
               }
           }
       }
       /* Wake SSE so UI receives the drain-final segment. */
       session->sse_cv.notify_all();

      /* Finalize: mark the session state so the UI knows this
            * is the last text, and free the per-session clone. */
       {
           std::lock_guard<std::mutex> lock(session->mu);
           if (session->error.empty()) {
               ApplyStableRealtimeCommit(session->total_samples,
                                         session->stable_text,
                                         session->last_inference_ms,
                                         true, session.get());
           }
           /* Push any remaining tail text. */
           if (!session->tail_text.empty()) {
               session->segments_text.push_back(session->tail_text);
               session->segments_sample_positions.push_back(
                   session->segment_cumulative_samples);
               session->tail_text.clear();
               RT_LOG("AsrWorkerLoop sid=%s finalize_tail_text=%s",
                      session->id.c_str(),
                      session->segments_text.back().c_str());
           }
          session->tail_audio.clear();
            /* Force-finalize remaining candidates — they represent the last
             * uncommitted text that the drain didn't cover (e.g. when drain
             * text was empty or fully overlapped with confirmed prefix). */
            for (const auto & s : session->candidates) {
                session->segments_sample_positions.push_back(
                    session->segment_cumulative_samples);
                session->segments_text.push_back(s);
                RT_LOG("AsrWorkerLoop sid=%s finalize-candidate=%s",
                       session->id.c_str(), s.c_str());
            }
            session->candidates.clear();
            session->finalized = true;
           session->sse_cv.notify_all();
           session->worker_done = true;
       }
        if (live_ctx) qwen_free(live_ctx);
        if (gpuPipeline && facade && gpuSessionId) {
            facade->engine()->CloseSession(gpuSessionId);
            RT_LOG("AsrWorkerLoop sid=%s GPU session closed esid=%lu",
                   session->id.c_str(),
                   static_cast<unsigned long>(gpuSessionId));
        }
        RT_LOG("AsrWorkerLoop sid=%s END", session->id.c_str());
    };

    auto StartRealtimeLiveWorker = [&](const std::shared_ptr<RealtimeSession> & session) -> Status {
        if (session == nullptr) {
            return Status(StatusCode::kInvalidArgument, "session must not be null");
        }
        RT_LOG("StartRealtimeLiveWorker sid=%s enter", session->id.c_str());

        auto worker = std::make_unique<RealtimeLiveWorker>();
        Status status = InitializeManualLiveAudio(&worker->live);
        if (!status.ok()) {
            return status;
        }
        worker->live_ready = true;
        worker->stop_requested.store(false, std::memory_order_release);
        worker->vad_thread_joined = false;
        worker->asr_thread_joined = false;

        InferHandle liveHandle = realtime_model->createInferHandle();
        qwen_ctx_t * live_ctx = liveHandle.nativeCtx;
        liveHandle.nativeCtx = nullptr;  // ownership transferred to AsrWorkerLoop
        const bool gpuPipeline = live_ctx == nullptr;
        if (gpuPipeline) {
            RT_LOG("StartRealtimeLiveWorker sid=%s nativeCtx null — using engine pipeline",
                   session->id.c_str());
        } else {
            RT_LOG("StartRealtimeLiveWorker sid=%s live_ctx=%p", session->id.c_str(), (void*)live_ctx);
            const float temperature = realtime_model->temperature();
            if (temperature >= 0.0f) {
                live_ctx->decode_temperature = temperature;
            }
        }

        const std::string forced_language = session->language;
         /* Per-session VAD: each session owns its own Silero VAD instance
         * so LSTM hidden state is not shared across sessions.  This
         * prevents cross-session state pollution and makes first-sentence
         * boundary detection stable.  Controlled by QASR_PER_SESSION_VAD
         * (default=1).  Fallback: if create fails, use shared VAD. */
      const bool use_per_session_vad = getenv("QASR_PER_SESSION_VAD") == nullptr ||
                                          atoi(getenv("QASR_PER_SESSION_VAD")) != 0;
         if (use_per_session_vad) {
             auto steady_ms = []() -> double {
                 return std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now().time_since_epoch()).count();
             };
             const double vad_create_t0 = steady_ms();
             QwenVadPtr local_vad(qwen_silero_vad_create(nullptr));
             const double vad_create_t1 = steady_ms();

             bool local_vad_active = local_vad && qwen_silero_vad_is_active(local_vad.get());
             if (local_vad_active) {
                 const int rc = qwen_silero_vad_reset(local_vad.get());
                 if (rc != 0) {
                     RT_LOG("StartRealtimeLiveWorker sid=%s session VAD reset failed; falling back to shared VAD",
                            session->id.c_str());
                     local_vad.reset();
                     local_vad_active = false;
                 }
             }

             if (local_vad_active) {
                 RT_LOG("VAD-session sid=%s create active=1 cost=%.1fms fallback_shared=0",
                        session->id.c_str(), vad_create_t1 - vad_create_t0);
                 worker->session_vad = std::move(local_vad);
                 worker->session_vad_active = true;
                 worker->session_vad_fallback_shared = false;
             } else {
                 RT_LOG("VAD-session sid=%s create active=0 cost=%.1fms fallback_shared=1",
                        session->id.c_str(), vad_create_t1 - vad_create_t0);
                 worker->session_vad_fallback_shared = true;
             }
         } else {
             worker->session_vad_fallback_shared = true;
         }

       /* Thread 1: VAD FACADE (producer).  Reads live audio, runs
         * VAD, pushes AudioSegments into worker->segment_queue. */
        worker->vad_thread = std::thread([
            session,
            worker_ptr = worker.get(),
            &vad_mu,
            &VadFacadeLoop,
            &dump_config,
            endpoint_mode,
            &cap_policy]() {
            RT_LOG("vad-thread sid=%s BEGIN", session->id.c_str());
            VadFacadeLoop(&worker_ptr->live, session, worker_ptr, &vad_mu,
                          &worker_ptr->stop_requested, &worker_ptr->segment_queue,
                          &worker_ptr->cumulative_decoded_samples,
                          dump_config, endpoint_mode, cap_policy);
            RT_LOG("vad-thread sid=%s END", session->id.c_str());
        });

      /* Thread 2: ASR WORKER (consumer).  Pops AudioSegments from
          * the queue, runs qwen_transcribe_audio (CPU) or
          * TranscribeSegment (GPU), updates session. */
         worker->asr_thread = std::thread([
             session,
             live_ctx,
             forced_language,
             facade = realtime_model,
             worker_ptr = worker.get(),
             &AsrWorkerLoop,
             &dump_config]() {
             RT_LOG("asr-thread sid=%s BEGIN", session->id.c_str());
             /* The AsrWorkerLoop owns live_ctx and will qwen_free()
              * it on exit.  When live_ctx is null (GPU path), it uses
              * the facade's engine pipeline (TranscribeSegment). */
             AsrWorkerLoop(live_ctx, session, forced_language,
                           &worker_ptr->segment_queue, dump_config, facade);
             RT_LOG("asr-thread sid=%s END", session->id.c_str());
         });

        session->live_worker = std::move(worker);
        return OkStatus();
    };

    auto StartHostCaptureLiveWorker = [&](const std::shared_ptr<HostCaptureSession> & capture) -> Status {
        if (capture == nullptr) {
            return Status(StatusCode::kInvalidArgument, "capture must not be null");
        }

        auto worker = std::make_unique<RealtimeLiveWorker>();
        Status status = InitializeManualLiveAudio(&worker->live);
        if (!status.ok()) {
            return status;
        }
        worker->live_ready = true;

        InferHandle captureHandle = realtime_model->createInferHandle();
        qwen_ctx_t * live_ctx = captureHandle.nativeCtx;
        captureHandle.nativeCtx = nullptr;  // ownership transferred to host capture thread
        if (live_ctx == nullptr) {
            DestroyManualLiveAudio(&worker->live);
            return Status(StatusCode::kInternal, "failed to clone capture model context");
        }

        const float stream_chunk_sec = RealtimeStreamChunkSeconds(realtime_policy);
        const int stream_max_new_tokens = RealtimeStreamMaxNewTokens(realtime_policy);
        const int verbosity = realtime_model->verbosity();

        worker->thread = std::thread([
            capture,
            worker_ptr = worker.get(),
            live_ctx,
            stream_chunk_sec,
            stream_max_new_tokens,
            verbosity]() {
            qwen_verbose = verbosity;
            live_ctx->segment_sec = 30.0f;
            live_ctx->past_text_conditioning = 1;
            live_ctx->stream_chunk_sec = stream_chunk_sec;
            live_ctx->stream_max_new_tokens = stream_max_new_tokens;

            if (qwen_set_prompt(live_ctx, nullptr) != 0) {
                std::lock_guard<std::mutex> lock(capture->mu);
                capture->error = "failed to set capture prompt";
                capture->worker_done = true;
                qwen_free(live_ctx);
                return;
            }

            std::function<void(const qwen_stream_chunk_t *)> chunk_callback = [&capture](const qwen_stream_chunk_t * chunk) {
                if (chunk == nullptr) {
                    return;
                }
                std::lock_guard<std::mutex> lock(capture->mu);
                ApplyChunkRealtimeCommit(chunk, capture->total_samples, capture.get());
            };

            qwen_set_chunk_callback(live_ctx, ForwardStreamChunk, &chunk_callback);
            char * raw = qwen_transcribe_stream_live(live_ctx, &worker_ptr->live);
            const bool was_cancelled = qwen_was_cancelled(live_ctx) != 0;
            qwen_set_chunk_callback(live_ctx, nullptr, nullptr);

            {
                std::lock_guard<std::mutex> lock(capture->mu);
                capture->last_inference_ms = live_ctx->perf_total_ms;
                if (raw != nullptr) {
                    ApplyStableRealtimeCommit(capture->total_samples, raw, live_ctx->perf_total_ms, true, capture.get());
                } else if (capture->error.empty()) {
                    capture->error = was_cancelled
                        ? "live capture transcription cancelled"
                        : "live capture transcription failed";
                }
                capture->finalized = true;
                capture->worker_done = true;
            }

            std::free(raw);
            qwen_free(live_ctx);
        });

        capture->live_worker = std::move(worker);
        return OkStatus();
    };

    auto FindRealtimeSession = [&](const std::string & session_id,
                                   std::shared_ptr<RealtimeSession> * session) -> Status {
        if (session == nullptr) {
            return Status(StatusCode::kInvalidArgument, "session output must not be null");
        }
        std::lock_guard<std::mutex> lock(realtime_mu);
        auto it = realtime_sessions.find(session_id);
        if (it == realtime_sessions.end()) {
            return Status(StatusCode::kNotFound, "session not found");
        }
        *session = it->second;
        return OkStatus();
    };

    auto CreateRealtimeSession = [&](std::string model_id,
                                     std::string language,
                                     RealtimeSessionSnapshot * created) -> Status {
        if (created == nullptr) {
            return Status(StatusCode::kInvalidArgument, "created session output must not be null");
        }
        RT_LOG("CreateRealtimeSession enter model=%s lang=%s", model_id.c_str(), language.c_str());

        auto session = std::make_shared<RealtimeSession>();
        session->id = std::to_string(session_counter.fetch_add(1));
        session->model = std::move(model_id);
        session->language = std::move(language);
        RT_LOG("CreateRealtimeSession sid=%s allocated", session->id.c_str());

        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            if (realtime_sessions.size() >= kMaxRealtimeSessions) {
                return Status(StatusCode::kFailedPrecondition, "too many realtime sessions");
            }
            realtime_sessions.emplace(session->id, session);
        }
        RT_LOG("CreateRealtimeSession sid=%s about to StartRealtimeLiveWorker", session->id.c_str());

        Status status = StartRealtimeLiveWorker(session);
        RT_LOG("CreateRealtimeSession sid=%s StartRealtimeLiveWorker status.ok=%d", session->id.c_str(), status.ok() ? 1 : 0);
        if (!status.ok()) {
            std::lock_guard<std::mutex> lock(realtime_mu);
            realtime_sessions.erase(session->id);
            return status;
        }

        metrics.realtime_sessions_started.fetch_add(1);
        return SnapshotRealtimeSessionState(session, false, created);
    };

    auto GetRealtimeSessionSnapshot = [&](const std::string & session_id,
                                          RealtimeSessionSnapshot * snapshot) -> Status {
        std::shared_ptr<RealtimeSession> session;
        Status status = FindRealtimeSession(session_id, &session);
        if (!status.ok()) {
            return status;
        }
        return SnapshotRealtimeSessionState(session, true, snapshot);
    };

    auto AppendRealtimeChunk = [&](const std::string & session_id,
                                   const std::vector<float> & chunk,
                                   RealtimeSessionSnapshot * snapshot) -> Status {
        if (snapshot == nullptr) {
            return Status(StatusCode::kInvalidArgument, "snapshot output must not be null");
        }
        /* Audio ingress diagnostic.  Throttled to once every 5 seconds
         * per session so we don't spam the log on a 4 Hz send loop. */
        {
            static std::atomic<uint64_t> s_chunk_seq{0};
            const uint64_t seq = s_chunk_seq.fetch_add(1);
            if ((seq % 20U) == 0U && !chunk.empty()) {
                float peak = 0.0f, sum_sq = 0.0f;
                for (float a : chunk) {
                    float aa = a < 0 ? -a : a;
                    if (aa > peak) peak = aa;
                    sum_sq += a * a;
                }
                const float rms = std::sqrt(sum_sq / static_cast<float>(chunk.size()));
                std::fprintf(stderr,
                             "[ingress] seq=%lu n=%zu peak=%.4f rms=%.4f session=%s\n",
                             static_cast<unsigned long>(seq), chunk.size(), peak, rms,
                             session_id.c_str());
            }
        }

        std::shared_ptr<RealtimeSession> session;
        Status status = FindRealtimeSession(session_id, &session);
        if (!status.ok()) {
            return status;
        }

        RealtimeLiveWorker * worker = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (session->finalized) {
                return Status(StatusCode::kFailedPrecondition, "session already finalized");
            }
            worker = session->live_worker.get();
        }
        if (worker == nullptr || !worker->live_ready) {
            return Status(StatusCode::kInternal, "realtime worker is not ready");
        }

        status = AppendManualLiveAudio(&worker->live, chunk.data(), chunk.size());
        if (!status.ok()) {
            return status;
        }

        /* Stash ingress peak/RMS on the session for the audio_diag
         * endpoint to surface to the UI.  Computed in O(n) per chunk
         * (a few microseconds for 1-3 KB chunks). */
        {
            float peak = 0.0f, sum_sq = 0.0f;
            for (float a : chunk) {
                float aa = a < 0 ? -a : a;
                if (aa > peak) peak = aa;
                sum_sq += a * a;
            }
            const float rms = std::sqrt(sum_sq / static_cast<float>(chunk.size()));
            std::lock_guard<std::mutex> lock(session->mu);
            session->last_ingress_peak = peak;
            session->last_ingress_rms = rms;
            if (peak > session->max_ingress_peak) {
                session->max_ingress_peak = peak;
            }
            session->ingress_chunks += 1;
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            AppendRealtimeSamples(realtime_policy, chunk, session.get());
        }
        metrics.realtime_decode_runs.fetch_add(1);
        status = SnapshotRealtimeSessionState(session, true, snapshot);
        if (!status.ok()) {
            return status;
        }
        if (!snapshot->error.empty()) {
            return Status(StatusCode::kInternal, snapshot->error);
        }
        return OkStatus();
    };

    auto FinalizeRealtimeSession = [&](const std::string & session_id,
                                       RealtimeSessionSnapshot * snapshot) -> Status {
        if (snapshot == nullptr) {
            return Status(StatusCode::kInvalidArgument, "session snapshot output must not be null");
        }
        RT_LOG("FinalizeRealtimeSession enter sid=%s", session_id.c_str());

        std::shared_ptr<RealtimeSession> session;
        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            auto it = realtime_sessions.find(session_id);
            if (it == realtime_sessions.end()) {
                return Status(StatusCode::kNotFound, "session not found");
            }
            session = it->second;
            realtime_sessions.erase(it);
        }

        RealtimeLiveWorker * worker = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            worker = session->live_worker.get();
        }
        RT_LOG("FinalizeRealtimeSession sid=%s about to join worker=%p", session_id.c_str(), (void*)worker);
        JoinRealtimeLiveWorker(worker);
        /* Phase 2.1 §3.6: log queue stats after join. */
        worker->segment_queue.LogStats(session_id.c_str());
        RT_LOG("FinalizeRealtimeSession sid=%s join returned", session_id.c_str());

        {
               std::lock_guard<std::mutex> lock(session->mu);
               session->live_worker.reset();
               session->finalized = true;
               session->sse_cv.notify_all();
           }
           metrics.realtime_finalizations.fetch_add(1);

          /* Snapshot VAD-segmented results immediately so the stop
            * handler can return without blocking on retranscription.
            * Full-audio retranscription runs asynchronously below. */
         {
             std::lock_guard<std::mutex> lock(session->mu);
             snapshot->id       = session_id;
             snapshot->model    = session->model;
             snapshot->language = session->language;
             snapshot->total_samples   = session->total_samples;
             snapshot->decoded_samples = session->decoded_samples;
             snapshot->retained_sample_count  = session->full_audio.size()
                                                 - session->retained_sample_offset;
             snapshot->retained_sample_offset = session->retained_sample_offset;
             snapshot->segments_text  = session->segments_text;
              snapshot->candidates     = session->candidates;
             snapshot->text           = session->text;
             snapshot->stable_text    = session->stable_text;
             snapshot->partial_text   = session->partial_text;
             snapshot->last_inference_ms  = session->last_inference_ms;
             snapshot->last_decode_ran    = session->last_decode_ran;
             snapshot->finalized        = true;
             snapshot->error            = session->error;
             snapshot->last_ingress_peak   = session->last_ingress_peak;
             snapshot->last_ingress_rms    = session->last_ingress_rms;
             snapshot->max_ingress_peak    = session->max_ingress_peak;
             snapshot->ingress_chunks      = session->ingress_chunks;
         }
         if (!snapshot->error.empty()) {
             return Status(StatusCode::kInternal, snapshot->error);
         }

         /* Post-stop full-audio retranscription runs on a detached
           * background thread.  It uses the larger batch model to
           * retranscribe the complete audio for higher quality.
           * Result updates session->text when ready — the session
           * shared_ptr keeps the object alive. */
         {
             std::shared_ptr<RealtimeSession> ssn = session;
             std::string sid = session_id;
             std::string lang = session->language;
             std::vector<float> audioCopy = session->full_audio;
             std::string vadText;
             {
                 std::lock_guard<std::mutex> lock(session->mu);
                 for (const auto & s : session->segments_text) {
                     if (!vadText.empty()) vadText += " ";
                     vadText += s;
                 }
             }
      std::thread([ssn, sid, lang, bm = batch_model, audioCopy = std::move(audioCopy),
                             vadText = std::move(vadText)]() mutable {
                    InferHandle rHandle = bm->createInferHandle();
                    qwen_ctx_t * reconcile_ctx = rHandle.nativeCtx;
                  if (reconcile_ctx && audioCopy.size() > 1024) {
                      reconcile_ctx->segment_sec = 0.0f;
                      reconcile_ctx->search_sec = 0.0f;
                      reconcile_ctx->past_text_conditioning = 0;
                      reconcile_ctx->stream_chunk_sec = 0.0f;
                      reconcile_ctx->stream_max_new_tokens = 0;
                      if (!lang.empty()) {
                          qwen_set_force_language(reconcile_ctx, lang.c_str());
                      }
                      qwen_set_prompt(reconcile_ctx, nullptr);

                      char * raw = qwen_transcribe_audio(reconcile_ctx,
                          audioCopy.data(),
                          static_cast<int>(audioCopy.size()));
                      if (raw) {
                          std::string full_text(raw);
                          std::free(raw);
                          while (!full_text.empty() &&
                                 (full_text.back() == ' ' || full_text.back() == '\n' || full_text.back() == '\t')) {
                              full_text.pop_back();
                          }
                          const bool should_replace = !full_text.empty() &&
                              full_text.size() >= vadText.size() * 0.9;
                           RT_LOG("RECONCILE sid=%s vad_len=%zd full_len=%zd gate=%d "
                                 "forced_lang=%s full_audio_sec=%.2f",
                                 sid.c_str(),
                                 vadText.size(),
                                 full_text.size(),
                                 should_replace ? 1 : 0,
                                 lang.empty() ? "<auto>" : lang.c_str(),
                                 static_cast<double>(audioCopy.size()) / 16000.0);
                               if (should_replace) {
                               if (qwen_verbose >= 1) {
                                   std::fprintf(stderr,
                                       "RECONCILE sid=%s: VAD=%zd chars retranscribed=%zd chars"
                                       " — delivering as reconcile event\n",
                                       sid.c_str(), vadText.size(), full_text.size());
                               }
                               {
                                   std::lock_guard<std::mutex> lock(ssn->mu);
                                   ssn->reconcile_text = full_text;
                                   ssn->reconcile_revised = (full_text != vadText);
                                   ssn->reconcile_ready = true;
                                  ssn->sse_cv.notify_all();
                              }
                          } else if (!full_text.empty() && qwen_verbose >= 1) {
                              std::fprintf(stderr,
                                  "RECONCILE sid=%s: retranscribed (%zd) shorter than "
                                  "VAD (%zd), keeping VAD text\n",
                                  sid.c_str(), full_text.size(), vadText.size());
                          }
                      }
                       bm->releaseInferHandle(rHandle);
                   } else if (reconcile_ctx) {
                       bm->releaseInferHandle(rHandle);
                   }
              }).detach();
         }

         return OkStatus();
     };

    server.Get("/", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "index.html", "text/html; charset=utf-8", "index.html");
    });
    server.Get("/app.js", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "app.js", "application/javascript; charset=utf-8", "app.js");
    });
    server.Get("/live_monitor.js", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "live_monitor.js", "application/javascript; charset=utf-8", "live_monitor.js");
    });
    server.Get("/state_pure.js", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "state_pure.js", "application/javascript; charset=utf-8", "state_pure.js");
    });
    server.Get("/state.js", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "state.js", "application/javascript; charset=utf-8", "state.js");
    });
    server.Get("/terminal.js", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "terminal.js", "application/javascript; charset=utf-8", "terminal.js");
    });
    server.Get("/style.css", [&](const HttpRequest &, HttpResponse & response) {
        ServeStaticTextFile(response, ui_dir / "style.css", "text/css; charset=utf-8", "style.css");
    });

    server.Get("/health", [&](const HttpRequest &, HttpResponse & response) {
        response.set_content(BuildHealthJson(), "application/json; charset=utf-8");
    });
    server.Get("/api/health", [&](const HttpRequest &, HttpResponse & response) {
        response.set_content(BuildHealthJson(), "application/json; charset=utf-8");
    });
    /* Debug endpoint: page JS POSTs DOM snapshots here, server keeps the
     * latest in `debug_state` (a mutex-guarded std::string).  Curl
     * /api/debug/get to read.  Used for live UI inspection during
     * development only. */
    static std::mutex debug_mu;
    static std::string debug_state;
    server.Post("/api/debug/state", [&](const HttpRequest & request, HttpResponse & response) {
        std::lock_guard<std::mutex> lock(debug_mu);
        debug_state = request.body;
        SetJsonResponse(response, Json::object({{"ok", true}, {"len", request.body.size()}}));
    });
    server.Get("/api/debug/get", [&](const HttpRequest &, HttpResponse & response) {
        std::lock_guard<std::mutex> lock(debug_mu);
        response.set_content(debug_state, "application/json; charset=utf-8");
    });
    server.Get("/v1/models", [&](const HttpRequest &, HttpResponse & response) {
        Json payload;
        payload["object"] = "list";
        payload["data"] = Json::array({
            Json::object({
                {"id", served_model_id},
                {"object", "model"},
                {"created", 0},
                {"owned_by", "qwen3asr_cpu"},
            })
        });
        SetJsonResponse(response, payload);
    });
    server.Get("/api/metrics", [&](const HttpRequest &, HttpResponse & response) {
        std::size_t active_realtime_sessions = 0;
        std::size_t queued_jobs = 0;
        bool host_capture_active = false;
        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            active_realtime_sessions = realtime_sessions.size();
        }
        {
            std::lock_guard<std::mutex> lock(jobs_mu);
            queued_jobs = jobs.size();
        }
        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            host_capture_active = static_cast<bool>(host_capture && host_capture->active);
        }
        const auto uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - server_start).count();
        Json payload;
        payload["backend"] = facade_->backendKind() == BackendKind::kCpu ? "cpu" : "cuda";
        payload["backend_fallback"] = facade_->backendFallback();
        payload["uptime_ms"] = uptime_ms;
        payload["offline_requests"] = metrics.offline_requests.load();
        payload["async_jobs_submitted"] = metrics.async_jobs_submitted.load();
        payload["async_job_cleanup_runs"] = metrics.async_job_cleanup_runs.load();
        payload["async_jobs_evicted"] = metrics.async_jobs_evicted.load();
        payload["chat_requests"] = metrics.chat_requests.load();
        payload["realtime_sessions_started"] = metrics.realtime_sessions_started.load();
        payload["realtime_decode_runs"] = metrics.realtime_decode_runs.load();
        payload["realtime_finalizations"] = metrics.realtime_finalizations.load();
        payload["host_capture_sessions_started"] = metrics.host_capture_sessions_started.load();
        payload["active_realtime_sessions"] = active_realtime_sessions;
        payload["job_count"] = queued_jobs;
        payload["host_capture_active"] = host_capture_active;
        SetJsonResponse(response, payload);
    });

    server.Post("/api/transcriptions", [&](const HttpRequest & request, HttpResponse & response) {
        const MultipartFormData * file = FindUploadedAudio(request);
        if (file == nullptr) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "multipart field 'audio' or 'file' is required"), 400);
            return;
        }

        metrics.offline_requests.fetch_add(1);
        TranscriptionApiOptions options;
        const Status parse_status = ParseTranscriptionApiOptions(request, &options);
        if (!parse_status.ok()) {
            SetErrorResponse(response, parse_status, StatusToHttpCode(parse_status));
            return;
        }
        if (options.stream) {
            SetErrorResponse(
                response,
                Status(StatusCode::kFailedPrecondition, "use /api/transcriptions/async + GET /api/jobs/:id for progressive results"),
                412);
            return;
        }

        PreparedAudioInput prepared;
        const Status prepare_status = PrepareUploadedAudio(*file, &prepared);
        if (!prepare_status.ok()) {
            SetErrorResponse(response, prepare_status, StatusToHttpCode(prepare_status));
            return;
        }

        ModelDecodeOptions decode;
        decode.prompt = options.prompt;
        decode.language = options.language;
        const AsrRunResult result = batch_model->TranscribeFile(prepared.wav_path, decode);
        CleanupPreparedAudio(&prepared);
        if (!result.status.ok()) {
            SetErrorResponse(response, result.status, StatusToHttpCode(result.status));
            return;
        }

        Json body = BuildBasicTranscriptionJson(result, options);
        SetJsonResponse(response, body);
    });

    server.Post("/api/transcriptions/async", [&](const HttpRequest & request, HttpResponse & response) {
        const MultipartFormData * file = FindUploadedAudio(request);
        if (file == nullptr) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "multipart field 'audio' or 'file' is required"), 400);
            return;
        }

        metrics.async_jobs_submitted.fetch_add(1);
        TranscriptionApiOptions options;
        const Status parse_status = ParseTranscriptionApiOptions(request, &options);
        if (!parse_status.ok()) {
            SetErrorResponse(response, parse_status, StatusToHttpCode(parse_status));
            return;
        }
        if (options.stream) {
            SetErrorResponse(response, Status(StatusCode::kFailedPrecondition, "async path does not support stream=true"), 412);
            return;
        }

        PreparedAudioInput prepared;
        const Status prepare_status = PrepareUploadedAudio(*file, &prepared);
        if (!prepare_status.ok()) {
            SetErrorResponse(response, prepare_status, StatusToHttpCode(prepare_status));
            return;
        }

        OfflineJob job;
        job.id = std::to_string(session_counter.fetch_add(1));
        job.cancel_flag = std::make_shared<std::atomic<bool>>(false);
        job.created_at = CurrentUnixSeconds();
        job.updated_at = job.created_at;
        {
            std::lock_guard<std::mutex> lock(jobs_mu);
            jobs.emplace(job.id, job);
        }

        const std::string job_id = job.id;
        const std::shared_ptr<std::atomic<bool>> cancel_flag = job.cancel_flag;
        std::thread([&, prepared, options, job_id, cancel_flag]() mutable {
            bool cancel_before_start = false;
            {
                std::lock_guard<std::mutex> lock(jobs_mu);
                OfflineJob & current = jobs[job_id];
                current.updated_at = CurrentUnixSeconds();
                if (current.cancel_requested || (cancel_flag && cancel_flag->load())) {
                    current.state = "cancelled";
                    cancel_before_start = true;
                } else {
                    current.state = "running";
                }
            }

            if (cancel_before_start) {
                CleanupPreparedAudio(&prepared);
                return;
            }

            ModelDecodeOptions decode;
            decode.prompt = options.prompt;
            decode.language = options.language;
            decode.cancel_callback = [cancel_flag]() {
                return cancel_flag && cancel_flag->load();
            };
            decode.token_callback = [&jobs, &jobs_mu, &job_id](std::string_view piece) {
                std::lock_guard<std::mutex> lock(jobs_mu);
                auto it = jobs.find(job_id);
                if (it != jobs.end()) {
                    it->second.text += std::string(piece);
                    // With boundary cleanup, callback fires per-segment with full text.
                    // Estimate token count: ~1.5 bytes/token for CJK, ~4 for Latin.
                    it->second.token_count += (std::max)(static_cast<std::int32_t>(piece.size() / 3), std::int32_t{1});
                }
            };

            /* Long audio (>40s) goes through VAD-segmented decode so
             * the encoder's 5×8s context window is never exceeded.
             * Short audio could in theory use the one-shot
             * TranscribeFile path, but to keep behavior identical
             * across audio lengths (and to stream progress for the
             * UI), we use VAD-segmented for everything.  The realtime
             * worker already proved this pattern works for live
             * audio; here we apply the same loop to a file.
             *
             * The on_segment callback fires once per VAD-committed
             * segment, updating the job's text and audio_ms under
             * jobs_mu so the UI sees growing text in real time. */
            InferHandle batchHandle = batch_model->createInferHandle();
                qwen_ctx_t *batch_ctx = batchHandle.nativeCtx;
                if (batch_ctx == nullptr) {
                /* GPU path: VAD-segmented per-segment TranscribeSegment
                 * with streaming progress (not whole-file TranscribeFile). */
                RT_LOG("batch sid=%s nativeCtx null — using engine VAD-segmented",
                       job_id.c_str());

                auto on_segment_gpu = [&jobs, &jobs_mu, &job_id,
                                       cancel_flag,
                                       prepared_wav = prepared.wav_path](
                                          int seg_idx,
                                          std::string_view seg_text,
                                          int seg_samples,
                                          int64_t total_samples) -> bool {
                    (void)seg_idx;
                    (void)seg_samples;
                    (void)prepared_wav;
                    if (cancel_flag && cancel_flag->load()) {
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(jobs_mu);
                    auto it = jobs.find(job_id);
                    if (it == jobs.end()) {
                        return false;
                    }
                    OfflineJob & current = it->second;
                    if (!seg_text.empty()) {
                        if (!current.text.empty() &&
                            current.text.back() != ' ' &&
                            current.text.back() != '\n' &&
                            current.text.back() != '\t') {
                            current.text.push_back(' ');
                        }
                        current.text.append(seg_text.data(), seg_text.size());
                        current.token_count += (std::max)(
                            static_cast<std::int32_t>(seg_text.size() / 3),
                            std::int32_t{1});
                    }
                    current.audio_ms = static_cast<double>(total_samples) *
                                       1000.0 / 16000.0;
                    current.updated_at = CurrentUnixSeconds();
                    return true;
                };

                VadSegmentedBatchResult vad_result;
                const std::string forced_lang = options.language;
                try {
                    vad_result = TranscribeFileVadSegmentedEngineImpl(
                        batch_model->engine(),
                        batch_model->vad(),
                        prepared.wav_path.string().c_str(),
                        forced_lang,
                        batch_model->verbosity(),
                        on_segment_gpu,
                        decode.cancel_callback,
                        &vad_mu);
                } catch (const std::exception & e) {
                    vad_result.status = Status(
                        StatusCode::kInternal,
                        std::string("vad-segmented exception: ") + e.what());
                } catch (...) {
                    vad_result.status = Status(StatusCode::kInternal,
                                               "vad-segmented unknown exception");
                }
                batch_model->releaseInferHandle(batchHandle);
                CleanupPreparedAudio(&prepared);

                {
                    std::lock_guard<std::mutex> lock(jobs_mu);
                    OfflineJob & current = jobs[job_id];
                    current.updated_at = CurrentUnixSeconds();
                    current.language = DetectLanguageLabel(options.language);
                    current.inference_ms = vad_result.inference_ms;
                    current.audio_ms = vad_result.audio_ms;
                    if (cancel_flag && cancel_flag->load()) {
                        current.state = "cancelled";
                        current.error.clear();
                    } else if (!vad_result.status.ok()) {
                        current.state = "failed";
                        current.error = vad_result.status.message();
                    } else {
                        current.state = "completed";
                        if (current.text.empty() && !vad_result.text.empty()) {
                            current.text = vad_result.text;
                        }
                    }
                    if (current.tokens == 0 && current.token_count > 0) {
                        current.tokens = current.token_count;
                    }
                }
                return;
            }
            const int batch_verbosity = batch_model->verbosity();
            const int batch_max_new_tokens = 64;  /* per-segment cap */
            const char *batch_lang =
                options.language.empty() ? nullptr : options.language.c_str();

            /* VAD-segmented per-segment callback.  Called from the
             * worker thread for each committed segment, before the
             * next segment is decoded.  Updates the job's text and
             * audio_ms in-place so the UI sees growing text in real
             * time.  Returns true to keep going, false to cancel
             * (e.g. user clicked Stop). */
            auto on_segment = [&jobs, &jobs_mu, &job_id, cancel_flag, prepared_wav = prepared.wav_path](
                                  int seg_idx,
                                  std::string_view seg_text,
                                  int seg_samples,
                                  int64_t total_samples) -> bool {
                (void)seg_idx;
                (void)seg_samples;
                (void)prepared_wav;
                if (cancel_flag && cancel_flag->load()) {
                    return false;  /* signal cancel to caller */
                }
                std::lock_guard<std::mutex> lock(jobs_mu);
                auto it = jobs.find(job_id);
                if (it == jobs.end()) {
                    return false;  /* job was deleted; treat as cancel */
                }
                OfflineJob &current = it->second;
                /* Append segment text (with a single space separator if
                 * the previous text doesn't end in punctuation or
                 * whitespace).  The model rarely emits a trailing
                 * space, so we always add one to keep words separated. */
                if (!seg_text.empty()) {
                    if (!current.text.empty() &&
                        current.text.back() != ' ' &&
                        current.text.back() != '\n' &&
                        current.text.back() != '\t') {
                        current.text.push_back(' ');
                    }
                    current.text.append(seg_text.data(), seg_text.size());
                    /* Rough token estimate: 1.5 chars/token CJK, 4 chars/token
                     * Latin.  Use 2.5 as a middle-of-the-road average. */
                    current.token_count += (std::max)(
                        static_cast<std::int32_t>(seg_text.size() / 3),
                        std::int32_t{1});
                }
                current.audio_ms = static_cast<double>(total_samples) * 1000.0 / 16000.0;
                current.updated_at = CurrentUnixSeconds();
                return true;
            };

              VadSegmentedBatchResult vad_result;
            try {
                vad_result = TranscribeFileVadSegmentedImpl(
                    batch_ctx,
                    batch_model->vad(),
                    prepared.wav_path.string().c_str(),
                    batch_lang,
                    batch_max_new_tokens,
                    batch_verbosity,
                    on_segment,
                    decode.cancel_callback,
                    &vad_mu);
            } catch (const std::exception &e) {
                vad_result.status = Status(StatusCode::kInternal,
                                           std::string("vad-segmented exception: ") + e.what());
            } catch (...) {
                vad_result.status = Status(StatusCode::kInternal,
                                           "vad-segmented unknown exception");
            }
      /* Free the batch clone; it holds per-batch decoder state
              * (KV cache, etc.) and must be released even on cancel. */
             batch_model->releaseInferHandle(batchHandle);
             CleanupPreparedAudio(&prepared);

            {
                std::lock_guard<std::mutex> lock(jobs_mu);
                OfflineJob & current = jobs[job_id];
                current.updated_at = CurrentUnixSeconds();
                current.language = DetectLanguageLabel(options.language);
                current.inference_ms = vad_result.inference_ms;
                current.audio_ms = vad_result.audio_ms;
                if (cancel_flag && cancel_flag->load()) {
                    current.state = "cancelled";
                    current.error.clear();
                    /* Keep whatever text the VAD segments emitted before cancel. */
                } else if (!vad_result.status.ok()) {
                    current.state = "failed";
                    current.error = vad_result.status.message();
                } else {
                    current.state = "completed";
                    /* If the VAD callback didn't already fill text
                     * (e.g. zero-segment edge case), fall back to
                     * vad_result.text. */
                    if (current.text.empty() && !vad_result.text.empty()) {
                        current.text = vad_result.text;
                    }
                }
                if (current.tokens == 0 && current.token_count > 0) {
                    current.tokens = current.token_count;
                }
            }
        }).detach();

        response.status = 202;
        SetJsonResponse(response, BuildJobJson(job));
    });

    server.Get("/api/jobs/:id", [&](const HttpRequest & request, HttpResponse & response) {
        const auto path_it = request.path_params.find("id");
        if (path_it == request.path_params.end()) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "job id is required"), 400);
            return;
        }
        std::lock_guard<std::mutex> lock(jobs_mu);
        const auto it = jobs.find(path_it->second);
        if (it == jobs.end()) {
            SetErrorResponse(response, Status(StatusCode::kNotFound, "job not found"), 404);
            return;
        }
        SetJsonResponse(response, BuildJobJson(it->second));
    });

    server.Post("/api/jobs/:id/cancel", [&](const HttpRequest & request, HttpResponse & response) {
        const auto path_it = request.path_params.find("id");
        if (path_it == request.path_params.end()) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "job id is required"), 400);
            return;
        }

        OfflineJob snapshot;
        {
            std::lock_guard<std::mutex> lock(jobs_mu);
            const auto it = jobs.find(path_it->second);
            if (it == jobs.end()) {
                SetErrorResponse(response, Status(StatusCode::kNotFound, "job not found"), 404);
                return;
            }

            OfflineJob & job = it->second;
            if (job.state != "completed" && job.state != "failed" && job.state != "cancelled") {
                job.cancel_requested = true;
                job.updated_at = CurrentUnixSeconds();
                if (job.cancel_flag) {
                    job.cancel_flag->store(true);
                }
                if (job.state == "queued" || job.state == "running") {
                    job.state = "cancelling";
                }
            }
            snapshot = job;
        }
        SetJsonResponse(response, BuildJobJson(snapshot));
    });

    server.Post("/v1/audio/transcriptions", [&](const HttpRequest & request, HttpResponse & response) {
        const MultipartFormData * file = FindUploadedAudio(request);
        if (file == nullptr) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "multipart field 'file' is required"), 400);
            return;
        }

        metrics.offline_requests.fetch_add(1);
        TranscriptionApiOptions options;
        const Status parse_status = ParseTranscriptionApiOptions(request, &options);
        if (!parse_status.ok()) {
            SetErrorResponse(response, parse_status, StatusToHttpCode(parse_status));
            return;
        }
        if (options.stream) {
            SetErrorResponse(
                response,
                Status(StatusCode::kFailedPrecondition, "OpenAI transcription stream is not enabled; use /v1/chat/completions with stream=true"),
                412);
            return;
        }

        PreparedAudioInput prepared;
        const Status prepare_status = PrepareUploadedAudio(*file, &prepared);
        if (!prepare_status.ok()) {
            SetErrorResponse(response, prepare_status, StatusToHttpCode(prepare_status));
            return;
        }

        ModelDecodeOptions decode;
        decode.prompt = options.prompt;
        decode.language = options.language;
        const AsrRunResult result = batch_model->TranscribeFile(prepared.wav_path, decode);
        CleanupPreparedAudio(&prepared);
        if (!result.status.ok()) {
            SetErrorResponse(response, result.status, StatusToHttpCode(result.status));
            return;
        }

        switch (options.response_format) {
            case TranscriptionResponseFormat::kText:
                response.set_content(result.text, "text/plain; charset=utf-8");
                return;
            case TranscriptionResponseFormat::kVerboseJson:
                SetJsonResponse(response, BuildVerboseTranscriptionJson(result, options));
                return;
            case TranscriptionResponseFormat::kJson:
            default:
                SetJsonResponse(response, Json::object({{"text", result.text}}));
                return;
        }
    });

    server.Post("/v1/chat/completions", [&](const HttpRequest & request, HttpResponse & response) {
        metrics.chat_requests.fetch_add(1);
        ChatCompletionRequestOptions options;
        const Status parse_status = ParseChatCompletionRequest(request, &options);
        if (!parse_status.ok()) {
            SetErrorResponse(response, parse_status, StatusToHttpCode(parse_status));
            return;
        }

        PreparedAudioInput prepared;
        const Status prepare_status = PrepareAudioLocator(options.audio_locator, &prepared);
        if (!prepare_status.ok()) {
            SetErrorResponse(response, prepare_status, StatusToHttpCode(prepare_status));
            return;
        }

        const std::string model_id = options.model.empty() ? served_model_id : options.model;
        const std::string request_id = "chatcmpl-" + std::to_string(session_counter.fetch_add(1));
        ModelDecodeOptions decode;
        decode.prompt = options.prompt;
        decode.language = options.language;
        const AsrRunResult result = batch_model->TranscribeFile(prepared.wav_path, decode);
        CleanupPreparedAudio(&prepared);
        if (!result.status.ok()) {
            SetErrorResponse(response, result.status, StatusToHttpCode(result.status));
            return;
        }

        if (!options.stream) {
            SetJsonResponse(response, BuildChatCompletionResponse(request_id, model_id, result));
            return;
        }

        response.set_header("Cache-Control", "no-cache");
        response.set_header("X-Accel-Buffering", "no");
        std::string sse;
        sse += BuildSseData(BuildChatChunk(request_id, model_id, "", true, false));
        sse += BuildSseData(BuildChatChunk(request_id, model_id, result.text, false, false));
        sse += BuildSseData(BuildChatChunk(request_id, model_id, "", false, true));
        sse += "data: [DONE]\n\n";
        response.set_content(sse, "text/event-stream");
    });

    server.Post("/v1/realtime", [&](const HttpRequest & request, HttpResponse & response) {
        OpenAiRealtimeRequest realtime_request;
        const Status parse_status = ParseOpenAiRealtimeRequest(request.body, &realtime_request);
        if (!parse_status.ok()) {
            SetErrorResponse(response, parse_status, StatusToHttpCode(parse_status));
            return;
        }

        DecodeRequestOptions decode_request;
        decode_request.task_mode = TaskMode::kStreaming;
        const Status validate_status = ValidateOpenAiRequest(
            OpenAiEndpoint::kRealtimeSessions,
            decode_request,
            realtime_request.stream);
        if (!validate_status.ok()) {
            SetErrorResponse(response, validate_status, StatusToHttpCode(validate_status));
            return;
        }

        const std::string model_id = realtime_request.model.empty() ? served_model_id : realtime_request.model;
        if (realtime_request.action == OpenAiRealtimeAction::kSessionCreate) {
            RealtimeSessionSnapshot session;
            const Status status = CreateRealtimeSession(model_id, realtime_request.language, &session);
            if (!status.ok()) {
                SetErrorResponse(response, status, StatusToHttpCode(status));
                return;
            }
            SetJsonResponse(
                response,
                BuildOpenAiRealtimeEventJson(
                    session,
                    "session.created",
                    false,
                    session.model.empty() ? model_id : session.model,
                    realtime_policy));
            return;
        }

        if (realtime_request.action == OpenAiRealtimeAction::kInputAudioBufferAppend) {
            std::vector<float> chunk;
            const Status decode_status = DecodeBase64Pcm16Le(realtime_request.audio, &chunk);
            if (!decode_status.ok()) {
                SetErrorResponse(response, decode_status, StatusToHttpCode(decode_status));
                return;
            }

            RealtimeSessionSnapshot session;
            const Status status = AppendRealtimeChunk(realtime_request.session_id, chunk, &session);
            if (!status.ok()) {
                SetErrorResponse(response, status, StatusToHttpCode(status));
                return;
            }
            SetJsonResponse(
                response,
                BuildOpenAiRealtimeEventJson(
                    session,
                    session.last_decode_ran ? "transcription.delta" : "input_audio_buffer.appended",
                    false,
                    session.model.empty() ? model_id : session.model,
                    realtime_policy));
            return;
        }

        RealtimeSessionSnapshot session;
        const Status status = FinalizeRealtimeSession(realtime_request.session_id, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        SetJsonResponse(
            response,
            BuildOpenAiRealtimeEventJson(
                session,
                "transcription.done",
                true,
                session.model.empty() ? model_id : session.model,
                realtime_policy));
    });

    server.Post("/api/realtime/start", [&](const HttpRequest &, HttpResponse & response) {
        RT_LOG("HTTP POST /api/realtime/start enter");
        RealtimeSessionSnapshot session;
        const Status status = CreateRealtimeSession(served_model_id, "", &session);
        RT_LOG("HTTP POST /api/realtime/start CreateRealtimeSession returned ok=%d sid=%s", status.ok() ? 1 : 0, session.id.c_str());
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        SetJsonResponse(response, Json::object({
            {"session_id", session.id},
            {"supported", true},
            {"decoded", false},
            {"sample_count", 0},
            {"retained_sample_count", 0},
            {"retained_sample_offset", 0},
            {"max_decode_window_ms", realtime_policy.max_decode_window_ms},
            {"stable_text", ""},
            {"partial_text", ""},
            {"text", ""},
        }));
    });

    server.Get("/api/realtime/status", [&](const HttpRequest & request, HttpResponse & response) {
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        RealtimeSessionSnapshot session;
        const Status status = GetRealtimeSessionSnapshot(session_id, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        SetJsonResponse(response, BuildRealtimeJson(session, false, true));
    });

    server.Post("/api/realtime/chunk", [&](const HttpRequest & request, HttpResponse & response) {
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        std::vector<float> chunk;
        {
            const Status decode_st = DecodePcm16Le(request.body, &chunk);
            if (!decode_st.ok()) {
                SetErrorResponse(response, decode_st, 400);
                return;
            }
            if (chunk.empty()) {
                SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "pcm16le body is required"), 400);
                return;
            }
        }

        RealtimeSessionSnapshot session;
        const Status status = AppendRealtimeChunk(session_id, chunk, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        SetJsonResponse(response, BuildRealtimeJson(session, false, true));
    });

    server.Post("/api/realtime/stop", [&](const HttpRequest & request, HttpResponse & response) {
        RT_LOG("HTTP POST /api/realtime/stop enter");
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        RT_LOG("HTTP POST /api/realtime/stop sid=%s", session_id.c_str());
        RealtimeSessionSnapshot session;
        const Status status = FinalizeRealtimeSession(session_id, &session);
        RT_LOG("HTTP POST /api/realtime/stop sid=%s Finalize returned ok=%d", session_id.c_str(), status.ok() ? 1 : 0);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
    Json body = BuildRealtimeJson(session, true, true);
         SetJsonResponse(response, body);
     });

     /* Translation proxy: browser calls this as same-origin, server
      * forwards to the configured MTranServer endpoint (which may be
      * on a different port or unreachable from the browser due to CORS
      * or mixed-content restrictions). */
     server.Post("/api/translation/translate",
         [&](const HttpRequest & request, HttpResponse & response) {
#ifdef QASR_CURL_AVAILABLE
         /* Read translation endpoint from env, fallback to default. */
         const char * env_endpoint = std::getenv("QASR_TRANSLATION_ENDPOINT");
         const std::string endpoint = env_endpoint ? env_endpoint : "http://127.0.0.1:8989/translate";

         /* Parse request body. */
         const std::string & body = request.body;
         if (body.empty()) {
             SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "request body required"), 400);
             return;
         }

         /* Build full URL: endpoint may or may not have trailing path. */
         std::string url = endpoint;
         if (!url.empty() && url.back() != '/') {
             url += "/";
         }
         /* Append /translate if not already present. */
         if (url.find("/translate") == std::string::npos) {
             url += "translate";
         }

         CURL * curl = curl_easy_init();
         if (!curl) {
             SetErrorResponse(response, Status(StatusCode::kInternal, "curl init failed"), 500);
             return;
         }

         std::string resp_body;
         long http_code = 0;
         {
             /* curl write callback must be C-linkage. */
             struct CurlWriteData { std::string * out; };
             CurlWriteData wdata;
             wdata.out = &resp_body;

             curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
             curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
             curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
             curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                 +[](char * ptr, size_t size, size_t nmemb, void * userdata) -> size_t {
                     CurlWriteData * wd = static_cast<CurlWriteData *>(userdata);
                     wd->out->append(ptr, size * nmemb);
                     return size * nmemb;
                 });
             curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wdata);
             struct curl_slist * headers = nullptr;
             headers = curl_slist_append(headers, "Content-Type: application/json");
             curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
             curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
             CURLcode res = curl_easy_perform(curl);
             if (res != CURLE_OK) {
         curl_slist_free_all(headers);
             curl_easy_cleanup(curl);
                 SetErrorResponse(response,
                     Status(StatusCode::kInternal, std::string("curl error: ") + curl_easy_strerror(res)),
                     502);
                 return;
             }
             curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
         }
         curl_easy_cleanup(curl);

         if (http_code < 200 || http_code >= 300) {
             SetErrorResponse(response,
                 Status(StatusCode::kInternal,
                     std::string("translation upstream returned ") + std::to_string(http_code) + ": " + resp_body),
                 502);
             return;
         }

         /* Pass through the response as JSON. */
         response.set_content(resp_body, "application/json");
#else
         SetErrorResponse(response, Status(StatusCode::kInternal, "translation proxy not available (curl not found)"), 501);
#endif
     });

    server.Post("/api/realtime/eof", [&](const HttpRequest & request, HttpResponse & response) {
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        std::shared_ptr<RealtimeSession> session;
        Status status = FindRealtimeSession(session_id, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        RealtimeLiveWorker * worker = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            worker = session->live_worker.get();
        }
        if (worker != nullptr && worker->live_ready) {
            FinishManualLiveAudio(&worker->live);
        }
        RealtimeSessionSnapshot snapshot;
        status = SnapshotRealtimeSessionState(session, false, &snapshot);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        SetJsonResponse(response, BuildRealtimeJson(snapshot, false, true));
    });

    /* Audio ingress diagnostic.  Surfaces last chunk's peak/RMS as
     * recorded server-side.  The UI polls this every 100 ms to render
     * an audio level meter that is impossible to miss, so we can
     * distinguish "browser mic muted" from "transport broken" from
     * "VAD model broken". */
    server.Get("/api/realtime/audio_diag", [&](const HttpRequest & request, HttpResponse & response) {
        const std::string session_id = request.get_param_value("session_id");
        if (session_id.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        std::shared_ptr<RealtimeSession> session;
        Status status = FindRealtimeSession(session_id, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        Json body;
        body["session_id"] = session_id;
        body["peak"] = session->last_ingress_peak;
        body["rms"] = session->last_ingress_rms;
        body["max_peak"] = session->max_ingress_peak;
        body["chunks"] = session->ingress_chunks;
        SetJsonResponse(response, body);
    });

    server.GetStream("/api/realtime/stream", [&](const HttpRequest & request, StreamWriter writer) {
        const std::string session_id = request.get_param_value("session_id");
        if (session_id.empty()) {
            writer(BuildSseData("{\"error\":\"session_id is required\"}"));
            return;
        }
        std::shared_ptr<RealtimeSession> session;
        {
            Status status = FindRealtimeSession(session_id, &session);
            if (!status.ok()) {
                writer(BuildSseData("{\"error\":\"session not found\"}"));
                return;
            }
        }
        /* Event-driven SSE: the ASR worker holds session->mu briefly
         * when committing a new segment or finalizing, and notifies
         * session->sse_cv.  We wait on that CV instead of polling.
         * Heartbeat every 5s keeps proxies from killing the
         * connection.  Per-session sse_last_segment_count lets us
         * emit only delta segments (full snapshot once at connect
         * time).  Paired with session->mu.
         *
         * Post-stop full-audio retranscription: after the VAD segments
         * are finalized, the SSE loop does NOT exit immediately.
         * Instead it enters Phase 2, waiting up to 60s for the
         * retranscription (reconciliation) result.  This way the
         * client receives the high-quality batch transcription on
         * the same SSE connection. */
        constexpr int kSseHeartbeatMs = 5000;
        constexpr int kReconcileTimeoutMs = 60000;
        std::size_t last_segment_count = 0;
        std::size_t last_candidate_count = 0;
        std::uint64_t last_partial_version = 0;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            last_segment_count = session->segments_text.size();
            last_candidate_count = session->candidates.size();
            session->sse_last_segment_count = last_segment_count;
            session->sse_last_candidate_count = last_candidate_count;
            last_partial_version = session->partial_version;
        }
        /* Initial full snapshot. */
        {
            RealtimeSessionSnapshot snapshot;
            SnapshotRealtimeSessionState(session, false, &snapshot);
            Json body = BuildRealtimeJson(snapshot, false, true);
            if (!writer(BuildSseData(body.dump()))) {
                return;
            }
            if (snapshot.finalized) {
                writer("data: [DONE]\n\n");
                return;
            }
        }
        while (true) {
            std::unique_lock<std::mutex> lock(session->mu);
            /* Wait for new segment or finalize or reconcile or
             * heartbeat timeout.  Also wake up on reconcile_ready
             * in case it fires before the SSE has seen finalized
             * (race-free on the CV). */
            session->sse_cv.wait_for(lock, std::chrono::milliseconds(kSseHeartbeatMs),
                [&] {
                    return session->finalized
                        || session->reconcile_ready
                        || session->segments_text.size() != session->sse_last_segment_count
                        || session->candidates.size() != session->sse_last_candidate_count
                        || session->partial_version != last_partial_version;
                });
            const std::uint64_t cur_partial_version = session->partial_version;
            const std::size_t cur_segment_count = session->segments_text.size();
            const std::size_t cur_candidate_count = session->candidates.size();
            const bool finalized = session->finalized;
            const bool reconcile_ready = session->reconcile_ready;

            /* ── P1: push new candidates (VAD tentative) ── */
            if (cur_candidate_count > last_candidate_count) {
                std::vector<std::string> new_candidates;
                for (std::size_t i = last_candidate_count; i < cur_candidate_count; ++i) {
                    new_candidates.push_back(session->candidates[i]);
                }
                last_candidate_count = cur_candidate_count;
                session->sse_last_candidate_count = cur_candidate_count;
                const double inf_ms = session->last_inference_ms;
                const std::size_t total_samples = session->total_samples;
                const std::string live_text = session->display_snapshot.live_text;
                lock.unlock();

                Json evt;
                evt["type"] = "update";
                evt["event_type"] = "transcript.candidate";
                evt["candidate_count"] = cur_candidate_count;
                Json cand_json = Json::array();
                for (const auto & s : new_candidates) cand_json.push_back(s);
                evt["new_candidates"] = cand_json;
                evt["total_samples"] = total_samples;
                evt["last_inference_ms"] = inf_ms;
                evt["live_text"] = live_text;
                if (qwen_verbose >= 1 && !new_candidates.empty()) {
                    std::fprintf(stderr,
                        "SSE-push sid=%s: %zu new candidates (total=%zu)\n",
                        session_id.c_str(), new_candidates.size(), cur_candidate_count);
                }
                if (!writer(BuildSseData(evt.dump()))) break;
                continue;
            }

            /* ── P1: push new finalized segments (two-pass confirmed) ── */
            if (cur_segment_count > session->sse_last_segment_count) {
                std::vector<std::string> new_finals;
                std::vector<std::size_t> new_positions;
                for (std::size_t i = session->sse_last_segment_count; i < cur_segment_count; ++i) {
                    new_finals.push_back(session->segments_text[i]);
                    if (i < session->segments_sample_positions.size()) {
                        new_positions.push_back(session->segments_sample_positions[i]);
                    } else {
                        new_positions.push_back(session->total_samples);
                    }
                }
                session->sse_last_segment_count = cur_segment_count;
                const double inf_ms = session->last_inference_ms;
                const std::size_t total_samples = session->total_samples;
                lock.unlock();

                Json evt;
                evt["type"] = "update";
                evt["event_type"] = "transcript.final";
                evt["segment_count"] = cur_segment_count;
                Json final_json = Json::array();
                for (const auto & s : new_finals) final_json.push_back(s);
                evt["new_segments"] = final_json;
                Json pos_json = Json::array();
                for (const auto & p : new_positions) pos_json.push_back(p);
                evt["new_segment_positions"] = pos_json;
                evt["total_samples"] = total_samples;
                evt["last_inference_ms"] = inf_ms;
                if (qwen_verbose >= 1 && !new_finals.empty()) {
                    std::fprintf(stderr,
                        "SSE-push sid=%s: %zu new final segments (total=%zu)\n",
                        session_id.c_str(), new_finals.size(), cur_segment_count);
                }
                if (!writer(BuildSseData(evt.dump()))) break;
                continue;
            }

            /* ── Phase 2: wait for retranscription ── */
            if (finalized && !reconcile_ready) {
                session->sse_last_segment_count = cur_segment_count;
                session->sse_last_candidate_count = cur_candidate_count;
                const std::string text = session->text;
                const std::string stable_text = session->stable_text;
                lock.unlock();

                /* Send final-fallback event now; candidates become
                 * the fallback if finalizer didn't produce results. */
                Json evt;
                evt["type"] = "update";
                evt["event_type"] = "transcript.final";
                evt["finalized"] = true;
                evt["text"] = text;
                evt["stable_text"] = stable_text;
                if (!writer(BuildSseData(evt.dump()))) break;

                /* Phase 2: wait for retranscription (up to 60s) */
                RT_LOG("SSE sid=%s entering reconcile wait (60s timeout)", session_id.c_str());
                {
                    std::unique_lock<std::mutex> rl(session->mu);
                    session->sse_cv.wait_for(rl,
                        std::chrono::milliseconds(kReconcileTimeoutMs),
                        [&] { return session->reconcile_ready; });
                    const bool ready = session->reconcile_ready;
                    const std::string rtext = session->reconcile_text;
                    rl.unlock();

                    if (ready && !rtext.empty()) {
                        RT_LOG("SSE sid=%s reconcile received: %s",
                               session_id.c_str(), rtext.c_str());
                        Json rev;
                        rev["type"] = "update";
                        Json rsegs = Json::array();
                        rsegs.push_back(rtext);
                        rev["new_segments"] = rsegs;
                        rev["new_segment_count"] = 1;
                        rev["text"] = rtext;
                        rev["stable_text"] = rtext;
                        rev["event_type"] = "transcript.final";
                        rev["reconciled"] = true;
                        rev["finalized"] = true;
                        rev["revised"] = session->reconcile_revised;
                        if (qwen_verbose >= 1) {
                            std::fprintf(stderr,
                                "SSE-push sid=%s: reconcile result: %s\n",
                                session_id.c_str(), rtext.c_str());
                        }
                        writer(BuildSseData(rev.dump()));
                    } else {
                        RT_LOG("SSE sid=%s reconcile timeout or empty", session_id.c_str());
                    }
                }
                writer("data: [DONE]\n\n");
                break;
            }

        /* ── Partial / live text push (no new candidates or segments) ── */
            if (cur_partial_version != last_partial_version) {
                session->sse_last_segment_count = cur_segment_count;
                const std::string live_stable = session->display_snapshot.live_stable_text;
                const std::string live_partial = session->display_snapshot.live_partial_text;
                const std::string live_text = session->display_snapshot.live_text;
                const double inf_ms = session->last_inference_ms;
                const std::size_t total_samples = session->total_samples;
                last_partial_version = cur_partial_version;
                lock.unlock();

                Json pev;
                pev["type"] = "update";
                pev["partial_version"] = cur_partial_version;
                pev["live_stable_text"] = live_stable;
                pev["live_partial_text"] = live_partial;
                pev["live_text"] = live_text;
                pev["last_inference_ms"] = inf_ms;
                pev["total_samples"] = total_samples;
                if (!writer(BuildSseData(pev.dump()))) break;
                continue;
            }

            /* Heartbeat — no new data, just keep the connection alive. */
            session->sse_last_segment_count = cur_segment_count;
            lock.unlock();
        }
    });

    server.Get("/api/capture/status", [&](const HttpRequest &, HttpResponse & response) {
        std::shared_ptr<HostCaptureSession> capture;
        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            capture = host_capture;
        }
        if (!capture) {
            SetJsonResponse(response, Json::object({{"active", false}, {"supported", true}}));
            return;
        }

        std::lock_guard<std::mutex> lock(capture->mu);
        if (capture->live_worker) {
            /* Use the worker's cumulative counter (VAD-updated),
             * NOT live->decoded_cursor (live-buffer-relative, reset
             * to 0 by TrimConsumedLiveAudio).  Same rationale as
             * the realtime snapshot path. */
            const int64_t dc = capture->live_worker->cumulative_decoded_samples.load(
                std::memory_order_relaxed);
            capture->decoded_samples = dc > 0 ? static_cast<std::size_t>(dc) : 0U;
        }
        Json body = BuildRealtimeJson(*capture, false, true);
        body["active"] = capture->active;
        body["capture_id"] = capture->id;
        body["backend"] = capture->backend;
        body["device"] = capture->device;
        body["error"] = capture->error;
        SetJsonResponse(response, body);
    });

    server.Post("/api/capture/start", [&](const HttpRequest & request, HttpResponse & response) {
        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            if (host_capture && host_capture->active) {
                SetErrorResponse(response, Status(StatusCode::kFailedPrecondition, "host capture is already active"), 409);
                return;
            }
        }

        std::string backend = "auto";
        std::string device;
        if (!request.body.empty()) {
            Json body = Json::parse(request.body);
            if (!body.is_discarded() && body.is_object()) {
                backend = body.value("backend", backend);
                device = body.value("device", device);
            }
        }
        if (request.has_param("backend")) {
            backend = request.get_param_value("backend");
        }
        if (request.has_param("device")) {
            device = request.get_param_value("device");
        }

        std::vector<std::string> argv;
        std::string selected_backend;
        const Status build_capture_status = BuildCaptureCommand(backend, device, &argv, &selected_backend);
        if (!build_capture_status.ok()) {
            SetErrorResponse(response, build_capture_status, StatusToHttpCode(build_capture_status));
            return;
        }

        auto capture = std::make_shared<HostCaptureSession>();
        capture->id = std::to_string(session_counter.fetch_add(1));
        capture->device = device;
        capture->backend = selected_backend;

#if defined(_WIN32)
        const Status spawn_status = SpawnCaptureProcess(argv, &capture->child_process, &capture->read_handle);
#else
        const Status spawn_status = SpawnCaptureProcess(argv, &capture->child_pid, &capture->read_fd);
#endif
        if (!spawn_status.ok()) {
            SetErrorResponse(response, spawn_status, StatusToHttpCode(spawn_status));
            return;
        }
        metrics.host_capture_sessions_started.fetch_add(1);

        const Status live_status = StartHostCaptureLiveWorker(capture);
        if (!live_status.ok()) {
            StopHostCaptureSession(capture);
            SetErrorResponse(response, live_status, StatusToHttpCode(live_status));
            return;
        }

       capture->reader = std::thread([capture, realtime_policy, &metrics]() {
            std::vector<char> buffer(6400);
        while (true) {
            #if defined(_WIN32)
                DWORD bytes_read = 0;
                if (!ReadFile(capture->read_handle, buffer.data(),
                              static_cast<DWORD>(buffer.size()), &bytes_read, NULL) ||
                    bytes_read == 0) {
                    break;
                }
                const std::size_t n_read = static_cast<std::size_t>(bytes_read);
#else
                const ssize_t raw_read = read(capture->read_fd, buffer.data(), buffer.size());
                if (raw_read <= 0) {
                    break;
                }
                const std::size_t n_read = static_cast<std::size_t>(raw_read);
#endif

              std::vector<float> chunk;
                 {
                     const Status decode_st = DecodePcm16Le(buffer.data(), static_cast<std::size_t>(n_read), &chunk);
                     if (!decode_st.ok()) {
                         RT_LOG("HostCaptureLoop sid=%s decode failed: %s",
                                capture->id.c_str(), decode_st.message().c_str());
                         continue;
                     }
                     if (chunk.empty()) continue;
                 }

                RealtimeLiveWorker * worker = nullptr;
                {
                    std::lock_guard<std::mutex> lock(capture->mu);
                    AppendRealtimeSamples(realtime_policy, chunk, capture.get());
                    worker = capture->live_worker.get();
                }

                if (worker == nullptr || !worker->live_ready) {
                    std::lock_guard<std::mutex> lock(capture->mu);
                    capture->error = "capture live worker is not ready";
                    break;
                }

                const Status append_status = AppendManualLiveAudio(&worker->live, chunk.data(), chunk.size());
                if (!append_status.ok()) {
                    std::lock_guard<std::mutex> lock(capture->mu);
                    capture->error = append_status.message();
                    break;
                }
                metrics.realtime_decode_runs.fetch_add(1);
            }

            bool stopped_by_request = false;
            {
                std::lock_guard<std::mutex> lock(capture->mu);
                if (capture->live_worker) {
                    FinishManualLiveAudio(&capture->live_worker->live);
                }
                capture->active = false;
                stopped_by_request = capture->stop_requested;
            }
            (void)stopped_by_request;
        });

        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            host_capture = capture;
        }

        Json body;
        body["capture_id"] = capture->id;
        body["backend"] = capture->backend;
        body["device"] = capture->device;
        body["supported"] = true;
        SetJsonResponse(response, body);
    });

    server.Post("/api/capture/stop", [&](const HttpRequest &, HttpResponse & response) {
        std::shared_ptr<HostCaptureSession> capture;
        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            capture = host_capture;
            host_capture.reset();
        }
        if (!capture) {
            SetErrorResponse(response, Status(StatusCode::kNotFound, "host capture is not active"), 404);
            return;
        }

        StopHostCaptureSession(capture);

        metrics.realtime_finalizations.fetch_add(1);
        std::lock_guard<std::mutex> lock(capture->mu);
        if (capture->live_worker) {
            /* See /api/capture/status above: use cumulative counter,
             * not live->decoded_cursor. */
            const int64_t dc = capture->live_worker->cumulative_decoded_samples.load(
                std::memory_order_relaxed);
            capture->decoded_samples = dc > 0 ? static_cast<std::size_t>(dc) : 0U;
        }
        Json body = BuildRealtimeJson(*capture, true, true);
        body["capture_id"] = capture->id;
        body["backend"] = capture->backend;
        body["device"] = capture->device;
        body["error"] = capture->error;
        SetJsonResponse(response, body);
    });

    std::fprintf(stderr, "qasr_server listening on %s:%d (verbosity=%d%s)\n",
                 config.host.c_str(), config.port, config.verbosity,
                 config.verbosity == 0 ? ", quiet mode" : "");
    const bool ok = server.listen(config.host, config.port);
    {
        std::lock_guard<std::mutex> lock(maintenance_mu);
        stop_maintenance = true;
    }
    maintenance_cv.notify_all();
    if (job_cleanup_thread.joinable()) {
        job_cleanup_thread.join();
    }
    /* VAD lives inside the Qwen context (ctx->vad, created by
     * qwen_load).  It is destroyed automatically when the
     * SharedAsrModel is destroyed (at scope exit, ~SharedAsrModel
     * calls qwen_free which calls qwen_silero_vad_destroy on
     * ctx->vad).  No explicit destruction needed here. */
    if (!ok) {
        std::fprintf(stderr, "qasr_server listen failed on %s:%d\n", config.host.c_str(), config.port);
        return 1;
    }
    return 0;
#endif
}

}  // namespace qasr
/* TODO(god-function-C1): RunServer (server.cc:2610-4394) is 1784 lines,
 * covering all HTTP route registration, lambdas for VAD-segmented batch
 * decode, live worker boot, async transcription job dispatch, and
 * shutdown.  Suggested split (whenever maintenance cost outweighs the
 * churn risk):
 *   - server_routes.cc     Register* handlers + HttpServer wiring
 *   - server_session.cc    SessionManager, RealtimeSession lifecycle
 *   - server_vad.cc        RunVadSegmentedDecode + TranscribeFileVadSegmentedImpl
 *   - server_live.cc       StartRealtimeLiveWorker + StartHostCaptureLiveWorker
 * See docs/AUDIT_C1.md §5.1 for the full god-function inventory and
 * docs/AUDIT_C1.md §5.2 for the god-file inventory. */
