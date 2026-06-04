#include "qasr/service/server.h"
#include "qasr/service/realtime.h"

#include <atomic>
#include <cmath>
#include <charconv>
#include <chrono>
#include <condition_variable>
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

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_silero_vad.h"
}
#include "qasr/base/http_server.h"
#include "qasr/base/process_spawn.h"
#endif

#include "qasr/runtime/model_bridge.h"

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

std::vector<float> DecodePcm16Le(const std::string & body) {
    std::vector<float> samples;
    if ((body.size() % 2U) != 0U) {
        return samples;
    }
    samples.resize(body.size() / 2U);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const unsigned char low = static_cast<unsigned char>(body[index * 2U]);
        const unsigned char high = static_cast<unsigned char>(body[index * 2U + 1U]);
        const std::int16_t value = static_cast<std::int16_t>(static_cast<std::uint16_t>(low) |
            (static_cast<std::uint16_t>(high) << 8U));
        samples[index] = static_cast<float>(value) / 32768.0f;
    }
    return samples;
}

#if defined(QASR_CPU_BACKEND_ENABLED)
std::vector<float> DecodePcm16Le(const char * data, std::size_t size) {
    return DecodePcm16Le(std::string(data, size));
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

class SharedAsrModel {
public:
    Status Load(const ServerConfig & config) {
        Status status = ValidateModelDirectory(config.model_dir);
        if (!status.ok()) {
            return status;
        }

        config_ = config;
        qwen_verbose = config.verbosity;
        qwen_monitor = 0;
        const int threads = config.threads > 0 ? config.threads : qwen_get_num_cpus();
        qwen_set_threads(threads);
        ctx_ = qwen_load(config.model_dir.c_str());
        if (ctx_ == nullptr) {
            return Status(StatusCode::kInternal, "qwen_load failed");
        }
        if (config.decoder_int8) {
            if (qwen_set_decoder_int8(ctx_, 1) != 0) {
                std::fprintf(stderr, "warning: decoder INT8 init failed, falling back to BF16\n");
            }
        }
        if (config.encoder_int8) {
            if (qwen_set_encoder_int8(ctx_, 1) != 0) {
                std::fprintf(stderr, "warning: encoder INT8 init failed, falling back to F32\n");
            }
        }
        ctx_->past_text_conditioning = 1;
        ctx_->segment_sec = 30.0f;
        if (config.temperature >= 0.0f) {
            ctx_->decode_temperature = config.temperature;
        }
        return OkStatus();
    }

    ~SharedAsrModel() {
        if (ctx_ != nullptr) {
            qwen_free(ctx_);
        }
    }

    AsrRunResult TranscribeFile(const fs::path & audio_path, const ModelDecodeOptions & decode) {
        std::lock_guard<std::mutex> lock(mu_);
        AsrRunResult result;
        if (ctx_ == nullptr) {
            result.status = Status(StatusCode::kFailedPrecondition, "model is not loaded");
            return result;
        }

        qwen_verbose = config_.verbosity;
        ctx_->stream_max_new_tokens = decode.stream_max_new_tokens;
        if (decode.stream_chunk_sec > 0.0f) {
            ctx_->stream_chunk_sec = decode.stream_chunk_sec;
        }
        if (decode.temperature >= 0.0f) {
            ctx_->decode_temperature = decode.temperature;
        } else if (config_.temperature >= 0.0f) {
            ctx_->decode_temperature = config_.temperature;
        }
        if (qwen_set_prompt(ctx_, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx_, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + decode.language);
            return result;
        }

        std::function<void(std::string_view)> token_callback = decode.token_callback;
        std::function<bool()> cancel_callback = decode.cancel_callback;
        qwen_set_token_callback(ctx_, token_callback ? ForwardTokenPiece : nullptr, token_callback ? &token_callback : nullptr);
        qwen_set_cancel_callback(ctx_, cancel_callback ? ForwardCancelRequest : nullptr, cancel_callback ? &cancel_callback : nullptr);

        char * raw = qwen_transcribe(ctx_, audio_path.string().c_str());
        const bool was_cancelled = qwen_was_cancelled(ctx_) != 0;
        qwen_set_cancel_callback(ctx_, nullptr, nullptr);
        qwen_set_token_callback(ctx_, nullptr, nullptr);
        if (raw == nullptr) {
            result.status = was_cancelled
                ? Status(StatusCode::kFailedPrecondition, "transcription cancelled")
                : Status(StatusCode::kInternal, "transcription failed");
            return result;
        }

        result.text = raw;
        std::free(raw);
        result.total_ms = ctx_->perf_total_ms;
        result.audio_ms = ctx_->perf_audio_ms;
        result.text_tokens = ctx_->perf_text_tokens;
        result.encode_ms = ctx_->perf_encode_ms;
        result.decode_ms = ctx_->perf_decode_ms;
        result.status = was_cancelled
            ? Status(StatusCode::kFailedPrecondition, "transcription cancelled")
            : OkStatus();
        return result;
    }

    AsrRunResult TranscribeRealtime(const std::vector<float> & samples, const ModelDecodeOptions & decode) {
        std::lock_guard<std::mutex> lock(mu_);
        AsrRunResult result;
        if (ctx_ == nullptr) {
            result.status = Status(StatusCode::kFailedPrecondition, "model is not loaded");
            return result;
        }

        qwen_verbose = config_.verbosity;
        ctx_->stream_max_new_tokens = decode.stream_max_new_tokens;
        if (decode.temperature >= 0.0f) {
            ctx_->decode_temperature = decode.temperature;
        } else if (config_.temperature >= 0.0f) {
            ctx_->decode_temperature = config_.temperature;
        }
        if (qwen_set_prompt(ctx_, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx_, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + decode.language);
            return result;
        }

        std::function<void(std::string_view)> token_callback = decode.token_callback;
        std::function<bool()> cancel_callback = decode.cancel_callback;
        qwen_set_token_callback(ctx_, token_callback ? ForwardTokenPiece : nullptr, token_callback ? &token_callback : nullptr);
        qwen_set_cancel_callback(ctx_, cancel_callback ? ForwardCancelRequest : nullptr, cancel_callback ? &cancel_callback : nullptr);

        char * raw = decode.use_stream_path
            ? qwen_transcribe_stream(ctx_, samples.data(), static_cast<int>(samples.size()))
            : qwen_transcribe_audio(ctx_, samples.data(), static_cast<int>(samples.size()));
        const bool was_cancelled = qwen_was_cancelled(ctx_) != 0;
        qwen_set_cancel_callback(ctx_, nullptr, nullptr);
        qwen_set_token_callback(ctx_, nullptr, nullptr);
        if (raw == nullptr) {
            result.status = was_cancelled
                ? Status(StatusCode::kFailedPrecondition, decode.use_stream_path ? "stream transcription cancelled" : "audio transcription cancelled")
                : Status(StatusCode::kInternal, decode.use_stream_path ? "stream transcription failed" : "audio transcription failed");
            return result;
        }

        result.text = raw;
        std::free(raw);
        result.total_ms = ctx_->perf_total_ms;
        result.audio_ms = ctx_->perf_audio_ms;
        result.text_tokens = ctx_->perf_text_tokens;
        result.encode_ms = ctx_->perf_encode_ms;
        result.decode_ms = ctx_->perf_decode_ms;
        result.status = was_cancelled
            ? Status(StatusCode::kFailedPrecondition, decode.use_stream_path ? "stream transcription cancelled" : "audio transcription cancelled")
            : OkStatus();
        return result;
    }

    qwen_ctx_t * CreateRealtimeClone() {
        std::lock_guard<std::mutex> lock(mu_);
        if (ctx_ == nullptr) {
            return nullptr;
        }
        qwen_ctx_t * clone = qwen_clone_shared(ctx_);
        if (clone == nullptr) {
            return nullptr;
        }
        /* qwen_clone_shared() resets decoder_int8/encoder_int8 to 0 on the
         * clone (INT8 resources are per-context, see qwen_asr.c). Without
         * re-applying the configured INT8 flags here, every realtime/host
         * capture session silently runs on the BF16/F32 path even when the
         * server was started with INT8 enabled.
         *
         * Decoder INT8 is the autoregressive Qwen3 LM and INT8 quantization
         * there has a measurable impact on language consistency (e.g. English
         * fragments leaking into Chinese audio with onomatopoeia, repeated
         * words, etc.). For realtime sessions we therefore default to BF16
         * decoder regardless of --decoder-int8, and only opt in when the
         * operator explicitly passes --realtime-decoder-int8. */
        if (config_.decoder_int8 && config_.realtime_decoder_int8) {
            if (qwen_set_decoder_int8(clone, 1) != 0) {
                std::fprintf(stderr,
                             "warning: realtime clone decoder INT8 init failed, falling back to BF16\n");
            }
        }
        if (config_.encoder_int8) {
            if (qwen_set_encoder_int8(clone, 1) != 0) {
                std::fprintf(stderr,
                             "warning: realtime clone encoder INT8 init failed, falling back to F32\n");
            }
        }
        return clone;
    }

    int verbosity() const noexcept {
        return config_.verbosity;
    }

    float temperature() const noexcept {
        return config_.temperature;
    }

private:
    ServerConfig config_;
    qwen_ctx_t * ctx_ = nullptr;
    std::mutex mu_;
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

struct RealtimeSession {
    std::mutex mu;
    std::string id;
    std::string model;
    std::string language;
    std::vector<float> samples;
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
    double current_segment_audio_sec = 0.0;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
    bool finalized = false;
    std::string error;
};

struct RealtimeLiveWorker {
    qwen_live_audio_t live{};
    std::thread thread;
    bool live_ready = false;
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

struct HostCaptureSession {
    std::string id;
    std::string backend;
    std::string device;
    std::vector<float> samples;
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
    FinishManualLiveAudio(&worker->live);
    if (worker->thread.joinable()) {
        worker->thread.join();
    }
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
    snapshot.current_segment_audio_sec = session.current_segment_audio_sec;
    snapshot.last_inference_ms = session.last_inference_ms;
    snapshot.last_decode_ran = session.last_decode_ran;
    snapshot.finalized = session.finalized;
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
    return ValidateModelDirectory(config.model_dir);
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
        if (arg == "--decoder-int8") {
            config->decoder_int8 = true;
            continue;
        }
        if (arg == "--encoder-int8") {
            config->encoder_int8 = true;
            continue;
        }
        if (arg == "--realtime-decoder-int8") {
            config->realtime_decoder_int8 = true;
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
    usage += "  --host <ip>\n";
    usage += "  --port <n>\n";
    usage += "  --ui-dir <dir>\n";
    usage += "  --threads <n>\n";
    usage += "  --verbosity <n>          0=silent (recommended for production),\n";
    usage += "                           1=commit/summary, 2=per-poll, 3=raw\n";
    usage += "  --quiet, -q              alias for --verbosity 0\n";
    usage += "  --temperature <float>  (default: auto, 0=greedy, >0=sampling)\n";
    usage += "  --decoder-int8           (decoder INT8: saves memory but DEGRADES quality:\n";
    usage += "                            language consistency drops, code-switch leakage,\n";
    usage += "                            hallucinations on low-confidence audio; use only when memory-bound)\n";
    usage += "  --encoder-int8\n";
    usage += "  --realtime-decoder-int8  (apply --decoder-int8 to realtime sessions; default off for language consistency)\n";
    usage += "  -h, --help\n";
    return usage;
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

    SharedAsrModel model;
    const Status load_status = model.Load(config);
    if (!load_status.ok()) {
        std::fprintf(stderr, "model load failed: %s\n", load_status.message().c_str());
        return 1;
    }

    const std::string served_model_id = ResolveServedModelId(config.model_dir);
    const fs::path ui_dir(config.ui_dir);
    const RealtimePolicyConfig realtime_policy;
    const auto server_start = std::chrono::steady_clock::now();
    ServerMetrics metrics;
    std::atomic<std::uint64_t> session_counter{1};
    std::unordered_map<std::string, std::shared_ptr<RealtimeSession>> realtime_sessions;
    std::mutex realtime_mu;
    std::unordered_map<std::string, OfflineJob> jobs;
    std::mutex jobs_mu;
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
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->live_worker) {
            LockLiveAudio(&session->live_worker->live);
            const int64_t dc = session->live_worker->live.decoded_cursor;
            UnlockLiveAudio(&session->live_worker->live);
            session->decoded_samples = dc > 0 ? static_cast<std::size_t>(dc) : 0U;
        }
        *snapshot = SnapshotRealtimeSession(*session);
        if (consume_decode_flag) {
            session->last_decode_ran = false;
        }
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
    constexpr int kVadSegmentSilenceFrames = 10;          /* 10 × 50 ms poll = 500 ms */
    constexpr int kVadSegmentMaxSamples = 40 * 16000;      /* 40 s safety cap */
    constexpr int kVadSegmentPollMs = 50;                 /* poll live buffer every 50 ms */
    constexpr float kVadSpeechProbThreshold = 0.35f;      /* match C-layer threshold */

    auto RunVadSegmentedDecode = [&](qwen_ctx_t * live_ctx,
                                     qwen_live_audio_t * live,
                                     const std::shared_ptr<RealtimeSession> & session,
                                     const std::string & forced_language) {
        /* Set up Silero VAD (best effort: if it fails to load, the
         * loop falls back to a pure 40 s timer, which is suboptimal
         * but still gives stable output). */
        qwen_silero_vad_t *vad = qwen_silero_vad_create(nullptr);
        const bool vad_active = qwen_silero_vad_is_active(vad);
        if (qwen_verbose >= 1) {
            std::fprintf(stderr, "VAD-segmented: Silero VAD %s\n",
                         vad_active ? "active" : "inactive (will use 40s timer only)");
        }

        /* Configure ctx for offline-style per-segment decode.  No
         * streaming chunk cadence, no coalesce step, no partial. */
        live_ctx->segment_sec = 0.0f;     /* never split inside a segment */
        live_ctx->search_sec = 0.0f;
        live_ctx->past_text_conditioning = 0;  /* each segment is independent */
        live_ctx->stream_chunk_sec = 0.0f;
        live_ctx->stream_max_new_tokens = 0;   /* no streaming cap */
        if (qwen_set_force_language(live_ctx,
                                    forced_language.empty() ? nullptr
                                                            : forced_language.c_str()) != 0) {
            std::lock_guard<std::mutex> lock(session->mu);
            session->error = "unsupported language: " + forced_language;
            qwen_silero_vad_destroy(vad);
            return;
        }

        std::vector<float> segment_buffer;
        segment_buffer.reserve(kVadSegmentMaxSamples * 2);
        int64_t consumed_samples = 0;   /* total live-audio samples already pulled into segment_buffer */
        int silence_run = 0;           /* number of consecutive silent VAD frames */
        bool segment_active = false;   /* true between VAD speech-start and segment-end commit */
        const int max_new_tokens = 64; /* per-segment cap; matches offline behavior */

        auto commit_segment = [&](const char * reason) {
            if (segment_buffer.empty()) {
                silence_run = 0;
                return;
            }
            const int n_samples = static_cast<int>(segment_buffer.size());
            if (qwen_verbose >= 1) {
                std::fprintf(stderr,
                             "VAD-segmented: committing segment (%.2fs, reason=%s)\n",
                             (double)n_samples / 16000.0, reason);
            }
            char * raw = qwen_transcribe_audio(live_ctx, segment_buffer.data(), n_samples);
            if (raw != nullptr) {
                std::string text(raw);
                std::free(raw);
                /* Trim trailing whitespace the model may emit. */
                while (!text.empty() &&
                       (text.back() == ' ' || text.back() == '\n' || text.back() == '\t')) {
                    text.pop_back();
                }
                std::lock_guard<std::mutex> lock(session->mu);
                if (!text.empty()) {
                    session->segments_text.push_back(text);
                    /* The current segment becomes the new live_stable
                     * (the UI shows it in the "live" area until the
                     * next segment starts).  Drain stable into recent
                     * segments to mimic the Web Speech API pattern. */
                    RealtimeTextUpdate update;
                    update.committed = true;
                    update.stable_text = text;
                    update.partial_text.clear();
                    update.text = text;
                    session->stable_text = text;
                    session->text_state.stable_text = text;
                    session->text_state.last_text = text;
                    session->text_state.last_decode_samples = session->total_samples;
                    session->text_state.unstable_since_samples = session->total_samples;
                    session->last_inference_ms = live_ctx->perf_total_ms;
                    ApplyRealtimeUpdate(update, session->last_inference_ms,
                                        true, false, session.get());
                }
            } else if (qwen_verbose >= 1) {
                std::fprintf(stderr,
                             "VAD-segmented: qwen_transcribe_audio returned null\n");
            }
            segment_buffer.clear();
            silence_run = 0;
            segment_active = false;
            /* Reset the VAD's LSTM state on segment commit so a brief
             * pre-segment noise burst doesn't bleed into the next
             * segment's silence detection. */
            if (vad_active) qwen_silero_vad_reset(vad);
        };

        /* Outer poll loop: drain new audio from the live buffer,
         * run VAD, decide whether to commit.  50 ms poll is fine
         * because we only need coarse sentence-end detection.
         *
         * CRITICAL: silence_run must also advance when *no* new
         * audio arrives.  Otherwise, after a single burst of audio
         * is fully drained, subsequent polls see take==0 and the
         * VAD code never runs — silence_run stays 0 forever, and
         * the segment never commits on a pause.  The fix: every
         * poll, if the VAD is in "speech" state and we have not
         * received new audio for >= 1 VAD frame, treat that as a
         * single silent frame and increment silence_run.  This
         * matches the C-level VAD fix in qwen_asr.c. */
        int idle_polls = 0;  /* polls since last new audio */
        while (true) {
            /* Snapshot live state. */
            int64_t live_n = 0;
            int live_eof = 0;
            {
                LockLiveAudio(live);
                live_n = live->n_samples;
                live_eof = live->eof;
                UnlockLiveAudio(live);
            }

            /* Drain any new audio into the segment buffer. */
            if (live_n > consumed_samples) {
                int64_t take = live_n - consumed_samples;
                if (take > 0) {
                    /* Diagnostic: peak in the source range BEFORE we
                     * copy.  If src_peak > 0 but the destination peak
                     * (computed below) is 0, the memcpy is broken. */
                    float src_peak = 0.0f;
                    LockLiveAudio(live);
                    for (int64_t i = 0; i < take; ++i) {
                        const float a = live->samples[consumed_samples + i];
                        const float aa = a < 0 ? -a : a;
                        if (aa > src_peak) src_peak = aa;
                    }
                    UnlockLiveAudio(live);

                    /* Copy in chunks to avoid holding the lock for
                     * the full memcpy; per call we cap at 32k samples
                     * (≈ 2 s) to keep the lock window bounded.
                     *
                     * CRITICAL: resize() BEFORE memcpy().  std::vector
                     * value-initializes new elements to 0 on grow, so
                     * if we memcpy first and resize second, the resize
                     * overwrites our data with zeros — exactly the
                     * "src_peak=0.99 dst_post_copy=0.00" symptom. */
                    constexpr int64_t kCopyChunk = 32000;
                    int64_t copied = 0;
                    const std::size_t buf_old_size = segment_buffer.size();
                    while (copied < take) {
                        int64_t want = take - copied;
                        if (want > kCopyChunk) want = kCopyChunk;
                        const std::size_t old_end = segment_buffer.size();
                        segment_buffer.resize(old_end + static_cast<std::size_t>(want));
                        LockLiveAudio(live);
                        std::memcpy(segment_buffer.data() + old_end,
                                    live->samples + (consumed_samples + copied),
                                    static_cast<std::size_t>(want) * sizeof(float));
                        UnlockLiveAudio(live);
                        copied += want;
                    }
                    consumed_samples += take;

                    /* Diagnostic 2: peak in the destination range,
                     * computed IMMEDIATELY after the copy.  If this
                     * is non-zero but the post-VAD peak (computed
                     * later in this iteration) is 0, the segment
                     * buffer is getting cleared between the copy
                     * and the post-VAD peak check. */
                    float dst_peak_post_copy = 0.0f;
                    {
                        const std::size_t dst_new = segment_buffer.size();
                        for (std::size_t i = buf_old_size; i < dst_new; ++i) {
                            const float a = segment_buffer[i];
                            const float aa = a < 0 ? -a : a;
                            if (aa > dst_peak_post_copy) dst_peak_post_copy = aa;
                        }
                    }

                    if (qwen_verbose >= 2) {
                        std::fprintf(stderr,
                                     "VAD-segmented: src_peak=%.4f dst_post_copy=%.4f buf_old=%zu buf_new=%zu take=%lld\n",
                                     (double)src_peak, (double)dst_peak_post_copy,
                                     buf_old_size, segment_buffer.size(),
                                     static_cast<long long>(take));
                    }
                    /* Compute peak amplitude of the new audio so
                     * we can tell "VAD said silence because input
                     * is silent" apart from "VAD said silence
                     * because model is broken on real input".  This
                     * is the first thing to check when VAD never
                     * reports speech. */
                    float peak = 0.0f;
                    float sum_sq = 0.0f;
                    const std::size_t new_tail = segment_buffer.size();
                    const std::size_t new_tail_start = new_tail - static_cast<std::size_t>(take);
                    for (std::size_t i = new_tail_start; i < new_tail; ++i) {
                        float a = segment_buffer[i];
                        if (a > peak) peak = a;
                        if (a < -peak) peak = -a;
                        sum_sq += a * a;
                    }
                    float rms = std::sqrt(sum_sq / static_cast<float>(take > 0 ? take : 1));

                    /* Run VAD over the new samples (in 512-sample
                     * frames, the silero_vad chunk size).  We process
                     * the entire segment_buffer tail that we just
                     * appended; the VAD maintains its own context
                     * across calls.
                     *
                     * IMPORTANT: use int64_t for the math.  The
                     * previous version used size_t and underflowed
                     * on the first iteration (segment_buffer.size()
                     * == 0, take > 0), which produced a huge
                     * aligned_start that skipped the entire loop —
                     * hence VAD never fired and silence_run was
                     * always 0, so the only commit ever was at EOF. */
                    if (vad_active) {
                        const int frame = 512;
                        int64_t tail_start = static_cast<int64_t>(segment_buffer.size()) - take;
                        if (tail_start < 0) tail_start = 0;
                        /* Align down to a 512-sample boundary so
                         * we don't double-process samples the
                         * prior iteration already saw. */
                        int64_t aligned_start = (tail_start / frame) * frame;
                        int64_t total = static_cast<int64_t>(segment_buffer.size());
                        int frames_processed = 0;
                        float last_prob = 0.0f;
                        for (int64_t off = aligned_start + frame;
                             off <= total; off += frame) {
                            float prob = 0.0f;
                            qwen_silero_vad_process(vad,
                                                    segment_buffer.data() + (off - frame),
                                                    frame, &prob);
                            last_prob = prob;
                            frames_processed++;
                            if (prob >= kVadSpeechProbThreshold) {
                                segment_active = true;
                                silence_run = 0;
                            } else {
                                /* Only count silence if the segment is
                                 * already active (a leading silent
                                 * region shouldn't trigger a fake
                                 * commit on an empty buffer). */
                                if (segment_active) {
                                    silence_run++;
                                }
                            }
                        }
                        /* Unconditional debug log every poll that
                         * processes new audio.  We need this to
                         * verify VAD is actually being called and
                         * what prob it's returning.  Throttled to
                         * once per ~10 frames to keep the log
                         * readable. */
                        static int debug_counter = 0;
                        if (++debug_counter % 10 == 0) {
                            std::fprintf(stderr,
                                         "VAD-segmented: take=%lld frames=%d "
                                         "active=%d silence_run=%d last_prob=%.3f "
                                         "buf=%.2fs rms=%.4f peak=%.4f\n",
                                         static_cast<long long>(take),
                                         frames_processed,
                                         segment_active ? 1 : 0,
                                         silence_run,
                                         last_prob,
                                         static_cast<double>(segment_buffer.size()) / 16000.0,
                                         rms, peak);
                        }
                    }
                    idle_polls = 0;  /* reset: we got new audio */
                } else {
                    /* No new audio since last poll.  Track how long
                     * we've been idle.  When the VAD has been in
                     * "speech" state and the audio stream goes
                     * silent (the speaker paused), this is the
                     * case.  Treat each 50 ms poll after the audio
                     * has gone quiet as one silent frame:
                     * silence_run advances once per poll, so 16
                     * polls (≈ 800 ms) trigger a commit.
                     *
                     * The poll granularity is 50 ms (kVadSegmentPollMs);
                     * 16 polls = 800 ms, slightly looser than the
                     * 512 ms of audio-based silence.  This is
                     * intentional: the pause detection should be
                     * generous on mic streams because a single
                     * mid-word breath (200-300 ms) shouldn't
                     * trigger a commit. */
                    idle_polls++;
                    if (vad_active && segment_active) {
                        silence_run++;
                    } else if (!vad_active && segment_active) {
                        /* No VAD available: fall back to wall-clock
                         * idle counting.  Convert idle_polls to
                         * frame-equivalent silence (50 ms per poll,
                         * 32 ms per frame → roughly 1.5 frames per
                         * poll, but treat as 1 for the threshold to
                         * avoid over-counting). */
                        silence_run++;
                    }
                    if (qwen_verbose >= 2 && idle_polls % 5 == 0) {
                        std::fprintf(stderr,
                                     "VAD-segmented: idle_polls=%d silence_run=%d "
                                     "active=%d buf=%.2fs\n",
                                     idle_polls, silence_run,
                                     segment_active ? 1 : 0,
                                     static_cast<double>(segment_buffer.size()) / 16000.0);
                    }
                }
            }

            /* Update heartbeat (UI shows "Listening... X.Xs"). */
            {
                std::lock_guard<std::mutex> lock(session->mu);
                session->current_segment_audio_sec =
                    static_cast<double>(segment_buffer.size()) / 16000.0;
            }

            /* Decide whether to commit. */
            bool should_commit = false;
            const char * commit_reason = nullptr;
            if (segment_active && silence_run >= kVadSegmentSilenceFrames) {
                should_commit = true;
                commit_reason = "vad_silence";
            } else if (static_cast<int>(segment_buffer.size()) >= kVadSegmentMaxSamples) {
                should_commit = true;
                commit_reason = "40s_safety_cap";
            }
            if (should_commit) {
                commit_segment(commit_reason);
            }

            /* Exit conditions: EOF and no in-flight segment to flush. */
            if (live_eof) {
                if (!segment_buffer.empty()) {
                    commit_segment("eof");
                }
                break;
            }

            /* Sleep until next poll, with an early wake when new
             * audio arrives.  We use a small fixed sleep (50 ms) —
             * fine-grained early wake would only matter for the
             * first 500 ms after the user stops talking, and the
             * extra latency is acceptable. */
            std::this_thread::sleep_for(std::chrono::milliseconds(kVadSegmentPollMs));
        }

        qwen_silero_vad_destroy(vad);
    };

    auto StartRealtimeLiveWorker = [&](const std::shared_ptr<RealtimeSession> & session) -> Status {
        if (session == nullptr) {
            return Status(StatusCode::kInvalidArgument, "session must not be null");
        }

        auto worker = std::make_unique<RealtimeLiveWorker>();
        Status status = InitializeManualLiveAudio(&worker->live);
        if (!status.ok()) {
            return status;
        }
        worker->live_ready = true;

        qwen_ctx_t * live_ctx = model.CreateRealtimeClone();
        if (live_ctx == nullptr) {
            DestroyManualLiveAudio(&worker->live);
            return Status(StatusCode::kInternal, "failed to clone realtime model context");
        }

        const float stream_chunk_sec = RealtimeStreamChunkSeconds(realtime_policy);
        const int stream_max_new_tokens = RealtimeStreamMaxNewTokens(realtime_policy);
        const int verbosity = model.verbosity();
        const float temperature = model.temperature();
        const std::string forced_language = session->language;

        worker->thread = std::thread([
            session,
            worker_ptr = worker.get(),
            live_ctx,
            verbosity,
            temperature,
            forced_language,
            &RunVadSegmentedDecode]() {
            qwen_verbose = verbosity;
            if (temperature >= 0.0f) {
                live_ctx->decode_temperature = temperature;
            }

            if (qwen_set_prompt(live_ctx, nullptr) != 0) {
                std::lock_guard<std::mutex> lock(session->mu);
                session->error = "failed to set realtime prompt";
                session->worker_done = true;
                qwen_free(live_ctx);
                return;
            }

            /* VAD-segmented decode: replaces the legacy rolling
             * qwen_transcribe_stream_live loop with a sentence-bounded
             * one.  See RunVadSegmentedDecode above for details. */
            RunVadSegmentedDecode(live_ctx, &worker_ptr->live, session, forced_language);

            {
                std::lock_guard<std::mutex> lock(session->mu);
                session->last_inference_ms = live_ctx->perf_total_ms;
                if (session->error.empty()) {
                    /* Mark the final segment as committed in display
                     * state so the UI knows the last text is
                     * finalized. */
                    ApplyStableRealtimeCommit(session->total_samples,
                                              session->stable_text,
                                              session->last_inference_ms,
                                              true, session.get());
                }
                session->finalized = true;
                session->worker_done = true;
            }
            qwen_free(live_ctx);
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

        qwen_ctx_t * live_ctx = model.CreateRealtimeClone();
        if (live_ctx == nullptr) {
            DestroyManualLiveAudio(&worker->live);
            return Status(StatusCode::kInternal, "failed to clone capture model context");
        }

        const float stream_chunk_sec = RealtimeStreamChunkSeconds(realtime_policy);
        const int stream_max_new_tokens = RealtimeStreamMaxNewTokens(realtime_policy);
        const int verbosity = model.verbosity();

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

        auto session = std::make_shared<RealtimeSession>();
        session->id = std::to_string(session_counter.fetch_add(1));
        session->model = std::move(model_id);
        session->language = std::move(language);

        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            if (realtime_sessions.size() >= kMaxRealtimeSessions) {
                return Status(StatusCode::kFailedPrecondition, "too many realtime sessions");
            }
            realtime_sessions.emplace(session->id, session);
        }

        Status status = StartRealtimeLiveWorker(session);
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
        JoinRealtimeLiveWorker(worker);

        {
            std::lock_guard<std::mutex> lock(session->mu);
            session->live_worker.reset();
            session->finalized = true;
        }
        metrics.realtime_finalizations.fetch_add(1);

        Status status = SnapshotRealtimeSessionState(session, false, snapshot);
        if (!status.ok()) {
            return status;
        }
        if (!snapshot->error.empty()) {
            return Status(StatusCode::kInternal, snapshot->error);
        }
        return OkStatus();
    };

    server.Get("/", [&](const HttpRequest &, HttpResponse & response) {
        const std::string body = LoadTextFile(ui_dir / "index.html");
        if (body.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInternal, "failed to load index.html"), 500);
            return;
        }
        response.set_content(body, "text/html; charset=utf-8");
    });
    server.Get("/app.js", [&](const HttpRequest &, HttpResponse & response) {
        const std::string body = LoadTextFile(ui_dir / "app.js");
        if (body.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInternal, "failed to load app.js"), 500);
            return;
        }
        response.set_content(body, "application/javascript; charset=utf-8");
    });
    server.Get("/live_monitor.js", [&](const HttpRequest &, HttpResponse & response) {
        const std::string body = LoadTextFile(ui_dir / "live_monitor.js");
        if (body.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInternal, "failed to load live_monitor.js"), 500);
            return;
        }
        response.set_content(body, "application/javascript; charset=utf-8");
    });
    server.Get("/style.css", [&](const HttpRequest &, HttpResponse & response) {
        const std::string body = LoadTextFile(ui_dir / "style.css");
        if (body.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInternal, "failed to load style.css"), 500);
            return;
        }
        response.set_content(body, "text/css; charset=utf-8");
    });

    server.Get("/health", [&](const HttpRequest &, HttpResponse & response) {
        SetJsonResponse(response, Json::object({{"status", "ok"}}));
    });
    server.Get("/api/health", [&](const HttpRequest &, HttpResponse & response) {
        SetJsonResponse(response, Json::object({{"status", "ok"}}));
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
        const AsrRunResult result = model.TranscribeFile(prepared.wav_path, decode);
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
            const AsrRunResult result = model.TranscribeFile(prepared.wav_path, decode);
            CleanupPreparedAudio(&prepared);

            {
                std::lock_guard<std::mutex> lock(jobs_mu);
                OfflineJob & current = jobs[job_id];
                current.updated_at = CurrentUnixSeconds();
                current.language = DetectLanguageLabel(options.language);
                current.inference_ms = result.total_ms;
                current.audio_ms = result.audio_ms;
                current.tokens = result.text_tokens;
                if (cancel_flag && cancel_flag->load()) {
                    current.state = "cancelled";
                    current.error.clear();
                    if (!result.text.empty()) {
                        current.text = result.text;
                    }
                } else if (!result.status.ok()) {
                    current.state = "failed";
                    current.error = result.status.message();
                } else {
                    current.state = "completed";
                    current.text = result.text;
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
        const AsrRunResult result = model.TranscribeFile(prepared.wav_path, decode);
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
        const AsrRunResult result = model.TranscribeFile(prepared.wav_path, decode);
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
        RealtimeSessionSnapshot session;
        const Status status = CreateRealtimeSession(served_model_id, "", &session);
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
        const std::vector<float> chunk = DecodePcm16Le(request.body);
        if (chunk.empty()) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "pcm16le body is required"), 400);
            return;
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
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        RealtimeSessionSnapshot session;
        const Status status = FinalizeRealtimeSession(session_id, &session);
        if (!status.ok()) {
            SetErrorResponse(response, status, StatusToHttpCode(status));
            return;
        }
        Json body = BuildRealtimeJson(session, true, true);
        SetJsonResponse(response, body);
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
        for (;;) {
            RealtimeSessionSnapshot snapshot;
            SnapshotRealtimeSessionState(session, false, &snapshot);
            Json body = BuildRealtimeJson(snapshot, false, true);
            if (!writer(BuildSseData(body.dump()))) {
                break;
            }
            if (snapshot.finalized) {
                writer("data: [DONE]\n\n");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
            LockLiveAudio(&capture->live_worker->live);
            const int64_t dc = capture->live_worker->live.decoded_cursor;
            UnlockLiveAudio(&capture->live_worker->live);
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
                BOOL ok = ReadFile(capture->read_handle, buffer.data(),
                    static_cast<DWORD>(buffer.size()), &bytes_read, nullptr);
                if (!ok || bytes_read == 0) {
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

                const std::vector<float> chunk = DecodePcm16Le(buffer.data(), static_cast<std::size_t>(n_read));
                if (chunk.empty()) {
                    continue;
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
            LockLiveAudio(&capture->live_worker->live);
            const int64_t dc = capture->live_worker->live.decoded_cursor;
            UnlockLiveAudio(&capture->live_worker->live);
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
    if (!ok) {
        std::fprintf(stderr, "qasr_server listen failed on %s:%d\n", config.host.c_str(), config.port);
        return 1;
    }
    return 0;
#endif
}

}  // namespace qasr
