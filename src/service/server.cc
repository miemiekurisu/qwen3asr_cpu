#include "qasr/service/server.h"
#include "qasr/service/realtime.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

#if defined(__linux__)
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
}
#include "qasr/base/http_server.h"
#include "qasr/base/json.h"
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

namespace {

namespace fs = std::filesystem;

#ifdef QASR_CPU_BACKEND_ENABLED
using Json = qasr::Json;
#endif

constexpr std::size_t kHttpWorkerQueueLimit = 64;
constexpr std::size_t kMaxRealtimeSessions = 64;

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

std::string ShellEscape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    escaped.push_back('\'');
    for (const char ch : value) {
        if (ch == '\'') {
            escaped += "'\"'\"'";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
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
    const std::string command = "command -v " + std::string(name) + " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
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

    const std::string command =
        "ffmpeg -loglevel error -nostdin -y -i " + ShellEscape(locator) +
        " -ar 16000 -ac 1 -f wav " + ShellEscape(output_path.string()) +
        " >/dev/null 2>&1";
    if (std::system(command.c_str()) != 0) {
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

#if defined(QASR_CPU_BACKEND_ENABLED) && defined(__linux__)
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
    bool use_stream_path = false;
    std::function<void(std::string_view)> token_callback;
};

void ForwardTokenPiece(const char * piece, void * userdata) {
    if (piece == nullptr || userdata == nullptr) {
        return;
    }
    auto * callback = static_cast<std::function<void(std::string_view)> *>(userdata);
    (*callback)(piece);
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
        ctx_->past_text_conditioning = 1;
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
        if (qwen_set_prompt(ctx_, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx_, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + decode.language);
            return result;
        }

        std::function<void(std::string_view)> callback = decode.token_callback;
        qwen_set_token_callback(ctx_, callback ? ForwardTokenPiece : nullptr, callback ? &callback : nullptr);

        char * raw = qwen_transcribe(ctx_, audio_path.string().c_str());
        qwen_set_token_callback(ctx_, nullptr, nullptr);
        if (raw == nullptr) {
            result.status = Status(StatusCode::kInternal, "transcription failed");
            return result;
        }

        result.text = raw;
        std::free(raw);
        result.total_ms = ctx_->perf_total_ms;
        result.audio_ms = ctx_->perf_audio_ms;
        result.text_tokens = ctx_->perf_text_tokens;
        result.encode_ms = ctx_->perf_encode_ms;
        result.decode_ms = ctx_->perf_decode_ms;
        result.status = OkStatus();
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
        if (qwen_set_prompt(ctx_, decode.prompt.empty() ? nullptr : decode.prompt.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
            return result;
        }
        if (qwen_set_force_language(ctx_, decode.language.empty() ? nullptr : decode.language.c_str()) != 0) {
            result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + decode.language);
            return result;
        }

        std::function<void(std::string_view)> callback = decode.token_callback;
        qwen_set_token_callback(ctx_, callback ? ForwardTokenPiece : nullptr, callback ? &callback : nullptr);

        char * raw = decode.use_stream_path
            ? qwen_transcribe_stream(ctx_, samples.data(), static_cast<int>(samples.size()))
            : qwen_transcribe_audio(ctx_, samples.data(), static_cast<int>(samples.size()));
        qwen_set_token_callback(ctx_, nullptr, nullptr);
        if (raw == nullptr) {
            result.status = Status(StatusCode::kInternal, decode.use_stream_path ? "stream transcription failed" : "audio transcription failed");
            return result;
        }

        result.text = raw;
        std::free(raw);
        result.total_ms = ctx_->perf_total_ms;
        result.audio_ms = ctx_->perf_audio_ms;
        result.text_tokens = ctx_->perf_text_tokens;
        result.encode_ms = ctx_->perf_encode_ms;
        result.decode_ms = ctx_->perf_decode_ms;
        result.status = OkStatus();
        return result;
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
    body["created_at"] = job.created_at;
    body["updated_at"] = job.updated_at;
    return body;
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
    std::string id;
    std::vector<float> samples;
    std::size_t total_samples = 0;
    std::size_t retained_sample_offset = 0;
    RealtimeTextState text_state;
    std::string text;
    std::string stable_text;
    std::string partial_text;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
};

struct ServerMetrics {
    std::atomic<std::uint64_t> offline_requests{0};
    std::atomic<std::uint64_t> async_jobs_submitted{0};
    std::atomic<std::uint64_t> chat_requests{0};
    std::atomic<std::uint64_t> realtime_sessions_started{0};
    std::atomic<std::uint64_t> realtime_decode_runs{0};
    std::atomic<std::uint64_t> realtime_finalizations{0};
    std::atomic<std::uint64_t> host_capture_sessions_started{0};
};

#if defined(__linux__)
struct HostCaptureSession {
    std::string id;
    std::string backend;
    std::string device;
    std::vector<float> samples;
    std::size_t total_samples = 0;
    std::size_t retained_sample_offset = 0;
    RealtimeTextState text_state;
    std::string text;
    std::string stable_text;
    std::string partial_text;
    std::string error;
    double last_inference_ms = 0.0;
    bool last_decode_ran = false;
    bool active = true;
    bool stop_requested = false;
    pid_t child_pid = -1;
    int read_fd = -1;
    std::thread reader;
    std::mutex mu;
};

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
    if (backend == "auto") {
        if (have_arecord) {
            backend = "arecord";
        } else if (have_parec) {
            backend = "parec";
        } else {
            return Status(StatusCode::kFailedPrecondition, "neither arecord nor parec is available");
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

    return Status(StatusCode::kInvalidArgument, "unsupported capture backend: " + backend);
}

void StopHostCaptureSession(const std::shared_ptr<HostCaptureSession> & capture) {
    if (!capture) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(capture->mu);
        capture->stop_requested = true;
    }

    if (capture->child_pid > 0) {
        kill(capture->child_pid, SIGTERM);
    }
    if (capture->read_fd >= 0) {
        close(capture->read_fd);
        capture->read_fd = -1;
    }
    if (capture->reader.joinable()) {
        capture->reader.join();
    }
    if (capture->child_pid > 0) {
        int status = 0;
        waitpid(capture->child_pid, &status, 0);
        capture->child_pid = -1;
    }

    std::lock_guard<std::mutex> lock(capture->mu);
    capture->active = false;
}
#endif

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
    SessionLike * session) {
    if (session == nullptr) {
        return;
    }
    session->stable_text = update.stable_text;
    session->partial_text = update.partial_text;
    session->text = update.text;
    session->last_inference_ms = inference_ms;
    session->last_decode_ran = decoded;
}

template <typename SessionLike>
Json BuildRealtimeJson(
    const SessionLike & session,
    bool finalized,
    bool supported) {
    Json body;
    body["session_id"] = session.id;
    body["sample_count"] = session.total_samples;
    body["retained_sample_count"] = session.samples.size();
    body["retained_sample_offset"] = session.retained_sample_offset;
    body["decoded"] = session.last_decode_ran;
    body["finalized"] = finalized;
    body["supported"] = supported;
    body["stable_text"] = session.stable_text;
    body["partial_text"] = session.partial_text;
    body["text"] = session.text;
    body["inference_ms"] = session.last_inference_ms;
    return body;
}

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
    usage += "  --verbosity <n>\n";
    usage += "  -h, --help\n";
    return usage;
}

int RunServer(const ServerConfig & config) {
#ifndef QASR_CPU_BACKEND_ENABLED
    (void)config;
    return 1;
#else
    const Status config_status = ValidateServerConfig(config);
    if (!config_status.ok()) {
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
    std::unordered_map<std::string, RealtimeSession> realtime_sessions;
    std::mutex realtime_mu;
    std::unordered_map<std::string, OfflineJob> jobs;
    std::mutex jobs_mu;
#if defined(__linux__)
    std::shared_ptr<HostCaptureSession> host_capture;
    std::mutex host_capture_mu;
#endif

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
    server.Get("/v1/models", [&](const HttpRequest &, HttpResponse & response) {
        Json payload;
        payload["object"] = "list";
        payload["data"] = Json::array({
            Json::object({
                {"id", served_model_id},
                {"object", "model"},
                {"created", 0},
                {"owned_by", "qwen-asr-provider"},
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
#if defined(__linux__)
        {
            std::lock_guard<std::mutex> lock(host_capture_mu);
            host_capture_active = static_cast<bool>(host_capture && host_capture->active);
        }
#endif
        const auto uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - server_start).count();
        Json payload;
        payload["uptime_ms"] = uptime_ms;
        payload["offline_requests"] = metrics.offline_requests.load();
        payload["async_jobs_submitted"] = metrics.async_jobs_submitted.load();
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
                Status(StatusCode::kFailedPrecondition, "use /v1/chat/completions with stream=true or /api/realtime/* for streaming"),
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
        job.created_at = CurrentUnixSeconds();
        job.updated_at = job.created_at;
        {
            std::lock_guard<std::mutex> lock(jobs_mu);
            jobs.emplace(job.id, job);
        }

        const std::string job_id = job.id;
        std::thread([prepared, options, job_id, &jobs, &jobs_mu, &model]() mutable {
            {
                std::lock_guard<std::mutex> lock(jobs_mu);
                OfflineJob & current = jobs[job_id];
                current.state = "running";
                current.updated_at = CurrentUnixSeconds();
            }

            ModelDecodeOptions decode;
            decode.prompt = options.prompt;
            decode.language = options.language;
            const AsrRunResult result = model.TranscribeFile(prepared.wav_path, decode);
            CleanupPreparedAudio(&prepared);

            std::lock_guard<std::mutex> lock(jobs_mu);
            OfflineJob & current = jobs[job_id];
            current.updated_at = CurrentUnixSeconds();
            current.language = DetectLanguageLabel(options.language);
            if (!result.status.ok()) {
                current.state = "failed";
                current.error = result.status.message();
                return;
            }
            current.state = "completed";
            current.text = result.text;
            current.inference_ms = result.total_ms;
            current.audio_ms = result.audio_ms;
            current.tokens = result.text_tokens;
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

    server.Post("/api/realtime/start", [&](const HttpRequest &, HttpResponse & response) {
        RealtimeSession session;
        session.id = std::to_string(session_counter.fetch_add(1));
        metrics.realtime_sessions_started.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            if (realtime_sessions.size() >= kMaxRealtimeSessions) {
                SetErrorResponse(response, Status(StatusCode::kFailedPrecondition, "too many realtime sessions"), 429);
                return;
            }
            realtime_sessions.emplace(session.id, session);
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

        std::vector<float> samples;
        std::size_t total_samples = 0;
        Json body;
        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            auto it = realtime_sessions.find(session_id);
            if (it == realtime_sessions.end()) {
                SetErrorResponse(response, Status(StatusCode::kNotFound, "session not found"), 404);
                return;
            }
            AppendRealtimeSamples(realtime_policy, chunk, &it->second);
            if (!RealtimeShouldDecode(
                    realtime_policy,
                    it->second.total_samples,
                    it->second.text_state.last_decode_samples,
                    false)) {
                it->second.last_decode_ran = false;
                body = BuildRealtimeJson(it->second, false, true);
                SetJsonResponse(response, body);
                return;
            }
            samples = it->second.samples;
            total_samples = it->second.total_samples;
        }

        ModelDecodeOptions decode;
        decode.use_stream_path = true;
        const AsrRunResult result = model.TranscribeRealtime(samples, decode);
        if (!result.status.ok()) {
            SetErrorResponse(response, result.status, StatusToHttpCode(result.status));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            auto it = realtime_sessions.find(session_id);
            if (it == realtime_sessions.end()) {
                SetErrorResponse(response, Status(StatusCode::kNotFound, "session not found"), 404);
                return;
            }
            RealtimeTextUpdate update;
            const Status update_status = AdvanceRealtimeTextState(
                realtime_policy,
                total_samples,
                result.text,
                false,
                &it->second.text_state,
                &update);
            if (!update_status.ok()) {
                SetErrorResponse(response, update_status, StatusToHttpCode(update_status));
                return;
            }
            ApplyRealtimeUpdate(update, result.total_ms, true, &it->second);
            body = BuildRealtimeJson(it->second, false, true);
        }
        metrics.realtime_decode_runs.fetch_add(1);
        SetJsonResponse(response, body);
    });

    server.Post("/api/realtime/stop", [&](const HttpRequest & request, HttpResponse & response) {
        if (!request.has_param("session_id")) {
            SetErrorResponse(response, Status(StatusCode::kInvalidArgument, "session_id is required"), 400);
            return;
        }
        const std::string session_id = request.get_param_value("session_id");
        RealtimeSession session;
        {
            std::lock_guard<std::mutex> lock(realtime_mu);
            auto it = realtime_sessions.find(session_id);
            if (it == realtime_sessions.end()) {
                SetErrorResponse(response, Status(StatusCode::kNotFound, "session not found"), 404);
                return;
            }
            session = it->second;
            realtime_sessions.erase(it);
        }

        if (!session.samples.empty()) {
            ModelDecodeOptions decode;
            decode.use_stream_path = false;
            const AsrRunResult result = model.TranscribeRealtime(session.samples, decode);
            if (result.status.ok()) {
                const std::string latest_text =
                    (session.retained_sample_offset == 0U || session.text.empty()) ? result.text : session.text;
                RealtimeTextUpdate update;
                const Status update_status = AdvanceRealtimeTextState(
                    realtime_policy,
                    session.total_samples,
                    latest_text,
                    true,
                    &session.text_state,
                    &update);
                if (update_status.ok()) {
                    ApplyRealtimeUpdate(update, result.total_ms, true, &session);
                }
            } else {
                RealtimeTextUpdate update;
                const Status update_status = AdvanceRealtimeTextState(
                    realtime_policy,
                    session.total_samples,
                    session.text.empty() ? session.text_state.last_text : session.text,
                    true,
                    &session.text_state,
                    &update);
                if (update_status.ok()) {
                    ApplyRealtimeUpdate(update, session.last_inference_ms, false, &session);
                }
            }
        }

        metrics.realtime_finalizations.fetch_add(1);
        Json body = BuildRealtimeJson(session, true, true);
        SetJsonResponse(response, body);
    });

#if defined(__linux__)
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

        const Status spawn_status = SpawnCaptureProcess(argv, &capture->child_pid, &capture->read_fd);
        if (!spawn_status.ok()) {
            SetErrorResponse(response, spawn_status, StatusToHttpCode(spawn_status));
            return;
        }
        metrics.host_capture_sessions_started.fetch_add(1);

        capture->reader = std::thread([capture, &model, realtime_policy]() {
            std::vector<char> buffer(6400);
            while (true) {
                const ssize_t n_read = read(capture->read_fd, buffer.data(), buffer.size());
                if (n_read <= 0) {
                    break;
                }

                const std::vector<float> chunk = DecodePcm16Le(buffer.data(), static_cast<std::size_t>(n_read));
                if (chunk.empty()) {
                    continue;
                }

                std::vector<float> samples;
                std::size_t total_samples = 0;
                {
                    std::lock_guard<std::mutex> lock(capture->mu);
                    AppendRealtimeSamples(realtime_policy, chunk, capture.get());
                    if (!RealtimeShouldDecode(
                            realtime_policy,
                            capture->total_samples,
                            capture->text_state.last_decode_samples,
                            false)) {
                        capture->last_decode_ran = false;
                        continue;
                    }
                    samples = capture->samples;
                    total_samples = capture->total_samples;
                }

                ModelDecodeOptions decode;
                decode.use_stream_path = true;
                const AsrRunResult result = model.TranscribeRealtime(samples, decode);
                std::lock_guard<std::mutex> lock(capture->mu);
                if (!result.status.ok()) {
                    capture->error = result.status.message();
                    break;
                }
                RealtimeTextUpdate update;
                Status update_status = AdvanceRealtimeTextState(
                    realtime_policy,
                    total_samples,
                    result.text,
                    false,
                    &capture->text_state,
                    &update);
                if (!update_status.ok()) {
                    capture->error = update_status.message();
                    break;
                }
                ApplyRealtimeUpdate(update, result.total_ms, true, capture.get());
            }

            std::lock_guard<std::mutex> lock(capture->mu);
            capture->active = false;
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

        {
            std::lock_guard<std::mutex> lock(capture->mu);
            if (!capture->samples.empty()) {
                ModelDecodeOptions decode;
                decode.use_stream_path = false;
                const AsrRunResult result = model.TranscribeRealtime(capture->samples, decode);
                if (result.status.ok()) {
                    const std::string latest_text =
                        (capture->retained_sample_offset == 0U || capture->text.empty()) ? result.text : capture->text;
                    RealtimeTextUpdate update;
                    const Status update_status = AdvanceRealtimeTextState(
                        realtime_policy,
                        capture->total_samples,
                        latest_text,
                        true,
                        &capture->text_state,
                        &update);
                    if (update_status.ok()) {
                        ApplyRealtimeUpdate(update, result.total_ms, true, capture.get());
                    }
                } else if (capture->error.empty()) {
                    capture->error = result.status.message();
                }
            } else {
                RealtimeTextUpdate update;
                const Status update_status = AdvanceRealtimeTextState(
                    realtime_policy,
                    capture->total_samples,
                    capture->text.empty() ? capture->text_state.last_text : capture->text,
                    true,
                    &capture->text_state,
                    &update);
                if (update_status.ok()) {
                    ApplyRealtimeUpdate(update, capture->last_inference_ms, false, capture.get());
                }
            }
        }

        metrics.realtime_finalizations.fetch_add(1);
        std::lock_guard<std::mutex> lock(capture->mu);
        Json body = BuildRealtimeJson(*capture, true, true);
        body["capture_id"] = capture->id;
        body["backend"] = capture->backend;
        body["device"] = capture->device;
        body["error"] = capture->error;
        SetJsonResponse(response, body);
    });
#else
    server.Get("/api/capture/status", [&](const HttpRequest &, HttpResponse & response) {
        SetJsonResponse(response, Json::object({{"active", false}, {"supported", false}}));
    });
    server.Post("/api/capture/start", [&](const HttpRequest &, HttpResponse & response) {
        SetErrorResponse(response, Status(StatusCode::kUnimplemented, "host capture backend is Linux-only"), 501);
    });
    server.Post("/api/capture/stop", [&](const HttpRequest &, HttpResponse & response) {
        SetErrorResponse(response, Status(StatusCode::kUnimplemented, "host capture backend is Linux-only"), 501);
    });
#endif

    std::fprintf(stderr, "qasr_server listening on %s:%d\n", config.host.c_str(), config.port);
    const bool ok = server.listen(config.host, config.port);
    if (!ok) {
        std::fprintf(stderr, "qasr_server listen failed on %s:%d\n", config.host.c_str(), config.port);
        return 1;
    }
    return 0;
#endif
}

}  // namespace qasr
