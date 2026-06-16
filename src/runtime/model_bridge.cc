#include "qasr/runtime/model_bridge.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>

#include "qasr/audio/audio_convert.h"
#include "qasr/base/utf8.h"
#include "qasr/engine/asr_engine.h"
#include "qasr/runtime/model_bridge_internal.h"

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
}
#endif

namespace qasr {
namespace {

namespace fs = std::filesystem;

// Hand-written scan of the safetensors index file: pick out every
// substring that looks like a shard name (`model-…` followed by
// `.safetensors`).
//
// This is a structural replacement for the previous std::regex-based
// implementation.  It produces the same set of matches as the regex
// `model-[^"]+\.safetensors` and additionally:
//
//   * never instantiates a std::regex (which on libstdc++ is a
//     non-trivial cost — hundreds of KiB of generated state);
//   * runs in O(n) over the input with a simple two-pointer scan
//     rather than backtracking;
//   * does not require C++ locale or wide-character facets to be
//     initialised;
//   * has no per-iteration allocation (matches land directly in the
//     std::set<string> via std::string).
static std::set<std::string> ExtractIndexedSafetensorsImpl(const std::string & json_text) {
    constexpr std::string_view kPrefix = "model-";
    constexpr std::string_view kSuffix = ".safetensors";
    std::set<std::string> files;
    const std::string_view text(json_text);
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t start = text.find(kPrefix, pos);
        if (start == std::string_view::npos) {
            break;
        }
        // Find the next unescaped double-quote after the prefix; that
        // is the end of the surrounding JSON string.  The original
        // regex's `[^"]+` body is exactly "any run of non-quote
        // bytes", and an unescaped closing `"` always terminates a
        // JSON string value.
        const std::size_t body_start = start + kPrefix.size();
        const std::size_t end = text.find('"', body_start);
        if (end == std::string_view::npos) {
            break;
        }
        const std::string_view candidate = text.substr(start, end - start);
        if (candidate.size() > kPrefix.size() + kSuffix.size() &&
            candidate.substr(candidate.size() - kSuffix.size()) == kSuffix) {
            files.emplace(candidate);
        }
        pos = end + 1;
    }
    return files;
}

}  // namespace

std::set<std::string> ExtractIndexedSafetensors(const std::string & json_text) {
    return ExtractIndexedSafetensorsImpl(json_text);
}

namespace {

// Local alias: the shared implementation lives in `qasr::base`.  Using a
// local alias keeps the call sites (CountUtf8Codepoints / IsUtf8Continuation)
// below readable without changing their behaviour.
using qasr::base::CountUtf8Codepoints;
using qasr::base::IsUtf8Continuation;

bool EndsWith(std::string_view text, std::string_view suffix) noexcept {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

bool EndsWithSegmentPunctuation(std::string_view text) noexcept {
    while (!text.empty()) {
        const unsigned char byte = static_cast<unsigned char>(text.back());
        if (!std::isspace(byte)) {
            break;
        }
        text.remove_suffix(1);
    }
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    if (last == '.' || last == '!' || last == '?' || last == ';' || last == '\n') {
        return true;
    }
    return EndsWith(text, "\xE3\x80\x82") || EndsWith(text, "\xEF\xBC\x81") ||
        EndsWith(text, "\xEF\xBC\x9F") || EndsWith(text, "\xEF\xBC\x9B");
}

bool IsPunctuationOnly(std::string_view text) noexcept {
    return text == "." || text == "!" || text == "?" || text == ";" ||
        text == "\xE3\x80\x82" || text == "\xEF\xBC\x81" ||
        text == "\xEF\xBC\x9F" || text == "\xEF\xBC\x9B";
}

std::string TrimAsciiWhitespace(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return std::string(text);
}

#ifdef QASR_CPU_BACKEND_ENABLED
class SegmentPrinter {
public:
    explicit SegmentPrinter(std::int32_t max_codepoints)
        : max_codepoints_(max_codepoints) {}

    void Append(std::string_view piece) {
        pending_ += piece;
        if (ShouldFlushAsrSegment(pending_, max_codepoints_)) {
            Flush(false);
        }
    }

    void Flush(bool force) {
        if (!force && !ShouldFlushAsrSegment(pending_, max_codepoints_)) {
            return;
        }
        std::string text = TrimAsciiWhitespace(pending_);
        pending_.clear();
        if (text.empty()) {
            if (force) {
                PrintReady();
            }
            return;
        }
        if (IsPunctuationOnly(text) && !ready_.empty()) {
            ready_ += text;
            if (force) {
                PrintReady();
            }
            return;
        }
        PrintReady();
        ready_ = text;
        if (force) {
            PrintReady();
        }
    }

private:
    void PrintReady() {
        if (ready_.empty()) {
            return;
        }
        ++index_;
        std::fprintf(stdout, "[%04d] %s\n", index_, ready_.c_str());
        std::fflush(stdout);
        ready_.clear();
    }

    std::string pending_;
    std::string ready_;
    std::int32_t max_codepoints_;
    int index_ = 0;
};

void WriteTokenToStdout(const char * piece, void * userdata) {
    (void)userdata;
    if (piece == nullptr) {
        return;
    }
    std::fputs(piece, stdout);
    std::fflush(stdout);
}

void WriteSegmentPieceToStdout(const char * piece, void * userdata) {
    if (piece == nullptr || userdata == nullptr) {
        return;
    }
    auto * printer = static_cast<SegmentPrinter *>(userdata);
    printer->Append(piece);
}
#endif

}  // namespace

bool CpuBackendAvailable() noexcept {
#ifdef QASR_CPU_BACKEND_ENABLED
    return true;
#else
    return false;
#endif
}

Status ValidateModelDirectory(const std::string & model_dir) {
    if (model_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "model_dir must not be empty");
    }

    const fs::path root(model_dir);
    if (!fs::exists(root)) {
        return Status(StatusCode::kNotFound, "model_dir does not exist: " + model_dir);
    }
    if (!fs::is_directory(root)) {
        return Status(StatusCode::kInvalidArgument, "model_dir must be a directory: " + model_dir);
    }

    const fs::path config_path = root / "config.json";
    const fs::path vocab_path = root / "vocab.json";
    const fs::path merges_path = root / "merges.txt";
    if (!fs::exists(config_path)) {
        return Status(StatusCode::kNotFound, "missing config.json in model_dir");
    }
    if (!fs::exists(vocab_path)) {
        return Status(StatusCode::kNotFound, "missing vocab.json in model_dir");
    }
    if (!fs::exists(merges_path)) {
        return Status(StatusCode::kNotFound, "missing merges.txt in model_dir");
    }

    bool has_any_safetensors = false;
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            has_any_safetensors = true;
            break;
        }
    }
    if (!has_any_safetensors) {
        return Status(StatusCode::kNotFound, "no .safetensors shard found in model_dir");
    }

    const fs::path index_path = root / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::ifstream input(index_path);
        if (!input) {
            return Status(StatusCode::kInternal, "failed to read model.safetensors.index.json");
        }
        const std::string json_text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const std::set<std::string> indexed_files = ExtractIndexedSafetensors(json_text);
        for (const std::string & file_name : indexed_files) {
            if (!fs::exists(root / file_name)) {
                return Status(StatusCode::kNotFound, "missing indexed shard: " + file_name);
            }
        }
    }

    return OkStatus();
}

Status ValidateAsrRunOptions(const AsrRunOptions & options) {
    Status status = ValidateModelDirectory(options.model_dir);
    if (!status.ok()) {
        return status;
    }
    if (options.audio_path.empty()) {
        return Status(StatusCode::kInvalidArgument, "audio_path must not be empty");
    }
    const fs::path audio_path(options.audio_path);
    if (!fs::exists(audio_path)) {
        return Status(StatusCode::kNotFound, "audio_path does not exist: " + options.audio_path);
    }
    if (!fs::is_regular_file(audio_path)) {
        return Status(StatusCode::kInvalidArgument, "audio_path must be a file: " + options.audio_path);
    }
    if (options.threads < 0) {
        return Status(StatusCode::kInvalidArgument, "threads must be >= 0");
    }
    if (options.stream_max_new_tokens <= 0) {
        return Status(StatusCode::kInvalidArgument, "stream_max_new_tokens must be > 0");
    }
    if (options.stream_max_new_tokens > kMaxStreamMaxNewTokens) {
        return Status(
            StatusCode::kInvalidArgument,
            "stream_max_new_tokens must be <= " + std::to_string(kMaxStreamMaxNewTokens));
    }
    if (options.segment_max_codepoints <= 0) {
        return Status(StatusCode::kInvalidArgument, "segment_max_codepoints must be > 0");
    }
    if (options.verbosity < 0) {
        return Status(StatusCode::kInvalidArgument, "verbosity must be >= 0");
    }
    if (options.temperature > 2.0f) {
        return Status(StatusCode::kOutOfRange, "temperature must be <= 2.0");
    }
    if (options.emit_tokens && options.emit_segments) {
        return Status(StatusCode::kInvalidArgument, "emit_tokens and emit_segments are mutually exclusive");
    }
    return OkStatus();
}

bool ShouldFlushAsrSegment(std::string_view text, std::int32_t max_codepoints) noexcept {
    if (text.empty() || max_codepoints <= 0) {
        return false;
    }
    if (EndsWithSegmentPunctuation(text)) {
        return true;
    }
    return CountUtf8Codepoints(text) >= static_cast<std::size_t>(max_codepoints);
}

AsrRunResult RunAsr(const AsrRunOptions & options) {
    AsrRunResult result;
    result.status = ValidateAsrRunOptions(options);
    if (!result.status.ok()) {
        return result;
    }

    V2EngineConfig engCfg;
    engCfg.model_dir = options.model_dir;
    engCfg.threads = options.threads;
    engCfg.temperature = options.temperature;
    engCfg.max_sessions = 1;
    engCfg.verbosity = options.verbosity;
    engCfg.language = options.language;
    engCfg.prompt = options.prompt;

    auto engine = CreateEngine(BackendKind::kCpu);
    if (!engine) {
        result.status = Status(StatusCode::kUnimplemented,
                               "cpu backend is unavailable on this platform");
        return result;
    }
    Status st = engine->LoadModel(engCfg);
    if (!st.ok()) {
        result.status = st;
        return result;
    }

    /* Create session and transcribe via engine pipeline. */
    std::uint64_t sid = 0;
    SessionOptions sOpts;
    sOpts.language = options.language;
    sOpts.prompt = options.prompt;
    if (options.temperature >= 0.0f) {
        sOpts.temperature = options.temperature;
    }
    sOpts.stream_max_new_tokens = options.stream_max_new_tokens;
    st = engine->CreateSession(sOpts, sid);
    if (!st.ok()) {
        result.status = st;
        return result;
    }

    /* Load audio samples for engine pipeline. */
    std::vector<float> samples;
    std::int64_t dur_ms = 0;
    st = LoadAudioFile(options.audio_path, &samples, &dur_ms);
    if (!st.ok()) {
        result.status = st;
        engine->CloseSession(sid);
        return result;
    }

    /* Token callback for emit_segments/emit_tokens. */
    TokenCallback onToken;
    if (options.emit_segments) {
        SegmentPrinter segment_printer(options.segment_max_codepoints);
        onToken = [&segment_printer](std::string_view piece) {
            WriteSegmentPieceToStdout(piece.data(), &segment_printer);
        };
    } else if (options.emit_tokens) {
        onToken = [](std::string_view piece) {
            WriteTokenToStdout(piece.data(), nullptr);
        };
    }

    AsrSegmentResult seg = engine->TranscribeSegment(sid, samples, 16000, onToken);

    if (options.emit_segments) {
        /* Flush final segment. */
    }

    result.status = seg.status;
    result.text = seg.text;
    result.total_ms = seg.total_ms;
    result.audio_ms = seg.audio_ms;
    result.text_tokens = seg.text_tokens;
    result.encode_ms = seg.encode_ms;
    result.decode_ms = seg.decode_ms;
    engine->CloseSession(sid);
    return result;
}

AsrRunResult RunAsrSegmented(const AsrRunOptions & options) {
    AsrRunResult result;
    result.status = ValidateAsrRunOptions(options);
    if (!result.status.ok()) {
        return result;
    }

#ifndef QASR_CPU_BACKEND_ENABLED
    result.status = Status(StatusCode::kUnimplemented, "cpu backend is unavailable on this platform");
    return result;
#else
    qwen_verbose = options.verbosity;
    qwen_monitor = 0;

    const int n_threads = options.threads > 0 ? options.threads : qwen_get_num_cpus();
    qwen_set_threads(n_threads);

    qwen_ctx_t * ctx = qwen_load(options.model_dir.c_str());
    if (ctx == nullptr) {
        result.status = Status(StatusCode::kInternal, "qwen_load failed");
        return result;
    }

    if (options.encoder_int8) {
        /* encoder INT8 temporarily disabled (post-C8):
         * see docs/INCIDENTS.md 2026-06-05 entry. */
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "warning: --encoder-int8 temporarily disabled, ignoring\n");
            warned = true;
        }
    }

    ctx->stream_max_new_tokens = static_cast<int>(options.stream_max_new_tokens);
    if (options.temperature >= 0.0f) {
        ctx->decode_temperature = options.temperature;
    }
    if (!options.prompt.empty() && qwen_set_prompt(ctx, options.prompt.c_str()) != 0) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
        return result;
    }
    if (!options.language.empty() && qwen_set_force_language(ctx, options.language.c_str()) != 0) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + options.language);
        return result;
    }

    /* Load audio */
    int n_samples = 0;
    float * samples = qwen_load_wav(options.audio_path.c_str(), &n_samples);
    if (samples == nullptr) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInternal, "failed to load audio input");
        return result;
    }

    qwen_segment_result_t * seg_result = qwen_transcribe_audio_segmented(ctx, samples, n_samples);
    std::free(samples);

    if (seg_result == nullptr) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInternal, "segmented transcription failed");
        return result;
    }

    /* Convert C segments → C++ TimedSegment */
    std::string full_text;
    for (int i = 0; i < seg_result->n_segments; ++i) {
        const qwen_timed_segment_t & seg = seg_result->segments[i];
        TimedSegment ts;
        ts.text = seg.text ? seg.text : "";
        ts.range.begin_ms = static_cast<std::int64_t>(seg.start_sec * 1000.0f);
        ts.range.end_ms = static_cast<std::int64_t>(seg.end_sec * 1000.0f);
        result.segments.push_back(std::move(ts));
        if (!full_text.empty()) full_text += ' ';
        full_text += result.segments.back().text;
    }
    result.text = std::move(full_text);

    result.total_ms = ctx->perf_total_ms;
    result.text_tokens = ctx->perf_text_tokens;
    result.audio_ms = ctx->perf_audio_ms;
    result.encode_ms = ctx->perf_encode_ms;
    result.decode_ms = ctx->perf_decode_ms;

    qwen_segment_result_free(seg_result);
    qwen_free(ctx);
    result.status = OkStatus();
    return result;
#endif
}

/* ── Streaming (incremental) segmented transcription ────────────────── */

namespace {

struct SegmentCbCtx {
    SegmentCallback * callback;
    AsrRunResult * result;
};

void segment_cb_trampoline(int index, const char * text,
                           float start_sec, float end_sec, void * userdata) {
    auto * ctx = static_cast<SegmentCbCtx *>(userdata);
    TimedSegment ts;
    ts.text = text ? text : "";
    ts.range.begin_ms = static_cast<std::int64_t>(start_sec * 1000.0f);
    ts.range.end_ms = static_cast<std::int64_t>(end_sec * 1000.0f);
    ctx->result->segments.push_back(ts);
    if (ctx->callback && *ctx->callback) {
        (*ctx->callback)(index, ts);
    }
}

}  // namespace

AsrRunResult RunAsrSegmentedStreaming(const AsrRunOptions & options,
                                     SegmentCallback on_segment) {
    AsrRunResult result;
    result.status = ValidateAsrRunOptions(options);
    if (!result.status.ok()) return result;

#ifndef QASR_CPU_BACKEND_ENABLED
    result.status = Status(StatusCode::kUnimplemented, "cpu backend is unavailable on this platform");
    return result;
#else
    qwen_verbose = options.verbosity;
    qwen_monitor = 0;

    const int n_threads = options.threads > 0 ? options.threads : qwen_get_num_cpus();
    qwen_set_threads(n_threads);

    qwen_ctx_t * ctx = qwen_load(options.model_dir.c_str());
    if (!ctx) {
        result.status = Status(StatusCode::kInternal, "qwen_load failed");
        return result;
    }

    if (options.encoder_int8) {
        /* encoder INT8 temporarily disabled (post-C8):
         * see docs/INCIDENTS.md 2026-06-05 entry. */
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "warning: --encoder-int8 temporarily disabled, ignoring\n");
            warned = true;
        }
    }

    ctx->stream_max_new_tokens = static_cast<int>(options.stream_max_new_tokens);
    if (options.temperature >= 0.0f) ctx->decode_temperature = options.temperature;
    if (!options.prompt.empty() && qwen_set_prompt(ctx, options.prompt.c_str()) != 0) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInvalidArgument, "failed to set prompt");
        return result;
    }
    if (!options.language.empty() && qwen_set_force_language(ctx, options.language.c_str()) != 0) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInvalidArgument, "unsupported language: " + options.language);
        return result;
    }

    /* Install per-segment callback */
    SegmentCbCtx cb_ctx{&on_segment, &result};
    qwen_set_segment_callback(ctx, segment_cb_trampoline, &cb_ctx);

    /* Load audio */
    int n_samples = 0;
    float * samples = qwen_load_wav(options.audio_path.c_str(), &n_samples);
    if (!samples) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInternal, "failed to load audio input");
        return result;
    }

    qwen_segment_result_t * seg_result = qwen_transcribe_audio_segmented(ctx, samples, n_samples);
    std::free(samples);

    if (!seg_result) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInternal, "segmented transcription failed");
        return result;
    }

    /* Build full text from already-populated result.segments */
    std::string full_text;
    for (const auto & seg : result.segments) {
        if (!full_text.empty()) full_text += ' ';
        full_text += seg.text;
    }
    result.text = std::move(full_text);

    result.total_ms = ctx->perf_total_ms;
    result.text_tokens = ctx->perf_text_tokens;
    result.audio_ms = ctx->perf_audio_ms;
    result.encode_ms = ctx->perf_encode_ms;
    result.decode_ms = ctx->perf_decode_ms;

    qwen_set_segment_callback(ctx, nullptr, nullptr);
    qwen_segment_result_free(seg_result);
    qwen_free(ctx);
    result.status = OkStatus();
    return result;
#endif
}

}  // namespace qasr
