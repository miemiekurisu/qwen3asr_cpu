#include "qasr/runtime/model_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <string>

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

std::set<std::string> ExtractIndexedSafetensors(const std::string & json_text) {
    const std::regex pattern(R"(model-[^"]+\.safetensors)");
    std::set<std::string> files;
    for (std::sregex_iterator it(json_text.begin(), json_text.end(), pattern), end; it != end; ++it) {
        files.insert(it->str());
    }
    return files;
}

#ifdef QASR_CPU_BACKEND_ENABLED
void WriteTokenToStdout(const char * piece, void * userdata) {
    (void)userdata;
    if (piece == nullptr) {
        return;
    }
    std::fputs(piece, stdout);
    std::fflush(stdout);
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
    if (options.verbosity < 0) {
        return Status(StatusCode::kInvalidArgument, "verbosity must be >= 0");
    }
    return OkStatus();
}

AsrRunResult RunAsr(const AsrRunOptions & options) {
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

    ctx->stream_max_new_tokens = static_cast<int>(options.stream_max_new_tokens);

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

    qwen_set_token_callback(ctx, options.emit_tokens ? WriteTokenToStdout : nullptr, nullptr);

    char * raw_text = nullptr;
    if (options.stream) {
        int n_samples = 0;
        float * samples = qwen_load_wav(options.audio_path.c_str(), &n_samples);
        if (samples == nullptr) {
            qwen_free(ctx);
            result.status = Status(StatusCode::kInternal, "failed to load wav input");
            return result;
        }
        raw_text = qwen_transcribe_stream(ctx, samples, n_samples);
        std::free(samples);
    } else {
        raw_text = qwen_transcribe(ctx, options.audio_path.c_str());
    }

    if (raw_text == nullptr) {
        qwen_free(ctx);
        result.status = Status(StatusCode::kInternal, "transcription failed");
        return result;
    }

    result.text = raw_text;
    std::free(raw_text);
    result.total_ms = ctx->perf_total_ms;
    result.text_tokens = ctx->perf_text_tokens;
    result.audio_ms = ctx->perf_audio_ms;
    result.encode_ms = ctx->perf_encode_ms;
    result.decode_ms = ctx->perf_decode_ms;
    qwen_free(ctx);
    result.status = OkStatus();
    return result;
#endif
}

}  // namespace qasr
