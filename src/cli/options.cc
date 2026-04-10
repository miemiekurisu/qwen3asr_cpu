#include "qasr/cli/options.h"

#include <charconv>
#include <string>

namespace qasr {
namespace {

Status ParseInt32Option(std::string_view text, const char * field_name, std::int32_t * value) {
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

}  // namespace

Status ParseCliArguments(int argc, const char * const argv[], CliOptions * options) {
    if (options == nullptr) {
        return Status(StatusCode::kInvalidArgument, "options must not be null");
    }
    if (argc <= 0 || argv == nullptr || argv[0] == nullptr) {
        return Status(StatusCode::kInvalidArgument, "argv must contain program name");
    }

    *options = CliOptions{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "-h" || arg == "--help") {
            options->show_help = true;
            continue;
        }
        if (arg == "--model-dir") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--model-dir", &value);
            if (!status.ok()) {
                return status;
            }
            options->asr.model_dir = value;
            ++index;
            continue;
        }
        if (arg == "--audio") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--audio", &value);
            if (!status.ok()) {
                return status;
            }
            options->asr.audio_path = value;
            ++index;
            continue;
        }
        if (arg == "--threads") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--threads", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Option(value, "threads", &options->asr.threads);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--stream-max-new-tokens") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--stream-max-new-tokens", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Option(value, "stream_max_new_tokens", &options->asr.stream_max_new_tokens);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--segment-max-codepoints") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--segment-max-codepoints", &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt32Option(value, "segment_max_codepoints", &options->asr.segment_max_codepoints);
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
            status = ParseInt32Option(value, "verbosity", &options->asr.verbosity);
            if (!status.ok()) {
                return status;
            }
            ++index;
            continue;
        }
        if (arg == "--prompt") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--prompt", &value);
            if (!status.ok()) {
                return status;
            }
            options->asr.prompt = value;
            ++index;
            continue;
        }
        if (arg == "--language") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--language", &value);
            if (!status.ok()) {
                return status;
            }
            options->asr.language = value;
            ++index;
            continue;
        }
        if (arg == "--stream") {
            options->asr.stream = true;
            continue;
        }
        if (arg == "--emit-tokens") {
            options->asr.emit_tokens = true;
            continue;
        }
        if (arg == "--emit-segments") {
            options->asr.emit_segments = true;
            continue;
        }
        return Status(StatusCode::kInvalidArgument, "unknown argument: " + std::string(arg));
    }

    if (options->show_help) {
        return OkStatus();
    }
    if (options->asr.model_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "--model-dir is required");
    }
    if (options->asr.audio_path.empty()) {
        return Status(StatusCode::kInvalidArgument, "--audio is required");
    }
    return ValidateAsrRunOptions(options->asr);
}

std::string BuildCliUsage(std::string_view program_name) {
    std::string usage;
    usage += std::string(program_name);
    usage += " --model-dir <dir> --audio <wav> [options]\n";
    usage += "  --threads <n>\n";
    usage += "  --stream\n";
    usage += "  --stream-max-new-tokens <n>  (default 32, max 128)\n";
    usage += "  --emit-segments\n";
    usage += "  --segment-max-codepoints <n>\n";
    usage += "  --prompt <text>\n";
    usage += "  --language <lang>\n";
    usage += "  --verbosity <n>\n";
    usage += "  --emit-tokens\n";
    usage += "  -h, --help\n";
    return usage;
}

}  // namespace qasr
