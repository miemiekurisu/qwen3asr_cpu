#include "qasr/cli/options.h"

#include <charconv>
#include <cstdlib>
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
            options->asr.temperature = t;
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
        if (arg == "--decoder-int8") {
            options->asr.decoder_int8 = true;
            continue;
        }
        if (arg == "--encoder-int8") {
            options->asr.encoder_int8 = true;
            continue;
        }
        if (arg == "--output-format") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--output-format", &value);
            if (!status.ok()) {
                return status;
            }
            options->output_format = value;
            ++index;
            continue;
        }
        if (arg == "--output") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--output", &value);
            if (!status.ok()) {
                return status;
            }
            options->output_path = value;
            ++index;
            continue;
        }
        if (arg == "--align") {
            options->align = true;
            continue;
        }
        if (arg == "--aligner-model-dir") {
            const char * value = nullptr;
            Status status = RequireValue(argc, argv, index, "--aligner-model-dir", &value);
            if (!status.ok()) {
                return status;
            }
            options->aligner_model_dir = value;
            ++index;
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
    if (options->align && options->aligner_model_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "--aligner-model-dir is required when --align is set");
    }
    return ValidateAsrRunOptions(options->asr);
}

std::string BuildCliUsage(std::string_view program_name) {
    std::string usage;
    usage += std::string(program_name);
    usage += " --model-dir <dir> --audio <file> [options]\n\n";
    usage += "必需参数 / Required:\n";
    usage += "  --model-dir <dir>              ASR 模型目录\n";
    usage += "  --audio <file>                 音频文件 (WAV/MP3/FLAC/AAC 等，非 WAV 自动调用 ffmpeg 转换)\n\n";
    usage += "输出选项 / Output:\n";
    usage += "  --output-format <fmt>          输出格式: text|srt|vtt|json (默认: text)\n";
    usage += "  --output <path>                输出文件路径 (默认: stdout / 字幕格式自动生成同名文件)\n\n";
    usage += "对齐选项 / Alignment:\n";
    usage += "  --align                        启用词级强制对齐 (生成更精确的字幕时间戳)\n";
    usage += "  --aligner-model-dir <dir>      ForcedAligner 模型目录\n\n";
    usage += "推理选项 / Inference:\n";
    usage += "  --threads <n>                  CPU 线程数 (默认: 自动检测)\n";
    usage += "  --language <lang>              强制语言 (如 Chinese, English)\n";
    usage += "  --prompt <text>                提示文本 (引导识别风格)\n";
    usage += "  --temperature <float>          采样温度 (默认: auto, 0=贪心, >0=采样)\n";
    usage += "  --decoder-int8                 解码器 INT8 量化 (减少内存，可能影响精度)\n";
    usage += "  --encoder-int8                 编码器 INT8 量化\n\n";
    usage += "高级选项 / Advanced:\n";
    usage += "  --stream                       流式分段推理\n";
    usage += "  --stream-max-new-tokens <n>    流式每段最大 token 数 (默认 32, max 128)\n";
    usage += "  --emit-tokens                  逐 token 输出到 stdout\n";
    usage += "  --emit-segments                按段输出到 stdout\n";
    usage += "  --segment-max-codepoints <n>   每段最大字符数\n";
    usage += "  --verbosity <n>                日志详细级别 (0=静默, 1=详细)\n";
    usage += "  -h, --help                     显示帮助\n";
    return usage;
}

}  // namespace qasr
