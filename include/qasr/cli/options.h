#pragma once

#include <string>
#include <string_view>

#include "qasr/core/status.h"
#include "qasr/runtime/model_bridge.h"

namespace qasr {

struct CliOptions {
    bool show_help = false;
    AsrRunOptions asr;
    std::string output_format;  // "text", "srt", "vtt", "json" (default: "text")
    std::string output_path;    // output file path (default: stdout, or <audio_stem>.<ext>)

    // ForcedAligner options
    bool align = false;                 // --align: enable word-level alignment
    std::string aligner_model_dir;      // --aligner-model-dir
};

Status ParseCliArguments(int argc, const char * const argv[], CliOptions * options);
std::string BuildCliUsage(std::string_view program_name);

}  // namespace qasr
