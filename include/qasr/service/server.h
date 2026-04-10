#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "qasr/core/status.h"

namespace qasr {

enum class TranscriptionResponseFormat {
    kJson = 0,
    kText,
    kVerboseJson,
};

struct ServerConfig {
    std::string model_dir;
    std::string host = "127.0.0.1";
    std::string ui_dir = "ui";
    std::int32_t port = 8080;
    std::int32_t threads = 0;
    std::int32_t verbosity = 0;
};

Status ParseBooleanText(std::string_view field_name, std::string_view text, bool * value);
Status ParseTranscriptionResponseFormat(
    std::string_view text,
    TranscriptionResponseFormat * format);
Status ValidateTimestampGranularities(bool want_segment_timestamps, bool want_word_timestamps);
std::string ResolveServedModelId(std::string_view model_dir);

Status ValidateServerConfig(const ServerConfig & config);
Status ParseServerArguments(int argc, const char * const argv[], ServerConfig * config, bool * show_help);
std::string BuildServerUsage(std::string_view program_name);
int RunServer(const ServerConfig & config);

}  // namespace qasr
