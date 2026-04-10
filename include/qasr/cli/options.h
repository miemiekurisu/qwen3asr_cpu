#pragma once

#include <string>
#include <string_view>

#include "qasr/core/status.h"
#include "qasr/runtime/model_bridge.h"

namespace qasr {

struct CliOptions {
    bool show_help = false;
    AsrRunOptions asr;
};

Status ParseCliArguments(int argc, const char * const argv[], CliOptions * options);
std::string BuildCliUsage(std::string_view program_name);

}  // namespace qasr
