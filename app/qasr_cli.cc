#include "qasr/cli/options.h"
#include "qasr/runtime/model_bridge.h"

#include <iostream>

int main(int argc, char ** argv) {
    qasr::CliOptions options;
    const qasr::Status parse_status = qasr::ParseCliArguments(argc, argv, &options);
    if (!parse_status.ok()) {
        std::cerr << parse_status.ToString() << "\n";
        std::cerr << qasr::BuildCliUsage(argc > 0 ? argv[0] : "qasr_cli");
        return 1;
    }

    if (options.show_help) {
        std::cout << qasr::BuildCliUsage(argv[0]);
        return 0;
    }

    const qasr::AsrRunResult result = qasr::RunAsr(options.asr);
    if (!result.status.ok()) {
        std::cerr << result.status.ToString() << "\n";
        return 1;
    }

    if (options.asr.emit_tokens) {
        std::cout << "\n";
    } else if (!options.asr.emit_segments) {
        std::cout << result.text << "\n";
    }
    std::cout.flush();

    std::cerr << "inference_ms=" << result.total_ms
              << " audio_ms=" << result.audio_ms
              << " tokens=" << result.text_tokens
              << " encode_ms=" << result.encode_ms
              << " decode_ms=" << result.decode_ms << "\n";
    return 0;
}
