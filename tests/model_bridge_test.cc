#include "tests/test_registry.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "qasr/runtime/model_bridge.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeTestDirectory(const std::string & name) {
    const fs::path dir = fs::temp_directory_path() / name;
    fs::create_directories(dir);
    return dir;
}

void WriteFile(const fs::path & path, const std::string & body) {
    std::ofstream output(path);
    output << body;
}

}  // namespace

QASR_TEST(ValidateModelDirectoryRejectsMissingShards) {
    const fs::path dir = MakeTestDirectory("qasr_model_missing_shard");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00002-of-00002.safetensors", "");
    WriteFile(dir / "model.safetensors.index.json",
        "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}");

    const qasr::Status status = qasr::ValidateModelDirectory(dir.string());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kNotFound);
    QASR_EXPECT(status.message().find("model-00001-of-00002.safetensors") != std::string::npos);
}

QASR_TEST(ValidateModelDirectoryAcceptsMinimalCompleteLayout) {
    const fs::path dir = MakeTestDirectory("qasr_model_ok");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00001-of-00002.safetensors", "");
    WriteFile(dir / "model-00002-of-00002.safetensors", "");
    WriteFile(dir / "model.safetensors.index.json",
        "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}");

    QASR_EXPECT(qasr::ValidateModelDirectory(dir.string()).ok());
}

QASR_TEST(ValidateAsrRunOptionsRejectsBrokenValues) {
    const fs::path dir = MakeTestDirectory("qasr_options");
    WriteFile(dir / "config.json", "{}");
    WriteFile(dir / "vocab.json", "{}");
    WriteFile(dir / "merges.txt", "");
    WriteFile(dir / "model-00001-of-00002.safetensors", "");

    qasr::AsrRunOptions options;
    options.model_dir = dir.string();
    options.audio_path = (dir / "missing.wav").string();
    options.stream_max_new_tokens = 0;

    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kNotFound);

    WriteFile(dir / "sample.wav", "wav");
    options.audio_path = (dir / "sample.wav").string();
    options.stream_max_new_tokens = 1;
    options.threads = -1;
    QASR_EXPECT_EQ(qasr::ValidateAsrRunOptions(options).code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(CpuBackendAvailabilityIsConsistent) {
#ifdef QASR_CPU_BACKEND_ENABLED
    QASR_EXPECT(qasr::CpuBackendAvailable());
#else
    QASR_EXPECT(!qasr::CpuBackendAvailable());
#endif
}
