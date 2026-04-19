#include "tests/test_registry.h"

#include <string>

#include "qasr/core/status.h"
#include "qasr/inference/aligner_client.h"
#include "qasr/inference/aligner_types.h"

// ========================================================================
// ValidateAlignerConfig
// ========================================================================

QASR_TEST(AlignerConfig_Valid) {
    qasr::AlignerConfig config;
    config.model_dir = "/some/path";
    QASR_EXPECT(qasr::ValidateAlignerConfig(config).ok());
}

QASR_TEST(AlignerConfig_EmptyModelDir) {
    qasr::AlignerConfig config;
    QASR_EXPECT_EQ(qasr::ValidateAlignerConfig(config).code(),
                   qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(AlignerConfig_DefaultThreads) {
    qasr::AlignerConfig config;
    QASR_EXPECT_EQ(config.threads, std::int32_t(0));
}

// ========================================================================
// ForcedAligner construction/lifecycle (no model)
// ========================================================================

QASR_TEST(ForcedAligner_DefaultNotLoaded) {
    qasr::ForcedAligner aligner;
    QASR_EXPECT(!aligner.IsLoaded());
}

QASR_TEST(ForcedAligner_MoveConstruct) {
    qasr::ForcedAligner a;
    qasr::ForcedAligner b(std::move(a));
    QASR_EXPECT(!b.IsLoaded());
}

QASR_TEST(ForcedAligner_UnloadWhenNotLoaded) {
    qasr::ForcedAligner aligner;
    // Should not crash
    aligner.Unload();
    QASR_EXPECT(!aligner.IsLoaded());
}

QASR_TEST(ForcedAligner_AlignBeforeLoad) {
    qasr::ForcedAligner aligner;
    qasr::AlignResult result;
    qasr::Status s = aligner.Align("audio.wav", "hello", "english", &result);
    QASR_EXPECT_EQ(s.code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(ForcedAligner_AlignNullResult) {
    qasr::ForcedAligner aligner;
    qasr::Status s = aligner.Align("audio.wav", "hello", "english", nullptr);
    QASR_EXPECT_EQ(s.code(), qasr::StatusCode::kInvalidArgument);
}
