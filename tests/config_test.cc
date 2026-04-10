#include "tests/test_registry.h"

#include "qasr/runtime/config.h"

QASR_TEST(ValidateEngineConfigAcceptsBaseline) {
    const qasr::EngineConfig config;
    QASR_EXPECT(qasr::ValidateEngineConfig(config).ok());
    QASR_EXPECT(qasr::HasAnyProtocolSurface(config));
}

QASR_TEST(ValidateEngineConfigRejectsBrokenValues) {
    qasr::EngineConfig config;
    config.intra_threads = -1;
    QASR_EXPECT_EQ(qasr::ValidateEngineConfig(config).code(), qasr::StatusCode::kInvalidArgument);

    config = qasr::EngineConfig{};
    config.max_sessions = 0;
    QASR_EXPECT_EQ(qasr::ValidateEngineConfig(config).code(), qasr::StatusCode::kInvalidArgument);

    config = qasr::EngineConfig{};
    config.max_queue_size = 0;
    QASR_EXPECT_EQ(qasr::ValidateEngineConfig(config).code(), qasr::StatusCode::kInvalidArgument);

    config = qasr::EngineConfig{};
    config.instance_name.clear();
    QASR_EXPECT_EQ(qasr::ValidateEngineConfig(config).code(), qasr::StatusCode::kInvalidArgument);

    config = qasr::EngineConfig{};
    config.enable_openai_compat = false;
    config.enable_vllm_compat = false;
    QASR_EXPECT_EQ(qasr::ValidateEngineConfig(config).code(), qasr::StatusCode::kFailedPrecondition);
    QASR_EXPECT(!qasr::HasAnyProtocolSurface(config));
}
