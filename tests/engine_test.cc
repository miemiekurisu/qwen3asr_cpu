#include "tests/test_registry.h"

#include "qasr/runtime/engine.h"

QASR_TEST(ValidateBootstrapInputsPropagatesConfigErrors) {
    qasr::EngineConfig config;
    config.enable_openai_compat = false;
    config.enable_vllm_compat = false;
    const qasr::DecodeRequestOptions options;
    QASR_EXPECT_EQ(qasr::ValidateBootstrapInputs(config, options).code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(ValidateBootstrapInputsEnforcesRuntimeFlags) {
    qasr::EngineConfig config;
    qasr::DecodeRequestOptions options;

    config.enable_streaming = false;
    options.task_mode = qasr::TaskMode::kStreaming;
    QASR_EXPECT_EQ(qasr::ValidateBootstrapInputs(config, options).code(), qasr::StatusCode::kFailedPrecondition);

    config = qasr::EngineConfig{};
    config.enable_timestamps = false;
    options = qasr::DecodeRequestOptions{};
    options.timestamp_mode = qasr::TimestampMode::kWord;
    QASR_EXPECT_EQ(qasr::ValidateBootstrapInputs(config, options).code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(BuildBootstrapPlanReflectsOptions) {
    qasr::EngineConfig config;
    qasr::DecodeRequestOptions options;
    options.want_async = true;
    options.timestamp_mode = qasr::TimestampMode::kWord;

    const qasr::BootstrapPlan plan = qasr::BuildBootstrapPlan(config, options);
    QASR_EXPECT(plan.start_async_executor);
    QASR_EXPECT(plan.start_openai_chat_surface);
    QASR_EXPECT(plan.start_openai_audio_surface);
    QASR_EXPECT(plan.start_vllm_surface);
    QASR_EXPECT(plan.start_forced_aligner);
}
