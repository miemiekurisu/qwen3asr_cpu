#include "tests/test_registry.h"
#include "qasr/core/inference_backend.h"

// --- Factory ---

QASR_TEST(CreateCpuBackendReturnsNonNull) {
    auto backend = qasr::CreateCpuBackend();
#ifdef QASR_CPU_BACKEND_ENABLED
    QASR_EXPECT(backend != nullptr);
#else
    QASR_EXPECT(backend == nullptr);
#endif
}

QASR_TEST(CpuBackendNotLoadedInitially) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;  // CPU backend not compiled
    QASR_EXPECT(!backend->IsLoaded());
    QASR_EXPECT_EQ(backend->EncoderOutputDim(), std::int32_t(0));
    QASR_EXPECT_EQ(backend->DecoderHiddenDim(), std::int32_t(0));
}

QASR_TEST(CpuBackendLoadFailsOnBadDir) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    qasr::Status s = backend->Load("/tmp/qasr-nonexistent-model-dir", 1);
    QASR_EXPECT(!s.ok());
    QASR_EXPECT(!backend->IsLoaded());
}

QASR_TEST(CpuBackendEncodeFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    std::vector<float> output;
    std::int32_t seq_len = 0;
    float dummy = 0.0f;
    qasr::Status s = backend->Encode(&dummy, 1, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(CpuBackendPrefillFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    float dummy = 0.0f;
    qasr::Status s = backend->Prefill(&dummy, 1);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(CpuBackendDecodeStepFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    float dummy = 0.0f;
    std::int32_t token = -1;
    qasr::Status s = backend->DecodeStep(&dummy, &token);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(CpuBackendResetDecoderSafeWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    backend->ResetDecoder();  // Must not crash
}

QASR_TEST(CpuBackendEncodeRejectsNull) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    qasr::Status s = backend->Encode(nullptr, 1, nullptr, nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(CpuBackendDecodeStepRejectsNull) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    qasr::Status s = backend->DecodeStep(nullptr, nullptr);
    QASR_EXPECT(!s.ok());
}
