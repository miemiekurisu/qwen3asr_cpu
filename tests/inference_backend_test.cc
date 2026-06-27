#include "tests/test_registry.h"
#include "qasr/backend/device_backend.h"
#include "qasr/backend/cpu_backend.h"

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
    if (!backend) return;
    QASR_EXPECT(!backend->PrepareWeights("/nonexistent").ok());
}

QASR_TEST(CpuBackendInitializeStub) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    QASR_EXPECT(backend->Initialize().ok());
}

QASR_TEST(CpuBackendEncodeFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    backend->Initialize();
    int out_tokens = 0;
    float dummy = 0.0f;
    QASR_EXPECT(!backend->EncodeMel(nullptr, &dummy, 1, nullptr, out_tokens).ok());
}

QASR_TEST(CpuBackendPrefillFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    backend->Initialize();
    QASR_EXPECT(!backend->DecoderPrefill(nullptr, nullptr, 0, nullptr, 0).ok());
}

QASR_TEST(CpuBackendDecodeStepFailsWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    backend->Initialize();
    std::int32_t token = -1;
    QASR_EXPECT(!backend->DecodeStep(nullptr, token).ok());
}

QASR_TEST(CpuBackendResetDecoderSafeWhenNotLoaded) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    QASR_EXPECT(backend->ResetDecoder(nullptr).ok());
}

QASR_TEST(CpuBackendKindReturnsCpu) {
    auto backend = qasr::CreateCpuBackend();
    if (!backend) return;
    QASR_EXPECT_EQ(backend->kind(), qasr::BackendKind::kCpu);
}
