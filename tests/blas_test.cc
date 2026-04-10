#include "tests/test_registry.h"

#include "qasr/runtime/blas.h"

QASR_TEST(CompiledBlasBackendMatchesPolicyForCurrentBuild) {
    const qasr::BlasBackend backend = qasr::CompiledBlasBackend();
#if defined(__APPLE__)
    QASR_EXPECT_EQ(backend, qasr::BlasBackend::kAccelerate);
#else
    QASR_EXPECT_EQ(backend, qasr::BlasBackend::kOpenBlas);
#endif
}

QASR_TEST(BlasBackendNameCoversAllValues) {
    QASR_EXPECT_EQ(qasr::BlasBackendName(qasr::BlasBackend::kUnknown), std::string_view("unknown"));
    QASR_EXPECT_EQ(qasr::BlasBackendName(qasr::BlasBackend::kAccelerate), std::string_view("accelerate"));
    QASR_EXPECT_EQ(qasr::BlasBackendName(qasr::BlasBackend::kOpenBlas), std::string_view("openblas"));
}

QASR_TEST(ValidateBlasPolicyEnforcesPlatformRule) {
    QASR_EXPECT(qasr::ValidateBlasPolicy("macos", qasr::BlasBackend::kAccelerate).ok());
    QASR_EXPECT(qasr::ValidateBlasPolicy("linux", qasr::BlasBackend::kOpenBlas).ok());
    QASR_EXPECT(qasr::ValidateBlasPolicy("windows", qasr::BlasBackend::kOpenBlas).ok());

    QASR_EXPECT_EQ(qasr::ValidateBlasPolicy("macos", qasr::BlasBackend::kOpenBlas).code(), qasr::StatusCode::kFailedPrecondition);
    QASR_EXPECT_EQ(qasr::ValidateBlasPolicy("linux", qasr::BlasBackend::kAccelerate).code(), qasr::StatusCode::kFailedPrecondition);
    QASR_EXPECT_EQ(qasr::ValidateBlasPolicy("", qasr::BlasBackend::kUnknown).code(), qasr::StatusCode::kInvalidArgument);
}
