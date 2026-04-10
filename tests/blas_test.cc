#include "tests/test_registry.h"

#include "qasr/runtime/blas.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#if defined(QASR_BLAS_ACCELERATE)
#  include <Accelerate/Accelerate.h>
#elif defined(QASR_BLAS_OPENBLAS)
#  include <cblas.h>
#endif

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

// ---------------------------------------------------------------------------
// BLAS runtime smoke tests — actually call cblas_sgemm to verify linkage
// and correct computation at runtime, not just at link time.
// ---------------------------------------------------------------------------

#if defined(QASR_BLAS_ACCELERATE) || defined(QASR_BLAS_OPENBLAS)

// Normal: 2x3 * 3x2 = 2x2 identity-like product
QASR_TEST(BlasSgemmNormalSmallMatmul) {
    // A (2x3), B (3x2), C (2x2)
    // A = [[1,0,0],[0,1,0]]  B = [[1,0],[0,1],[0,0]]  => C = [[1,0],[0,1]]
    const float A[] = {1, 0, 0, 0, 1, 0};
    const float B[] = {1, 0, 0, 1, 0, 0};
    float C[] = {0, 0, 0, 0};

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 3, 1.0f, A, 3, B, 2, 0.0f, C, 2);

    QASR_EXPECT(std::fabs(C[0] - 1.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(C[1] - 0.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(C[2] - 0.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(C[3] - 1.0f) < 1e-6f);
}

// Normal: alpha/beta scaling
QASR_TEST(BlasSgemmAlphaBetaScaling) {
    // C = alpha * A * B + beta * C
    const float A[] = {2.0f, 3.0f};  // 1x2
    const float B[] = {4.0f, 5.0f};  // 2x1
    float C[] = {10.0f};             // 1x1, pre-loaded

    // C = 2.0 * [2,3]*[4;5] + 0.5 * 10 = 2*(8+15) + 5 = 46+5 = 51
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 2, 2.0f, A, 2, B, 1, 0.5f, C, 1);

    QASR_EXPECT(std::fabs(C[0] - 51.0f) < 1e-5f);
}

// Extreme: 1x1 degenerate
QASR_TEST(BlasSgemmExtremeSingleElement) {
    const float A[] = {7.0f};
    const float B[] = {3.0f};
    float C[] = {0.0f};

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0f, A, 1, B, 1, 0.0f, C, 1);

    QASR_EXPECT(std::fabs(C[0] - 21.0f) < 1e-6f);
}

// Extreme: larger matrix (64x64) to stress OpenBLAS threading
QASR_TEST(BlasSgemmExtremeLargerMatrix) {
    const int N = 64;
    std::vector<float> A(static_cast<std::size_t>(N * N), 0.0f);
    std::vector<float> B(static_cast<std::size_t>(N * N), 0.0f);
    std::vector<float> C(static_cast<std::size_t>(N * N), 0.0f);

    // Set A = I, B = constant 2.0 => C should equal B
    for (int i = 0; i < N; ++i) {
        A[static_cast<std::size_t>(i * N + i)] = 1.0f;
        for (int j = 0; j < N; ++j) {
            B[static_cast<std::size_t>(i * N + j)] = 2.0f;
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A.data(), N, B.data(), N, 0.0f, C.data(), N);

    for (int i = 0; i < N * N; ++i) {
        QASR_EXPECT(std::fabs(C[static_cast<std::size_t>(i)] - 2.0f) < 1e-5f);
    }
}

// Random: pseudo-random matrix multiply, verify via naive O(n^3)
QASR_TEST(BlasSgemmRandomVerifyAgainstNaive) {
    const int M = 8, K = 6, N = 4;
    std::vector<float> A(static_cast<std::size_t>(M * K));
    std::vector<float> B(static_cast<std::size_t>(K * N));
    std::vector<float> C_blas(static_cast<std::size_t>(M * N), 0.0f);
    std::vector<float> C_naive(static_cast<std::size_t>(M * N), 0.0f);

    // Deterministic pseudo-random fill
    unsigned seed = 42;
    auto next_float = [&seed]() -> float {
        seed = seed * 1103515245u + 12345u;
        return static_cast<float>(static_cast<int>((seed >> 16) & 0x7FFF)) / 32768.0f - 0.5f;
    };
    for (auto & v : A) v = next_float();
    for (auto & v : B) v = next_float();

    // BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_blas.data(), N);

    // Naive
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<std::size_t>(i * K + k)] * B[static_cast<std::size_t>(k * N + j)];
            }
            C_naive[static_cast<std::size_t>(i * N + j)] = sum;
        }
    }

    for (int i = 0; i < M * N; ++i) {
        QASR_EXPECT(std::fabs(C_blas[static_cast<std::size_t>(i)] - C_naive[static_cast<std::size_t>(i)]) < 1e-4f);
    }
}

// Error: zero alpha produces beta * C regardless of A, B
QASR_TEST(BlasSgemmErrorZeroAlphaPreservesC) {
    const float A[] = {999.0f};
    const float B[] = {999.0f};
    float C[] = {7.0f};

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 0.0f, A, 1, B, 1, 1.0f, C, 1);

    QASR_EXPECT(std::fabs(C[0] - 7.0f) < 1e-6f);
}

// Transpose: A^T * B
QASR_TEST(BlasSgemmTransposeA) {
    // A stored as 3x2 (row-major) but used as 2x3 transposed
    const float A[] = {1, 4, 2, 5, 3, 6};  // A^T = [[1,2,3],[4,5,6]]
    const float B[] = {1, 1, 1};            // 3x1
    float C[] = {0, 0};                     // 2x1

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                2, 1, 3, 1.0f, A, 2, B, 1, 0.0f, C, 1);

    // A^T * B = [[1+2+3],[4+5+6]] = [[6],[15]]
    QASR_EXPECT(std::fabs(C[0] - 6.0f) < 1e-5f);
    QASR_EXPECT(std::fabs(C[1] - 15.0f) < 1e-5f);
}

#endif  // QASR_BLAS_ACCELERATE || QASR_BLAS_OPENBLAS
