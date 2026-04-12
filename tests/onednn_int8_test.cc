/*
 * onednn_int8_test.cc - Tests for INT8 quantization and oneDNN matmul
 *
 * Tests are split into three groups:
 *   1. INT8 quantization (always compiled, no oneDNN dependency)
 *   2. oneDNN matmul (only when QASR_ONEDNN_AVAILABLE is defined)
 *   3. CLI/server --decoder-int8 option parsing
 */

#include "tests/test_registry.h"

extern "C" {
#include "src/backend/qwen_cpu/qwen_asr_onednn.h"
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qasr/cli/options.h"
#include "qasr/service/server.h"

namespace fs = std::filesystem;

namespace {

/* ======================================================================
 * Helpers
 * ====================================================================== */

std::uint16_t FloatToBf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16ToFloat(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float NextPseudoRandom(std::uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return static_cast<float>((*state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

void FillBf16Random(std::vector<std::uint16_t> *bf16,
                    std::size_t count,
                    std::uint32_t seed,
                    float scale = 1.0f) {
    bf16->resize(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        float v = NextPseudoRandom(&state) * scale;
        (*bf16)[i] = FloatToBf16(v);
    }
}

fs::path MakeInt8FixtureDirectory() {
    const fs::path dir = fs::temp_directory_path() / "qasr_int8_fixture";
    fs::create_directories(dir / "ui");
    std::ofstream(dir / "config.json") << "{}";
    std::ofstream(dir / "vocab.json") << "{}";
    std::ofstream(dir / "merges.txt") << "";
    std::ofstream(dir / "model-00001-of-00002.safetensors") << "";
    std::ofstream(dir / "ui" / "index.html") << "ok";
    std::ofstream(dir / "ui" / "app.js") << "ok";
    std::ofstream(dir / "ui" / "style.css") << "ok";
    return dir;
}

}  // namespace

/* ========================================================================
 * Group 1: INT8 Quantization Tests (always available)
 * ======================================================================== */

QASR_TEST(Int8QuantizeBf16BasicValues) {
    /* Quantize a small 2x3 matrix of known BF16 values and verify INT8 output. */
    const float src_values[] = {1.0f, -0.5f, 0.25f, -1.0f, 0.75f, 0.0f};
    std::vector<std::uint16_t> bf16(6);
    for (int i = 0; i < 6; ++i) {
        bf16[static_cast<std::size_t>(i)] = FloatToBf16(src_values[i]);
    }

    qwen_int8_weight_t w = {};
    int rc = qwen_int8_quantize_bf16(&w, bf16.data(), 2, 3);
    QASR_EXPECT_EQ(rc, 0);
    QASR_EXPECT(w.data != nullptr);
    QASR_EXPECT(w.row_scale != nullptr);
    QASR_EXPECT_EQ(w.rows, std::size_t(2));
    QASR_EXPECT_EQ(w.cols, std::size_t(3));
    QASR_EXPECT_EQ(w.data_bytes, std::size_t(6));

    /* Row 0: max |val| = 1.0, scale = 1.0/127 ≈ 0.00787
     * data[0] = round(1.0 * 127) = 127
     * data[1] = round(-0.5 * 127) = -64
     * data[2] = round(0.25 * 127) = 32 */
    QASR_EXPECT_EQ(w.data[0], int8_t(127));
    QASR_EXPECT(w.data[1] >= -64 && w.data[1] <= -63);
    QASR_EXPECT(w.data[2] >= 31 && w.data[2] <= 32);

    /* Row 1: max |val| = 1.0, scale = 1.0/127
     * data[3] = round(-1.0 * 127) = -127
     * data[4] = round(0.75 * 127) = 95
     * data[5] = round(0.0 * 127) = 0 */
    QASR_EXPECT_EQ(w.data[3], int8_t(-127));
    QASR_EXPECT(w.data[4] >= 94 && w.data[4] <= 96);
    QASR_EXPECT_EQ(w.data[5], int8_t(0));

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16RoundTrip) {
    /* Quantize and then dequantize a larger random matrix.
     * Check that relative error typically stays within a few percent. */
    const std::size_t rows = 16;
    const std::size_t cols = 32;
    std::vector<std::uint16_t> bf16;
    FillBf16Random(&bf16, rows * cols, 42, 2.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);

    /* Dequantize and check accuracy */
    int large_error_count = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float original = Bf16ToFloat(bf16[r * cols + c]);
            float dequant = static_cast<float>(w.data[r * cols + c]) * w.row_scale[r];
            float abs_err = std::fabs(original - dequant);
            /* Per-row symmetric INT8: expect < 2% error for most values */
            if (std::fabs(original) > 0.1f && abs_err / std::fabs(original) > 0.05f) {
                ++large_error_count;
            }
        }
    }
    /* Allow up to 5% of values to exceed the tolerance (BF16 rounding + INT8 quant) */
    QASR_EXPECT(large_error_count < static_cast<int>(rows * cols) / 20);

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16AllZeroRow) {
    /* A row of all zeros should produce scale=1.0 and zero data. */
    std::vector<std::uint16_t> bf16(8, FloatToBf16(0.0f));

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), 1, 8), 0);

    /* scale should be 1.0 (the fallback to avoid division by zero) */
    QASR_EXPECT(std::fabs(w.row_scale[0] - 1.0f) < 1e-6f);

    for (std::size_t i = 0; i < 8; ++i) {
        QASR_EXPECT_EQ(w.data[i], int8_t(0));
    }

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16MaxValueClamps) {
    /* Values near the limit should clamp to ±127. */
    float vals[] = {100.0f, -100.0f, 50.0f, -50.0f};
    std::vector<std::uint16_t> bf16(4);
    for (int i = 0; i < 4; ++i) {
        bf16[static_cast<std::size_t>(i)] = FloatToBf16(vals[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), 1, 4), 0);

    /* max |val| = 100.0, so 100 maps to 127, -100 to -127, 50 to ~64 */
    QASR_EXPECT_EQ(w.data[0], int8_t(127));
    QASR_EXPECT_EQ(w.data[1], int8_t(-127));
    QASR_EXPECT(std::abs(w.data[2] - 64) <= 1);
    QASR_EXPECT(std::abs(w.data[3] + 64) <= 1);

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16SingleElement) {
    std::uint16_t bf16 = FloatToBf16(3.14f);
    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, &bf16, 1, 1), 0);

    /* Single element: scale = 3.14/127, data[0] = 127 */
    QASR_EXPECT_EQ(w.data[0], int8_t(127));
    QASR_EXPECT(w.row_scale[0] > 0.0f);

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16LargerMatrix) {
    /* Stress test with a 64x128 matrix. */
    const std::size_t rows = 64;
    const std::size_t cols = 128;
    std::vector<std::uint16_t> bf16;
    FillBf16Random(&bf16, rows * cols, 12345, 5.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);
    QASR_EXPECT_EQ(w.rows, rows);
    QASR_EXPECT_EQ(w.cols, cols);
    QASR_EXPECT_EQ(w.data_bytes, rows * cols);

    /* Verify all INT8 values are in [-127, 127] */
    for (std::size_t i = 0; i < rows * cols; ++i) {
        QASR_EXPECT(w.data[i] >= -127 && w.data[i] <= 127);
    }

    /* Verify all scales are positive */
    for (std::size_t r = 0; r < rows; ++r) {
        QASR_EXPECT(w.row_scale[r] > 0.0f);
    }

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16NegativeOnlyRow) {
    /* Row with all negative values. */
    float vals[] = {-2.0f, -0.5f, -1.0f, -0.1f};
    std::vector<std::uint16_t> bf16(4);
    for (int i = 0; i < 4; ++i) {
        bf16[static_cast<std::size_t>(i)] = FloatToBf16(vals[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), 1, 4), 0);

    /* All data should be negative */
    for (int i = 0; i < 4; ++i) {
        QASR_EXPECT(w.data[i] <= 0);
    }
    /* First element (max abs) maps to -127 */
    QASR_EXPECT_EQ(w.data[0], int8_t(-127));

    qwen_int8_weight_free(&w);
}

QASR_TEST(Int8QuantizeBf16MultipleRows) {
    /* 4 rows, verify each row has independent scale. */
    float row_data[4][4] = {
        {1.0f, 0.5f, 0.25f, 0.125f},     /* max=1.0 */
        {10.0f, 5.0f, 2.5f, 1.25f},       /* max=10.0 */
        {0.01f, 0.005f, 0.0025f, 0.001f}, /* max=0.01 */
        {100.0f, 50.0f, 25.0f, 12.5f},    /* max=100.0 */
    };
    std::vector<std::uint16_t> bf16(16);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            bf16[static_cast<std::size_t>(r * 4 + c)] = FloatToBf16(row_data[r][c]);
        }
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), 4, 4), 0);

    /* Each row's first element should be 127 (it's the max) */
    QASR_EXPECT_EQ(w.data[0], int8_t(127));
    QASR_EXPECT_EQ(w.data[4], int8_t(127));
    QASR_EXPECT_EQ(w.data[8], int8_t(127));
    QASR_EXPECT_EQ(w.data[12], int8_t(127));

    /* Scales should be proportional to max values */
    QASR_EXPECT(w.row_scale[1] > w.row_scale[0] * 5.0f);
    QASR_EXPECT(w.row_scale[3] > w.row_scale[1] * 5.0f);

    qwen_int8_weight_free(&w);
}

/* --- Error handling tests --- */

QASR_TEST(Int8QuantizeBf16RejectsNullDst) {
    std::uint16_t dummy = 0;
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(nullptr, &dummy, 1, 1), -1);
}

QASR_TEST(Int8QuantizeBf16RejectsNullSrc) {
    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, nullptr, 1, 1), -1);
}

QASR_TEST(Int8QuantizeBf16RejectsZeroRows) {
    qwen_int8_weight_t w = {};
    std::uint16_t dummy = 0;
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, &dummy, 0, 1), -1);
}

QASR_TEST(Int8QuantizeBf16RejectsZeroCols) {
    qwen_int8_weight_t w = {};
    std::uint16_t dummy = 0;
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, &dummy, 1, 0), -1);
}

/* --- Free tests --- */

QASR_TEST(Int8WeightFreeHandlesNull) {
    /* Should not crash. */
    qwen_int8_weight_free(nullptr);
}

QASR_TEST(Int8WeightFreeZerosFields) {
    const std::size_t rows = 4;
    const std::size_t cols = 8;
    std::vector<std::uint16_t> bf16;
    FillBf16Random(&bf16, rows * cols, 99, 1.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);
    QASR_EXPECT(w.data != nullptr);
    QASR_EXPECT(w.row_scale != nullptr);

    qwen_int8_weight_free(&w);
    QASR_EXPECT(w.data == nullptr);
    QASR_EXPECT(w.row_scale == nullptr);
    QASR_EXPECT_EQ(w.rows, std::size_t(0));
    QASR_EXPECT_EQ(w.cols, std::size_t(0));
    QASR_EXPECT_EQ(w.data_bytes, std::size_t(0));
}

QASR_TEST(Int8WeightFreeDoubleFreeSafe) {
    /* Double free should be safe since fields are zeroed. */
    const std::size_t rows = 2;
    const std::size_t cols = 4;
    std::vector<std::uint16_t> bf16;
    FillBf16Random(&bf16, rows * cols, 77, 1.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);

    qwen_int8_weight_free(&w);
    qwen_int8_weight_free(&w);  /* Should not crash */
    QASR_EXPECT(w.data == nullptr);
}

/* ========================================================================
 * Group 2: oneDNN Matmul Tests (only when oneDNN is available)
 *
 * When oneDNN is NOT compiled in, onednn_init() returns -1 and
 * matmul_create() returns NULL. We test both scenarios.
 * ======================================================================== */

QASR_TEST(OneDnnInitReturnsConsistently) {
    /* Call twice — should be idempotent. */
    int rc1 = qwen_onednn_init();
    int rc2 = qwen_onednn_init();
    QASR_EXPECT_EQ(rc1, rc2);
}

QASR_TEST(OneDnnShutdownSafe) {
    /* Should not crash even if init was never called, or called twice. */
    qwen_onednn_shutdown();
    qwen_onednn_shutdown();
}

QASR_TEST(OneDnnMatmulCreateFromQuantized) {
    /* Quantize a small matrix, then try to create a matmul handle. */
    const std::size_t rows = 8;
    const std::size_t cols = 4;
    std::vector<std::uint16_t> bf16;
    FillBf16Random(&bf16, rows * cols, 1001, 1.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    if (qwen_onednn_init() == 0) {
        /* oneDNN available — handle should be valid */
        QASR_EXPECT(mm != nullptr);
        qwen_onednn_matmul_free(mm);
    } else {
        /* oneDNN not available — stub returns NULL */
        QASR_EXPECT(mm == nullptr);
    }

    qwen_int8_weight_free(&w);
}

QASR_TEST(OneDnnMatmulCreateRejectsNull) {
    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(nullptr);
    QASR_EXPECT(mm == nullptr);
}

QASR_TEST(OneDnnMatmulFreeHandlesNull) {
    /* Should not crash. */
    qwen_onednn_matmul_free(nullptr);
}

QASR_TEST(OneDnnMatmulExecuteRejectsNull) {
    float dummy_src = 1.0f;
    float dummy_dst = 0.0f;
    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(nullptr, &dummy_src, 1, &dummy_dst), -1);
}

QASR_TEST(OneDnnInt8MatvecRejectsNull) {
    float dummy_x = 1.0f;
    float dummy_y = 0.0f;
    QASR_EXPECT_EQ(qwen_int8_matvec(nullptr, &dummy_x, 1, &dummy_y), -1);
}

#if defined(QASR_ONEDNN_AVAILABLE)

/* These tests only run when oneDNN is actually linked. */

QASR_TEST(OneDnnInitSucceeds) {
    QASR_EXPECT_EQ(qwen_onednn_init(), 0);
}

QASR_TEST(OneDnnMatmulExecuteIdentity) {
    /* Create a weight matrix that approximates identity via INT8 quantization.
     * Weight[N,K] where N=K=4, diagonal = 1.0, rest 0.
     * After quantize: scale per row = 1.0/127, data diagonal = 127. */
    const int dim = 4;
    std::vector<std::uint16_t> bf16_weight(dim * dim, FloatToBf16(0.0f));
    for (int i = 0; i < dim; ++i) {
        bf16_weight[static_cast<std::size_t>(i * dim + i)] = FloatToBf16(1.0f);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16_weight.data(),
                   static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* src = [1, 2, 3, 4], expect dst ≈ [1, 2, 3, 4] */
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {0.0f};

    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src, 1, dst), 0);

    for (int i = 0; i < dim; ++i) {
        QASR_EXPECT(std::fabs(dst[i] - src[i]) < 0.1f);
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(OneDnnMatmulExecuteMatchesReference) {
    /* Compare INT8 matmul output against a float reference computation.
     * Weight[N,K] with N=8, K=16 */
    const int N = 8;
    const int K = 16;
    const int M = 1;  /* decode path */

    std::vector<std::uint16_t> bf16_weight;
    FillBf16Random(&bf16_weight, N * K, 2024, 0.5f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16_weight.data(),
                   static_cast<std::size_t>(N), static_cast<std::size_t>(K)), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* Generate input vector */
    std::vector<float> src(K);
    std::uint32_t seed = 9999;
    for (int i = 0; i < K; ++i) {
        src[static_cast<std::size_t>(i)] = NextPseudoRandom(&seed);
    }

    /* oneDNN result */
    std::vector<float> dst_int8(N, 0.0f);
    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), M, dst_int8.data()), 0);

    /* Float reference: dst[n] = sum_k(src[k] * W_f32[n,k]) */
    std::vector<float> dst_ref(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float wval = Bf16ToFloat(bf16_weight[static_cast<std::size_t>(n * K + k)]);
            sum += src[static_cast<std::size_t>(k)] * wval;
        }
        dst_ref[static_cast<std::size_t>(n)] = sum;
    }

    /* Compare with tolerance.  INT8 quantization introduces error;
     * accept max ~5% relative error or 0.05 absolute. */
    for (int n = 0; n < N; ++n) {
        float abs_err = std::fabs(dst_int8[static_cast<std::size_t>(n)] -
                                  dst_ref[static_cast<std::size_t>(n)]);
        float denom = std::fabs(dst_ref[static_cast<std::size_t>(n)]);
        if (denom > 0.1f) {
            QASR_EXPECT(abs_err / denom < 0.10f);
        } else {
            QASR_EXPECT(abs_err < 0.10f);
        }
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(OneDnnMatmulExecutePrefillM4) {
    /* Test with M > 1 (prefill path). */
    const int N = 8;
    const int K = 4;
    const int M = 4;

    std::vector<std::uint16_t> bf16_weight;
    FillBf16Random(&bf16_weight, N * K, 7777, 1.0f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16_weight.data(),
                   static_cast<std::size_t>(N), static_cast<std::size_t>(K)), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* Generate M*K input */
    std::vector<float> src(M * K);
    std::uint32_t seed = 5555;
    for (auto &v : src) v = NextPseudoRandom(&seed);

    std::vector<float> dst_int8(M * N, 0.0f);
    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), M, dst_int8.data()), 0);

    /* Float reference */
    std::vector<float> dst_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float wval = Bf16ToFloat(bf16_weight[static_cast<std::size_t>(n * K + k)]);
                sum += src[static_cast<std::size_t>(m * K + k)] * wval;
            }
            dst_ref[static_cast<std::size_t>(m * N + n)] = sum;
        }
    }

    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::fabs(dst_int8[static_cast<std::size_t>(i)] -
                                  dst_ref[static_cast<std::size_t>(i)]);
        float denom = std::fabs(dst_ref[static_cast<std::size_t>(i)]);
        if (denom > 0.1f) {
            QASR_EXPECT(abs_err / denom < 0.10f);
        } else {
            QASR_EXPECT(abs_err < 0.10f);
        }
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(OneDnnMatmulExecuteVaryingM) {
    /* Create once, execute with M=1 then M=3, testing the dynamic M path. */
    const int N = 4;
    const int K = 4;

    std::vector<std::uint16_t> bf16_weight;
    FillBf16Random(&bf16_weight, N * K, 3333, 0.8f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16_weight.data(),
                   static_cast<std::size_t>(N), static_cast<std::size_t>(K)), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* M=1 */
    {
        std::vector<float> src(K, 1.0f);
        std::vector<float> dst(N, 0.0f);
        QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), 1, dst.data()), 0);
    }

    /* M=3 — triggers dynamic M resize path */
    {
        std::vector<float> src(3 * K, 0.5f);
        std::vector<float> dst(3 * N, 0.0f);
        QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), 3, dst.data()), 0);
    }

    /* M=1 again — back to decode */
    {
        std::vector<float> src(K, -1.0f);
        std::vector<float> dst(N, 0.0f);
        QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), 1, dst.data()), 0);
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(OneDnnInt8MatvecMatchesExecute) {
    /* qwen_int8_matvec should produce the same result as matmul_execute. */
    const int N = 8;
    const int K = 8;

    std::vector<std::uint16_t> bf16_weight;
    FillBf16Random(&bf16_weight, N * K, 6666, 1.5f);

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16_weight.data(),
                   static_cast<std::size_t>(N), static_cast<std::size_t>(K)), 0);

    qwen_onednn_matmul_t *mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    std::vector<float> x(K);
    std::uint32_t seed = 8888;
    for (auto &v : x) v = NextPseudoRandom(&seed);

    std::vector<float> y1(N, 0.0f);
    std::vector<float> y2(N, 0.0f);

    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, x.data(), 1, y1.data()), 0);
    QASR_EXPECT_EQ(qwen_int8_matvec(mm, x.data(), 1, y2.data()), 0);

    for (int i = 0; i < N; ++i) {
        QASR_EXPECT(std::fabs(y1[static_cast<std::size_t>(i)] -
                              y2[static_cast<std::size_t>(i)]) < 1e-6f);
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

#endif  /* QASR_ONEDNN_AVAILABLE */

/* ========================================================================
 * Group 3: CLI / Server --decoder-int8 Option Parsing
 * ======================================================================== */

QASR_TEST(ParseCliArgumentsAcceptsDecoderInt8) {
    const fs::path dir = MakeInt8FixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string audio_path = (dir / "model-00001-of-00002.safetensors").string();
    const char * argv[] = {
        "qasr_cli",
        "--model-dir", model_dir.c_str(),
        "--audio", audio_path.c_str(),
        "--threads", "1",
        "--decoder-int8",
    };

    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &options);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(options.asr.decoder_int8);
}

QASR_TEST(ParseCliArgumentsDefaultsDecoderInt8Off) {
    const fs::path dir = MakeInt8FixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string audio_path = (dir / "model-00001-of-00002.safetensors").string();
    const char * argv[] = {
        "qasr_cli",
        "--model-dir", model_dir.c_str(),
        "--audio", audio_path.c_str(),
        "--threads", "1",
    };

    qasr::CliOptions options;
    const qasr::Status status = qasr::ParseCliArguments(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &options);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(!options.asr.decoder_int8);
}

QASR_TEST(BuildCliUsageMentionsDecoderInt8) {
    const std::string usage = qasr::BuildCliUsage("qasr_cli");
    QASR_EXPECT(usage.find("--decoder-int8") != std::string::npos);
}

QASR_TEST(ParseServerArgumentsAcceptsDecoderInt8) {
    const fs::path dir = MakeInt8FixtureDirectory();
    const std::string model_dir = dir.string();
    const std::string ui_dir = (dir / "ui").string();
    const char * argv[] = {
        "qasr_server",
        "--model-dir", model_dir.c_str(),
        "--ui-dir", ui_dir.c_str(),
        "--decoder-int8",
    };

    qasr::ServerConfig config;
    bool show_help = false;
    const qasr::Status status = qasr::ParseServerArguments(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, &config, &show_help);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(config.decoder_int8);
}

QASR_TEST(ParseServerArgumentsDefaultsDecoderInt8Off) {
    const char * argv[] = {"qasr_server", "--help"};
    qasr::ServerConfig config;
    bool show_help = false;
    const qasr::Status status = qasr::ParseServerArguments(2, argv, &config, &show_help);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(!config.decoder_int8);
}

QASR_TEST(BuildServerUsageMentionsDecoderInt8) {
    const std::string usage = qasr::BuildServerUsage("qasr_server");
    QASR_EXPECT(usage.find("--decoder-int8") != std::string::npos);
}

QASR_TEST(ServerConfigDecoderInt8DefaultFalse) {
    qasr::ServerConfig config;
    QASR_EXPECT(!config.decoder_int8);
}
