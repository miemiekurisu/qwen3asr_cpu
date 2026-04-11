#include "tests/test_registry.h"

extern "C" {
#include "src/backend/qwen_cpu/qwen_asr_kernels.h"
}

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

float NextPseudoRandom(std::uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return static_cast<float>((*state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

void FillPseudoRandom(std::vector<float> *values, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (float &value : *values) {
        value = NextPseudoRandom(&state);
    }
}

std::uint16_t FloatToBfloat16(float value) {
    std::uint32_t bits = 0;
    std::uint16_t bf16 = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bf16 = static_cast<std::uint16_t>(bits >> 16);
    return bf16;
}

float Bfloat16ToFloat(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void RunReferenceBfloat16Linear(std::vector<float> *output,
                                const std::vector<float> &x,
                                const std::vector<std::uint16_t> &weights,
                                const std::vector<float> *bias,
                                int seq_len,
                                int in_dim,
                                int out_dim) {
    for (int row = 0; row < seq_len; ++row) {
        for (int out = 0; out < out_dim; ++out) {
            float sum = bias ? (*bias)[static_cast<std::size_t>(out)] : 0.0f;
            for (int column = 0; column < in_dim; ++column) {
                sum += x[static_cast<std::size_t>(row * in_dim + column)] *
                       Bfloat16ToFloat(weights[static_cast<std::size_t>(out * in_dim + column)]);
            }
            (*output)[static_cast<std::size_t>(row * out_dim + out)] = sum;
        }
    }
}

void ExpectAllClose(const std::vector<float> &lhs,
                    const std::vector<float> &rhs,
                    float epsilon) {
    QASR_EXPECT_EQ(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        QASR_EXPECT(std::fabs(lhs[i] - rhs[i]) < epsilon);
    }
}

void RunSeparateQkv(std::vector<float> *q,
                    std::vector<float> *k,
                    std::vector<float> *v,
                    const std::vector<float> &x,
                    const std::vector<float> &wq,
                    const std::vector<float> &wk,
                    const std::vector<float> &wv,
                    const float *bq,
                    const float *bk,
                    const float *bv,
                    int seq_len,
                    int in_dim,
                    int q_dim,
                    int kv_dim) {
    qwen_linear(q->data(), x.data(), wq.data(), bq, seq_len, in_dim, q_dim);
    qwen_linear(k->data(), x.data(), wk.data(), bk, seq_len, in_dim, kv_dim);
    qwen_linear(v->data(), x.data(), wv.data(), bv, seq_len, in_dim, kv_dim);
}

void RunPackedQkv(std::vector<float> *q,
                  std::vector<float> *k,
                  std::vector<float> *v,
                  std::vector<float> *qkv_out,
                  const std::vector<float> &x,
                  const std::vector<float> &wq,
                  const std::vector<float> &wk,
                  const std::vector<float> &wv,
                  const float *bq,
                  const float *bk,
                  const float *bv,
                  int seq_len,
                  int in_dim,
                  int q_dim,
                  int kv_dim) {
    const int total_out = q_dim + 2 * kv_dim;
    std::vector<float> packed_weights(static_cast<std::size_t>(total_out * in_dim));
    std::vector<float> packed_biases(static_cast<std::size_t>(total_out));

    std::copy(wq.begin(), wq.end(), packed_weights.begin());
    std::copy(wk.begin(), wk.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>(q_dim * in_dim));
    std::copy(wv.begin(), wv.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>((q_dim + kv_dim) * in_dim));
    if (bq) {
        std::copy(bq, bq + q_dim, packed_biases.begin());
    } else {
        std::fill(packed_biases.begin(), packed_biases.begin() + q_dim, 0.0f);
    }
    if (bk) {
        std::copy(bk, bk + kv_dim, packed_biases.begin() + q_dim);
    } else {
        std::fill(packed_biases.begin() + q_dim, packed_biases.begin() + q_dim + kv_dim, 0.0f);
    }
    if (bv) {
        std::copy(bv, bv + kv_dim, packed_biases.begin() + q_dim + kv_dim);
    } else {
        std::fill(packed_biases.begin() + q_dim + kv_dim, packed_biases.end(), 0.0f);
    }

    qwen_linear_qkv_f32_packed(q->data(), k->data(), v->data(),
                               qkv_out->data(),
                               x.data(),
                               packed_weights.data(),
                               packed_biases.data(),
                               seq_len, in_dim, q_dim, kv_dim);
}

}  // namespace

QASR_TEST(QwenLinearQkvF32MatchesSeparateNormalShape) {
    const int seq_len = 4;
    const int in_dim = 5;
    const int q_dim = 3;
    const int kv_dim = 2;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<float> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> bq(static_cast<std::size_t>(q_dim));
    std::vector<float> bk(static_cast<std::size_t>(kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(kv_dim));

    FillPseudoRandom(&x, 1u);
    FillPseudoRandom(&wq, 2u);
    FillPseudoRandom(&wk, 3u);
    FillPseudoRandom(&wv, 4u);
    FillPseudoRandom(&bq, 5u);
    FillPseudoRandom(&bk, 6u);
    FillPseudoRandom(&bv, 7u);

    std::vector<float> q_fused(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_fused(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_fused(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_separate(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));
    std::vector<float> w_qkv(static_cast<std::size_t>(total_out * in_dim));
    std::vector<float> b_qkv(static_cast<std::size_t>(total_out));

    qwen_linear_qkv_f32(q_fused.data(), k_fused.data(), v_fused.data(),
                        qkv_out.data(), w_qkv.data(), b_qkv.data(),
                        x.data(),
                        wq.data(), wk.data(), wv.data(),
                        bq.data(), bk.data(), bv.data(),
                        seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   bq.data(), bk.data(), bv.data(),
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_fused, q_separate, 1e-4f);
    ExpectAllClose(k_fused, k_separate, 1e-4f);
    ExpectAllClose(v_fused, v_separate, 1e-4f);
}

QASR_TEST(QwenLinearQkvF32PackedMatchesSeparateNormalShape) {
    const int seq_len = 4;
    const int in_dim = 5;
    const int q_dim = 3;
    const int kv_dim = 2;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<float> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> bq(static_cast<std::size_t>(q_dim));
    std::vector<float> bk(static_cast<std::size_t>(kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(kv_dim));

    FillPseudoRandom(&x, 11u);
    FillPseudoRandom(&wq, 12u);
    FillPseudoRandom(&wk, 13u);
    FillPseudoRandom(&wv, 14u);
    FillPseudoRandom(&bq, 15u);
    FillPseudoRandom(&bk, 16u);
    FillPseudoRandom(&bv, 17u);

    std::vector<float> q_packed(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_separate(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));

    RunPackedQkv(&q_packed, &k_packed, &v_packed, &qkv_out,
                 x, wq, wk, wv,
                 bq.data(), bk.data(), bv.data(),
                 seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   bq.data(), bk.data(), bv.data(),
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_packed, q_separate, 1e-4f);
    ExpectAllClose(k_packed, k_separate, 1e-4f);
    ExpectAllClose(v_packed, v_separate, 1e-4f);
}

QASR_TEST(QwenLinearQkvF32MatchesSeparateSingleTokenWithNullBiases) {
    const int seq_len = 1;
    const int in_dim = 1;
    const int q_dim = 1;
    const int kv_dim = 1;

    const std::vector<float> x = {1.25f};
    const std::vector<float> wq = {2.0f};
    const std::vector<float> wk = {-3.0f};
    const std::vector<float> wv = {4.5f};
    const std::vector<float> bk = {0.75f};

    std::vector<float> q_fused(1, 0.0f);
    std::vector<float> k_fused(1, 0.0f);
    std::vector<float> v_fused(1, 0.0f);
    std::vector<float> q_separate(1, 0.0f);
    std::vector<float> k_separate(1, 0.0f);
    std::vector<float> v_separate(1, 0.0f);
    std::vector<float> qkv_out(3, 0.0f);
    std::vector<float> w_qkv(3, 0.0f);
    std::vector<float> b_qkv(3, 0.0f);

    qwen_linear_qkv_f32(q_fused.data(), k_fused.data(), v_fused.data(),
                        qkv_out.data(), w_qkv.data(), b_qkv.data(),
                        x.data(),
                        wq.data(), wk.data(), wv.data(),
                        nullptr, bk.data(), nullptr,
                        seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   nullptr, bk.data(), nullptr,
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_fused, q_separate, 1e-6f);
    ExpectAllClose(k_fused, k_separate, 1e-6f);
    ExpectAllClose(v_fused, v_separate, 1e-6f);
}

QASR_TEST(QwenLinearQkvF32PackedMatchesSeparateSingleTokenWithNullBiases) {
    const int seq_len = 1;
    const int in_dim = 1;
    const int q_dim = 1;
    const int kv_dim = 1;
    const int total_out = q_dim + 2 * kv_dim;

    const std::vector<float> x = {1.25f};
    const std::vector<float> wq = {2.0f};
    const std::vector<float> wk = {-3.0f};
    const std::vector<float> wv = {4.5f};
    const std::vector<float> bk = {0.75f};

    std::vector<float> q_packed(1, 0.0f);
    std::vector<float> k_packed(1, 0.0f);
    std::vector<float> v_packed(1, 0.0f);
    std::vector<float> q_separate(1, 0.0f);
    std::vector<float> k_separate(1, 0.0f);
    std::vector<float> v_separate(1, 0.0f);
    std::vector<float> qkv_out(static_cast<std::size_t>(total_out), 0.0f);

    RunPackedQkv(&q_packed, &k_packed, &v_packed, &qkv_out,
                 x, wq, wk, wv,
                 nullptr, bk.data(), nullptr,
                 seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   nullptr, bk.data(), nullptr,
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_packed, q_separate, 1e-6f);
    ExpectAllClose(k_packed, k_separate, 1e-6f);
    ExpectAllClose(v_packed, v_separate, 1e-6f);
}

QASR_TEST(QwenLinearQkvF32MatchesSeparatePseudoRandomShape) {
    const int seq_len = 7;
    const int in_dim = 11;
    const int q_dim = 5;
    const int kv_dim = 3;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<float> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> bq(static_cast<std::size_t>(q_dim));
    std::vector<float> bk(static_cast<std::size_t>(kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(kv_dim));

    FillPseudoRandom(&x, 101u);
    FillPseudoRandom(&wq, 202u);
    FillPseudoRandom(&wk, 303u);
    FillPseudoRandom(&wv, 404u);
    FillPseudoRandom(&bq, 505u);
    FillPseudoRandom(&bk, 606u);
    FillPseudoRandom(&bv, 707u);

    std::vector<float> q_fused(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_fused(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_fused(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_separate(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));
    std::vector<float> w_qkv(static_cast<std::size_t>(total_out * in_dim));
    std::vector<float> b_qkv(static_cast<std::size_t>(total_out));

    qwen_linear_qkv_f32(q_fused.data(), k_fused.data(), v_fused.data(),
                        qkv_out.data(), w_qkv.data(), b_qkv.data(),
                        x.data(),
                        wq.data(), wk.data(), wv.data(),
                        bq.data(), bk.data(), bv.data(),
                        seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   bq.data(), bk.data(), bv.data(),
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_fused, q_separate, 2e-4f);
    ExpectAllClose(k_fused, k_separate, 2e-4f);
    ExpectAllClose(v_fused, v_separate, 2e-4f);
}

QASR_TEST(QwenLinearQkvF32PackedMatchesSeparatePseudoRandomShape) {
    const int seq_len = 7;
    const int in_dim = 11;
    const int q_dim = 5;
    const int kv_dim = 3;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<float> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> bq(static_cast<std::size_t>(q_dim));
    std::vector<float> bk(static_cast<std::size_t>(kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(kv_dim));

    FillPseudoRandom(&x, 111u);
    FillPseudoRandom(&wq, 222u);
    FillPseudoRandom(&wk, 333u);
    FillPseudoRandom(&wv, 444u);
    FillPseudoRandom(&bq, 555u);
    FillPseudoRandom(&bk, 666u);
    FillPseudoRandom(&bv, 777u);

    std::vector<float> q_packed(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_separate(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));

    RunPackedQkv(&q_packed, &k_packed, &v_packed, &qkv_out,
                 x, wq, wk, wv,
                 bq.data(), bk.data(), bv.data(),
                 seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   bq.data(), bk.data(), bv.data(),
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_packed, q_separate, 2e-4f);
    ExpectAllClose(k_packed, k_separate, 2e-4f);
    ExpectAllClose(v_packed, v_separate, 2e-4f);
}

QASR_TEST(QwenLinearQkvF32ZeroSequenceLeavesOutputsUntouched) {
    std::vector<float> x(1, 0.0f);
    std::vector<float> wq(1, 1.0f);
    std::vector<float> wk(1, 2.0f);
    std::vector<float> wv(1, 3.0f);
    std::vector<float> q(1, 9.0f);
    std::vector<float> k(1, 8.0f);
    std::vector<float> v(1, 7.0f);
    std::vector<float> qkv_out(1, 0.0f);
    std::vector<float> w_qkv(3, 0.0f);
    std::vector<float> b_qkv(3, 0.0f);

    qwen_linear_qkv_f32(q.data(), k.data(), v.data(),
                        qkv_out.data(), w_qkv.data(), b_qkv.data(),
                        x.data(),
                        wq.data(), wk.data(), wv.data(),
                        nullptr, nullptr, nullptr,
                        0, 1, 1, 1);

    QASR_EXPECT(std::fabs(q[0] - 9.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(k[0] - 8.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(v[0] - 7.0f) < 1e-6f);
}

QASR_TEST(QwenLinearQkvF32PackedZeroSequenceLeavesOutputsUntouched) {
    std::vector<float> x(1, 0.0f);
    std::vector<float> q(1, 9.0f);
    std::vector<float> k(1, 8.0f);
    std::vector<float> v(1, 7.0f);
    std::vector<float> qkv_out(3, 0.0f);
    std::vector<float> packed_weights(3, 0.0f);
    std::vector<float> packed_biases(3, 0.0f);

    qwen_linear_qkv_f32_packed(q.data(), k.data(), v.data(),
                               qkv_out.data(),
                               x.data(),
                               packed_weights.data(),
                               packed_biases.data(),
                               0, 1, 1, 1);

    QASR_EXPECT(std::fabs(q[0] - 9.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(k[0] - 8.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(v[0] - 7.0f) < 1e-6f);
}

QASR_TEST(QwenEncoderQkvPolicyBestKeepsCurrentProductionPath) {
    const qwen_enc_qkv_impl_t below_threshold = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_BEST, 4, 1024, 1);
    const qwen_enc_qkv_impl_t normal_prefill = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_BEST, 13, 1024, 1);

    QASR_EXPECT_EQ(below_threshold, QWEN_ENC_QKV_IMPL_SEPARATE);
    QASR_EXPECT_EQ(normal_prefill, QWEN_ENC_QKV_IMPL_PACKED);
}

QASR_TEST(QwenEncoderQkvPolicyShapeAutoFallsBackOnLargeWideShapes) {
    const qwen_enc_qkv_impl_t large_wide = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 104, 1024, 1);
    const qwen_enc_qkv_impl_t medium = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 52, 1024, 1);
    const qwen_enc_qkv_impl_t no_packed = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_FORCE_PACKED, 13, 1024, 0);

    QASR_EXPECT_EQ(large_wide, QWEN_ENC_QKV_IMPL_SEPARATE);
    QASR_EXPECT_EQ(medium, QWEN_ENC_QKV_IMPL_PACKED);
    QASR_EXPECT_EQ(no_packed, QWEN_ENC_QKV_IMPL_SEPARATE);
}

QASR_TEST(QwenLinearBf16MatchesReferenceForSingleToken) {
    const int seq_len = 1;
    const int in_dim = 9;
    const int out_dim = 7;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<std::uint16_t> weights_bf16(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<float> bias(static_cast<std::size_t>(out_dim));
    std::vector<float> actual(static_cast<std::size_t>(seq_len * out_dim), 0.0f);
    std::vector<float> expected(static_cast<std::size_t>(seq_len * out_dim), 0.0f);

    FillPseudoRandom(&x, 909u);
    FillPseudoRandom(&weights_f32, 1001u);
    FillPseudoRandom(&bias, 1003u);
    for (std::size_t index = 0; index < weights_f32.size(); ++index) {
        weights_bf16[index] = FloatToBfloat16(weights_f32[index]);
    }

    qwen_linear_bf16(actual.data(), x.data(), weights_bf16.data(), bias.data(),
                     seq_len, in_dim, out_dim);
    RunReferenceBfloat16Linear(&expected, x, weights_bf16, &bias, seq_len, in_dim, out_dim);

    ExpectAllClose(actual, expected, 2e-4f);
}

QASR_TEST(QwenArgmaxMatvecBf16MatchesReference) {
    const int in_dim = 11;
    const int out_dim = 13;

    std::vector<float> x(static_cast<std::size_t>(in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<std::uint16_t> weights_bf16(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<float> logits(static_cast<std::size_t>(out_dim), 0.0f);

    FillPseudoRandom(&x, 2001u);
    FillPseudoRandom(&weights_f32, 2003u);
    for (std::size_t index = 0; index < weights_f32.size(); ++index) {
        weights_bf16[index] = FloatToBfloat16(weights_f32[index]);
    }

    RunReferenceBfloat16Linear(&logits, x, weights_bf16, nullptr, 1, in_dim, out_dim);

    int expected = 0;
    for (int index = 1; index < out_dim; ++index) {
        if (logits[static_cast<std::size_t>(index)] > logits[static_cast<std::size_t>(expected)]) {
            expected = index;
        }
    }

    const int actual = qwen_argmax_matvec_bf16(x.data(), weights_bf16.data(), in_dim, out_dim);
    QASR_EXPECT_EQ(actual, expected);
}