#include "tests/test_registry.h"

extern "C" {
#include "src/backend/qwen_cpu/qwen_asr.h"
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

void RunPackedQkvNoBias(std::vector<float> *q,
                        std::vector<float> *k,
                        std::vector<float> *v,
                        std::vector<float> *qkv_out,
                        const std::vector<float> &x,
                        const std::vector<float> &wq,
                        const std::vector<float> &wk,
                        const std::vector<float> &wv,
                        int seq_len,
                        int in_dim,
                        int q_dim,
                        int kv_dim) {
    const int total_out = q_dim + 2 * kv_dim;
    std::vector<float> packed_weights(static_cast<std::size_t>(total_out * in_dim));

    std::copy(wq.begin(), wq.end(), packed_weights.begin());
    std::copy(wk.begin(), wk.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>(q_dim * in_dim));
    std::copy(wv.begin(), wv.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>((q_dim + kv_dim) * in_dim));

    qwen_linear_nobias_qkv_f32_packed(q->data(), k->data(), v->data(),
                                      qkv_out->data(),
                                      x.data(),
                                      packed_weights.data(),
                                      seq_len, in_dim, q_dim, kv_dim);
}

void RunSeparateQkvBfloat16(std::vector<float> *q,
                            std::vector<float> *k,
                            std::vector<float> *v,
                            const std::vector<float> &x,
                            const std::vector<std::uint16_t> &wq,
                            const std::vector<std::uint16_t> &wk,
                            const std::vector<std::uint16_t> &wv,
                            int seq_len,
                            int in_dim,
                            int q_dim,
                            int kv_dim) {
    qwen_linear_nobias_bf16(q->data(), x.data(), wq.data(), seq_len, in_dim, q_dim);
    qwen_linear_nobias_bf16(k->data(), x.data(), wk.data(), seq_len, in_dim, kv_dim);
    qwen_linear_nobias_bf16(v->data(), x.data(), wv.data(), seq_len, in_dim, kv_dim);
}

void RunBfloat16NoBiasReference(std::vector<float> *output,
                                const std::vector<float> &x,
                                const std::vector<std::uint16_t> &weights,
                                int seq_len,
                                int in_dim,
                                int out_dim) {
    RunReferenceBfloat16Linear(output, x, weights, nullptr, seq_len, in_dim, out_dim);
}

void FillPreparedWeight(std::vector<std::uint16_t> *values, float seed) {
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = FloatToBfloat16(seed + static_cast<float>(index) * 0.03125f);
    }
}

void InitializeDecoderPrepareContext(qwen_ctx_t *ctx,
                                     int dec_layers,
                                     int hidden,
                                     int q_dim,
                                     int kv_dim,
                                     int intermediate,
                                     std::vector<std::vector<std::uint16_t>> *wq_storage,
                                     std::vector<std::vector<std::uint16_t>> *wk_storage,
                                     std::vector<std::vector<std::uint16_t>> *wv_storage,
                                     std::vector<std::vector<std::uint16_t>> *gate_up_storage) {
    ctx->config.dec_layers = dec_layers;
    ctx->config.dec_hidden = hidden;
    ctx->config.dec_heads = q_dim / 2;
    ctx->config.dec_head_dim = 2;
    ctx->config.dec_kv_heads = kv_dim / 2;
    ctx->config.dec_intermediate = intermediate;

    wq_storage->resize(static_cast<std::size_t>(dec_layers));
    wk_storage->resize(static_cast<std::size_t>(dec_layers));
    wv_storage->resize(static_cast<std::size_t>(dec_layers));
    gate_up_storage->resize(static_cast<std::size_t>(dec_layers));

    for (int layer = 0; layer < dec_layers; ++layer) {
        (*wq_storage)[static_cast<std::size_t>(layer)].resize(static_cast<std::size_t>(q_dim * hidden));
        (*wk_storage)[static_cast<std::size_t>(layer)].resize(static_cast<std::size_t>(kv_dim * hidden));
        (*wv_storage)[static_cast<std::size_t>(layer)].resize(static_cast<std::size_t>(kv_dim * hidden));
        (*gate_up_storage)[static_cast<std::size_t>(layer)].resize(static_cast<std::size_t>(2 * intermediate * hidden));
        FillPreparedWeight(&(*wq_storage)[static_cast<std::size_t>(layer)], 1.0f + static_cast<float>(layer));
        FillPreparedWeight(&(*wk_storage)[static_cast<std::size_t>(layer)], 2.0f + static_cast<float>(layer));
        FillPreparedWeight(&(*wv_storage)[static_cast<std::size_t>(layer)], 3.0f + static_cast<float>(layer));
        FillPreparedWeight(&(*gate_up_storage)[static_cast<std::size_t>(layer)], 4.0f + static_cast<float>(layer));
        ctx->decoder.layers[layer].wq_weight_bf16 = (*wq_storage)[static_cast<std::size_t>(layer)].data();
        ctx->decoder.layers[layer].wk_weight_bf16 = (*wk_storage)[static_cast<std::size_t>(layer)].data();
        ctx->decoder.layers[layer].wv_weight_bf16 = (*wv_storage)[static_cast<std::size_t>(layer)].data();
        ctx->decoder.layers[layer].gate_up_fused_bf16 = (*gate_up_storage)[static_cast<std::size_t>(layer)].data();
    }
}

void FreePreparedDecoderContext(qwen_ctx_t *ctx) {
    for (int layer = 0; layer < ctx->config.dec_layers; ++layer) {
        std::free(ctx->decoder.layers[layer].prefill_qkv_prepared.f32_data);
        ctx->decoder.layers[layer].prefill_qkv_prepared.f32_data = nullptr;
        std::free(ctx->decoder.layers[layer].prefill_gate_up_prepared.f32_data);
        ctx->decoder.layers[layer].prefill_gate_up_prepared.f32_data = nullptr;
    }
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

QASR_TEST(QwenLinearNobiasQkvF32PackedMatchesSeparatePseudoRandomShape) {
    const int seq_len = 6;
    const int in_dim = 9;
    const int q_dim = 4;
    const int kv_dim = 3;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<float> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> q_packed(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_packed(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_separate(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_separate(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));

    FillPseudoRandom(&x, 901u);
    FillPseudoRandom(&wq, 902u);
    FillPseudoRandom(&wk, 903u);
    FillPseudoRandom(&wv, 904u);

    RunPackedQkvNoBias(&q_packed, &k_packed, &v_packed, &qkv_out,
                       x, wq, wk, wv,
                       seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkv(&q_separate, &k_separate, &v_separate,
                   x, wq, wk, wv,
                   nullptr, nullptr, nullptr,
                   seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_packed, q_separate, 2e-4f);
    ExpectAllClose(k_packed, k_separate, 2e-4f);
    ExpectAllClose(v_packed, v_separate, 2e-4f);
}

QASR_TEST(QwenLinearNobiasQkvF32PackedZeroSequenceLeavesOutputsUntouched) {
    std::vector<float> x(1, 0.0f);
    std::vector<float> q(1, 9.0f);
    std::vector<float> k(1, 8.0f);
    std::vector<float> v(1, 7.0f);
    std::vector<float> qkv_out(3, 0.0f);
    std::vector<float> packed_weights(3, 0.0f);

    qwen_linear_nobias_qkv_f32_packed(q.data(), k.data(), v.data(),
                                      qkv_out.data(), x.data(), packed_weights.data(),
                                      0, 1, 1, 1);

    QASR_EXPECT(std::fabs(q[0] - 9.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(k[0] - 8.0f) < 1e-6f);
    QASR_EXPECT(std::fabs(v[0] - 7.0f) < 1e-6f);
}

QASR_TEST(QwenLinearNobiasBf16QkvPrefillMatchesSeparatePseudoRandomShape) {
    const int seq_len = 5;
    const int in_dim = 7;
    const int q_dim = 4;
    const int kv_dim = 3;
    const int total_out = q_dim + 2 * kv_dim;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>(total_out * in_dim));
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> q_actual(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_actual(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_actual(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> q_expected(static_cast<std::size_t>(seq_len * q_dim));
    std::vector<float> k_expected(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> v_expected(static_cast<std::size_t>(seq_len * kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(seq_len * total_out));
    std::vector<float> qkv_weights(static_cast<std::size_t>(total_out * in_dim));

    FillPseudoRandom(&x, 10001u);
    FillPseudoRandom(&weights_f32, 10003u);
    for (int out = 0; out < q_dim; ++out) {
        for (int col = 0; col < in_dim; ++col) {
            wq[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>(out * in_dim + col)]);
        }
    }
    for (int out = 0; out < kv_dim; ++out) {
        for (int col = 0; col < in_dim; ++col) {
            wk[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>((q_dim + out) * in_dim + col)]);
            wv[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>((q_dim + kv_dim + out) * in_dim + col)]);
        }
    }

    qwen_linear_nobias_bf16_qkv_prefill(q_actual.data(), k_actual.data(), v_actual.data(),
                                        qkv_out.data(), qkv_weights.data(), x.data(),
                                        wq.data(), wk.data(), wv.data(),
                                        seq_len, in_dim, q_dim, kv_dim);
    RunSeparateQkvBfloat16(&q_expected, &k_expected, &v_expected,
                           x, wq, wk, wv,
                           seq_len, in_dim, q_dim, kv_dim);

    ExpectAllClose(q_actual, q_expected, 3e-4f);
    ExpectAllClose(k_actual, k_expected, 3e-4f);
    ExpectAllClose(v_actual, v_expected, 3e-4f);
}

QASR_TEST(QwenLinearNobiasBf16QkvPrefillSingleTokenUsesDecodePath) {
    const int in_dim = 8;
    const int q_dim = 5;
    const int kv_dim = 3;

    std::vector<float> x(static_cast<std::size_t>(in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>((q_dim + 2 * kv_dim) * in_dim));
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(q_dim * in_dim));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(kv_dim * in_dim));
    std::vector<float> q_actual(static_cast<std::size_t>(q_dim));
    std::vector<float> k_actual(static_cast<std::size_t>(kv_dim));
    std::vector<float> v_actual(static_cast<std::size_t>(kv_dim));
    std::vector<float> q_expected(static_cast<std::size_t>(q_dim));
    std::vector<float> k_expected(static_cast<std::size_t>(kv_dim));
    std::vector<float> v_expected(static_cast<std::size_t>(kv_dim));

    FillPseudoRandom(&x, 10101u);
    FillPseudoRandom(&weights_f32, 10103u);
    for (int out = 0; out < q_dim; ++out) {
        for (int col = 0; col < in_dim; ++col) {
            wq[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>(out * in_dim + col)]);
        }
    }
    for (int out = 0; out < kv_dim; ++out) {
        for (int col = 0; col < in_dim; ++col) {
            wk[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>((q_dim + out) * in_dim + col)]);
            wv[static_cast<std::size_t>(out * in_dim + col)] = FloatToBfloat16(
                weights_f32[static_cast<std::size_t>((q_dim + kv_dim + out) * in_dim + col)]);
        }
    }

    qwen_linear_nobias_bf16_qkv_prefill(q_actual.data(), k_actual.data(), v_actual.data(),
                                        nullptr, nullptr, x.data(),
                                        wq.data(), wk.data(), wv.data(),
                                        1, in_dim, q_dim, kv_dim);
    qwen_linear_nobias_bf16_qkv(q_expected.data(), k_expected.data(), v_expected.data(),
                                x.data(), wq.data(), wk.data(), wv.data(),
                                in_dim, q_dim, kv_dim);

    ExpectAllClose(q_actual, q_expected, 3e-4f);
    ExpectAllClose(k_actual, k_expected, 3e-4f);
    ExpectAllClose(v_actual, v_expected, 3e-4f);
}

QASR_TEST(QwenLinearNobiasBf16QkvPrefillNullScratchLeavesOutputsUntouched) {
    const int seq_len = 3;
    const int in_dim = 4;
    const int q_dim = 2;
    const int kv_dim = 2;
    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim), 0.5f);
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(q_dim * in_dim), FloatToBfloat16(1.0f));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(kv_dim * in_dim), FloatToBfloat16(2.0f));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(kv_dim * in_dim), FloatToBfloat16(3.0f));
    std::vector<float> q(static_cast<std::size_t>(seq_len * q_dim), 9.0f);
    std::vector<float> k(static_cast<std::size_t>(seq_len * kv_dim), 8.0f);
    std::vector<float> v(static_cast<std::size_t>(seq_len * kv_dim), 7.0f);

    qwen_linear_nobias_bf16_qkv_prefill(q.data(), k.data(), v.data(),
                                        nullptr, nullptr, x.data(),
                                        wq.data(), wk.data(), wv.data(),
                                        seq_len, in_dim, q_dim, kv_dim);

    QASR_EXPECT(std::all_of(q.begin(), q.end(), [](float value) { return std::fabs(value - 9.0f) < 1e-6f; }));
    QASR_EXPECT(std::all_of(k.begin(), k.end(), [](float value) { return std::fabs(value - 8.0f) < 1e-6f; }));
    QASR_EXPECT(std::all_of(v.begin(), v.end(), [](float value) { return std::fabs(value - 7.0f) < 1e-6f; }));
}

QASR_TEST(QwenLinearNobiasBf16ScratchMatchesReferencePseudoRandomShape) {
    const int seq_len = 4;
    const int in_dim = 6;
    const int out_dim = 5;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<std::uint16_t> weights_bf16(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<float> actual(static_cast<std::size_t>(seq_len * out_dim), 0.0f);
    std::vector<float> expected(static_cast<std::size_t>(seq_len * out_dim), 0.0f);
    std::vector<float> scratch(static_cast<std::size_t>(out_dim * in_dim), 0.0f);

    FillPseudoRandom(&x, 12001u);
    FillPseudoRandom(&weights_f32, 12003u);
    for (std::size_t index = 0; index < weights_f32.size(); ++index) {
        weights_bf16[index] = FloatToBfloat16(weights_f32[index]);
    }

    qwen_linear_nobias_bf16_scratch(actual.data(), x.data(), weights_bf16.data(),
                                    scratch.data(), seq_len, in_dim, out_dim);
    RunBfloat16NoBiasReference(&expected, x, weights_bf16, seq_len, in_dim, out_dim);

    ExpectAllClose(actual, expected, 3e-4f);
}

QASR_TEST(QwenLinearNobiasBf16ScratchSingleTokenUsesDecodePath) {
    const int seq_len = 1;
    const int in_dim = 7;
    const int out_dim = 4;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim));
    std::vector<float> weights_f32(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<std::uint16_t> weights_bf16(static_cast<std::size_t>(out_dim * in_dim));
    std::vector<float> actual(static_cast<std::size_t>(seq_len * out_dim), 0.0f);
    std::vector<float> expected(static_cast<std::size_t>(seq_len * out_dim), 0.0f);

    FillPseudoRandom(&x, 12101u);
    FillPseudoRandom(&weights_f32, 12103u);
    for (std::size_t index = 0; index < weights_f32.size(); ++index) {
        weights_bf16[index] = FloatToBfloat16(weights_f32[index]);
    }

    qwen_linear_nobias_bf16_scratch(actual.data(), x.data(), weights_bf16.data(),
                                    nullptr, seq_len, in_dim, out_dim);
    qwen_linear_nobias_bf16(expected.data(), x.data(), weights_bf16.data(),
                            seq_len, in_dim, out_dim);

    ExpectAllClose(actual, expected, 3e-4f);
}

QASR_TEST(QwenLinearNobiasBf16ScratchNullScratchLeavesOutputsUntouched) {
    const int seq_len = 3;
    const int in_dim = 4;
    const int out_dim = 3;

    std::vector<float> x(static_cast<std::size_t>(seq_len * in_dim), 0.25f);
    std::vector<std::uint16_t> weights_bf16(static_cast<std::size_t>(out_dim * in_dim), FloatToBfloat16(1.0f));
    std::vector<float> actual(static_cast<std::size_t>(seq_len * out_dim), 5.0f);

    qwen_linear_nobias_bf16_scratch(actual.data(), x.data(), weights_bf16.data(),
                                    nullptr, seq_len, in_dim, out_dim);

    QASR_EXPECT(std::all_of(actual.begin(), actual.end(), [](float value) { return std::fabs(value - 5.0f) < 1e-6f; }));
}

QASR_TEST(QwenRuntimeProfileSelectorsHonorBudget) {
    qwen_runtime_profile_config_t profile = {};
    profile.kind = QWEN_RUNTIME_PROFILE_BALANCED;
    profile.decoder_prefill_qkv_persist_f32 = 1;
    profile.decoder_prefill_qkv_budget_bytes = static_cast<std::size_t>(512ull * 1024ull * 1024ull);
    profile.decoder_prefill_gate_up_persist_f32 = 1;
    profile.decoder_prefill_gate_up_budget_bytes = static_cast<std::size_t>(300ull * 1024ull * 1024ull);

    QASR_EXPECT(qwen_should_prepare_decoder_prefill_qkv(&profile, 1024, 2048, 1024, 28) != 0);
    QASR_EXPECT(qwen_should_prepare_decoder_prefill_qkv(&profile, 2048, 2048, 1024, 28) == 0);
    QASR_EXPECT(qwen_should_prepare_decoder_prefill_gate_up(&profile, 1024, 3072, 28) == 0);
    profile.decoder_prefill_gate_up_budget_bytes = static_cast<std::size_t>(800ull * 1024ull * 1024ull);
    QASR_EXPECT(qwen_should_prepare_decoder_prefill_gate_up(&profile, 1024, 3072, 28) != 0);
    QASR_EXPECT(qwen_should_prepare_decoder_prefill_gate_up(&profile, 2048, 6144, 28) == 0);
    QASR_EXPECT(std::strcmp(qwen_runtime_profile_name(QWEN_RUNTIME_PROFILE_REALTIME), "realtime") == 0);
}

QASR_TEST(QwenFloatArenaReserveAllocResetAndFreeWork) {
    qwen_float_arena_t arena = {};
    float *first = nullptr;
    float *second = nullptr;

    QASR_EXPECT_EQ(qwen_float_arena_reserve(&arena, 8), 0);
    QASR_EXPECT(arena.capacity >= 8);
    first = qwen_float_arena_alloc(&arena, 3);
    second = qwen_float_arena_alloc(&arena, 5);
    QASR_EXPECT(first != nullptr);
    QASR_EXPECT(second != nullptr);
    QASR_EXPECT_EQ(static_cast<int>(arena.offset), 8);
    QASR_EXPECT(qwen_float_arena_alloc(&arena, 0) == nullptr);

    qwen_float_arena_reset(&arena);
    QASR_EXPECT_EQ(static_cast<int>(arena.offset), 0);
    first = qwen_float_arena_alloc(&arena, 9);
    QASR_EXPECT(first != nullptr);
    QASR_EXPECT(arena.capacity >= 9);

    qwen_float_arena_free(&arena);
    QASR_EXPECT(arena.data == nullptr);
    QASR_EXPECT_EQ(static_cast<int>(arena.capacity), 0);
    QASR_EXPECT_EQ(static_cast<int>(arena.offset), 0);
}

QASR_TEST(QwenPerfNowMsIsMonotonic) {
    const double t0 = qwen_perf_now_ms();
    const double t1 = qwen_perf_now_ms();

    QASR_EXPECT(t1 >= t0);
}

QASR_TEST(QwenDecoderPrepareRuntimeBuildsPreparedWeightsWhenBudgetAllows) {
    qwen_ctx_t ctx = {};
    std::vector<std::vector<std::uint16_t>> wq_storage;
    std::vector<std::vector<std::uint16_t>> wk_storage;
    std::vector<std::vector<std::uint16_t>> wv_storage;
    std::vector<std::vector<std::uint16_t>> gate_up_storage;
    const int dec_layers = 2;
    const int hidden = 4;
    const int q_dim = 4;
    const int kv_dim = 2;
    const int intermediate = 3;
    const std::size_t expected_qkv_bytes = static_cast<std::size_t>(q_dim + 2 * kv_dim) * hidden * sizeof(float);
    const std::size_t expected_gate_up_bytes = static_cast<std::size_t>(2 * intermediate) * hidden * sizeof(float);
    const int old_verbose = qwen_verbose;

    qwen_verbose = 0;
    InitializeDecoderPrepareContext(&ctx, dec_layers, hidden, q_dim, kv_dim, intermediate,
                                    &wq_storage, &wk_storage, &wv_storage, &gate_up_storage);
    ctx.runtime_profile.kind = QWEN_RUNTIME_PROFILE_REALTIME;
    ctx.runtime_profile.decoder_prefill_qkv_persist_f32 = 1;
    ctx.runtime_profile.decoder_prefill_qkv_budget_bytes = expected_qkv_bytes * dec_layers;
    ctx.runtime_profile.decoder_prefill_gate_up_persist_f32 = 1;
    ctx.runtime_profile.decoder_prefill_gate_up_budget_bytes = expected_gate_up_bytes * dec_layers;
    ctx.runtime_profile.decoder_layer_timing = 0;

    QASR_EXPECT_EQ(qwen_decoder_prepare_runtime(&ctx), 0);
    QASR_EXPECT_EQ(ctx.runtime_perf.decoder_prefill_qkv_layers, dec_layers);
    QASR_EXPECT_EQ(ctx.runtime_perf.decoder_prefill_gate_up_layers, dec_layers);
    QASR_EXPECT_EQ(static_cast<int>(ctx.runtime_perf.decoder_prefill_qkv_bytes), static_cast<int>(expected_qkv_bytes * dec_layers));
    QASR_EXPECT_EQ(static_cast<int>(ctx.runtime_perf.decoder_prefill_gate_up_bytes), static_cast<int>(expected_gate_up_bytes * dec_layers));
    QASR_EXPECT(ctx.runtime_perf.decoder_prefill_qkv_prepare_ms >= 0.0);
    QASR_EXPECT(ctx.runtime_perf.decoder_prefill_gate_up_prepare_ms >= 0.0);
    QASR_EXPECT(ctx.decoder.layers[0].prefill_qkv_prepared.f32_data != nullptr);
    QASR_EXPECT(ctx.decoder.layers[0].prefill_gate_up_prepared.f32_data != nullptr);
    QASR_EXPECT_EQ(static_cast<int>(ctx.decoder.layers[0].prefill_qkv_prepared.rows), q_dim + 2 * kv_dim);
    QASR_EXPECT_EQ(static_cast<int>(ctx.decoder.layers[0].prefill_qkv_prepared.cols), hidden);
    QASR_EXPECT_EQ(static_cast<int>(ctx.decoder.layers[0].prefill_gate_up_prepared.rows), 2 * intermediate);
    QASR_EXPECT_EQ(static_cast<int>(ctx.decoder.layers[0].prefill_gate_up_prepared.cols), hidden);
    QASR_EXPECT(std::fabs(ctx.decoder.layers[0].prefill_qkv_prepared.f32_data[0] -
                          Bfloat16ToFloat(wq_storage[0][0])) < 1e-6f);
    QASR_EXPECT(std::fabs(ctx.decoder.layers[0].prefill_gate_up_prepared.f32_data[0] -
                          Bfloat16ToFloat(gate_up_storage[0][0])) < 1e-6f);

    FreePreparedDecoderContext(&ctx);
    qwen_verbose = old_verbose;
}

QASR_TEST(QwenDecoderPrepareRuntimeSkipsPreparedWeightsWhenBudgetTooSmall) {
    qwen_ctx_t ctx = {};
    std::vector<std::vector<std::uint16_t>> wq_storage;
    std::vector<std::vector<std::uint16_t>> wk_storage;
    std::vector<std::vector<std::uint16_t>> wv_storage;
    std::vector<std::vector<std::uint16_t>> gate_up_storage;
    const int old_verbose = qwen_verbose;

    qwen_verbose = 0;
    InitializeDecoderPrepareContext(&ctx, 1, 4, 4, 2, 3,
                                    &wq_storage, &wk_storage, &wv_storage, &gate_up_storage);
    ctx.runtime_profile.kind = QWEN_RUNTIME_PROFILE_EDGE_LOWMEM;
    ctx.runtime_profile.decoder_prefill_qkv_persist_f32 = 1;
    ctx.runtime_profile.decoder_prefill_qkv_budget_bytes = 8;
    ctx.runtime_profile.decoder_prefill_gate_up_persist_f32 = 1;
    ctx.runtime_profile.decoder_prefill_gate_up_budget_bytes = 8;
    ctx.runtime_profile.decoder_layer_timing = 0;

    QASR_EXPECT_EQ(qwen_decoder_prepare_runtime(&ctx), 0);
    QASR_EXPECT(ctx.decoder.layers[0].prefill_qkv_prepared.f32_data == nullptr);
    QASR_EXPECT(ctx.decoder.layers[0].prefill_gate_up_prepared.f32_data == nullptr);
    QASR_EXPECT_EQ(ctx.runtime_perf.decoder_prefill_qkv_layers, 0);
    QASR_EXPECT_EQ(ctx.runtime_perf.decoder_prefill_gate_up_layers, 0);

    FreePreparedDecoderContext(&ctx);
    qwen_verbose = old_verbose;
}

QASR_TEST(QwenEncoderQkvPolicyBestUsesPackedFromSeq4) {
    qwen_set_thread_policy_override(8, 8);
    const qwen_enc_qkv_impl_t below_threshold = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_BEST, 3, 1024, 1);
    const qwen_enc_qkv_impl_t normal_prefill = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_BEST, 4, 1024, 1);

    QASR_EXPECT_EQ(below_threshold, QWEN_ENC_QKV_IMPL_SEPARATE);
    QASR_EXPECT_EQ(normal_prefill, QWEN_ENC_QKV_IMPL_PACKED);
    qwen_clear_thread_policy_override();
}

QASR_TEST(QwenEncoderQkvPolicyShapeAutoFallsBackOnLargeWideShapes) {
    qwen_set_thread_policy_override(8, 8);
    const qwen_enc_qkv_impl_t large_wide = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 104, 1024, 1);
    const qwen_enc_qkv_impl_t medium = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 52, 1024, 1);
    const qwen_enc_qkv_impl_t no_packed = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_FORCE_PACKED, 13, 1024, 0);

    QASR_EXPECT_EQ(large_wide, QWEN_ENC_QKV_IMPL_SEPARATE);
    QASR_EXPECT_EQ(medium, QWEN_ENC_QKV_IMPL_PACKED);
    QASR_EXPECT_EQ(no_packed, QWEN_ENC_QKV_IMPL_SEPARATE);
    qwen_clear_thread_policy_override();
}

QASR_TEST(QwenEncoderQkvPolicyShapeAutoKeepsPackedOnHighPrefillThreads) {
    qwen_set_thread_policy_override(12, 8);
    const qwen_enc_qkv_impl_t large_wide = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 104, 1024, 1);
    const qwen_enc_qkv_impl_t medium = qwen_select_encoder_qkv_impl(
        QWEN_ENC_QKV_POLICY_SHAPE_AUTO, 52, 1024, 1);

    QASR_EXPECT_EQ(large_wide, QWEN_ENC_QKV_IMPL_PACKED);
    QASR_EXPECT_EQ(medium, QWEN_ENC_QKV_IMPL_PACKED);
    qwen_clear_thread_policy_override();
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