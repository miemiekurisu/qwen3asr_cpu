/*
 * qwen_asr_kernels_generic.c - architecture-generic hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#include <math.h>
#include <string.h>

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

float qwen_dot_f32_generic(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_generic(float *dst, float scale, int n) {
    for (int i = 0; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n) {
    for (int i = 0; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

void qwen_softmax_causal_generic(float *S, int seq_q, int seq_k, int q_offset) {
    for (int i = 0; i < seq_q; i++) {
        float *row = S + (size_t)i * seq_k;
        int valid_k = q_offset + i + 1;
        if (valid_k > seq_k) valid_k = seq_k;

        for (int j = valid_k; j < seq_k; j++) row[j] = -1e30f;

        float max_val = row[0];
        for (int j = 1; j < valid_k; j++) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_k; j++) {
            float v = expf(row[j] - max_val);
            row[j] = v;
            sum += v;
        }

        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int j = 0; j < seq_k; j++) row[j] *= inv_sum;
    }
}
