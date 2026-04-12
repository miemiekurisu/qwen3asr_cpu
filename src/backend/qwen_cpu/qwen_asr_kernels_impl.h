/*
 * qwen_asr_kernels_impl.h - internal architecture dispatch for hot kernels
 */

#ifndef QWEN_ASR_KERNELS_IMPL_H
#define QWEN_ASR_KERNELS_IMPL_H

#include <stdint.h>

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out);
float qwen_dot_f32_generic(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_generic(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#ifdef __ARM_NEON
void qwen_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float qwen_dot_f32_neon(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_neon(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

#elif defined(QWEN_X86_AVX2_AVAILABLE)
void qwen_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float qwen_dot_f32_avx(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_avx(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_avx(float *dst, const float *src, float correction, int n);
void qwen_softmax_causal_avx(float *S, int seq_q, int seq_k, int q_offset);
#endif

void qwen_softmax_causal_generic(float *S, int seq_q, int seq_k, int q_offset);

#endif /* QWEN_ASR_KERNELS_IMPL_H */
