#ifndef QWEN_ASR_PERF_H
#define QWEN_ASR_PERF_H

typedef enum {
    QWEN_ENC_QKV_POLICY_BEST = 0,
    QWEN_ENC_QKV_POLICY_FORCE_SEPARATE = 1,
    QWEN_ENC_QKV_POLICY_FORCE_PACKED = 2,
    QWEN_ENC_QKV_POLICY_SHAPE_AUTO = 3,
} qwen_enc_qkv_policy_t;

typedef enum {
    QWEN_ENC_QKV_IMPL_SEPARATE = 0,
    QWEN_ENC_QKV_IMPL_PACKED = 1,
} qwen_enc_qkv_impl_t;

qwen_enc_qkv_policy_t qwen_get_encoder_qkv_policy(void);
qwen_enc_qkv_impl_t qwen_select_encoder_qkv_impl(qwen_enc_qkv_policy_t policy,
                                                 int seq_len,
                                                 int d_model,
                                                 int has_packed_weights);
const char *qwen_encoder_qkv_policy_name(qwen_enc_qkv_policy_t policy);
const char *qwen_encoder_qkv_impl_name(qwen_enc_qkv_impl_t impl);

int qwen_get_prefill_threads(void);
int qwen_get_decode_threads(void);
void qwen_set_thread_policy_override(int prefill_threads, int decode_threads);
void qwen_clear_thread_policy_override(void);
void qwen_apply_prefill_thread_policy(void);
void qwen_apply_decode_thread_policy(void);

int qwen_x86_cpu_supports_avx2_fma(void);

#endif /* QWEN_ASR_PERF_H */