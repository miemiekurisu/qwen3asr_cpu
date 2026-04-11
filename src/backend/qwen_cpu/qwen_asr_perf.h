#ifndef QWEN_ASR_PERF_H
#define QWEN_ASR_PERF_H

#include <stddef.h>

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

typedef enum {
    QWEN_RUNTIME_PROFILE_BALANCED = 0,
    QWEN_RUNTIME_PROFILE_REALTIME = 1,
    QWEN_RUNTIME_PROFILE_OFFLINE = 2,
    QWEN_RUNTIME_PROFILE_EDGE_LOWMEM = 3,
} qwen_runtime_profile_t;

typedef struct {
    qwen_runtime_profile_t kind;
    int decoder_prefill_qkv_persist_f32;
    size_t decoder_prefill_qkv_budget_bytes;
    int decoder_prefill_gate_up_persist_f32;
    size_t decoder_prefill_gate_up_budget_bytes;
    int decoder_layer_timing;
} qwen_runtime_profile_config_t;

typedef struct {
    float *data;
    size_t capacity;
    size_t offset;
} qwen_float_arena_t;

qwen_enc_qkv_policy_t qwen_get_encoder_qkv_policy(void);
qwen_enc_qkv_impl_t qwen_select_encoder_qkv_impl(qwen_enc_qkv_policy_t policy,
                                                 int seq_len,
                                                 int d_model,
                                                 int has_packed_weights);
const char *qwen_encoder_qkv_policy_name(qwen_enc_qkv_policy_t policy);
const char *qwen_encoder_qkv_impl_name(qwen_enc_qkv_impl_t impl);

const qwen_runtime_profile_config_t *qwen_get_runtime_profile_config(void);
const char *qwen_runtime_profile_name(qwen_runtime_profile_t profile);
int qwen_should_prepare_decoder_prefill_qkv(const qwen_runtime_profile_config_t *config,
                                            int hidden,
                                            int q_dim,
                                            int kv_dim,
                                            int layers);
int qwen_should_prepare_decoder_prefill_gate_up(const qwen_runtime_profile_config_t *config,
                                                int hidden,
                                                int intermediate,
                                                int layers);

int qwen_float_arena_reserve(qwen_float_arena_t *arena, size_t min_capacity);
void qwen_float_arena_reset(qwen_float_arena_t *arena);
float *qwen_float_arena_alloc(qwen_float_arena_t *arena, size_t count);
void qwen_float_arena_free(qwen_float_arena_t *arena);

double qwen_perf_now_ms(void);

int qwen_get_prefill_threads(void);
int qwen_get_decode_threads(void);
void qwen_set_thread_policy_override(int prefill_threads, int decode_threads);
void qwen_clear_thread_policy_override(void);
void qwen_apply_prefill_thread_policy(void);
void qwen_apply_decode_thread_policy(void);

int qwen_x86_cpu_supports_avx2_fma(void);

#endif /* QWEN_ASR_PERF_H */