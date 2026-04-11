/*
 * qwen_asr_decoder.c - Qwen3 LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA 2:1,
 * tied embeddings (tok_embeddings == lm_head).
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static void bf16_to_f32_linear(float *dst, const uint16_t *src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(dst + i, &bits, sizeof(bits));
    }
}

static void clear_prepared_weight(qwen_prepared_f32_weight_t *prepared) {
    free(prepared->f32_data);
    prepared->f32_data = NULL;
    prepared->rows = 0;
    prepared->cols = 0;
    prepared->bytes = 0;
}

static void clear_decoder_prefill_qkv_prepared(qwen_ctx_t *ctx) {
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        clear_prepared_weight(&ctx->decoder.layers[i].prefill_qkv_prepared);
    }
    ctx->runtime_perf.decoder_prefill_qkv_prepare_ms = 0.0;
    ctx->runtime_perf.decoder_prefill_qkv_bytes = 0;
    ctx->runtime_perf.decoder_prefill_qkv_layers = 0;
}

static void clear_decoder_prefill_gate_up_prepared(qwen_ctx_t *ctx) {
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        clear_prepared_weight(&ctx->decoder.layers[i].prefill_gate_up_prepared);
    }
    ctx->runtime_perf.decoder_prefill_gate_up_prepare_ms = 0.0;
    ctx->runtime_perf.decoder_prefill_gate_up_bytes = 0;
    ctx->runtime_perf.decoder_prefill_gate_up_layers = 0;
}

static void prepare_decoder_prefill_qkv_block(float *dst,
                                              const qwen_dec_layer_t *layer,
                                              size_t q_count,
                                              size_t kv_count) {
    bf16_to_f32_linear(dst, layer->wq_weight_bf16, q_count);
    bf16_to_f32_linear(dst + q_count, layer->wk_weight_bf16, kv_count);
    bf16_to_f32_linear(dst + q_count + kv_count, layer->wv_weight_bf16, kv_count);
}

static void prepare_decoder_prefill_gate_up_block(float *dst,
                                                  const qwen_dec_layer_t *layer,
                                                  size_t count) {
    bf16_to_f32_linear(dst, layer->gate_up_fused_bf16, count);
}

int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    dec->tok_embeddings_bf16 = load_bf16_direct(ms,
        "thinker.model.embed_tokens.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_proj.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_proj.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.v_proj.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.o_proj.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* Per-head Q/K RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm_weight = load_f32(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.input_layernorm.weight", i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_f32(ms, name);

        /* SwiGLU MLP weights (bf16, no bias) */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.gate_proj.weight", i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.up_proj.weight", i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.down_proj.weight", i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, gate_row1, up_row1, ...] */
        {
            int inter = cfg->dec_intermediate;
            int hidden = cfg->dec_hidden;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            if (!l->gate_up_fused_bf16) {
                fprintf(stderr, "decoder: failed to allocate gate_up_fused for layer %d\n", i);
                return -1;
            }
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }

    }

    /* Final RMSNorm */
    dec->norm = load_f32(ms, "thinker.model.norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

int qwen_decoder_prepare_runtime(qwen_ctx_t *ctx) {
    const qwen_config_t *cfg = &ctx->config;
    const int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    const int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    const int gate_up_dim = 2 * cfg->dec_intermediate;
    const size_t q_count = (size_t)q_dim * (size_t)cfg->dec_hidden;
    const size_t kv_count = (size_t)kv_dim * (size_t)cfg->dec_hidden;
    const size_t total_rows = (size_t)q_dim + (size_t)(2 * kv_dim);
    const size_t total_count = total_rows * (size_t)cfg->dec_hidden;
    const size_t total_bytes = total_count * sizeof(float);
    const size_t gate_up_count = (size_t)gate_up_dim * (size_t)cfg->dec_hidden;
    const size_t gate_up_bytes = gate_up_count * sizeof(float);
    const int prepare_qkv = qwen_should_prepare_decoder_prefill_qkv(&ctx->runtime_profile,
                                                                    cfg->dec_hidden,
                                                                    q_dim,
                                                                    kv_dim,
                                                                    cfg->dec_layers);
    const int prepare_gate_up = qwen_should_prepare_decoder_prefill_gate_up(&ctx->runtime_profile,
                                                                            cfg->dec_hidden,
                                                                            cfg->dec_intermediate,
                                                                            cfg->dec_layers);

    if (!prepare_qkv && !prepare_gate_up) {
        return 0;
    }

    if (prepare_qkv) {
        const double total_start = qwen_perf_now_ms();
        clear_decoder_prefill_qkv_prepared(ctx);

        for (int layer_index = 0; layer_index < cfg->dec_layers; layer_index++) {
            qwen_dec_layer_t *layer = &ctx->decoder.layers[layer_index];
            double layer_start = 0.0;
            float *prepared = (float *)malloc(total_bytes);

            if (!prepared) {
                clear_decoder_prefill_qkv_prepared(ctx);
                if (qwen_verbose >= 1) {
                    fprintf(stderr,
                            "decoder: runtime prepared prefill QKV skipped (alloc failed, %.1f MB target)\n",
                            (double)total_bytes * (double)cfg->dec_layers / (1024.0 * 1024.0));
                }
                return 0;
            }

            if (ctx->runtime_profile.decoder_layer_timing) {
                layer_start = qwen_perf_now_ms();
            }
            prepare_decoder_prefill_qkv_block(prepared, layer, q_count, kv_count);

            layer->prefill_qkv_prepared.f32_data = prepared;
            layer->prefill_qkv_prepared.rows = total_rows;
            layer->prefill_qkv_prepared.cols = (size_t)cfg->dec_hidden;
            layer->prefill_qkv_prepared.bytes = total_bytes;
            ctx->runtime_perf.decoder_prefill_qkv_bytes += total_bytes;
            ctx->runtime_perf.decoder_prefill_qkv_layers += 1;

            if (ctx->runtime_profile.decoder_layer_timing && qwen_verbose >= 2) {
                fprintf(stderr,
                        "decoder: prepared prefill QKV layer=%d rows=%zu cols=%zu ms=%.3f\n",
                        layer_index,
                        total_rows,
                        (size_t)cfg->dec_hidden,
                        qwen_perf_now_ms() - layer_start);
            }
        }

        ctx->runtime_perf.decoder_prefill_qkv_prepare_ms = qwen_perf_now_ms() - total_start;
        if (qwen_verbose >= 1 && ctx->runtime_perf.decoder_prefill_qkv_layers > 0) {
            fprintf(stderr,
                    "decoder: prepared prefill QKV layers=%d bytes=%.1f MB profile=%s ms=%.3f\n",
                    ctx->runtime_perf.decoder_prefill_qkv_layers,
                    (double)ctx->runtime_perf.decoder_prefill_qkv_bytes / (1024.0 * 1024.0),
                    qwen_runtime_profile_name(ctx->runtime_profile.kind),
                    ctx->runtime_perf.decoder_prefill_qkv_prepare_ms);
        }
    }

    if (prepare_gate_up) {
        const double total_start = qwen_perf_now_ms();
        clear_decoder_prefill_gate_up_prepared(ctx);

        for (int layer_index = 0; layer_index < cfg->dec_layers; layer_index++) {
            qwen_dec_layer_t *layer = &ctx->decoder.layers[layer_index];
            double layer_start = 0.0;
            float *prepared = (float *)malloc(gate_up_bytes);

            if (!prepared) {
                clear_decoder_prefill_gate_up_prepared(ctx);
                if (qwen_verbose >= 1) {
                    fprintf(stderr,
                            "decoder: runtime prepared prefill GateUp skipped (alloc failed, %.1f MB target)\n",
                            (double)gate_up_bytes * (double)cfg->dec_layers / (1024.0 * 1024.0));
                }
                return 0;
            }

            if (ctx->runtime_profile.decoder_layer_timing) {
                layer_start = qwen_perf_now_ms();
            }
            prepare_decoder_prefill_gate_up_block(prepared, layer, gate_up_count);

            layer->prefill_gate_up_prepared.f32_data = prepared;
            layer->prefill_gate_up_prepared.rows = (size_t)gate_up_dim;
            layer->prefill_gate_up_prepared.cols = (size_t)cfg->dec_hidden;
            layer->prefill_gate_up_prepared.bytes = gate_up_bytes;
            ctx->runtime_perf.decoder_prefill_gate_up_bytes += gate_up_bytes;
            ctx->runtime_perf.decoder_prefill_gate_up_layers += 1;

            if (ctx->runtime_profile.decoder_layer_timing && qwen_verbose >= 2) {
                fprintf(stderr,
                        "decoder: prepared prefill GateUp layer=%d rows=%d cols=%d ms=%.3f\n",
                        layer_index,
                        gate_up_dim,
                        cfg->dec_hidden,
                        qwen_perf_now_ms() - layer_start);
            }
        }

        ctx->runtime_perf.decoder_prefill_gate_up_prepare_ms = qwen_perf_now_ms() - total_start;
        if (qwen_verbose >= 1 && ctx->runtime_perf.decoder_prefill_gate_up_layers > 0) {
            fprintf(stderr,
                    "decoder: prepared prefill GateUp layers=%d bytes=%.1f MB profile=%s ms=%.3f\n",
                    ctx->runtime_perf.decoder_prefill_gate_up_layers,
                    (double)ctx->runtime_perf.decoder_prefill_gate_up_bytes / (1024.0 * 1024.0),
                    qwen_runtime_profile_name(ctx->runtime_profile.kind),
                    ctx->runtime_perf.decoder_prefill_gate_up_prepare_ms);
        }
    }

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(qwen_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qwen_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int ensure_prefill_buffers(qwen_ctx_t *ctx, int seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qwen_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int ensure_rope_cache(qwen_ctx_t *ctx, int required_pos, int head_dim, float theta) {
    if (required_pos <= ctx->rope_cache_cap) return 0;
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cache_cos, n * sizeof(float));
    if (!new_cos) return -1;
    ctx->rope_cache_cos = new_cos;

    float *new_sin = (float *)realloc(ctx->rope_cache_sin, n * sizeof(float));
    if (!new_sin) return -1;
    ctx->rope_cache_sin = new_sin;

    int half = head_dim / 2;
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * ctx->rope_inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int total_qkv_dim = q_dim + 2 * kv_dim;
    float *qkv_out_scratch = NULL;
    float *linear_weight_scratch = NULL;
    size_t linear_weight_scratch_count = 0;

    qwen_apply_prefill_thread_policy();

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    qwen_float_arena_reset(&ctx->prefill_scratch);
    if (seq_len > 1) {
        for (int layer = 0; layer < cfg->dec_layers; layer++) {
            const qwen_dec_layer_t *prepared = &dec->layers[layer];
            if (!prepared->prefill_qkv_prepared.f32_data) {
                const size_t qkv_weight_count = (size_t)total_qkv_dim * (size_t)dim;
                if (qkv_weight_count > linear_weight_scratch_count) {
                    linear_weight_scratch_count = qkv_weight_count;
                }
            }
            if (!prepared->prefill_gate_up_prepared.f32_data) {
                const size_t gate_up_weight_count = (size_t)(2 * intermediate) * (size_t)dim;
                if (gate_up_weight_count > linear_weight_scratch_count) {
                    linear_weight_scratch_count = gate_up_weight_count;
                }
            }
        }
        if ((size_t)dim * (size_t)q_dim > linear_weight_scratch_count) {
            linear_weight_scratch_count = (size_t)dim * (size_t)q_dim;
        }
        if ((size_t)dim * (size_t)intermediate > linear_weight_scratch_count) {
            linear_weight_scratch_count = (size_t)dim * (size_t)intermediate;
        }
        {
            size_t scratch_count = (size_t)seq_len * (size_t)total_qkv_dim;
            if (linear_weight_scratch_count > 0) {
                scratch_count += linear_weight_scratch_count;
            }
            if (qwen_float_arena_reserve(&ctx->prefill_scratch, scratch_count) != 0) return;
            qkv_out_scratch = qwen_float_arena_alloc(&ctx->prefill_scratch,
                                                     (size_t)seq_len * (size_t)total_qkv_dim);
            if (!qkv_out_scratch) return;
            if (linear_weight_scratch_count > 0) {
                linear_weight_scratch = qwen_float_arena_alloc(&ctx->prefill_scratch,
                                                               linear_weight_scratch_count);
                if (!linear_weight_scratch) return;
            }
        }
    }

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_cache(ctx, start_pos + seq_len, head_dim, theta) != 0) return;
    const float *rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        /* Input RMSNorm */
        qwen_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections (fused single GEMM) */
        if (ctx->runtime_profile.decoder_layer_timing) {
            const double qkv_start = qwen_perf_now_ms();
            if (seq_len > 1 && l->prefill_qkv_prepared.f32_data) {
                qwen_linear_nobias_qkv_f32_packed(q, k, v,
                                                  qkv_out_scratch,
                                                  x_norm,
                                                  l->prefill_qkv_prepared.f32_data,
                                                  seq_len, dim, q_dim, kv_dim);
            } else {
                qwen_linear_nobias_bf16_qkv_prefill(q, k, v,
                                                    qkv_out_scratch,
                                                    linear_weight_scratch,
                                                    x_norm,
                                                    l->wq_weight_bf16,
                                                    l->wk_weight_bf16,
                                                    l->wv_weight_bf16,
                                                    seq_len, dim, q_dim, kv_dim);
            }
            ctx->runtime_perf.decoder_prefill_qkv_ms += qwen_perf_now_ms() - qkv_start;
        } else if (seq_len > 1 && l->prefill_qkv_prepared.f32_data) {
            qwen_linear_nobias_qkv_f32_packed(q, k, v,
                                              qkv_out_scratch,
                                              x_norm,
                                              l->prefill_qkv_prepared.f32_data,
                                              seq_len, dim, q_dim, kv_dim);
        } else {
            qwen_linear_nobias_bf16_qkv_prefill(q, k, v,
                                                qkv_out_scratch,
                                                linear_weight_scratch,
                                                x_norm,
                                                l->wq_weight_bf16,
                                                l->wk_weight_bf16,
                                                l->wv_weight_bf16,
                                                seq_len, dim, q_dim, kv_dim);
        }

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        qwen_causal_attention(attn_out, q, full_k, full_v,
                               seq_len, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, start_pos);

        /* Output projection + residual */
        qwen_linear_nobias_bf16_scratch(proj_out, attn_out, l->wo_weight_bf16,
                        linear_weight_scratch,
                        seq_len, q_dim, dim);
        qwen_add_inplace(x, proj_out, seq_len * dim);

        /* Post-attention RMSNorm */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        /* SwiGLU MLP */
        if (ctx->runtime_profile.decoder_layer_timing) {
            const double gate_up_start = qwen_perf_now_ms();
            if (l->prefill_gate_up_prepared.f32_data) {
                qwen_linear_nobias(gate_up, x_norm, l->prefill_gate_up_prepared.f32_data,
                                   seq_len, dim, 2 * intermediate);
            } else {
                qwen_linear_nobias_bf16_scratch(gate_up, x_norm, l->gate_up_fused_bf16,
                                                linear_weight_scratch,
                                                seq_len, dim, 2 * intermediate);
            }
            ctx->runtime_perf.decoder_prefill_gate_up_ms += qwen_perf_now_ms() - gate_up_start;
        } else {
            if (l->prefill_gate_up_prepared.f32_data) {
                qwen_linear_nobias(gate_up, x_norm, l->prefill_gate_up_prepared.f32_data,
                                   seq_len, dim, 2 * intermediate);
            } else {
                qwen_linear_nobias_bf16_scratch(gate_up, x_norm, l->gate_up_fused_bf16,
                                                linear_weight_scratch,
                                                seq_len, dim, 2 * intermediate);
            }
        }
        qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
        qwen_linear_nobias_bf16_scratch(ffn_out, gate, l->down_weight_bf16,
                                        linear_weight_scratch,
                                        seq_len, intermediate, dim);

        qwen_add_inplace(x, ffn_out, seq_len * dim);

    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

static void ensure_dec_buffers(qwen_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    qwen_apply_decode_thread_policy();

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Grow KV cache if needed */
    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return QWEN_TOKEN_IM_END;
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        return QWEN_TOKEN_IM_END;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        /* Fused gate+up matvec: one pass over x_norm, output interleaved [g0,u0,g1,u1,...] */
        qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                 1, dim, 2 * intermediate);
        /* In-place for seq=1: gate_buf[0:inter] receives SwiGLU output. */
        qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
        qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
        qwen_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    return qwen_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}
