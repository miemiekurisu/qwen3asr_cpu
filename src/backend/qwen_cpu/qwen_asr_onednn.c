/*
 * qwen_asr_onednn.c - oneDNN INT8 decoder acceleration
 *
 * Provides:
 *   1. Per-row symmetric INT8 quantization of BF16 weights
 *   2. oneDNN matmul primitive wrappers for U8×S8→F32
 *   3. Decoder INT8 preparation (called from qwen_decoder_prepare_runtime)
 *
 * When USE_ONEDNN is not defined, all functions compile as stubs that
 * return failure codes so the caller falls back to the BF16 path.
 */

#include "qwen_asr_onednn.h"
#include "qwen_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern int qwen_verbose;

/* ========================================================================
 * INT8 Per-Row Symmetric Quantization (always available)
 * ======================================================================== */

static inline float bf16_to_f32_scalar(uint16_t bf) {
    uint32_t bits = ((uint32_t)bf) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int qwen_int8_quantize_bf16(qwen_int8_weight_t *dst,
                            const uint16_t *src_bf16,
                            size_t rows, size_t cols) {
    if (!dst || !src_bf16 || rows == 0 || cols == 0) return -1;

    size_t data_bytes = rows * cols;
    int8_t *data = (int8_t *)malloc(data_bytes);
    float *scales = (float *)malloc(rows * sizeof(float));
    if (!data || !scales) {
        free(data);
        free(scales);
        return -1;
    }

    for (size_t r = 0; r < rows; r++) {
        const uint16_t *row_src = src_bf16 + r * cols;
        float amax = 0.0f;

        /* Find absolute max of this row */
        for (size_t c = 0; c < cols; c++) {
            float v = bf16_to_f32_scalar(row_src[c]);
            float av = fabsf(v);
            if (av > amax) amax = av;
        }

        float scale = amax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;  /* avoid division by zero */
        scales[r] = scale;

        float inv_scale = 127.0f / amax;
        if (amax == 0.0f) inv_scale = 0.0f;

        int8_t *row_dst = data + r * cols;
        for (size_t c = 0; c < cols; c++) {
            float v = bf16_to_f32_scalar(row_src[c]);
            int32_t q = (int32_t)roundf(v * inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            row_dst[c] = (int8_t)q;
        }
    }

    dst->data = data;
    dst->row_scale = scales;
    dst->rows = rows;
    dst->cols = cols;
    dst->data_bytes = data_bytes;
    return 0;
}

void qwen_int8_weight_free(qwen_int8_weight_t *w) {
    if (!w) return;
    free(w->data);
    free(w->row_scale);
    w->data = NULL;
    w->row_scale = NULL;
    w->rows = 0;
    w->cols = 0;
    w->data_bytes = 0;
}

/* ========================================================================
 * oneDNN Implementation (when USE_ONEDNN is defined)
 * ======================================================================== */

#if defined(USE_ONEDNN)

#include <dnnl.h>

/* Global engine + stream (lazy-init) */
static dnnl_engine_t g_engine = NULL;
static dnnl_stream_t g_stream = NULL;
static int g_onednn_init_done = 0;
static int g_onednn_init_ok = 0;

#ifdef _WIN32
#include <windows.h>
static INIT_ONCE g_onednn_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK onednn_init_callback(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    dnnl_status_t s;
    s = dnnl_engine_create(&g_engine, dnnl_cpu, 0);
    if (s != dnnl_success) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "oneDNN: engine_create failed (%d)\n", (int)s);
        g_onednn_init_done = 1;
        return TRUE;
    }
    s = dnnl_stream_create(&g_stream, g_engine, dnnl_stream_default_flags);
    if (s != dnnl_success) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "oneDNN: stream_create failed (%d)\n", (int)s);
        dnnl_engine_destroy(g_engine);
        g_engine = NULL;
        g_onednn_init_done = 1;
        return TRUE;
    }
    g_onednn_init_ok = 1;
    g_onednn_init_done = 1;
    return TRUE;
}
#else
#include <pthread.h>
static pthread_once_t g_onednn_once = PTHREAD_ONCE_INIT;
static void onednn_init_impl(void) {
    dnnl_status_t s;
    s = dnnl_engine_create(&g_engine, dnnl_cpu, 0);
    if (s != dnnl_success) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "oneDNN: engine_create failed (%d)\n", (int)s);
        g_onednn_init_done = 1;
        return;
    }
    s = dnnl_stream_create(&g_stream, g_engine, dnnl_stream_default_flags);
    if (s != dnnl_success) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "oneDNN: stream_create failed (%d)\n", (int)s);
        dnnl_engine_destroy(g_engine);
        g_engine = NULL;
        g_onednn_init_done = 1;
        return;
    }
    g_onednn_init_ok = 1;
    g_onednn_init_done = 1;
}
#endif

int qwen_onednn_init(void) {
#ifdef _WIN32
    InitOnceExecuteOnce(&g_onednn_once, onednn_init_callback, NULL, NULL);
#else
    pthread_once(&g_onednn_once, onednn_init_impl);
#endif
    return g_onednn_init_ok ? 0 : -1;
}

void qwen_onednn_shutdown(void) {
    if (g_stream) { dnnl_stream_destroy(g_stream); g_stream = NULL; }
    if (g_engine) { dnnl_engine_destroy(g_engine); g_engine = NULL; }
    g_onednn_init_done = 0;
    g_onednn_init_ok = 0;
#ifdef _WIN32
    InitOnceInitialize(&g_onednn_once);
#else
    g_onednn_once = PTHREAD_ONCE_INIT;
#endif
}

/* ========================================================================
 * MatMul Primitive Handle
 *
 * We create a matmul: src(u8)[M,K] × wei(s8)[K,N] → dst(f32)[M,N]
 * with per-oc (per-output-channel = per-N) scales for dequantization
 * and a common source scale for dynamic activation quantization.
 *
 * The weight is stored as [rows=N, cols=K] (row-major), but oneDNN matmul
 * expects wei as [K, N]. We use the ba (transposed) tag.
 *
 * At execution time, src f32 is dynamically quantized to u8, and the
 * output is dequantized via: dst = src_scale * wei_scale[n] * (u8_src * s8_wei)
 * ======================================================================== */

struct qwen_onednn_matmul_s {
    dnnl_primitive_t primitive;
    dnnl_memory_t   src_mem;       /* u8, [M_max, K] */
    dnnl_memory_t   wei_mem;       /* s8, [K, N] (logically, ba layout) */
    dnnl_memory_t   dst_mem;       /* f32, [M_max, N] */
    dnnl_memory_t   wei_scales_mem;/* f32, [N] per-output-channel scales */
    dnnl_memory_t   src_scales_mem;/* f32, [1] common source scale */
    int K;                         /* input dim */
    int N;                         /* output dim (rows in original weight) */
    int M_cur;                     /* current allocated M */
    uint8_t *src_u8_buf;           /* scratch for quantized src */
    int src_u8_buf_size;           /* allocated size */
    float src_scale_val;           /* current src scale (written before execute) */
};

qwen_onednn_matmul_t *qwen_onednn_matmul_create(
    const qwen_int8_weight_t *weight)
{
    if (!weight || !weight->data || !weight->row_scale) return NULL;
    if (qwen_onednn_init() != 0) return NULL;

    int N = (int)weight->rows;     /* output dim */
    int K = (int)weight->cols;     /* input dim */
    int M = 1;                     /* default for decode; grows for prefill */

    dnnl_status_t s;
    qwen_onednn_matmul_t *h = (qwen_onednn_matmul_t *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->K = K;
    h->N = N;
    h->M_cur = M;

    /* Allocate u8 scratch buffer for source quantization */
    h->src_u8_buf_size = M * K;
    h->src_u8_buf = (uint8_t *)malloc((size_t)h->src_u8_buf_size);
    if (!h->src_u8_buf) { free(h); return NULL; }
    h->src_scale_val = 1.0f;

    /* Memory descriptors: u8(src) × s8(wei) → f32(dst)
     * Use DNNL_RUNTIME_DIM_VAL for M to support dynamic batch sizes. */
    dnnl_memory_desc_t src_md, wei_md, dst_md;
    dnnl_dims_t src_dims = {DNNL_RUNTIME_DIM_VAL, K};
    dnnl_dims_t wei_dims = {K, N};
    dnnl_dims_t dst_dims = {DNNL_RUNTIME_DIM_VAL, N};

    s = dnnl_memory_desc_create_with_tag(&src_md, 2, src_dims, dnnl_u8, dnnl_ab);
    if (s != dnnl_success) goto fail;

    /* Weight [K, N] in s8, stored as [N, K] row-major → use ba (transpose) tag */
    s = dnnl_memory_desc_create_with_tag(&wei_md, 2, wei_dims, dnnl_s8, dnnl_ba);
    if (s != dnnl_success) { dnnl_memory_desc_destroy(src_md); goto fail; }

    s = dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_f32, dnnl_ab);
    if (s != dnnl_success) { dnnl_memory_desc_destroy(src_md); dnnl_memory_desc_destroy(wei_md); goto fail; }

    /* Create primitive descriptor with scales for dequantization */
    dnnl_primitive_desc_t pd = NULL;
    dnnl_primitive_attr_t attr = NULL;
    s = dnnl_primitive_attr_create(&attr);
    if (s != dnnl_success) goto fail_md;

    /* Source scale: common (mask=0) — dynamic quantization scale */
    s = dnnl_primitive_attr_set_scales_mask(attr, DNNL_ARG_SRC, /* mask */ 0);
    if (s != dnnl_success) { dnnl_primitive_attr_destroy(attr); goto fail_md; }

    /* Weight scales: per-output-channel (mask=2 = bit 1 set for dim N) */
    s = dnnl_primitive_attr_set_scales_mask(attr, DNNL_ARG_WEIGHTS, /* mask */ 2);
    if (s != dnnl_success) { dnnl_primitive_attr_destroy(attr); goto fail_md; }

    /* Zero point for src (u8 has zero-point 128 for symmetric mapping) */
    s = dnnl_primitive_attr_set_zero_points_mask(attr, DNNL_ARG_SRC, /* mask */ 0);
    if (s != dnnl_success) { dnnl_primitive_attr_destroy(attr); goto fail_md; }

    s = dnnl_matmul_primitive_desc_create(&pd, g_engine, src_md, wei_md, NULL, dst_md, attr);
    dnnl_primitive_attr_destroy(attr);
    if (s != dnnl_success) goto fail_md;

    /* Create primitive */
    s = dnnl_primitive_create(&h->primitive, pd);
    dnnl_primitive_desc_destroy(pd);
    if (s != dnnl_success) goto fail_md;

    /* Create memory objects — need concrete dims for initial allocation */
    {
        dnnl_memory_desc_t concrete_src_md, concrete_dst_md;
        dnnl_dims_t c_src_dims = {M, K};
        dnnl_dims_t c_dst_dims = {M, N};
        s = dnnl_memory_desc_create_with_tag(&concrete_src_md, 2, c_src_dims, dnnl_u8, dnnl_ab);
        if (s != dnnl_success) goto fail_prim;
        s = dnnl_memory_desc_create_with_tag(&concrete_dst_md, 2, c_dst_dims, dnnl_f32, dnnl_ab);
        if (s != dnnl_success) { dnnl_memory_desc_destroy(concrete_src_md); goto fail_prim; }

        s = dnnl_memory_create(&h->src_mem, concrete_src_md, g_engine, h->src_u8_buf);
        dnnl_memory_desc_destroy(concrete_src_md);
        if (s != dnnl_success) { dnnl_memory_desc_destroy(concrete_dst_md); goto fail_prim; }

        s = dnnl_memory_create(&h->dst_mem, concrete_dst_md, g_engine, DNNL_MEMORY_ALLOCATE);
        dnnl_memory_desc_destroy(concrete_dst_md);
        if (s != dnnl_success) { dnnl_memory_destroy(h->src_mem); goto fail_prim; }
    }

    s = dnnl_memory_create(&h->wei_mem, wei_md, g_engine, (void *)weight->data);
    if (s != dnnl_success) { dnnl_memory_destroy(h->dst_mem); dnnl_memory_destroy(h->src_mem); goto fail_prim; }

    /* Weight scales memory: per-output-channel float32 scales */
    {
        dnnl_memory_desc_t wei_sc_md;
        dnnl_dims_t wei_sc_dims = {N};
        s = dnnl_memory_desc_create_with_tag(&wei_sc_md, 1, wei_sc_dims, dnnl_f32, dnnl_a);
        if (s != dnnl_success) goto fail_all_mem;
        s = dnnl_memory_create(&h->wei_scales_mem, wei_sc_md, g_engine, (void *)weight->row_scale);
        dnnl_memory_desc_destroy(wei_sc_md);
        if (s != dnnl_success) goto fail_all_mem;
    }

    /* Source scale memory: single float */
    {
        dnnl_memory_desc_t src_sc_md;
        dnnl_dims_t src_sc_dims = {1};
        s = dnnl_memory_desc_create_with_tag(&src_sc_md, 1, src_sc_dims, dnnl_f32, dnnl_a);
        if (s != dnnl_success) goto fail_wei_sc;
        s = dnnl_memory_create(&h->src_scales_mem, src_sc_md, g_engine, &h->src_scale_val);
        dnnl_memory_desc_destroy(src_sc_md);
        if (s != dnnl_success) goto fail_wei_sc;
    }

    /* Clean up memory descriptors */
    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(wei_md);
    dnnl_memory_desc_destroy(dst_md);

    return h;

fail_wei_sc:
    dnnl_memory_destroy(h->wei_scales_mem);
fail_all_mem:
    dnnl_memory_destroy(h->dst_mem);
    dnnl_memory_destroy(h->wei_mem);
    dnnl_memory_destroy(h->src_mem);
fail_prim:
    dnnl_primitive_destroy(h->primitive);
fail_md:
    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(wei_md);
    dnnl_memory_desc_destroy(dst_md);
fail:
    free(h->src_u8_buf);
    free(h);
    return NULL;
}

int qwen_onednn_matmul_execute(qwen_onednn_matmul_t *handle,
                               const float *src, int M,
                               float *dst)
{
    if (!handle || !src || !dst || M <= 0) return -1;

    dnnl_status_t s;
    int K = handle->K;
    int total = M * K;

    /* Ensure u8 scratch buffer is large enough */
    if (total > handle->src_u8_buf_size) {
        uint8_t *newbuf = (uint8_t *)realloc(handle->src_u8_buf, (size_t)total);
        if (!newbuf) return -1;
        handle->src_u8_buf = newbuf;
        handle->src_u8_buf_size = total;
    }

    /* Dynamic quantization: f32 → u8 with zero-point 128 (symmetric mapping)
     * u8_val = clamp(round(src_val / src_scale) + 128, 0, 255)
     * src_scale = max(|src|) / 127  (so that -127..+127 maps to 1..255) */
    float max_abs = 0.0f;
    for (int i = 0; i < total; ++i) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float src_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / src_scale;

    for (int i = 0; i < total; ++i) {
        float v = src[i] * inv_scale + 128.0f;
        int iv = (int)(v + 0.5f);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        handle->src_u8_buf[i] = (uint8_t)iv;
    }
    handle->src_scale_val = src_scale;

    /* If M changed, we need to update src/dst memory descriptors.
     * For decode (M=1 always) this is a no-op after first call. */
    if (M != handle->M_cur) {
        dnnl_memory_desc_t new_src_md, new_dst_md;
        dnnl_dims_t src_dims = {M, K};
        dnnl_dims_t dst_dims = {M, handle->N};

        s = dnnl_memory_desc_create_with_tag(&new_src_md, 2, src_dims, dnnl_u8, dnnl_ab);
        if (s != dnnl_success) return -1;
        s = dnnl_memory_desc_create_with_tag(&new_dst_md, 2, dst_dims, dnnl_f32, dnnl_ab);
        if (s != dnnl_success) { dnnl_memory_desc_destroy(new_src_md); return -1; }

        dnnl_memory_destroy(handle->src_mem);
        dnnl_memory_destroy(handle->dst_mem);
        dnnl_memory_create(&handle->src_mem, new_src_md, g_engine, handle->src_u8_buf);
        dnnl_memory_create(&handle->dst_mem, new_dst_md, g_engine, DNNL_MEMORY_NONE);
        dnnl_memory_desc_destroy(new_src_md);
        dnnl_memory_desc_destroy(new_dst_md);
        handle->M_cur = M;
    } else {
        /* Same M — just update data handle */
        dnnl_memory_set_data_handle(handle->src_mem, handle->src_u8_buf);
    }

    dnnl_memory_set_data_handle(handle->dst_mem, dst);
    dnnl_memory_set_data_handle(handle->src_scales_mem, &handle->src_scale_val);

    /* Zero-point memory for src (u8 zero-point = 128) */
    dnnl_memory_desc_t zp_md;
    dnnl_dims_t zp_dims = {1};
    s = dnnl_memory_desc_create_with_tag(&zp_md, 1, zp_dims, dnnl_s32, dnnl_a);
    if (s != dnnl_success) return -1;
    int32_t zp_val = 128;
    dnnl_memory_t zp_mem;
    s = dnnl_memory_create(&zp_mem, zp_md, g_engine, &zp_val);
    dnnl_memory_desc_destroy(zp_md);
    if (s != dnnl_success) return -1;

    /* Execute */
    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC,                                handle->src_mem},
        {DNNL_ARG_WEIGHTS,                             handle->wei_mem},
        {DNNL_ARG_DST,                                handle->dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,         handle->src_scales_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,     handle->wei_scales_mem},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,    zp_mem},
    };
    s = dnnl_primitive_execute(handle->primitive, g_stream, 6, args);
    if (s != dnnl_success) {
        dnnl_memory_destroy(zp_mem);
        return -1;
    }

    s = dnnl_stream_wait(g_stream);
    dnnl_memory_destroy(zp_mem);
    return (s == dnnl_success) ? 0 : -1;
}

void qwen_onednn_matmul_free(qwen_onednn_matmul_t *handle) {
    if (!handle) return;
    if (handle->src_scales_mem) dnnl_memory_destroy(handle->src_scales_mem);
    if (handle->wei_scales_mem) dnnl_memory_destroy(handle->wei_scales_mem);
    if (handle->dst_mem) dnnl_memory_destroy(handle->dst_mem);
    if (handle->wei_mem) dnnl_memory_destroy(handle->wei_mem);
    if (handle->src_mem) dnnl_memory_destroy(handle->src_mem);
    if (handle->primitive) dnnl_primitive_destroy(handle->primitive);
    free(handle->src_u8_buf);
    free(handle);
}

int qwen_int8_matvec(qwen_onednn_matmul_t *handle,
                     const float *x, int seq_len, float *y) {
    return qwen_onednn_matmul_execute(handle, x, seq_len, y);
}

/* ========================================================================
 * Decoder INT8 Preparation
 * ======================================================================== */

static void free_int8_dec_layer(qwen_int8_dec_layer_t *il) {
    if (!il) return;
    qwen_onednn_matmul_free(il->mm_wq);
    qwen_onednn_matmul_free(il->mm_wk);
    qwen_onednn_matmul_free(il->mm_wv);
    qwen_onednn_matmul_free(il->mm_wo);
    qwen_onednn_matmul_free(il->mm_gate_up);
    qwen_onednn_matmul_free(il->mm_down);
    qwen_int8_weight_free(&il->wq_int8);
    qwen_int8_weight_free(&il->wk_int8);
    qwen_int8_weight_free(&il->wv_int8);
    qwen_int8_weight_free(&il->wo_int8);
    qwen_int8_weight_free(&il->gate_up_int8);
    qwen_int8_weight_free(&il->down_int8);
    memset(il, 0, sizeof(*il));
}

int qwen_decoder_prepare_int8(void *ctx_ptr) {
    qwen_ctx_t *ctx = (qwen_ctx_t *)ctx_ptr;
    if (!ctx) return -1;

    const qwen_config_t *cfg = &ctx->config;
    const int q_dim  = cfg->dec_heads * cfg->dec_head_dim;
    const int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    const int intermediate = cfg->dec_intermediate;
    const int hidden = cfg->dec_hidden;

    if (qwen_onednn_init() != 0) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "decoder: INT8 preparation skipped (oneDNN init failed)\n");
        return -1;
    }

    double start_ms = qwen_perf_now_ms();

    /* Allocate int8 layer array */
    if (ctx->int8_dec_layers) {
        qwen_decoder_free_int8(ctx);
    }
    ctx->int8_dec_layers = calloc((size_t)cfg->dec_layers, sizeof(qwen_int8_dec_layer_t));
    if (!ctx->int8_dec_layers) return -1;
    ctx->n_int8_dec_layers = cfg->dec_layers;

    size_t total_int8_bytes = 0;

    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        qwen_int8_dec_layer_t *il = &((qwen_int8_dec_layer_t *)ctx->int8_dec_layers)[i];

        /* Quantize all projections */
        if (qwen_int8_quantize_bf16(&il->wq_int8, l->wq_weight_bf16,
                                    (size_t)q_dim, (size_t)hidden) != 0) goto fail;
        if (qwen_int8_quantize_bf16(&il->wk_int8, l->wk_weight_bf16,
                                    (size_t)kv_dim, (size_t)hidden) != 0) goto fail;
        if (qwen_int8_quantize_bf16(&il->wv_int8, l->wv_weight_bf16,
                                    (size_t)kv_dim, (size_t)hidden) != 0) goto fail;
        if (qwen_int8_quantize_bf16(&il->wo_int8, l->wo_weight_bf16,
                                    (size_t)hidden, (size_t)q_dim) != 0) goto fail;
        if (qwen_int8_quantize_bf16(&il->gate_up_int8, l->gate_up_fused_bf16,
                                    (size_t)(2 * intermediate), (size_t)hidden) != 0) goto fail;
        if (qwen_int8_quantize_bf16(&il->down_int8, l->down_weight_bf16,
                                    (size_t)hidden, (size_t)intermediate) != 0) goto fail;

        /* Create oneDNN primitives */
        il->mm_wq = qwen_onednn_matmul_create(&il->wq_int8);
        il->mm_wk = qwen_onednn_matmul_create(&il->wk_int8);
        il->mm_wv = qwen_onednn_matmul_create(&il->wv_int8);
        il->mm_wo = qwen_onednn_matmul_create(&il->wo_int8);
        il->mm_gate_up = qwen_onednn_matmul_create(&il->gate_up_int8);
        il->mm_down = qwen_onednn_matmul_create(&il->down_int8);

        if (!il->mm_wq || !il->mm_wk || !il->mm_wv ||
            !il->mm_wo || !il->mm_gate_up || !il->mm_down) {
            if (qwen_verbose >= 1)
                fprintf(stderr, "decoder: INT8 oneDNN primitive creation failed at layer %d\n", i);
            goto fail;
        }

        total_int8_bytes += il->wq_int8.data_bytes + il->wk_int8.data_bytes +
                            il->wv_int8.data_bytes + il->wo_int8.data_bytes +
                            il->gate_up_int8.data_bytes + il->down_int8.data_bytes;

        if (qwen_verbose >= 2) {
            fprintf(stderr, "decoder: INT8 layer %d prepared\n", i);
        }
    }

    double elapsed = qwen_perf_now_ms() - start_ms;
    if (qwen_verbose >= 1) {
        fprintf(stderr,
                "decoder: INT8 prepared layers=%d int8_bytes=%.1f MB ms=%.1f\n",
                cfg->dec_layers,
                (double)total_int8_bytes / (1024.0 * 1024.0),
                elapsed);
    }

    return 0;

fail:
    qwen_decoder_free_int8(ctx);
    return -1;
}

void qwen_decoder_free_int8(void *ctx_ptr) {
    qwen_ctx_t *ctx = (qwen_ctx_t *)ctx_ptr;
    if (!ctx || !ctx->int8_dec_layers) return;

    qwen_int8_dec_layer_t *layers = (qwen_int8_dec_layer_t *)ctx->int8_dec_layers;
    for (int i = 0; i < ctx->n_int8_dec_layers; i++) {
        free_int8_dec_layer(&layers[i]);
    }
    free(layers);
    ctx->int8_dec_layers = NULL;
    ctx->n_int8_dec_layers = 0;
}

#else /* !USE_ONEDNN */

/* ========================================================================
 * Stub implementations when oneDNN is not available
 * ======================================================================== */

int qwen_onednn_init(void) { return -1; }
void qwen_onednn_shutdown(void) {}

qwen_onednn_matmul_t *qwen_onednn_matmul_create(const qwen_int8_weight_t *w) {
    (void)w; return NULL;
}
int qwen_onednn_matmul_execute(qwen_onednn_matmul_t *h, const float *s, int M, float *d) {
    (void)h; (void)s; (void)M; (void)d; return -1;
}
void qwen_onednn_matmul_free(qwen_onednn_matmul_t *h) { (void)h; }

int qwen_int8_matvec(qwen_onednn_matmul_t *h, const float *x, int s, float *y) {
    (void)h; (void)x; (void)s; (void)y; return -1;
}

int qwen_decoder_prepare_int8(void *ctx) {
    (void)ctx;
    if (qwen_verbose >= 1)
        fprintf(stderr, "decoder: INT8 not available (compiled without oneDNN)\n");
    return -1;
}

void qwen_decoder_free_int8(void *ctx) { (void)ctx; }

#endif /* USE_ONEDNN */
