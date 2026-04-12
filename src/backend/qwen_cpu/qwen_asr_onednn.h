/*
 * qwen_asr_onednn.h - oneDNN INT8 decoder acceleration (optional)
 *
 * When USE_ONEDNN is defined, provides per-row absmax INT8 quantization of
 * decoder BF16 weights and oneDNN matmul primitives for decode (M=1) and
 * prefill (M>1) paths.
 *
 * When USE_ONEDNN is NOT defined, all functions are no-ops / stubs so the
 * rest of the codebase compiles unchanged.
 */

#ifndef QWEN_ASR_ONEDNN_H
#define QWEN_ASR_ONEDNN_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * INT8 Quantized Weight (per-row symmetric absmax)
 *
 * For a BF16 weight matrix W[rows, cols]:
 *   scale[r] = max(|W[r,:]|) / 127.0
 *   data[r * cols + c] = round(W_f32[r,c] / scale[r])
 * ======================================================================== */

typedef struct {
    int8_t *data;              /* [rows, cols] row-major INT8 weights */
    float  *row_scale;         /* [rows] per-row dequant scale */
    size_t  rows;
    size_t  cols;
    size_t  data_bytes;        /* rows * cols */
} qwen_int8_weight_t;

/* Quantize a BF16 weight matrix to per-row symmetric INT8.
 * Returns 0 on success, -1 on allocation failure. */
int qwen_int8_quantize_bf16(qwen_int8_weight_t *dst,
                            const uint16_t *src_bf16,
                            size_t rows, size_t cols);

/* Quantize an F32 weight matrix to per-row symmetric INT8.
 * Returns 0 on success, -1 on allocation failure. */
int qwen_int8_quantize_f32(qwen_int8_weight_t *dst,
                           const float *src_f32,
                           size_t rows, size_t cols);

/* Free quantized weight buffers. */
void qwen_int8_weight_free(qwen_int8_weight_t *w);

/* ========================================================================
 * oneDNN MatMul Primitive Handle
 *
 * Wraps a single dnnl_matmul primitive configured for:
 *   src: f32 [M, K]   (activation, dynamic M)
 *   wei: s8  [K, N]   (quantized weight, static)
 *   dst: f32 [M, N]   (output)
 *   + per-oc (per-column) scales for dequantization
 * ======================================================================== */

typedef struct qwen_onednn_matmul_s qwen_onednn_matmul_t;

/* Create a matmul primitive for INT8 weight.
 * K = weight cols (input dim), N = weight rows (output dim).
 * The weight data is reordered/packed by oneDNN internally.
 * Returns NULL on failure. */
qwen_onednn_matmul_t *qwen_onednn_matmul_create(
    const qwen_int8_weight_t *weight);

/* Execute: dst[M, N] = src[M, K] @ W_int8^T, with per-row dequant scales.
 * M can vary between calls (decode M=1, prefill M>1). */
int qwen_onednn_matmul_execute(qwen_onednn_matmul_t *handle,
                               const float *src, int M,
                               float *dst);

/* Free the primitive and all oneDNN resources. */
void qwen_onednn_matmul_free(qwen_onednn_matmul_t *handle);

/* ========================================================================
 * Per-Layer INT8 Decoder Weights
 * ======================================================================== */

typedef struct {
    /* Attention projections */
    qwen_int8_weight_t  wq_int8;
    qwen_int8_weight_t  wk_int8;
    qwen_int8_weight_t  wv_int8;
    qwen_int8_weight_t  wo_int8;

    /* MLP projections */
    qwen_int8_weight_t  gate_up_int8;    /* [2*intermediate, hidden] */
    qwen_int8_weight_t  down_int8;       /* [hidden, intermediate] */

    /* oneDNN matmul handles (NULL if not using oneDNN) */
    qwen_onednn_matmul_t *mm_wq;
    qwen_onednn_matmul_t *mm_wk;
    qwen_onednn_matmul_t *mm_wv;
    qwen_onednn_matmul_t *mm_wo;
    qwen_onednn_matmul_t *mm_gate_up;
    qwen_onednn_matmul_t *mm_down;
} qwen_int8_dec_layer_t;

/* Prepare INT8 weights + oneDNN primitives for all decoder layers.
 * Returns 0 on success, -1 on failure (caller should fall back to BF16). */
int qwen_decoder_prepare_int8(void *ctx_ptr);

/* Free all INT8 decoder resources for all layers. */
void qwen_decoder_free_int8(void *ctx_ptr);

/* ========================================================================
 * Per-Layer INT8 Encoder Weights
 * ======================================================================== */

typedef struct {
    /* Fused QKV [3*d_model, d_model] */
    qwen_int8_weight_t  qkv_int8;

    /* Output projection [d_model, d_model] */
    qwen_int8_weight_t  wo_int8;

    /* FFN fc1 [ffn_dim, d_model], fc2 [d_model, ffn_dim] */
    qwen_int8_weight_t  fc1_int8;
    qwen_int8_weight_t  fc2_int8;

    /* oneDNN matmul handles */
    qwen_onednn_matmul_t *mm_qkv;
    qwen_onednn_matmul_t *mm_wo;
    qwen_onednn_matmul_t *mm_fc1;
    qwen_onednn_matmul_t *mm_fc2;
} qwen_int8_enc_layer_t;

/* Prepare INT8 weights + oneDNN primitives for all encoder layers.
 * Returns 0 on success, -1 on failure (caller should fall back to F32). */
int qwen_encoder_prepare_int8(void *ctx_ptr);

/* Free all INT8 encoder resources for all layers. */
void qwen_encoder_free_int8(void *ctx_ptr);

/* Execute a single INT8 matmul (decode M=1 or general).
 * Returns 0 on success, -1 on failure. */
int qwen_int8_matvec(qwen_onednn_matmul_t *handle,
                     const float *x, int seq_len, float *y);

/* ========================================================================
 * Global oneDNN Engine (lazy-init singleton)
 * ======================================================================== */

/* Initialize the global oneDNN engine + stream. Thread-safe, idempotent.
 * Returns 0 on success, -1 on failure. */
int qwen_onednn_init(void);

/* Shutdown global oneDNN resources. */
void qwen_onednn_shutdown(void);

#endif /* QWEN_ASR_ONEDNN_H */
