/*
 * qwen_asr.h - Qwen3-ASR Pure C Inference Engine
 *
 * Supports both Qwen3-ASR-1.7B and Qwen3-ASR-0.6B models.
 */

#ifndef QWEN_ASR_H
#define QWEN_ASR_H

#include "qwen_asr_perf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
/* MSVC names POSIX functions with underscore prefix */
#ifndef strdup
#define strdup _strdup
#endif
#else
#include <pthread.h>
#endif

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_SAMPLE_RATE      16000
#define QWEN_MEL_BINS         128
#define QWEN_HOP_LENGTH       160
#define QWEN_WINDOW_SIZE      400
#define QWEN_VOCAB_SIZE       151936
#define QWEN_STREAM_MAX_NEW_TOKENS_LIMIT 128

/* Maximum layer counts (for static array sizing) */
#define QWEN_MAX_ENC_LAYERS   24
#define QWEN_MAX_DEC_LAYERS   28

/* Special token IDs */
#define QWEN_TOKEN_IM_START     151644
#define QWEN_TOKEN_IM_END       151645
#define QWEN_TOKEN_ENDOFTEXT    151643
#define QWEN_TOKEN_AUDIO_START  151669
#define QWEN_TOKEN_AUDIO_END    151670
#define QWEN_TOKEN_AUDIO_PAD    151676
#define QWEN_TOKEN_ASR_TEXT     151704
#define QWEN_TOKEN_TIMESTAMP    151705

/* Conv2D stem constants */
#define QWEN_CONV_HIDDEN      480
#define QWEN_CONV_KERNEL      3

/* ========================================================================
 * Model Configuration (populated from config.json)
 * ======================================================================== */

typedef struct {
    /* Audio encoder */
    int enc_d_model;           /* 1024 or 896 */
    int enc_layers;            /* 24 or 18 */
    int enc_heads;             /* 16 or 14 */
    int enc_head_dim;          /* 64 */
    int enc_ffn_dim;           /* 4096 or 3584 */
    int enc_output_dim;        /* 2048 or 1024 */
    int enc_n_window;          /* 50 */
    int enc_n_window_infer;    /* 800 */
    int enc_chunk_size;        /* n_window * 2 = 100 */
    int enc_conv_proj_dim;     /* CONV_HIDDEN * 16 = 7680 */

    /* LLM decoder */
    int dec_hidden;            /* 2048 or 1024 */
    int dec_layers;            /* 28 */
    int dec_heads;             /* 16 */
    int dec_kv_heads;          /* 8 */
    int dec_head_dim;          /* 128 */
    int dec_intermediate;      /* 6144 or 3072 */
    int vocab_size;            /* 151936 (ASR) or 152064 (Aligner) */
    float dec_rms_norm_eps;    /* 1e-6 */
    float dec_rope_theta;      /* 1e6 */

    /* ForcedAligner-specific (zero for ASR models) */
    int classify_num;          /* 5000 (Aligner) or 0 (ASR) */
    int timestamp_token_id;    /* 151705 (Aligner) or 0 (ASR) */
    float timestamp_segment_time; /* 80.0 ms (Aligner) or 0 (ASR) */
    int tie_word_embeddings;   /* 1 (ASR) or 0 (Aligner) */
} qwen_config_t;

/* ========================================================================
 * Audio Encoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (ALL have biases) - pre-converted to f32 */
    float *wq_weight;          /* [d_model, d_model] */
    float *wq_bias;            /* [d_model] */
    float *wk_weight;          /* [d_model, d_model] */
    float *wk_bias;            /* [d_model] */
    float *wv_weight;          /* [d_model, d_model] */
    float *wv_bias;            /* [d_model] */
    float *qkv_weight_packed;  /* [3*d_model, d_model], owned packed QKV block */
    float *qkv_bias_packed;    /* [3*d_model], owned packed QKV bias block */
    float *wo_weight;          /* [d_model, d_model] */
    float *wo_bias;            /* [d_model] */

    /* Pre-attention LayerNorm (with bias) */
    float *attn_norm_weight;   /* [d_model] */
    float *attn_norm_bias;     /* [d_model] */

    /* FFN: GELU(fc1(x)) -> fc2 (ALL have biases) - pre-converted to f32 */
    float *fc1_weight;         /* [ffn_dim, d_model] */
    float *fc1_bias;           /* [ffn_dim] */
    float *fc2_weight;         /* [d_model, ffn_dim] */
    float *fc2_bias;           /* [d_model] */

    /* Pre-FFN LayerNorm (with bias) */
    float *ffn_norm_weight;    /* [d_model] */
    float *ffn_norm_bias;      /* [d_model] */
} qwen_enc_layer_t;

typedef struct {
    /* Conv2D stem (3 layers, each 3x3, stride 2) */
    float *conv1_weight;       /* [480, 1, 3, 3] */
    float *conv1_bias;         /* [480] */
    float *conv2_weight;       /* [480, 480, 3, 3] */
    float *conv2_bias;         /* [480] */
    float *conv3_weight;       /* [480, 480, 3, 3] */
    float *conv3_bias;         /* [480] */

    /* Conv output projection - pre-converted to f32 */
    float *conv_out_weight;    /* [d_model, 7680] */

    /* Transformer layers */
    qwen_enc_layer_t layers[QWEN_MAX_ENC_LAYERS];

    /* Final LayerNorm */
    float *ln_post_weight;     /* [d_model] */
    float *ln_post_bias;       /* [d_model] */

    /* Projection layers - pre-converted to f32 */
    float *proj1_weight;       /* [d_model, d_model] */
    float *proj1_bias;         /* [d_model] */
    float *proj2_weight;       /* [output_dim, d_model] */
    float *proj2_bias;         /* [output_dim] */
} qwen_encoder_t;

/* ========================================================================
 * LLM Decoder Layer
 * ======================================================================== */

typedef struct {
    float *f32_data;
    size_t rows;
    size_t cols;
    size_t bytes;
} qwen_prepared_f32_weight_t;

typedef struct {
    /* Self-attention (NO biases in decoder) */
    uint16_t *wq_weight_bf16;  /* [n_heads*head_dim, hidden] */
    uint16_t *wk_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wv_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wo_weight_bf16;  /* [hidden, n_heads*head_dim] */

    /* Per-head Q/K RMSNorm */
    float *q_norm_weight;      /* [head_dim] = [128] */
    float *k_norm_weight;      /* [head_dim] = [128] */

    /* RMSNorm (no bias) */
    float *input_norm;         /* [hidden] */
    float *post_attn_norm;     /* [hidden] */

    /* SwiGLU MLP (NO biases) */
    uint16_t *gate_weight_bf16; /* [intermediate, hidden] */
    uint16_t *up_weight_bf16;   /* [intermediate, hidden] */
    uint16_t *down_weight_bf16; /* [hidden, intermediate] */

    /* Fused gate+up weight for single-token matvec [2*intermediate, hidden] */
    uint16_t *gate_up_fused_bf16;

    qwen_prepared_f32_weight_t prefill_qkv_prepared;
    qwen_prepared_f32_weight_t prefill_gate_up_prepared;
} qwen_dec_layer_t;

typedef struct {
    /* Token embeddings (tied with lm_head for ASR) */
    uint16_t *tok_embeddings_bf16; /* [vocab_size, hidden] */

    /* Separate lm_head (Aligner only; NULL when tied) */
    uint16_t *lm_head_bf16;    /* [classify_num, hidden] or NULL */

    /* Suffix max-abs for argmax early termination.
     * tok_embed_suffix_max[r] = max over rows r..N-1 of max(|W[row][k]|).
     * Owned, NULL when compute failed. */
    float *tok_embed_suffix_max;     /* [vocab_size] */
    float *lm_head_suffix_max;       /* [classify_num] or NULL */

    /* Transformer layers */
    qwen_dec_layer_t layers[QWEN_MAX_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;               /* [hidden] */
} qwen_decoder_t;

/* ========================================================================
 * Token Callback (streaming output)
 * ======================================================================== */

/* Called for each decoded text token during autoregressive generation.
 * 'piece' is the decoded token string (UTF-8). */
typedef void (*qwen_token_cb)(const char *piece, void *userdata);

/* Per-chunk snapshot delivered once after each streaming decode step.
 *
 * The C-side stream_impl already implements the industrial-grade
 * rollback / tail-stable-promote / cross-chunk-dup-suppress pipeline.
 * Most UIs need to surface BOTH the just-committed text and the still
 * tentative tail in real time, but a per-token callback forces the
 * caller to treat every token as "stable" (which is the source of the
 * "front-end flicker" bug).  This callback gives the caller the natural
 * unit of streaming display: one chunk's worth of new stable text plus
 * the current tentative tail.
 *
 * Lifetime: 'stable_piece' and 'tentative_piece' are valid only for the
 * duration of the callback.  If the receiver needs to keep them, it
 * must copy.  The strings are owned by the streaming context.
 *
 * 'stable_piece'        UTF-8 text newly promoted to committed in this
 *                        chunk (already stripped of any tentative tail
 *                        and cross-chunk duplicates).  May be empty.
 * 'tentative_piece'     UTF-8 text that the decoder is still working on
 *                        and which may be revised or rolled back on the
 *                        next chunk.  Empty on the final chunk (EOF).
 * 'audio_cursor'        Total number of input samples consumed up to the
 *                        end of this chunk (0 for non-live mode).
 * 'decode_ms'           Wall time spent inside the decode for this chunk,
 *                        excluding the background encoder overlap.
 * 'is_first' / 'is_final' are set on the first and last chunk.
 */
typedef struct qwen_stream_chunk_t {
    int chunk_index;
    int is_first;
    int is_final;
    const char *stable_piece;
    const char *tentative_piece;
    int stable_token_count;
    int tentative_token_count;
    int64_t audio_cursor;
    double decode_ms;
} qwen_stream_chunk_t;

typedef void (*qwen_stream_chunk_cb_t)(const qwen_stream_chunk_t *chunk,
                                        void *userdata);

/* Called to decide whether the current transcription should stop early.
 * Return non-zero to cancel the active run. */
typedef int (*qwen_cancel_cb)(void *userdata);

/* Called after each segment is transcribed in segmented mode.
 * Receives the segment index (0-based), text, start/end in seconds. */
typedef void (*qwen_segment_cb)(int index, const char *text,
                                float start_sec, float end_sec,
                                void *userdata);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    double decoder_prefill_qkv_prepare_ms;
    double decoder_prefill_qkv_ms;
    double decoder_prefill_gate_up_prepare_ms;
    double decoder_prefill_gate_up_ms;
    double decoder_prefill_attn_ms;
    double decoder_prefill_wo_ms;
    double decoder_prefill_down_ms;
    size_t decoder_prefill_qkv_bytes;
    size_t decoder_prefill_gate_up_bytes;
    int decoder_prefill_qkv_layers;
    int decoder_prefill_gate_up_layers;
} qwen_runtime_perf_t;

typedef struct {
    qwen_config_t config;
    qwen_encoder_t encoder;
    qwen_decoder_t decoder;
    qwen_runtime_profile_config_t runtime_profile;
    qwen_runtime_perf_t runtime_perf;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];
    int owns_model_data;

    /* KV cache for decoder */
    float *kv_cache_k;         /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_up, *dec_ffn_out;
    float *dec_rope_cos, *dec_rope_sin;

    /* Persistent decoder prefill buffers (multi-token prefill) */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;
    qwen_float_arena_t prefill_scratch;

    /* Cached RoPE tables for decoder positions */
    float *rope_cache_cos, *rope_cache_sin;   /* [pos, head_dim] */
    float *rope_inv_freq;                     /* [head_dim / 2] */
    int rope_cache_cap;                       /* cached positions */
    int rope_inv_freq_half;                   /* cached half-dim */

    /* Token streaming callback (optional) */
    qwen_token_cb token_cb;
    void *token_cb_userdata;

    /* Per-chunk streaming callback (optional).  Fires once per chunk after
     * the rollback / tail-stable / dup-suppress pipeline has decided which
     * tokens are committed and which are still tentative.  See
     * qwen_stream_chunk_t for fields.  Stable / tentative pieces are owned
     * by the streaming context and only valid for the duration of the
     * callback.  Reuse a single ctx-level scratch buffer to avoid per-
     * chunk allocations. */
    qwen_stream_chunk_cb_t chunk_cb;
    void *chunk_cb_userdata;
    char *chunk_stable_buf;       /* heap, grown on demand */
    size_t chunk_stable_cap;
    char *chunk_tentative_buf;
    size_t chunk_tentative_cap;

    /* Cooperative cancellation callback (optional) */
    qwen_cancel_cb cancel_cb;
    void *cancel_cb_userdata;
    int last_run_cancelled;

    /* Per-segment callback (optional, for incremental subtitle output) */
    qwen_segment_cb segment_cb;
    void *segment_cb_userdata;

    /* Segmentation settings */
    float segment_sec;             /* 0 = no splitting, default full-audio decode */
    float search_sec;              /* segment-cutting silence search window ± seconds (default 3) */

    /* Streaming settings */
    float stream_chunk_sec;        /* chunk interval in seconds (default 2.0) */
    int stream_rollback;           /* tokens to roll back per chunk (default 5) */
    int stream_unfixed_chunks;     /* cold-start chunks without prefix (default 2) */
    int stream_max_new_tokens;     /* max generated tokens per streaming step (default 32) */
    int stream_idle_flush_ms;      /* live mode: wake the decode loop every N ms while
                                    * waiting for a full chunk so a buffered tail can be
                                    * decoded promptly when the speaker pauses. 0 disables
                                    * (legacy infinite wait, only EOF flushes the tail). */
    float stream_idle_flush_min_sec; /* minimum buffered audio (seconds) before an idle
                                    * timeout will trigger a partial-chunk decode. */
    int stream_idle_flush_max_new_tokens; /* when an idle tail-flush decode fires
                                    * (stream_idle_flush_ms timed out with no new audio),
                                    * override the per-step max_new_tokens to this value
                                    * to force the model to commit its current hypothesis
                                    * instead of "imagining" more tokens. 0 disables
                                    * (always use stream_max_new_tokens).  Default 16. */
    int past_text_conditioning;    /* 1=enable past text conditioning in -S/--stream (default: off).
                                    * In segmented mode, this also enables boundary cleanup/post-processing. */
    int skip_silence;              /* 1=drop long silent spans before transcription */

    /* Optional prompt/language controls */
    char *prompt;                  /* system prompt text (UTF-8) */
    char *force_language;          /* normalized language name, or NULL */
    int *prompt_tokens;            /* cached token ids for prompt text */
    int n_prompt_tokens;
    int *force_prompt_tokens;      /* cached token ids for "language X" + <asr_text> */
    int n_force_prompt_tokens;
    int prompt_tokens_ready;       /* cache valid flag */

    /* Per-run performance stats (populated by last transcription call) */
    double perf_total_ms;          /* end-to-end inference time in milliseconds */
    int perf_text_tokens;          /* emitted text tokens (after <asr_text>) */
    double perf_audio_ms;          /* input audio duration in milliseconds */
    double perf_encode_ms;         /* mel + encoder time in milliseconds */
    double perf_decode_ms;         /* decoder prefill + decode time in milliseconds */

    /* Temperature sampling for decoder (used by batch fallback).
     * When decode_temperature > 0, decoder_forward uses top-k sampling
     * instead of greedy argmax to break repetition loops. */
    float decode_temperature;      /* 0 = greedy (default) */
    float *dec_logits_buf;         /* lazily allocated [vocab_size] */
    unsigned int sample_rng_state; /* LCG state for sampling */

    /* Repetition penalty for decoder.  Standard CTRL-paper approach:
     * logit > 0 → logit /= penalty;  logit < 0 → logit *= penalty.
     * Applied to tokens recently generated (stored in ring buffer).
     * Value of 1.0 = disabled (default), >1.0 = penalize repeats. */
    float decode_repetition_penalty; /* 1.0 = none (default) */
    #define QWEN_REP_PEN_RING_SIZE 256
    int rep_pen_ring[QWEN_REP_PEN_RING_SIZE]; /* ring buffer of recent token IDs */
    int rep_pen_ring_pos;            /* next write position in ring */
    int rep_pen_ring_count;          /* valid entries (up to RING_SIZE) */

    /* INT8 decoder acceleration (optional, via oneDNN) */
    int decoder_int8;              /* 0=disabled (default), 1=enabled */
    void *int8_dec_layers;         /* qwen_int8_dec_layer_t[] or NULL */
    int n_int8_dec_layers;         /* number of valid entries */

    /* INT8 encoder acceleration (optional, via oneDNN) */
    int encoder_int8;              /* 0=disabled (default), 1=enabled */
    void *int8_enc_layers;         /* qwen_int8_dec_layer_t[] or NULL */
    int n_int8_enc_layers;         /* number of valid entries */

    /* Optional Silero VAD for endpoint detection.  NULL = disabled
     * (use legacy timeout-based detection).  Owned by ctx; destroyed
     * in qwen_free.  See qwen_silero_vad.h. */
    struct qwen_silero_vad_t *vad;
    /* Count of consecutive frames where VAD prob < threshold.  When
     * this exceeds a small N, the speaker is considered to have
     * stopped and the streaming loop forces a partial-chunk decode
     * (tail-flush).  Reset on every chunk. */
    int vad_silence_run;
    /* Last speech prob reported by the VAD.  -1 = never updated. */
    float vad_last_prob;
} qwen_ctx_t;

/* ========================================================================
 * Live Audio (incremental stdin streaming)
 * ======================================================================== */

typedef struct {
    /* Written by reader thread under mutex */
    float *samples;
    int64_t sample_offset;      /* global index of samples[0] */
    int64_t n_samples;          /* number of valid samples in buffer */
    int64_t capacity;           /* allocated capacity (in samples) */
    int eof;
    /* Updated by stream_impl decoder thread after each chunk decode.
     * Readers should acquire mutex before reading. */
    int64_t decoded_cursor;     /* audio sample position up to which decode completed */
#ifdef _WIN32
    CRITICAL_SECTION mutex;
    CONDITION_VARIABLE cond;
    HANDLE thread;
#else
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
#endif
} qwen_live_audio_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
qwen_ctx_t *qwen_load(const char *model_dir);

/* Clone a context for an independent decode session while sharing read-only model data.
 * The source context must outlive the clone. */
qwen_ctx_t *qwen_clone_shared(const qwen_ctx_t *src);

/* Trim a UTF-8 byte string to the last complete character boundary.
 * Scans backwards from the end; if the trailing bytes are a partial
 * multi-byte sequence, truncates them.  Returns the new length in
 * *out_len (or 0 if input is empty).  The string is mutated in place
 * and re-terminated with NUL.  No-op for inputs <= 1 byte.
 *
 * Why we need this: Qwen3's BPE token boundaries do not align with
 * UTF-8 character boundaries — a single Chinese character can span 1-3
 * BPE tokens, each carrying a partial byte sequence.  When the decoder
 * cuts mid-character (VAD early-stop, max_new_tokens cap, or recovery
 * reset), the resulting string has a partial UTF-8 tail that renders
 * as garbled bytes in the UI.  Trimming to the last complete char
 * boundary removes the garbled tail and lets the next decode re-emit
 * the rest of the character. */
size_t qwen_utf8_truncate(char *s, size_t len);

/* Free all resources */
void qwen_free(qwen_ctx_t *ctx);

/* Internal runtime preparation step used after decoder weights are loaded. */
int qwen_decoder_prepare_runtime(qwen_ctx_t *ctx);

/* Enable or disable INT8 decoder acceleration (requires oneDNN).
 * Must be called after qwen_load(). Quantizes weights on first enable.
 * enable=0 frees INT8 resources and reverts to BF16 path.
 * Returns 0 on success, -1 on failure (remains on BF16 path). */
int qwen_set_decoder_int8(qwen_ctx_t *ctx, int enable);

/* Enable or disable INT8 encoder acceleration (requires oneDNN).
 * Must be called after qwen_load(). Quantizes F32 encoder weights to INT8.
 * enable=0 frees INT8 resources and reverts to F32 path.
 * Returns 0 on success, -1 on failure (remains on F32 path). */
int qwen_set_encoder_int8(qwen_ctx_t *ctx, int enable);

/* Set a callback to receive each decoded token as it's generated.
 * Set cb=NULL to disable. The callback is invoked during transcription. */
void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata);

/* Set a callback used to cooperatively cancel an active transcription.
 * Set cb=NULL to disable. */
void qwen_set_cancel_callback(qwen_ctx_t *ctx, qwen_cancel_cb cb, void *userdata);

/* Set a callback invoked after each segment finishes in segmented transcription.
 * Set cb=NULL to disable. */
void qwen_set_segment_callback(qwen_ctx_t *ctx, qwen_segment_cb cb, void *userdata);

/* Set a per-chunk callback.  See qwen_stream_chunk_t above for semantics.
 * This is the preferred way to drive a real-time UI: every chunk, the
 * callback receives both the just-committed text and the still-tentative
 * tail, so the UI can show a confirmed line + a "typing" line.
 *
 * If BOTH token_cb and chunk_cb are set, both are fired: token_cb once
 * per token (legacy / debugging), chunk_cb once per chunk (UI).  The
 * two are not redundant because the C-side rollback logic only operates
 * at chunk boundaries; the per-token callback cannot know which tokens
 * are "tentative" without the per-chunk view.
 *
 * Set cb=NULL to disable. */
void qwen_set_chunk_callback(qwen_ctx_t *ctx,
                             qwen_stream_chunk_cb_t cb, void *userdata);

/* Returns non-zero if the most recent transcription exited due to cancellation. */
int qwen_was_cancelled(const qwen_ctx_t *ctx);

/* Set optional system prompt text (UTF-8). Pass NULL or "" to clear.
 * Returns 0 on success, -1 on allocation/encoding errors. */
int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt);

/* Set optional forced language. Pass NULL or "" to clear.
 * Returns 0 on success, -1 if language is unsupported. */
int qwen_set_force_language(qwen_ctx_t *ctx, const char *language);

/* Comma-separated supported language names for --language. */
const char *qwen_supported_languages_csv(void);

/* ========================================================================
 * Segmented Transcription (with per-segment timestamps)
 * ======================================================================== */

/* A single timed segment returned by segmented transcription. */
typedef struct {
    char  *text;        /* malloc'd UTF-8 text for this segment (caller frees) */
    float  start_sec;   /* segment start in seconds from audio begin */
    float  end_sec;     /* segment end in seconds from audio begin */
} qwen_timed_segment_t;

/* Result of segmented transcription: array of timed segments.
 * Caller must free via qwen_segment_result_free(). */
typedef struct {
    qwen_timed_segment_t *segments;   /* array, length = n_segments */
    int                    n_segments;
} qwen_segment_result_t;

/* Transcribe raw audio and return per-segment timestamps.
 * Pre: ctx must be a loaded context, samples mono float32 16kHz.
 * Post: caller owns result, must free via qwen_segment_result_free().
 * Returns NULL on fatal error. */
qwen_segment_result_t *qwen_transcribe_audio_segmented(qwen_ctx_t *ctx,
                                                        const float *samples,
                                                        int n_samples);

/* Free a segment result and all its text strings. */
void qwen_segment_result_free(qwen_segment_result_t *result);

/* Transcribe a WAV file, returns allocated string (caller must free) */
char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path);

/* Transcribe from raw audio samples (mono float32, 16kHz) */
char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Transcribe from stdin (auto-detect WAV or raw s16le) */
char *qwen_transcribe_stdin(qwen_ctx_t *ctx);

/* Streaming transcription: process audio in chunks with prefix rollback.
 * Re-encodes growing audio and uses previous text as decoder context.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Live streaming transcription from an incrementally-filled audio source.
 * The streaming loop waits for new data instead of terminating at EOF.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Audio encoder forward pass */
float *qwen_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames,
                             int *out_seq_len);

/* Decoder prefill (multiple tokens) */
void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len);

/* KV cache shift: correct RoPE on K entries and move positions after eviction */
void qwen_kv_cache_shift(qwen_ctx_t *ctx, int prefix_keep, int shift);

/* Decoder forward (single token, uses KV cache, returns greedy token) */
int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed);

/* Global verbose flag */
extern int qwen_verbose;

/* Monitor mode: show inline Unicode symbols on stderr for streaming diagnostics.
 * Symbols: ▶ encoder  · prefill  ▪ decode  ▸ slow decode  ⟳ window eviction */
extern int qwen_monitor;

/* ========================================================================
 * Forced Alignment (NAR classifier mode)
 * ======================================================================== */

/* Result of forced alignment for a single word/character. */
typedef struct {
    char *text;       /* malloc'd UTF-8 word text (caller frees) */
    float start_sec;  /* word start time in seconds */
    float end_sec;    /* word end time in seconds */
} qwen_aligned_word_t;

/* Result of forced alignment for an audio+text pair.
 * Caller must free via qwen_align_result_free(). */
typedef struct {
    qwen_aligned_word_t *words;  /* array, length = n_words */
    int n_words;
} qwen_align_result_t;

/* Run forced alignment: given audio samples and transcript text, produce
 * word-level timestamps using a loaded ForcedAligner model.
 * Pre: ctx loaded from a ForcedAligner model dir (classify_num > 0).
 *      samples: mono float32 16kHz. text: UTF-8 transcript.
 *      language: "chinese", "english", etc.
 * Post: caller owns result, must free via qwen_align_result_free().
 * Returns NULL on error. */
qwen_align_result_t *qwen_forced_align(qwen_ctx_t *ctx,
                                       const float *samples, int n_samples,
                                       const char *text,
                                       const char *language);

/* Free an alignment result and all its text strings. */
void qwen_align_result_free(qwen_align_result_t *result);

#endif /* QWEN_ASR_H */
