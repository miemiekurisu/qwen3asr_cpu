/*
 * qwen_asr_aligner.c - Forced alignment via NAR classifier
 *
 * Implements word-level timestamp alignment using Qwen3-ForcedAligner-0.6B.
 * Reuses the same encoder, decoder layer, and kernel infrastructure as ASR.
 *
 * Key differences from ASR inference:
 *   - Non-autoregressive: single prefill pass, no token-by-token decode
 *   - Separate lm_head [classify_num, hidden] (not tied with embeddings)
 *   - Output: per-timestamp-position argmax -> class_id * 80ms
 *   - Input: <audio_start><audio_pad*N><audio_end>word1<ts><ts>word2<ts><ts>...
 */

#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_tokenizer.h"
#include "qwen_asr_perf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * UTF-8 Utilities
 * ======================================================================== */

/* Decode one UTF-8 codepoint from *p, advance *p past it. Returns 0 on end/error. */
static uint32_t utf8_next(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    if (!*s) return 0;
    uint32_t cp;
    int len;
    if (s[0] < 0x80)      { cp = s[0]; len = 1; }
    else if (s[0] < 0xE0) { cp = s[0] & 0x1F; len = 2; }
    else if (s[0] < 0xF0) { cp = s[0] & 0x0F; len = 3; }
    else                   { cp = s[0] & 0x07; len = 4; }
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) { *p = (const char *)(s + 1); return 0xFFFD; }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    *p = (const char *)(s + len);
    return cp;
}

static int is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x4E00  && cp <= 0x9FFF)   /* CJK Unified */
        || (cp >= 0x3400  && cp <= 0x4DBF)   /* Extension A */
        || (cp >= 0x20000 && cp <= 0x2A6DF)  /* Extension B */
        || (cp >= 0x2A700 && cp <= 0x2B73F)  /* Extension C */
        || (cp >= 0x2B740 && cp <= 0x2B81F)  /* Extension D */
        || (cp >= 0x2B820 && cp <= 0x2CEAF)  /* Extension E */
        || (cp >= 0xF900  && cp <= 0xFAFF);  /* Compatibility */
}

/* Check if a Unicode codepoint is a letter (L*) or digit (N*) or apostrophe.
 * Simplified version of Python unicodedata.category check. */
static int is_kept_char(uint32_t cp) {
    if (cp == '\'') return 1;
    /* ASCII letters and digits */
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
        (cp >= '0' && cp <= '9')) return 1;
    /* CJK */
    if (is_cjk_codepoint(cp)) return 1;
    /* Hangul */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;
    /* Hiragana/Katakana */
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;
    /* Latin extended, Cyrillic, Arabic, Devanagari, etc. — broad letter ranges */
    if (cp >= 0x00C0 && cp <= 0x024F) return 1; /* Latin Extended */
    if (cp >= 0x0400 && cp <= 0x04FF) return 1; /* Cyrillic */
    if (cp >= 0x0600 && cp <= 0x06FF) return 1; /* Arabic */
    if (cp >= 0x0900 && cp <= 0x097F) return 1; /* Devanagari */
    if (cp >= 0x3000 && cp <= 0x303F) return 0; /* CJK Symbols — NOT kept */
    if (cp >= 0xFF00 && cp <= 0xFF60) return 0; /* Fullwidth punctuation — NOT kept */
    /* Fullwidth letters/digits */
    if (cp >= 0xFF21 && cp <= 0xFF3A) return 1;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return 1;
    if (cp >= 0xFF10 && cp <= 0xFF19) return 1;
    /* General: if above basic ASCII printable and not already matched, assume kept
     * for other scripts (Thai, Greek, Hebrew, etc.) */
    if (cp > 0x00FF && cp < 0x2000) return 1;
    return 0;
}

/* Encode a codepoint to UTF-8 into buf. Returns number of bytes written. */
static int utf8_encode(char *buf, uint32_t cp) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/* ========================================================================
 * Word Splitting (mirrors Python Qwen3ForceAlignProcessor)
 * ======================================================================== */

typedef struct {
    char **words;   /* array of malloc'd UTF-8 strings */
    int n_words;
    int capacity;
} word_list_t;

static void word_list_init(word_list_t *wl) {
    wl->words = NULL;
    wl->n_words = 0;
    wl->capacity = 0;
}

static void word_list_push(word_list_t *wl, const char *word) {
    if (wl->n_words >= wl->capacity) {
        int new_cap = wl->capacity ? wl->capacity * 2 : 32;
        wl->words = (char **)realloc(wl->words, (size_t)new_cap * sizeof(char *));
        wl->capacity = new_cap;
    }
    wl->words[wl->n_words++] = strdup(word);
}

static void word_list_free(word_list_t *wl) {
    for (int i = 0; i < wl->n_words; i++)
        free(wl->words[i]);
    free(wl->words);
    wl->words = NULL;
    wl->n_words = 0;
    wl->capacity = 0;
}

/* Clean a token: keep only letter/digit/apostrophe characters. */
static char *clean_token(const char *token) {
    char buf[512];
    int pos = 0;
    const char *p = token;
    while (*p) {
        uint32_t cp = utf8_next(&p);
        if (cp == 0) break;
        if (is_kept_char(cp)) {
            int n = utf8_encode(buf + pos, cp);
            pos += n;
            if (pos >= 500) break;
        }
    }
    buf[pos] = '\0';
    return strdup(buf);
}

/* Split a segment: CJK characters become individual words,
 * non-CJK characters are grouped together. */
static void split_with_cjk(word_list_t *wl, const char *seg) {
    char buf[256];
    int buf_pos = 0;
    const char *p = seg;
    while (*p) {
        const char *prev = p;
        uint32_t cp = utf8_next(&p);
        if (cp == 0) break;
        if (is_cjk_codepoint(cp)) {
            /* Flush non-CJK buffer */
            if (buf_pos > 0) {
                buf[buf_pos] = '\0';
                word_list_push(wl, buf);
                buf_pos = 0;
            }
            /* CJK character as individual word */
            char cjk_buf[8];
            int n = utf8_encode(cjk_buf, cp);
            cjk_buf[n] = '\0';
            word_list_push(wl, cjk_buf);
        } else {
            int byte_len = (int)(p - prev);
            if (buf_pos + byte_len < 250) {
                memcpy(buf + buf_pos, prev, (size_t)byte_len);
                buf_pos += byte_len;
            }
        }
    }
    if (buf_pos > 0) {
        buf[buf_pos] = '\0';
        word_list_push(wl, buf);
    }
}

/* Split text into word list (space-delimited languages + CJK per-character). */
static void tokenize_text(word_list_t *wl, const char *text) {
    /* Split by whitespace */
    const char *p = text;
    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;
        /* Collect segment until next whitespace */
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        int seg_len = (int)(p - start);
        char *seg = (char *)malloc((size_t)(seg_len + 1));
        memcpy(seg, start, (size_t)seg_len);
        seg[seg_len] = '\0';
        /* Clean */
        char *cleaned = clean_token(seg);
        free(seg);
        if (cleaned[0] != '\0') {
            split_with_cjk(wl, cleaned);
        }
        free(cleaned);
    }
}

/* ========================================================================
 * Timestamp Post-Processing (LIS-based monotonicity fix)
 * ======================================================================== */

/* Fix non-monotonic timestamps using Longest Increasing Subsequence.
 * Modifies data[] in-place. */
static void fix_timestamps(int *data, int n) {
    if (n <= 1) return;

    int *dp = (int *)malloc((size_t)n * sizeof(int));
    int *parent = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) { dp[i] = 1; parent[i] = -1; }

    /* Compute LIS */
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    int max_len = 0, max_idx = 0;
    for (int i = 0; i < n; i++) {
        if (dp[i] > max_len) { max_len = dp[i]; max_idx = i; }
    }

    /* Mark normal (in-LIS) positions */
    int *is_normal = (int *)calloc((size_t)n, sizeof(int));
    {
        int idx = max_idx;
        while (idx != -1) {
            is_normal[idx] = 1;
            idx = parent[idx];
        }
    }

    /* Interpolate anomalous positions */
    int i = 0;
    while (i < n) {
        if (!is_normal[i]) {
            int j = i;
            while (j < n && !is_normal[j]) j++;
            int anomaly_count = j - i;

            /* Find left and right normal values */
            int left_val = -1, right_val = -1;
            for (int k = i - 1; k >= 0; k--) {
                if (is_normal[k]) { left_val = data[k]; break; }
            }
            for (int k = j; k < n; k++) {
                if (is_normal[k]) { right_val = data[k]; break; }
            }

            if (anomaly_count <= 2) {
                for (int k = i; k < j; k++) {
                    if (left_val < 0)
                        data[k] = right_val;
                    else if (right_val < 0)
                        data[k] = left_val;
                    else
                        data[k] = (k - (i - 1) <= j - k) ? left_val : right_val;
                }
            } else {
                if (left_val >= 0 && right_val >= 0) {
                    float step = (float)(right_val - left_val) / (float)(anomaly_count + 1);
                    for (int k = i; k < j; k++)
                        data[k] = left_val + (int)(step * (float)(k - i + 1));
                } else if (left_val >= 0) {
                    for (int k = i; k < j; k++) data[k] = left_val;
                } else if (right_val >= 0) {
                    for (int k = i; k < j; k++) data[k] = right_val;
                }
            }
            i = j;
        } else {
            i++;
        }
    }

    free(dp);
    free(parent);
    free(is_normal);
}

/* ========================================================================
 * Embedding Helper
 * ======================================================================== */

static void tok_embed(float *dst, const uint16_t *emb_bf16, int token_id, int dim) {
    const uint16_t *src = emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

/* ========================================================================
 * NAR Forward: Forced Alignment
 * ======================================================================== */

qwen_align_result_t *qwen_forced_align(qwen_ctx_t *ctx,
                                       const float *samples, int n_samples,
                                       const char *text,
                                       const char *language) {
    const qwen_config_t *cfg = &ctx->config;
    if (cfg->classify_num <= 0) {
        fprintf(stderr, "qwen_forced_align: model is not a ForcedAligner (classify_num=0)\n");
        return NULL;
    }

    int dim = cfg->dec_hidden; /* = enc_output_dim for Aligner */
    int ts_token = cfg->timestamp_token_id;
    float ts_seg_ms = cfg->timestamp_segment_time;
    int classify_num = cfg->classify_num;

    /* ---- 1. Split text into words ---- */
    word_list_t wl;
    word_list_init(&wl);
    tokenize_text(&wl, text);
    if (wl.n_words == 0) {
        fprintf(stderr, "qwen_forced_align: no words after tokenization\n");
        return NULL;
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Aligner: %d words from text\n", wl.n_words);

    /* ---- 2. Tokenize each word, build token sequence ---- */
    char vocab_path[600];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", ctx->model_dir);
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(vocab_path);
    if (!tokenizer) {
        fprintf(stderr, "qwen_forced_align: cannot load tokenizer from %s\n", vocab_path);
        word_list_free(&wl);
        return NULL;
    }

    /* Build token ID sequence: <audio_start> [audio_pad * N] <audio_end> word_tokens... */
    int *token_ids = NULL;
    int n_token_ids = 0;
    int token_cap = 0;
    int *ts_positions = NULL;  /* positions of timestamp tokens in token_ids */
    int n_ts_positions = 0;
    int ts_cap = 0;

    #define PUSH_TOKEN(tid) do { \
        if (n_token_ids >= token_cap) { \
            token_cap = token_cap ? token_cap * 2 : 256; \
            token_ids = (int *)realloc(token_ids, (size_t)token_cap * sizeof(int)); \
        } \
        token_ids[n_token_ids++] = (tid); \
    } while (0)

    #define PUSH_TS_POS(pos) do { \
        if (n_ts_positions >= ts_cap) { \
            ts_cap = ts_cap ? ts_cap * 2 : 64; \
            ts_positions = (int *)realloc(ts_positions, (size_t)ts_cap * sizeof(int)); \
        } \
        ts_positions[n_ts_positions++] = (pos); \
    } while (0)

    /* Audio prefix: <audio_start> (audio_pad will be filled by encoder later) */
    PUSH_TOKEN(QWEN_TOKEN_AUDIO_START);
    int audio_pad_start = 1;  /* index where audio_pad tokens begin */

    /* Placeholder: we'll know enc_seq_len after running encoder.
     * For now, insert a marker. We'll expand later. */

    /* Tokenize each word and interleave timestamps */
    /* First, prepare the text part token IDs */
    int *text_part_ids = NULL;
    int n_text_part = 0;
    int text_part_cap = 0;
    int *text_ts_markers = NULL; /* 1 if position is a timestamp token */

    #define PUSH_TEXT(tid, is_ts) do { \
        if (n_text_part >= text_part_cap) { \
            text_part_cap = text_part_cap ? text_part_cap * 2 : 256; \
            text_part_ids = (int *)realloc(text_part_ids, (size_t)text_part_cap * sizeof(int)); \
            text_ts_markers = (int *)realloc(text_ts_markers, (size_t)text_part_cap * sizeof(int)); \
        } \
        text_part_ids[n_text_part] = (tid); \
        text_ts_markers[n_text_part] = (is_ts); \
        n_text_part++; \
    } while (0)

    for (int w = 0; w < wl.n_words; w++) {
        /* Tokenize word */
        int n_word_tokens = 0;
        int *word_tokens = qwen_tokenizer_encode(tokenizer, wl.words[w], &n_word_tokens);
        if (word_tokens && n_word_tokens > 0) {
            for (int t = 0; t < n_word_tokens; t++)
                PUSH_TEXT(word_tokens[t], 0);
        }
        free(word_tokens);
        /* Two timestamp tokens after each word */
        PUSH_TEXT(ts_token, 1);
        PUSH_TEXT(ts_token, 1);
    }
    qwen_tokenizer_free(tokenizer);

    /* ---- 3. Compute mel + encoder ---- */
    double t0 = qwen_perf_now_ms();
    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) {
        fprintf(stderr, "qwen_forced_align: mel spectrogram failed\n");
        word_list_free(&wl);
        free(text_part_ids);
        free(text_ts_markers);
        free(token_ids);
        free(ts_positions);
        return NULL;
    }

    int enc_seq_len = 0;
    float *enc_output = qwen_encoder_forward(ctx, mel, mel_frames, &enc_seq_len);
    free(mel);
    if (!enc_output) {
        fprintf(stderr, "qwen_forced_align: encoder forward failed\n");
        word_list_free(&wl);
        free(text_part_ids);
        free(text_ts_markers);
        free(token_ids);
        free(ts_positions);
        return NULL;
    }

    double encode_ms = qwen_perf_now_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Aligner encode: %d frames -> %d tokens (%.0f ms)\n",
                mel_frames, enc_seq_len, encode_ms);

    /* ---- 4. Build full token ID sequence now that we know enc_seq_len ---- */
    n_token_ids = 0;  /* reset */
    PUSH_TOKEN(QWEN_TOKEN_AUDIO_START);
    for (int i = 0; i < enc_seq_len; i++)
        PUSH_TOKEN(QWEN_TOKEN_AUDIO_PAD);
    PUSH_TOKEN(QWEN_TOKEN_AUDIO_END);

    int text_start_pos = n_token_ids;
    for (int i = 0; i < n_text_part; i++) {
        if (text_ts_markers[i])
            PUSH_TS_POS(n_token_ids);
        PUSH_TOKEN(text_part_ids[i]);
    }
    free(text_part_ids);
    free(text_ts_markers);

    int total_seq = n_token_ids;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Aligner sequence: %d tokens (%d audio + %d text, %d timestamps)\n",
                total_seq, enc_seq_len, n_text_part, n_ts_positions);

    /* ---- 5. Build input embeddings ---- */
    float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    if (!input_embeds) {
        free(enc_output);
        word_list_free(&wl);
        free(token_ids);
        free(ts_positions);
        return NULL;
    }

    /* audio_start embed */
    tok_embed(input_embeds, ctx->decoder.tok_embeddings_bf16,
              QWEN_TOKEN_AUDIO_START, dim);

    /* Replace audio_pad positions with encoder output */
    for (int i = 0; i < enc_seq_len; i++) {
        memcpy(input_embeds + (size_t)(1 + i) * dim,
               enc_output + (size_t)i * dim,
               (size_t)dim * sizeof(float));
    }
    free(enc_output);

    /* audio_end embed */
    tok_embed(input_embeds + (size_t)(1 + enc_seq_len) * dim,
              ctx->decoder.tok_embeddings_bf16,
              QWEN_TOKEN_AUDIO_END, dim);

    /* Text token embeds */
    for (int i = text_start_pos; i < total_seq; i++) {
        tok_embed(input_embeds + (size_t)i * dim,
                  ctx->decoder.tok_embeddings_bf16,
                  token_ids[i], dim);
    }
    free(token_ids);

    /* ---- 6. Decoder prefill (NAR: all positions at once) ---- */
    t0 = qwen_perf_now_ms();
    ctx->kv_cache_len = 0;  /* Reset KV cache */
    qwen_decoder_prefill(ctx, input_embeds, total_seq);

    /* After prefill, ctx->pref_x contains hidden states [total_seq, dim].
     * Apply final RMSNorm to all positions. */
    float *hidden = ctx->pref_x;
    qwen_rms_norm(hidden, hidden, ctx->decoder.norm, total_seq, dim, cfg->dec_rms_norm_eps);

    double prefill_ms = qwen_perf_now_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Aligner prefill + norm: %.0f ms\n", prefill_ms);

    free(input_embeds);

    /* ---- 7. Extract timestamps from timestamp positions ---- */
    int *raw_timestamps = (int *)malloc((size_t)n_ts_positions * sizeof(int));
    if (!raw_timestamps) {
        word_list_free(&wl);
        free(ts_positions);
        return NULL;
    }

    t0 = qwen_perf_now_ms();
    for (int i = 0; i < n_ts_positions; i++) {
        int pos = ts_positions[i];
        float *h = hidden + (size_t)pos * dim;
        /* Apply lm_head [classify_num, hidden] -> argmax */
        int class_id = qwen_argmax_matvec_bf16(h, ctx->decoder.lm_head_bf16,
                                                dim, classify_num);
        raw_timestamps[i] = class_id;
    }
    free(ts_positions);

    double lmhead_ms = qwen_perf_now_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Aligner lm_head: %d positions (%.0f ms)\n",
                n_ts_positions, lmhead_ms);

    /* ---- 8. Fix timestamp monotonicity ---- */
    fix_timestamps(raw_timestamps, n_ts_positions);

    /* ---- 9. Convert to result ---- */
    qwen_align_result_t *result = (qwen_align_result_t *)calloc(1, sizeof(qwen_align_result_t));
    result->n_words = wl.n_words;
    result->words = (qwen_aligned_word_t *)calloc((size_t)wl.n_words,
                                                   sizeof(qwen_aligned_word_t));

    for (int w = 0; w < wl.n_words; w++) {
        int start_ts = raw_timestamps[w * 2];
        int end_ts   = raw_timestamps[w * 2 + 1];
        result->words[w].text = strdup(wl.words[w]);
        result->words[w].start_sec = (float)start_ts * ts_seg_ms / 1000.0f;
        result->words[w].end_sec   = (float)end_ts   * ts_seg_ms / 1000.0f;
    }

    free(raw_timestamps);
    word_list_free(&wl);

    if (qwen_verbose >= 1) {
        double total_ms = encode_ms + prefill_ms + lmhead_ms;
        float audio_sec = (float)n_samples / 16000.0f;
        fprintf(stderr, "Aligner: %d words, %.1fs audio, %.0f ms total "
                "(encode=%.0f prefill=%.0f lmhead=%.0f)\n",
                result->n_words, audio_sec, total_ms,
                encode_ms, prefill_ms, lmhead_ms);
    }

    return result;
}

void qwen_align_result_free(qwen_align_result_t *result) {
    if (!result) return;
    for (int i = 0; i < result->n_words; i++)
        free(result->words[i].text);
    free(result->words);
    free(result);
}
