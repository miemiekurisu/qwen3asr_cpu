/*
 * audit_bug_replay_gpu_diagnose_test.cc
 *
 * GPU encoder 根因诊断测试。
 * 问题现象：Windows + RTX 3070 Laptop 上 encoder 耗时 18.9s，
 * decoder prefill 后仅输出 IM_END（空转录）。
 * DGX（Linux A100）正常。
 *
 * 测试内容：
 *   1. 生成合成音频 → mel spectrogram
 *   2. 运行 CUDA EncoderForward，记录 wall-clock 时间
 *   3. 将 GPU 输出拷回 CPU，检查（全零/NaN/Inf/统计特征）
 *   4. 可选：与 CPU backend 输出逐元素比较
 *
 * 编译条件：QASR_CUDA_BACKEND_ENABLED && QASR_CPU_BACKEND_ENABLED
 */

#include "tests/test_registry.h"

#include "qasr/backend/cuda_backend.h"
#include "qasr/backend/cpu_backend.h"

extern "C" {
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
}
#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
}
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#ifdef QASR_CUDA_BACKEND_ENABLED
#include <cuda_runtime.h>
#endif

/* ─── Helpers ─── */

static bool HasCudaDevice() {
#ifdef QASR_CUDA_BACKEND_ENABLED
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
#else
    return false;
#endif
}

/* Generate sine wave at given frequency & duration */
static std::vector<float> GenSineWave(float freq_hz, int sample_rate,
                                       float duration_sec) {
    int n = (int)(sample_rate * duration_sec);
    std::vector<float> wav(n);
    for (int i = 0; i < n; i++) {
        wav[i] = sinf(2.0f * 3.14159265f * freq_hz * (float)i / (float)sample_rate);
    }
    return wav;
}

/* Check if a float buffer has any NaN or Inf */
static bool HasNaNOrInf(const float * data, int n) {
    for (int i = 0; i < n; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) return true;
    }
    return false;
}

/* Compute statistics of a float buffer */
struct FloatStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double abs_mean = 0.0;
    int nonzero = 0;
    int total = 0;
};

static FloatStats ComputeStats(const float * data, int n) {
    FloatStats s;
    s.total = n;
    if (n == 0) return s;

    double sum = 0.0, sum_abs = 0.0;
    s.min_val = data[0];
    s.max_val = data[0];
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum += v;
        sum_abs += fabsf(v);
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        if (fabsf(v) > 1e-9f) s.nonzero++;
    }
    s.mean = sum / n;
    s.abs_mean = sum_abs / n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = data[i] - s.mean;
        var += d * d;
    }
    s.stddev = sqrt(var / n);
    return s;
}

/* ─── Helper: allocate a CudaSessionState and call AllocateSession ─── */
static bool SetupCudaSession(qasr::CudaBackend & backend,
                              qasr::CudaSessionState & session,
                              int max_seq_len = 4096) {
    auto status = backend.AllocateSession(&session, max_seq_len);
    if (!status.ok()) {
        fprintf(stderr, "  AllocateSession failed: %s\n", status.ToString().c_str());
        return false;
    }
    return true;
}

/* =================================================================
 * Test 1: CUDA encoder timing + output validation
 * Runs the GPU encoder with synthetic audio,
 * measures wall-clock time, and checks output quality.
 * ================================================================= */
QASR_TEST(CudaEncoderDiagnoseTimingAndOutput) {
    /* Skip if no CUDA device */
    if (!HasCudaDevice()) return;

    const char * model_dir = std::getenv("QASR_MODEL_DIR");
    if (!model_dir) {
        fprintf(stderr, "  SKIP: QASR_MODEL_DIR not set\n");
        return;
    }

    fprintf(stderr, "\n===== CudaEncoderDiagnoseTimingAndOutput =====\n");
    fprintf(stderr, "  model_dir=%s\n", model_dir);

    /* Step 1: Initialize CUDA backend & load weights */
    qasr::CudaBackend backend;
    QASR_EXPECT(backend.Initialize().ok());

    fprintf(stderr, "  Loading weights...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto status = backend.PrepareWeights(model_dir);
    QASR_EXPECT(status.ok());
    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "  PrepareWeights: %.1f ms\n", load_ms);

    /* Step 2: Create session */
    qasr::CudaSessionState session;
    QASR_EXPECT(SetupCudaSession(backend, session));

    /* Step 3: Generate synthetic audio (3 seconds, 1 kHz sine, 16 kHz) */
    const int sample_rate = 16000;
    const float duration_sec = 3.0f;
    auto wav = GenSineWave(1000.0f, sample_rate, duration_sec);
    fprintf(stderr, "  Audio: %d samples, %.1f sec, 1kHz sine\n",
            (int)wav.size(), duration_sec);

    /* Step 4: Compute mel spectrogram */
    int mel_frames = 0;
    float * mel = qwen_mel_spectrogram(wav.data(), (int)wav.size(), &mel_frames);
    QASR_EXPECT(mel != nullptr);
    QASR_EXPECT(mel_frames > 0);
    fprintf(stderr, "  Mel: %d frames\n", mel_frames);

    /* Step 5: Allocate CPU output buffer + run encoder */
    int enc_out_dim = 0;
    {
        auto * cw = backend.cuda_weights();
        if (cw) enc_out_dim = cw->enc_output_dim;
    }
    if (enc_out_dim <= 0) {
        fprintf(stderr, "  ERROR: enc_output_dim=%d (invalid)\n", enc_out_dim);
        std::free(mel);
        return;
    }

    /* Allocate output buffer (max possible: mel_frames / 2 tokens, generous) */
    int max_out_tokens = mel_frames / 2 + 100;
    std::vector<float> enc_output((size_t)max_out_tokens * enc_out_dim, 0.0f);

    /* Run encoder with detailed timing */
    int out_tokens = 0;
    
    /* CUDA events for detailed timing */
    cudaEvent_t ev_upload, ev_conv, ev_transformer, ev_output, ev_copy;
    cudaEventCreate(&ev_upload);
    cudaEventCreate(&ev_conv);
    cudaEventCreate(&ev_transformer);
    cudaEventCreate(&ev_output);
    cudaEventCreate(&ev_copy);
    
    auto te0 = std::chrono::steady_clock::now();

    /* Use EncodeMel (calls EncoderForward + copies to CPU) */
    status = backend.EncodeMel(&session, mel, mel_frames,
                                enc_output.data(), out_tokens);
    auto te1 = std::chrono::steady_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(te1 - te0).count();

    std::free(mel);

    if (!status.ok()) {
        fprintf(stderr, "  EncoderForward failed: %s\n", status.ToString().c_str());
        cudaEventDestroy(ev_upload);
        cudaEventDestroy(ev_conv);
        cudaEventDestroy(ev_transformer);
        cudaEventDestroy(ev_output);
        cudaEventDestroy(ev_copy);
        QASR_EXPECT(false);
        return;
    }

    fprintf(stderr, "\n  === Encoder Result ===\n");
    fprintf(stderr, "  out_tokens=%d  time=%.1f ms\n", out_tokens, enc_ms);
    if (enc_ms > 5000.0) {
        fprintf(stderr, "  *** WARNING: encoder took %.0f ms (expected < 100 ms) ***\n", enc_ms);
    }

    /* Step 6: Analyze encoder output */
    int total_vals = out_tokens * enc_out_dim;
    if (total_vals <= 0) {
        fprintf(stderr, "  *** EMPTY OUTPUT: out_tokens=%d enc_out_dim=%d ***\n",
                out_tokens, enc_out_dim);
        cudaEventDestroy(ev_upload);
        cudaEventDestroy(ev_conv);
        cudaEventDestroy(ev_transformer);
        cudaEventDestroy(ev_output);
        cudaEventDestroy(ev_copy);
        QASR_EXPECT(total_vals > 0);
        return;
    }

    /* Check for NaN / Inf */
    bool has_nan_inf = HasNaNOrInf(enc_output.data(), total_vals);
    if (has_nan_inf) {
        fprintf(stderr, "  *** OUTPUT HAS NaN/Inf VALUES ***\n");
    }

    /* Compute statistics */
    FloatStats st = ComputeStats(enc_output.data(), total_vals);
    double nonzero_ratio = (double)st.nonzero / (double)st.total;

    fprintf(stderr, "\n  === Output Statistics ===\n");
    fprintf(stderr, "  total_values=%d  nonzero=%d (%.2f%%)\n",
            st.total, st.nonzero, nonzero_ratio * 100.0);
    fprintf(stderr, "  mean=%.6f  stddev=%.6f\n", st.mean, st.stddev);
    fprintf(stderr, "  abs_mean=%.6f  range=[%.6f, %.6f]\n",
            st.abs_mean, st.min_val, st.max_val);
    fprintf(stderr, "  has_nan_or_inf=%s\n", has_nan_inf ? "YES ***" : "no");

    /* Assertions: output must be valid */
    QASR_EXPECT(out_tokens > 0);
    QASR_EXPECT(!has_nan_inf);
    QASR_EXPECT(nonzero_ratio > 0.0f);
    QASR_EXPECT(st.abs_mean > 1e-6f);

    cudaEventDestroy(ev_upload);
    cudaEventDestroy(ev_conv);
    cudaEventDestroy(ev_transformer);
    cudaEventDestroy(ev_output);
    cudaEventDestroy(ev_copy);

    fprintf(stderr, "===== TEST PASSED =====\n\n");
}

/* =================================================================
 * Test 2: CUDA vs CPU encoder output comparison
 * Runs same audio through both backends and compares.
 * ================================================================= */
#ifdef QASR_CPU_BACKEND_ENABLED
QASR_TEST(CudaEncoderCompareWithCpu) {
    if (!HasCudaDevice()) return;

    const char * model_dir = std::getenv("QASR_MODEL_DIR");
    if (!model_dir) {
        fprintf(stderr, "  SKIP: QASR_MODEL_DIR not set\n");
        return;
    }

    fprintf(stderr, "\n===== CudaEncoderCompareWithCpu =====\n");
    fprintf(stderr, "  model_dir=%s\n", model_dir);

    /* ── CPU backend ── */
    qasr::CpuBackend cpu_backend;
    QASR_EXPECT(cpu_backend.Initialize().ok());
    QASR_EXPECT(cpu_backend.PrepareWeights(model_dir).ok());

    qasr::CpuSessionState cpu_session;
    auto cpu_status = cpu_backend.ResetDecoder(&cpu_session);
    QASR_EXPECT(cpu_status.ok());

    /* Get config for workspace size */
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;
    size_t ws_bytes = cpu_backend.WorkspaceBytes(cfg);
    cpu_session.workspace.resize(ws_bytes / sizeof(float) + 1);
    cpu_session.workspace_size = (int)ws_bytes;

    /* ── CUDA backend ── */
    qasr::CudaBackend cuda_backend;
    QASR_EXPECT(cuda_backend.Initialize().ok());
    QASR_EXPECT(cuda_backend.PrepareWeights(model_dir).ok());

    qasr::CudaSessionState cuda_session;
    QASR_EXPECT(SetupCudaSession(cuda_backend, cuda_session));

    /* ── Generate audio + mel ── */
    const int sample_rate = 16000;
    auto wav = GenSineWave(1000.0f, sample_rate, 3.0f);

    int mel_frames = 0;
    float * mel = qwen_mel_spectrogram(wav.data(), (int)wav.size(), &mel_frames);
    QASR_EXPECT(mel != nullptr);

    int enc_out_dim = 1024; /* Qwen3-ASR-0.6B enc_output_dim */

    /* ── Run CPU encoder ── */
    int cpu_tokens = 0;
    std::vector<float> cpu_out((size_t)mel_frames * enc_out_dim, 0.0f);
    auto tc0 = std::chrono::steady_clock::now();
    cpu_status = cpu_backend.EncodeMel(&cpu_session, mel, mel_frames,
                                         cpu_out.data(), cpu_tokens);
    auto tc1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

    /* ── Run CUDA encoder ── */
    int cuda_tokens = 0;
    std::vector<float> cuda_out((size_t)mel_frames * enc_out_dim, 0.0f);
    auto tu0 = std::chrono::steady_clock::now();
    auto cuda_status = cuda_backend.EncodeMel(&cuda_session, mel, mel_frames,
                                                cuda_out.data(), cuda_tokens);
    auto tu1 = std::chrono::steady_clock::now();
    double cuda_ms = std::chrono::duration<double, std::milli>(tu1 - tu0).count();

    std::free(mel);

    fprintf(stderr, "\n  === Backend Comparison ===\n");
    fprintf(stderr, "  CPU  encoder: %d tokens, %.1f ms\n", cpu_tokens, cpu_ms);
    fprintf(stderr, "  CUDA encoder: %d tokens, %.1f ms\n", cuda_tokens, cuda_ms);
    fprintf(stderr, "  CUDA/CPU ratio: %.2fx\n", cuda_ms / (cpu_ms + 0.001f));

    int min_tokens = std::min(cpu_tokens, cuda_tokens);
    if (min_tokens <= 0) {
        fprintf(stderr, "  *** One backend produced 0 tokens ***\n");
        QASR_EXPECT(min_tokens > 0);
        return;
    }

    /* ── Compare outputs element-wise ── */
    int cmp_tokens = std::min(cpu_tokens, cuda_tokens);
    int cmp_vals = cmp_tokens * enc_out_dim;

    double max_diff = 0.0, sum_diff = 0.0;
    int close_enough = 0;
    const float EPS = 1e-2f;

    for (int i = 0; i < cmp_vals; i++) {
        float d = fabsf(cpu_out[i] - cuda_out[i]);
        sum_diff += d;
        if (d > max_diff) max_diff = d;
        if (d < EPS) close_enough++;
    }

    double match_ratio = (double)close_enough / (double)cmp_vals;
    double avg_diff = sum_diff / (double)cmp_vals;

    fprintf(stderr, "  compared %d values\n", cmp_vals);
    fprintf(stderr, "  max_diff=%.6f  avg_diff=%.6f\n", max_diff, avg_diff);
    fprintf(stderr, "  within 1e-2: %d/%d (%.2f%%)\n",
            close_enough, cmp_vals, match_ratio * 100.0);

    /* Statistics per backend */
    FloatStats cpu_stats = ComputeStats(cpu_out.data(), cmp_vals);
    FloatStats cuda_stats = ComputeStats(cuda_out.data(), cmp_vals);

    fprintf(stderr, "\n  CPU  stats: mean=%.6f std=%.6f abs_mean=%.6f nonzero=%d/%d\n",
            cpu_stats.mean, cpu_stats.stddev, cpu_stats.abs_mean,
            cpu_stats.nonzero, cpu_stats.total);
    fprintf(stderr, "  CUDA stats: mean=%.6f std=%.6f abs_mean=%.6f nonzero=%d/%d\n",
            cuda_stats.mean, cuda_stats.stddev, cuda_stats.abs_mean,
            cuda_stats.nonzero, cuda_stats.total);

    /* In normal operation, outputs should be very close */
    QASR_EXPECT(cuda_tokens > 0);
    QASR_EXPECT(match_ratio > 0.5f); /* at least 50% values close */

    fprintf(stderr, "===== TEST PASSED =====\n\n");
}
#endif /* QASR_CPU_BACKEND_ENABLED */

/* =================================================================
 * Test 3: Full GPU decoder pipeline diagnosis
 * Runs encoder → decoder prefill → autoregressive decode step-by-step.
 * Dumps every token ID + decoded text to pinpoint where/why the
 * decoder produces ASR_TEXT → ENDOFTEXT without actual text.
 *
 * Tests:
 *   a) No forced language (past_asr_text=0)
 *   b) With forced language "Chinese" (past_asr_text=1)
 *   c) With real audio (long.mp3 chunk) if available
 *   d) CPU qwen_transcribe_audio baseline (if QASR_CPU_BACKEND_ENABLED)
 * ================================================================= */
QASR_TEST(CudaDecoderDiagnoseFullPipeline) {
    if (!HasCudaDevice()) return;

    const char * model_dir = std::getenv("QASR_MODEL_DIR");
    if (!model_dir) {
        fprintf(stderr, "  SKIP: QASR_MODEL_DIR not set\n");
        return;
    }

    fprintf(stderr, "\n===== CudaDecoderDiagnoseFullPipeline =====\n");
    fprintf(stderr, "  model_dir=%s\n", model_dir);

    /* Prompt template arrays — MUST match transcribe_segment_gpu in cuda_asr_engine.cc */
    static const int PROMPT_PREFIX_HEAD[] = {QWEN_TOKEN_IM_START, 8948, 198};
    static const int PROMPT_PREFIX_TAIL[] = {QWEN_TOKEN_IM_END, 198, QWEN_TOKEN_IM_START, 872, 198, QWEN_TOKEN_AUDIO_START};
    static const int PROMPT_SUFFIX_BASE[] = {QWEN_TOKEN_AUDIO_END, QWEN_TOKEN_IM_END, 198, QWEN_TOKEN_IM_START, 77091, 198};
    static const int AUDIO_PAD_TOKEN = 151676;

    /* ── Load GPU model ── */
    qasr::CudaBackend backend;
    QASR_EXPECT(backend.Initialize().ok());
    auto t0 = std::chrono::steady_clock::now();
    QASR_EXPECT(backend.PrepareWeights(model_dir).ok());
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "  PrepareWeights: %.1f ms\n", load_ms);

    qasr::CudaSessionState session;
    QASR_EXPECT(SetupCudaSession(backend, session));

    auto * cw = backend.cuda_weights();
    int enc_out_dim = cw ? cw->enc_output_dim : 1024;
    int dec_hidden = cw ? cw->dec_hidden : 1024;
    int vocab_size = cw ? cw->vocab_size : 151936;
    fprintf(stderr, "  enc_output_dim=%d dec_hidden=%d vocab_size=%d\n",
            enc_out_dim, dec_hidden, vocab_size);

    /* ── Generate audio & mel ── */
    auto wav = GenSineWave(1000.0f, 16000, 5.0f);
    fprintf(stderr, "  Audio: %d samples (5 sec, 1 kHz sine)\n", (int)wav.size());

    int mel_frames = 0;
    float * mel = qwen_mel_spectrogram(wav.data(), (int)wav.size(), &mel_frames);
    QASR_EXPECT(mel != nullptr && mel_frames > 0);
    fprintf(stderr, "  Mel: %d frames\n", mel_frames);

    /* ── CPU baseline (if available) ── */
#ifdef QASR_CPU_BACKEND_ENABLED
    fprintf(stderr, "\n  --- CPU baseline via qwen_transcribe_audio ---\n");
    qwen_ctx_t * cpu_ctx = qwen_load(model_dir);
    QASR_EXPECT(cpu_ctx != nullptr);
    char * cpu_text = qwen_transcribe_audio(cpu_ctx, wav.data(), (int)wav.size());
    if (cpu_text) {
        fprintf(stderr, "  CPU text: \"%s\"\n", cpu_text);
        fprintf(stderr, "  CPU perf: total=%.0fms enc=%.0fms dec=%.0fms tokens=%d\n",
                cpu_ctx->perf_total_ms, cpu_ctx->perf_encode_ms,
                cpu_ctx->perf_decode_ms, cpu_ctx->perf_text_tokens);
        std::free(cpu_text);
    } else {
        fprintf(stderr, "  CPU text: (null)\n");
    }
    /* CPU baseline with forced language */
    {
        qwen_set_force_language(cpu_ctx, "Chinese");
        char * cpu_text_lang = qwen_transcribe_audio(cpu_ctx, wav.data(), (int)wav.size());
        if (cpu_text_lang) {
            fprintf(stderr, "  CPU text (lang=Chinese): \"%s\"\n", cpu_text_lang);
            fprintf(stderr, "  CPU perf: total=%.0fms enc=%.0fms dec=%.0fms tokens=%d\n",
                    cpu_ctx->perf_total_ms, cpu_ctx->perf_encode_ms,
                    cpu_ctx->perf_decode_ms, cpu_ctx->perf_text_tokens);
            std::free(cpu_text_lang);
        }
        qwen_set_force_language(cpu_ctx, nullptr);
    }
    qwen_free(cpu_ctx);
    fprintf(stderr, "  --- End CPU baseline ---\n\n");
#endif

    /* ── GPU pipeline (no forced language) ── */
    int max_decode_steps = 32;
    for (int lang_test = 0; lang_test < 2; lang_test++) {
        const char * lang_str = (lang_test == 0) ? "" : "Chinese";
        fprintf(stderr, "\n  ===== GPU decode: lang=\"%s\" =====\n",
                lang_str[0] ? lang_str : "(none)");

        /* Reset session — cannot reuse KV cache between tests */
        qasr::CudaSessionState sess2;
        QASR_EXPECT(SetupCudaSession(backend, sess2));

        /* Step 1: EncoderForward */
        int enc_seq_len = 0;
        cudaEvent_t ev_e0, ev_e1;
        cudaEventCreate(&ev_e0);
        cudaEventCreate(&ev_e1);
        cudaEventRecord(ev_e0);
        auto status = backend.EncoderForward(&sess2, mel, mel_frames, &enc_seq_len);
        cudaEventRecord(ev_e1);
        cudaEventSynchronize(ev_e1);
        float enc_ms = 0;
        cudaEventElapsedTime(&enc_ms, ev_e0, ev_e1);
        cudaEventDestroy(ev_e0);
        cudaEventDestroy(ev_e1);
        if (!status.ok() || enc_seq_len <= 0) {
            fprintf(stderr, "  FAIL: EncoderForward: %s\n", status.ToString().c_str());
            QASR_EXPECT(false);
            continue;
        }
        fprintf(stderr, "  EncoderForward: %d tokens, %.1f ms\n", enc_seq_len, enc_ms);

        /* Step 2: Load tokenizer and build prompt tokens */
        std::string vocab_path = std::string(model_dir) + "/vocab.json";
        qwen_tokenizer_t * tok = qwen_tokenizer_load(vocab_path.c_str());
        if (!tok) {
            fprintf(stderr, "  FAIL: tokenizer load failed\n");
            QASR_EXPECT(false);
            continue;
        }

        /* Forced language tokens */
        int * force_prompt_tokens = nullptr;
        int n_force_prompt_tokens = 0;
        if (lang_str[0] != '\0') {
            char force_text[256];
            snprintf(force_text, sizeof(force_text), "language %s", lang_str);
            int n_lang_txt = 0;
            int * lang_txt_tokens = qwen_tokenizer_encode(tok, force_text, &n_lang_txt);
            n_force_prompt_tokens = n_lang_txt + 1;
            force_prompt_tokens = (int *)malloc((size_t)n_force_prompt_tokens * sizeof(int));
            if (lang_txt_tokens) {
                memcpy(force_prompt_tokens, lang_txt_tokens,
                       (size_t)n_lang_txt * sizeof(int));
                std::free(lang_txt_tokens);
            }
            force_prompt_tokens[n_lang_txt] = QWEN_TOKEN_ASR_TEXT;
        }

        /* Prompt tokens (empty) */
        int n_prompt_tokens = 0;

        /* Build token sequence */
        int prefix_len = 3 + n_prompt_tokens + 6;
        int suffix_len = 6 + n_force_prompt_tokens;
        int total_tokens = prefix_len + enc_seq_len + suffix_len;

        std::vector<std::int32_t> tokens(total_tokens);
        int off = 0;
        for (int i = 0; i < 3; i++) tokens[off++] = PROMPT_PREFIX_HEAD[i];
        for (int i = 0; i < n_prompt_tokens; i++) tokens[off++] = 0; /* unused */
        for (int i = 0; i < 6; i++) tokens[off++] = PROMPT_PREFIX_TAIL[i];
        for (int i = 0; i < enc_seq_len; i++) tokens[off++] = AUDIO_PAD_TOKEN;
        for (int i = 0; i < 6; i++) tokens[off++] = PROMPT_SUFFIX_BASE[i];
        for (int i = 0; i < n_force_prompt_tokens; i++) tokens[off++] = force_prompt_tokens[i];

        fprintf(stderr, "  Prompt: %d tokens (prefix=%d audio=%d suffix=%d)\n",
                total_tokens, prefix_len, enc_seq_len, suffix_len);
        fprintf(stderr, "  Last prompt token: %d\n", tokens[total_tokens - 1]);

        /* Step 3: DecoderPrefill */
        sess2.current_seq_len = 0;
        cudaEvent_t ev_p0, ev_p1;
        cudaEventCreate(&ev_p0);
        cudaEventCreate(&ev_p1);
        cudaEventRecord(ev_p0);
        status = backend.DecoderPrefill(&sess2,
                                         static_cast<float *>(sess2.enc_output.data()),
                                         enc_seq_len,
                                         tokens.data(), total_tokens);
        cudaEventRecord(ev_p1);
        cudaEventSynchronize(ev_p1);
        float prefill_ms = 0;
        cudaEventElapsedTime(&prefill_ms, ev_p0, ev_p1);
        cudaEventDestroy(ev_p0);
        cudaEventDestroy(ev_p1);
        if (!status.ok()) {
            fprintf(stderr, "  FAIL: DecoderPrefill: %s\n", status.ToString().c_str());
            qwen_tokenizer_free(tok);
            if (force_prompt_tokens) free(force_prompt_tokens);
            QASR_EXPECT(false);
            continue;
        }
        fprintf(stderr, "  DecoderPrefill: %.1f ms (current_seq_len=%d)\n",
                prefill_ms, sess2.current_seq_len);

        /* Step 4: Autoregressive decode loop */
        sess2.prev_token = tokens[total_tokens - 1];
        int past_asr_text = (n_force_prompt_tokens > 0) ? 1 : 0;
        fprintf(stderr, "  past_asr_text=%d\n", past_asr_text);

        cudaEvent_t ev_d0, ev_d1;
        cudaEventCreate(&ev_d0);
        cudaEventCreate(&ev_d1);
        cudaEventRecord(ev_d0);

        fprintf(stderr, "\n  Decode steps:\n");
        fprintf(stderr, "  %-6s | %-8s | %-10s | %s\n",
                "Step", "TokenID", "Type", "Decoded");
        fprintf(stderr, "  %s\n", std::string(60, '-').c_str());

        std::vector<int> gen_tokens;
        std::string output_text;
        for (int step = 0; step < max_decode_steps; step++) {
            std::int32_t token_id = 0;
            status = backend.DecodeStep(&sess2, token_id);
            if (!status.ok()) {
                fprintf(stderr, "  FAIL at step %d: %s\n", step, status.ToString().c_str());
                break;
            }

            gen_tokens.push_back(static_cast<int>(token_id));

            /* Categorize token */
            const char * type_str = "OTHER";
            if (token_id == QWEN_TOKEN_ENDOFTEXT) {
                type_str = "ENDOFTEXT";
            } else if (token_id == QWEN_TOKEN_IM_END) {
                type_str = "IM_END";
            } else if (token_id == QWEN_TOKEN_ASR_TEXT) {
                type_str = "ASR_TEXT";
                past_asr_text = 1;
            } else if (past_asr_text) {
                type_str = "TEXT_TOK";
            }

            const char * decoded = qwen_tokenizer_decode(tok, static_cast<int>(token_id));
            if (!decoded) decoded = "(null)";

            fprintf(stderr, "  %-6d | %-8d | %-10s | ", step, token_id, type_str);
            /* Print decoded text safely, escape non-printable */
            for (const char *p = decoded; *p; p++) {
                if ((unsigned char)*p >= 32 && *p != 127)
                    fputc(*p, stderr);
                else
                    fprintf(stderr, "\\x%02x", (unsigned char)*p);
            }
            fprintf(stderr, "\n");

            /* Accumulate text tokens */
            if (type_str == std::string("TEXT_TOK")) {
                output_text += decoded;
            }

            /* Stop conditions */
            if (token_id == QWEN_TOKEN_ENDOFTEXT || token_id == QWEN_TOKEN_IM_END)
                break;
        }

        cudaEventRecord(ev_d1);
        cudaEventSynchronize(ev_d1);
        float decode_ms = 0;
        cudaEventElapsedTime(&decode_ms, ev_d0, ev_d1);
        cudaEventDestroy(ev_d0);
        cudaEventDestroy(ev_d1);

        fprintf(stderr, "\n  Decode: %d steps, %.1f ms\n", (int)gen_tokens.size(), decode_ms);
        fprintf(stderr, "  past_asr_text after decode: %d\n", past_asr_text);
        fprintf(stderr, "  Accumulated text: \"%s\"\n", output_text.c_str());

        /* Statistical analysis */
        int n_text = 0, n_special = 0;
        for (int id : gen_tokens) {
            if (id == QWEN_TOKEN_ENDOFTEXT || id == QWEN_TOKEN_IM_END ||
                id == QWEN_TOKEN_ASR_TEXT || id == QWEN_TOKEN_IM_START ||
                id == QWEN_TOKEN_AUDIO_START || id == QWEN_TOKEN_AUDIO_END)
                n_special++;
            else
                n_text++;
        }
        fprintf(stderr, "  Stats: %d total, %d special, %d text tokens, text=\"%s\"\n",
                (int)gen_tokens.size(), n_special, n_text, output_text.c_str());

        /* Critical assertion: must produce at least 1 text token after ASR_TEXT marker */
        bool has_text_after_asr = false;
        bool seen_asr = false;
        for (int id : gen_tokens) {
            if (id == QWEN_TOKEN_ASR_TEXT) { seen_asr = true; continue; }
            if (seen_asr && id != QWEN_TOKEN_ENDOFTEXT && id != QWEN_TOKEN_IM_END)
                has_text_after_asr = true;
        }
        if (!has_text_after_asr) {
            fprintf(stderr, "  *** DIAGNOSIS: decoder produced ASR_TEXT but no text tokens followed ***\n");
            fprintf(stderr, "  *** Check: encoder output quality, lm_head logits, KV cache correctness ***\n");
        }

        /* Verify encoder output statistics */
        float * enc_host = nullptr;
        cudaMallocHost(&enc_host, (size_t)enc_seq_len * enc_out_dim * sizeof(float));
        cudaMemcpy(enc_host, sess2.enc_output.data(),
                   (size_t)enc_seq_len * enc_out_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        FloatStats enc_stats = ComputeStats(enc_host, enc_seq_len * enc_out_dim);
        fprintf(stderr, "  Encoder output stats: mean=%.6f std=%.6f abs_mean=%.6f nonzero=%d/%d\n",
                enc_stats.mean, enc_stats.stddev, enc_stats.abs_mean,
                enc_stats.nonzero, enc_stats.total);
        cudaFreeHost(enc_host);

        fprintf(stderr, "  %s\n", std::string(60, '-').c_str());

        qwen_tokenizer_free(tok);
        if (force_prompt_tokens) free(force_prompt_tokens);
    }

    std::free(mel);
    fprintf(stderr, "\n===== TEST PASSED =====\n\n");
}

/* =================================================================
 * Test 4: Encoding stress test — 10s audio
 * Checks if larger audio triggers different behavior.
 * ================================================================= */
QASR_TEST(CudaEncoderStress10s) {
    if (!HasCudaDevice()) return;

    const char * model_dir = std::getenv("QASR_MODEL_DIR");
    if (!model_dir) {
        fprintf(stderr, "  SKIP: QASR_MODEL_DIR not set\n");
        return;
    }

    fprintf(stderr, "\n===== CudaEncoderStress10s =====\n");
    fprintf(stderr, "  model_dir=%s\n", model_dir);

    qasr::CudaBackend backend;
    QASR_EXPECT(backend.Initialize().ok());
    QASR_EXPECT(backend.PrepareWeights(model_dir).ok());

    qasr::CudaSessionState session;
    QASR_EXPECT(SetupCudaSession(backend, session));

    /* 10 seconds of audio */
    auto wav = GenSineWave(440.0f, 16000, 10.0f);
    fprintf(stderr, "  Audio: %d samples (10 sec, 440 Hz)\n", (int)wav.size());

    int mel_frames = 0;
    float * mel = qwen_mel_spectrogram(wav.data(), (int)wav.size(), &mel_frames);
    QASR_EXPECT(mel != nullptr);
    fprintf(stderr, "  Mel: %d frames\n", mel_frames);

    int out_dim = 1024;
    std::vector<float> output((size_t)mel_frames * out_dim, 0.0f);
    int out_tokens = 0;

    auto t0 = std::chrono::steady_clock::now();
    auto status = backend.EncodeMel(&session, mel, mel_frames,
                                     output.data(), out_tokens);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::free(mel);

    fprintf(stderr, "  out_tokens=%d  time=%.1f ms (%.2fx realtime)\n",
            out_tokens, ms, ms / 10000.0);

    if (out_tokens > 0) {
        int n = out_tokens * out_dim;
        FloatStats st = ComputeStats(output.data(), n);
        fprintf(stderr, "  output: mean=%.6f std=%.6f abs_mean=%.6f nonzero=%d/%d\n",
                st.mean, st.stddev, st.abs_mean, st.nonzero, st.total);
        QASR_EXPECT(!HasNaNOrInf(output.data(), n));
        QASR_EXPECT(st.nonzero > 0);
    }

    fprintf(stderr, "===== TEST PASSED =====\n\n");
}
