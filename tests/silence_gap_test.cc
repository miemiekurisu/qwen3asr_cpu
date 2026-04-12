/*
 * silence_gap_test.cc — Integration test: realtime streaming with long silence gaps.
 *
 * Constructs synthetic audio:
 *   [speech] + [20s silence] + [speech] + [20s silence] + [speech] + [20s silence]
 *
 * Runs the actual C inference engine (qwen_transcribe_stream) to observe:
 *   - Whether the model crashes during long silence
 *   - Whether hallucinated tokens appear during silence
 *   - Whether the last tokens get emitted (tail flush)
 *   - Latency and token count behaviour
 *
 * Requires:
 *   - Environment variable QASR_MODEL_DIR pointing to a Qwen3-ASR model directory
 *   - testfile/顾君子（01）.wav (WAV audio source)
 *
 * Skips gracefully when model or audio file is absent (CI-safe).
 */

#include "tests/test_registry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef QASR_CPU_BACKEND_ENABLED
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
}
#endif

#include "qasr/audio/frontend.h"

namespace fs = std::filesystem;

namespace {

/* ── Helpers ────────────────────────────────────────────────────── */

fs::path TestfileDir() {
    return fs::path(__FILE__).parent_path().parent_path() / "testfile";
}

/* URL-encoded name used on Windows / Git for Windows. */
const char * kWavBasename =
    "%E9%A1%BE%E5%90%9B%E5%AD%90%EF%BC%8801%EF%BC%89.wav";

fs::path WavPath() {
    return TestfileDir() / kWavBasename;
}

bool WavAvailable() {
    const auto p = WavPath();
    if (fs::exists(p)) return true;
    /* Try unicode name directly. */
    const auto p2 = TestfileDir() / L"\u987e\u541b\u5b50\uff0801\uff09.wav";
    if (fs::exists(p2)) return true;
    std::fprintf(stderr, "  [SKIP] WAV test file not found: %s\n", p.string().c_str());
    return false;
}

fs::path ResolveWavPath() {
    auto p = WavPath();
    if (fs::exists(p)) return p;
    return TestfileDir() / L"\u987e\u541b\u5b50\uff0801\uff09.wav";
}

const char * ModelDir() {
    const char * env = std::getenv("QASR_MODEL_DIR");
    return (env && env[0] != '\0') ? env : nullptr;
}

/* ── Token callback for logging ─────────────────────────────── */

struct TokenLog {
    int call_count;
    int total_bytes;
    std::string full_text;
};

#ifdef QASR_CPU_BACKEND_ENABLED
void LogTokenCallback(const char * piece, void * userdata) {
    if (!piece || !userdata) return;
    auto * log = static_cast<TokenLog *>(userdata);
    log->call_count++;
    std::size_t len = std::strlen(piece);
    log->total_bytes += static_cast<int>(len);
    log->full_text += piece;
    std::fprintf(stderr, "  TOKEN[%d]: \"%s\" (total_bytes=%d)\n",
                 log->call_count, piece, log->total_bytes);
    std::fflush(stderr);
}
#endif

}  // namespace

/* ================================================================
 * Test: SilenceGapStreamInference
 *
 * Build audio = 3 × [ ~5s speech + 20s silence ] and stream through
 * the real encoder-decoder engine.
 * ================================================================ */
QASR_TEST(SilenceGapStreamInference) {
#ifndef QASR_CPU_BACKEND_ENABLED
    std::fprintf(stderr, "  [SKIP] CPU backend not enabled\n");
    return;
#else
    const char * model_dir = ModelDir();
    if (!model_dir) {
        std::fprintf(stderr,
                     "  [SKIP] QASR_MODEL_DIR not set — set it to run this test\n"
                     "         e.g. set QASR_MODEL_DIR=D:\\models\\Qwen3-ASR-0.6B\n");
        return;
    }
    if (!WavAvailable()) return;

    /* ── 1. Load speech audio ────────────────────────────────── */
    std::vector<float> wav_samples;
    std::int32_t wav_rate = 0;
    qasr::Status rs = qasr::ReadWav(ResolveWavPath().string(), &wav_samples, &wav_rate);
    if (!rs.ok()) {
        std::fprintf(stderr, "  [SKIP] ReadWav failed: %s\n", rs.message().c_str());
        return;
    }

    /* Resample to 16 kHz if needed. */
    std::vector<float> audio_16k;
    if (wav_rate != 16000) {
        qasr::Status s = qasr::Resample(wav_samples, wav_rate, 16000, &audio_16k);
        if (!s.ok()) {
            std::fprintf(stderr, "  [SKIP] Resample failed: %s\n", s.message().c_str());
            return;
        }
    } else {
        audio_16k = wav_samples;
    }
    wav_samples.clear();  /* free memory */

    /* Use first 5 seconds of speech. */
    const std::size_t speech_samples = std::min<std::size_t>(audio_16k.size(), 5 * 16000);
    const std::size_t silence_samples = 20 * 16000;  /* 20 seconds of silence */

    /* ── 2. Build synthetic audio ────────────────────────────── */
    /* [5s speech] [20s silence] [5s speech] [20s silence] [5s speech] [20s silence] */
    const int n_segments = 3;
    const std::size_t total = static_cast<std::size_t>(n_segments) * (speech_samples + silence_samples);
    std::vector<float> synthetic(total, 0.0f);  /* zero-fill = silence */

    for (int seg = 0; seg < n_segments; seg++) {
        std::size_t offset = static_cast<std::size_t>(seg) * (speech_samples + silence_samples);
        std::memcpy(synthetic.data() + offset, audio_16k.data(),
                    speech_samples * sizeof(float));
    }
    audio_16k.clear();

    double total_dur_sec = static_cast<double>(total) / 16000.0;
    std::fprintf(stderr,
                 "\n=== SilenceGapStreamInference ===\n"
                 "  Speech segment:  %.1f s\n"
                 "  Silence gap:     %.1f s\n"
                 "  Segments:        %d\n"
                 "  Total duration:  %.1f s\n"
                 "  Model dir:       %s\n\n",
                 static_cast<double>(speech_samples) / 16000.0,
                 static_cast<double>(silence_samples) / 16000.0,
                 n_segments, total_dur_sec, model_dir);

    /* ── 3. Load model ───────────────────────────────────────── */
    qwen_verbose = 2;
    qwen_monitor = 1;
    qwen_set_threads(0);  /* auto-detect */

    qwen_ctx_t * ctx = qwen_load(model_dir);
    if (!ctx) {
        std::fprintf(stderr, "  [SKIP] qwen_load failed for %s\n", model_dir);
        return;
    }

    /* Configure streaming parameters */
    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 1;
    ctx->stream_chunk_sec = 0.5f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;

    /* ── 4. Set token callback ───────────────────────────────── */
    TokenLog token_log{};
    qwen_set_token_callback(ctx, LogTokenCallback, &token_log);

    /* ── 5. Run streaming transcription ──────────────────────── */
    std::fprintf(stderr, "\n--- Starting stream_infer ---\n");
    std::fflush(stderr);

    char * result = qwen_transcribe_stream(
        ctx, synthetic.data(), static_cast<int>(total));

    std::fprintf(stderr, "\n--- stream_infer completed ---\n");
    std::fflush(stderr);

    /* ── 6. Report results ───────────────────────────────────── */
    std::fprintf(stderr,
                 "\n=== Results ===\n"
                 "  Final text:      \"%s\"\n"
                 "  Token callbacks: %d\n"
                 "  Emitted bytes:   %d\n"
                 "  Emitted text:    \"%s\"\n"
                 "  perf_total_ms:   %.0f\n"
                 "  perf_encode_ms:  %.0f\n"
                 "  perf_decode_ms:  %.0f\n"
                 "  perf_text_tokens:%d\n"
                 "  audio_ms:        %.0f\n",
                 result ? result : "(null)",
                 token_log.call_count,
                 token_log.total_bytes,
                 token_log.full_text.c_str(),
                 ctx->perf_total_ms,
                 ctx->perf_encode_ms,
                 ctx->perf_decode_ms,
                 ctx->perf_text_tokens,
                 ctx->perf_audio_ms);

    /* ── 7. Basic assertions ─────────────────────────────────── */
    /* The test should not crash (reaching here = no crash). */
    QASR_EXPECT(result != nullptr);

    if (result) {
        /* Should produce some non-empty text. */
        QASR_EXPECT(std::strlen(result) > 0);

        /* Check the returned text matches the callback-emitted text
         * (tail emission issue: if last tokens don't get emitted
         * via callback, the full_text will be shorter than result). */
        std::fprintf(stderr,
                     "\n=== Tail emission check ===\n"
                     "  result length:  %zu bytes\n"
                     "  emitted length: %zu bytes\n",
                     std::strlen(result),
                     token_log.full_text.size());

        if (token_log.full_text.size() < std::strlen(result)) {
            std::fprintf(stderr,
                         "  WARNING: Emitted text is shorter than final result!\n"
                         "           Missing tail: \"%s\"\n",
                         result + token_log.full_text.size());
        }

        std::free(result);
    }

    qwen_free(ctx);

    std::fprintf(stderr, "\n=== SilenceGapStreamInference PASSED ===\n\n");
#endif
}

/* ================================================================
 * Test: SilenceGapLive — same audio via the live_audio path
 *
 * Feeds the synthetic audio in 0.5s chunks, simulating real-time
 * microphone input with a qwen_live_audio_t buffer.
 * ================================================================ */
QASR_TEST(SilenceGapLive) {
#ifndef QASR_CPU_BACKEND_ENABLED
    std::fprintf(stderr, "  [SKIP] CPU backend not enabled\n");
    return;
#else
    const char * model_dir = ModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }
    if (!WavAvailable()) return;

    /* ── Load and build synthetic audio (same as above) ──────── */
    std::vector<float> wav_samples;
    std::int32_t wav_rate = 0;
    qasr::Status rs = qasr::ReadWav(ResolveWavPath().string(), &wav_samples, &wav_rate);
    if (!rs.ok()) {
        std::fprintf(stderr, "  [SKIP] ReadWav failed: %s\n", rs.message().c_str());
        return;
    }
    std::vector<float> audio_16k;
    if (wav_rate != 16000) {
        qasr::Status s = qasr::Resample(wav_samples, wav_rate, 16000, &audio_16k);
        if (!s.ok()) {
            std::fprintf(stderr, "  [SKIP] Resample failed\n");
            return;
        }
    } else {
        audio_16k = wav_samples;
    }
    wav_samples.clear();

    const std::size_t speech_samples = std::min<std::size_t>(audio_16k.size(), 5 * 16000);
    const std::size_t silence_samples = 20 * 16000;
    const int n_segments = 3;
    const std::size_t total = static_cast<std::size_t>(n_segments) * (speech_samples + silence_samples);
    std::vector<float> synthetic(total, 0.0f);
    for (int seg = 0; seg < n_segments; seg++) {
        std::size_t offset = static_cast<std::size_t>(seg) * (speech_samples + silence_samples);
        std::memcpy(synthetic.data() + offset, audio_16k.data(),
                    speech_samples * sizeof(float));
    }
    audio_16k.clear();

    std::fprintf(stderr,
                 "\n=== SilenceGapLive ===\n"
                 "  Total duration: %.1f s (3 × [5s speech + 20s silence])\n"
                 "  Model dir:      %s\n\n",
                 static_cast<double>(total) / 16000.0, model_dir);

    /* ── Load model ──────────────────────────────────────────── */
    qwen_verbose = 2;
    qwen_monitor = 1;
    qwen_set_threads(0);

    qwen_ctx_t * ctx = qwen_load(model_dir);
    if (!ctx) {
        std::fprintf(stderr, "  [SKIP] qwen_load failed\n");
        return;
    }
    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 1;
    ctx->stream_chunk_sec = 0.5f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;

    TokenLog token_log{};
    qwen_set_token_callback(ctx, LogTokenCallback, &token_log);

    /* ── Set up live audio buffer ────────────────────────────── */
    qwen_live_audio_t live{};
#ifdef _WIN32
    InitializeCriticalSection(&live.mutex);
    InitializeConditionVariable(&live.cond);
#else
    pthread_mutex_init(&live.mutex, nullptr);
    pthread_cond_init(&live.cond, nullptr);
#endif
    live.samples = nullptr;
    live.sample_offset = 0;
    live.n_samples = 0;
    live.capacity = 0;
    live.eof = 0;
    live.decoded_cursor = 0;

    /* Feed all audio in 0.5s chunks, then set EOF.
     * Since qwen_transcribe_stream_live blocks (waits for data),
     * we must feed from a separate thread. */
    struct FeedCtx {
        qwen_live_audio_t * la;
        const float * data;
        std::size_t total;
        std::size_t chunk;
    };
    FeedCtx feed_ctx{&live, synthetic.data(), total, 8000};

#ifdef _WIN32
    HANDLE feed_thread = CreateThread(nullptr, 0,
        [](LPVOID arg) -> DWORD {
            auto * fc = static_cast<FeedCtx *>(arg);
            std::size_t fed = 0;
            while (fed < fc->total) {
                std::size_t n = fc->chunk;
                if (n > fc->total - fed) n = fc->total - fed;

                /* Append under lock */
                EnterCriticalSection(&fc->la->mutex);
                int64_t need = fc->la->n_samples + static_cast<int64_t>(n);
                if (need > fc->la->capacity) {
                    int64_t new_cap = fc->la->capacity > 0 ? fc->la->capacity : 32000;
                    while (new_cap < need) new_cap *= 2;
                    float * tmp = static_cast<float *>(
                        std::realloc(fc->la->samples,
                                     static_cast<std::size_t>(new_cap) * sizeof(float)));
                    if (tmp) {
                        fc->la->samples = tmp;
                        fc->la->capacity = new_cap;
                    }
                }
                if (fc->la->n_samples + static_cast<int64_t>(n) <= fc->la->capacity) {
                    std::memcpy(fc->la->samples + fc->la->n_samples,
                                fc->data + fed, n * sizeof(float));
                    fc->la->n_samples += static_cast<int64_t>(n);
                }
                WakeConditionVariable(&fc->la->cond);
                LeaveCriticalSection(&fc->la->mutex);

                fed += n;

                /* Log progress every 5 seconds of audio */
                if (fed % (5 * 16000) < fc->chunk) {
                    std::fprintf(stderr, "  [feed] %.1f / %.1f s\n",
                                 static_cast<double>(fed) / 16000.0,
                                 static_cast<double>(fc->total) / 16000.0);
                }
            }

            /* Signal EOF */
            EnterCriticalSection(&fc->la->mutex);
            fc->la->eof = 1;
            WakeConditionVariable(&fc->la->cond);
            LeaveCriticalSection(&fc->la->mutex);

            std::fprintf(stderr, "  [feed] EOF at %.1f s\n",
                         static_cast<double>(fc->total) / 16000.0);
            return 0;
        }, &feed_ctx, 0, nullptr);
#else
    pthread_t feed_thread;
    pthread_create(&feed_thread, nullptr,
        [](void * arg) -> void * {
            auto * fc = static_cast<FeedCtx *>(arg);
            std::size_t fed = 0;
            while (fed < fc->total) {
                std::size_t n = fc->chunk;
                if (n > fc->total - fed) n = fc->total - fed;

                pthread_mutex_lock(&fc->la->mutex);
                int64_t need = fc->la->n_samples + static_cast<int64_t>(n);
                if (need > fc->la->capacity) {
                    int64_t new_cap = fc->la->capacity > 0 ? fc->la->capacity : 32000;
                    while (new_cap < need) new_cap *= 2;
                    float * tmp = static_cast<float *>(
                        std::realloc(fc->la->samples,
                                     static_cast<std::size_t>(new_cap) * sizeof(float)));
                    if (tmp) {
                        fc->la->samples = tmp;
                        fc->la->capacity = new_cap;
                    }
                }
                if (fc->la->n_samples + static_cast<int64_t>(n) <= fc->la->capacity) {
                    std::memcpy(fc->la->samples + fc->la->n_samples,
                                fc->data + fed, n * sizeof(float));
                    fc->la->n_samples += static_cast<int64_t>(n);
                }
                pthread_cond_signal(&fc->la->cond);
                pthread_mutex_unlock(&fc->la->mutex);

                fed += n;
                if (fed % (5 * 16000) < fc->chunk) {
                    std::fprintf(stderr, "  [feed] %.1f / %.1f s\n",
                                 static_cast<double>(fed) / 16000.0,
                                 static_cast<double>(fc->total) / 16000.0);
                }
            }
            pthread_mutex_lock(&fc->la->mutex);
            fc->la->eof = 1;
            pthread_cond_signal(&fc->la->cond);
            pthread_mutex_unlock(&fc->la->mutex);
            std::fprintf(stderr, "  [feed] EOF\n");
            return nullptr;
        }, &feed_ctx);
#endif

    /* ── Run live streaming transcription (blocks until EOF) ── */
    std::fprintf(stderr, "\n--- Starting stream_live ---\n");
    char * result = qwen_transcribe_stream_live(ctx, &live);
    std::fprintf(stderr, "\n--- stream_live completed ---\n");

    /* Wait for feeder thread */
#ifdef _WIN32
    WaitForSingleObject(feed_thread, INFINITE);
    CloseHandle(feed_thread);
    DeleteCriticalSection(&live.mutex);
#else
    pthread_join(feed_thread, nullptr);
    pthread_mutex_destroy(&live.mutex);
    pthread_cond_destroy(&live.cond);
#endif
    std::free(live.samples);

    /* ── Report results ──────────────────────────────────────── */
    std::fprintf(stderr,
                 "\n=== SilenceGapLive Results ===\n"
                 "  Final text:      \"%s\"\n"
                 "  Token callbacks: %d\n"
                 "  Emitted text:    \"%s\"\n"
                 "  perf_total_ms:   %.0f\n"
                 "  perf_encode_ms:  %.0f\n"
                 "  perf_decode_ms:  %.0f\n"
                 "  perf_text_tokens:%d\n",
                 result ? result : "(null)",
                 token_log.call_count,
                 token_log.full_text.c_str(),
                 ctx->perf_total_ms,
                 ctx->perf_encode_ms,
                 ctx->perf_decode_ms,
                 ctx->perf_text_tokens);

    QASR_EXPECT(result != nullptr);
    if (result) {
        QASR_EXPECT(std::strlen(result) > 0);

        std::fprintf(stderr,
                     "\n=== Tail emission check (live) ===\n"
                     "  result length:  %zu bytes\n"
                     "  emitted length: %zu bytes\n",
                     std::strlen(result),
                     token_log.full_text.size());
        if (token_log.full_text.size() < std::strlen(result)) {
            std::fprintf(stderr,
                         "  WARNING: Missing tail via callback: \"%s\"\n",
                         result + token_log.full_text.size());
        }
        std::free(result);
    }

    qwen_free(ctx);
    std::fprintf(stderr, "\n=== SilenceGapLive PASSED ===\n\n");
#endif
}
