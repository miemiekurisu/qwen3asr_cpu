// qasr_blas_bench — real-time-stream BLAS benchmark
//
// Exercises qwen_transcribe_stream_live on a real long audio file (chunked
// 2 s pieces, fed from a feeder thread) and reports:
//   * total wall time
//   * audio-RTF (audio_duration / wall_time)
//   * per-chunk decode time (from stream_impl internal stats)
//   * first-token latency (audio time at first decoded token)
//
// This is the realistic microbenchmark for comparing OpenBLAS / BLIS / MKL,
// because the matmul shapes that dominate qwen_asr_kernels.c sgemm calls
// (Q×K^T, S×V, QKV proj, FFN up/gate, im2col conv2d) are all hit during
// streaming decode of one chunk.
//
// Usage:
//   qasr_blas_bench --model-dir <dir> --audio <wav> [options]
//     --chunk-ms   N   feeder push size in milliseconds (default 2000)
//     --threads    N   OMP threads (default 0 = qwen decides)
//     --language  S   force language tag (default Chinese)
//     --rounds     N   repeat the whole pipeline N times (default 3)
//     --backend   S   override backend tag in JSON output
//     --verbose        enable qwen verbose log

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "qasr/audio/frontend.h"

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
}

namespace {

struct Options {
    std::string model_dir;
    std::string audio_path;
    int chunk_ms = 2000;          // feeder push size in ms
    int threads = 0;              // 0 = qwen decides
    std::string language = "Chinese";
    int rounds = 3;               // amortise warmup; 3 is a good default
    bool verbose = false;
    std::string backend_tag;      // for JSON output; CMake passes QASR_BENCH_BACKEND
};

bool ParseArgs(int argc, char ** argv, Options * opts) {
#if defined(QASR_BENCH_BACKEND)
    opts->backend_tag = QASR_BENCH_BACKEND;
#endif
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->model_dir = v;
        } else if (a == "--audio") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->audio_path = v;
        } else if (a == "--chunk-ms") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->chunk_ms = std::atoi(v);
        } else if (a == "--threads") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->threads = std::atoi(v);
        } else if (a == "--language") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->language = v;
        } else if (a == "--rounds") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->rounds = std::atoi(v);
        } else if (a == "--backend") {
            const char * v = need(a.c_str()); if (!v) return false;
            opts->backend_tag = v;
        } else if (a == "--verbose") {
            opts->verbose = true;
        } else if (a == "-h" || a == "--help") {
            std::printf(
                "Usage: %s --model-dir <dir> --audio <wav> [options]\n"
                "  --chunk-ms   N   feeder push size in milliseconds (default 2000)\n"
                "  --threads    N   OMP threads (default 0 = qwen decides)\n"
                "  --language  S   force language tag (default Chinese)\n"
                "  --rounds     N   repeat the whole pipeline N times (default 3)\n"
                "  --backend   S   override backend tag in JSON output\n"
                "  --verbose        enable qwen verbose log\n",
                argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return false;
        }
    }
    if (opts->model_dir.empty() || opts->audio_path.empty()) {
        std::fprintf(stderr, "--model-dir and --audio are required\n");
        return false;
    }
    return true;
}

struct FeedCtx {
    qwen_live_audio_t * la;
    const float * data;
    std::size_t total;
    std::size_t chunk;
    std::int64_t sample_rate;
};

#ifdef _WIN32
DWORD WINAPI FeederThread(LPVOID arg) {
#else
void * FeederThread(void * arg) {
#endif
    auto * fc = static_cast<FeedCtx *>(arg);
    std::size_t fed = 0;
    while (fed < fc->total) {
        std::size_t n = fc->chunk;
        if (n > fc->total - fed) n = fc->total - fed;

#ifdef _WIN32
        EnterCriticalSection(&fc->la->mutex);
#else
        pthread_mutex_lock(&fc->la->mutex);
#endif
        std::int64_t need = fc->la->n_samples + static_cast<std::int64_t>(n);
        if (need > fc->la->capacity) {
            std::int64_t new_cap = fc->la->capacity > 0 ? fc->la->capacity
                                                        : static_cast<std::int64_t>(fc->sample_rate);
            while (new_cap < need) new_cap *= 2;
            float * tmp = static_cast<float *>(
                std::realloc(fc->la->samples,
                             static_cast<std::size_t>(new_cap) * sizeof(float)));
            if (tmp) {
                fc->la->samples = tmp;
                fc->la->capacity = new_cap;
            }
        }
        if (fc->la->n_samples + static_cast<std::int64_t>(n) <= fc->la->capacity) {
            std::memcpy(fc->la->samples + fc->la->n_samples,
                        fc->data + fed, n * sizeof(float));
            fc->la->n_samples += static_cast<std::int64_t>(n);
        }
#ifdef _WIN32
        WakeConditionVariable(&fc->la->cond);
        LeaveCriticalSection(&fc->la->mutex);
#else
        pthread_cond_signal(&fc->la->cond);
        pthread_mutex_unlock(&fc->la->mutex);
#endif

        fed += n;
    }
#ifdef _WIN32
        EnterCriticalSection(&fc->la->mutex);
#else
        pthread_mutex_lock(&fc->la->mutex);
#endif
    fc->la->eof = 1;
#ifdef _WIN32
        WakeConditionVariable(&fc->la->cond);
        LeaveCriticalSection(&fc->la->mutex);
#else
        pthread_cond_signal(&fc->la->cond);
        pthread_mutex_unlock(&fc->la->mutex);
#endif
#ifdef _WIN32
    return 0;
#else
    return nullptr;
#endif
}

}  // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!ParseArgs(argc, argv, &opts)) return 1;

    // ── Load + decode + resample to 16 kHz mono ─────────────────
    // Use ReadWav directly (LoadAudioFile returns duration_ms, not sample rate).
    std::vector<float> audio_16k;
    {
        std::vector<float> raw;
        std::int32_t raw_sr = 0;
        qasr::Status ls = qasr::ReadWav(opts.audio_path, &raw, &raw_sr);
        if (!ls.ok()) {
            std::fprintf(stderr, "ReadWav failed: %s\n", ls.ToString().c_str());
            return 1;
        }
        if (raw_sr != 16000) {
            qasr::Status rs = qasr::Resample(raw, raw_sr, 16000, &audio_16k);
            if (!rs.ok()) {
                std::fprintf(stderr, "Resample failed: %s\n", rs.ToString().c_str());
                return 1;
            }
        } else {
            audio_16k = std::move(raw);
        }
    }
    const double audio_sec = static_cast<double>(audio_16k.size()) / 16000.0;
    const std::size_t chunk_samples = static_cast<std::size_t>(opts.chunk_ms) * 16;
    std::fprintf(stderr,
                 "[bench] audio=%.2f s @ 16 kHz, chunk=%d ms, threads=%d, rounds=%d\n",
                 audio_sec, opts.chunk_ms, opts.threads, opts.rounds);

    // ── Model load (timed, but amortized across rounds) ─────────
    if (opts.verbose) qwen_verbose = 2;
    qwen_monitor = 1;
    qwen_set_threads(opts.threads);

    auto t_load0 = std::chrono::steady_clock::now();
    qwen_ctx_t * ctx = qwen_load(opts.model_dir.c_str());
    auto t_load1 = std::chrono::steady_clock::now();
    if (!ctx) {
        std::fprintf(stderr, "qwen_load failed\n");
        return 1;
    }
    double load_ms =
        std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
    std::fprintf(stderr, "[bench] model load: %.0f ms\n", load_ms);

    if (!opts.language.empty()) {
        qwen_set_force_language(ctx, opts.language.c_str());
    }
    ctx->stream_max_new_tokens = 32;
    ctx->stream_chunk_sec = static_cast<float>(opts.chunk_ms) / 1000.0f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;
    ctx->stream_idle_flush_ms = 500;  // 0.5 s idle wake
    ctx->stream_idle_flush_min_sec = 0.3f;

    // ── Token callback: just count tokens, no per-piece work ───
    std::int64_t total_tokens_holder = 0;
    qwen_set_token_callback(ctx,
        [](const char * piece, void * u) {
            (void)piece;
            *static_cast<std::int64_t *>(u) += 1;
        },
        &total_tokens_holder);

    // ── Run N rounds ────────────────────────────────────────────
    double best_wall_ms = 1e18;
    double sum_wall_ms = 0;
    std::int64_t total_tokens = 0;
    for (int r = 0; r < opts.rounds; ++r) {
        // Reset live audio state for each round.
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

        FeedCtx fc{&live, audio_16k.data(), audio_16k.size(), chunk_samples, 16000};
#ifdef _WIN32
        HANDLE feeder = CreateThread(nullptr, 0, FeederThread, &fc, 0, nullptr);
#else
        pthread_t feeder;
        pthread_create(&feeder, nullptr, FeederThread, &fc);
#endif

        total_tokens_holder = 0;
        auto t0 = std::chrono::steady_clock::now();
        char * result = qwen_transcribe_stream_live(ctx, &live);
        auto t1 = std::chrono::steady_clock::now();
#ifdef _WIN32
        WaitForSingleObject(feeder, INFINITE);
        CloseHandle(feeder);
#else
        pthread_join(feeder, nullptr);
#endif
        double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (result) std::free(result);

        // Tear down live state.
        if (live.samples) std::free(live.samples);
#ifdef _WIN32
        DeleteCriticalSection(&live.mutex);
#else
        pthread_mutex_destroy(&live.mutex);
        pthread_cond_destroy(&live.cond);
#endif

        double rtf = audio_sec / (wall_ms / 1000.0);
        std::fprintf(stderr,
                     "[bench] round %d: wall=%.1f ms  rtf=%.2fx  tokens=%lld  "
                     "decoded_cursor=%lld\n",
                     r + 1, wall_ms, rtf,
                     static_cast<long long>(total_tokens_holder),
                     static_cast<long long>(live.decoded_cursor));
        if (wall_ms < best_wall_ms) best_wall_ms = wall_ms;
        sum_wall_ms += wall_ms;
        total_tokens += total_tokens_holder;
    }

    double avg_wall_ms = sum_wall_ms / opts.rounds;
    double best_rtf = audio_sec / (best_wall_ms / 1000.0);
    double avg_rtf = audio_sec / (avg_wall_ms / 1000.0);

    // ── JSON to stdout for downstream parsing ─────────────────
    const char * backend = opts.backend_tag.empty()
                               ? "unknown"
                               : opts.backend_tag.c_str();
    std::printf(
        "{\"backend\":\"%s\",\"audio_sec\":%.3f,\"chunk_ms\":%d,"
        "\"threads\":%d,\"rounds\":%d,"
        "\"load_ms\":%.1f,\"wall_ms_avg\":%.1f,\"wall_ms_best\":%.1f,"
        "\"rtf_avg\":%.3f,\"rtf_best\":%.3f,\"tokens_total\":%lld}\n",
        backend,
        audio_sec, opts.chunk_ms, opts.threads, opts.rounds,
        load_ms, avg_wall_ms, best_wall_ms, avg_rtf, best_rtf,
        static_cast<long long>(total_tokens));
    std::fflush(stdout);

    qwen_free(ctx);
    return 0;
}
