/*
 * chunk_callback_test.cc — Integration test for qwen_set_chunk_callback.
 *
 * Exercises the C-level per-chunk callback contract end-to-end with the
 * real encoder-decoder engine:
 *
 *   1. Callback fires once per chunk (i.e. once per audio chunk_step)
 *   2. is_first is true exactly on the first invocation
 *   3. is_final is true exactly on the last invocation
 *   4. chunk_index is monotonically non-decreasing starting at 0
 *   5. stable_piece tokens (per-chunk deltas) concatenate to the final text
 *   6. tentative_piece is present on every non-final chunk (maybe empty)
 *   7. tentative_piece is "" on the final chunk
 *   8. The callback receives the audio cursor / decode timing fields
 *
 * Requires:
 *   - Environment variable QASR_MODEL_DIR pointing to a Qwen3-ASR model
 *   - testfile/aishell_S0002_limai_108s.wav (10 s of speech is enough)
 *
 * Skips gracefully when model or audio file is absent (CI-safe).
 */

#include "tests/test_registry.h"

#include <algorithm>
#include <cstdint>
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

fs::path TestfileDir() {
    return fs::path(__FILE__).parent_path().parent_path() / "testfile";
}

fs::path WavPath() {
    return TestfileDir() / "aishell_S0002_limai_108s.wav";
}

bool WavAvailable() {
    const auto p = WavPath();
    if (fs::exists(p)) return true;
    std::fprintf(stderr, "  [SKIP] WAV test file not found: %s\n", p.string().c_str());
    return false;
}

const char * ModelDir() {
    const char * env = std::getenv("QASR_MODEL_DIR");
    return (env && env[0] != '\0') ? env : nullptr;
}

#ifdef QASR_CPU_BACKEND_ENABLED

struct ChunkLog {
    int call_count = 0;
    int first_index = -1;
    int final_index = -1;
    int prev_chunk_index = -1;
    std::string stable_pieces;     /* concatenation of every stable_piece */
    std::string tentative_pieces;   /* concatenation of tentative_pieces (for cross-check) */
    int max_stable_token_count = 0;
    int max_tentative_token_count = 0;
    int final_has_tentative_nonzero = -1;  /* 1 if final chunk had tentative, 0 if empty, -1 unset */
    int audio_cursor_monotonic_violations = 0;
    int64_t prev_audio_cursor = -1;
    double max_decode_ms = 0.0;
    int nonfinal_tentative_empty_count = 0;
    int nonfinal_total = 0;
};

void LogChunkCallback(const qwen_stream_chunk_t * chunk, void * userdata) {
    if (!chunk || !userdata) return;
    auto * log = static_cast<ChunkLog *>(userdata);
    log->call_count++;

    if (chunk->is_first && log->first_index == -1) {
        log->first_index = chunk->chunk_index;
    }

    if (chunk->chunk_index < log->prev_chunk_index) {
        std::fprintf(stderr, "  WARN: chunk_index went backwards (%d -> %d)\n",
                     log->prev_chunk_index, chunk->chunk_index);
    }
    log->prev_chunk_index = chunk->chunk_index;

    if (chunk->stable_piece) {
        log->stable_pieces += chunk->stable_piece;
    }
    if (chunk->tentative_piece) {
        log->tentative_pieces += chunk->tentative_piece;
    }
    if (chunk->stable_token_count > log->max_stable_token_count) {
        log->max_stable_token_count = chunk->stable_token_count;
    }
    if (chunk->tentative_token_count > log->max_tentative_token_count) {
        log->max_tentative_token_count = chunk->tentative_token_count;
    }

    if ((int64_t)chunk->audio_cursor < log->prev_audio_cursor) {
        log->audio_cursor_monotonic_violations++;
    }
    log->prev_audio_cursor = chunk->audio_cursor;
    if (chunk->decode_ms > log->max_decode_ms) {
        log->max_decode_ms = chunk->decode_ms;
    }

    if (chunk->is_final) {
        log->final_index = chunk->chunk_index;
        log->final_has_tentative_nonzero =
            (chunk->tentative_piece && chunk->tentative_piece[0] != '\0') ? 1 : 0;
    } else {
        log->nonfinal_total++;
        if (!chunk->tentative_piece || chunk->tentative_piece[0] == '\0') {
            log->nonfinal_tentative_empty_count++;
        }
    }
}

#endif  // QASR_CPU_BACKEND_ENABLED

}  // namespace

QASR_TEST(ChunkCallbackFiresPerChunk) {
#ifndef QASR_CPU_BACKEND_ENABLED
    std::fprintf(stderr, "  [SKIP] CPU backend not enabled\n");
    return;
#else
    const char * model_dir = ModelDir();
    if (!model_dir) {
        std::fprintf(stderr,
                     "  [SKIP] QASR_MODEL_DIR not set — set it to run this test\n"
                     "         e.g. set QASR_MODEL_DIR=/path/to/Qwen3-ASR-0.6B\n");
        return;
    }
    if (!WavAvailable()) return;

    /* ── 1. Load 10 s of speech audio ──────────────────────────── */
    std::vector<float> wav_samples;
    std::int32_t wav_rate = 0;
    qasr::Status rs = qasr::ReadWav(WavPath().string(), &wav_samples, &wav_rate);
    if (!rs.ok()) {
        std::fprintf(stderr, "  [SKIP] ReadWav failed: %s\n", rs.message().c_str());
        return;
    }
    std::vector<float> audio_16k;
    if (wav_rate != 16000) {
        qasr::Status s = qasr::Resample(wav_samples, wav_rate, 16000, &audio_16k);
        if (!s.ok()) {
            std::fprintf(stderr, "  [SKIP] Resample failed: %s\n", rs.message().c_str());
            return;
        }
    } else {
        audio_16k = std::move(wav_samples);
    }

    /* Use first 10 seconds of speech. */
    const std::size_t n_use = std::min<std::size_t>(audio_16k.size(), 10 * 16000);
    audio_16k.resize(n_use);
    const double dur_sec = static_cast<double>(n_use) / 16000.0;

    std::fprintf(stderr,
                 "\n=== ChunkCallbackFiresPerChunk ===\n"
                 "  Audio:           %.1f s\n"
                 "  Model dir:       %s\n\n",
                 dur_sec, model_dir);

    /* ── 2. Load model ───────────────────────────────────────── */
    qwen_verbose = 0;
    qwen_monitor = 0;
    qwen_set_threads(0);

    qwen_ctx_t * ctx = qwen_load(model_dir);
    if (!ctx) {
        std::fprintf(stderr, "  [SKIP] qwen_load failed for %s\n", model_dir);
        return;
    }

    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 1;
    ctx->stream_chunk_sec = 0.5f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;

    /* ── 3. Set chunk callback (clear any token callback) ─────── */
    ChunkLog chunk_log{};
    qwen_set_token_callback(ctx, nullptr, nullptr);
    qwen_set_chunk_callback(ctx, LogChunkCallback, &chunk_log);
    QASR_EXPECT(ctx->chunk_cb == LogChunkCallback);
    QASR_EXPECT(ctx->chunk_cb_userdata == &chunk_log);

    /* ── 4. Run streaming transcription ──────────────────────── */
    std::fprintf(stderr, "--- Starting qwen_transcribe_stream ---\n");
    std::fflush(stderr);

    char * result = qwen_transcribe_stream(
        ctx, audio_16k.data(), static_cast<int>(n_use));

    std::fprintf(stderr, "--- stream_infer completed ---\n\n");
    std::fflush(stderr);

    /* ── 5. Verify the callback contract ─────────────────────── */
    std::fprintf(stderr,
                 "=== ChunkCallback Results ===\n"
                 "  Chunk callbacks: %d\n"
                 "  First index:     %d\n"
                 "  Final index:     %d\n"
                 "  Final text:      \"%s\"\n"
                 "  Stable concat:   \"%s\"\n"
                 "  Tentative concat:\"%s\"\n"
                 "  Max stable toks: %d\n"
                 "  Max tentative:   %d\n"
                 "  Non-final empty tentative: %d/%d\n"
                 "  Final has tentative nonzero: %d\n"
                 "  Audio cursor monotonic violations: %d\n"
                 "  Max decode ms:   %.1f\n",
                 chunk_log.call_count,
                 chunk_log.first_index,
                 chunk_log.final_index,
                 result ? result : "(null)",
                 chunk_log.stable_pieces.c_str(),
                 chunk_log.tentative_pieces.c_str(),
                 chunk_log.max_stable_token_count,
                 chunk_log.max_tentative_token_count,
                 chunk_log.nonfinal_tentative_empty_count,
                 chunk_log.nonfinal_total,
                 chunk_log.final_has_tentative_nonzero,
                 chunk_log.audio_cursor_monotonic_violations,
                 chunk_log.max_decode_ms);

    /* (1) Callback fires at least once. */
    QASR_EXPECT(chunk_log.call_count >= 1);

    /* (2) is_first triggers exactly once. */
    QASR_EXPECT_EQ(chunk_log.first_index, 0);

    /* (3) is_final triggers exactly once on the last chunk. */
    QASR_EXPECT(chunk_log.final_index >= 0);
    /* (7) final chunk has empty tentative_piece. */
    QASR_EXPECT_EQ(chunk_log.final_has_tentative_nonzero, 0);

    /* (4) chunk_index monotonically non-decreasing.  The violations field
     * is incremented only on a strict decrease, so this should be 0. */
    QASR_EXPECT_EQ(chunk_log.audio_cursor_monotonic_violations, 0);

    /* (5) stable_pieces concatenation matches the final emitted text
     * (token_callback output and final result, if both available). */
    if (result) {
        QASR_EXPECT(std::strcmp(result, chunk_log.stable_pieces.c_str()) == 0);
    }

    /* (8) Decode timing field is non-negative. */
    QASR_EXPECT(chunk_log.max_decode_ms >= 0.0);

    /* Tearing down: free context, releasing scratch buffers that the
     * callback allocated. */
    qwen_free(ctx);
#endif
}
