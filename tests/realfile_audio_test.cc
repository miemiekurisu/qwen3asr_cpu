/*
 * realfile_audio_test.cc — Integration tests using real WAV files from testfile/.
 *
 * Tests exercise the audio frontend pipeline (ReadWav → Resample → Mel →
 * CompactSilence → StreamingAudioRing) on actual Chinese speech, plus
 * INT8 quantisation of realistic mel-spectrogram data.
 *
 * Each test early-returns (passes with a note) when the test file is
 * absent, so CI without asset checkout still succeeds.
 */

#include "tests/test_registry.h"
#include "qasr/audio/frontend.h"
#include "qasr/core/audio_types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(QASR_ENABLE_CPU_BACKEND) || defined(QASR_ONEDNN_AVAILABLE)
extern "C" {
#include "src/backend/qwen_cpu/qwen_asr_onednn.h"
}
#endif

namespace fs = std::filesystem;

namespace {

/* -------------------------------------------------------------------
 * Locate the WAV test file.
 *
 * Filename on disk is URL-encoded (Git for Windows created it that way):
 *   %E9%A1%BE%E5%90%9B%E5%AD%90%EF%BC%8801%EF%BC%89.wav
 * This file is 顾君子（01）.wav — ~55 MB, mono/stereo 16-bit PCM.
 * ------------------------------------------------------------------- */

fs::path TestfileDir() {
    return fs::path(__FILE__).parent_path().parent_path() / "testfile";
}

const char * kWavBasename =
    "%E9%A1%BE%E5%90%9B%E5%AD%90%EF%BC%8801%EF%BC%89.wav";

fs::path WavPath() {
    return TestfileDir() / kWavBasename;
}

bool WavAvailable() {
    const auto p = WavPath();
    if (fs::exists(p)) return true;
    std::fprintf(stderr, "  [SKIP] %s not found\n", p.string().c_str());
    return false;
}

/* Short helper to load the WAV and fail-fast on error. */
struct LoadedWav {
    std::vector<float> samples;
    std::int32_t sample_rate_hz = 0;
};

bool LoadTestWav(LoadedWav * out) {
    if (!WavAvailable()) return false;
    qasr::Status s = qasr::ReadWav(WavPath().string(), &out->samples, &out->sample_rate_hz);
    if (!s.ok()) {
        std::fprintf(stderr, "  [SKIP] ReadWav failed: %s\n", s.message().c_str());
        return false;
    }
    return true;
}

}  // namespace

/* ========================================================================
 * Group 1 — WAV Loading
 * ======================================================================== */

QASR_TEST(RealWavLoadSuccess) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* The file is non-trivial spoken Chinese audio. */
    QASR_EXPECT(wav.sample_rate_hz > 0);
    QASR_EXPECT(wav.samples.size() > 16000);  /* at least 1 second of audio */
}

QASR_TEST(RealWavSampleRate) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* Expect standard PCM rates: 8000, 16000, 22050, 44100, 48000 */
    const int32_t rate = wav.sample_rate_hz;
    QASR_EXPECT(rate == 8000 || rate == 16000 || rate == 22050 ||
                rate == 44100 || rate == 48000);
}

QASR_TEST(RealWavSampleRange) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* All samples should be in [-1, 1] after PCM normalisation. */
    float min_val = 0.0f, max_val = 0.0f;
    for (float s : wav.samples) {
        QASR_EXPECT(std::isfinite(s));
        if (s < min_val) min_val = s;
        if (s > max_val) max_val = s;
    }
    QASR_EXPECT(min_val >= -1.0f);
    QASR_EXPECT(max_val <= 1.0f);

    /* Real speech should have some non-zero dynamic range. */
    QASR_EXPECT(max_val - min_val > 0.01f);
}

QASR_TEST(RealWavDuration) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* Duration in seconds. The file is a multi-minute reading. */
    const double dur_sec = static_cast<double>(wav.samples.size()) /
                           static_cast<double>(wav.sample_rate_hz);
    /* Should be at least 5 seconds and no more than 60 minutes. */
    QASR_EXPECT(dur_sec > 5.0);
    QASR_EXPECT(dur_sec < 3600.0);
}

QASR_TEST(RealWavAudioSpanValidation) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    qasr::AudioSpan span{};
    span.samples = wav.samples.data();
    span.sample_count = static_cast<std::int64_t>(wav.samples.size());
    span.sample_rate_hz = wav.sample_rate_hz;
    span.channels = 1;

    QASR_EXPECT(qasr::ValidateAudioSpan(span).ok());
    QASR_EXPECT(qasr::AudioDurationMs(span) > 5000);  /* > 5 s */
}

/* ========================================================================
 * Group 2 — Resampling
 * ======================================================================== */

QASR_TEST(RealWavResampleTo16k) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> out;
    qasr::Status s = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &out);
    QASR_EXPECT(s.ok());

    /* Target length should be proportional to rate ratio. */
    const double ratio = 16000.0 / wav.sample_rate_hz;
    const auto expected_len = static_cast<std::size_t>(wav.samples.size() * ratio);
    /* Allow 1-sample rounding tolerance. */
    QASR_EXPECT(out.size() >= expected_len - 1);
    QASR_EXPECT(out.size() <= expected_len + 1);

    /* All resampled values in [-1, 1]. */
    for (float v : out) {
        QASR_EXPECT(std::isfinite(v));
        QASR_EXPECT(v >= -1.0f && v <= 1.0f);
    }
}

QASR_TEST(RealWavResampleTo8k) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> out;
    qasr::Status s = qasr::Resample(wav.samples, wav.sample_rate_hz, 8000, &out);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(out.size() > 0);

    const double expected = static_cast<double>(wav.samples.size()) *
                            (8000.0 / wav.sample_rate_hz);
    QASR_EXPECT(std::abs(static_cast<double>(out.size()) - expected) < 2.0);
}

QASR_TEST(RealWavResampleIdentity) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> out;
    qasr::Status s = qasr::Resample(wav.samples, wav.sample_rate_hz,
                                     wav.sample_rate_hz, &out);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(out.size(), wav.samples.size());

    /* Identity resample should produce exact copy. */
    for (std::size_t i = 0; i < out.size(); ++i) {
        QASR_EXPECT(out[i] == wav.samples[i]);
    }
}

/* ========================================================================
 * Group 3 — Mel Spectrogram
 * ======================================================================== */

QASR_TEST(RealWavMelSpectrogram80Bins) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* Resample to 16 kHz if needed. */
    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), audio_16k.size(), 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(n_frames > 0);
    QASR_EXPECT_EQ(mel.size(), static_cast<std::size_t>(n_frames) * 80);

    /* For 16 kHz audio with 10 ms hop, expect ~100 frames/sec. */
    const double dur_sec = static_cast<double>(audio_16k.size()) / 16000.0;
    const double frames_per_sec = n_frames / dur_sec;
    QASR_EXPECT(frames_per_sec > 90.0 && frames_per_sec < 110.0);
}

QASR_TEST(RealWavMelSpectrogram128Bins) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), audio_16k.size(), 128, &n_frames, &mel);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(n_frames > 0);
    QASR_EXPECT_EQ(mel.size(), static_cast<std::size_t>(n_frames) * 128);
}

QASR_TEST(RealWavMelValuesFinite) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), audio_16k.size(), 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    /* Every mel coefficient should be finite. */
    for (float v : mel) {
        QASR_EXPECT(std::isfinite(v));
    }

    /* Speech audio should produce non-constant mel features. */
    float mel_min = mel[0], mel_max = mel[0];
    for (float v : mel) {
        if (v < mel_min) mel_min = v;
        if (v > mel_max) mel_max = v;
    }
    QASR_EXPECT(mel_max - mel_min > 0.1f);
}

QASR_TEST(RealWavMelSpectrogram1SecSlice) {
    /* Process only the first 1 second — fast, deterministic. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    /* Take first 16000 samples (1 sec). */
    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    /* 1 sec at 16kHz with 10ms hop → ~98 frames (approx). */
    QASR_EXPECT(n_frames >= 90 && n_frames <= 110);
    QASR_EXPECT_EQ(mel.size(), static_cast<std::size_t>(n_frames) * 80);
}

/* ========================================================================
 * Group 4 — CompactSilence
 * ======================================================================== */

QASR_TEST(RealWavCompactSilence) {
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* Work on a copy. */
    std::vector<float> audio = wav.samples;
    const auto original_size = audio.size();

    qasr::Status s = qasr::CompactSilence(&audio, -40.0f, 200, 50);
    QASR_EXPECT(s.ok());

    /* We expect the result to be no larger than the original. */
    QASR_EXPECT(audio.size() <= original_size);
    QASR_EXPECT(audio.size() > 0);

    /* All values still in [-1, 1]. */
    for (float v : audio) {
        QASR_EXPECT(std::isfinite(v));
        QASR_EXPECT(v >= -1.0f && v <= 1.0f);
    }
}

QASR_TEST(RealWavCompactSilenceTight) {
    /* Aggressive silence compaction: threshold=-20 dB. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio = wav.samples;
    const auto original_size = audio.size();

    qasr::Status s = qasr::CompactSilence(&audio, -20.0f, 100, 20);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(audio.size() <= original_size);
    QASR_EXPECT(audio.size() > 0);
}

/* ========================================================================
 * Group 5 — StreamingAudioRing with real data
 * ======================================================================== */

QASR_TEST(RealWavStreamingRingChunkedAppend) {
    /* Simulate streaming: feed real audio in 320-sample (20 ms) chunks. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    /* Use first 5 seconds max. */
    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }
    const std::size_t max_samples = std::min<std::size_t>(audio_16k.size(), 80000);

    /* 10-second ring buffer. */
    qasr::StreamingAudioRing ring(160000);
    const std::size_t chunk_size = 320;  /* 20 ms at 16 kHz */

    std::size_t fed = 0;
    while (fed < max_samples) {
        const std::size_t n = std::min(chunk_size, max_samples - fed);
        ring.Append(audio_16k.data() + fed, n);
        fed += n;
    }

    QASR_EXPECT_EQ(ring.total_appended(), fed);
    QASR_EXPECT(ring.current_size() == fed || ring.current_size() == ring.max_samples());

    std::vector<float> copied;
    ring.CopyTo(&copied);
    QASR_EXPECT_EQ(copied.size(), ring.current_size());
}

QASR_TEST(RealWavStreamingRingEviction) {
    /* Ring smaller than audio → eviction should occur. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    /* Small ring: 1 second. */
    const std::size_t ring_cap = 16000;
    qasr::StreamingAudioRing ring(ring_cap);

    /* Feed first 3 seconds. */
    const std::size_t feed = std::min<std::size_t>(audio_16k.size(), 48000);
    ring.Append(audio_16k.data(), feed);
    QASR_EXPECT_EQ(ring.current_size(), ring_cap);
    QASR_EXPECT_EQ(ring.total_appended(), feed);

    /* Content should be the LAST ring_cap samples. */
    std::vector<float> out;
    ring.CopyTo(&out);
    for (std::size_t i = 0; i < out.size(); ++i) {
        QASR_EXPECT(out[i] == audio_16k[feed - ring_cap + i]);
    }
}

QASR_TEST(RealWavStreamingMelPipeline) {
    /* Full streaming pipeline: ring buffer → mel spectrogram.
     * Simulates the path used for real-time inference. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    /* Feed first 2 seconds into ring. */
    const std::size_t two_sec = std::min<std::size_t>(audio_16k.size(), 32000);
    qasr::StreamingAudioRing ring(160000);
    ring.Append(audio_16k.data(), two_sec);

    /* Copy out and compute mel. */
    std::vector<float> buf;
    ring.CopyTo(&buf);
    QASR_EXPECT_EQ(buf.size(), two_sec);

    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(buf.data(), buf.size(), 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(n_frames > 0);
    QASR_EXPECT_EQ(mel.size(), static_cast<std::size_t>(n_frames) * 80);
}

/* ========================================================================
 * Group 6 — INT8 quantisation of real mel features
 *
 * The mel spectrogram represents realistic input distributions that the
 * decoder INT8 path would encounter.  These tests verify quantisation
 * accuracy on that data.
 * ======================================================================== */

#if defined(QASR_ENABLE_CPU_BACKEND) || defined(QASR_ONEDNN_AVAILABLE)

namespace {

/* Helper: BF16 encode/decode (same as onednn_int8_test.cc). */
std::uint16_t FloatToBf16R(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16ToFloatR(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

QASR_TEST(RealMelInt8QuantizeAccuracy) {
    /* Load audio → mel → treat first N mel frames as a BF16 weight matrix
     * → quantise → check round-trip accuracy. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    /* First 1 second mel. */
    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(n_frames > 0);

    /* Convert mel to BF16, treating [n_frames, 80] as a weight matrix. */
    const std::size_t rows = static_cast<std::size_t>(n_frames);
    const std::size_t cols = 80;
    std::vector<std::uint16_t> bf16(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        bf16[i] = FloatToBf16R(mel[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);
    QASR_EXPECT_EQ(w.rows, rows);
    QASR_EXPECT_EQ(w.cols, cols);

    /* Check round-trip accuracy.
     * Mel features are log-energy values with skewed per-row distributions
     * (a few large values, many small).  Per-row symmetric INT8 quantisation
     * loses precision on small values that share a row with a large peak.
     * We use a generous threshold: > 10% relative error, and accept up to
     * 15% of values exceeding that. */
    int large_error = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float original = Bf16ToFloatR(bf16[r * cols + c]);
            float dequant = static_cast<float>(w.data[r * cols + c]) * w.row_scale[r];
            float abs_err = std::fabs(original - dequant);
            if (std::fabs(original) > 0.5f && abs_err / std::fabs(original) > 0.10f) {
                ++large_error;
            }
        }
    }
    /* Allow up to 15% of values to exceed 10% relative error. */
    QASR_EXPECT(large_error < static_cast<int>(rows * cols) * 15 / 100);

    qwen_int8_weight_free(&w);
}

QASR_TEST(RealMelInt8AllValuesInRange) {
    /* All INT8 quantised values should be in [-127, 127]. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    const std::size_t rows = static_cast<std::size_t>(n_frames);
    const std::size_t cols = 80;
    std::vector<std::uint16_t> bf16(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        bf16[i] = FloatToBf16R(mel[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), rows, cols), 0);

    for (std::size_t i = 0; i < rows * cols; ++i) {
        QASR_EXPECT(w.data[i] >= -127 && w.data[i] <= 127);
    }
    for (std::size_t r = 0; r < rows; ++r) {
        QASR_EXPECT(w.row_scale[r] > 0.0f);
        QASR_EXPECT(std::isfinite(w.row_scale[r]));
    }

    qwen_int8_weight_free(&w);
}

#if defined(QASR_ONEDNN_AVAILABLE)

QASR_TEST(RealMelInt8MatmulDecode) {
    /* Full pipeline: audio → mel → BF16 weight → INT8 quantise →
     * oneDNN matmul execute (M = 1, simulating single-step decode). */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    /* Use first-second mel as a simulated weight [N, K]. */
    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    /* Treat as [N=n_frames, K=80]. */
    const std::size_t N = static_cast<std::size_t>(n_frames);
    const std::size_t K = 80;

    std::vector<std::uint16_t> bf16(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
        bf16[i] = FloatToBf16R(mel[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), N, K), 0);

    qwen_onednn_matmul_t * mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* Create a realistic input vector (K = 80). */
    std::vector<float> src(K, 0.0f);
    for (std::size_t i = 0; i < K; ++i) {
        src[i] = std::sin(static_cast<float>(i) * 0.1f);  /* smooth input */
    }

    std::vector<float> dst(N, 0.0f);
    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), 1, dst.data()), 0);

    /* Compare with float reference. */
    std::vector<float> dst_ref(N, 0.0f);
    for (std::size_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            sum += src[k] * Bf16ToFloatR(bf16[n * K + k]);
        }
        dst_ref[n] = sum;
    }

    int out_of_tol = 0;
    for (std::size_t n = 0; n < N; ++n) {
        float abs_err = std::fabs(dst[n] - dst_ref[n]);
        float denom = std::fabs(dst_ref[n]);
        if (denom > 0.1f && abs_err / denom > 0.10f) {
            ++out_of_tol;
        }
    }
    /* Allow up to 10% of outputs to exceed 10% relative error. */
    QASR_EXPECT(out_of_tol < static_cast<int>(N) / 10);

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(RealMelInt8MatmulPrefill) {
    /* Prefill path: M = 4 tokens at once. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    const std::size_t N = static_cast<std::size_t>(n_frames);
    const std::size_t K = 80;
    const int M = 4;

    std::vector<std::uint16_t> bf16(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
        bf16[i] = FloatToBf16R(mel[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), N, K), 0);

    qwen_onednn_matmul_t * mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    /* M = 4 input rows, each K = 80. */
    std::vector<float> src(M * K, 0.0f);
    for (std::size_t i = 0; i < static_cast<std::size_t>(M * K); ++i) {
        src[i] = std::cos(static_cast<float>(i) * 0.05f) * 0.5f;
    }

    std::vector<float> dst(M * N, 0.0f);
    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, src.data(), M, dst.data()), 0);

    /* Float reference. */
    std::vector<float> dst_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += src[m * K + k] * Bf16ToFloatR(bf16[n * K + k]);
            }
            dst_ref[m * N + n] = sum;
        }
    }

    int out_of_tol = 0;
    for (int i = 0; i < M * static_cast<int>(N); ++i) {
        float abs_err = std::fabs(dst[i] - dst_ref[i]);
        float denom = std::fabs(dst_ref[i]);
        if (denom > 0.1f && abs_err / denom > 0.10f) {
            ++out_of_tol;
        }
    }
    QASR_EXPECT(out_of_tol < static_cast<int>(M * N) / 10);

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

QASR_TEST(RealMelInt8MatvecConsistency) {
    /* qwen_int8_matvec should produce the same result as matmul_execute
     * when tested with mel-derived weights. */
    LoadedWav wav;
    if (!LoadTestWav(&wav)) return;

    std::vector<float> audio_16k;
    if (wav.sample_rate_hz != 16000) {
        qasr::Status rs = qasr::Resample(wav.samples, wav.sample_rate_hz, 16000, &audio_16k);
        QASR_EXPECT(rs.ok());
    } else {
        audio_16k = wav.samples;
    }

    const std::size_t one_sec = std::min<std::size_t>(audio_16k.size(), 16000);
    std::vector<float> mel;
    std::int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(
        audio_16k.data(), one_sec, 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());

    const std::size_t N = static_cast<std::size_t>(n_frames);
    const std::size_t K = 80;

    std::vector<std::uint16_t> bf16(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
        bf16[i] = FloatToBf16R(mel[i]);
    }

    qwen_int8_weight_t w = {};
    QASR_EXPECT_EQ(qwen_int8_quantize_bf16(&w, bf16.data(), N, K), 0);

    qwen_onednn_matmul_t * mm = qwen_onednn_matmul_create(&w);
    QASR_EXPECT(mm != nullptr);

    std::vector<float> x(K);
    for (std::size_t i = 0; i < K; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.2f);
    }

    std::vector<float> y1(N, 0.0f);
    std::vector<float> y2(N, 0.0f);

    QASR_EXPECT_EQ(qwen_onednn_matmul_execute(mm, x.data(), 1, y1.data()), 0);
    QASR_EXPECT_EQ(qwen_int8_matvec(mm, x.data(), 1, y2.data()), 0);

    for (std::size_t i = 0; i < N; ++i) {
        QASR_EXPECT(std::fabs(y1[i] - y2[i]) < 1e-6f);
    }

    qwen_onednn_matmul_free(mm);
    qwen_int8_weight_free(&w);
}

#endif  /* QASR_ONEDNN_AVAILABLE */
#endif  /* QASR_ENABLE_CPU_BACKEND || QASR_ONEDNN_AVAILABLE */
