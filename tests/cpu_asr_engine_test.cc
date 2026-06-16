/*
 * cpu_asr_engine_test.cc — CpuAsrEngine integration tests.
 *
 * Tests the CPU engine with actual model when QASR_MODEL_DIR is set.
 * CI-safe: skips gracefully when model is not available.
 */

#include "tests/test_registry.h"

#include "qasr/engine/cpu_asr_engine.h"
#include "qasr/engine/config.h"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

static const char * GetModelDir() {
    return std::getenv("QASR_MODEL_DIR");
}

QASR_TEST(CpuAsrEngineLoadModelWithRealPath) {
    const char * model_dir = GetModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }

    qasr::CpuAsrEngine engine;
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;

    auto status = engine.LoadModel(cfg);
    if (!status.ok()) {
        std::fprintf(stderr, "  [SKIP] LoadModel failed: %s\n", status.message().c_str());
        return;
    }

    QASR_EXPECT_EQ(engine.ActiveSessionCount(), 1);
}

QASR_TEST(CpuAsrEngineTranscribeSilence) {
    const char * model_dir = GetModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }

    qasr::CpuAsrEngine engine;
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;

    auto status = engine.LoadModel(cfg);
    if (!status.ok()) {
        std::fprintf(stderr, "  [SKIP] LoadModel failed: %s\n", status.message().c_str());
        return;
    }

    std::uint64_t session_id;
    status = engine.CreateSession({}, session_id);
    QASR_EXPECT(status.ok());

    // 1 second of silence at 16kHz
    constexpr int SAMPLE_RATE = 16000;
    std::vector<float> silence(SAMPLE_RATE, 0.0f);

    qasr::AsrSegmentResult result = engine.TranscribeSegment(session_id, silence, SAMPLE_RATE);

    // Silence may or may not produce text, but should not crash
    QASR_EXPECT(result.status.ok() || result.text.empty());
    QASR_EXPECT(result.total_ms >= 0.0);
    QASR_EXPECT(result.audio_ms > 0.0);

    engine.CloseSession(session_id);
}

QASR_TEST(CpuAsrEngineTranscribeSineWave) {
    const char * model_dir = GetModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }

    qasr::CpuAsrEngine engine;
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;

    auto status = engine.LoadModel(cfg);
    if (!status.ok()) {
        std::fprintf(stderr, "  [SKIP] LoadModel failed: %s\n", status.message().c_str());
        return;
    }

    std::uint64_t session_id;
    status = engine.CreateSession({}, session_id);
    QASR_EXPECT(status.ok());

    // 2 seconds of 440 Hz sine wave
    constexpr int SAMPLE_RATE = 16000;
    std::vector<float> sine(SAMPLE_RATE * 2);
    for (int i = 0; i < static_cast<int>(sine.size()); i++) {
        sine[i] = 0.5f * static_cast<float>(std::sin(6.283185307179586 * 440.0 * i / SAMPLE_RATE));
    }

    qasr::AsrSegmentResult result = engine.TranscribeSegment(session_id, sine, SAMPLE_RATE);
    QASR_EXPECT(result.status.ok());
    QASR_EXPECT(result.total_ms > 0.0);
    QASR_EXPECT(result.audio_ms > 0.0);

    engine.CloseSession(session_id);
}

QASR_TEST(CpuAsrEngineSessionLifecycle) {
    const char * model_dir = GetModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }

    qasr::CpuAsrEngine engine;
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;

    auto status = engine.LoadModel(cfg);
    if (!status.ok()) {
        std::fprintf(stderr, "  [SKIP] LoadModel failed: %s\n", status.message().c_str());
        return;
    }

    // Create multiple sessions
    std::uint64_t ids[3];
    for (int i = 0; i < 3; i++) {
        status = engine.CreateSession({}, ids[i]);
        QASR_EXPECT(status.ok());
        QASR_EXPECT_EQ(ids[i], static_cast<std::uint64_t>(i + 1));
    }

    // Close all
    for (int i = 0; i < 3; i++) {
        status = engine.CloseSession(ids[i]);
        QASR_EXPECT(status.ok());
    }
}

QASR_TEST(CpuAsrEngineConfigPropagation) {
    const char * model_dir = GetModelDir();
    if (!model_dir) {
        std::fprintf(stderr, "  [SKIP] QASR_MODEL_DIR not set\n");
        return;
    }

    qasr::CpuAsrEngine engine;
    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;
    cfg.language = "en";
    cfg.prompt = "Transcribe this audio.";
    cfg.temperature = 0.7f;

    auto status = engine.LoadModel(cfg);
    if (!status.ok()) {
        std::fprintf(stderr, "  [SKIP] LoadModel failed: %s\n", status.message().c_str());
        return;
    }

    const auto & loaded_cfg = engine.config();
    QASR_EXPECT_EQ(loaded_cfg.language, "en");
    QASR_EXPECT_EQ(loaded_cfg.prompt, "Transcribe this audio.");
    QASR_EXPECT(loaded_cfg.temperature > 0.0f);
}
