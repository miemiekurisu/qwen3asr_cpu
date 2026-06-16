#include "qasr/engine/asr_engine.h"
#include "qasr/engine/config.h"
#include "qasr/audio/audio_convert.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

static std::vector<float> load_wav_samples(const char * path) {
    std::vector<float> samples;
    std::int64_t duration_ms = 0;
    auto status = qasr::LoadAudioFile(path, &samples, &duration_ms);
    if (!status.ok()) {
        std::cerr << "LoadAudioFile: " << status.ToString() << "\n";
        exit(1);
    }
    std::cout << "Audio: " << samples.size() << " samples, "
              << duration_ms << " ms\n";
    return samples;
}

static void run_transcription(const std::string & model_dir,
                               const std::string & audio_path,
                               qasr::BackendKind backend) {
    auto engine = qasr::CreateEngine(backend);
    if (!engine) {
        std::cerr << "CreateEngine(" << (int)backend << ") returned null\n";
        exit(1);
    }

    qasr::V2EngineConfig cfg;
    cfg.model_dir = model_dir;
    cfg.backend = backend;
    cfg.allow_backend_fallback = true;
    cfg.verbosity = 1;

    auto t0 = std::chrono::steady_clock::now();
    auto status = engine->LoadModel(cfg);
    if (!status.ok()) {
        std::cerr << "LoadModel: " << status.ToString() << "\n";
        exit(1);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "LoadModel: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n";

    qasr::SessionOptions opts;
    /* Language auto-detection: leave empty or set to match audio content */
    opts.language = "";
    std::uint64_t sid;
    status = engine->CreateSession(opts, sid);
    if (!status.ok()) {
        std::cerr << "CreateSession: " << status.ToString() << "\n";
        exit(1);
    }
    std::cout << "Session created: id=" << sid << "\n";

    auto samples = load_wav_samples(audio_path.c_str());

    t0 = std::chrono::steady_clock::now();
    auto result = engine->TranscribeSegment(sid, samples, 16000);
    t1 = std::chrono::steady_clock::now();
    std::cout << "TranscribeSegment: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n";

    if (!result.status.ok()) {
        std::cerr << "TranscribeSegment: " << result.status.ToString() << "\n";
    } else {
        std::cout << "Text: [" << result.text << "]\n";
        std::cout << "Tokens: " << result.text_tokens << "\n";
        std::cout << "Audio: " << result.audio_ms << " ms\n";
        std::cout << "Encode: " << result.encode_ms << " ms\n";
        std::cout << "Decode: " << result.decode_ms << " ms\n";
        std::cout << "Total: " << result.total_ms << " ms\n";
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <audio.wav> [cpu|cuda|verify]\n";
        return 1;
    }

    std::string model_dir = argv[1];
    std::string audio_path = argv[2];
    qasr::BackendKind backend = qasr::BackendKind::kCpu;

    if (argc >= 4) {
        std::string bk = argv[3];
        if (bk == "verify") {
            std::cout << "=== VERIFY: CPU vs CUDA ===\n";
            run_transcription(model_dir, audio_path, qasr::BackendKind::kCpu);
            std::cout << "\n=== --- CUDA --- ===\n";
            run_transcription(model_dir, audio_path, qasr::BackendKind::kCuda);
            return 0;
        }
        backend = qasr::ParseBackendKind(bk);
    }

    const char * name = (backend == qasr::BackendKind::kCuda) ? "CUDA" : "CPU";
    std::cout << "=== " << name << " backend ===\n";

    run_transcription(model_dir, audio_path, backend);
    return 0;
}
