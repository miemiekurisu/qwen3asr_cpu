#pragma once

#include <cstdint>
#include <string>

#include "qasr/core/status.h"
#include "qasr/inference/aligner_types.h"

namespace qasr {

/// Configuration for the native ForcedAligner backend.
struct AlignerConfig {
    std::string model_dir;    // path to ForcedAligner model directory
    std::int32_t threads = 0; // 0 = auto
};

/// Validate aligner configuration.
Status ValidateAlignerConfig(const AlignerConfig & config);

/// Native ForcedAligner wrapping the C qwen_forced_align() backend.
///
/// Ownership: owns the loaded model context.
/// Thread-safety: NOT thread-safe; use one instance per thread.
class ForcedAligner {
public:
    ForcedAligner();
    ~ForcedAligner();

    ForcedAligner(const ForcedAligner &) = delete;
    ForcedAligner & operator=(const ForcedAligner &) = delete;
    ForcedAligner(ForcedAligner &&) noexcept;
    ForcedAligner & operator=(ForcedAligner &&) noexcept;

    /// Load the ForcedAligner model.
    Status Load(const AlignerConfig & config);

    /// Release model resources.
    void Unload();

    /// Check if model is loaded.
    bool IsLoaded() const noexcept;

    /// Align a WAV audio file against its transcript.
    /// Pre: IsLoaded() == true. audio_path must be a WAV file.
    Status Align(const std::string & audio_path,
                 const std::string & text,
                 const std::string & language,
                 AlignResult * result);

    /// Align raw audio samples (16 kHz mono float32) against a transcript.
    /// Pre: IsLoaded() == true. samples must not be null when n_samples > 0.
    Status AlignSamples(const float * samples, int n_samples,
                        const std::string & text,
                        const std::string & language,
                        AlignResult * result);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

}  // namespace qasr
