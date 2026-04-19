#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Check if a file extension indicates a WAV file.
/// Thread-safe: yes.
bool IsWavFile(const std::string & path) noexcept;

/// Check if ffmpeg is available on the system PATH.
/// Thread-safe: yes.
bool FfmpegAvailable() noexcept;

/// Convert any audio/video file to 16 kHz mono WAV via ffmpeg.
/// Pre: ffmpeg must be in PATH; input_path must exist.
/// Post: output_wav_path contains a valid PCM WAV file.
/// Thread-safe: yes (uses unique temp files).
Status ConvertToWav(const std::string & input_path,
                    const std::string & output_wav_path);

/// Load audio from any supported format (WAV native, others via ffmpeg).
/// Pre: path must exist and be a readable file.
/// Post: samples contain mono float32 at 16 kHz, duration_ms set.
/// Thread-safe: yes.
Status LoadAudioFile(const std::string & path,
                     std::vector<float> * samples,
                     std::int64_t * duration_ms);

}  // namespace qasr
