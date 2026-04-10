#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qasr/core/audio_types.h"
#include "qasr/core/status.h"

namespace qasr {

/// WAV file reader. Extracts mono float32 samples at 16 kHz.
/// Pre: path must be a readable WAV file.
/// Post: samples are in [-1, 1] range, mono, 16 kHz.
/// Thread-safe: yes (stateless after call).
Status ReadWav(const std::string & path, std::vector<float> * samples, std::int32_t * sample_rate_hz);

/// Parse a WAV buffer in memory.
/// Pre: data must contain a valid WAV with RIFF header.
/// Post: same as ReadWav.
/// Thread-safe: yes.
Status ParseWavBuffer(const void * data, std::size_t size,
                      std::vector<float> * samples, std::int32_t * sample_rate_hz);

/// Resample audio from source rate to target rate.
/// Pre: source_rate > 0, target_rate > 0.
/// Post: output sized proportionally.
/// Thread-safe: yes.
Status Resample(const std::vector<float> & input, std::int32_t source_rate,
                std::int32_t target_rate, std::vector<float> * output);

/// Compute log-mel spectrogram from 16 kHz mono audio.
/// Pre: audio must be mono 16 kHz float32.
/// Post: output is [mel_bins, n_frames].
/// Thread-safe: yes.
Status ComputeMelSpectrogram(const float * samples, std::size_t n_samples,
                             std::int32_t mel_bins, std::int32_t * out_frames,
                             std::vector<float> * mel);

/// Compact silence regions by replacing long silence runs with short ones.
/// Pre: samples must be mono audio, threshold >= 0.
/// Post: modifies samples in-place, returns new length.
/// Thread-safe: yes.
Status CompactSilence(std::vector<float> * samples, float threshold_db,
                      std::int32_t min_silence_ms, std::int32_t keep_ms);

/// Ring buffer for streaming audio ingestion.
/// Pre: max_samples > 0.
/// Post: oldest samples are evicted when capacity exceeded.
/// Thread-safe: NOT thread-safe; caller must synchronize.
class StreamingAudioRing {
public:
    explicit StreamingAudioRing(std::size_t max_samples);

    void Append(const float * data, std::size_t count);
    std::size_t CopyTo(std::vector<float> * output) const;
    void Clear();

    std::size_t total_appended() const noexcept { return total_appended_; }
    std::size_t current_size() const noexcept { return buffer_.size(); }
    std::size_t max_samples() const noexcept { return max_samples_; }

private:
    std::vector<float> buffer_;
    std::size_t max_samples_;
    std::size_t total_appended_ = 0;
};

}  // namespace qasr
