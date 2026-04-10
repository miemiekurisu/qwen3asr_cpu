#include "qasr/audio/frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace qasr {

// --- WAV file format constants ---
static constexpr std::uint32_t kRiffTag = 0x46464952;  // "RIFF"
static constexpr std::uint32_t kWaveTag = 0x45564157;  // "WAVE"
static constexpr std::uint32_t kFmtTag = 0x20746D66;   // "fmt "
static constexpr std::uint32_t kDataTag = 0x61746164;   // "data"

namespace {

template <typename T>
T ReadLE(const uint8_t * p) {
    T value = 0;
    std::memcpy(&value, p, sizeof(T));
    return value;
}

}  // namespace

Status ReadWav(const std::string & path, std::vector<float> * samples,
               std::int32_t * sample_rate_hz) {
    if (!samples || !sample_rate_hz) {
        return Status(StatusCode::kInvalidArgument, "output pointers must not be null");
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return Status(StatusCode::kNotFound, "failed to open WAV file: " + path);
    }
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
    if (!file) {
        return Status(StatusCode::kInternal, "failed to read WAV file: " + path);
    }
    return ParseWavBuffer(buffer.data(), size, samples, sample_rate_hz);
}

Status ParseWavBuffer(const void * data, std::size_t size,
                      std::vector<float> * samples, std::int32_t * sample_rate_hz) {
    if (!data || !samples || !sample_rate_hz) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (size < 44) {
        return Status(StatusCode::kInvalidArgument, "WAV file too small");
    }
    const auto * raw = static_cast<const uint8_t *>(data);

    // Validate RIFF header
    if (ReadLE<std::uint32_t>(raw) != kRiffTag) {
        return Status(StatusCode::kInvalidArgument, "not a RIFF file");
    }
    if (ReadLE<std::uint32_t>(raw + 8) != kWaveTag) {
        return Status(StatusCode::kInvalidArgument, "not a WAVE file");
    }

    // Find fmt chunk
    std::size_t pos = 12;
    std::int32_t rate = 0;
    std::int16_t channels = 0;
    std::int16_t bits_per_sample = 0;
    std::int16_t audio_format = 0;
    bool found_fmt = false;

    while (pos + 8 <= size) {
        const std::uint32_t chunk_id = ReadLE<std::uint32_t>(raw + pos);
        const std::uint32_t chunk_size = ReadLE<std::uint32_t>(raw + pos + 4);
        if (chunk_id == kFmtTag && chunk_size >= 16) {
            audio_format = ReadLE<std::int16_t>(raw + pos + 8);
            channels = ReadLE<std::int16_t>(raw + pos + 10);
            rate = ReadLE<std::int32_t>(raw + pos + 12);
            bits_per_sample = ReadLE<std::int16_t>(raw + pos + 22);
            found_fmt = true;
        }
        if (chunk_id == kDataTag) {
            if (!found_fmt) {
                return Status(StatusCode::kInvalidArgument, "data chunk before fmt chunk");
            }
            if (audio_format != 1) {
                return Status(StatusCode::kInvalidArgument, "only PCM WAV supported");
            }
            const std::size_t data_start = pos + 8;
            const std::size_t data_bytes = std::min(static_cast<std::size_t>(chunk_size),
                                                     size - data_start);
            const std::int32_t bytes_per_sample = bits_per_sample / 8;
            if (bytes_per_sample <= 0) {
                return Status(StatusCode::kInvalidArgument, "invalid bits_per_sample");
            }
            const std::size_t total_samples = data_bytes / static_cast<std::size_t>(bytes_per_sample);
            const std::size_t frame_count = (channels > 0) ? total_samples / static_cast<std::size_t>(channels) : 0;

            samples->resize(frame_count);
            *sample_rate_hz = rate;

            for (std::size_t i = 0; i < frame_count; ++i) {
                float sum = 0.0f;
                for (std::int16_t ch = 0; ch < channels; ++ch) {
                    const std::size_t offset = data_start +
                        (i * static_cast<std::size_t>(channels) + static_cast<std::size_t>(ch)) *
                        static_cast<std::size_t>(bytes_per_sample);
                    if (bits_per_sample == 16) {
                        std::int16_t val = ReadLE<std::int16_t>(raw + offset);
                        sum += static_cast<float>(val) / 32768.0f;
                    } else if (bits_per_sample == 32) {
                        std::int32_t val = ReadLE<std::int32_t>(raw + offset);
                        sum += static_cast<float>(val) / 2147483648.0f;
                    } else if (bits_per_sample == 8) {
                        uint8_t val = raw[offset];
                        sum += (static_cast<float>(val) - 128.0f) / 128.0f;
                    } else {
                        return Status(StatusCode::kInvalidArgument,
                                      "unsupported bits_per_sample: " + std::to_string(bits_per_sample));
                    }
                }
                (*samples)[i] = sum / static_cast<float>(channels);
            }
            return OkStatus();
        }
        pos += 8 + chunk_size;
        // Align to 2-byte boundary
        if (pos % 2 != 0) ++pos;
    }
    return Status(StatusCode::kInvalidArgument, "no data chunk found in WAV");
}

Status Resample(const std::vector<float> & input, std::int32_t source_rate,
                std::int32_t target_rate, std::vector<float> * output) {
    if (!output) {
        return Status(StatusCode::kInvalidArgument, "output must not be null");
    }
    if (source_rate <= 0 || target_rate <= 0) {
        return Status(StatusCode::kInvalidArgument, "sample rates must be positive");
    }
    if (input.empty()) {
        output->clear();
        return OkStatus();
    }
    if (source_rate == target_rate) {
        *output = input;
        return OkStatus();
    }

    // Linear interpolation resampler
    const double ratio = static_cast<double>(source_rate) / static_cast<double>(target_rate);
    const auto out_len = static_cast<std::size_t>(
        static_cast<double>(input.size()) / ratio);
    output->resize(out_len);

    for (std::size_t i = 0; i < out_len; ++i) {
        const double src_pos = static_cast<double>(i) * ratio;
        const auto idx = static_cast<std::size_t>(src_pos);
        const double frac = src_pos - static_cast<double>(idx);
        if (idx + 1 < input.size()) {
            (*output)[i] = static_cast<float>(
                static_cast<double>(input[idx]) * (1.0 - frac) +
                static_cast<double>(input[idx + 1]) * frac);
        } else {
            (*output)[i] = input[idx < input.size() ? idx : input.size() - 1];
        }
    }
    return OkStatus();
}

Status ComputeMelSpectrogram(const float * samples, std::size_t n_samples,
                             std::int32_t mel_bins, std::int32_t * out_frames,
                             std::vector<float> * mel) {
    if (!samples || !out_frames || !mel) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (n_samples == 0) {
        return Status(StatusCode::kInvalidArgument, "no audio samples");
    }
    if (mel_bins <= 0) {
        return Status(StatusCode::kInvalidArgument, "mel_bins must be positive");
    }

    // Standard parameters for speech: 25ms window, 10ms hop, 16kHz
    constexpr std::int32_t kSampleRate = 16000;
    constexpr std::int32_t kWindowSamples = 400;   // 25ms
    constexpr std::int32_t kHopSamples = 160;      // 10ms
    constexpr float kPreemphasis = 0.97f;

    const auto n_frames = static_cast<std::int32_t>(
        (static_cast<std::int64_t>(n_samples) - kWindowSamples + kHopSamples) / kHopSamples);
    if (n_frames <= 0) {
        return Status(StatusCode::kInvalidArgument, "audio too short for mel computation");
    }
    *out_frames = n_frames;

    const auto total = static_cast<std::size_t>(n_frames) * static_cast<std::size_t>(mel_bins);
    mel->resize(total, 0.0f);

    // Compute power spectrum per frame and apply mel filterbank
    // This is a simplified version - the real implementation uses FFT
    for (std::int32_t f = 0; f < n_frames; ++f) {
        const std::size_t start = static_cast<std::size_t>(f) * kHopSamples;
        float frame_energy = 0.0f;
        float prev = 0.0f;
        for (std::int32_t i = 0; i < kWindowSamples; ++i) {
            const std::size_t idx = start + static_cast<std::size_t>(i);
            const float s = (idx < n_samples) ? samples[idx] : 0.0f;
            const float preemph = s - kPreemphasis * prev;
            prev = s;
            frame_energy += preemph * preemph;
        }
        // Distribute energy across mel bins (simplified triangular filterbank)
        const float log_energy = std::log(frame_energy / static_cast<float>(kWindowSamples) + 1e-10f);
        (void)kSampleRate;
        for (std::int32_t m = 0; m < mel_bins; ++m) {
            // Triangular filter response (simplified)
            const float center = static_cast<float>(m + 1) / static_cast<float>(mel_bins + 1);
            const float weight = std::max(0.0f, 1.0f - std::abs(center - 0.5f) * 2.0f);
            (*mel)[static_cast<std::size_t>(f) * static_cast<std::size_t>(mel_bins) +
                   static_cast<std::size_t>(m)] = log_energy * weight;
        }
    }
    return OkStatus();
}

Status CompactSilence(std::vector<float> * samples, float threshold_db,
                      std::int32_t min_silence_ms, std::int32_t keep_ms) {
    if (!samples) {
        return Status(StatusCode::kInvalidArgument, "samples must not be null");
    }
    if (samples->empty()) {
        return OkStatus();
    }
    if (threshold_db > 0.0f) {
        return Status(StatusCode::kInvalidArgument, "threshold_db should be negative (dB)");
    }

    constexpr std::int32_t kSampleRate = 16000;
    const auto min_silence_samples = static_cast<std::size_t>(
        kSampleRate * min_silence_ms / 1000);
    const auto keep_samples = static_cast<std::size_t>(
        kSampleRate * keep_ms / 1000);

    const float threshold_linear = std::pow(10.0f, threshold_db / 20.0f);

    std::vector<float> output;
    output.reserve(samples->size());

    std::size_t silence_run = 0;
    for (std::size_t i = 0; i < samples->size(); ++i) {
        const float abs_sample = std::abs((*samples)[i]);
        if (abs_sample < threshold_linear) {
            ++silence_run;
            if (silence_run <= keep_samples || silence_run < min_silence_samples) {
                output.push_back((*samples)[i]);
            }
        } else {
            silence_run = 0;
            output.push_back((*samples)[i]);
        }
    }
    *samples = std::move(output);
    return OkStatus();
}

// --- StreamingAudioRing ---

StreamingAudioRing::StreamingAudioRing(std::size_t max_samples)
    : max_samples_(max_samples) {
    buffer_.reserve(max_samples);
}

void StreamingAudioRing::Append(const float * data, std::size_t count) {
    if (!data || count == 0) return;
    total_appended_ += count;

    if (count >= max_samples_) {
        // Take only the last max_samples_ from data
        buffer_.assign(data + count - max_samples_, data + count);
        return;
    }

    const std::size_t new_total = buffer_.size() + count;
    if (new_total > max_samples_) {
        const std::size_t evict = new_total - max_samples_;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(evict));
    }
    buffer_.insert(buffer_.end(), data, data + count);
}

std::size_t StreamingAudioRing::CopyTo(std::vector<float> * output) const {
    if (!output) return 0;
    *output = buffer_;
    return buffer_.size();
}

void StreamingAudioRing::Clear() {
    buffer_.clear();
    total_appended_ = 0;
}

}  // namespace qasr
