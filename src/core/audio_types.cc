#include "qasr/core/audio_types.h"

namespace qasr {

Status ValidateAudioSpan(const AudioSpan & audio) {
    if (audio.sample_count < 0) {
        return Status(StatusCode::kInvalidArgument, "sample_count must be >= 0");
    }
    if (audio.sample_rate_hz <= 0) {
        return Status(StatusCode::kInvalidArgument, "sample_rate_hz must be > 0");
    }
    if (audio.channels <= 0) {
        return Status(StatusCode::kInvalidArgument, "channels must be > 0");
    }
    if (audio.sample_count > 0 && audio.samples == nullptr) {
        return Status(StatusCode::kInvalidArgument, "samples must not be null when sample_count > 0");
    }
    return OkStatus();
}

bool IsMono16kAudio(const AudioSpan & audio) noexcept {
    return audio.sample_rate_hz == 16000 && audio.channels == 1;
}

std::int64_t AudioDurationMs(const AudioSpan & audio) noexcept {
    if (audio.sample_rate_hz <= 0 || audio.channels <= 0 || audio.sample_count <= 0) {
        return 0;
    }
    const std::int64_t frames = audio.sample_count / audio.channels;
    return (frames * 1000) / audio.sample_rate_hz;
}

}  // namespace qasr
