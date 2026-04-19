#include "qasr/audio/audio_convert.h"
#include "qasr/audio/frontend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace qasr {
namespace {

namespace fs = std::filesystem;

std::string ToLower(std::string_view s) {
    std::string result(s);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

}  // namespace

bool IsWavFile(const std::string & path) noexcept {
    const fs::path p(path);
    const std::string ext = ToLower(p.extension().string());
    return ext == ".wav" || ext == ".wave";
}

bool FfmpegAvailable() noexcept {
#ifdef _WIN32
    const int rc = std::system("where ffmpeg >nul 2>&1");
#else
    const int rc = std::system("which ffmpeg >/dev/null 2>&1");
#endif
    return rc == 0;
}

Status ConvertToWav(const std::string & input_path,
                    const std::string & output_wav_path) {
    if (input_path.empty()) {
        return Status(StatusCode::kInvalidArgument, "input_path must not be empty");
    }
    if (output_wav_path.empty()) {
        return Status(StatusCode::kInvalidArgument, "output_wav_path must not be empty");
    }
    if (!fs::exists(input_path)) {
        return Status(StatusCode::kNotFound, "input file not found: " + input_path);
    }
    if (!FfmpegAvailable()) {
        return Status(StatusCode::kFailedPrecondition, "ffmpeg not found in PATH");
    }

    // Build ffmpeg command: convert to 16kHz mono s16le WAV
    // -y: overwrite output, -loglevel error: suppress info
    std::string cmd = "ffmpeg -y -loglevel error -i \"" + input_path +
                      "\" -ar 16000 -ac 1 -c:a pcm_s16le \"" + output_wav_path + "\"";

#ifdef _WIN32
    cmd += " 2>nul";
#else
    cmd += " 2>/dev/null";
#endif

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        return Status(StatusCode::kInternal,
                      "ffmpeg conversion failed (exit " + std::to_string(rc) + ")");
    }
    if (!fs::exists(output_wav_path)) {
        return Status(StatusCode::kInternal, "ffmpeg produced no output file");
    }
    return OkStatus();
}

Status LoadAudioFile(const std::string & path,
                     std::vector<float> * samples,
                     std::int64_t * duration_ms) {
    if (!samples || !duration_ms) {
        return Status(StatusCode::kInvalidArgument, "output pointers must not be null");
    }
    if (path.empty()) {
        return Status(StatusCode::kInvalidArgument, "path must not be empty");
    }
    if (!fs::exists(path)) {
        return Status(StatusCode::kNotFound, "file not found: " + path);
    }

    constexpr std::int32_t kTargetRate = 16000;

    if (IsWavFile(path)) {
        // Direct WAV read
        std::int32_t sample_rate = 0;
        Status status = ReadWav(path, samples, &sample_rate);
        if (!status.ok()) return status;

        if (sample_rate != kTargetRate) {
            std::vector<float> resampled;
            status = Resample(*samples, sample_rate, kTargetRate, &resampled);
            if (!status.ok()) return status;
            *samples = std::move(resampled);
        }
        *duration_ms = static_cast<std::int64_t>(samples->size()) * 1000 / kTargetRate;
        return OkStatus();
    }

    // Non-WAV: convert via ffmpeg to temp WAV, then read
    const fs::path input_path(path);
    const fs::path temp_dir = fs::temp_directory_path();
    const fs::path temp_wav = temp_dir / (input_path.stem().string() + "_qasr_tmp.wav");
    const std::string temp_wav_str = temp_wav.string();

    Status status = ConvertToWav(path, temp_wav_str);
    if (!status.ok()) return status;

    std::int32_t sample_rate = 0;
    status = ReadWav(temp_wav_str, samples, &sample_rate);

    // Clean up temp file regardless of result
    std::error_code ec;
    fs::remove(temp_wav, ec);

    if (!status.ok()) return status;

    if (sample_rate != kTargetRate) {
        std::vector<float> resampled;
        status = Resample(*samples, sample_rate, kTargetRate, &resampled);
        if (!status.ok()) return status;
        *samples = std::move(resampled);
    }
    *duration_ms = static_cast<std::int64_t>(samples->size()) * 1000 / kTargetRate;
    return OkStatus();
}

}  // namespace qasr
