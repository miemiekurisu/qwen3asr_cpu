#include "tests/test_registry.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qasr/audio/audio_convert.h"

namespace {
namespace fs = std::filesystem;

// Create a minimal valid 16kHz mono 16-bit PCM WAV in memory.
std::vector<uint8_t> MakeTestWav(const std::vector<int16_t> & pcm, int32_t sample_rate = 16000) {
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    const uint32_t file_size = 36 + data_size;
    std::vector<uint8_t> buf(44 + pcm.size() * 2);
    auto write16 = [&](size_t off, uint16_t v) { std::memcpy(&buf[off], &v, 2); };
    auto write32 = [&](size_t off, uint32_t v) { std::memcpy(&buf[off], &v, 4); };

    buf[0]='R'; buf[1]='I'; buf[2]='F'; buf[3]='F';
    write32(4, file_size);
    buf[8]='W'; buf[9]='A'; buf[10]='V'; buf[11]='E';
    buf[12]='f'; buf[13]='m'; buf[14]='t'; buf[15]=' ';
    write32(16, 16);
    write16(20, 1);
    write16(22, 1);
    write32(24, static_cast<uint32_t>(sample_rate));
    write32(28, static_cast<uint32_t>(sample_rate * 2));
    write16(32, 2);
    write16(34, 16);
    buf[36]='d'; buf[37]='a'; buf[38]='t'; buf[39]='a';
    write32(40, data_size);
    std::memcpy(&buf[44], pcm.data(), pcm.size() * 2);
    return buf;
}

std::string WriteTempFile(const std::string & name, const std::vector<uint8_t> & data) {
    const fs::path temp = fs::temp_directory_path() / ("qasr_test_" + name);
    std::ofstream out(temp, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return temp.string();
}

void RemoveTempFile(const std::string & path) {
    std::error_code ec;
    fs::remove(path, ec);
}

}  // namespace

// ========================================================================
// IsWavFile
// ========================================================================

// Normal
QASR_TEST(IsWavFile_NormalWav) {
    QASR_EXPECT(qasr::IsWavFile("test.wav"));
    QASR_EXPECT(qasr::IsWavFile("test.WAV"));
    QASR_EXPECT(qasr::IsWavFile("test.wave"));
    QASR_EXPECT(qasr::IsWavFile("path/to/audio.wav"));
}

QASR_TEST(IsWavFile_NormalNonWav) {
    QASR_EXPECT(!qasr::IsWavFile("test.mp3"));
    QASR_EXPECT(!qasr::IsWavFile("test.flac"));
    QASR_EXPECT(!qasr::IsWavFile("test.mp4"));
    QASR_EXPECT(!qasr::IsWavFile("test.ogg"));
}

// Extreme: empty
QASR_TEST(IsWavFile_EmptyString) {
    QASR_EXPECT(!qasr::IsWavFile(""));
}

// Extreme: no extension
QASR_TEST(IsWavFile_NoExtension) {
    QASR_EXPECT(!qasr::IsWavFile("wavfile"));
    QASR_EXPECT(!qasr::IsWavFile("."));
}

// Error: tricky names
QASR_TEST(IsWavFile_TrickyNames) {
    QASR_EXPECT(!qasr::IsWavFile("test.wav.mp3"));
    QASR_EXPECT(qasr::IsWavFile("test.mp3.wav"));
}

// Random: fuzz
QASR_TEST(IsWavFile_RandomPaths) {
    std::srand(77);
    for (int i = 0; i < 100; ++i) {
        std::string s;
        const int len = std::rand() % 50;
        for (int j = 0; j < len; ++j) {
            s += static_cast<char>(32 + std::rand() % 95);
        }
        // Should not crash
        (void)qasr::IsWavFile(s);
    }
}

// ========================================================================
// FfmpegAvailable
// ========================================================================

// Normal: just call it, should not crash
QASR_TEST(FfmpegAvailable_DoesNotCrash) {
    // Returns true or false depending on system; must not throw/crash
    (void)qasr::FfmpegAvailable();
}

// ========================================================================
// ConvertToWav
// ========================================================================

// Error: empty paths
QASR_TEST(ConvertToWav_EmptyInput) {
    QASR_EXPECT_EQ(qasr::ConvertToWav("", "out.wav").code(), qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(ConvertToWav_EmptyOutput) {
    QASR_EXPECT_EQ(qasr::ConvertToWav("in.mp3", "").code(), qasr::StatusCode::kInvalidArgument);
}

// Error: nonexistent input
QASR_TEST(ConvertToWav_NonexistentInput) {
    QASR_EXPECT_EQ(qasr::ConvertToWav("nonexistent_file_xyz.mp3", "out.wav").code(),
                   qasr::StatusCode::kNotFound);
}

// ========================================================================
// LoadAudioFile
// ========================================================================

// Error: null outputs
QASR_TEST(LoadAudioFile_NullSamples) {
    std::int64_t dur = 0;
    QASR_EXPECT_EQ(qasr::LoadAudioFile("test.wav", nullptr, &dur).code(),
                   qasr::StatusCode::kInvalidArgument);
}

QASR_TEST(LoadAudioFile_NullDuration) {
    std::vector<float> samples;
    QASR_EXPECT_EQ(qasr::LoadAudioFile("test.wav", &samples, nullptr).code(),
                   qasr::StatusCode::kInvalidArgument);
}

// Error: empty path
QASR_TEST(LoadAudioFile_EmptyPath) {
    std::vector<float> samples;
    std::int64_t dur = 0;
    QASR_EXPECT_EQ(qasr::LoadAudioFile("", &samples, &dur).code(),
                   qasr::StatusCode::kInvalidArgument);
}

// Error: nonexistent file
QASR_TEST(LoadAudioFile_NonexistentFile) {
    std::vector<float> samples;
    std::int64_t dur = 0;
    QASR_EXPECT_EQ(qasr::LoadAudioFile("nonexistent_xyz.wav", &samples, &dur).code(),
                   qasr::StatusCode::kNotFound);
}

// Normal: load a real WAV from memory-created temp file
QASR_TEST(LoadAudioFile_NormalWav) {
    // 1 second of 16kHz silence
    std::vector<int16_t> pcm(16000, 0);
    pcm[8000] = 1000;  // one blip
    const auto wav = MakeTestWav(pcm);
    const std::string path = WriteTempFile("load_normal.wav", wav);

    std::vector<float> samples;
    std::int64_t dur = 0;
    qasr::Status s = qasr::LoadAudioFile(path, &samples, &dur);
    RemoveTempFile(path);

    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(dur, std::int64_t(1000));
    QASR_EXPECT_EQ(samples.size(), std::size_t(16000));
}

// Extreme: very short WAV (1 sample)
QASR_TEST(LoadAudioFile_SingleSampleWav) {
    std::vector<int16_t> pcm = {12345};
    const auto wav = MakeTestWav(pcm);
    const std::string path = WriteTempFile("single_sample.wav", wav);

    std::vector<float> samples;
    std::int64_t dur = 0;
    qasr::Status s = qasr::LoadAudioFile(path, &samples, &dur);
    RemoveTempFile(path);

    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(samples.size(), std::size_t(1));
    QASR_EXPECT_EQ(dur, std::int64_t(0));  // 1/16000 * 1000 = 0 (integer truncation)
}

// Random: generate random PCM and verify round-trip
QASR_TEST(LoadAudioFile_RandomPcm) {
    std::srand(55);
    for (int trial = 0; trial < 5; ++trial) {
        const int n = 100 + std::rand() % 10000;
        std::vector<int16_t> pcm(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            pcm[static_cast<std::size_t>(j)] = static_cast<int16_t>(std::rand() % 65536 - 32768);
        }
        const auto wav = MakeTestWav(pcm);
        const std::string path = WriteTempFile("random_" + std::to_string(trial) + ".wav", wav);

        std::vector<float> samples;
        std::int64_t dur = 0;
        qasr::Status s = qasr::LoadAudioFile(path, &samples, &dur);
        RemoveTempFile(path);

        QASR_EXPECT(s.ok());
        QASR_EXPECT_EQ(samples.size(), static_cast<std::size_t>(n));
    }
}
