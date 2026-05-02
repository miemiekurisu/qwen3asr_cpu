#include "tests/test_registry.h"
#include "tests/test_paths.h"
#include "qasr/audio/frontend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

// Create a minimal valid 16kHz mono 16-bit PCM WAV in memory.
std::vector<uint8_t> MakeMinimalWav(const std::vector<int16_t> & pcm, int32_t sample_rate = 16000) {
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    const uint32_t file_size = 36 + data_size;
    std::vector<uint8_t> buf(44 + pcm.size() * 2);
    auto write16 = [&](size_t off, uint16_t v) { std::memcpy(&buf[off], &v, 2); };
    auto write32 = [&](size_t off, uint32_t v) { std::memcpy(&buf[off], &v, 4); };

    // RIFF header
    buf[0]='R'; buf[1]='I'; buf[2]='F'; buf[3]='F';
    write32(4, file_size);
    buf[8]='W'; buf[9]='A'; buf[10]='V'; buf[11]='E';
    // fmt chunk
    buf[12]='f'; buf[13]='m'; buf[14]='t'; buf[15]=' ';
    write32(16, 16);  // chunk size
    write16(20, 1);   // PCM
    write16(22, 1);   // mono
    write32(24, static_cast<uint32_t>(sample_rate));
    write32(28, static_cast<uint32_t>(sample_rate * 2));  // byte rate
    write16(32, 2);   // block align
    write16(34, 16);  // bits per sample
    // data chunk
    buf[36]='d'; buf[37]='a'; buf[38]='t'; buf[39]='a';
    write32(40, data_size);
    std::memcpy(&buf[44], pcm.data(), pcm.size() * 2);
    return buf;
}

std::string WriteTempWav(const std::string & name, const std::vector<uint8_t> & wav) {
    const std::string path = qasr_test::TempPath(__FILE__, "qasr_test_" + name).string();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(wav.data()), static_cast<std::streamsize>(wav.size()));
    return path;
}

}  // namespace

// --- Normal: ParseWavBuffer ---

QASR_TEST(ParseWavBufferMono16k) {
    std::vector<int16_t> pcm = {0, 1000, -1000, 32767, -32768};
    auto wav = MakeMinimalWav(pcm);
    std::vector<float> samples;
    int32_t rate = 0;
    qasr::Status s = qasr::ParseWavBuffer(wav.data(), wav.size(), &samples, &rate);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(rate, int32_t(16000));
    QASR_EXPECT_EQ(samples.size(), pcm.size());
    // First sample should be zero
    QASR_EXPECT(std::abs(samples[0]) < 1e-6f);
}

QASR_TEST(ReadWavFromFile) {
    std::vector<int16_t> pcm = {100, 200, 300};
    auto wav = MakeMinimalWav(pcm);
    auto path = WriteTempWav("read_test.wav", wav);
    std::vector<float> samples;
    int32_t rate = 0;
    qasr::Status s = qasr::ReadWav(path, &samples, &rate);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(samples.size(), pcm.size());
    std::remove(path.c_str());
}

// --- Error: invalid inputs ---

QASR_TEST(ParseWavBufferNull) {
    std::vector<float> samples;
    int32_t rate = 0;
    qasr::Status s = qasr::ParseWavBuffer(nullptr, 100, &samples, &rate);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(ParseWavBufferTooSmall) {
    uint8_t tiny[10] = {};
    std::vector<float> samples;
    int32_t rate = 0;
    qasr::Status s = qasr::ParseWavBuffer(tiny, sizeof(tiny), &samples, &rate);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(ReadWavNonexistent) {
    std::vector<float> samples;
    int32_t rate = 0;
    qasr::Status s = qasr::ReadWav(
        qasr_test::MissingTempPath(__FILE__, "qasr_fake_wav_12345.wav").string(),
        &samples,
        &rate);
    QASR_EXPECT(!s.ok());
}

// --- Resample ---

QASR_TEST(ResampleSameRate) {
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> output;
    qasr::Status s = qasr::Resample(input, 16000, 16000, &output);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(output.size(), input.size());
}

QASR_TEST(ResampleDownsample) {
    std::vector<float> input(32000, 0.5f);  // 2 seconds at 16kHz
    std::vector<float> output;
    qasr::Status s = qasr::Resample(input, 16000, 8000, &output);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(output.size() < input.size());
}

QASR_TEST(ResampleEmptyInput) {
    std::vector<float> input;
    std::vector<float> output;
    qasr::Status s = qasr::Resample(input, 16000, 8000, &output);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(output.empty());
}

QASR_TEST(ResampleInvalidRate) {
    std::vector<float> input = {0.1f};
    std::vector<float> output;
    qasr::Status s = qasr::Resample(input, 0, 16000, &output);
    QASR_EXPECT(!s.ok());
}

// --- ComputeMelSpectrogram ---

QASR_TEST(MelSpectrogramBasic) {
    // 1 second of silence at 16kHz
    std::vector<float> audio(16000, 0.0f);
    std::vector<float> mel;
    int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(audio.data(), audio.size(), 80, &n_frames, &mel);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(n_frames > 0);
    QASR_EXPECT_EQ(mel.size(), static_cast<std::size_t>(n_frames) * 80);
}

QASR_TEST(MelSpectrogramNullInput) {
    std::vector<float> mel;
    int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(nullptr, 100, 80, &n_frames, &mel);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(MelSpectrogramZeroSamples) {
    float dummy = 0.0f;
    std::vector<float> mel;
    int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(&dummy, 0, 80, &n_frames, &mel);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(MelSpectrogramInvalidBins) {
    std::vector<float> audio(16000, 0.0f);
    std::vector<float> mel;
    int32_t n_frames = 0;
    qasr::Status s = qasr::ComputeMelSpectrogram(audio.data(), audio.size(), 0, &n_frames, &mel);
    QASR_EXPECT(!s.ok());
}

// --- CompactSilence ---

QASR_TEST(CompactSilenceEmpty) {
    std::vector<float> samples;
    qasr::Status s = qasr::CompactSilence(&samples, -40.0f, 200, 50);
    QASR_EXPECT(s.ok());
}

QASR_TEST(CompactSilenceKeepsSpeech) {
    // Non-silent samples should pass through
    std::vector<float> samples(1000, 0.5f);
    const auto original_size = samples.size();
    qasr::Status s = qasr::CompactSilence(&samples, -40.0f, 200, 50);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(samples.size(), original_size);
}

QASR_TEST(CompactSilenceNull) {
    qasr::Status s = qasr::CompactSilence(nullptr, -40.0f, 200, 50);
    QASR_EXPECT(!s.ok());
}

// --- StreamingAudioRing ---

QASR_TEST(AudioRingAppendAndCopy) {
    qasr::StreamingAudioRing ring(100);
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    ring.Append(data.data(), data.size());
    QASR_EXPECT_EQ(ring.current_size(), std::size_t(3));
    QASR_EXPECT_EQ(ring.total_appended(), std::size_t(3));

    std::vector<float> output;
    ring.CopyTo(&output);
    QASR_EXPECT_EQ(output.size(), std::size_t(3));
}

QASR_TEST(AudioRingEvictsOldSamples) {
    qasr::StreamingAudioRing ring(5);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    ring.Append(data.data(), data.size());
    std::vector<float> more = {5.0f, 6.0f, 7.0f};
    ring.Append(more.data(), more.size());
    QASR_EXPECT_EQ(ring.current_size(), std::size_t(5));
    QASR_EXPECT_EQ(ring.total_appended(), std::size_t(7));
}

QASR_TEST(AudioRingClear) {
    qasr::StreamingAudioRing ring(100);
    std::vector<float> data = {1.0f, 2.0f};
    ring.Append(data.data(), data.size());
    ring.Clear();
    QASR_EXPECT_EQ(ring.current_size(), std::size_t(0));
    QASR_EXPECT_EQ(ring.total_appended(), std::size_t(0));
}

QASR_TEST(AudioRingOverflowSingleAppend) {
    qasr::StreamingAudioRing ring(3);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ring.Append(data.data(), data.size());
    QASR_EXPECT_EQ(ring.current_size(), std::size_t(3));

    std::vector<float> output;
    ring.CopyTo(&output);
    // Should keep last 3 samples
    QASR_EXPECT_EQ(output.size(), std::size_t(3));
}
