#include "tests/test_registry.h"
#include "qasr/inference/encoder.h"

// --- BuildEncoderWindowPlan ---

QASR_TEST(EncoderPlanSingleWindow) {
    qasr::EncoderWindowPlan plan;
    qasr::Status s = qasr::BuildEncoderWindowPlan(100, 100, 100, &plan);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(plan.n_windows, int32_t(1));
    QASR_EXPECT_EQ(plan.total_frames, int32_t(100));
    QASR_EXPECT_EQ(plan.window_starts.size(), std::size_t(1));
    QASR_EXPECT_EQ(plan.window_starts[0], int32_t(0));
}

QASR_TEST(EncoderPlanMultipleWindows) {
    qasr::EncoderWindowPlan plan;
    qasr::Status s = qasr::BuildEncoderWindowPlan(250, 100, 150, &plan);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(plan.n_windows >= 3);
}

QASR_TEST(EncoderPlanNullOutput) {
    qasr::Status s = qasr::BuildEncoderWindowPlan(100, 50, 50, nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncoderPlanZeroFrames) {
    qasr::EncoderWindowPlan plan;
    qasr::Status s = qasr::BuildEncoderWindowPlan(0, 50, 50, &plan);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncoderPlanZeroChunk) {
    qasr::EncoderWindowPlan plan;
    qasr::Status s = qasr::BuildEncoderWindowPlan(100, 0, 50, &plan);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncoderPlanWindowSmallerThanChunk) {
    qasr::EncoderWindowPlan plan;
    qasr::Status s = qasr::BuildEncoderWindowPlan(100, 50, 30, &plan);
    QASR_EXPECT(!s.ok());
}

// --- EncodeChunk ---

QASR_TEST(EncodeChunkUnloadedWeights) {
    qasr::EncoderWeights w;
    w.loaded = false;
    qasr::EncoderWindowPlan plan;
    plan.n_windows = 1;
    plan.window_starts = {0};
    plan.window_frames = 100;
    std::vector<float> mel(100, 0.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeChunk(w, mel.data(), 100, 0, plan, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncodeChunkNullMel) {
    qasr::EncoderWeights w;
    w.loaded = true;
    w.d_model = 256;
    qasr::EncoderWindowPlan plan;
    plan.n_windows = 1;
    plan.window_starts = {0};
    plan.window_frames = 100;
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeChunk(w, nullptr, 100, 0, plan, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncodeChunkValidProducesOutput) {
    qasr::EncoderWeights w;
    w.loaded = true;
    w.d_model = 256;
    qasr::EncoderWindowPlan plan;
    plan.n_windows = 1;
    plan.window_starts = {0};
    plan.window_frames = 100;
    std::vector<float> mel(100 * 80, 0.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeChunk(w, mel.data(), 100, 0, plan, &output, &seq_len);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(seq_len > 0);
    QASR_EXPECT(!output.empty());
}

QASR_TEST(EncodeChunkOutOfRange) {
    qasr::EncoderWeights w;
    w.loaded = true;
    w.d_model = 256;
    qasr::EncoderWindowPlan plan;
    plan.n_windows = 1;
    plan.window_starts = {0};
    plan.window_frames = 100;
    std::vector<float> mel(100, 0.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeChunk(w, mel.data(), 100, 5, plan, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

// --- EncodeAudio ---

QASR_TEST(EncodeAudioUnloadedWeights) {
    qasr::EncoderWeights w;
    w.loaded = false;
    std::vector<float> mel(100, 0.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeAudio(w, mel.data(), 100, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EncodeAudioZeroFrames) {
    qasr::EncoderWeights w;
    w.loaded = true;
    w.d_model = 256;
    std::vector<float> mel(100, 0.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::EncodeAudio(w, mel.data(), 0, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

// --- ConcatEncoderWindows ---

QASR_TEST(ConcatEncoderWindowsEmpty) {
    std::vector<std::vector<float>> windows;
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::ConcatEncoderWindows(windows, 256, &output, &seq_len);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(seq_len, int32_t(0));
    QASR_EXPECT(output.empty());
}

QASR_TEST(ConcatEncoderWindowsTwoWindows) {
    std::vector<float> w1(256 * 2, 1.0f);
    std::vector<float> w2(256 * 3, 2.0f);
    std::vector<std::vector<float>> windows = {w1, w2};
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::ConcatEncoderWindows(windows, 256, &output, &seq_len);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(seq_len, int32_t(5));
    QASR_EXPECT_EQ(output.size(), std::size_t(256 * 5));
}

QASR_TEST(ConcatEncoderWindowsMismatchedDModel) {
    std::vector<float> w1(255, 1.0f);  // Not divisible by 256
    std::vector<std::vector<float>> windows = {w1};
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::ConcatEncoderWindows(windows, 256, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(ConcatEncoderWindowsInvalidDModel) {
    std::vector<std::vector<float>> windows;
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::ConcatEncoderWindows(windows, 0, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}
