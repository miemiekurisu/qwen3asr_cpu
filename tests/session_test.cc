/*
 * session_test.cc — QwenSession workspace and lifecycle tests.
 *
 * Tests:
 *   - QwenSession initialization
 *   - workspace allocation
 *   - AudioRingBuffer push/drain
 *   - VadState defaults
 *   - ReorderBuffer operations
 *
 * CI-safe: all tests run without model or GPU.
 */

#include "tests/test_registry.h"

#include "qasr/engine/qwen_session.h"
#include "qasr/engine/config.h"

#include <vector>

QASR_TEST(SessionDefaultConstruction) {
    qasr::QwenSession session;
    QASR_EXPECT_EQ(session.session_id, 0u);
    QASR_EXPECT(!session.model);
    QASR_EXPECT(!session.backend);
    QASR_EXPECT(!session.vad.speech_detected);
    QASR_EXPECT_EQ(session.segment_state.current_segment_id, 0u);
    QASR_EXPECT_EQ(session.priority, 0);
    QASR_EXPECT(!session.realtime);
}

QASR_TEST(SessionAllocateWorkspaceZeroBytes) {
    qasr::QwenSession session;
    auto status = session.AllocateWorkspace(0);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(!session.workspace());
}

QASR_TEST(SessionAllocateWorkspaceValid) {
    qasr::QwenSession session;
    auto status = session.AllocateWorkspace(1024);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(session.workspace() != nullptr);
}

QASR_TEST(SessionAllocateWorkspaceReallocates) {
    qasr::QwenSession session;
    auto status1 = session.AllocateWorkspace(256);
    QASR_EXPECT(status1.ok());
    void * ptr1 = session.workspace();

    auto status2 = session.AllocateWorkspace(512);
    QASR_EXPECT(status2.ok());
    QASR_EXPECT(session.workspace() != nullptr);
}

QASR_TEST(AudioRingBufferDefaultEmpty) {
    qasr::AudioRingBuffer buffer;
    QASR_EXPECT(buffer.empty());
    QASR_EXPECT_EQ(buffer.size(), 0);
}

QASR_TEST(AudioRingBufferPushDrain) {
    qasr::AudioRingBuffer buffer;

    std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f};
    buffer.Push(data.data(), 4);
    QASR_EXPECT_EQ(buffer.size(), 4);

    std::vector<float> out;
    int n = buffer.Drain(out, 2);
    QASR_EXPECT_EQ(n, 2);
    QASR_EXPECT_EQ(out.size(), 2u);
    QASR_EXPECT_EQ(out[0], 0.1f);
    QASR_EXPECT_EQ(out[1], 0.2f);
    QASR_EXPECT_EQ(buffer.size(), 2);

    int n2 = buffer.Drain(out, 10);
    QASR_EXPECT_EQ(n2, 2);
    QASR_EXPECT(buffer.empty());
}

QASR_TEST(AudioRingBufferDrainMoreThanAvailable) {
    qasr::AudioRingBuffer buffer;

    std::vector<float> data = {1.0f, 2.0f};
    buffer.Push(data.data(), 2);

    std::vector<float> out;
    int n = buffer.Drain(out, 100);
    QASR_EXPECT_EQ(n, 2);
    QASR_EXPECT_EQ(out.size(), 2u);
    QASR_EXPECT(buffer.empty());
}

QASR_TEST(AudioRingBufferZeroSize) {
    qasr::AudioRingBuffer buffer;
    std::vector<float> out;
    int n = buffer.Drain(out, 10);
    QASR_EXPECT_EQ(n, 0);
    QASR_EXPECT(buffer.empty());
}

QASR_TEST(VadStateDefaults) {
    qasr::VadState vad;
    QASR_EXPECT(!vad.speech_detected);
    QASR_EXPECT_EQ(vad.last_prob, 0.0f);
    QASR_EXPECT_EQ(vad.silent_frames, 0);
}

QASR_TEST(SegmentStateDefaults) {
    qasr::SegmentState seg;
    QASR_EXPECT_EQ(seg.current_segment_id, 0u);
    QASR_EXPECT_EQ(seg.frame_offset, 0);
}

QASR_TEST(ReorderBufferDefaultEmpty) {
    qasr::ReorderBuffer reorder;
    QASR_EXPECT(reorder.entries.empty());
}

QASR_TEST(SessionPerfStatsDefaults) {
    qasr::SessionPerfStats perf;
    QASR_EXPECT_EQ(perf.total_infer_ms, 0.0);
    QASR_EXPECT_EQ(perf.total_encode_ms, 0.0);
    QASR_EXPECT_EQ(perf.total_decode_ms, 0.0);
    QASR_EXPECT_EQ(perf.total_segments, 0);
    QASR_EXPECT_EQ(perf.total_tokens, 0);
}

QASR_TEST(SessionInitializeWithCpuBackend) {
    qasr::QwenSession session;
    qasr::V2EngineConfig config;
    config.backend = qasr::BackendKind::kCpu;
    auto status = session.Initialize(config, nullptr);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(session.backend != nullptr);
}
