#pragma once

#include "qasr/engine/qwen_model.h"
#include "qasr/engine/types.h"
#include "qasr/engine/perf_stats.h"
#include "qasr/backend/device_backend.h"
#include "qasr/backend/cpu_backend.h"
#include "qasr/backend/cuda_backend.h"
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <cstdint>
#include <mutex>

namespace qasr {

struct VadState {
    bool speech_detected = false;
    float last_prob = 0.0f;
    int silent_frames = 0;
};

struct SegmentState {
    std::uint64_t current_segment_id = 0;
    int frame_offset = 0;
};

struct ReorderEntry {
    std::uint64_t segment_id;
    std::string text;
    double timestamp_ms;
    int version = 1;
};

struct ReorderBuffer {
    std::deque<ReorderEntry> entries;
};

class AudioRingBuffer {
public:
    void Push(const float * data, int count);
    int Drain(std::vector<float> & out, int max_samples);
    bool empty() const { return buffer_.empty(); }
    int size() const { return static_cast<int>(buffer_.size()); }

private:
    std::vector<float> buffer_;
    std::mutex mu_;
};

class QwenSession {
public:
    std::uint64_t session_id = 0;
    std::shared_ptr<QwenModel> model;
    std::unique_ptr<DeviceBackend> backend;

    VadState vad;
    AudioRingBuffer audio_buffer;
    SegmentState segment_state;

    std::unique_ptr<BackendSessionState> backend_state;

    std::string prompt;
    std::string language;
    int priority = 0;
    bool realtime = false;

    ReorderBuffer output_reorder;
    SessionPerfStats perf;

    std::mutex mu_;

    QwenSession() = default;
    ~QwenSession() = default;
    Status Initialize(const V2EngineConfig & config, std::shared_ptr<QwenModel> model);
};

}  // namespace qasr
