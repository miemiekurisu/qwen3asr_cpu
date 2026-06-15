#pragma once

#include "qasr/core/status.h"
#include "qasr/engine/types.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <atomic>

namespace qasr {

struct SegmentJob {
    std::uint64_t session_id;
    std::uint64_t segment_id;
    std::vector<float> samples;
    int64_t sample_rate = 16000;
    bool realtime = true;
    int priority = 0;
    std::chrono::steady_clock::time_point enqueue_time;
    std::chrono::steady_clock::time_point deadline;
    std::string language;
    std::string prompt;
};

struct SegmentResult {
    std::uint64_t session_id;
    std::uint64_t segment_id;
    Status status;
    std::string text;
    double encode_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    int tokens = 0;
};

using JobCompleteCallback = std::function<void(const SegmentResult &)>;

class SessionFairQueue {
public:
    Status Push(const SegmentJob & job);
    bool Pop(SegmentJob & out);
    bool empty() const { return queue_.empty(); }
    size_t size() const { return queue_.size(); }
    bool Full(int max_size) const { return queue_.size() >= static_cast<size_t>(max_size); }

private:
    std::queue<SegmentJob> queue_;
    mutable std::mutex mu_;
};

class GpuScheduler {
public:
    GpuScheduler();
    ~GpuScheduler();

    Status Submit(const SegmentJob & job);
    void SetWorker(int worker_count = 1);
    void SetCallback(JobCompleteCallback cb);
    void Start();
    void Shutdown();

    int max_sessions() const { return max_sessions_; }
    int queue_depth() const { return static_cast<int>(realtime_queue_.size() + batch_queue_.size()); }

private:
    void WorkerLoop();

    int max_sessions_ = 3;
    int max_active_gpu_jobs_ = 1;
    int max_pending_per_session_ = 4;
    int max_batch_queue_ = 16;

    SessionFairQueue realtime_queue_;
    SessionFairQueue batch_queue_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;

    JobCompleteCallback on_complete_;
};

}  // namespace qasr
