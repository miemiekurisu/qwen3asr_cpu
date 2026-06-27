#pragma once

#include "qasr/core/status.h"
#include "qasr/engine/asr_engine.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <condition_variable>
#include <atomic>

namespace qasr {

struct SegmentResult;

using JobCompleteCallback = std::function<void(const SegmentResult &)>;

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

    /* Optional pre-existing engine session ID.  If non-zero,
     * ExecuteJob reuses this session instead of creating a
     * transient one.  The caller is responsible for cleanup. */
    std::uint64_t engine_session_id = 0;

    /* Per-job callback — invoked by ExecuteJob on completion.
     * Takes ownership of the callback to avoid global callback
     * race in SubmitAndAwait. */
    JobCompleteCallback on_job_complete;
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

class SessionFairQueue {
public:
    Status Push(const SegmentJob & job);
    bool Pop(SegmentJob & out);
    bool empty() const { return queue_.empty(); }
    size_t size() const { return queue_.size(); }
    bool Full(int max_size) const { return queue_.size() >= static_cast<size_t>(max_size); }

    /* Check if session has a pending job (backpressure). */
    bool SessionHasPending(std::uint64_t session_id) const;
    /* Return pending count for a given session. Thread-safe. */
    int SessionPendingCount(std::uint64_t session_id) const;

private:
    std::queue<SegmentJob> queue_;
    /* Per-session job count — tracks how many jobs each session has in queue. */
    std::unordered_map<std::uint64_t, int> session_count_;
    mutable std::mutex mu_;
};

class GpuScheduler {
public:
    GpuScheduler();
    ~GpuScheduler();

    /* Set the engine that performs inference. Must be called before Start. */
    void SetEngine(AsrEngine * engine);

    Status Submit(const SegmentJob & job);

    /* Synchronous submit-and-await: submit a job and block until
     * the result is ready.  Useful for batch paths that need the
     * result inline.  Returns the SegmentResult directly. */
    SegmentResult SubmitAndAwait(const SegmentJob & job);

    void SetWorker(int worker_count = 1);
    void SetCallback(JobCompleteCallback cb);
    void Start();
    void Shutdown();

    int max_sessions() const { return max_sessions_; }
    int queue_depth() const { return static_cast<int>(realtime_queue_.size() + batch_queue_.size()); }

private:
    void WorkerLoop();
    void ExecuteJob(SegmentJob & job);

    int max_sessions_ = 3;
    int max_active_gpu_jobs_ = 1;
    int max_pending_per_session_ = 4;
    int max_batch_queue_ = 16;

    SessionFairQueue realtime_queue_;
    SessionFairQueue batch_queue_;

    AsrEngine * engine_ = nullptr;
    JobCompleteCallback on_complete_;

    /* Per-session inflight tracking — max 1 inflight per session. */
    std::unordered_set<std::uint64_t> inflight_sessions_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;
};

}  // namespace qasr
