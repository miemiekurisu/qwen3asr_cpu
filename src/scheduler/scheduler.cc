#include "qasr/scheduler/scheduler.h"
#include <chrono>
#include <cstdio>

namespace qasr {

Status SessionFairQueue::Push(const SegmentJob & job) {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push(job);
    return OkStatus();
}

bool SessionFairQueue::Pop(SegmentJob & out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) return false;
    out = queue_.front();
    queue_.pop();
    return true;
}

GpuScheduler::GpuScheduler() = default;

GpuScheduler::~GpuScheduler() {
    Shutdown();
}

Status GpuScheduler::Submit(const SegmentJob & job) {
    if (shutdown_.load()) {
        return Status(StatusCode::kFailedPrecondition, "scheduler shut down");
    }
    if (job.realtime) {
        if (realtime_queue_.Full(max_pending_per_session_ * max_sessions_)) {
            return Status(StatusCode::kResourceExhausted, "realtime queue full");
        }
        realtime_queue_.Push(job);
    } else {
        if (batch_queue_.Full(max_batch_queue_)) {
            return Status(StatusCode::kResourceExhausted, "batch queue full");
        }
        batch_queue_.Push(job);
    }
    cv_.notify_one();
    return OkStatus();
}

void GpuScheduler::SetWorker(int worker_count) {
    (void)worker_count;
}

void GpuScheduler::SetCallback(JobCompleteCallback cb) {
    on_complete_ = std::move(cb);
}

void GpuScheduler::Start() {
    if (running_.exchange(true)) return;
    workers_.emplace_back([this]() { WorkerLoop(); });
}

void GpuScheduler::Shutdown() {
    if (!running_.exchange(false)) return;
    shutdown_.store(true);
    cv_.notify_all();
    for (auto & w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

void GpuScheduler::WorkerLoop() {
    while (!shutdown_.load()) {
        SegmentJob job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                return shutdown_.load() || !realtime_queue_.empty() || !batch_queue_.empty();
            });
            if (shutdown_.load()) break;

            SegmentResult res;
            if (realtime_queue_.Pop(job)) {
                res.session_id = job.session_id;
                res.segment_id = job.segment_id;
                res.status = Status(StatusCode::kUnimplemented,
                                    "CUDA decode worker not yet implemented");
                res.total_ms = 0;
            } else if (batch_queue_.Pop(job)) {
                res.session_id = job.session_id;
                res.segment_id = job.segment_id;
                res.status = Status(StatusCode::kUnimplemented,
                                    "CUDA decode worker not yet implemented");
                res.total_ms = 0;
            } else {
                continue;
            }

            if (on_complete_) {
                on_complete_(res);
            }
        }
    }
}

}  // namespace qasr
