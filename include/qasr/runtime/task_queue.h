#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>

#include "qasr/core/state_machine.h"
#include "qasr/core/status.h"

namespace qasr {

struct TaskItem {
    std::string request_id;
    RequestState state = RequestState::kAccepted;
    std::function<void()> work;
};

/// Bounded task queue with backpressure and cancellation support.
/// Pre: capacity > 0.
/// Post: Enqueue blocks or rejects when full.
/// Thread-safe: yes (internally synchronized).
class TaskQueue {
public:
    explicit TaskQueue(std::int32_t capacity = 64);
    ~TaskQueue();

    /// Enqueue a task. Returns kFailedPrecondition if queue is full (backpressure).
    Status Enqueue(TaskItem item);

    /// Try to dequeue a task. Returns false if queue is empty.
    bool TryDequeue(TaskItem * item);

    /// Cancel a pending task by request_id.
    /// Returns Ok if found and removed, kNotFound otherwise.
    Status CancelTask(const std::string & request_id);

    /// Reject overload: returns true if queue is at capacity.
    bool IsOverloaded() const noexcept;

    std::int32_t size() const noexcept;
    std::int32_t capacity() const noexcept { return capacity_; }
    bool is_shutdown() const noexcept { return shutdown_.load(); }

    void Shutdown();

private:
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::queue<TaskItem> queue_;
    std::int32_t capacity_;
    std::atomic<bool> shutdown_{false};
};

}  // namespace qasr
