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
    /// The caller must invoke NotifyCompleted() exactly once for every
    /// successful dequeue, after the work callback has returned.
    bool TryDequeue(TaskItem * item);

    /// Block until a task is available, then dequeue it. Returns false
    /// if the queue is shut down while waiting.  Use this in worker
    /// loops in preference to TryDequeue, otherwise a worker can exit
    /// prematurely when the queue is momentarily empty (e.g. on a
    /// freshly-constructed pool).  The caller must invoke
    /// NotifyCompleted() exactly once for every successful wait, after
    /// the work callback has returned.
    bool WaitForItem(TaskItem * item);

    /// Decrement the pending count for a previously-dequeued task.
    /// Must be called exactly once per successful TryDequeue, after
    /// the work callback returns (or after the dequeued item is
    /// discarded without being run).
    void NotifyCompleted() noexcept;

    /// Cancel a pending task by request_id.
    /// Returns Ok if found and removed, kNotFound otherwise.
    Status CancelTask(const std::string & request_id);

    /// Reject overload: returns true if queue is at capacity.
    bool IsOverloaded() const noexcept;

    /// Returns true if no tasks are currently queued.
    bool IsEmpty() const noexcept;

    /// Returns the number of tasks that have been accepted via
    /// Enqueue() but whose work callback has not yet been invoked.
    ///
    /// "Pending" is incremented under the same lock that publishes the
    /// item to the queue, and decremented only after the callback
    /// returns.  This gives callers a load-bearing view of "submitted
    /// but not yet executed" that is consistent with the queue
    /// contents: an observer that sees `pending_count() == 0` is
    /// guaranteed to see the queue empty AND no worker currently
    /// running a callback.
    std::int32_t pending_count() const noexcept;

    std::int32_t size() const noexcept;
    std::int32_t capacity() const noexcept { return capacity_; }
    bool is_shutdown() const noexcept { return shutdown_.load(); }

    void Shutdown();

private:
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::queue<TaskItem> queue_;
    std::int32_t capacity_;
    std::int32_t pending_count_ = 0;  // guarded by mu_
    std::atomic<bool> shutdown_{false};
};

}  // namespace qasr
