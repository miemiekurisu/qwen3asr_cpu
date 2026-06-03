#include "qasr/runtime/task_queue.h"

#include <algorithm>

namespace qasr {

TaskQueue::TaskQueue(std::int32_t capacity) : capacity_(capacity) {}

TaskQueue::~TaskQueue() {
    Shutdown();
}

Status TaskQueue::Enqueue(TaskItem item) {
    if (shutdown_.load()) {
        return Status(StatusCode::kFailedPrecondition, "queue is shut down");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<std::int32_t>(queue_.size()) >= capacity_) {
        return Status(StatusCode::kFailedPrecondition,
                      "queue at capacity (backpressure)");
    }
    item.state = RequestState::kQueued;
    queue_.push(std::move(item));
    // Increment under the same lock that publishes the item, so any
    // observer that subsequently sees `pending_count() > 0` is also
    // guaranteed to see the item in the queue.
    ++pending_count_;
    not_empty_.notify_one();
    return OkStatus();
}

bool TaskQueue::TryDequeue(TaskItem * item) {
    if (!item) return false;
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) return false;
    *item = std::move(queue_.front());
    queue_.pop();
    // Item is now owned by the caller.  pending_count_ remains
    // non-zero until the caller invokes NotifyCompleted() after
    // running the work callback.
    return true;
}

bool TaskQueue::WaitForItem(TaskItem * item) {
    if (!item) return false;
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [this]() {
        return !queue_.empty() || shutdown_.load();
    });
    if (queue_.empty()) {
        // Spurious wakeup or shutdown — re-check inside the lock and
        // return false to let the worker exit cleanly.
        return false;
    }
    *item = std::move(queue_.front());
    queue_.pop();
    // pending_count_ remains non-zero until the caller invokes
    // NotifyCompleted() after running the work callback.
    return true;
}

void TaskQueue::NotifyCompleted() noexcept {
    {
        std::lock_guard<std::mutex> lock(mu_);
        --pending_count_;
    }
    not_empty_.notify_all();
}

Status TaskQueue::CancelTask(const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mu_);

    // Rebuild queue without the cancelled task
    std::queue<TaskItem> new_queue;
    bool found = false;
    while (!queue_.empty()) {
        auto item = std::move(queue_.front());
        queue_.pop();
        if (!found && item.request_id == request_id) {
            found = true;
            // Don't add to new queue — effectively cancelled
            --pending_count_;
            continue;
        }
        new_queue.push(std::move(item));
    }
    queue_ = std::move(new_queue);

    if (!found) {
        return Status(StatusCode::kNotFound, "task not found: " + request_id);
    }
    return OkStatus();
}

bool TaskQueue::IsOverloaded() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<std::int32_t>(queue_.size()) >= capacity_;
}

bool TaskQueue::IsEmpty() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.empty();
}

std::int32_t TaskQueue::pending_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_count_;
}

std::int32_t TaskQueue::size() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<std::int32_t>(queue_.size());
}

void TaskQueue::Shutdown() {
    shutdown_.store(true);
    std::lock_guard<std::mutex> lock(mu_);
    not_empty_.notify_all();
}

}  // namespace qasr
