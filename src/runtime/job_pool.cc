#include "qasr/runtime/job_pool.h"

#include <atomic>
#include <chrono>

namespace qasr {

JobPool::JobPool(std::int32_t num_threads, std::int32_t queue_capacity)
    : num_threads_(num_threads), queue_(queue_capacity) {
  for (std::int32_t i = 0; i < num_threads_; ++i) {
    workers_.emplace_back(&JobPool::WorkerLoop, this);
  }
}

JobPool::~JobPool() {
  Shutdown();
}

Status JobPool::Submit(const std::string & request_id, std::function<void()> work) {
  if (shutdown_.load()) {
    return Status(StatusCode::kFailedPrecondition, "pool has been shut down");
  }
  TaskItem item;
  item.request_id = request_id;
  item.work = std::move(work);
  return queue_.Enqueue(std::move(item));
}

void JobPool::Shutdown() {
  if (shutdown_.exchange(true)) {
    return;  // Already shut down
  }
  // Don't call queue_.Shutdown() yet — let workers drain the queue
  for (auto & w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
  queue_.Shutdown();
}

std::int32_t JobPool::queue_size() const {
  return queue_.size();
}

void JobPool::WorkerLoop() {
  TaskItem item;
  while (true) {
    if (queue_.TryDequeue(&item)) {
      if (item.work) {
        active_tasks_.fetch_add(1);
        item.work();
        active_tasks_.fetch_sub(1);
      }
      continue;
    }
    // Queue empty — only exit if shutdown and no in-flight tasks
    if (shutdown_.load() && active_tasks_.load() == 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

}  // namespace qasr
