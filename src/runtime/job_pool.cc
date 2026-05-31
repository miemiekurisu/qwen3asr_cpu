#include "qasr/runtime/job_pool.h"

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
  queue_.Shutdown();
  for (auto & w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
}

std::int32_t JobPool::queue_size() const {
  return queue_.size();
}

void JobPool::WorkerLoop() {
  TaskItem item;
  while (queue_.TryDequeue(&item)) {
    if (item.work) {
      item.work();
    }
  }
}

}  // namespace qasr
