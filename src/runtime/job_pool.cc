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
  // Wake any pending WaitForIdle callers and let them observe idle.
  {
    std::lock_guard<std::mutex> lock(idle_mu_);
    idle_cv_.notify_all();
  }
}

std::int32_t JobPool::queue_size() const {
  return queue_.size();
}

bool JobPool::WaitForIdle(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(idle_mu_);
  return idle_cv_.wait_for(lock, timeout, [this]() {
    return shutdown_.load() || queue_.pending_count() == 0;
  });
}

void JobPool::WorkerLoop() {
  // We must block on the queue's "not empty" signal, not poll
  // TryDequeue, otherwise all workers may exit if the queue is empty
  // when the thread starts up (e.g. on a freshly-constructed pool).
  TaskItem item;
  while (queue_.WaitForItem(&item)) {
    if (item.work) {
      item.work();
    }
    queue_.NotifyCompleted();
    // Wake WaitForIdle: the predicate re-checks pending_count() and
    // can now observe zero.
    {
      std::lock_guard<std::mutex> lock(idle_mu_);
      idle_cv_.notify_all();
    }
  }
  // Reached only on shutdown.  Shutdown's notify_all on the queue's
  // not_empty_ condition variable will wake us from WaitForItem.
}

}  // namespace qasr
