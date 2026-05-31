#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/runtime/task_queue.h"

namespace qasr {

/// Bounded thread pool backed by a TaskQueue.
/// Pre: num_threads > 0, capacity > 0.
/// Post: processes jobs concurrently up to num_threads workers.
/// Thread-safe: yes (internally synchronized).
class JobPool {
 public:
  JobPool(std::int32_t num_threads, std::int32_t queue_capacity = 64);
  ~JobPool();

  /// Submit a job for asynchronous execution.
  /// Returns kResourceExhausted when the internal queue is full.
  Status Submit(const std::string & request_id, std::function<void()> work);

  /// Graceful shutdown: stops accepting work, drains queue, joins threads.
  void Shutdown();

  /// Returns true after Shutdown() has completed.
  bool is_shutdown() const noexcept { return shutdown_.load(); }

  std::int32_t num_threads() const noexcept { return num_threads_; }
  std::int32_t queue_size() const;
  std::int32_t queue_capacity() const noexcept { return queue_.capacity(); }

 private:
  void WorkerLoop();

  const std::int32_t num_threads_;
  TaskQueue queue_;
  std::vector<std::thread> workers_;
  std::atomic<bool> shutdown_{false};
};

}  // namespace qasr
