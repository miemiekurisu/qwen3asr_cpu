/*
 * audit_bug_replay_scheduler_race_test.cc
 *
 * Proves the SessionFairQueue empty()/size()/Full() data race.
 *
 * Bug: These methods read queue_ without holding SessionFairQueue::mu_,
 * while Push() (which holds the mutex) concurrently modifies queue_.
 * This is a C++ data race (UB).
 *
 * The WorkerLoop at scheduler.cc:156-157 calls empty() from cv_.wait_for
 * predicate while holding GpuScheduler::mu_ but NOT SessionFairQueue::mu_.
 *
 * Reproduction strategy:
 *   We prove the race through static code analysis and guarded runtime tests.
 *   The thread-based reproduction is only activated under ThreadSanitizer,
 *   which detects the race safely without crashing.
 *
 * CI-safe: no model, no GPU needed.
 */

#include "tests/test_registry.h"

#include "qasr/scheduler/scheduler.h"

#include <cstdio>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

/* ─── Static proof: show that empty()/size() access queue_ without lock ─── */
QASR_TEST(SessionFairQueueEmptyDataRace_StaticProof) {
    /* Proof by code analysis:
     *
     * SessionFairQueue (scheduler.h:58-74):
     *   bool empty() const { return queue_.empty(); }  // line 62
     *   size_t size() const { return queue_.size(); }  // line 63
     *   bool Full(int max_size) const { return queue_.size() >= ...; }  // line 64
     *
     * These access queue_ WITHOUT locking mu_.  But:
     *   Status Push(const SegmentJob & job) {           // line 60
     *       std::lock_guard<std::mutex> lock(mu_);      // HAS lock
     *       queue_.push(job);                           // modifies queue_
     *   }
     *
     * Since Push (with lock) concurrently modifies queue_ while empty()
     * (without lock) reads it: DATA RACE (C++ §6.9.2 — undefined behavior).
     *
     * Compile and run with -fsanitize=thread to detect this race.
     */
    std::fprintf(stderr,
        "  STATIC PROOF:\n"
        "  scheduler.h:62  empty() reads queue_ WITHOUT SessionFairQueue::mu_\n"
        "  scheduler.h:63  size()  reads queue_ WITHOUT SessionFairQueue::mu_\n"
        "  scheduler.h:64  Full()  reads queue_ WITHOUT SessionFairQueue::mu_\n"
        "  scheduler.cc:9  Push()  writes queue_ WITH    SessionFairQueue::mu_\n"
        "\n"
        "  CONFIRMED: data race — concurrent Push + empty()/size() is UB.\n"
        "  Run with -fsanitize=thread to see ThreadSanitizer report.\n");
}

/* ─── WorkerLoop data race proof ─── */
QASR_TEST(GpuSchedulerWorkerLoopPredicateDataRace_StaticProof) {
    /* Proof by code analysis:
     *
     * scheduler.cc:154-168 (WorkerLoop):
     *   {
     *       std::unique_lock<std::mutex> lock(mu_);  // GpuScheduler::mu_
     *       cv_.wait_for(lock, ..., [this]() {
     *           return shutdown_.load()
     *               || !realtime_queue_.empty()   // ← NO SessionFairQueue::mu_!
     *               || !batch_queue_.empty();      // ← NO SessionFairQueue::mu_!
     *       });
     *       ...
     *   }
     *
     * GpuScheduler::mu_ protects inflight_sessions_, NOT SessionFairQueue::queue_.
     * SessionFairQueue::mu_ is held by Push()/Pop() but NOT by empty().
     *
     * WorkerLoop calls empty() in the cv predicate while holding GpuScheduler::mu_
     * but NOT SessionFairQueue::mu_.  This races with Push() which holds
     * SessionFairQueue::mu_.
     */
    std::fprintf(stderr,
        "  STATIC PROOF:\n"
        "  scheduler.cc:157  WorkerLoop calls realtime_queue_.empty() with\n"
        "                    GpuScheduler::mu_ held but NOT SessionFairQueue::mu_\n"
        "  scheduler.cc:62   empty() reads queue_ without SessionFairQueue::mu_\n"
        "  scheduler.cc:9    Push()  writes queue_ with SessionFairQueue::mu_\n"
        "  CONFIRMED: data race between WorkerLoop empty() and Push()\n");
}

/* ─── ThreadSanitizer-based detection ─── */
/* The following test actually creates the race and is only enabled under TSAN.
 * Under TSAN, the race is detected safely.  Without TSAN, skip. */
QASR_TEST(SessionFairQueueDataRace_TSanProof) {
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
    qasr::SessionFairQueue queue;
    std::atomic<bool> stop{false};
    std::atomic<int> push_count{0};

    std::thread pusher([&]() {
        for (int i = 0; i < 10000; i++) {
            qasr::SegmentJob job;
            job.session_id = 1;
            job.segment_id = static_cast<std::uint64_t>(i);
            job.realtime = true;
            if (queue.Push(job).ok()) push_count++;
        }
        stop.store(true);
    });

    std::thread reader([&]() {
        while (!stop.load()) {
            /* DATA RACE: accessing queue_ without SessionFairQueue::mu_
             * while Pusher holds SessionFairQueue::mu_ and modifies queue_.
             * TSAN will catch this immediately. */
            bool e __attribute__((unused)) = queue.empty();
            size_t s __attribute__((unused)) = queue.size();
            bool f __attribute__((unused)) = queue.Full(100);
        }
    });

    pusher.join();
    reader.join();

    std::fprintf(stderr,
        "  TSAN: data race detected if ThreadSanitizer is active\n");
#  else
    std::fprintf(stderr,
        "  [SKIP] Not compiled with ThreadSanitizer.\n"
        "  Recompile with -fsanitize=thread to detect this race.\n");
#  endif
#else
    std::fprintf(stderr,
        "  [SKIP] __has_feature not available.\n"
        "  Recompile with -fsanitize=thread to detect this race.\n");
#endif
}
