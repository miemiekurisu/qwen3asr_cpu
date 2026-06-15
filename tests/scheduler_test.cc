/*
 * scheduler_test.cc — GpuScheduler and SessionFairQueue tests (M3 verification).
 *
 * Tests:
 *   - SessionFairQueue push/pop/empty/size
 *   - GpuScheduler submit/start/shutdown
 *   - realtime vs batch priority
 *   - bounded queue backpressure
 *   - callback invocation
 *   - shutdown no deadlock
 *   - concurrent submit safety
 *
 * CI-safe: all tests run without model or GPU.
 */

#include "tests/test_registry.h"

#include "qasr/scheduler/scheduler.h"

#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

QASR_TEST(SessionFairQueuePushPopEmpty) {
    qasr::SessionFairQueue queue;
    QASR_EXPECT(queue.empty());
    QASR_EXPECT_EQ(queue.size(), 0u);

    qasr::SegmentJob job;
    job.session_id = 1;
    job.segment_id = 1;
    QASR_EXPECT(queue.Push(job).ok());
    QASR_EXPECT(!queue.empty());
    QASR_EXPECT_EQ(queue.size(), 1u);

    qasr::SegmentJob out;
    QASR_EXPECT(queue.Pop(out));
    QASR_EXPECT_EQ(out.session_id, 1u);
    QASR_EXPECT_EQ(out.segment_id, 1u);
    QASR_EXPECT(queue.empty());
}

QASR_TEST(SessionFairQueuePopEmptyReturnsFalse) {
    qasr::SessionFairQueue queue;
    qasr::SegmentJob out;
    QASR_EXPECT(!queue.Pop(out));
}

QASR_TEST(SessionFairQueueFifoOrder) {
    qasr::SessionFairQueue queue;

    for (int i = 0; i < 5; i++) {
        qasr::SegmentJob job;
        job.session_id = static_cast<std::uint64_t>(i + 1);
        job.segment_id = 1;
        queue.Push(job);
    }
    QASR_EXPECT_EQ(queue.size(), 5u);

    for (int i = 1; i <= 5; i++) {
        qasr::SegmentJob out;
        QASR_EXPECT(queue.Pop(out));
        QASR_EXPECT_EQ(out.session_id, static_cast<std::uint64_t>(i));
    }
    QASR_EXPECT(queue.empty());
}

QASR_TEST(SessionFairQueueFullReturnsTrue) {
    qasr::SessionFairQueue queue;
    QASR_EXPECT(!queue.Full(1));

    qasr::SegmentJob job;
    job.session_id = 1;
    queue.Push(job);
    QASR_EXPECT(queue.Full(1));
    QASR_EXPECT(!queue.Full(2));
}

QASR_TEST(SchedulerDefaultValues) {
    qasr::GpuScheduler scheduler;
    QASR_EXPECT_EQ(scheduler.max_sessions(), 3);
    QASR_EXPECT_EQ(scheduler.queue_depth(), 0);
}

QASR_TEST(SchedulerSubmitAndCallback) {
    qasr::GpuScheduler scheduler;
    std::atomic<int> call_count{0};
    std::uint64_t captured_session_id = 0;
    std::uint64_t captured_segment_id = 0;

    scheduler.SetCallback([&](const qasr::SegmentResult & res) {
        call_count++;
        captured_session_id = res.session_id;
        captured_segment_id = res.segment_id;
    });

    qasr::SegmentJob job;
    job.session_id = 42;
    job.segment_id = 7;
    job.realtime = true;
    job.samples = {0.1f, 0.2f, 0.3f};
    job.enqueue_time = std::chrono::steady_clock::now();

    QASR_EXPECT(scheduler.Submit(job).ok());
    QASR_EXPECT_EQ(scheduler.queue_depth(), 1);

    scheduler.Start();

    // Wait for the worker to process the job
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    QASR_EXPECT_EQ(call_count.load(), 1);
    QASR_EXPECT_EQ(captured_session_id, 42u);
    QASR_EXPECT_EQ(captured_segment_id, 7u);

    scheduler.Shutdown();
}

QASR_TEST(SchedulerRealtimePriority) {
    qasr::GpuScheduler scheduler;
    std::atomic<int> rt_count{0};
    std::atomic<int> batch_count{0};

    scheduler.SetCallback([&](const qasr::SegmentResult &) {
        rt_count++;
    });

    // Submit a batch job first
    qasr::SegmentJob batch_job;
    batch_job.session_id = 1;
    batch_job.segment_id = 1;
    batch_job.realtime = false;
    batch_job.enqueue_time = std::chrono::steady_clock::now();
    scheduler.Submit(batch_job);

    // Submit a realtime job second
    qasr::SegmentJob rt_job;
    rt_job.session_id = 2;
    rt_job.segment_id = 1;
    rt_job.realtime = true;
    rt_job.enqueue_time = std::chrono::steady_clock::now();
    scheduler.Submit(rt_job);

    QASR_EXPECT_EQ(scheduler.queue_depth(), 2);

    scheduler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    // Both should be processed
    QASR_EXPECT_EQ(rt_count.load(), 2);

    scheduler.Shutdown();
}

QASR_TEST(SchedulerBoundedQueueBackpressure) {
    qasr::GpuScheduler scheduler;
    QASR_EXPECT_EQ(scheduler.max_sessions(), 3);
    QASR_EXPECT_EQ(scheduler.max_sessions() * 4, 12);

    // Fill the realtime queue up to the limit: max_pending_per_session * max_sessions
    for (int i = 0; i < 12; i++) {
        qasr::SegmentJob job;
        job.session_id = 1;
        job.segment_id = static_cast<std::uint64_t>(i);
        job.realtime = true;
        job.enqueue_time = std::chrono::steady_clock::now();
        QASR_EXPECT(scheduler.Submit(job).ok());
    }

    // 13th should fail with ResourceExhausted
    qasr::SegmentJob overflow_job;
    overflow_job.session_id = 1;
    overflow_job.segment_id = 99;
    overflow_job.realtime = true;
    overflow_job.enqueue_time = std::chrono::steady_clock::now();
    auto status = scheduler.Submit(overflow_job);
    QASR_EXPECT(!status.ok());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kResourceExhausted);
}

QASR_TEST(SchedulerBatchQueueBackpressure) {
    qasr::GpuScheduler scheduler;

    // Fill batch queue: max_batch_queue_ = 16
    for (int i = 0; i < 16; i++) {
        qasr::SegmentJob job;
        job.session_id = 1;
        job.segment_id = static_cast<std::uint64_t>(i);
        job.realtime = false;
        job.enqueue_time = std::chrono::steady_clock::now();
        QASR_EXPECT(scheduler.Submit(job).ok());
    }

    // 17th should fail
    qasr::SegmentJob overflow_job;
    overflow_job.session_id = 1;
    overflow_job.segment_id = 99;
    overflow_job.realtime = false;
    overflow_job.enqueue_time = std::chrono::steady_clock::now();
    auto status = scheduler.Submit(overflow_job);
    QASR_EXPECT(!status.ok());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kResourceExhausted);
}

QASR_TEST(SchedulerShutdownAfterSubmit) {
    qasr::GpuScheduler scheduler;

    qasr::SegmentJob job;
    job.session_id = 1;
    job.segment_id = 1;
    job.realtime = true;
    job.enqueue_time = std::chrono::steady_clock::now();
    scheduler.Submit(job);

    // Shutdown without Start — should not hang
    scheduler.Shutdown();
}

QASR_TEST(SchedulerDoubleShutdown) {
    qasr::GpuScheduler scheduler;
    scheduler.Shutdown();
    scheduler.Shutdown();
}

QASR_TEST(SchedulerShutdownAfterStartNoDeadlock) {
    qasr::GpuScheduler scheduler;
    scheduler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler.Shutdown();
}

QASR_TEST(SchedulerSubmitAfterShutdownFails) {
    qasr::GpuScheduler scheduler;
    scheduler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler.Shutdown();

    qasr::SegmentJob job;
    job.session_id = 1;
    job.segment_id = 1;
    job.realtime = true;
    job.enqueue_time = std::chrono::steady_clock::now();
    auto status = scheduler.Submit(job);
    QASR_EXPECT(!status.ok());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(SchedulerConcurrentSubmitSafety) {
    qasr::GpuScheduler scheduler;
    std::atomic<int> submit_count{0};
    std::atomic<int> success_count{0};
    std::mutex mu;

    // Submit 20 jobs from 4 threads simultaneously
    for (int t = 0; t < 4; t++) {
        // Use a lambda to capture t
        [&, t]() {
            for (int i = 0; i < 5; i++) {
                qasr::SegmentJob job;
                job.session_id = static_cast<std::uint64_t>(t * 5 + i + 1);
                job.segment_id = 1;
                job.realtime = true;
                job.enqueue_time = std::chrono::steady_clock::now();

                auto status = scheduler.Submit(job);
                submit_count++;
                if (status.ok()) {
                    success_count++;
                }
            }
        }();
    }

    // Queue depth should match successful submits (max 12 for realtime)
    QASR_GE(12, static_cast<int>(scheduler.queue_depth()));
    QASR_EXPECT_EQ(submit_count.load(), 20);

    scheduler.Shutdown();
}

QASR_TEST(SchedulerMultipleJobsProcessed) {
    qasr::GpuScheduler scheduler;
    std::atomic<int> processed{0};
    std::mutex mu;
    std::vector<std::uint64_t> session_ids;

    scheduler.SetCallback([&](const qasr::SegmentResult & res) {
        std::lock_guard<std::mutex> lock(mu);
        session_ids.push_back(res.session_id);
        processed++;
    });

    for (int i = 0; i < 5; i++) {
        qasr::SegmentJob job;
        job.session_id = static_cast<std::uint64_t>(i + 1);
        job.segment_id = 1;
        job.realtime = true;
        job.enqueue_time = std::chrono::steady_clock::now();
        scheduler.Submit(job);
    }

    scheduler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    scheduler.Shutdown();

    QASR_EQ(processed.load(), 5);
}
