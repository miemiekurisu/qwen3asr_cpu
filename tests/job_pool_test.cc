#include "tests/test_registry.h"
#include "qasr/runtime/job_pool.h"

#include <atomic>
#include <chrono>
#include <thread>

// --- Normal ---

QASR_TEST(JobPoolConstruction) {
    qasr::JobPool pool(2, 10);
    QASR_EQ(pool.num_threads(), 2);
    QASR_EQ(pool.queue_capacity(), 10);
    QASR_EXPECT(!pool.is_shutdown());
}

QASR_TEST(JobPoolSubmitAndExecute) {
    qasr::JobPool pool(2, 10);
    std::atomic<int> counter{0};

    pool.Submit("job-1", [&counter]() { counter.fetch_add(1); });
    pool.Submit("job-2", [&counter]() { counter.fetch_add(1); });

    // Wait for jobs to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.Shutdown();

    QASR_EQ(counter.load(), 2);
}

QASR_TEST(JobPoolFIFOExecution) {
    qasr::JobPool pool(1, 10);
    std::atomic<int> order{0};
    std::atomic<int> lastOrder{0};

    // Submit 3 jobs with ordering assertions
    pool.Submit("job-1", [&]() {
        int v = order.fetch_add(1) + 1;
        QASR_EQ(v, lastOrder.fetch_add(1) + 1);
    });
    pool.Submit("job-2", [&]() {
        int v = order.fetch_add(1) + 1;
        QASR_EQ(v, lastOrder.fetch_add(1) + 1);
    });
    pool.Submit("job-3", [&]() {
        int v = order.fetch_add(1) + 1;
        QASR_EQ(v, lastOrder.fetch_add(1) + 1);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();
}

QASR_TEST(JobPoolQueueSize) {
    qasr::JobPool pool(1, 10);

    // Submit a blocking job to keep the worker busy
    std::atomic<bool> release{false};
    pool.Submit("blocking", [&release]() {
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    QASR_EQ(pool.queue_size(), 0);  // Worker picked it up

    // Submit more jobs that will queue up
    pool.Submit("queued-1", []() {});
    pool.Submit("queued-2", []() {});

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    QASR_GE(pool.queue_size(), 1);

    release.store(true);
    pool.Shutdown();
}

// --- Backpressure ---

QASR_TEST(JobPoolBackpressure) {
    qasr::JobPool pool(1, 2);

    // Block the worker
    std::atomic<bool> release{false};
    pool.Submit("blocking", [&release]() {
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Fill the queue
    qasr::Status s1 = pool.Submit("fill-1", []() {});
    QASR_EXPECT(s1.ok());
    qasr::Status s2 = pool.Submit("fill-2", []() {});
    QASR_EXPECT(s2.ok());

    // Next should fail with backpressure
    qasr::Status s3 = pool.Submit("overflow", []() {});
    QASR_EXPECT(!s3.ok());

    release.store(true);
    pool.Shutdown();
}

// --- Shutdown ---

QASR_TEST(JobPoolShutdown) {
    qasr::JobPool pool(2, 10);
    pool.Shutdown();
    QASR_EXPECT(pool.is_shutdown());

    // Submit after shutdown should fail
    qasr::Status s = pool.Submit("post-shutdown", []() {});
    QASR_EXPECT(!s.ok());
}

QASR_TEST(JobPoolDrainOnShutdown) {
    qasr::JobPool pool(2, 10);
    std::atomic<int> counter{0};

    // Submit many jobs
    for (int i = 0; i < 20; ++i) {
        pool.Submit("drain-" + std::to_string(i), [&counter]() {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }

    // Shutdown should drain queue
    pool.Shutdown();
    QASR_EQ(counter.load(), 20);
}

QASR_TEST(JobPoolDoubleShutdown) {
    qasr::JobPool pool(2, 10);
    pool.Shutdown();
    pool.Shutdown();  // Should be safe
    QASR_EXPECT(pool.is_shutdown());
}

QASR_TEST(JobPoolDestructorShutdown) {
    std::atomic<int> counter{0};
    {
        qasr::JobPool pool(2, 10);
        pool.Submit("destructor-test", [&counter]() {
            counter.fetch_add(1);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Destructor should shutdown gracefully
    }
    QASR_EQ(counter.load(), 1);
}

// --- Extreme ---

QASR_TEST(JobPoolSingleThread) {
    qasr::JobPool pool(1, 5);
    std::atomic<int> counter{0};

    for (int i = 0; i < 5; ++i) {
        pool.Submit("single-" + std::to_string(i), [&counter]() {
            counter.fetch_add(1);
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.Shutdown();
    QASR_EQ(counter.load(), 5);
}

QASR_TEST(JobPoolManyThreads) {
    qasr::JobPool pool(8, 100);
    std::atomic<int> counter{0};

    for (int i = 0; i < 64; ++i) {
        pool.Submit("many-" + std::to_string(i), [&counter]() {
            counter.fetch_add(1);
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();
    QASR_EQ(counter.load(), 64);
}

QASR_TEST(JobPoolEmptyWork) {
    qasr::JobPool pool(2, 10);
    pool.Submit("empty", nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pool.Shutdown();
}

QASR_TEST(JobPoolConcurrentSubmit) {
    qasr::JobPool pool(4, 50);
    std::atomic<int> counter{0};
    std::atomic<int> successCount{0};

    // Submit from multiple threads concurrently
    std::vector<std::thread> submitters;
    for (int t = 0; t < 4; ++t) {
        submitters.emplace_back([&pool, &counter, &successCount, t]() {
            for (int i = 0; i < 10; ++i) {
                qasr::Status s = pool.Submit(
                    "concurrent-" + std::to_string(t) + "-" + std::to_string(i),
                    [&counter]() { counter.fetch_add(1); });
                if (s.ok()) {
                    successCount.fetch_add(1);
                }
            }
        });
    }

    for (auto & t : submitters) {
        t.join();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();

    QASR_EQ(counter.load(), successCount.load());
}
