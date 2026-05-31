#include "tests/test_registry.h"
#include "qasr/runtime/job_pool.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

// --- Normal ---

QASR_TEST(JobPoolConstruction) {
    qasr::JobPool pool(2, 10);
    QASR_EXPECT_EQ(pool.num_threads(), std::int32_t(2));
    QASR_EXPECT_EQ(pool.queue_capacity(), std::int32_t(10));
    QASR_EXPECT(!pool.is_shutdown());
}

QASR_TEST(JobPoolSubmitAndExecute) {
    qasr::JobPool pool(2, 10);
    std::atomic<int> counter{0};

    pool.Submit("job-1", [&counter]() { counter.fetch_add(1); });
    pool.Submit("job-2", [&counter]() { counter.fetch_add(1); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.Shutdown();

    QASR_EXPECT_EQ(counter.load(), 2);
}

QASR_TEST(JobPoolFIFOExecution) {
    qasr::JobPool pool(1, 10);
    std::vector<int> execution_order;
    std::mutex mu;

    pool.Submit("job-1", [&execution_order, &mu]() {
        std::lock_guard<std::mutex> lock(mu);
        execution_order.push_back(1);
    });
    pool.Submit("job-2", [&execution_order, &mu]() {
        std::lock_guard<std::mutex> lock(mu);
        execution_order.push_back(2);
    });
    pool.Submit("job-3", [&execution_order, &mu]() {
        std::lock_guard<std::mutex> lock(mu);
        execution_order.push_back(3);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.Shutdown();

    QASR_EXPECT_EQ(execution_order.size(), std::size_t(3));
    QASR_EXPECT_EQ(execution_order[0], 1);
    QASR_EXPECT_EQ(execution_order[1], 2);
    QASR_EXPECT_EQ(execution_order[2], 3);
}

QASR_TEST(JobPoolQueueSize) {
    qasr::JobPool pool(1, 10);

    std::atomic<bool> release{false};
    pool.Submit("blocking", [&release]() {
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    QASR_EXPECT_EQ(pool.queue_size(), std::int32_t(0));

    pool.Submit("queued-1", []() {});
    pool.Submit("queued-2", []() {});

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    QASR_EXPECT(pool.queue_size() >= 1);

    release.store(true);
    pool.Shutdown();
}

// --- Backpressure ---

QASR_TEST(JobPoolBackpressure) {
    qasr::JobPool pool(1, 2);

    std::atomic<bool> release{false};
    pool.Submit("blocking", [&release]() {
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    qasr::Status s1 = pool.Submit("fill-1", []() {});
    QASR_EXPECT(s1.ok());
    qasr::Status s2 = pool.Submit("fill-2", []() {});
    QASR_EXPECT(s2.ok());

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

    qasr::Status s = pool.Submit("post-shutdown", []() {});
    QASR_EXPECT(!s.ok());
}

QASR_TEST(JobPoolDrainOnShutdown) {
    qasr::JobPool pool(2, 64);
    std::atomic<int> counter{0};

    for (int i = 0; i < 20; ++i) {
        qasr::Status s = pool.Submit("drain-" + std::to_string(i), [&counter]() {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        QASR_EXPECT(s.ok());
    }

    pool.Shutdown();
    QASR_EXPECT_EQ(counter.load(), 20);
}

QASR_TEST(JobPoolDoubleShutdown) {
    qasr::JobPool pool(2, 10);
    pool.Shutdown();
    pool.Shutdown();
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
    }
    QASR_EXPECT_EQ(counter.load(), 1);
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
    QASR_EXPECT_EQ(counter.load(), 5);
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
    QASR_EXPECT_EQ(counter.load(), 64);
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

    QASR_EXPECT_EQ(counter.load(), successCount.load());
}
