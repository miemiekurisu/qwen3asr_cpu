#include "tests/test_registry.h"
#include "qasr/runtime/task_queue.h"

#include <string>

// --- Normal ---

QASR_TEST(TaskQueueEnqueueDequeue) {
    qasr::TaskQueue queue(10);
    qasr::TaskItem item;
    item.request_id = "req-1";
    item.work = []() {};
    QASR_EXPECT(queue.Enqueue(std::move(item)).ok());
    QASR_EXPECT_EQ(queue.size(), int32_t(1));

    qasr::TaskItem out;
    QASR_EXPECT(queue.TryDequeue(&out));
    QASR_EXPECT_EQ(out.request_id, std::string("req-1"));
    QASR_EXPECT_EQ(queue.size(), int32_t(0));
}

QASR_TEST(TaskQueueDequeueEmpty) {
    qasr::TaskQueue queue(10);
    qasr::TaskItem item;
    QASR_EXPECT(!queue.TryDequeue(&item));
}

QASR_TEST(TaskQueueFIFOOrder) {
    qasr::TaskQueue queue(10);
    for (int i = 0; i < 3; ++i) {
        qasr::TaskItem item;
        item.request_id = "req-" + std::to_string(i);
        item.work = []() {};
        queue.Enqueue(std::move(item));
    }
    for (int i = 0; i < 3; ++i) {
        qasr::TaskItem item;
        queue.TryDequeue(&item);
        QASR_EXPECT_EQ(item.request_id, "req-" + std::to_string(i));
    }
}

QASR_TEST(TaskQueueCancelTask) {
    qasr::TaskQueue queue(10);
    for (int i = 0; i < 3; ++i) {
        qasr::TaskItem item;
        item.request_id = "req-" + std::to_string(i);
        item.work = []() {};
        queue.Enqueue(std::move(item));
    }
    QASR_EXPECT(queue.CancelTask("req-1").ok());
    QASR_EXPECT_EQ(queue.size(), int32_t(2));

    qasr::TaskItem first;
    queue.TryDequeue(&first);
    QASR_EXPECT_EQ(first.request_id, std::string("req-0"));

    qasr::TaskItem second;
    queue.TryDequeue(&second);
    QASR_EXPECT_EQ(second.request_id, std::string("req-2"));
}

QASR_TEST(TaskQueueCancelNonexistent) {
    qasr::TaskQueue queue(10);
    qasr::Status s = queue.CancelTask("does-not-exist");
    QASR_EXPECT(!s.ok());
}

// --- Backpressure ---

QASR_TEST(TaskQueueBackpressure) {
    qasr::TaskQueue queue(2);
    for (int i = 0; i < 2; ++i) {
        qasr::TaskItem item;
        item.request_id = "req-" + std::to_string(i);
        item.work = []() {};
        QASR_EXPECT(queue.Enqueue(std::move(item)).ok());
    }
    QASR_EXPECT(queue.IsOverloaded());

    qasr::TaskItem overflow;
    overflow.request_id = "req-overflow";
    overflow.work = []() {};
    qasr::Status s = queue.Enqueue(std::move(overflow));
    QASR_EXPECT(!s.ok());  // Rejected
}

QASR_TEST(TaskQueueNotOverloaded) {
    qasr::TaskQueue queue(10);
    QASR_EXPECT(!queue.IsOverloaded());
}

// --- Shutdown ---

QASR_TEST(TaskQueueShutdown) {
    qasr::TaskQueue queue(10);
    queue.Shutdown();
    QASR_EXPECT(queue.is_shutdown());

    qasr::TaskItem item;
    item.request_id = "post-shutdown";
    item.work = []() {};
    qasr::Status s = queue.Enqueue(std::move(item));
    QASR_EXPECT(!s.ok());
}

// --- Extreme ---

QASR_TEST(TaskQueueCapacityOne) {
    qasr::TaskQueue queue(1);
    qasr::TaskItem item;
    item.request_id = "only";
    item.work = []() {};
    QASR_EXPECT(queue.Enqueue(std::move(item)).ok());
    QASR_EXPECT(queue.IsOverloaded());

    qasr::TaskItem out;
    QASR_EXPECT(queue.TryDequeue(&out));
    QASR_EXPECT_EQ(out.request_id, std::string("only"));
    QASR_EXPECT(!queue.IsOverloaded());
}

QASR_TEST(TaskQueueStateTransition) {
    qasr::TaskQueue queue(10);
    qasr::TaskItem item;
    item.request_id = "req-1";
    item.state = qasr::RequestState::kAccepted;
    item.work = []() {};
    queue.Enqueue(std::move(item));

    qasr::TaskItem out;
    queue.TryDequeue(&out);
    // After enqueue, state should be kQueued
    QASR_EXPECT(out.state == qasr::RequestState::kQueued);
}
