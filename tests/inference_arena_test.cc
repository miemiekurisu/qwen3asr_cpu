#include "tests/test_registry.h"
#include "qasr/core/inference_arena.h"

#include <cstring>

QASR_TEST(InferenceArenaDefaultEmpty) {
    qasr::InferenceArena arena;
    QASR_EXPECT_EQ(arena.used(), std::size_t(0));
    QASR_EXPECT_EQ(arena.capacity(), std::size_t(0));
}

QASR_TEST(InferenceArenaInitialCapacity) {
    qasr::InferenceArena arena(4096);
    QASR_EXPECT_EQ(arena.used(), std::size_t(0));
    QASR_EXPECT(arena.capacity() >= std::size_t(4096));
}

QASR_TEST(InferenceArenaAllocateZeroReturnsNull) {
    qasr::InferenceArena arena(1024);
    float * ptr = arena.Allocate(0);
    QASR_EXPECT(ptr == nullptr);
    QASR_EXPECT_EQ(arena.used(), std::size_t(0));
}

QASR_TEST(InferenceArenaAllocateReturnsValid) {
    qasr::InferenceArena arena(1024);
    float * a = arena.Allocate(256);
    QASR_EXPECT(a != nullptr);
    QASR_EXPECT_EQ(arena.used(), std::size_t(256));

    float * b = arena.Allocate(128);
    QASR_EXPECT(b != nullptr);
    QASR_EXPECT(b != a);
    QASR_EXPECT_EQ(arena.used(), std::size_t(384));
}

QASR_TEST(InferenceArenaAllocateIsContiguous) {
    qasr::InferenceArena arena(1024);
    float * a = arena.Allocate(100);
    float * b = arena.Allocate(100);
    // b should start right after a
    QASR_EXPECT_EQ(b, a + 100);
}

QASR_TEST(InferenceArenaResetReusesMemory) {
    qasr::InferenceArena arena(1024);
    float * first = arena.Allocate(512);
    QASR_EXPECT_EQ(arena.used(), std::size_t(512));

    arena.Reset();
    QASR_EXPECT_EQ(arena.used(), std::size_t(0));
    QASR_EXPECT(arena.capacity() >= std::size_t(1024));

    float * second = arena.Allocate(512);
    QASR_EXPECT_EQ(first, second);  // Reuses same memory
}

QASR_TEST(InferenceArenaGrowsBeyondInitial) {
    qasr::InferenceArena arena(64);
    float * ptr = arena.Allocate(256);
    QASR_EXPECT(ptr != nullptr);
    QASR_EXPECT(arena.capacity() >= std::size_t(256));
    QASR_EXPECT_EQ(arena.used(), std::size_t(256));
}

QASR_TEST(InferenceArenaReserveGrows) {
    qasr::InferenceArena arena;
    arena.Reserve(8192);
    QASR_EXPECT(arena.capacity() >= std::size_t(8192));
    QASR_EXPECT_EQ(arena.used(), std::size_t(0));
}

QASR_TEST(InferenceArenaWriteAndRead) {
    qasr::InferenceArena arena(1024);
    float * buf = arena.Allocate(4);
    buf[0] = 1.0f;
    buf[1] = 2.0f;
    buf[2] = 3.0f;
    buf[3] = 4.0f;

    arena.Reset();
    float * buf2 = arena.Allocate(4);
    // Same memory, values should persist (no zeroing on reset)
    QASR_EXPECT_EQ(buf2[0], 1.0f);
    QASR_EXPECT_EQ(buf2[3], 4.0f);
}
