/*
 * qasr_alloc_test.cc - Unit tests for the safe-realloc helper used by the
 * Qwen3-ASR C backend (see qwen_asr_alloc.h).
 *
 * The C++ side wraps the C API in std::byte / std::size_t calls.  These
 * tests exercise:
 *   1. Fresh growth from NULL (initial allocation)
 *   2. Doubling strategy for repeated grows
 *   3. No-op when the new size is <= the current capacity
 *   4. Arithmetic-overflow rejection (asked for > SIZE_MAX / element_size)
 *   5. Argument validation (NULL buffer pointer, zero element size, ...)
 *   6. The critical property: on realloc failure the previous buffer is
 *      preserved so the caller can continue without heap overflow.
 */
#include "tests/test_registry.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" {
#include "qwen_asr_alloc.h"
}

namespace {

// Returns true iff `p` points to a writable region of at least `n` bytes.
bool IsWritable(const void * p, std::size_t n) {
    if (p == nullptr) return false;
    volatile char * v = static_cast<volatile char *>(const_cast<void *>(p));
    // Touch both ends to make ASan/UBSan flag any out-of-bounds access.
    v[0] = v[0];
    if (n > 1) v[n - 1] = v[n - 1];
    return true;
}

}  // namespace

QASR_TEST(GrowBufferFreshFromNull) {
    void * buffer = nullptr;
    std::size_t new_cap = 0;
    int ok = qwen_grow_buffer(&buffer, sizeof(int), 0, 8, &new_cap);
    QASR_EXPECT_EQ(ok, 1);
    QASR_EXPECT(buffer != nullptr);
    QASR_EXPECT(new_cap >= 8);
    QASR_EXPECT(IsWritable(buffer, new_cap * sizeof(int)));
    std::free(buffer);
}

QASR_TEST(GrowBufferNoOpWhenAlreadyLarge) {
    // Pre-allocate a buffer of capacity 16, then "grow" to 16.  The
    // helper must return success without changing the capacity or the
    // pointer.  (Asking for strictly less than the current capacity is
    // invalid usage and is rejected by a separate test.)
    std::vector<int> backing(16);
    int * original = backing.data();
    void * buffer = original;
    std::size_t new_cap = 0;
    int ok = qwen_grow_buffer(&buffer, sizeof(int), 16, 16, &new_cap);
    QASR_EXPECT_EQ(ok, 1);
    QASR_EXPECT_EQ(buffer, static_cast<void *>(original));
}

QASR_TEST(GrowBufferDoublesUntilNeeded) {
    void * buffer = nullptr;
    std::size_t cap1 = 0;
    QASR_EXPECT_EQ(qwen_grow_buffer(&buffer, sizeof(int), 0, 4, &cap1), 1);
    QASR_EXPECT(cap1 >= 4);
    QASR_EXPECT(IsWritable(buffer, cap1 * sizeof(int)));

    std::size_t cap2 = 0;
    QASR_EXPECT_EQ(qwen_grow_buffer(&buffer, sizeof(int), cap1, cap1 * 2, &cap2), 1);
    QASR_EXPECT(cap2 >= cap1 * 2);
    QASR_EXPECT(IsWritable(buffer, cap2 * sizeof(int)));

    std::free(buffer);
}

QASR_TEST(GrowBufferPreservesPreviousBufferOnRejection) {
    // Mock: pre-allocate, then trigger overflow rejection by asking for
    // needed_capacity > SIZE_MAX / element_size.  The previous buffer
    // must be untouched and remain writable.
    void * buffer = nullptr;
    QASR_EXPECT_EQ(qwen_grow_buffer(&buffer, sizeof(int), 0, 4, nullptr), 1);
    QASR_EXPECT(buffer != nullptr);
    QASR_EXPECT(IsWritable(buffer, 4 * sizeof(int)));

    int * sentinel = static_cast<int *>(buffer);
    sentinel[0] = 0xAABBCCDD;
    sentinel[3] = 0x11223344;

    int ok = qwen_grow_buffer(
        &buffer,
        sizeof(int),
        4,
        std::numeric_limits<std::size_t>::max(),
        nullptr);
    QASR_EXPECT_EQ(ok, 0);
    QASR_EXPECT(buffer != nullptr);
    QASR_EXPECT_EQ(static_cast<int *>(buffer), sentinel);
    QASR_EXPECT_EQ(sentinel[0], 0xAABBCCDD);
    QASR_EXPECT_EQ(sentinel[3], 0x11223344);
    std::free(buffer);
}

QASR_TEST(GrowBufferRejectsNullBufferArg) {
    int ok = qwen_grow_buffer(nullptr, sizeof(int), 0, 4, nullptr);
    QASR_EXPECT_EQ(ok, 0);
}

QASR_TEST(GrowBufferRejectsZeroElementSize) {
    void * buffer = nullptr;
    int ok = qwen_grow_buffer(&buffer, 0, 0, 4, nullptr);
    QASR_EXPECT_EQ(ok, 0);
    QASR_EXPECT(buffer == nullptr);
}

QASR_TEST(GrowBufferRejectsShrinkRequest) {
    void * buffer = nullptr;
    QASR_EXPECT_EQ(qwen_grow_buffer(&buffer, sizeof(int), 0, 16, nullptr), 1);
    // Asking for less than the current capacity is invalid usage; the
    // helper must reject it without freeing or shrinking.
    int * original = static_cast<int *>(buffer);
    int ok = qwen_grow_buffer(&buffer, sizeof(int), 16, 8, nullptr);
    QASR_EXPECT_EQ(ok, 0);
    QASR_EXPECT_EQ(static_cast<int *>(buffer), original);
    std::free(buffer);
}

QASR_TEST(GrowBufferOutNewCapacityOptional) {
    void * buffer = nullptr;
    // Caller is allowed to pass NULL for out_new_capacity.
    int ok = qwen_grow_buffer(&buffer, sizeof(int), 0, 4, nullptr);
    QASR_EXPECT_EQ(ok, 1);
    QASR_EXPECT(buffer != nullptr);
    QASR_EXPECT(IsWritable(buffer, 4 * sizeof(int)));
    std::free(buffer);
}
