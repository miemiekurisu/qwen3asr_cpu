/*
 * qwen_asr_alloc.h - Allocation helpers for the Qwen3-ASR C backend.
 *
 * Helpers in this header are deliberately small and side-effect-free
 * wrappers around the platform allocator.  They exist so the
 * realloc-grow-or-keep pattern (used in the streaming decoder tail
 * buffer) can be exercised directly from unit tests.
 */
#ifndef QWEN_ASR_ALLOC_H
#define QWEN_ASR_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Grow the buffer pointed to by *buffer so that it can hold at least
 * `needed_capacity` elements of size `element_size`.
 *
 * On success, *buffer points to the new (possibly moved) allocation,
 * the returned capacity is >= needed_capacity, and the new bytes
 * (from the old capacity up to the new capacity) are uninitialised.
 *
 * On failure (allocation error or arithmetic overflow), *buffer is
 * UNCHANGED and the caller's previous buffer remains valid.  This is
 * the property that makes the call site safe against heap-buffer
 * overflow when the new size is larger than the old size.
 *
 * The growth strategy is doubling, capped at 1<<20 elements, to keep
 * amortised cost O(n) and avoid pathological realloc patterns.
 *
 * Pre:
 *   - *buffer may be NULL (initial capacity is treated as 0).
 *   - current_capacity must equal the current allocation capacity in
 *     elements, not the number of valid elements.
 *   - needed_capacity must be >= current_capacity.
 *   - element_size must be > 0.
 *
 * Post:
 *   - Returns 1 on success, 0 on failure.
 *   - On success, *out_new_capacity is set to the new capacity in
 *     elements.  May be NULL if the caller does not need it.
 */
int qwen_grow_buffer(
    void ** buffer,
    size_t element_size,
    size_t current_capacity,
    size_t needed_capacity,
    size_t * out_new_capacity);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* QWEN_ASR_ALLOC_H */
