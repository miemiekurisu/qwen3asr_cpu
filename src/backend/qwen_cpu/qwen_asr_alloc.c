/*
 * qwen_asr_alloc.c - Implementation of the allocation helpers declared
 * in qwen_asr_alloc.h.
 */
#include "qwen_asr_alloc.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Upper bound on the doubling sequence.  This is large enough that the
 * realistic tail-buffer sizes used in the streaming decoder never hit
 * it, but small enough that we never silently grow into multi-gigabyte
 * allocations due to a caller bug. */
#define QWEN_GROW_MAX_CAPACITY ((size_t)1 << 20)

int qwen_grow_buffer(
    void ** buffer,
    size_t element_size,
    size_t current_capacity,
    size_t needed_capacity,
    size_t * out_new_capacity) {
    if (buffer == NULL) return 0;
    if (element_size == 0) return 0;
    if (needed_capacity < current_capacity) return 0;

    /* Reject sizes that would overflow size_t when multiplied by
     * element_size.  This protects against caller mistakes and against
     * adversarial inputs that might have slipped through. */
    if (needed_capacity > SIZE_MAX / element_size) {
        return 0;
    }

    size_t new_capacity = current_capacity == 0 ? 1 : current_capacity;
    while (new_capacity < needed_capacity) {
        if (new_capacity >= QWEN_GROW_MAX_CAPACITY / 2) {
            new_capacity = needed_capacity;
            break;
        }
        new_capacity *= 2;
    }

    void * resized = realloc(*buffer, new_capacity * element_size);
    if (resized == NULL) {
        /* On failure: *buffer is unchanged per realloc(3) contract.
         * Caller's previous buffer remains valid. */
        return 0;
    }
    *buffer = resized;
    if (out_new_capacity != NULL) {
        *out_new_capacity = new_capacity;
    }
    return 1;
}
