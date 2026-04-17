#pragma once

#include <cstddef>
#include <vector>

namespace qasr {

/// Pre-allocated scratch memory arena for inference operations.
///
/// Eliminates per-call malloc/free overhead by maintaining a persistent
/// buffer that grows once and is reused across invocations via Reset().
/// Typical usage: reserve once at session start, then Allocate/Reset per
/// inference step.
///
/// Thread-safety: NOT thread-safe; owned per session.
class InferenceArena {
public:
    explicit InferenceArena(std::size_t initial_capacity = 0);

    /// Ensure at least @p min_capacity floats of total capacity.
    void Reserve(std::size_t min_capacity);

    /// Bump-allocate a contiguous block of @p count floats.
    /// Returns nullptr only when count == 0.
    float * Allocate(std::size_t count);

    /// Reset the allocation offset to zero (no deallocation).
    void Reset() noexcept;

    /// Number of floats currently in use.
    std::size_t used() const noexcept { return offset_; }

    /// Total capacity in floats.
    std::size_t capacity() const noexcept { return buffer_.size(); }

private:
    std::vector<float> buffer_;
    std::size_t offset_ = 0;
};

}  // namespace qasr
