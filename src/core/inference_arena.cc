#include "qasr/core/inference_arena.h"

namespace qasr {

InferenceArena::InferenceArena(std::size_t initial_capacity) {
    if (initial_capacity > 0) {
        buffer_.resize(initial_capacity);
    }
}

void InferenceArena::Reserve(std::size_t min_capacity) {
    if (min_capacity > buffer_.size()) {
        buffer_.resize(min_capacity);
    }
}

float * InferenceArena::Allocate(std::size_t count) {
    if (count == 0) {
        return nullptr;
    }
    const std::size_t required = offset_ + count;
    if (required > buffer_.size()) {
        std::size_t new_cap = buffer_.size() > 0 ? buffer_.size() : 1024;
        while (new_cap < required) {
            new_cap *= 2;
        }
        buffer_.resize(new_cap);
    }
    float * ptr = buffer_.data() + offset_;
    offset_ = required;
    return ptr;
}

void InferenceArena::Reset() noexcept {
    offset_ = 0;
}

}  // namespace qasr
