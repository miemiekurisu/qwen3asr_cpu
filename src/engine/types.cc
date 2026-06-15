#include "qasr/engine/types.h"

namespace qasr {

std::int64_t TensorShape::size() const {
    std::int64_t s = 1;
    for (int i = 0; i < ndim; i++) {
        s *= dims[i];
    }
    return s;
}

bool TensorShape::valid() const {
    if (ndim <= 0 || ndim > 8) return false;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return false;
    }
    return true;
}

size_t TensorHandle::nbytes() const {
    size_t bytes_per_elem = 4;
    switch (dtype) {
        case Dtype::kBf16:
        case Dtype::kFp16:
        case Dtype::kInt16:  bytes_per_elem = 2; break;
        case Dtype::kInt32:  bytes_per_elem = 4; break;
        case Dtype::kFp32:   bytes_per_elem = 4; break;
    }
    return static_cast<size_t>(shape.size()) * bytes_per_elem;
}

bool TensorHandle::valid() const {
    return opaque && shape.valid();
}

}  // namespace qasr
