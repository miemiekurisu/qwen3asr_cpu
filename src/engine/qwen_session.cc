#include "qasr/engine/qwen_session.h"
#include <algorithm>

namespace qasr {

void AudioRingBuffer::Push(const float * data, int count) {
    std::lock_guard<std::mutex> lock(mu_);
    buffer_.insert(buffer_.end(), data, data + count);
}

int AudioRingBuffer::Drain(std::vector<float> & out, int max_samples) {
    std::lock_guard<std::mutex> lock(mu_);
    int n = std::min(max_samples, static_cast<int>(buffer_.size()));
    out.assign(buffer_.begin(), buffer_.begin() + n);
    buffer_.erase(buffer_.begin(), buffer_.begin() + n);
    return n;
}

Status QwenSession::Initialize(const V2EngineConfig & config, std::shared_ptr<QwenModel> mdl) {
    session_id = 0;
    model = std::move(mdl);
    if (config.backend == BackendKind::kCuda) {
        backend = std::make_unique<CudaBackend>();
    } else {
        backend = std::make_unique<CpuBackend>();
    }
    return OkStatus();
}

}  // namespace qasr
