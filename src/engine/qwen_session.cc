#include "qasr/engine/qwen_session.h"
#include <cstring>
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

QwenSession::QwenSession()
    : session_id(0), model(nullptr), backend_workspace(nullptr) {}

Status QwenSession::Initialize(const V2EngineConfig & config, std::shared_ptr<QwenModel> mdl) {
    model = std::move(mdl);
    if (config.backend == BackendKind::kCuda) {
        backend = std::make_unique<CudaBackend>();
    } else {
        backend = std::make_unique<CpuBackend>();
    }
    return OkStatus();
}

Status QwenSession::AllocateWorkspace(size_t bytes) {
    if (bytes == 0) return OkStatus();
    if (backend_workspace) {
        std::free(backend_workspace);
        backend_workspace = nullptr;
    }
    backend_workspace = std::calloc(bytes, 1);
    if (!backend_workspace) {
        return Status(StatusCode::kResourceExhausted,
                      "session workspace allocation failed: " + std::to_string(bytes) + " bytes");
    }
    return OkStatus();
}

}  // namespace qasr
