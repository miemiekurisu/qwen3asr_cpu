#pragma once

#include "qasr/engine/asr_engine.h"
#include "qasr/backend/cpu_backend.h"
#include "qasr/backend/cuda_backend.h"
#include "qasr/engine/qwen_model.h"
#include "qasr/engine/qwen_session.h"
#include "qasr/scheduler/scheduler.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>

namespace qasr {

class CudaAsrEngine final : public AsrEngine {
public:
    CudaAsrEngine() = default;
    ~CudaAsrEngine() override = default;

    Status LoadModel(const V2EngineConfig & config) override;
    Status CreateSession(const SessionOptions & opts, std::uint64_t & out_id) override;
    Status CloseSession(std::uint64_t session_id) override;
    AsrSegmentResult TranscribeSegment(std::uint64_t session_id,
                                        const std::vector<float> & samples,
                                        int64_t sample_rate = 16000,
                                        TokenCallback on_token = {},
                                        CancelCallback on_cancel = {}) override;
    int ActiveSessionCount() const override;
    const V2EngineConfig & config() const override { return config_; }

private:
    V2EngineConfig config_;
    std::shared_ptr<QwenModel> model_;
    std::shared_ptr<CudaBackend> cuda_backend_;
    std::shared_ptr<CpuBackend> cpu_fallback_;
    GpuScheduler scheduler_;

    std::unordered_map<std::uint64_t, std::shared_ptr<QwenSession>> sessions_;
    uint64_t next_session_id_ = 1;
    mutable std::mutex mu_;
};

}  // namespace qasr
