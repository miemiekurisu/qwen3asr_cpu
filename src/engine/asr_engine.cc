#include "qasr/engine/asr_engine.h"
#include "qasr/engine/cpu_asr_engine.h"
#include "qasr/engine/cuda_asr_engine.h"

namespace qasr {

std::unique_ptr<AsrEngine> CreateEngine(BackendKind backend) {
    switch (backend) {
        case BackendKind::kCuda:
            return std::make_unique<CudaAsrEngine>();
        case BackendKind::kMlx:
            return nullptr;
        case BackendKind::kCpu:
        default:
            return std::make_unique<CpuAsrEngine>();
    }
}

}  // namespace qasr
