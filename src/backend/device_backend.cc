#include "qasr/backend/device_backend.h"
#include "qasr/backend/cpu_backend.h"

namespace qasr {

std::unique_ptr<DeviceBackend> CreateCpuBackend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace qasr
