#pragma once

#include <cstddef>
#include <cstdint>

namespace qasr {

enum class BackendKind {
    kCpu,
    kCuda,
    kMlx
};

enum class PlatformProfile {
    kGenericCpu,
    kLinuxX86_64Cpu,
    kLinuxAarch64Cpu,
    kDgxSparkCuda13Sm121,
    kH200Cuda12Sm90,
    kWindowsCuda12Sm89,
    kMacosArm64Mlx
};

struct TensorShape {
    std::int64_t dims[8] = {0};
    int ndim = 0;

    TensorShape() = default;
    TensorShape(std::int64_t d0) : dims{d0}, ndim{1} {}
    TensorShape(std::int64_t d0, std::int64_t d1) : dims{d0, d1}, ndim{2} {}
    std::int64_t size() const;
    bool valid() const;
};

struct TensorHandle {
    enum class Device { kCpu, kCuda, kMlx };
    enum class Dtype { kFp32, kBf16, kFp16, kInt32, kInt16 };

    Device device = Device::kCpu;
    Dtype dtype = Dtype::kFp32;
    TensorShape shape;
    void* opaque = nullptr;

    size_t nbytes() const;
    bool valid() const;
};

}  // namespace qasr
