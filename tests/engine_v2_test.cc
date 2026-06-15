/*
 * engine_v2_test.cc — Engine v2 type and utility tests (M1 verification).
 *
 * Tests Engine v2 types, config parsing, TensorShape/TensorHandle utilities.
 * CPU engine lifecycle tests are deferred to integration test with model.
 *
 * CI-safe: all tests run without external model or GPU.
 */

#include "tests/test_registry.h"

#include "qasr/engine/config.h"
#include "qasr/engine/types.h"
#include "qasr/engine/qwen_model.h"
#include "qasr/core/status.h"

#include <string>

QASR_TEST(ParseBackendKindIdentifiesCpu) {
    QASR_EXPECT_EQ(qasr::ParseBackendKind("cpu"), qasr::BackendKind::kCpu);
    QASR_EXPECT_EQ(qasr::ParseBackendKind(""), qasr::BackendKind::kCpu);
    QASR_EXPECT_EQ(qasr::ParseBackendKind("unknown"), qasr::BackendKind::kCpu);
}

QASR_TEST(ParseBackendKindIdentifiesCuda) {
    QASR_EXPECT_EQ(qasr::ParseBackendKind("cuda"), qasr::BackendKind::kCuda);
    QASR_EXPECT_EQ(qasr::ParseBackendKind("CUDA"), qasr::BackendKind::kCuda);
}

QASR_TEST(ParseBackendKindIdentifiesMlx) {
    QASR_EXPECT_EQ(qasr::ParseBackendKind("mlx"), qasr::BackendKind::kMlx);
    QASR_EXPECT_EQ(qasr::ParseBackendKind("MLX"), qasr::BackendKind::kMlx);
}

QASR_TEST(V2EngineConfigDefaults) {
    qasr::V2EngineConfig cfg;
    QASR_EXPECT_EQ(cfg.backend, qasr::BackendKind::kCpu);
    QASR_EXPECT_EQ(cfg.platform, qasr::PlatformProfile::kGenericCpu);
    QASR_EXPECT_EQ(cfg.device_id, 0);
    QASR_EXPECT_EQ(cfg.threads, 0);
    QASR_EXPECT(!cfg.allow_backend_fallback);
    QASR_EXPECT_EQ(cfg.max_sessions, 1);
    QASR_EXPECT_EQ(cfg.max_active_gpu_jobs, 1);
    QASR_EXPECT(!cfg.enable_decode_microbatch);
    QASR_EXPECT_EQ(cfg.residency, qasr::V2EngineConfig::Residency::kCpuOnly);
    QASR_EXPECT_EQ(cfg.precision, qasr::V2EngineConfig::Precision::kFp32);
}

QASR_TEST(TensorShapeDefaultIsEmpty) {
    qasr::TensorShape shape;
    QASR_EXPECT_EQ(shape.ndim, 0);
    QASR_EXPECT(!shape.valid());
    QASR_EXPECT_EQ(shape.size(), 1LL);
}

QASR_TEST(TensorShape1D) {
    qasr::TensorShape shape(128);
    QASR_EXPECT_EQ(shape.ndim, 1);
    QASR_EXPECT(shape.valid());
    QASR_EXPECT_EQ(shape.size(), 128LL);
}

QASR_TEST(TensorShape2D) {
    qasr::TensorShape shape(64, 128);
    QASR_EXPECT_EQ(shape.ndim, 2);
    QASR_EXPECT(shape.valid());
    QASR_EXPECT_EQ(shape.size(), 8192LL);
}

QASR_TEST(TensorShapeInvalidNegative) {
    qasr::TensorShape shape;
    shape.dims[0] = -1;
    shape.ndim = 1;
    QASR_EXPECT(!shape.valid());
}

QASR_TEST(TensorShapeInvalidZeroDim) {
    qasr::TensorShape shape(0);
    QASR_EXPECT(!shape.valid());
}

QASR_TEST(TensorHandleDefaultsAreCpuFp32) {
    qasr::TensorHandle handle;
    QASR_EXPECT_EQ(handle.device, qasr::TensorHandle::Device::kCpu);
    QASR_EXPECT_EQ(handle.dtype, qasr::TensorHandle::Dtype::kFp32);
    QASR_EXPECT(!handle.valid());
}

QASR_TEST(TensorHandleFp32Nbytes) {
    qasr::TensorHandle handle;
    handle.shape = qasr::TensorShape(128, 64);
    handle.dtype = qasr::TensorHandle::Dtype::kFp32;
    QASR_EXPECT_EQ(handle.nbytes(), 128UL * 64 * 4UL);
}

QASR_TEST(TensorHandleBf16Nbytes) {
    qasr::TensorHandle handle;
    handle.shape = qasr::TensorShape(128, 64);
    handle.dtype = qasr::TensorHandle::Dtype::kBf16;
    QASR_EXPECT_EQ(handle.nbytes(), 128UL * 64 * 2UL);
}

QASR_TEST(TensorHandleValidWithOpaque) {
    qasr::TensorHandle handle;
    handle.shape = qasr::TensorShape(32);
    handle.opaque = reinterpret_cast<void *>(0x1);
    QASR_EXPECT(handle.valid());
}

QASR_TEST(TensorHandleInvalidWithoutOpaque) {
    qasr::TensorHandle handle;
    handle.shape = qasr::TensorShape(32);
    handle.opaque = nullptr;
    QASR_EXPECT(!handle.valid());
}

QASR_TEST(TensorHandleInvalidWithBadShape) {
    qasr::TensorHandle handle;
    handle.shape.ndim = 0;
    handle.opaque = reinterpret_cast<void *>(0x1);
    QASR_EXPECT(!handle.valid());
}

QASR_TEST(AutoDetectPlatformKnown) {
    auto p = qasr::AutoDetectPlatform();
    QASR_EXPECT(p == qasr::PlatformProfile::kGenericCpu ||
                p == qasr::PlatformProfile::kLinuxAarch64Cpu ||
                p == qasr::PlatformProfile::kLinuxX86_64Cpu ||
                p == qasr::PlatformProfile::kMacosArm64Mlx);
}

QASR_TEST(StatusCodeResourceExhausted) {
    QASR_EXPECT_EQ(std::string(qasr::StatusCodeName(qasr::StatusCode::kResourceExhausted)),
                    "RESOURCE_EXHAUSTED");
    QASR_EXPECT_EQ(std::string(qasr::StatusCodeName(qasr::StatusCode::kOk)), "OK");
    QASR_EXPECT_EQ(std::string(qasr::StatusCodeName(qasr::StatusCode::kUnimplemented)),
                    "UNIMPLEMENTED");
}

QASR_TEST(ModelConfigDefaults) {
    qasr::ModelConfig cfg;
    QASR_EXPECT(cfg.model_dir.empty());
    QASR_EXPECT_EQ(cfg.num_layers, 0);
    QASR_EXPECT_EQ(cfg.num_heads, 0);
    QASR_EXPECT_EQ(cfg.hidden_size, 0);
    QASR_EXPECT_EQ(cfg.vocab_size, 0);
    QASR_EXPECT_EQ(cfg.mel_dim, 0);
}
