/*
 * cuda_backend_test.cc — CUDA backend stub and shell tests.
 *
 * Tests the CUDA backend stub (CPU-only build):
 *   - CudaBuffer allocation/free
 *   - CudaBackend initialize (stub path)
 *   - CudaStreamHandle stub
 *   - CublasHandle stub
 *   - CudaWeights defaults
 *   - CudaSessionState defaults
 *
 * When compiled with CUDA (QASR_CUDA_BACKEND_ENABLED):
 *   - CudaBackend initialize (real CUDA)
 *   - Device properties populated
 *
 * CI-safe: runs in CPU-only mode when CUDA not available.
 */

#include "tests/test_registry.h"

#include "qasr/backend/cuda_backend.h"
#include "qasr/engine/config.h"

#include <cstdio>

QASR_TEST(CudaBufferDefaultEmpty) {
    qasr::CudaBuffer buf;
    QASR_EXPECT(!buf.data());
    QASR_EXPECT_EQ(buf.size(), 0u);
}

QASR_TEST(CudaBufferAllocateAndReset) {
    qasr::CudaBuffer buf;
    auto status = buf.Allocate(256);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(buf.data() != nullptr);
    QASR_EXPECT_EQ(buf.size(), 256u);

    buf.Reset();
    QASR_EXPECT(!buf.data());
    QASR_EXPECT_EQ(buf.size(), 0u);
}

QASR_TEST(CudaBufferAllocateZeroIsNoop) {
    qasr::CudaBuffer buf;
    auto status = buf.Allocate(0);
    QASR_EXPECT(status.ok());
    QASR_EXPECT(!buf.data());
}

QASR_TEST(CudaBufferMoveConstructor) {
    qasr::CudaBuffer buf;
    buf.Allocate(128);
    void * ptr = buf.data();
    QASR_EXPECT(ptr != nullptr);

    qasr::CudaBuffer moved(std::move(buf));
    QASR_EXPECT_EQ(moved.data(), ptr);
    QASR_EXPECT_EQ(moved.size(), 128u);
    QASR_EXPECT(!buf.data());
}

QASR_TEST(CudaBufferMoveAssignment) {
    qasr::CudaBuffer buf;
    buf.Allocate(128);
    void * ptr = buf.data();

    qasr::CudaBuffer target;
    target.Allocate(64);
    target = std::move(buf);
    QASR_EXPECT_EQ(target.data(), ptr);
    QASR_EXPECT_EQ(target.size(), 128u);
    QASR_EXPECT(!buf.data());
}

QASR_TEST(CudaBackendInitializeStub) {
    qasr::CudaBackend backend;
    auto status = backend.Initialize();
    QASR_EXPECT(status.ok());
}

QASR_TEST(CudaBackendPrepareWeightsFailsOnBadDir) {
    qasr::CudaBackend backend;
    backend.Initialize();
    auto status = backend.PrepareWeights("/nonexistent/model");
    QASR_EXPECT(!status.ok());
}

QASR_TEST(CudaBackendPrepareWeightsSuccess) {
    const char * env = std::getenv("QASR_MODEL_DIR");
    if (!env) {
        return; /* Skip if no model dir */
    }
    qasr::CudaBackend backend;
    auto status = backend.Initialize();
    QASR_EXPECT(status.ok());
    status = backend.PrepareWeights(env);
    QASR_EXPECT(status.ok());
}

QASR_TEST(CudaBackendEncodeMelNeedsEncoder) {
    qasr::CudaBackend backend;
    backend.Initialize();
    int out_tokens = 0;
    auto status = backend.EncodeMel(nullptr, nullptr, 0, nullptr, out_tokens);
    QASR_EXPECT(!status.ok());
    /* EncoderForward requires encoder_ready + valid session; fails with kFailedPrecondition */
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kFailedPrecondition);
}

QASR_TEST(CudaBackendDecoderPrefillNeedsWeights) {
    qasr::CudaBackend backend;
    backend.Initialize();
    auto status = backend.DecoderPrefill(nullptr, nullptr, 0, nullptr, 0);
    QASR_EXPECT(!status.ok());
}

QASR_TEST(CudaBackendDecodeStepNeedsWeights) {
    qasr::CudaBackend backend;
    backend.Initialize();
    std::int32_t out_token = 0;
    auto status = backend.DecodeStep(nullptr, out_token);
    QASR_EXPECT(!status.ok());
}

QASR_TEST(CudaBackendResetDecoderOk) {
    qasr::CudaBackend backend;
    backend.Initialize();
    auto status = backend.ResetDecoder(nullptr);
    QASR_EXPECT(status.ok());
}

QASR_TEST(CudaBackendWorkspaceBytesNonZero) {
    qasr::CudaBackend backend;
    backend.Initialize();
    qasr::V2EngineConfig cfg;
    size_t bytes = backend.WorkspaceBytes(cfg);
    QASR_EXPECT(bytes > 0u);
    /* Expect ~100 MB per session (KV cache + decoder buffers) */
    QASR_EXPECT(bytes > 50ULL * 1024 * 1024);
    QASR_EXPECT(bytes < 200ULL * 1024 * 1024);
}

QASR_TEST(CudaBackendShutdown) {
    qasr::CudaBackend backend;
    backend.Initialize();
    auto status = backend.Shutdown();
    QASR_EXPECT(status.ok());

    // Re-initialize should work
    status = backend.Initialize();
    QASR_EXPECT(status.ok());
}

QASR_TEST(CudaBackendDoubleShutdown) {
    qasr::CudaBackend backend;
    backend.Initialize();
    backend.Shutdown();
    backend.Shutdown();
}

QASR_TEST(CudaBackendKindReturnsCuda) {
    qasr::CudaBackend backend;
    QASR_EXPECT_EQ(backend.kind(), qasr::BackendKind::kCuda);
}

QASR_TEST(CudaBackendDeviceIdDefault) {
    qasr::CudaBackend backend;
    QASR_EXPECT_EQ(backend.device_id(), 0);
}

QASR_TEST(CudaWeightsDefaults) {
    qasr::CudaWeights weights;
    QASR_EXPECT(!weights.decoder_ready);
    QASR_EXPECT(!weights.lm_head_ready);
    QASR_EXPECT(!weights.encoder_ready);
}

QASR_TEST(CudaSessionStateDefaults) {
    qasr::CudaSessionState state;
    QASR_EXPECT_EQ(state.current_seq_len, 0);
    QASR_EXPECT_EQ(state.stream_index, 0);
}

QASR_TEST(CudaStreamHandleStubCreate) {
    qasr::CudaStreamHandle handle;
    auto status = handle.Create();
    QASR_EXPECT(status.ok());
}

QASR_TEST(CublasHandleStubCreate) {
    qasr::CublasHandle handle;
    auto status = handle.Create();
    QASR_EXPECT(status.ok());
}

QASR_TEST(CublasHandleStubSetStream) {
    qasr::CublasHandle handle;
    handle.Create();
    auto status = handle.SetStream(nullptr);
    QASR_EXPECT(status.ok());
}
