/*
 * qasr_safety_microbench.cc - Micro-benchmark for the safety hardening
 * changes introduced in this change set.
 *
 * Goals:
 *   1. Measure the cost of the new header_size bounds checks in
 *      safetensors_open / SafeTensorIndex::Build, exercising the
 *      happy path (valid file) and the rejection path (crafted file).
 *   2. Measure the cost of qwen_grow_buffer for the in-place grow
 *      pattern used by the streaming decoder tail buffer.
 *
 * This is intentionally a small, deterministic benchmark that runs in
 * under a second on a workstation.  Numbers should match those produced
 * by the existing `qasr_cpu_bench` in terms of methodology (warmup,
 * min/avg/max over a fixed iteration count).
 *
 * Run:  build/linux-openblas/qasr_safety_microbench [--iterations N]
 */
#include "tests/test_registry.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "qasr/storage/safetensors_loader.h"

extern "C" {
#include "qwen_asr_alloc.h"
}

namespace {

struct BenchStats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

BenchStats Measure(int iterations, const std::function<void()> & fn) {
    // Warmup
    for (int i = 0; i < 3; ++i) fn();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }
    BenchStats s;
    s.min_ms = *std::min_element(times.begin(), times.end());
    s.max_ms = *std::max_element(times.begin(), times.end());
    s.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) /
                static_cast<double>(times.size());
    return s;
}

std::vector<std::uint8_t> MakeValidSafetensorsFile() {
    // Minimal valid safetensors: 8-byte header_size=2 + "{}".
    std::vector<std::uint8_t> bytes(10, 0);
    bytes[0] = 2;  // header_size LE = 2
    bytes[8] = '{';
    bytes[9] = '}';
    return bytes;
}

std::vector<std::uint8_t> MakeOversizedHeaderFile() {
    // 8-byte header_size = 1 KiB, but the file is only 16 bytes.
    std::vector<std::uint8_t> bytes(16, 0);
    std::uint64_t header_size = 1024;
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>((header_size >> (i * 8)) & 0xFFu);
    }
    return bytes;
}

std::string WriteTempFile(const std::string & name, const std::vector<std::uint8_t> & bytes) {
    const std::string path = "/tmp/qasr_safety_microbench_" + name;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
}

void Cleanup(const std::string & path) {
    std::remove(path.c_str());
}

}  // namespace

QASR_TEST(Microbench_SafetensorsOpen_Valid) {
    const auto bytes = MakeValidSafetensorsFile();
    const std::string path = WriteTempFile("valid.bin", bytes);
    const int iterations = 1000;

    auto fn = [&]() {
        qasr::MappedFile mapped;
        qasr::MappedFile::Open(path, &mapped);
        qasr::SafeTensorIndex index;
        qasr::SafeTensorIndex::Build(mapped, &index);
    };
    const BenchStats s = Measure(iterations, fn);
    std::printf("[microbench] safetensors_open valid    : avg=%.4f ms  min=%.4f ms  max=%.4f ms (n=%d)\n",
                s.avg_ms, s.min_ms, s.max_ms, iterations);
    Cleanup(path);
}

QASR_TEST(Microbench_SafetensorsOpen_OversizedHeader) {
    const auto bytes = MakeOversizedHeaderFile();
    const std::string path = WriteTempFile("oversize.bin", bytes);
    const int iterations = 1000;

    auto fn = [&]() {
        qasr::MappedFile mapped;
        qasr::MappedFile::Open(path, &mapped);
        qasr::SafeTensorIndex index;
        qasr::SafeTensorIndex::Build(mapped, &index);
    };
    const BenchStats s = Measure(iterations, fn);
    std::printf("[microbench] safetensors_open reject   : avg=%.4f ms  min=%.4f ms  max=%.4f ms (n=%d)\n",
                s.avg_ms, s.min_ms, s.max_ms, iterations);
    Cleanup(path);
}

QASR_TEST(Microbench_GrowBuffer_RepeatGrow) {
    // Simulate the streaming tail-buffer grow pattern: every round
    // asks for +1 element, doubling kicks in periodically.
    const int iterations = 1000;
    auto fn = []() {
        void * buffer = nullptr;
        std::size_t cap = 0;
        for (int i = 1; i <= 32; ++i) {
            qwen_grow_buffer(&buffer, sizeof(int), cap, static_cast<std::size_t>(i), &cap);
        }
        std::free(buffer);
    };
    const BenchStats s = Measure(iterations, fn);
    std::printf("[microbench] grow_buffer 1->32 ints   : avg=%.4f ms  min=%.4f ms  max=%.4f ms (n=%d)\n",
                s.avg_ms, s.min_ms, s.max_ms, iterations);
}

QASR_TEST(Microbench_GrowBuffer_Stable) {
    // Steady-state: caller already has enough capacity, so the helper
    // must return success without doing any work.
    const int iterations = 1000;
    std::vector<int> backing(64);
    void * buffer = backing.data();
    std::size_t cap = 64;
    auto fn = [&]() {
        void * local = buffer;
        std::size_t new_cap = 0;
        qwen_grow_buffer(&local, sizeof(int), cap, 32, &new_cap);
    };
    const BenchStats s = Measure(iterations, fn);
    std::printf("[microbench] grow_buffer no-op         : avg=%.4f ms  min=%.4f ms  max=%.4f ms (n=%d)\n",
                s.avg_ms, s.min_ms, s.max_ms, iterations);
}
