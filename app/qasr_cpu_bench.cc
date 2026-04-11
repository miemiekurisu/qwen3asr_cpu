#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include "qwen_asr_kernels.h"
}

namespace {

struct Scenario {
    const char *name;
    int seq_len;
    int in_dim;
    int q_dim;
    int kv_dim;
    int iterations;
};

struct Options {
    int threads = 8;
    int warmup = 5;
    int scale = 1;
};

struct Stats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

float NextPseudoRandom(std::uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return static_cast<float>((*state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

void FillPseudoRandom(std::vector<float> *values, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (float &value : *values) {
        value = NextPseudoRandom(&state);
    }
}

void SetThreadEnvironment(int threads) {
    std::ostringstream text;
    text << threads;
#ifdef _WIN32
    _putenv_s("OPENBLAS_NUM_THREADS", text.str().c_str());
    _putenv_s("OMP_NUM_THREADS", text.str().c_str());
#else
    setenv("OPENBLAS_NUM_THREADS", text.str().c_str(), 1);
    setenv("OMP_NUM_THREADS", text.str().c_str(), 1);
#endif
    qwen_set_threads(threads);
}

Stats Measure(const std::function<void()> &fn, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        times.push_back(elapsed.count());
    }

    Stats stats;
    stats.min_ms = *std::min_element(times.begin(), times.end());
    stats.max_ms = *std::max_element(times.begin(), times.end());
    stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    return stats;
}

Stats BenchmarkSeparate(const Scenario &scenario, const Options &options) {
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.in_dim));
    std::vector<float> wq(static_cast<std::size_t>(scenario.q_dim * scenario.in_dim));
    std::vector<float> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> bq(static_cast<std::size_t>(scenario.q_dim));
    std::vector<float> bk(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> q(static_cast<std::size_t>(scenario.seq_len * scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));

    FillPseudoRandom(&x, 1u);
    FillPseudoRandom(&wq, 2u);
    FillPseudoRandom(&wk, 3u);
    FillPseudoRandom(&wv, 4u);
    FillPseudoRandom(&bq, 5u);
    FillPseudoRandom(&bk, 6u);
    FillPseudoRandom(&bv, 7u);

    return Measure([&]() {
        qwen_linear(q.data(), x.data(), wq.data(), bq.data(), scenario.seq_len, scenario.in_dim, scenario.q_dim);
        qwen_linear(k.data(), x.data(), wk.data(), bk.data(), scenario.seq_len, scenario.in_dim, scenario.kv_dim);
        qwen_linear(v.data(), x.data(), wv.data(), bv.data(), scenario.seq_len, scenario.in_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkFusedScratch(const Scenario &scenario, const Options &options) {
    const int total_out = scenario.q_dim + 2 * scenario.kv_dim;
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.in_dim));
    std::vector<float> wq(static_cast<std::size_t>(scenario.q_dim * scenario.in_dim));
    std::vector<float> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> bq(static_cast<std::size_t>(scenario.q_dim));
    std::vector<float> bk(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> q(static_cast<std::size_t>(scenario.seq_len * scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(scenario.seq_len * total_out));
    std::vector<float> packed_weights(static_cast<std::size_t>(total_out * scenario.in_dim));
    std::vector<float> packed_biases(static_cast<std::size_t>(total_out));

    FillPseudoRandom(&x, 11u);
    FillPseudoRandom(&wq, 12u);
    FillPseudoRandom(&wk, 13u);
    FillPseudoRandom(&wv, 14u);
    FillPseudoRandom(&bq, 15u);
    FillPseudoRandom(&bk, 16u);
    FillPseudoRandom(&bv, 17u);

    return Measure([&]() {
        qwen_linear_qkv_f32(q.data(), k.data(), v.data(),
                            qkv_out.data(), packed_weights.data(), packed_biases.data(),
                            x.data(),
                            wq.data(), wk.data(), wv.data(),
                            bq.data(), bk.data(), bv.data(),
                            scenario.seq_len, scenario.in_dim, scenario.q_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkFusedPacked(const Scenario &scenario, const Options &options) {
    const int total_out = scenario.q_dim + 2 * scenario.kv_dim;
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.in_dim));
    std::vector<float> wq(static_cast<std::size_t>(scenario.q_dim * scenario.in_dim));
    std::vector<float> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.in_dim));
    std::vector<float> bq(static_cast<std::size_t>(scenario.q_dim));
    std::vector<float> bk(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> bv(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> q(static_cast<std::size_t>(scenario.seq_len * scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(scenario.seq_len * total_out));
    std::vector<float> packed_weights(static_cast<std::size_t>(total_out * scenario.in_dim));
    std::vector<float> packed_biases(static_cast<std::size_t>(total_out));

    FillPseudoRandom(&x, 21u);
    FillPseudoRandom(&wq, 22u);
    FillPseudoRandom(&wk, 23u);
    FillPseudoRandom(&wv, 24u);
    FillPseudoRandom(&bq, 25u);
    FillPseudoRandom(&bk, 26u);
    FillPseudoRandom(&bv, 27u);

    std::copy(wq.begin(), wq.end(), packed_weights.begin());
    std::copy(wk.begin(), wk.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>(scenario.q_dim * scenario.in_dim));
    std::copy(wv.begin(), wv.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>((scenario.q_dim + scenario.kv_dim) * scenario.in_dim));
    std::copy(bq.begin(), bq.end(), packed_biases.begin());
    std::copy(bk.begin(), bk.end(), packed_biases.begin() + scenario.q_dim);
    std::copy(bv.begin(), bv.end(), packed_biases.begin() + scenario.q_dim + scenario.kv_dim);

    return Measure([&]() {
        qwen_linear_qkv_f32_packed(q.data(), k.data(), v.data(),
                                   qkv_out.data(),
                                   x.data(),
                                   packed_weights.data(),
                                   packed_biases.data(),
                                   scenario.seq_len, scenario.in_dim, scenario.q_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkPolicy(const Scenario &scenario,
                     const Options &options,
                     qwen_enc_qkv_policy_t policy) {
    const qwen_enc_qkv_impl_t impl = qwen_select_encoder_qkv_impl(
        policy, scenario.seq_len, scenario.q_dim, 1);
    if (impl == QWEN_ENC_QKV_IMPL_PACKED) {
        return BenchmarkFusedPacked(scenario, options);
    }
    return BenchmarkSeparate(scenario, options);
}

bool ParseInt(std::string_view text, int *value) {
    try {
        *value = std::stoi(std::string(text));
        return true;
    } catch (...) {
        return false;
    }
}

void PrintUsage(const char *program) {
    std::cout << program << " [--threads <n>] [--warmup <n>] [--scale <n>]\n";
}

}  // namespace

int main(int argc, char **argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--threads" && index + 1 < argc) {
            if (!ParseInt(argv[++index], &options.threads) || options.threads <= 0) {
                std::cerr << "invalid --threads\n";
                return 1;
            }
            continue;
        }
        if (arg == "--warmup" && index + 1 < argc) {
            if (!ParseInt(argv[++index], &options.warmup) || options.warmup < 0) {
                std::cerr << "invalid --warmup\n";
                return 1;
            }
            continue;
        }
        if (arg == "--scale" && index + 1 < argc) {
            if (!ParseInt(argv[++index], &options.scale) || options.scale <= 0) {
                std::cerr << "invalid --scale\n";
                return 1;
            }
            continue;
        }
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argc > 0 ? argv[0] : "qasr_cpu_bench");
            return 0;
        }
        std::cerr << "unknown arg: " << arg << "\n";
        PrintUsage(argc > 0 ? argv[0] : "qasr_cpu_bench");
        return 1;
    }

    SetThreadEnvironment(options.threads);

    const Scenario scenarios[] = {
        {"enc_qkv_seq4_d1024", 4, 1024, 1024, 1024, 80},
        {"enc_qkv_seq13_d1024", 13, 1024, 1024, 1024, 60},
        {"enc_qkv_seq52_d1024", 52, 1024, 1024, 1024, 24},
        {"enc_qkv_seq104_d1024", 104, 1024, 1024, 1024, 12},
        {"enc_qkv_seq13_d896", 13, 896, 896, 896, 72}
    };

    std::cout << "threads=" << options.threads << " warmup=" << options.warmup << " scale=" << options.scale << "\n";
    std::cout << "runtime_kernel_backend=" << qwen_get_runtime_kernel_backend_name()
              << " default_qkv_policy=" << qwen_encoder_qkv_policy_name(qwen_get_encoder_qkv_policy())
              << "\n";
    std::cout << std::left << std::setw(20) << "scenario"
              << std::right << std::setw(12) << "separate"
              << std::setw(14) << "fused_copy"
              << std::setw(16) << "fused_packed"
              << std::setw(14) << "shape_auto"
              << std::setw(14) << "auto_path"
              << std::setw(16) << "copy_speedup"
              << std::setw(18) << "packed_speedup"
              << std::setw(16) << "auto_speedup" << "\n";

    for (const Scenario &scenario : scenarios) {
        const Stats separate = BenchmarkSeparate(scenario, options);
        const Stats fused_copy = BenchmarkFusedScratch(scenario, options);
        const Stats fused_packed = BenchmarkFusedPacked(scenario, options);
        const Stats shape_auto = BenchmarkPolicy(scenario, options, QWEN_ENC_QKV_POLICY_SHAPE_AUTO);
        const qwen_enc_qkv_impl_t shape_impl = qwen_select_encoder_qkv_impl(
            QWEN_ENC_QKV_POLICY_SHAPE_AUTO, scenario.seq_len, scenario.q_dim, 1);
        const double copy_speedup = separate.avg_ms / fused_copy.avg_ms;
        const double packed_speedup = separate.avg_ms / fused_packed.avg_ms;
        const double auto_speedup = separate.avg_ms / shape_auto.avg_ms;

        std::cout << std::left << std::setw(20) << scenario.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << separate.avg_ms
                  << std::setw(14) << fused_copy.avg_ms
                  << std::setw(16) << fused_packed.avg_ms
                  << std::setw(14) << shape_auto.avg_ms
                  << std::setw(14) << qwen_encoder_qkv_impl_name(shape_impl)
                  << std::setw(16) << copy_speedup
                  << std::setw(18) << packed_speedup
                  << std::setw(16) << auto_speedup << "\n";
    }

    return 0;
}