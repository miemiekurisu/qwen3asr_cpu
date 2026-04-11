#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    int prefill_threads = 0;
    int decode_threads = 0;
    int warmup = 5;
    int scale = 1;
};

struct Stats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct DecoderQkvScenario {
    const char *name;
    int seq_len;
    int hidden;
    int q_dim;
    int kv_dim;
    int iterations;
};

struct DecoderGateUpScenario {
    const char *name;
    int seq_len;
    int hidden;
    int intermediate;
    int iterations;
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

std::uint16_t FloatToBfloat16(float value) {
    std::uint32_t bits = 0;
    std::uint16_t bf16 = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bf16 = static_cast<std::uint16_t>(bits >> 16);
    return bf16;
}

void FillPseudoRandomBfloat16(std::vector<std::uint16_t> *values, std::uint32_t seed) {
    std::vector<float> scratch(values->size());
    FillPseudoRandom(&scratch, seed);
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = FloatToBfloat16(scratch[index]);
    }
}

void ConvertBfloat16ToFloat(std::vector<float> *dst, const std::vector<std::uint16_t> &src) {
    for (std::size_t index = 0; index < src.size(); ++index) {
        std::uint32_t bits = static_cast<std::uint32_t>(src[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        (*dst)[index] = value;
    }
}

int EffectivePrefillThreads(const Options &options) {
    return options.prefill_threads > 0 ? options.prefill_threads : options.threads;
}

int EffectiveDecodeThreads(const Options &options) {
    return options.decode_threads > 0 ? options.decode_threads : options.threads;
}

void SetThreadEnvironment(const Options &options) {
    std::ostringstream text;
    text << EffectivePrefillThreads(options);
#ifdef _WIN32
    _putenv_s("OPENBLAS_NUM_THREADS", text.str().c_str());
    _putenv_s("OMP_NUM_THREADS", text.str().c_str());
#else
    setenv("OPENBLAS_NUM_THREADS", text.str().c_str(), 1);
    setenv("OMP_NUM_THREADS", text.str().c_str(), 1);
#endif
    qwen_set_thread_policy_override(EffectivePrefillThreads(options), EffectiveDecodeThreads(options));
    qwen_apply_prefill_thread_policy();
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

    qwen_apply_prefill_thread_policy();

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

    qwen_apply_prefill_thread_policy();

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

    qwen_apply_prefill_thread_policy();

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

Stats BenchmarkDecoderPrefillQkvBf16(const DecoderQkvScenario &scenario, const Options &options) {
    const int total_out = scenario.q_dim + 2 * scenario.kv_dim;
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.hidden));
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(scenario.q_dim * scenario.hidden));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<float> q(static_cast<std::size_t>(scenario.seq_len * scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> qkv_out(static_cast<std::size_t>(scenario.seq_len * total_out));
    std::vector<float> qkv_weights(static_cast<std::size_t>(total_out * scenario.hidden));

    FillPseudoRandom(&x, 3001u);
    FillPseudoRandomBfloat16(&wq, 3003u);
    FillPseudoRandomBfloat16(&wk, 3005u);
    FillPseudoRandomBfloat16(&wv, 3007u);

    qwen_apply_prefill_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias_bf16_qkv_prefill(q.data(), k.data(), v.data(),
                                            qkv_out.data(), qkv_weights.data(), x.data(),
                                            wq.data(), wk.data(), wv.data(),
                                            scenario.seq_len, scenario.hidden,
                                            scenario.q_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkDecoderPrefillQkvPacked(const DecoderQkvScenario &scenario, const Options &options) {
    const int total_out = scenario.q_dim + 2 * scenario.kv_dim;
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.hidden));
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(scenario.q_dim * scenario.hidden));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<float> packed_weights(static_cast<std::size_t>(total_out * scenario.hidden));
    std::vector<float> qkv_out(static_cast<std::size_t>(scenario.seq_len * total_out));
    std::vector<float> q(static_cast<std::size_t>(scenario.seq_len * scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.seq_len * scenario.kv_dim));
    std::vector<float> tmp_q(static_cast<std::size_t>(scenario.q_dim * scenario.hidden));
    std::vector<float> tmp_k(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<float> tmp_v(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));

    FillPseudoRandom(&x, 3011u);
    FillPseudoRandomBfloat16(&wq, 3013u);
    FillPseudoRandomBfloat16(&wk, 3015u);
    FillPseudoRandomBfloat16(&wv, 3017u);
    ConvertBfloat16ToFloat(&tmp_q, wq);
    ConvertBfloat16ToFloat(&tmp_k, wk);
    ConvertBfloat16ToFloat(&tmp_v, wv);
    std::copy(tmp_q.begin(), tmp_q.end(), packed_weights.begin());
    std::copy(tmp_k.begin(), tmp_k.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>(scenario.q_dim * scenario.hidden));
    std::copy(tmp_v.begin(), tmp_v.end(), packed_weights.begin() + static_cast<std::ptrdiff_t>((scenario.q_dim + scenario.kv_dim) * scenario.hidden));

    qwen_apply_prefill_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias_qkv_f32_packed(q.data(), k.data(), v.data(),
                                          qkv_out.data(), x.data(), packed_weights.data(),
                                          scenario.seq_len, scenario.hidden,
                                          scenario.q_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkDecoderPrefillGateUpBf16(const DecoderGateUpScenario &scenario, const Options &options) {
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.hidden));
    std::vector<std::uint16_t> gate_up(static_cast<std::size_t>(scenario.hidden * 2 * scenario.intermediate));
    std::vector<float> out(static_cast<std::size_t>(scenario.seq_len * 2 * scenario.intermediate));

    FillPseudoRandom(&x, 4001u);
    FillPseudoRandomBfloat16(&gate_up, 4003u);

    qwen_apply_prefill_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias_bf16(out.data(), x.data(), gate_up.data(),
                                scenario.seq_len, scenario.hidden,
                                2 * scenario.intermediate);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkDecoderPrefillGateUpPacked(const DecoderGateUpScenario &scenario, const Options &options) {
    std::vector<float> x(static_cast<std::size_t>(scenario.seq_len * scenario.hidden));
    std::vector<std::uint16_t> gate_up_bf16(static_cast<std::size_t>(scenario.hidden * 2 * scenario.intermediate));
    std::vector<float> gate_up_f32(static_cast<std::size_t>(scenario.hidden * 2 * scenario.intermediate));
    std::vector<float> out(static_cast<std::size_t>(scenario.seq_len * 2 * scenario.intermediate));

    FillPseudoRandom(&x, 4011u);
    FillPseudoRandomBfloat16(&gate_up_bf16, 4013u);
    ConvertBfloat16ToFloat(&gate_up_f32, gate_up_bf16);

    qwen_apply_prefill_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias(out.data(), x.data(), gate_up_f32.data(),
                           scenario.seq_len, scenario.hidden,
                           2 * scenario.intermediate);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkDecoderDecodeQkvSingleToken(const DecoderQkvScenario &scenario, const Options &options) {
    std::vector<float> x(static_cast<std::size_t>(scenario.hidden));
    std::vector<std::uint16_t> wq(static_cast<std::size_t>(scenario.q_dim * scenario.hidden));
    std::vector<std::uint16_t> wk(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<std::uint16_t> wv(static_cast<std::size_t>(scenario.kv_dim * scenario.hidden));
    std::vector<float> q(static_cast<std::size_t>(scenario.q_dim));
    std::vector<float> k(static_cast<std::size_t>(scenario.kv_dim));
    std::vector<float> v(static_cast<std::size_t>(scenario.kv_dim));

    FillPseudoRandom(&x, 5001u);
    FillPseudoRandomBfloat16(&wq, 5003u);
    FillPseudoRandomBfloat16(&wk, 5005u);
    FillPseudoRandomBfloat16(&wv, 5007u);

    qwen_apply_decode_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias_bf16_qkv(q.data(), k.data(), v.data(), x.data(),
                                    wq.data(), wk.data(), wv.data(),
                                    scenario.hidden, scenario.q_dim, scenario.kv_dim);
    }, options.warmup, scenario.iterations * options.scale);
}

Stats BenchmarkDecoderDecodeGateUpSingleToken(const DecoderGateUpScenario &scenario, const Options &options) {
    std::vector<float> x(static_cast<std::size_t>(scenario.hidden));
    std::vector<std::uint16_t> gate_up(static_cast<std::size_t>(scenario.hidden * 2 * scenario.intermediate));
    std::vector<float> out(static_cast<std::size_t>(2 * scenario.intermediate));

    FillPseudoRandom(&x, 6001u);
    FillPseudoRandomBfloat16(&gate_up, 6003u);

    qwen_apply_decode_thread_policy();

    return Measure([&]() {
        qwen_linear_nobias_bf16(out.data(), x.data(), gate_up.data(),
                                1, scenario.hidden, 2 * scenario.intermediate);
    }, options.warmup, scenario.iterations * options.scale);
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
    std::cout << program << " [--threads <n>] [--prefill-threads <n>] [--decode-threads <n>] [--warmup <n>] [--scale <n>]\n";
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
        if (arg == "--prefill-threads" && index + 1 < argc) {
            if (!ParseInt(argv[++index], &options.prefill_threads) || options.prefill_threads <= 0) {
                std::cerr << "invalid --prefill-threads\n";
                return 1;
            }
            continue;
        }
        if (arg == "--decode-threads" && index + 1 < argc) {
            if (!ParseInt(argv[++index], &options.decode_threads) || options.decode_threads <= 0) {
                std::cerr << "invalid --decode-threads\n";
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

    SetThreadEnvironment(options);

    const Scenario scenarios[] = {
        {"enc_qkv_seq4_d1024", 4, 1024, 1024, 1024, 80},
        {"enc_qkv_seq13_d1024", 13, 1024, 1024, 1024, 60},
        {"enc_qkv_seq52_d1024", 52, 1024, 1024, 1024, 24},
        {"enc_qkv_seq104_d1024", 104, 1024, 1024, 1024, 12},
        {"enc_qkv_seq13_d896", 13, 896, 896, 896, 72}
    };
    const DecoderQkvScenario decoder_qkv_scenarios[] = {
        {"dec_prefill_qkv_seq64_h2048", 64, 2048, 2048, 1024, 8},
        {"dec_prefill_qkv_seq32_h1024", 32, 1024, 2048, 1024, 12},
    };
    const DecoderGateUpScenario decoder_gate_up_scenarios[] = {
        {"dec_gate_up_seq64_h2048_i6144", 64, 2048, 6144, 4},
        {"dec_gate_up_seq32_h1024_i3072", 32, 1024, 3072, 6},
    };
    const DecoderQkvScenario decode_qkv_scenarios[] = {
        {"dec_decode_qkv_h2048", 1, 2048, 2048, 1024, 120},
        {"dec_decode_qkv_h1024", 1, 1024, 2048, 1024, 180},
    };
    const DecoderGateUpScenario decode_gate_up_scenarios[] = {
        {"dec_decode_gate_up_h2048_i6144", 1, 2048, 6144, 48},
        {"dec_decode_gate_up_h1024_i3072", 1, 1024, 3072, 80},
    };

    std::cout << "threads=" << options.threads
              << " prefill_threads=" << EffectivePrefillThreads(options)
              << " decode_threads=" << EffectiveDecodeThreads(options)
              << " warmup=" << options.warmup << " scale=" << options.scale << "\n";
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

    std::cout << "\n[decoder_prefill_qkv]\n";
    std::cout << std::left << std::setw(28) << "scenario"
              << std::right << std::setw(14) << "bf16_path"
              << std::setw(16) << "packed_f32"
              << std::setw(16) << "speedup"
              << std::setw(16) << "cache_mb" << "\n";
    for (const DecoderQkvScenario &scenario : decoder_qkv_scenarios) {
        const Stats bf16_path = BenchmarkDecoderPrefillQkvBf16(scenario, options);
        const Stats packed_f32 = BenchmarkDecoderPrefillQkvPacked(scenario, options);
        const double speedup = bf16_path.avg_ms / packed_f32.avg_ms;
        const double cache_mb = static_cast<double>((scenario.q_dim + 2 * scenario.kv_dim) * scenario.hidden * sizeof(float)) /
                                (1024.0 * 1024.0);
        std::cout << std::left << std::setw(28) << scenario.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << bf16_path.avg_ms
                  << std::setw(16) << packed_f32.avg_ms
                  << std::setw(16) << speedup
                  << std::setw(16) << cache_mb << "\n";
    }

    std::cout << "\n[decoder_prefill_gate_up]\n";
    std::cout << std::left << std::setw(32) << "scenario"
              << std::right << std::setw(14) << "bf16_path"
              << std::setw(16) << "packed_f32"
              << std::setw(16) << "speedup"
              << std::setw(16) << "cache_mb" << "\n";
    for (const DecoderGateUpScenario &scenario : decoder_gate_up_scenarios) {
        const Stats bf16_path = BenchmarkDecoderPrefillGateUpBf16(scenario, options);
        const Stats packed_f32 = BenchmarkDecoderPrefillGateUpPacked(scenario, options);
        const double speedup = bf16_path.avg_ms / packed_f32.avg_ms;
        const double cache_mb = static_cast<double>(scenario.hidden * 2 * scenario.intermediate * sizeof(float)) /
                                (1024.0 * 1024.0);
        std::cout << std::left << std::setw(32) << scenario.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << bf16_path.avg_ms
                  << std::setw(16) << packed_f32.avg_ms
                  << std::setw(16) << speedup
                  << std::setw(16) << cache_mb << "\n";
    }

    std::cout << "\n[decoder_decode]\n";
    std::cout << std::left << std::setw(28) << "scenario"
              << std::right << std::setw(14) << "qkv_ms"
              << std::setw(16) << "gate_up_ms" << "\n";
    for (std::size_t index = 0; index < (sizeof(decode_qkv_scenarios) / sizeof(decode_qkv_scenarios[0])); ++index) {
        const Stats qkv_stats = BenchmarkDecoderDecodeQkvSingleToken(decode_qkv_scenarios[index], options);
        const Stats gate_up_stats = BenchmarkDecoderDecodeGateUpSingleToken(decode_gate_up_scenarios[index], options);
        std::cout << std::left << std::setw(28) << decode_qkv_scenarios[index].name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << qkv_stats.avg_ms
                  << std::setw(16) << gate_up_stats.avg_ms << "\n";
    }

    qwen_clear_thread_policy_override();

    return 0;
}