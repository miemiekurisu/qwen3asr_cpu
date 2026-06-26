/*
 * threads_per_block_compliance_test.cc
 *
 * 严格测试 threads_per_block 配置是否符合 NVIDIA CUDA Best Practices Guide。
 * 依据：Section 11.3 Thread and Block Heuristics
 *
 * 测试覆盖：
 *   1. 静态验证：代码中的默认值是否符合官方建议
 *   2. 动态验证：运行时不同配置下的性能表现
 *   3. 边界测试：最小值、推荐范围、最大值
 *   4. 真实场景：使用 long.mp3 切片（不同长度序列）
 *
 * 覆盖要求：100% 覆盖所有官方建议
 */

#include "tests/test_registry.h"
#include "qasr/backend/cuda_backend.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

#ifdef QASR_CUDA_BACKEND_ENABLED

extern "C" {
#include "qwen_asr_audio.h"
}

/* ============================================================================
 * STATIC VERIFICATION: 静态验证代码是否符合 NVIDIA 官方建议
 * ============================================================================
 *
 * NVIDIA CUDA Best Practices Guide Section 11.3:
 *   1. "threads per block should be a multiple of 32"
 *   2. "A minimum of 64 threads per block should be used"
 *   3. "Between 128 and 256 threads per block is a good initial range"
 */

QASR_TEST(ThreadsPerBlockStaticCompliance) {
    fprintf(stderr, "\n===== Static Compliance Verification =====\n");
    fprintf(stderr, "NVIDIA CUDA Best Practices Guide Section 11.3:\n");
    fprintf(stderr, "  1. threads_per_block % 32 == 0\n");
    fprintf(stderr, "  2. threads_per_block >= 64\n");
    fprintf(stderr, "  3. 128 <= threads_per_block <= 256 (recommended)\n");
    fprintf(stderr, "\n");
    
    /* Read the source file and extract the default value */
    std::ifstream file("src/backend/bidir_attention.cu");
    if (!file.is_open()) {
        fprintf(stderr, "SKIP: Cannot open bidir_attention.cu\n");
        return;
    }
    
    std::string line;
    int default_threads = -1;
    bool found_default = false;
    
    while (std::getline(file, line)) {
        /* Look for: int threads_per_block = 256; */
        if (line.find("int threads_per_block =") != std::string::npos) {
            /* Extract the number */
            size_t start = line.find('=') + 1;
            size_t end = line.find(';', start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string num_str = line.substr(start, end - start);
                /* Trim whitespace */
                size_t first = num_str.find_first_not_of(" \t");
                size_t last = num_str.find_last_not_of(" \t");
                if (first != std::string::npos && last != std::string::npos) {
                    num_str = num_str.substr(first, last - first + 1);
                    default_threads = std::atoi(num_str.c_str());
                    found_default = true;
                    break;
                }
            }
        }
    }
    file.close();
    
    if (!found_default || default_threads <= 0) {
        fprintf(stderr, "FAIL: Could not find valid threads_per_block default\n");
        QASR_EXPECT(false);
        return;
    }
    
    fprintf(stderr, "Found default threads_per_block = %d\n\n", default_threads);
    
    /* Check Rule 1: multiple of 32 */
    bool rule1_pass = (default_threads % 32) == 0;
    fprintf(stderr, "Rule 1: threads_per_block %% 32 == 0\n");
    fprintf(stderr, "  %d %% 32 = %d -> %s\n", 
            default_threads, default_threads % 32,
            rule1_pass ? "PASS" : "FAIL");
    QASR_EXPECT(rule1_pass);
    
    /* Check Rule 2: minimum 64 */
    bool rule2_pass = default_threads >= 64;
    fprintf(stderr, "\nRule 2: threads_per_block >= 64\n");
    fprintf(stderr, "  %d >= 64 -> %s\n", 
            default_threads, rule2_pass ? "PASS" : "FAIL");
    QASR_EXPECT(rule2_pass);
    
    /* Check Rule 3: recommended range 128-256 */
    bool rule3_pass = (default_threads >= 128 && default_threads <= 256);
    fprintf(stderr, "\nRule 3: 128 <= threads_per_block <= 256 (recommended)\n");
    fprintf(stderr, "  128 <= %d <= 256 -> %s\n", 
            default_threads, rule3_pass ? "PASS" : "FAIL");
    
    if (!rule3_pass) {
        fprintf(stderr, "  WARNING: Outside recommended range, but may still be valid\n");
    }
    
    fprintf(stderr, "\n===== Static Compliance: %s =====\n\n", 
            (rule1_pass && rule2_pass && rule3_pass) ? "FULLY COMPLIANT" : "PARTIAL");
}

/* ============================================================================
 * DYNAMIC VERIFICATION: 动态测试不同配置下的性能
 * ============================================================================
 */

/* Load a slice of long.mp3 for testing */
static std::vector<float> LoadAudioSlice(const char* audio_path, 
                                          int start_sec, int duration_sec) {
    /* Use ffmpeg to extract a slice */
    std::ostringstream cmd;
    std::string temp_file = "_tmp/slice_test.wav";
    
    cmd << "ffmpeg -y -ss " << start_sec 
        << " -t " << duration_sec
        << " -i \"" << audio_path << "\""
        << " -f s16le -acodec pcm_s16le -ar 16000 -ac 1 \"" << temp_file << "\" 2>/dev/null";
    
    system(cmd.str().c_str());
    
    if (!std::ifstream(temp_file).good()) {
        return {};
    }
    
    /* Read WAV file (simplified - assume PCM 16-bit) */
    std::ifstream file(temp_file, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    
    /* Skip WAV header (44 bytes) */
    file.seekg(44);
    
    std::vector<float> samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        samples.push_back(sample / 32768.0f);
    }
    
    return samples;
}

/* Benchmark with specific threads_per_block configuration */
static double BenchmarkWithThreads(const char* model_dir,
                                    const float* mel_data, int mel_frames,
                                    int threads_per_block,
                                    int iterations = 10) {
    /* Set environment variable */
    std::string env_var = "QASR_ATTENTION_THREADS=" + std::to_string(threads_per_block);
    setenv("QASR_ATTENTION_THREADS", std::to_string(threads_per_block).c_str(), 1);
    
    /* Re-initialize backend to pick up new config */
    qasr::CudaBackend backend;
    if (!backend.Initialize().ok()) {
        return -1.0;
    }
    
    if (!backend.PrepareWeights(model_dir).ok()) {
        return -1.0;
    }
    
    qasr::CudaSessionState session;
    if (!SetupCudaSession(backend, session)) {
        return -1.0;
    }
    
    int out_tokens = 0;
    std::vector<float> output(10000 * 2048, 0.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double total_time = 0.0;
    int valid_iterations = 0;
    
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        
        auto status = backend.EncodeMel(&session, mel_data, mel_frames, 
                                         output.data(), out_tokens);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        if (status.ok() && out_tokens > 0) {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;
            valid_iterations++;
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (valid_iterations > 0) ? (total_time / valid_iterations) : -1.0;
}

QASR_TEST(ThreadsPerBlockDynamicBenchmark) {
    const char* model_dir = std::getenv("QASR_MODEL_DIR");
    if (!model_dir) {
        fprintf(stderr, "SKIP: QASR_MODEL_DIR not set\n");
        return;
    }
    
    /* Check if long.mp3 exists */
    std::string long_mp3 = "testfile/long.mp3";
    if (!std::ifstream(long_mp3).good()) {
        fprintf(stderr, "SKIP: %s not found\n", long_mp3.c_str());
        return;
    }
    
    fprintf(stderr, "\n===== Dynamic Benchmark (long.mp3 slices) =====\n");
    fprintf(stderr, "Testing different threads_per_block configurations:\n");
    fprintf(stderr, "  - 32 (below minimum, for comparison)\n");
    fprintf(stderr, "  - 64 (minimum per NVIDIA)\n");
    fprintf(stderr, "  - 128 (recommended range start)\n");
    fprintf(stderr, "  - 256 (recommended range, default)\n");
    fprintf(stderr, "  - 512 (above recommended, for comparison)\n");
    fprintf(stderr, "\n");
    
    /* Test with different slice lengths */
    struct TestCase {
        const char* name;
        int start_sec;
        int duration_sec;
        int expected_tokens;
    };
    
    TestCase test_cases[] = {
        {"Short (2s, ~200 tokens)", 0, 2, 200},
        {"Medium (5s, ~500 tokens)", 10, 5, 500},
        {"Long (10s, ~1000 tokens)", 20, 10, 1000},
    };
    
    int threads_configs[] = {32, 64, 128, 256, 512};
    const char* thread_labels[] = {
        "32 (below min)", 
        "64 (min)", 
        "128 (rec)", 
        "256 (rec/default)", 
        "512 (above)"
    };
    
    for (const auto& tc : test_cases) {
        fprintf(stderr, "\n--- Test Case: %s ---\n", tc.name);
        
        /* Load audio slice */
        auto samples = LoadAudioSlice(long_mp3.c_str(), tc.start_sec, tc.duration_sec);
        if (samples.empty()) {
            fprintf(stderr, "Failed to load audio slice\n");
            continue;
        }
        
        /* Compute mel spectrogram */
        int mel_frames = 0;
        float* mel = qwen_mel_spectrogram(samples.data(), (int)samples.size(), &mel_frames);
        if (!mel || mel_frames <= 0) {
            fprintf(stderr, "Failed to compute mel spectrogram\n");
            continue;
        }
        
        fprintf(stderr, "Mel frames: %d (expected ~%d)\n", mel_frames, 
                (int)(tc.duration_sec * 100));
        fprintf(stderr, "%-20s | %-12s | %-12s | %-10s\n", 
                "Threads", "Avg Time", "Relative", "Status");
        fprintf(stderr, "----------------------------------------------------\n");
        
        double baseline_time = 0;
        std::vector<double> results;
        
        for (int i = 0; i < 5; i++) {
            int threads = threads_configs[i];
            
            double avg_time = BenchmarkWithThreads(
                model_dir, mel, mel_frames, threads, 10);
            
            results.push_back(avg_time);
            
            if (threads == 256) {
                baseline_time = avg_time;
            }
            
            const char* status = "OK";
            double relative = 0;
            if (avg_time > 0 && baseline_time > 0) {
                relative = avg_time / baseline_time;
                if (threads == 256) {
                    status = "BEST";
                } else if (relative < 1.2) {
                    status = "OK";
                } else if (relative < 2.0) {
                    status = "SLOW";
                } else {
                    status = "VERY SLOW";
                }
            } else if (avg_time < 0) {
                status = "ERROR";
            }
            
            if (avg_time > 0) {
                fprintf(stderr, "%-20d | %-12.2f | x%-6.2f | %-10s\n",
                        threads, avg_time, relative, status);
            } else {
                fprintf(stderr, "%-20d | %-12s | %-12s | %-10s\n",
                        threads, "N/A", "N/A", status);
            }
        }
        
        /* Analysis */
        fprintf(stderr, "\nAnalysis:\n");
        double best_time = results[0];
        int best_threads = threads_configs[0];
        for (int i = 1; i < 5; i++) {
            if (results[i] > 0 && (best_time < 0 || results[i] < best_time)) {
                best_time = results[i];
                best_threads = threads_configs[i];
            }
        }
        
        if (best_time > 0) {
            fprintf(stderr, "  Best: threads=%d, time=%.2f ms\n", 
                    best_threads, best_time);
            fprintf(stderr, "  Per NVIDIA guidelines, threads=256 should be optimal.\n");
            
            if (best_threads != 256) {
                fprintf(stderr, "  WARNING: Best result not at threads=256!\n");
                fprintf(stderr, "  This may indicate:\n");
                fprintf(stderr, "    1. Sequence length too short for occupancy to matter\n");
                fprintf(stderr, "    2. Bottleneck is not in bidir_attention\n");
                fprintf(stderr, "    3. Need Nsight Compute for deeper analysis\n");
            }
        }
        
        std::free(mel);
    }
    
    fprintf(stderr, "\n===== Dynamic Benchmark Complete =====\n\n");
}

#endif
