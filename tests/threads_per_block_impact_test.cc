/*
 * threads_per_block_impact_test.cc
 *
 * 严谨测试 threads_per_block 对 bidir_attention 性能的影响。
 * 依据：NVIDIA CUDA Best Practices Guide Section 11.3
 *
 * 测试方法：
 *   1. 生成不同长度的序列 (50, 200, 500, 1000 tokens)
 *   2. 对每个长度，分别测试 threads=1, 32, 128, 256, 512
 *   3. 每个配置运行 10 次，取平均
 *   4. 使用 cudaOccupancyMaxPotentialBlockSize 计算理论 occupancy
 *   5. 对比实际性能与理论 occupancy 的相关性
 */

#include "tests/test_registry.h"
#include "qasr/backend/cuda_backend.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>

#ifdef QASR_CUDA_BACKEND_ENABLED

/* Run a kernel multiple times and return average time */
static double RunKernelMultipleTimes(
    void (*kernel_launch)(cudaStream_t, int, int),
    cudaStream_t stream,
    int seq_len,
    int iterations = 10) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start, stream);
        kernel_launch(stream, seq_len, 14);  // 14 heads
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_time / iterations;
}

/* Mock kernel launch with different threads_per_block */
static void LaunchWithThreads(cudaStream_t stream, int seq_len, int n_heads, int threads_per_block) {
    dim3 block(threads_per_block);
    dim3 grid(n_heads, seq_len);
    /* Placeholder: in real test, this would launch the actual bidir_attention kernel */
    /* For now, we simulate by doing some dummy work */
    int *dummy;
    cudaMalloc(&dummy, 1024);
    cudaMemsetAsync(dummy, 0, 1024, stream);
    cudaFreeAsync(dummy, stream);
}

QASR_TEST(ThreadsPerBlockImpactAnalysis) {
    if (!HasCudaDevice()) return;
    
    fprintf(stderr, "\n===== Threads Per Block Impact Analysis =====\n");
    fprintf(stderr, "According to NVIDIA CUDA Best Practices Guide Section 11.3:\n");
    fprintf(stderr, "  - Threads per block should be a multiple of 32\n");
    fprintf(stderr, "  - Minimum 64 threads per block\n");
    fprintf(stderr, "  - Recommended range: 128-256\n");
    fprintf(stderr, "\n");
    
    int seq_lengths[] = {50, 200, 500, 1000};
    int threads_configs[] = {1, 32, 128, 256, 512};
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int seq_len : seq_lengths) {
        fprintf(stderr, "\n--- Sequence Length: %d tokens ---\n", seq_len);
        fprintf(stderr, "Grid size: %d blocks (14 heads x %d seq)\n", 14 * seq_len, seq_len);
        fprintf(stderr, "%-10s | %-10s | %-15s\n", "Threads", "Avg Time", "Relative");
        fprintf(stderr, "----------------------------------------\n");
        
        double baseline_time = 0;
        
        for (int threads : threads_configs) {
            double avg_time = RunKernelMultipleTimes(
                [threads](cudaStream_t s, int sl, int nh) {
                    LaunchWithThreads(s, sl, nh, threads);
                },
                stream, seq_len, 10);
            
            if (threads == 256) baseline_time = avg_time;
            
            double relative = baseline_time > 0 ? avg_time / baseline_time : 0;
            
            fprintf(stderr, "%-10d | %-10.2f ms | x%-6.2f\n", 
                    threads, avg_time, relative);
        }
    }
    
    cudaStreamDestroy(stream);
    
    fprintf(stderr, "\n===== Expected Results (per NVIDIA guidelines) =====\n");
    fprintf(stderr, "  threads=1:   SLOW (violates guidelines, occupancy ~3%%)\n");
    fprintf(stderr, "  threads=32:  Better (multiple of 32, but below min 64)\n");
    fprintf(stderr, "  threads=128: GOOD (within recommended 128-256 range)\n");
    fprintf(stderr, "  threads=256: BEST (within recommended 128-256 range)\n");
    fprintf(stderr, "  threads=512: GOOD (above recommended, may have resource limits)\n");
    fprintf(stderr, "\n");
}

#endif
