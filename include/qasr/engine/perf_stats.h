#pragma once

#include <atomic>
#include <string>
#include <cstdint>

namespace qasr {

/* Server-level perf metrics */
struct ServerPerfStats {
    std::atomic<int> active_sessions{0};
    std::atomic<int> queue_depth{0};
    double queue_wait_ms = 0.0;
    double gpu_busy_ratio = 0.0;
    std::atomic<int> deadline_miss_count{0};
};

/* Per-session perf metrics */
struct SessionPerfStats {
    std::uint64_t session_id = 0;
    std::uint64_t segment_id = 0;
    double segment_e2e_ms = 0.0;
    double queue_wait_ms = 0.0;
    double infer_ms = 0.0;
    double output_lag_ms = 0.0;
    double total_infer_ms = 0.0;
    double total_encode_ms = 0.0;
    double total_decode_ms = 0.0;
    int total_segments = 0;
    int total_tokens = 0;
};

/* Backend-level perf metrics (CPU / CUDA) */
struct BackendPerfStats {
    double encoder_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ms_per_token = 0.0;
    double tokens_per_sec = 0.0;

    /* CUDA-specific */
    double cuda_h2d_ms = 0.0;
    double cuda_d2h_ms = 0.0;
    double cuda_sync_ms = 0.0;
    int cuda_fallback_count = 0;

    /* Derived metrics */
    double xrt() const { return tokens_per_sec > 0 ? 1.0 / tokens_per_sec : 0.0; }
    double rtf() const { return tokens_per_sec > 0 ? tokens_per_sec : 0.0; }
};

std::string FormatPerfJson(const ServerPerfStats & server,
                           const SessionPerfStats & session,
                           const BackendPerfStats & backend);

}  // namespace qasr
