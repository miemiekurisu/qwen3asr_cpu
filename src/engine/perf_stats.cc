#include "qasr/engine/perf_stats.h"
#include <sstream>
#include <iomanip>

namespace qasr {

std::string FormatPerfJson(const ServerPerfStats & server,
                           const SessionPerfStats & session,
                           const BackendPerfStats & backend) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);

    os << "{\n";

    /* Server metrics */
    os << "  \"server\": {\n";
    os << "    \"active_sessions\": " << server.active_sessions.load() << ",\n";
    os << "    \"queue_depth\": " << server.queue_depth.load() << ",\n";
    os << "    \"queue_wait_ms\": " << server.queue_wait_ms << ",\n";
    os << "    \"gpu_busy_ratio\": " << server.gpu_busy_ratio << ",\n";
    os << "    \"deadline_miss_count\": " << server.deadline_miss_count.load() << "\n";
    os << "  },\n";

    /* Session metrics */
    os << "  \"session\": {\n";
    os << "    \"session_id\": " << session.session_id << ",\n";
    os << "    \"segment_id\": " << session.segment_id << ",\n";
    os << "    \"segment_e2e_ms\": " << session.segment_e2e_ms << ",\n";
    os << "    \"queue_wait_ms\": " << session.queue_wait_ms << ",\n";
    os << "    \"infer_ms\": " << session.infer_ms << ",\n";
    os << "    \"output_lag_ms\": " << session.output_lag_ms << ",\n";
    os << "    \"total_segments\": " << session.total_segments << ",\n";
    os << "    \"total_tokens\": " << session.total_tokens << "\n";
    os << "  },\n";

    /* Backend metrics */
    os << "  \"backend\": {\n";
    os << "    \"encoder_ms\": " << backend.encoder_ms << ",\n";
    os << "    \"prefill_ms\": " << backend.prefill_ms << ",\n";
    os << "    \"decode_ms\": " << backend.decode_ms << ",\n";
    os << "    \"ms_per_token\": " << backend.ms_per_token << ",\n";
    os << "    \"tokens_per_sec\": " << backend.tokens_per_sec << ",\n";
    os << "    \"cuda_h2d_ms\": " << backend.cuda_h2d_ms << ",\n";
    os << "    \"cuda_d2h_ms\": " << backend.cuda_d2h_ms << ",\n";
    os << "    \"cuda_sync_ms\": " << backend.cuda_sync_ms << ",\n";
    os << "    \"cuda_fallback_count\": " << backend.cuda_fallback_count << "\n";
    os << "  }\n";

    os << "}";
    return os.str();
}

}  // namespace qasr
