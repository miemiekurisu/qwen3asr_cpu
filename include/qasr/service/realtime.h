#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

struct RealtimePolicyConfig {
    int sample_rate_hz = 16000;
    int min_decode_interval_ms = 800;
    int max_unstable_ms = 12000;
    int max_decode_window_ms = 32000;
};

struct RealtimeTextState {
    std::string stable_text;
    std::string last_text;
    std::size_t last_decode_samples = 0;
    std::size_t unstable_since_samples = 0;
};

struct RealtimeTextUpdate {
    bool committed = false;
    std::string stable_text;
    std::string partial_text;
    std::string text;
};

Status ValidateRealtimePolicyConfig(const RealtimePolicyConfig & config);
std::size_t RealtimeMaxDecodeSamples(const RealtimePolicyConfig & config);
std::size_t TrimRealtimeSamples(std::vector<float> * samples, std::size_t max_samples);
bool RealtimeShouldDecode(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::size_t last_decode_samples,
    bool force);
Status AdvanceRealtimeTextState(
    const RealtimePolicyConfig & config,
    std::size_t total_samples,
    std::string_view latest_text,
    bool force_finalize,
    RealtimeTextState * state,
    RealtimeTextUpdate * update);

}  // namespace qasr
