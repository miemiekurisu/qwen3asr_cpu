#pragma once

#include <cstdint>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Encoder weight view: thin C++ reference to mmap'd weight data.
/// Pre: must be populated by LoadEncoderWeights from a ShardRegistry.
/// Post: pointers are valid as long as the ShardRegistry is alive.
/// Thread-safe: immutable after construction (read-only shared).
struct EncoderWeights {
    std::int32_t d_model = 0;
    std::int32_t n_layers = 0;
    std::int32_t n_heads = 0;
    std::int32_t head_dim = 0;
    std::int32_t ffn_dim = 0;
    std::int32_t output_dim = 0;
    bool loaded = false;
};

/// Plan for chunked encoder windowing.
/// Pre: total_frames > 0, chunk_frames > 0.
/// Post: windows covers all frames with overlap.
/// Thread-safe: immutable after construction.
struct EncoderWindowPlan {
    std::int32_t total_frames = 0;
    std::int32_t chunk_frames = 0;
    std::int32_t window_frames = 0;
    std::int32_t n_windows = 0;
    std::vector<std::int32_t> window_starts;
};

/// Pre: total_frames > 0.
/// Post: plan covers all input frames.
/// Thread-safe: yes.
Status BuildEncoderWindowPlan(std::int32_t total_frames,
                              std::int32_t chunk_frames,
                              std::int32_t window_frames,
                              EncoderWindowPlan * plan);

/// Pre: mel is [mel_bins, n_frames], chunk defined by plan.
/// Post: output is encoder hidden states for the chunk.
/// Thread-safe: no (uses internal scratch buffers).
Status EncodeChunk(const EncoderWeights & weights,
                   const float * mel, std::int32_t mel_frames,
                   std::int32_t chunk_index, const EncoderWindowPlan & plan,
                   std::vector<float> * output, std::int32_t * out_seq_len);

/// Pre: mel is [mel_bins, n_frames], weights loaded.
/// Post: output is full encoder hidden state sequence.
/// Thread-safe: no.
Status EncodeAudio(const EncoderWeights & weights,
                   const float * mel, std::int32_t mel_frames,
                   std::vector<float> * output, std::int32_t * out_seq_len);

/// Pre: windows are compatible (same d_model).
/// Post: output contains concatenated windows.
/// Thread-safe: yes.
Status ConcatEncoderWindows(const std::vector<std::vector<float>> & windows,
                            std::int32_t d_model,
                            std::vector<float> * output, std::int32_t * out_seq_len);

}  // namespace qasr
