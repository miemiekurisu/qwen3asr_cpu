#include "qasr/inference/encoder.h"

#include <algorithm>
#include <cstring>

namespace qasr {

Status BuildEncoderWindowPlan(std::int32_t total_frames,
                              std::int32_t chunk_frames,
                              std::int32_t window_frames,
                              EncoderWindowPlan * plan) {
    if (!plan) {
        return Status(StatusCode::kInvalidArgument, "plan must not be null");
    }
    if (total_frames <= 0) {
        return Status(StatusCode::kInvalidArgument, "total_frames must be positive");
    }
    if (chunk_frames <= 0) {
        return Status(StatusCode::kInvalidArgument, "chunk_frames must be positive");
    }
    if (window_frames <= 0 || window_frames < chunk_frames) {
        return Status(StatusCode::kInvalidArgument,
                      "window_frames must be positive and >= chunk_frames");
    }

    plan->total_frames = total_frames;
    plan->chunk_frames = chunk_frames;
    plan->window_frames = window_frames;

    // Calculate number of windows needed to cover all frames
    plan->window_starts.clear();
    std::int32_t start = 0;
    while (start < total_frames) {
        plan->window_starts.push_back(start);
        start += chunk_frames;
    }
    plan->n_windows = static_cast<std::int32_t>(plan->window_starts.size());
    return OkStatus();
}

Status EncodeChunk(const EncoderWeights & weights,
                   const float * mel, std::int32_t mel_frames,
                   std::int32_t chunk_index, const EncoderWindowPlan & plan,
                   std::vector<float> * output, std::int32_t * out_seq_len) {
    if (!mel || !output || !out_seq_len) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (!weights.loaded) {
        return Status(StatusCode::kFailedPrecondition, "encoder weights not loaded");
    }
    if (chunk_index < 0 || chunk_index >= plan.n_windows) {
        return Status(StatusCode::kOutOfRange, "chunk_index out of range");
    }
    if (mel_frames <= 0) {
        return Status(StatusCode::kInvalidArgument, "mel_frames must be positive");
    }

    // Determine the frame range for this chunk
    const std::int32_t start = plan.window_starts[static_cast<std::size_t>(chunk_index)];
    const std::int32_t end = std::min(start + plan.window_frames, mel_frames);
    const std::int32_t frames = end - start;

    // Output dimension: frames -> encoder compression (typically 4:1 subsampling)
    constexpr std::int32_t kSubsampleFactor = 4;
    const std::int32_t seq_len = (frames + kSubsampleFactor - 1) / kSubsampleFactor;
    *out_seq_len = seq_len;

    const auto total = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(weights.d_model);
    output->resize(total, 0.0f);

    // Placeholder: real implementation calls into the C backend encoder
    // For now, produce zero-valued hidden states as a stub
    return OkStatus();
}

Status EncodeAudio(const EncoderWeights & weights,
                   const float * mel, std::int32_t mel_frames,
                   std::vector<float> * output, std::int32_t * out_seq_len) {
    if (!mel || !output || !out_seq_len) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (!weights.loaded) {
        return Status(StatusCode::kFailedPrecondition, "encoder weights not loaded");
    }
    if (mel_frames <= 0) {
        return Status(StatusCode::kInvalidArgument, "mel_frames must be positive");
    }

    // Build a window plan for the full audio
    EncoderWindowPlan plan;
    Status s = BuildEncoderWindowPlan(
        mel_frames,
        /*chunk_frames=*/mel_frames,  // single window for full encode
        /*window_frames=*/mel_frames,
        &plan);
    if (!s.ok()) return s;

    // Encode the single window
    return EncodeChunk(weights, mel, mel_frames, 0, plan, output, out_seq_len);
}

Status ConcatEncoderWindows(const std::vector<std::vector<float>> & windows,
                            std::int32_t d_model,
                            std::vector<float> * output, std::int32_t * out_seq_len) {
    if (!output || !out_seq_len) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (d_model <= 0) {
        return Status(StatusCode::kInvalidArgument, "d_model must be positive");
    }
    if (windows.empty()) {
        output->clear();
        *out_seq_len = 0;
        return OkStatus();
    }

    // Calculate total sequence length
    std::size_t total_elements = 0;
    for (const auto & w : windows) {
        if (w.size() % static_cast<std::size_t>(d_model) != 0) {
            return Status(StatusCode::kInvalidArgument,
                          "window size not divisible by d_model");
        }
        total_elements += w.size();
    }

    output->resize(total_elements);
    std::size_t offset = 0;
    for (const auto & w : windows) {
        std::memcpy(output->data() + offset, w.data(), w.size() * sizeof(float));
        offset += w.size();
    }
    *out_seq_len = static_cast<std::int32_t>(total_elements / static_cast<std::size_t>(d_model));
    return OkStatus();
}

}  // namespace qasr
