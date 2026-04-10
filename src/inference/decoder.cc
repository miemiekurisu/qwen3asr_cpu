#include "qasr/inference/decoder.h"

#include <algorithm>
#include <cstring>

namespace qasr {

// --- KvCache ---

Status KvCache::Allocate(std::int32_t n_layers, std::int32_t n_kv_heads,
                          std::int32_t head_dim, std::int32_t max_seq_len) {
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) {
        return Status(StatusCode::kInvalidArgument,
                      "all KvCache dimensions must be positive");
    }

    const auto per_layer = static_cast<std::size_t>(n_kv_heads) *
                           static_cast<std::size_t>(head_dim) *
                           static_cast<std::size_t>(max_seq_len);
    const auto total = per_layer * static_cast<std::size_t>(n_layers);

    keys_.resize(total, 0.0f);
    values_.resize(total, 0.0f);
    length_ = 0;
    capacity_ = max_seq_len;
    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    return OkStatus();
}

void KvCache::Reset() noexcept {
    length_ = 0;
    if (!keys_.empty()) {
        std::fill(keys_.begin(), keys_.end(), 0.0f);
        std::fill(values_.begin(), values_.end(), 0.0f);
    }
}

// --- Decode functions ---

Status Prefill(const DecoderWeights & weights,
               const float * embeddings, std::int32_t seq_len,
               KvCache * cache) {
    if (!embeddings || !cache) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (!weights.loaded) {
        return Status(StatusCode::kFailedPrecondition, "decoder weights not loaded");
    }
    if (!cache->is_allocated()) {
        return Status(StatusCode::kFailedPrecondition, "KV cache not allocated");
    }
    if (seq_len <= 0) {
        return Status(StatusCode::kInvalidArgument, "seq_len must be positive");
    }
    if (seq_len > cache->capacity()) {
        return Status(StatusCode::kOutOfRange, "seq_len exceeds KV cache capacity");
    }

    // Placeholder: real implementation runs transformer prefill through C backend
    cache->set_length(seq_len);
    return OkStatus();
}

Status DecodeStep(const DecoderWeights & weights,
                  const float * embed,
                  KvCache * cache,
                  std::int32_t * out_token_id) {
    if (!embed || !cache || !out_token_id) {
        return Status(StatusCode::kInvalidArgument, "null pointer argument");
    }
    if (!weights.loaded) {
        return Status(StatusCode::kFailedPrecondition, "decoder weights not loaded");
    }
    if (!cache->is_allocated()) {
        return Status(StatusCode::kFailedPrecondition, "KV cache not allocated");
    }
    if (cache->length() >= cache->capacity()) {
        return Status(StatusCode::kOutOfRange, "KV cache full");
    }

    // Placeholder: real implementation runs single decode step through C backend
    // Returns EOS token (151643) as default
    *out_token_id = 151643;
    cache->set_length(cache->length() + 1);
    return OkStatus();
}

Status BuildPromptEmbeddings(const DecoderWeights & weights,
                             const std::int32_t * prompt_token_ids,
                             std::int32_t n_prompt_tokens,
                             const float * audio_embeds,
                             std::int32_t audio_seq_len,
                             std::int32_t hidden_dim,
                             std::vector<float> * output,
                             std::int32_t * out_seq_len) {
    if (!output || !out_seq_len) {
        return Status(StatusCode::kInvalidArgument, "output pointers must not be null");
    }
    if (!weights.loaded) {
        return Status(StatusCode::kFailedPrecondition, "decoder weights not loaded");
    }
    if (hidden_dim <= 0) {
        return Status(StatusCode::kInvalidArgument, "hidden_dim must be positive");
    }

    const std::int32_t total_seq = n_prompt_tokens + audio_seq_len;
    if (total_seq <= 0) {
        return Status(StatusCode::kInvalidArgument, "no tokens to embed");
    }
    *out_seq_len = total_seq;

    const auto total = static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(hidden_dim);
    output->resize(total, 0.0f);

    // Placeholder: real implementation looks up prompt embeddings from weight table
    // and concatenates with audio embeddings
    if (audio_embeds && audio_seq_len > 0) {
        const auto audio_offset = static_cast<std::size_t>(n_prompt_tokens) *
                                  static_cast<std::size_t>(hidden_dim);
        const auto audio_bytes = static_cast<std::size_t>(audio_seq_len) *
                                 static_cast<std::size_t>(hidden_dim) * sizeof(float);
        std::memcpy(output->data() + audio_offset, audio_embeds, audio_bytes);
    }

    (void)prompt_token_ids;  // Used in real embedding lookup
    return OkStatus();
}

}  // namespace qasr
