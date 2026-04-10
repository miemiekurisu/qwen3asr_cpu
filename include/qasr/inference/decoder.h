#pragma once

#include <cstdint>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Decoder weight view: thin C++ reference to mmap'd weight data.
/// Pre: must be populated from a ShardRegistry.
/// Post: immutable; valid as long as ShardRegistry is alive.
/// Thread-safe: read-only shared.
struct DecoderWeights {
    std::int32_t hidden_dim = 0;
    std::int32_t n_layers = 0;
    std::int32_t n_heads = 0;
    std::int32_t n_kv_heads = 0;
    std::int32_t head_dim = 0;
    std::int32_t intermediate_dim = 0;
    std::int32_t vocab_size = 0;
    bool loaded = false;
};

/// KV cache for autoregressive decoding.
/// Pre: capacity > 0, valid dimensions from DecoderWeights.
/// Post: stores key/value tensors up to capacity.
/// Thread-safe: NOT thread-safe; owned by one session.
class KvCache {
public:
    KvCache() = default;

    Status Allocate(std::int32_t n_layers, std::int32_t n_kv_heads,
                    std::int32_t head_dim, std::int32_t max_seq_len);
    void Reset() noexcept;

    std::int32_t length() const noexcept { return length_; }
    std::int32_t capacity() const noexcept { return capacity_; }
    bool is_allocated() const noexcept { return capacity_ > 0; }

    float * key_data() noexcept { return keys_.data(); }
    float * value_data() noexcept { return values_.data(); }
    const float * key_data() const noexcept { return keys_.data(); }
    const float * value_data() const noexcept { return values_.data(); }

    void set_length(std::int32_t len) noexcept { length_ = len; }

private:
    std::vector<float> keys_;
    std::vector<float> values_;
    std::int32_t length_ = 0;
    std::int32_t capacity_ = 0;
    std::int32_t n_layers_ = 0;
    std::int32_t n_kv_heads_ = 0;
    std::int32_t head_dim_ = 0;
};

/// Multi-token prefill of the decoder.
/// Pre: embeddings is [seq_len, hidden_dim], cache allocated.
/// Post: KV cache updated with prefilled keys/values.
/// Thread-safe: no.
Status Prefill(const DecoderWeights & weights,
               const float * embeddings, std::int32_t seq_len,
               KvCache * cache);

/// Single-token decode step.
/// Pre: embed is [1, hidden_dim], cache has prior context.
/// Post: returns greedy token ID; cache extended by 1.
/// Thread-safe: no.
Status DecodeStep(const DecoderWeights & weights,
                  const float * embed,
                  KvCache * cache,
                  std::int32_t * out_token_id);

/// Assemble prompt + audio embeddings for prefill.
/// Pre: prompt_ids/audio_embeds valid.
/// Post: output is concatenated embedding sequence.
/// Thread-safe: yes.
Status BuildPromptEmbeddings(const DecoderWeights & weights,
                             const std::int32_t * prompt_token_ids,
                             std::int32_t n_prompt_tokens,
                             const float * audio_embeds,
                             std::int32_t audio_seq_len,
                             std::int32_t hidden_dim,
                             std::vector<float> * output,
                             std::int32_t * out_seq_len);

}  // namespace qasr
