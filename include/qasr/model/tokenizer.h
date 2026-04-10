#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// BPE tokenizer for Qwen2/Qwen3 models.
/// Pre: must be loaded from a valid vocab.json + merges.txt.
/// Post: encode/decode are stateless, may be called from any thread.
/// Thread-safe: immutable after Load().
class Tokenizer {
public:
    Tokenizer() = default;

    static Status Load(const std::string & vocab_json_path,
                       const std::string & merges_txt_path,
                       Tokenizer * out);

    Status Encode(const std::string & text, std::vector<std::int32_t> * token_ids) const;
    Status Decode(const std::vector<std::int32_t> & token_ids, std::string * text) const;
    Status DecodeSingle(std::int32_t token_id, std::string * piece) const;

    std::int32_t vocab_size() const noexcept { return vocab_size_; }
    bool is_loaded() const noexcept { return vocab_size_ > 0; }

private:
    std::vector<std::string> id_to_text_;
    std::int32_t vocab_size_ = 0;
    // Internal merge/vocab hash maps are opaque.
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

/// Pre: path must be a readable vocab.json file.
/// Post: populates id_to_text mapping.
/// Thread-safe: yes (file I/O only).
Status LoadVocabJson(const std::string & path, std::vector<std::string> * id_to_text);

/// Pre: path must be a readable merges.txt file.
/// Post: populates merge list in BPE priority order.
/// Thread-safe: yes (file I/O only).
Status LoadMergesTxt(const std::string & path, std::vector<std::pair<std::string, std::string>> * merges);

/// Pre: text must be valid UTF-8.
/// Post: returns byte-level BPE token IDs.
/// Thread-safe: yes.
Status EncodeUtf8(const Tokenizer & tokenizer, const std::string & text, std::vector<std::int32_t> * ids);

/// Pre: token_ids must be valid indices.
/// Post: returns decoded UTF-8 string.
/// Thread-safe: yes.
Status DecodeIds(const Tokenizer & tokenizer, const std::vector<std::int32_t> & ids, std::string * text);

}  // namespace qasr
