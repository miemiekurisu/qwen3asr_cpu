#include "qasr/model/tokenizer.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace qasr {

struct Tokenizer::Impl {
    std::unordered_map<std::string, std::int32_t> text_to_id;
    std::vector<std::pair<std::string, std::string>> merges;
};

Status Tokenizer::Load(const std::string & vocab_json_path,
                       const std::string & merges_txt_path,
                       Tokenizer * out) {
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }

    std::vector<std::string> id_to_text;
    Status s = LoadVocabJson(vocab_json_path, &id_to_text);
    if (!s.ok()) return s;

    std::vector<std::pair<std::string, std::string>> merges;
    s = LoadMergesTxt(merges_txt_path, &merges);
    if (!s.ok()) return s;

    auto impl = std::make_shared<Impl>();
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(id_to_text.size()); ++i) {
        impl->text_to_id[id_to_text[static_cast<std::size_t>(i)]] = i;
    }
    impl->merges = std::move(merges);

    out->id_to_text_ = std::move(id_to_text);
    out->vocab_size_ = static_cast<std::int32_t>(out->id_to_text_.size());
    out->impl_ = std::move(impl);
    return OkStatus();
}

Status Tokenizer::Encode(const std::string & text, std::vector<std::int32_t> * token_ids) const {
    if (!is_loaded()) {
        return Status(StatusCode::kFailedPrecondition, "tokenizer not loaded");
    }
    if (!token_ids) {
        return Status(StatusCode::kInvalidArgument, "token_ids must not be null");
    }
    return EncodeUtf8(*this, text, token_ids);
}

Status Tokenizer::Decode(const std::vector<std::int32_t> & token_ids, std::string * text) const {
    if (!is_loaded()) {
        return Status(StatusCode::kFailedPrecondition, "tokenizer not loaded");
    }
    if (!text) {
        return Status(StatusCode::kInvalidArgument, "text must not be null");
    }
    return DecodeIds(*this, token_ids, text);
}

Status Tokenizer::DecodeSingle(std::int32_t token_id, std::string * piece) const {
    if (!is_loaded()) {
        return Status(StatusCode::kFailedPrecondition, "tokenizer not loaded");
    }
    if (!piece) {
        return Status(StatusCode::kInvalidArgument, "piece must not be null");
    }
    if (token_id < 0 || token_id >= vocab_size_) {
        return Status(StatusCode::kOutOfRange, "token_id out of range");
    }
    *piece = id_to_text_[static_cast<std::size_t>(token_id)];
    return OkStatus();
}

// --- Free functions ---

Status LoadVocabJson(const std::string & path, std::vector<std::string> * id_to_text) {
    if (!id_to_text) {
        return Status(StatusCode::kInvalidArgument, "id_to_text must not be null");
    }
    std::ifstream input(path);
    if (!input) {
        return Status(StatusCode::kNotFound, "failed to open vocab file: " + path);
    }
    // Minimal JSON object parser: expects {"token": id, ...}
    const std::string json_text((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());

    // Find max id to size the vector
    std::regex entry_pattern(R"("([^"\\]|\\.)*"\s*:\s*(\d+))");
    std::int32_t max_id = -1;
    std::vector<std::pair<std::string, std::int32_t>> entries;

    for (std::sregex_iterator it(json_text.begin(), json_text.end(), entry_pattern), end;
         it != end; ++it) {
        const std::string token_raw = (*it)[0].str();
        // Extract the token string: everything between the first pair of quotes
        const auto first_quote = token_raw.find('"');
        auto second_quote = token_raw.find('"', first_quote + 1);
        // Handle escaped quotes
        while (second_quote != std::string::npos && second_quote > 0 &&
               token_raw[second_quote - 1] == '\\') {
            second_quote = token_raw.find('"', second_quote + 1);
        }
        if (first_quote == std::string::npos || second_quote == std::string::npos) continue;
        const std::string token_str = token_raw.substr(first_quote + 1, second_quote - first_quote - 1);

        // Extract the id
        const auto colon_pos = token_raw.find(':', second_quote);
        if (colon_pos == std::string::npos) continue;
        const std::string id_str = token_raw.substr(colon_pos + 1);
        std::int32_t id = 0;
        try {
            id = static_cast<std::int32_t>(std::stol(id_str));
        } catch (...) {
            continue;
        }
        entries.emplace_back(token_str, id);
        if (id > max_id) max_id = id;
    }

    if (max_id < 0) {
        return Status(StatusCode::kInvalidArgument, "no valid entries in vocab file");
    }
    id_to_text->resize(static_cast<std::size_t>(max_id + 1));
    for (const auto & [tok, id] : entries) {
        (*id_to_text)[static_cast<std::size_t>(id)] = tok;
    }
    return OkStatus();
}

Status LoadMergesTxt(const std::string & path,
                     std::vector<std::pair<std::string, std::string>> * merges) {
    if (!merges) {
        return Status(StatusCode::kInvalidArgument, "merges must not be null");
    }
    std::ifstream input(path);
    if (!input) {
        return Status(StatusCode::kNotFound, "failed to open merges file: " + path);
    }
    merges->clear();
    std::string line;
    while (std::getline(input, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        // Each line is "token1 token2"
        const auto space = line.find(' ');
        if (space == std::string::npos || space == 0 || space == line.size() - 1) continue;
        merges->emplace_back(line.substr(0, space), line.substr(space + 1));
    }
    if (merges->empty()) {
        return Status(StatusCode::kInvalidArgument, "no valid merge entries");
    }
    return OkStatus();
}

Status EncodeUtf8(const Tokenizer & tokenizer, const std::string & text,
                  std::vector<std::int32_t> * ids) {
    if (!ids) {
        return Status(StatusCode::kInvalidArgument, "ids must not be null");
    }
    if (!tokenizer.is_loaded()) {
        return Status(StatusCode::kFailedPrecondition, "tokenizer not loaded");
    }
    ids->clear();

    // Byte-level BPE: start with individual byte tokens
    // Each byte maps to a token in the vocabulary
    // Then iteratively apply merges in priority order
    // Simplified implementation: try to find full text or character sequences
    for (std::size_t i = 0; i < text.size();) {
        bool found = false;
        // Try longest match first (greedy)
        for (std::size_t len = std::min(text.size() - i, std::size_t(64)); len > 0; --len) {
            std::string piece;
            try {
                tokenizer.DecodeSingle(0, &piece);  // validate loaded
            } catch (...) {
                break;
            }

            // Try to find this substring in vocab
            // This is a simplified approach - real BPE uses merge priorities
            std::string sub = text.substr(i, len);
            std::int32_t token_id = -1;
            for (std::int32_t t = 0; t < tokenizer.vocab_size(); ++t) {
                std::string tok_text;
                if (tokenizer.DecodeSingle(t, &tok_text).ok() && tok_text == sub) {
                    token_id = t;
                    break;
                }
            }
            if (token_id >= 0) {
                ids->push_back(token_id);
                i += len;
                found = true;
                break;
            }
        }
        if (!found) {
            // Fall back: single byte as unknown token (id 0)
            ids->push_back(0);
            ++i;
        }
    }
    return OkStatus();
}

Status DecodeIds(const Tokenizer & tokenizer, const std::vector<std::int32_t> & ids,
                 std::string * text) {
    if (!text) {
        return Status(StatusCode::kInvalidArgument, "text must not be null");
    }
    if (!tokenizer.is_loaded()) {
        return Status(StatusCode::kFailedPrecondition, "tokenizer not loaded");
    }
    text->clear();
    for (const auto id : ids) {
        std::string piece;
        Status s = tokenizer.DecodeSingle(id, &piece);
        if (!s.ok()) return s;
        text->append(piece);
    }
    return OkStatus();
}

}  // namespace qasr
