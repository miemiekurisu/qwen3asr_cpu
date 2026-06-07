#include "qasr/model/tokenizer.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
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

    // Hand-written scan for `"<token>":<id>` entries.  Replaces the
    // previous std::regex-based implementation.  Walks the input in
    // O(n) time with a single pass; no regex state machine.
    //
    // Escape handling: a `"` is considered escaped if it is preceded
    // by an odd number of backslashes.  This matches JSON's own
    // backslash-escape rule and is more careful than the original
    // implementation, which only checked the single preceding
    // character.
    auto is_digit = [](unsigned char c) { return std::isdigit(c) != 0; };

    std::int32_t max_id = -1;
    std::vector<std::pair<std::string, std::int32_t>> entries;

    std::size_t pos = 0;
    while (pos < json_text.size()) {
        // Find the next opening quote.
        const std::size_t open = json_text.find('"', pos);
        if (open == std::string::npos) break;

        // Read characters until we hit the matching closing quote.
        // Track backslash runs to detect escaped quotes.
        std::size_t close = std::string::npos;
        std::size_t i = open + 1;
        std::size_t backslash_run = 0;
        while (i < json_text.size()) {
            const char c = json_text[i];
            if (c == '\\') {
                ++backslash_run;
                ++i;
                continue;
            }
            if (c == '"') {
                if ((backslash_run & 1U) == 0U) {
                    close = i;
                    break;
                }
                // Escaped quote — consume the backslash and the quote
                // as part of the token body.
                backslash_run = 0;
                ++i;
                continue;
            }
            backslash_run = 0;
            ++i;
        }
        if (close == std::string::npos) break;

        const std::string token_str = json_text.substr(open + 1, close - open - 1);
        std::size_t j = close + 1;

        // Skip optional whitespace before ':'.
        while (j < json_text.size() &&
               std::isspace(static_cast<unsigned char>(json_text[j]))) {
            ++j;
        }
        if (j >= json_text.size() || json_text[j] != ':') {
            pos = j + 1;
            continue;
        }
        ++j;

        // Skip optional whitespace before the id.
        while (j < json_text.size() &&
               std::isspace(static_cast<unsigned char>(json_text[j]))) {
            ++j;
        }
        if (j >= json_text.size() || !is_digit(static_cast<unsigned char>(json_text[j]))) {
            pos = j + 1;
            continue;
        }
        // Parse the id (decimal non-negative integer).
        std::int64_t id = 0;
        while (j < json_text.size() &&
               is_digit(static_cast<unsigned char>(json_text[j]))) {
            id = id * 10 + (json_text[j] - '0');
            if (id > 0x7FFFFFFFLL) {
                // Overflow int32_t — skip the entry.
                while (j < json_text.size() &&
                       is_digit(static_cast<unsigned char>(json_text[j]))) {
                    ++j;
                }
                id = -1;
                break;
            }
            ++j;
        }
        if (id < 0) {
            pos = j + 1;
            continue;
        }

        entries.emplace_back(token_str, static_cast<std::int32_t>(id));
        if (id > max_id) max_id = static_cast<std::int32_t>(id);
        pos = j + 1;
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
