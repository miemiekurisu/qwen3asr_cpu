#include "qasr/base/json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace qasr {

namespace {

const Json kNullSentinel;

void EscapeJsonString(const std::string & input, std::string & output) {
    output.push_back('"');
    for (const unsigned char ch : input) {
        switch (ch) {
            case '"':  output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(ch));
                    output += buf;
                } else {
                    output.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    output.push_back('"');
}

class Parser {
public:
    explicit Parser(const std::string & text)
        : data_(text.data()), end_(text.data() + text.size()), pos_(text.data()) {}

    Json Run() {
        SkipWhitespace();
        Json result = ParseValue();
        if (result.is_discarded()) return result;
        SkipWhitespace();
        if (pos_ != end_) return MakeDiscarded();
        return result;
    }

private:
    const char * data_;
    const char * end_;
    const char * pos_;

    static Json MakeDiscarded() {
        Json j;
        // Access private type_ via a friend-free hack: parse a sentinel.
        // Instead, use a static helper via the public API to create discarded.
        // We set the type by constructing then assigning.  The class exposes
        // no public way to set kDiscarded, so we use a helper struct.
        struct Hack : Json { void SetDiscarded() { type_ = Type::kDiscarded; } };
        Hack h;
        h.SetDiscarded();
        return h;
    }

    bool AtEnd() const { return pos_ >= end_; }

    char Peek() const { return AtEnd() ? '\0' : *pos_; }

    char Advance() { return AtEnd() ? '\0' : *pos_++; }

    void SkipWhitespace() {
        while (!AtEnd() && (*pos_ == ' ' || *pos_ == '\t' || *pos_ == '\n' || *pos_ == '\r')) {
            ++pos_;
        }
    }

    bool Match(char expected) {
        SkipWhitespace();
        if (AtEnd() || *pos_ != expected) return false;
        ++pos_;
        return true;
    }

    Json ParseValue() {
        SkipWhitespace();
        if (AtEnd()) return MakeDiscarded();
        const char ch = Peek();
        switch (ch) {
            case '"': return ParseString();
            case '{': return ParseObject();
            case '[': return ParseArray();
            case 't': case 'f': return ParseBool();
            case 'n': return ParseNull();
            default:
                if (ch == '-' || (ch >= '0' && ch <= '9')) return ParseNumber();
                return MakeDiscarded();
        }
    }

    Json ParseNull() {
        if (end_ - pos_ >= 4 && std::memcmp(pos_, "null", 4) == 0) {
            pos_ += 4;
            return Json(nullptr);
        }
        return MakeDiscarded();
    }

    Json ParseBool() {
        if (end_ - pos_ >= 4 && std::memcmp(pos_, "true", 4) == 0) {
            pos_ += 4;
            return Json(true);
        }
        if (end_ - pos_ >= 5 && std::memcmp(pos_, "false", 5) == 0) {
            pos_ += 5;
            return Json(false);
        }
        return MakeDiscarded();
    }

    Json ParseString() {
        std::string result;
        if (!ParseStringInto(result)) return MakeDiscarded();
        return Json(std::move(result));
    }

    bool ParseStringInto(std::string & out) {
        if (Advance() != '"') return false;
        while (!AtEnd()) {
            const char ch = Advance();
            if (ch == '"') return true;
            if (ch == '\\') {
                if (AtEnd()) return false;
                const char esc = Advance();
                switch (esc) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        std::uint32_t cp = 0;
                        if (!ParseHex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (Advance() != '\\' || Advance() != 'u') return false;
                            std::uint32_t low = 0;
                            if (!ParseHex4(low)) return false;
                            if (low < 0xDC00 || low > 0xDFFF) return false;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        }
                        AppendUtf8(out, cp);
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(ch);
            }
        }
        return false;
    }

    bool ParseHex4(std::uint32_t & value) {
        value = 0;
        for (int i = 0; i < 4; ++i) {
            if (AtEnd()) return false;
            const char ch = Advance();
            value <<= 4;
            if (ch >= '0' && ch <= '9') value |= static_cast<std::uint32_t>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') value |= static_cast<std::uint32_t>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') value |= static_cast<std::uint32_t>(ch - 'A' + 10);
            else return false;
        }
        return true;
    }

    static void AppendUtf8(std::string & out, std::uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x110000) {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    Json ParseNumber() {
        const char * start = pos_;
        bool is_float = false;
        if (Peek() == '-') ++pos_;
        if (AtEnd() || (*pos_ < '0' || *pos_ > '9')) return MakeDiscarded();
        if (*pos_ == '0') {
            ++pos_;
        } else {
            while (!AtEnd() && *pos_ >= '0' && *pos_ <= '9') ++pos_;
        }
        if (!AtEnd() && *pos_ == '.') {
            is_float = true;
            ++pos_;
            if (AtEnd() || *pos_ < '0' || *pos_ > '9') return MakeDiscarded();
            while (!AtEnd() && *pos_ >= '0' && *pos_ <= '9') ++pos_;
        }
        if (!AtEnd() && (*pos_ == 'e' || *pos_ == 'E')) {
            is_float = true;
            ++pos_;
            if (!AtEnd() && (*pos_ == '+' || *pos_ == '-')) ++pos_;
            if (AtEnd() || *pos_ < '0' || *pos_ > '9') return MakeDiscarded();
            while (!AtEnd() && *pos_ >= '0' && *pos_ <= '9') ++pos_;
        }
        const std::string tok(start, static_cast<std::size_t>(pos_ - start));
        if (is_float) {
            char * endp = nullptr;
            const double val = std::strtod(tok.c_str(), &endp);
            if (endp != tok.c_str() + tok.size()) return MakeDiscarded();
            return Json(val);
        }
        char * endp = nullptr;
        const long long val = std::strtoll(tok.c_str(), &endp, 10);
        if (endp != tok.c_str() + tok.size()) return MakeDiscarded();
        return Json(static_cast<std::int64_t>(val));
    }

    Json ParseObject() {
        if (Advance() != '{') return MakeDiscarded();
        Json obj = Json::object();
        SkipWhitespace();
        if (Match('}')) return obj;
        while (true) {
            SkipWhitespace();
            std::string key;
            if (!ParseStringInto(key)) return MakeDiscarded();
            if (!Match(':')) return MakeDiscarded();
            Json val = ParseValue();
            if (val.is_discarded()) return MakeDiscarded();
            obj[key] = std::move(val);
            SkipWhitespace();
            if (Match('}')) return obj;
            if (!Match(',')) return MakeDiscarded();
        }
    }

    Json ParseArray() {
        if (Advance() != '[') return MakeDiscarded();
        Json arr = Json::array();
        SkipWhitespace();
        if (Match(']')) return arr;
        while (true) {
            Json val = ParseValue();
            if (val.is_discarded()) return MakeDiscarded();
            arr.push_back(std::move(val));
            SkipWhitespace();
            if (Match(']')) return arr;
            if (!Match(',')) return MakeDiscarded();
        }
    }
};

}  // namespace

bool Json::empty() const noexcept {
    switch (type_) {
        case Type::kArray:  return array_.empty();
        case Type::kObject: return object_.empty();
        case Type::kString: return string_value_.empty();
        case Type::kNull:   return true;
        default:            return false;
    }
}

std::size_t Json::size() const noexcept {
    switch (type_) {
        case Type::kArray:  return array_.size();
        case Type::kObject: return object_.size();
        default:            return 0;
    }
}

bool Json::contains(const std::string & key) const {
    if (type_ != Type::kObject) return false;
    for (const auto & entry : object_) {
        if (entry.first == key) return true;
    }
    return false;
}

Json & Json::operator[](const std::string & key) {
    if (type_ == Type::kNull) {
        type_ = Type::kObject;
    }
    if (type_ != Type::kObject) {
        static Json discard;
        discard = Json();
        return discard;
    }
    for (auto & entry : object_) {
        if (entry.first == key) return entry.second;
    }
    object_.emplace_back(key, Json());
    return object_.back().second;
}

const Json & Json::operator[](const std::string & key) const {
    if (type_ != Type::kObject) return kNullSentinel;
    for (const auto & entry : object_) {
        if (entry.first == key) return entry.second;
    }
    return kNullSentinel;
}

void Json::push_back(Json value) {
    if (type_ == Type::kNull) {
        type_ = Type::kArray;
    }
    if (type_ == Type::kArray) {
        array_.push_back(std::move(value));
    }
}

Json Json::array() {
    Json j;
    j.type_ = Type::kArray;
    return j;
}

Json Json::array(std::initializer_list<Json> values) {
    Json j;
    j.type_ = Type::kArray;
    j.array_.assign(values.begin(), values.end());
    return j;
}

Json Json::object() {
    Json j;
    j.type_ = Type::kObject;
    return j;
}

Json Json::object(std::initializer_list<ObjectEntry> entries) {
    Json j;
    j.type_ = Type::kObject;
    j.object_.assign(entries.begin(), entries.end());
    return j;
}

Json Json::parse(const std::string & text) {
    if (text.empty()) {
        Json j;
        j.type_ = Type::kDiscarded;
        return j;
    }
    Parser parser(text);
    return parser.Run();
}

std::string Json::dump() const {
    std::string out;
    switch (type_) {
        case Type::kNull:
        case Type::kDiscarded:
            out += "null";
            break;
        case Type::kBool:
            out += bool_value_ ? "true" : "false";
            break;
        case Type::kInteger: {
            char buf[32];
            const int len = std::snprintf(buf, sizeof(buf), "%lld",
                static_cast<long long>(int_value_));
            out.append(buf, static_cast<std::size_t>(len));
            break;
        }
        case Type::kFloat: {
            if (std::isnan(float_value_) || std::isinf(float_value_)) {
                out += "null";
            } else {
                char buf[64];
                const int len = std::snprintf(buf, sizeof(buf), "%g", float_value_);
                out.append(buf, static_cast<std::size_t>(len));
            }
            break;
        }
        case Type::kString:
            EscapeJsonString(string_value_, out);
            break;
        case Type::kArray:
            out.push_back('[');
            for (std::size_t i = 0; i < array_.size(); ++i) {
                if (i > 0) out.push_back(',');
                out += array_[i].dump();
            }
            out.push_back(']');
            break;
        case Type::kObject:
            out.push_back('{');
            for (std::size_t i = 0; i < object_.size(); ++i) {
                if (i > 0) out.push_back(',');
                EscapeJsonString(object_[i].first, out);
                out.push_back(':');
                out += object_[i].second.dump();
            }
            out.push_back('}');
            break;
    }
    return out;
}

Json::Array::iterator Json::begin() noexcept { return array_.begin(); }
Json::Array::iterator Json::end() noexcept { return array_.end(); }
Json::Array::const_iterator Json::begin() const noexcept { return array_.begin(); }
Json::Array::const_iterator Json::end() const noexcept { return array_.end(); }

}  // namespace qasr
