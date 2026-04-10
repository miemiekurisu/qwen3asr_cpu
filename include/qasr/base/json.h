#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace qasr {

/// Minimal ordered JSON value type.
/// Pre: construction arguments match documented types.
/// Post: value is immutable once published; read access is thread-safe.
/// Thread-safe: read-only after construction.
class Json {
public:
    using ObjectEntry = std::pair<std::string, Json>;
    using Array = std::vector<Json>;
    using Object = std::vector<ObjectEntry>;

    enum class Type {
        kNull,
        kBool,
        kInteger,
        kFloat,
        kString,
        kArray,
        kObject,
        kDiscarded,
    };

    Json() noexcept = default;
    Json(std::nullptr_t) noexcept {}
    Json(bool value) noexcept : type_(Type::kBool), bool_value_(value) {}
    Json(const char * value) : type_(Type::kString), string_value_(value ? value : "") {}
    Json(std::string value) noexcept : type_(Type::kString), string_value_(std::move(value)) {}
    Json(std::string_view value) : type_(Type::kString), string_value_(value) {}

    template <typename T, std::enable_if_t<
        std::is_integral_v<T> && !std::is_same_v<std::decay_t<T>, bool>, int> = 0>
    Json(T value) noexcept : type_(Type::kInteger), int_value_(static_cast<std::int64_t>(value)) {}

    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    Json(T value) noexcept : type_(Type::kFloat), float_value_(static_cast<double>(value)) {}

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::kNull; }
    bool is_bool() const noexcept { return type_ == Type::kBool; }
    bool is_number_integer() const noexcept { return type_ == Type::kInteger; }
    bool is_number_float() const noexcept { return type_ == Type::kFloat; }
    bool is_number() const noexcept { return is_number_integer() || is_number_float(); }
    bool is_string() const noexcept { return type_ == Type::kString; }
    bool is_array() const noexcept { return type_ == Type::kArray; }
    bool is_object() const noexcept { return type_ == Type::kObject; }
    bool is_discarded() const noexcept { return type_ == Type::kDiscarded; }

    bool empty() const noexcept;
    std::size_t size() const noexcept;
    bool contains(const std::string & key) const;

    Json & operator[](const std::string & key);
    const Json & operator[](const std::string & key) const;

    void push_back(Json value);

    template <typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return string_value_;
        } else if constexpr (std::is_same_v<T, bool>) {
            return bool_value_;
        } else if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(int_value_);
        } else if constexpr (std::is_floating_point_v<T>) {
            return type_ == Type::kFloat ? static_cast<T>(float_value_)
                                         : static_cast<T>(int_value_);
        } else {
            static_assert(!std::is_same_v<T, T>, "unsupported get type");
        }
    }

    template <typename T>
    T value(const std::string & key, const T & default_val) const {
        if (type_ != Type::kObject) return default_val;
        for (const auto & entry : object_) {
            if (entry.first != key) continue;
            const Json & v = entry.second;
            if constexpr (std::is_same_v<T, std::string>) {
                return v.type_ == Type::kString ? v.string_value_ : default_val;
            } else if constexpr (std::is_same_v<T, bool>) {
                return v.type_ == Type::kBool ? v.bool_value_ : default_val;
            } else if constexpr (std::is_integral_v<T>) {
                return v.type_ == Type::kInteger ? static_cast<T>(v.int_value_) : default_val;
            } else if constexpr (std::is_floating_point_v<T>) {
                if (v.type_ == Type::kFloat) return static_cast<T>(v.float_value_);
                if (v.type_ == Type::kInteger) return static_cast<T>(v.int_value_);
                return default_val;
            } else {
                return default_val;
            }
        }
        return default_val;
    }

    /// Explicit factories.
    static Json array();
    static Json array(std::initializer_list<Json> values);
    static Json object();
    static Json object(std::initializer_list<ObjectEntry> entries);

    /// Parse JSON text.  Never throws; returns discarded on error.
    static Json parse(const std::string & text);

    /// Compact JSON serialization.
    std::string dump() const;

    /// Array iteration.
    Array::iterator begin() noexcept;
    Array::iterator end() noexcept;
    Array::const_iterator begin() const noexcept;
    Array::const_iterator end() const noexcept;

private:
    Type type_ = Type::kNull;
    bool bool_value_ = false;
    std::int64_t int_value_ = 0;
    double float_value_ = 0.0;
    std::string string_value_;
    Array array_;
    Object object_;
};

}  // namespace qasr
