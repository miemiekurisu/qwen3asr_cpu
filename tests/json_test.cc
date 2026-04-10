#include "test_registry.h"

#include "qasr/base/json.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using qasr::Json;

// ==================== Construction ====================

QASR_TEST(Json_DefaultIsNull) {
    Json j;
    QASR_EXPECT(j.is_null());
    QASR_EXPECT(!j.is_object());
    QASR_EXPECT(!j.is_array());
    QASR_EXPECT(!j.is_string());
}

QASR_TEST(Json_NullptrIsNull) {
    Json j(nullptr);
    QASR_EXPECT(j.is_null());
}

QASR_TEST(Json_BoolTrue) {
    Json j(true);
    QASR_EXPECT(j.is_bool());
    QASR_EXPECT_EQ(j.get<bool>(), true);
}

QASR_TEST(Json_BoolFalse) {
    Json j(false);
    QASR_EXPECT(j.is_bool());
    QASR_EXPECT_EQ(j.get<bool>(), false);
}

QASR_TEST(Json_IntZero) {
    Json j(0);
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<int>(), 0);
}

QASR_TEST(Json_IntNegative) {
    Json j(-42);
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<std::int64_t>(), -42);
}

QASR_TEST(Json_Int64Max) {
    Json j(std::numeric_limits<std::int64_t>::max());
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<std::int64_t>(), std::numeric_limits<std::int64_t>::max());
}

QASR_TEST(Json_UnsignedInt) {
    const std::size_t val = 123456;
    Json j(val);
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<std::int64_t>(), 123456);
}

QASR_TEST(Json_Double) {
    Json j(3.14);
    QASR_EXPECT(j.is_number_float());
    QASR_EXPECT(std::abs(j.get<double>() - 3.14) < 1e-10);
}

QASR_TEST(Json_StringLiteral) {
    Json j("hello");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("hello"));
}

QASR_TEST(Json_StdString) {
    std::string s = "world";
    Json j(s);
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("world"));
}

QASR_TEST(Json_EmptyString) {
    Json j("");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string(""));
}

// ==================== Factories ====================

QASR_TEST(Json_ArrayEmpty) {
    Json j = Json::array();
    QASR_EXPECT(j.is_array());
    QASR_EXPECT(j.empty());
    QASR_EXPECT_EQ(j.size(), std::size_t(0));
}

QASR_TEST(Json_ArrayWithElements) {
    Json j = Json::array({Json(1), Json(2), Json(3)});
    QASR_EXPECT(j.is_array());
    QASR_EXPECT_EQ(j.size(), std::size_t(3));
}

QASR_TEST(Json_ObjectEmpty) {
    Json j = Json::object();
    QASR_EXPECT(j.is_object());
    QASR_EXPECT(j.empty());
}

QASR_TEST(Json_ObjectWithEntries) {
    Json j = Json::object({{"name", "test"}, {"count", 42}});
    QASR_EXPECT(j.is_object());
    QASR_EXPECT_EQ(j.size(), std::size_t(2));
    QASR_EXPECT(j.contains("name"));
    QASR_EXPECT(j.contains("count"));
}

// ==================== Object access ====================

QASR_TEST(Json_SubscriptCreatesField) {
    Json j;
    j["key"] = "value";
    QASR_EXPECT(j.is_object());
    QASR_EXPECT(j.contains("key"));
    QASR_EXPECT_EQ(j["key"].get<std::string>(), std::string("value"));
}

QASR_TEST(Json_SubscriptMultipleFields) {
    Json j;
    j["a"] = 1;
    j["b"] = 2;
    j["c"] = 3;
    QASR_EXPECT_EQ(j.size(), std::size_t(3));
    QASR_EXPECT_EQ(j["a"].get<int>(), 1);
    QASR_EXPECT_EQ(j["b"].get<int>(), 2);
    QASR_EXPECT_EQ(j["c"].get<int>(), 3);
}

QASR_TEST(Json_SubscriptOverwrite) {
    Json j;
    j["key"] = 1;
    j["key"] = 2;
    QASR_EXPECT_EQ(j.size(), std::size_t(1));
    QASR_EXPECT_EQ(j["key"].get<int>(), 2);
}

QASR_TEST(Json_ConstSubscriptMissing) {
    const Json j = Json::object({{"exists", true}});
    const Json & missing = j["nonexistent"];
    QASR_EXPECT(missing.is_null());
}

QASR_TEST(Json_Contains) {
    Json j = Json::object({{"key", "val"}});
    QASR_EXPECT(j.contains("key"));
    QASR_EXPECT(!j.contains("other"));
}

QASR_TEST(Json_ContainsOnNonObject) {
    Json j(42);
    QASR_EXPECT(!j.contains("key"));
}

// ==================== value() with default ====================

QASR_TEST(Json_ValueStringExists) {
    Json j = Json::object({{"model", "qwen"}});
    QASR_EXPECT_EQ(j.value("model", std::string()), std::string("qwen"));
}

QASR_TEST(Json_ValueStringMissing) {
    Json j = Json::object();
    QASR_EXPECT_EQ(j.value("model", std::string("default")), std::string("default"));
}

QASR_TEST(Json_ValueBoolExists) {
    Json j = Json::object({{"stream", true}});
    QASR_EXPECT_EQ(j.value("stream", false), true);
}

QASR_TEST(Json_ValueBoolMissing) {
    Json j = Json::object();
    QASR_EXPECT_EQ(j.value("stream", false), false);
}

QASR_TEST(Json_ValueIntExists) {
    Json j = Json::object({{"count", 42}});
    QASR_EXPECT_EQ(j.value("count", 0), 42);
}

QASR_TEST(Json_ValueOnNonObject) {
    Json j(42);
    QASR_EXPECT_EQ(j.value("key", std::string("default")), std::string("default"));
}

// ==================== get<T>() ====================

QASR_TEST(Json_GetString) {
    Json j("hello");
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("hello"));
}

QASR_TEST(Json_GetInt) {
    Json j(99);
    QASR_EXPECT_EQ(j.get<int>(), 99);
}

QASR_TEST(Json_GetDouble) {
    Json j(2.5);
    QASR_EXPECT(std::abs(j.get<double>() - 2.5) < 1e-10);
}

QASR_TEST(Json_GetDoubleFromInt) {
    Json j(10);
    QASR_EXPECT(std::abs(j.get<double>() - 10.0) < 1e-10);
}

// ==================== push_back ====================

QASR_TEST(Json_PushBackOnNull) {
    Json j;
    j.push_back(Json(1));
    j.push_back(Json(2));
    QASR_EXPECT(j.is_array());
    QASR_EXPECT_EQ(j.size(), std::size_t(2));
}

QASR_TEST(Json_PushBackOnArray) {
    Json j = Json::array();
    j.push_back(Json("a"));
    j.push_back(Json("b"));
    QASR_EXPECT_EQ(j.size(), std::size_t(2));
}

// ==================== Iteration ====================

QASR_TEST(Json_IterateArray) {
    Json j = Json::array({Json(1), Json(2), Json(3)});
    int sum = 0;
    for (const Json & item : j) {
        sum += item.get<int>();
    }
    QASR_EXPECT_EQ(sum, 6);
}

QASR_TEST(Json_IterateEmptyArray) {
    Json j = Json::array();
    int count = 0;
    for (const Json & item : j) {
        (void)item;
        ++count;
    }
    QASR_EXPECT_EQ(count, 0);
}

// ==================== Parse ====================

QASR_TEST(Json_ParseNull) {
    Json j = Json::parse("null");
    QASR_EXPECT(j.is_null());
}

QASR_TEST(Json_ParseTrue) {
    Json j = Json::parse("true");
    QASR_EXPECT(j.is_bool());
    QASR_EXPECT_EQ(j.get<bool>(), true);
}

QASR_TEST(Json_ParseFalse) {
    Json j = Json::parse("false");
    QASR_EXPECT(j.is_bool());
    QASR_EXPECT_EQ(j.get<bool>(), false);
}

QASR_TEST(Json_ParseInteger) {
    Json j = Json::parse("42");
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<int>(), 42);
}

QASR_TEST(Json_ParseNegativeInteger) {
    Json j = Json::parse("-100");
    QASR_EXPECT(j.is_number_integer());
    QASR_EXPECT_EQ(j.get<int>(), -100);
}

QASR_TEST(Json_ParseFloat) {
    Json j = Json::parse("3.14");
    QASR_EXPECT(j.is_number_float());
    QASR_EXPECT(std::abs(j.get<double>() - 3.14) < 1e-10);
}

QASR_TEST(Json_ParseExponent) {
    Json j = Json::parse("1e3");
    QASR_EXPECT(j.is_number_float());
    QASR_EXPECT(std::abs(j.get<double>() - 1000.0) < 1e-10);
}

QASR_TEST(Json_ParseString) {
    Json j = Json::parse("\"hello\"");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("hello"));
}

QASR_TEST(Json_ParseStringEscapes) {
    Json j = Json::parse("\"line1\\nline2\\t\\\"quoted\\\"\"");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("line1\nline2\t\"quoted\""));
}

QASR_TEST(Json_ParseUnicode) {
    Json j = Json::parse("\"\\u0041\"");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>(), std::string("A"));
}

QASR_TEST(Json_ParseSurrogatePair) {
    // U+1F600 GRINNING FACE = \uD83D\uDE00
    Json j = Json::parse("\"\\uD83D\\uDE00\"");
    QASR_EXPECT(j.is_string());
    QASR_EXPECT_EQ(j.get<std::string>().size(), std::size_t(4));
}

QASR_TEST(Json_ParseEmptyObject) {
    Json j = Json::parse("{}");
    QASR_EXPECT(j.is_object());
    QASR_EXPECT(j.empty());
}

QASR_TEST(Json_ParseObject) {
    Json j = Json::parse("{\"key\":\"value\",\"num\":42}");
    QASR_EXPECT(j.is_object());
    QASR_EXPECT_EQ(j["key"].get<std::string>(), std::string("value"));
    QASR_EXPECT_EQ(j["num"].get<int>(), 42);
}

QASR_TEST(Json_ParseEmptyArray) {
    Json j = Json::parse("[]");
    QASR_EXPECT(j.is_array());
    QASR_EXPECT(j.empty());
}

QASR_TEST(Json_ParseArray) {
    Json j = Json::parse("[1,2,3]");
    QASR_EXPECT(j.is_array());
    QASR_EXPECT_EQ(j.size(), std::size_t(3));
}

QASR_TEST(Json_ParseNested) {
    Json j = Json::parse("{\"arr\":[1,{\"inner\":true}],\"obj\":{\"k\":null}}");
    QASR_EXPECT(j.is_object());
    QASR_EXPECT(j["arr"].is_array());
    QASR_EXPECT_EQ(j["arr"].size(), std::size_t(2));
    QASR_EXPECT(j["obj"]["k"].is_null());
}

QASR_TEST(Json_ParseWhitespace) {
    Json j = Json::parse("  { \"key\" : \"value\" }  ");
    QASR_EXPECT(j.is_object());
    QASR_EXPECT_EQ(j["key"].get<std::string>(), std::string("value"));
}

// ==================== Parse errors ====================

QASR_TEST(Json_ParseEmpty) {
    Json j = Json::parse("");
    QASR_EXPECT(j.is_discarded());
}

QASR_TEST(Json_ParseInvalid) {
    Json j = Json::parse("xyz");
    QASR_EXPECT(j.is_discarded());
}

QASR_TEST(Json_ParseTrailingGarbage) {
    Json j = Json::parse("42 abc");
    QASR_EXPECT(j.is_discarded());
}

QASR_TEST(Json_ParseUnclosedString) {
    Json j = Json::parse("\"unterminated");
    QASR_EXPECT(j.is_discarded());
}

QASR_TEST(Json_ParseUnclosedObject) {
    Json j = Json::parse("{\"key\": 1");
    QASR_EXPECT(j.is_discarded());
}

QASR_TEST(Json_ParseUnclosedArray) {
    Json j = Json::parse("[1, 2");
    QASR_EXPECT(j.is_discarded());
}

// ==================== Dump ====================

QASR_TEST(Json_DumpNull) {
    QASR_EXPECT_EQ(Json(nullptr).dump(), std::string("null"));
}

QASR_TEST(Json_DumpBoolTrue) {
    QASR_EXPECT_EQ(Json(true).dump(), std::string("true"));
}

QASR_TEST(Json_DumpBoolFalse) {
    QASR_EXPECT_EQ(Json(false).dump(), std::string("false"));
}

QASR_TEST(Json_DumpInt) {
    QASR_EXPECT_EQ(Json(42).dump(), std::string("42"));
}

QASR_TEST(Json_DumpNegativeInt) {
    QASR_EXPECT_EQ(Json(-7).dump(), std::string("-7"));
}

QASR_TEST(Json_DumpFloat) {
    Json j(3.5);
    QASR_EXPECT_EQ(j.dump(), std::string("3.5"));
}

QASR_TEST(Json_DumpString) {
    QASR_EXPECT_EQ(Json("hello").dump(), std::string("\"hello\""));
}

QASR_TEST(Json_DumpStringEscapes) {
    Json j("line1\nline2\t\"quoted\"");
    std::string expected = "\"line1\\nline2\\t\\\"quoted\\\"\"";
    QASR_EXPECT_EQ(j.dump(), expected);
}

QASR_TEST(Json_DumpEmptyArray) {
    QASR_EXPECT_EQ(Json::array().dump(), std::string("[]"));
}

QASR_TEST(Json_DumpArray) {
    Json j = Json::array({Json(1), Json("two"), Json(true)});
    QASR_EXPECT_EQ(j.dump(), std::string("[1,\"two\",true]"));
}

QASR_TEST(Json_DumpEmptyObject) {
    QASR_EXPECT_EQ(Json::object().dump(), std::string("{}"));
}

QASR_TEST(Json_DumpObject) {
    Json j = Json::object({{"a", 1}, {"b", "two"}});
    QASR_EXPECT_EQ(j.dump(), std::string("{\"a\":1,\"b\":\"two\"}"));
}

QASR_TEST(Json_DumpPreservesInsertionOrder) {
    Json j;
    j["z"] = 1;
    j["a"] = 2;
    j["m"] = 3;
    QASR_EXPECT_EQ(j.dump(), std::string("{\"z\":1,\"a\":2,\"m\":3}"));
}

// ==================== Round-trip ====================

QASR_TEST(Json_RoundTripComplex) {
    const std::string input = "{\"text\":\"hello world\",\"tokens\":42,\"segments\":[{\"start\":0,\"end\":1.5}]}";
    Json j = Json::parse(input);
    QASR_EXPECT(!j.is_discarded());
    std::string output = j.dump();
    Json j2 = Json::parse(output);
    QASR_EXPECT(!j2.is_discarded());
    QASR_EXPECT_EQ(j2["text"].get<std::string>(), std::string("hello world"));
    QASR_EXPECT_EQ(j2["tokens"].get<int>(), 42);
    QASR_EXPECT(j2["segments"].is_array());
    QASR_EXPECT_EQ(j2["segments"].size(), std::size_t(1));
}

// ==================== empty() and size() ====================

QASR_TEST(Json_EmptyNull) {
    Json j;
    QASR_EXPECT(j.empty());
}

QASR_TEST(Json_EmptyStringValue) {
    Json j("");
    QASR_EXPECT(j.empty());
}

QASR_TEST(Json_NonEmptyString) {
    Json j("x");
    QASR_EXPECT(!j.empty());
}

QASR_TEST(Json_SizeOfArray) {
    Json j = Json::array({Json(1), Json(2)});
    QASR_EXPECT_EQ(j.size(), std::size_t(2));
}

QASR_TEST(Json_SizeOfObject) {
    Json j = Json::object({{"a", 1}});
    QASR_EXPECT_EQ(j.size(), std::size_t(1));
}

// ==================== Type checks ====================

QASR_TEST(Json_IsNumber) {
    QASR_EXPECT(Json(42).is_number());
    QASR_EXPECT(Json(3.14).is_number());
    QASR_EXPECT(!Json("42").is_number());
    QASR_EXPECT(!Json(true).is_number());
}

QASR_TEST(Json_Discarded) {
    Json j = Json::parse("invalid");
    QASR_EXPECT(j.is_discarded());
    QASR_EXPECT_EQ(j.dump(), std::string("null"));
}
