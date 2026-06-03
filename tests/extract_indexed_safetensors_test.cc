#include "tests/test_registry.h"

#include <set>
#include <string>
#include <vector>

#include "qasr/runtime/model_bridge_internal.h"

namespace {

QASR_TEST(ExtractIndexedSafetensorsReturnsEmptyForEmptyInput) {
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors("");
    QASR_EXPECT(result.empty());
}

QASR_TEST(ExtractIndexedSafetensorsFindsSingleShard) {
    const std::string json =
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors"}})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
    QASR_EXPECT(result.count("model-00001-of-00002.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsDeduplicates) {
    // Two keys referencing the same shard should collapse to one entry.
    const std::string json =
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors",)"
        R"("b":"model-00001-of-00002.safetensors"}})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
}

QASR_TEST(ExtractIndexedSafetensorsFindsMultipleShards) {
    const std::string json =
        R"({"weight_map":{)"
        R"("a":"model-00001-of-00003.safetensors",)"
        R"("b":"model-00002-of-00003.safetensors",)"
        R"("c":"model-00003-of-00003.safetensors"}})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{3});
    QASR_EXPECT(result.count("model-00001-of-00003.safetensors") == 1);
    QASR_EXPECT(result.count("model-00002-of-00003.safetensors") == 1);
    QASR_EXPECT(result.count("model-00003-of-00003.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsIgnoresNonShardStrings) {
    // A `model-` prefix that does not end with `.safetensors` must not
    // be returned.  A `.safetensors` suffix without the `model-` prefix
    // must also be skipped.
    const std::string json =
        R"({"comment":"see model-README for details",)"
        R"("other":"random.safetensors",)"
        R"("good":"model-00001-of-00002.safetensors"})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
    QASR_EXPECT(result.count("model-00001-of-00002.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsSkipsBareModelPrefix) {
    // `model-` with no body must not match (regex required `[^"]+`).
    const std::string json =
        R"({"a":"model-","b":"model-00001-of-00002.safetensors"})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
    QASR_EXPECT(result.count("model-00001-of-00002.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsSkipsBareSuffix) {
    // `.safetensors` with no `model-` prefix and no body must not match.
    const std::string json =
        R"({"a":".safetensors","b":"model-00001-of-00002.safetensors"})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
    QASR_EXPECT(result.count("model-00001-of-00002.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsHandlesUnterminatedString) {
    // A `model-` prefix that never sees a closing quote must not loop
    // forever or yield a partial match.
    const std::string json = R"({"a":"model-00001-of-00002.safetensors)";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT(result.empty());
}

QASR_TEST(ExtractIndexedSafetensorsHandlesDottedBody) {
    // Shard bodies that themselves contain dots (rare but legal) should
    // still be captured intact.
    const std::string json =
        R"({"a":"model-foo.bar-00001-of-00002.safetensors"})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{1});
    QASR_EXPECT(result.count("model-foo.bar-00001-of-00002.safetensors") == 1);
}

QASR_TEST(ExtractIndexedSafetensorsRealisticIndex) {
    // A realistic-looking HuggingFace index.json with metadata,
    // a string containing `model-` in a non-shard context (the
    // `name` field of a configuration block), and a multi-shard
    // weight_map.
    const std::string json = R"({
  "metadata": {"total_size": 12345678901},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.weight": "model-00001-of-00003.safetensors",
    "model.layers.1.weight": "model-00002-of-00003.safetensors",
    "model.layers.50.weight": "model-00003-of-00003.safetensors",
    "model.norm.weight": "model-00003-of-00003.safetensors"
  }
})";
    const std::set<std::string> result = qasr::ExtractIndexedSafetensors(json);
    QASR_EXPECT_EQ(result.size(), std::size_t{3});
    QASR_EXPECT(result.count("model-00001-of-00003.safetensors") == 1);
    QASR_EXPECT(result.count("model-00002-of-00003.safetensors") == 1);
    QASR_EXPECT(result.count("model-00003-of-00003.safetensors") == 1);
}

}  // namespace
