#include "tests/test_registry.h"
#include "qasr/inference/streaming_policy.h"

// --- ValidateStreamPolicyConfig ---

QASR_TEST(StreamPolicyConfigDefaultValid) {
    qasr::StreamPolicyConfig config;
    qasr::Status s = qasr::ValidateStreamPolicyConfig(config);
    QASR_EXPECT(s.ok());
}

QASR_TEST(StreamPolicyConfigInvalidChunkSec) {
    qasr::StreamPolicyConfig config;
    config.chunk_sec = 0.0f;
    QASR_EXPECT(!qasr::ValidateStreamPolicyConfig(config).ok());
}

QASR_TEST(StreamPolicyConfigWindowTooSmall) {
    qasr::StreamPolicyConfig config;
    config.chunk_sec = 5.0f;
    config.window_sec = 2.0f;  // smaller than chunk
    QASR_EXPECT(!qasr::ValidateStreamPolicyConfig(config).ok());
}

QASR_TEST(StreamPolicyConfigNegativeRollback) {
    qasr::StreamPolicyConfig config;
    config.rollback_tokens = -1;
    QASR_EXPECT(!qasr::ValidateStreamPolicyConfig(config).ok());
}

QASR_TEST(StreamPolicyConfigZeroMaxNewTokens) {
    qasr::StreamPolicyConfig config;
    config.max_new_tokens = 0;
    QASR_EXPECT(!qasr::ValidateStreamPolicyConfig(config).ok());
}

// --- StreamChunkPlanner ---

QASR_TEST(StreamChunkPlannerShouldDecodeInitially) {
    qasr::StreamPolicyConfig config;
    config.chunk_sec = 2.0f;
    qasr::StreamChunkPlanner planner(config, 16000);
    // Need 2s * 16000 = 32000 samples before first decode
    QASR_EXPECT(!planner.ShouldDecode(16000));  // 1 second - not enough
    QASR_EXPECT(planner.ShouldDecode(32000));   // 2 seconds - enough
}

QASR_TEST(StreamChunkPlannerMarkDecoded) {
    qasr::StreamPolicyConfig config;
    config.chunk_sec = 1.0f;
    qasr::StreamChunkPlanner planner(config, 16000);
    QASR_EXPECT(planner.ShouldDecode(16000));
    planner.MarkDecoded(16000);
    QASR_EXPECT(!planner.ShouldDecode(20000));  // Not enough since last decode
    QASR_EXPECT(planner.ShouldDecode(32000));   // Another full chunk
}

QASR_TEST(StreamChunkPlannerSampleCounts) {
    qasr::StreamPolicyConfig config;
    config.chunk_sec = 2.0f;
    config.window_sec = 8.0f;
    qasr::StreamChunkPlanner planner(config, 16000);
    QASR_EXPECT_EQ(planner.chunk_samples(), int32_t(32000));
    QASR_EXPECT_EQ(planner.window_samples(), int32_t(128000));
}

// --- EncoderCache ---

QASR_TEST(EncoderCacheStoreAndHas) {
    qasr::EncoderCache cache;
    std::vector<float> data(256, 1.0f);
    cache.Store(0, data, 10);
    QASR_EXPECT(cache.Has(0));
    QASR_EXPECT(!cache.Has(1));
    QASR_EXPECT_EQ(cache.size(), std::size_t(1));
}

QASR_TEST(EncoderCacheReplaceExisting) {
    qasr::EncoderCache cache;
    std::vector<float> data1(256, 1.0f);
    std::vector<float> data2(256, 2.0f);
    cache.Store(0, data1, 10);
    cache.Store(0, data2, 20);
    QASR_EXPECT_EQ(cache.size(), std::size_t(1));  // No duplicates
}

QASR_TEST(EncoderCacheEvict) {
    qasr::EncoderCache cache;
    for (int i = 0; i < 5; ++i) {
        cache.Store(i, std::vector<float>(10, 0.0f), 1);
    }
    QASR_EXPECT_EQ(cache.size(), std::size_t(5));
    cache.Evict(3);  // Evict windows 0, 1, 2
    QASR_EXPECT_EQ(cache.size(), std::size_t(2));
    QASR_EXPECT(!cache.Has(0));
    QASR_EXPECT(!cache.Has(2));
    QASR_EXPECT(cache.Has(3));
    QASR_EXPECT(cache.Has(4));
}

// --- LongestCommonStablePrefix ---

QASR_TEST(LongestCommonPrefixIdentical) {
    std::size_t len = qasr::LongestCommonStablePrefix("hello", "hello");
    QASR_EXPECT_EQ(len, std::size_t(5));
}

QASR_TEST(LongestCommonPrefixPartial) {
    std::size_t len = qasr::LongestCommonStablePrefix("hello world", "hello there");
    QASR_EXPECT_EQ(len, std::size_t(6));  // "hello "
}

QASR_TEST(LongestCommonPrefixEmpty) {
    QASR_EXPECT_EQ(qasr::LongestCommonStablePrefix("", "hello"), std::size_t(0));
    QASR_EXPECT_EQ(qasr::LongestCommonStablePrefix("hello", ""), std::size_t(0));
    QASR_EXPECT_EQ(qasr::LongestCommonStablePrefix("", ""), std::size_t(0));
}

QASR_TEST(LongestCommonPrefixNoCommon) {
    QASR_EXPECT_EQ(qasr::LongestCommonStablePrefix("abc", "xyz"), std::size_t(0));
}

// --- DetectDegenerateTail ---

QASR_TEST(DegenerateTailRepeating) {
    QASR_EXPECT(qasr::DetectDegenerateTail("abcabcabc", 3));
}

QASR_TEST(DegenerateTailNotRepeating) {
    QASR_EXPECT(!qasr::DetectDegenerateTail("hello world", 5));
}

QASR_TEST(DegenerateTailTooShort) {
    QASR_EXPECT(!qasr::DetectDegenerateTail("ab", 5));
}

// --- CommitFrontier ---

QASR_TEST(CommitFrontierExtendStable) {
    std::string stable = "hello";
    std::string unstable;
    qasr::Status s = qasr::CommitFrontier("hello world", &stable, &unstable, 0);
    QASR_EXPECT(s.ok());
}

QASR_TEST(CommitFrontierNullOutput) {
    qasr::Status s = qasr::CommitFrontier("test", nullptr, nullptr, 0);
    QASR_EXPECT(!s.ok());
}

// --- ForceFreezeAgedSuffix ---

QASR_TEST(ForceFreezeMovesToStable) {
    std::string stable = "hello ";
    std::string unstable = "world";
    std::string frozen;
    qasr::Status s = qasr::ForceFreezeAgedSuffix(&stable, &unstable, &frozen);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(stable, std::string("hello world"));
    QASR_EXPECT(unstable.empty());
    QASR_EXPECT_EQ(frozen, std::string("world"));
}

QASR_TEST(ForceFreezeEmptyUnstable) {
    std::string stable = "hello";
    std::string unstable;
    std::string frozen;
    qasr::Status s = qasr::ForceFreezeAgedSuffix(&stable, &unstable, &frozen);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(frozen.empty());
}

// --- ReanchorContext ---

QASR_TEST(ReanchorContextResetsToStable) {
    std::string context = "old context with extra stuff";
    qasr::Status s = qasr::ReanchorContext("stable only", &context);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(context, std::string("stable only"));
}

QASR_TEST(ReanchorContextNull) {
    qasr::Status s = qasr::ReanchorContext("test", nullptr);
    QASR_EXPECT(!s.ok());
}

// --- EvictOldHistory ---

QASR_TEST(EvictOldHistoryNull) {
    qasr::Status s = qasr::EvictOldHistory(nullptr, 100000, 50000);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EvictOldHistoryInvalidSamples) {
    qasr::EncoderCache cache;
    qasr::Status s = qasr::EvictOldHistory(&cache, 100, 0);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(EvictOldHistoryNoEvictionNeeded) {
    qasr::EncoderCache cache;
    cache.Store(0, std::vector<float>(10), 1);
    qasr::Status s = qasr::EvictOldHistory(&cache, 10000, 50000);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(cache.size(), std::size_t(1));
}

// --- EncoderCache capacity limit ---

QASR_TEST(EncoderCacheUnlimitedByDefault) {
    qasr::EncoderCache cache;
    QASR_EXPECT_EQ(cache.max_entries(), std::size_t(0));
    for (int i = 0; i < 100; ++i) {
        cache.Store(i, std::vector<float>(10, 0.0f), 1);
    }
    QASR_EXPECT_EQ(cache.size(), std::size_t(100));
}

QASR_TEST(EncoderCacheCapacityEvictsOldest) {
    qasr::EncoderCache cache(3);
    QASR_EXPECT_EQ(cache.max_entries(), std::size_t(3));
    cache.Store(10, std::vector<float>(10, 0.0f), 1);
    cache.Store(20, std::vector<float>(10, 0.0f), 1);
    cache.Store(30, std::vector<float>(10, 0.0f), 1);
    QASR_EXPECT_EQ(cache.size(), std::size_t(3));

    // Adding a 4th entry should evict the oldest (window 10)
    cache.Store(40, std::vector<float>(10, 0.0f), 1);
    QASR_EXPECT_EQ(cache.size(), std::size_t(3));
    QASR_EXPECT(!cache.Has(10));
    QASR_EXPECT(cache.Has(20));
    QASR_EXPECT(cache.Has(30));
    QASR_EXPECT(cache.Has(40));
}

QASR_TEST(EncoderCacheCapacityReplaceDoesNotEvict) {
    qasr::EncoderCache cache(2);
    cache.Store(0, std::vector<float>(10, 1.0f), 1);
    cache.Store(1, std::vector<float>(10, 2.0f), 1);
    // Replacing window 0 should not trigger eviction
    cache.Store(0, std::vector<float>(10, 3.0f), 5);
    QASR_EXPECT_EQ(cache.size(), std::size_t(2));
    QASR_EXPECT(cache.Has(0));
    QASR_EXPECT(cache.Has(1));
}

QASR_TEST(EncoderCacheCapacityOneSlot) {
    qasr::EncoderCache cache(1);
    cache.Store(0, std::vector<float>(4, 0.0f), 1);
    QASR_EXPECT_EQ(cache.size(), std::size_t(1));
    cache.Store(1, std::vector<float>(4, 0.0f), 1);
    QASR_EXPECT_EQ(cache.size(), std::size_t(1));
    QASR_EXPECT(!cache.Has(0));
    QASR_EXPECT(cache.Has(1));
}
