#include "tests/test_registry.h"
#include "qasr/inference/decoder.h"

// --- KvCache ---

QASR_TEST(KvCacheDefaultNotAllocated) {
    qasr::KvCache cache;
    QASR_EXPECT(!cache.is_allocated());
    QASR_EXPECT_EQ(cache.length(), int32_t(0));
    QASR_EXPECT_EQ(cache.capacity(), int32_t(0));
}

QASR_TEST(KvCacheAllocateValid) {
    qasr::KvCache cache;
    qasr::Status s = cache.Allocate(4, 2, 64, 512);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(cache.is_allocated());
    QASR_EXPECT_EQ(cache.capacity(), int32_t(512));
    QASR_EXPECT_EQ(cache.length(), int32_t(0));
}

QASR_TEST(KvCacheAllocateInvalidDimensions) {
    qasr::KvCache cache;
    QASR_EXPECT(!cache.Allocate(0, 2, 64, 512).ok());
    QASR_EXPECT(!cache.Allocate(4, 0, 64, 512).ok());
    QASR_EXPECT(!cache.Allocate(4, 2, 0, 512).ok());
    QASR_EXPECT(!cache.Allocate(4, 2, 64, 0).ok());
}

QASR_TEST(KvCacheReset) {
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    cache.set_length(100);
    QASR_EXPECT_EQ(cache.length(), int32_t(100));
    cache.Reset();
    QASR_EXPECT_EQ(cache.length(), int32_t(0));
    QASR_EXPECT(cache.is_allocated());  // Still allocated after reset
}

QASR_TEST(KvCacheDataPointers) {
    qasr::KvCache cache;
    cache.Allocate(2, 2, 32, 64);
    QASR_EXPECT(cache.key_data() != nullptr);
    QASR_EXPECT(cache.value_data() != nullptr);
}

// --- Prefill ---

QASR_TEST(PrefillUnloadedWeights) {
    qasr::DecoderWeights w;
    w.loaded = false;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    std::vector<float> embed(256, 0.0f);
    qasr::Status s = qasr::Prefill(w, embed.data(), 1, &cache);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(PrefillCacheNotAllocated) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    std::vector<float> embed(256, 0.0f);
    qasr::Status s = qasr::Prefill(w, embed.data(), 1, &cache);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(PrefillNullInput) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    qasr::Status s = qasr::Prefill(w, nullptr, 1, &cache);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(PrefillZeroSeqLen) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    std::vector<float> embed(256, 0.0f);
    qasr::Status s = qasr::Prefill(w, embed.data(), 0, &cache);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(PrefillExceedsCapacity) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 10);
    std::vector<float> embed(256, 0.0f);
    qasr::Status s = qasr::Prefill(w, embed.data(), 20, &cache);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(PrefillValid) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    std::vector<float> embed(256, 0.0f);
    qasr::Status s = qasr::Prefill(w, embed.data(), 10, &cache);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(cache.length(), int32_t(10));
}

// --- DecodeStep ---

QASR_TEST(DecodeStepUnloaded) {
    qasr::DecoderWeights w;
    w.loaded = false;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    std::vector<float> embed(256, 0.0f);
    int32_t token = -1;
    qasr::Status s = qasr::DecodeStep(w, embed.data(), &cache, &token);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(DecodeStepCacheFull) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 2);
    cache.set_length(2);
    std::vector<float> embed(256, 0.0f);
    int32_t token = -1;
    qasr::Status s = qasr::DecodeStep(w, embed.data(), &cache, &token);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(DecodeStepValid) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::KvCache cache;
    cache.Allocate(4, 2, 64, 512);
    std::vector<float> embed(256, 0.0f);
    int32_t token = -1;
    qasr::Status s = qasr::DecodeStep(w, embed.data(), &cache, &token);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(token >= 0);
    QASR_EXPECT_EQ(cache.length(), int32_t(1));
}

// --- BuildPromptEmbeddings ---

QASR_TEST(BuildPromptEmbeddingsNullOutput) {
    qasr::DecoderWeights w;
    w.loaded = true;
    qasr::Status s = qasr::BuildPromptEmbeddings(w, nullptr, 0, nullptr, 0, 256, nullptr, nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(BuildPromptEmbeddingsUnloaded) {
    qasr::DecoderWeights w;
    w.loaded = false;
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::BuildPromptEmbeddings(w, nullptr, 0, nullptr, 0, 256, &output, &seq_len);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(BuildPromptEmbeddingsValid) {
    qasr::DecoderWeights w;
    w.loaded = true;
    std::int32_t prompt[] = {1, 2, 3};
    std::vector<float> audio(256 * 5, 1.0f);
    std::vector<float> output;
    int32_t seq_len = 0;
    qasr::Status s = qasr::BuildPromptEmbeddings(w, prompt, 3, audio.data(), 5, 256, &output, &seq_len);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(seq_len, int32_t(8));
    QASR_EXPECT_EQ(output.size(), std::size_t(8 * 256));
}
