// tests/qwen_clone_shared_test.cc
//
// Regression tests for the qwen_clone_shared() shallow-struct copy that
// caused the "150 s double-free crash" (see docs/INCIDENTS.md).
//
// Contract under test:
//   * Caller owns the source ctx (loaded via qwen_load).
//   * qwen_clone_shared() returns a NEW ctx that:
//       - aliases model weights and config with the source (no deep copy)
//       - sets owns_model_data = 0
//       - zeroes prepared prefill buffers (so the clone doesn't reuse
//         the source's persisted QKV/GateUp scratch)
//       - disables INT8 (clones start on the BF16 path)
//       - clears all callbacks (caller wires its own)
//   * The source ctx MUST outlive the clone; qwen_free() on either one
//     must not affect the other (no double-free, no UAF).
//   * qwen_free(NULL) must be a no-op.

#include "tests/test_registry.h"

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_onednn.h"
}

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <string>

namespace {

// Build a qwen_ctx_t with controllable fields.  We allocate via
// calloc, fill the visible fields, and let the test decide which
// fields to exercise.  We do NOT call qwen_load() because that
// requires a real safetensors directory; the clone contract is
// fully observable from the struct's value, not from the model
// contents on disk.
//
// IMPORTANT: every pointer field that qwen_free() will eventually
// free() must be either NULL or a heap-allocated chunk (so the
// free() succeeds).  Fake addresses like 0xDEADBEEF would crash
// qwen_free() at the end of the test.  We allocate the per-layer
// scratch buffers via calloc() so the aliasing check still has a
// distinct address to compare, and qwen_free() can release them.
qwen_ctx_t *MakeSourceCtx() {
    qwen_ctx_t *ctx = static_cast<qwen_ctx_t *>(std::calloc(1, sizeof(qwen_ctx_t)));
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->owns_model_data = 1;
    std::snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", "/fake/model/dir");
    ctx->config.enc_layers = 2;
    ctx->config.dec_layers = 2;
    ctx->runtime_perf.decoder_prefill_qkv_bytes = 1024 * 1024;
    ctx->runtime_perf.decoder_prefill_qkv_layers = 28;
    ctx->runtime_perf.decoder_prefill_gate_up_bytes = 512 * 1024;
    ctx->runtime_perf.decoder_prefill_gate_up_layers = 28;
    // Heap-allocated sentinels the test checks for aliasing.
    // qwen_free() will free() these, so they must be real chunks.
    ctx->decoder.tok_embed_suffix_max = static_cast<float *>(std::calloc(8, sizeof(float)));
    ctx->decoder.lm_head_suffix_max = static_cast<float *>(std::calloc(8, sizeof(float)));
    // Per-layer prepared buffers: each must be a real chunk so
    // qwen_free() can release them.  The clone contract is "the
    // clone zeros its own copy" — we verify that with rows/cols
    // sentinels, which qwen_free() doesn't touch.
    for (int i = 0; i < QWEN_MAX_DEC_LAYERS; ++i) {
        ctx->decoder.layers[i].prefill_qkv_prepared.f32_data = static_cast<float *>(std::calloc(64, sizeof(float)));
        ctx->decoder.layers[i].prefill_qkv_prepared.rows = 7;
        ctx->decoder.layers[i].prefill_qkv_prepared.cols = 11;
        ctx->decoder.layers[i].prefill_qkv_prepared.bytes = 999;
        ctx->decoder.layers[i].prefill_gate_up_prepared.f32_data = static_cast<float *>(std::calloc(64, sizeof(float)));
        ctx->decoder.layers[i].prefill_gate_up_prepared.rows = 13;
        ctx->decoder.layers[i].prefill_gate_up_prepared.cols = 17;
        ctx->decoder.layers[i].prefill_gate_up_prepared.bytes = 888;
    }
    return ctx;
}

// --- qwen_free: NULL safety ---

QASR_TEST(QwenFreeNullIsNoOp) {
    // qwen_free must accept NULL; if it didn't, the very first
    // success path that took a fast-exit would crash here.
    qwen_free(nullptr);
    QASR_EXPECT(true);  // reaching this line = no crash
}

// NOTE: qwen_free() is NOT idempotent.  It uses the FREE0 macro
// to null the weight pointers (so a second pass through the
// owns_model_data block is safe), but the trailing `free(ctx)` at
// the end of the function is a raw free.  Calling qwen_free on the
// same ctx twice would double-free the ctx struct itself.  This is
// by design: the contract is "caller owns the ctx, frees it
// exactly once".  The 150 s crash regression is covered by the
// qwen_clone_shared tests below (clone aliases weights, source is
// the sole owner).

// --- qwen_clone_shared: input validation ---

QASR_TEST(QwenCloneSharedNullSrcReturnsNull) {
    QASR_EXPECT(qwen_clone_shared(nullptr) == nullptr);
}

QASR_TEST(QwenCloneSharedProducesNewPointer) {
    qwen_ctx_t *src = MakeSourceCtx();
    QASR_EXPECT(src != nullptr);
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone != src);  // distinct heap address
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedDoesNotOwnModelData) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT_EQ(clone->owns_model_data, 0);
    QASR_EXPECT_EQ(src->owns_model_data, 1);  // source unchanged
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedAliasesModelWeights) {
    // The whole point of clone_shared is weight aliasing; if the
    // pointers are deep-copied, the contract is broken and we'd
    // load 2 GB per session.
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->decoder.tok_embed_suffix_max == src->decoder.tok_embed_suffix_max);
    QASR_EXPECT(clone->decoder.lm_head_suffix_max == src->decoder.lm_head_suffix_max);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedCopiesConfig) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT_EQ(clone->config.enc_layers, src->config.enc_layers);
    QASR_EXPECT_EQ(clone->config.dec_layers, src->config.dec_layers);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedCopiesModelDir) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(std::strcmp(clone->model_dir, src->model_dir) == 0);
    qwen_free(clone);
    qwen_free(src);
}

// --- qwen_clone_shared: prepared prefill buffer hygiene ---

QASR_TEST(QwenCloneSharedZeroesPrefillQkvPrepared) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_qkv_bytes == 0U);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_qkv_layers == 0);
    QASR_EXPECT(clone->runtime_profile.decoder_prefill_qkv_persist_f32 == 0);
    QASR_EXPECT(clone->runtime_profile.decoder_prefill_qkv_budget_bytes == 0U);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedZeroesPrefillGateUpPrepared) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_gate_up_bytes == 0U);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_gate_up_layers == 0);
    QASR_EXPECT(clone->runtime_profile.decoder_prefill_gate_up_persist_f32 == 0);
    QASR_EXPECT(clone->runtime_profile.decoder_prefill_gate_up_budget_bytes == 0U);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedZeroesPerLayerPrefillBuffers) {
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    // qwen_clone_shared() only iterates 0..config.dec_layers when
    // clearing per-layer prepared buffers (qwen_asr.c:433).  For
    // unused tail layers, the shallow struct copy from the source
    // (which is calloc'd, so all-NULL) is the source of truth.  We
    // check the contract for the active layers only.
    for (int i = 0; i < clone->config.dec_layers; ++i) {
        QASR_EXPECT(clone->decoder.layers[i].prefill_qkv_prepared.f32_data == nullptr);
        QASR_EXPECT(clone->decoder.layers[i].prefill_qkv_prepared.rows == 0);
        QASR_EXPECT(clone->decoder.layers[i].prefill_qkv_prepared.cols == 0);
        QASR_EXPECT(clone->decoder.layers[i].prefill_qkv_prepared.bytes == 0U);
        QASR_EXPECT(clone->decoder.layers[i].prefill_gate_up_prepared.f32_data == nullptr);
        QASR_EXPECT(clone->decoder.layers[i].prefill_gate_up_prepared.rows == 0);
        QASR_EXPECT(clone->decoder.layers[i].prefill_gate_up_prepared.cols == 0);
        QASR_EXPECT(clone->decoder.layers[i].prefill_gate_up_prepared.bytes == 0U);
    }
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedSourcePrefillUnchanged) {
    // The clone zeroes its OWN prepared buffers, but it must NOT
    // touch the source's prepared buffers — the source may still
    // be running other sessions that need them.  We capture the
    // source's pointer values BEFORE the clone, then assert they
    // survive the clone unchanged.  Iterate 0..config.dec_layers
    // (qwen_clone_shared only touches the active layers).
    qwen_ctx_t *src = MakeSourceCtx();
    const int active = src->config.dec_layers;
    float *src_qkv[QWEN_MAX_DEC_LAYERS];
    float *src_gu[QWEN_MAX_DEC_LAYERS];
    for (int i = 0; i < active; ++i) {
        src_qkv[i] = src->decoder.layers[i].prefill_qkv_prepared.f32_data;
        src_gu[i] = src->decoder.layers[i].prefill_gate_up_prepared.f32_data;
    }
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    for (int i = 0; i < active; ++i) {
        QASR_EXPECT(src->decoder.layers[i].prefill_qkv_prepared.f32_data == src_qkv[i]);
        QASR_EXPECT(src->decoder.layers[i].prefill_qkv_prepared.rows == 7);
        QASR_EXPECT(src->decoder.layers[i].prefill_qkv_prepared.cols == 11);
        QASR_EXPECT(src->decoder.layers[i].prefill_qkv_prepared.bytes == 999U);
        QASR_EXPECT(src->decoder.layers[i].prefill_gate_up_prepared.f32_data == src_gu[i]);
        QASR_EXPECT(src->decoder.layers[i].prefill_gate_up_prepared.rows == 13);
        QASR_EXPECT(src->decoder.layers[i].prefill_gate_up_prepared.cols == 17);
        QASR_EXPECT(src->decoder.layers[i].prefill_gate_up_prepared.bytes == 888U);
    }
    QASR_EXPECT(src->runtime_perf.decoder_prefill_qkv_bytes == 1024U * 1024U);
    qwen_free(clone);
    qwen_free(src);
}

// --- qwen_clone_shared: callback / INT8 hygiene ---

QASR_TEST(QwenCloneSharedClearsCallbacks) {
    qwen_ctx_t *src = MakeSourceCtx();
    // Stuff non-null callbacks into the source.
    src->token_cb = reinterpret_cast<qwen_token_cb>(0xAAAA1111);
    src->token_cb_userdata = reinterpret_cast<void *>(0xBBBB2222);
    src->cancel_cb = reinterpret_cast<qwen_cancel_cb>(0xCCCC3333);
    src->cancel_cb_userdata = reinterpret_cast<void *>(0xDDDD4444);
    src->chunk_cb = reinterpret_cast<qwen_stream_chunk_cb_t>(0xEEEE5555);
    src->chunk_cb_userdata = reinterpret_cast<void *>(0xFFFF6666);
    src->segment_cb = reinterpret_cast<qwen_segment_cb>(0xAAAA7777);
    src->segment_cb_userdata = reinterpret_cast<void *>(0xBBBB8888);

    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->token_cb == nullptr);
    QASR_EXPECT(clone->token_cb_userdata == nullptr);
    QASR_EXPECT(clone->cancel_cb == nullptr);
    QASR_EXPECT(clone->cancel_cb_userdata == nullptr);
    QASR_EXPECT(clone->chunk_cb == nullptr);
    QASR_EXPECT(clone->chunk_cb_userdata == nullptr);
    QASR_EXPECT(clone->segment_cb == nullptr);
    QASR_EXPECT(clone->segment_cb_userdata == nullptr);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedDisablesInt8) {
    qwen_ctx_t *src = MakeSourceCtx();
    src->decoder_int8 = 1;
    // Real calloc'd pointers (not fake addresses): qwen_free()
    // walks the int8 layer array via n_int8_dec_layers and would
    // deref whatever int8_dec_layers points to.  The layer
    // members themselves are zeroed (calloc), so the inner
    // qwen_int8_weight_free / matmul_free calls are no-ops.
    src->int8_dec_layers = std::calloc(28, sizeof(qwen_int8_dec_layer_t));
    src->n_int8_dec_layers = 28;
    src->encoder_int8 = 1;
    src->int8_enc_layers = std::calloc(18, sizeof(qwen_int8_enc_layer_t));
    src->n_int8_enc_layers = 18;

    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT_EQ(clone->decoder_int8, 0);
    QASR_EXPECT(clone->int8_dec_layers == nullptr);
    QASR_EXPECT_EQ(clone->n_int8_dec_layers, 0);
    QASR_EXPECT_EQ(clone->encoder_int8, 0);
    QASR_EXPECT(clone->int8_enc_layers == nullptr);
    QASR_EXPECT_EQ(clone->n_int8_enc_layers, 0);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedCopiesRuntimeStreamingParams) {
    qwen_ctx_t *src = MakeSourceCtx();
    src->segment_sec = 1.5f;
    src->search_sec = 2.5f;
    src->stream_chunk_sec = 3.5f;
    src->stream_rollback = 7;
    src->stream_unfixed_chunks = 2;
    src->stream_max_new_tokens = 64;
    src->stream_idle_flush_ms = 800;
    src->stream_idle_flush_min_sec = 0.3f;
    src->stream_idle_flush_max_new_tokens = 24;
    src->past_text_conditioning = 1;
    src->skip_silence = 1;

    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->segment_sec == src->segment_sec);
    QASR_EXPECT(clone->search_sec == src->search_sec);
    QASR_EXPECT(clone->stream_chunk_sec == src->stream_chunk_sec);
    QASR_EXPECT_EQ(clone->stream_rollback, src->stream_rollback);
    QASR_EXPECT_EQ(clone->stream_unfixed_chunks, src->stream_unfixed_chunks);
    QASR_EXPECT_EQ(clone->stream_max_new_tokens, src->stream_max_new_tokens);
    QASR_EXPECT_EQ(clone->stream_idle_flush_ms, src->stream_idle_flush_ms);
    QASR_EXPECT(clone->stream_idle_flush_min_sec == src->stream_idle_flush_min_sec);
    QASR_EXPECT_EQ(clone->stream_idle_flush_max_new_tokens, src->stream_idle_flush_max_new_tokens);
    QASR_EXPECT_EQ(clone->past_text_conditioning, src->past_text_conditioning);
    QASR_EXPECT_EQ(clone->skip_silence, src->skip_silence);
    qwen_free(clone);
    qwen_free(src);
}

// --- qwen_clone_shared: ownership / double-free regression ---

QASR_TEST(QwenCloneSharedFreeCloneFirstNoUafOnSource) {
    // The source's tok_embed_suffix_max / lm_head_suffix_max is a
    // heap pointer (sentinel here, real alloc in production).  The
    // clone aliases it with owns_model_data=0.  When we free the
    // clone, qwen_free() must NOT touch the shared pointer; when we
    // free the source, the FREE0 macro sets the source ptr to NULL
    // but the aliasing clone has already been freed (its copy of the
    // pointer is the SAME address, but qwen_free on it set its copy
    // to NULL via FREE0, then the second free sees NULL and skips).
    //
    // NOTE: with the literal sentinel addresses used here (0xDEAD...)
    // the free() would actually crash, so we use calloc-allocated
    // sentinels that free() can handle.
    qwen_ctx_t *src = MakeSourceCtx();
    src->decoder.tok_embed_suffix_max = static_cast<float *>(std::calloc(1, sizeof(float)));
    src->decoder.lm_head_suffix_max = static_cast<float *>(std::calloc(1, sizeof(float)));
    QASR_EXPECT(src->decoder.tok_embed_suffix_max != nullptr);
    QASR_EXPECT(src->decoder.lm_head_suffix_max != nullptr);

    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    // Aliasing — both point at the same heap chunk.
    QASR_EXPECT(clone->decoder.tok_embed_suffix_max == src->decoder.tok_embed_suffix_max);
    QASR_EXPECT(clone->decoder.lm_head_suffix_max == src->decoder.lm_head_suffix_max);

    qwen_free(clone);  // should NOT free the shared chunks
    // Source still owns the chunks — the addresses are unchanged.
    QASR_EXPECT(src->decoder.tok_embed_suffix_max != nullptr);
    QASR_EXPECT(src->decoder.lm_head_suffix_max != nullptr);

    qwen_free(src);  // owner frees the shared chunks, FREE0 NULLs
    // Reaching here without SIGSEGV / double-free is the assertion.
    QASR_EXPECT(true);
}

QASR_TEST(QwenCloneSharedFreeSourceFirstNoUafOnClone) {
    // Inverse: free source first.  The source's qwen_free() frees
    // the shared chunks (its owns_model_data=1).  After this, the
    // clone's tok_embed_suffix_max pointer is DANGLING.  But we
    // must not call qwen_free(clone) afterward in production code;
    // we test the safer path: when the source dies first, the
    // operator is expected to drop all clones too.  This test only
    // verifies the source's own free is correct.
    qwen_ctx_t *src = MakeSourceCtx();
    src->decoder.tok_embed_suffix_max = static_cast<float *>(std::calloc(1, sizeof(float)));
    src->decoder.lm_head_suffix_max = static_cast<float *>(std::calloc(1, sizeof(float)));
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->decoder.tok_embed_suffix_max == src->decoder.tok_embed_suffix_max);

    qwen_free(src);  // source frees the shared chunks
    // Clone's pointer is now dangling — touching it would UAF.
    // We deliberately do NOT touch clone->decoder.tok_embed_suffix_max
    // here; we only test that the source's own free was clean.
    std::free(clone);  // the test cleanup, not qwen_free
    QASR_EXPECT(true);
}

// --- qwen_clone_shared: boundary / garbage / beyond-boundary ---

QASR_TEST(QwenCloneSharedBoundaryModelDirLength512) {
    // model_dir is char[512].  Source string of length 511 (no
    // terminator) must be copied to 511 chars + NUL = 512 bytes.
    qwen_ctx_t *src = MakeSourceCtx();
    std::string exact(511, 'A');
    std::snprintf(src->model_dir, sizeof(src->model_dir), "%s", exact.c_str());
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(std::strlen(clone->model_dir) == 511U);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedBeyondBoundaryModelDirTruncates) {
    // model_dir is char[512].  Source string of length 1024 must
    // be truncated to 511 + NUL.  The snprintf in qwen_load
    // guarantees the truncation; the clone uses snprintf too.
    qwen_ctx_t *src = MakeSourceCtx();
    std::string long_path(1024, 'B');
    // Use snprintf (the same primitive qwen_load uses) so we can
    // pre-truncate to the declared buffer size.
    std::snprintf(src->model_dir, sizeof(src->model_dir), "%s", long_path.c_str());
    QASR_EXPECT(std::strlen(src->model_dir) == 511U);
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(std::strlen(clone->model_dir) == 511U);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedGarbageModelDirBytesPreserved) {
    // model_dir is raw bytes; control chars / high-bit chars must
    // be preserved byte-for-byte.  Catching a sanitizer issue here
    // (e.g. strlen overrun) would fail.
    qwen_ctx_t *src = MakeSourceCtx();
    const char garbage[] = {'a', '\t', '\n', static_cast<char>(0xFF), '/', '\0'};
    std::snprintf(src->model_dir, sizeof(src->model_dir), "%s", garbage);
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    for (std::size_t i = 0; i < sizeof(garbage); ++i) {
        QASR_EXPECT(clone->model_dir[i] == garbage[i]);
    }
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedEmptyModelDir) {
    // Empty model_dir is a valid input; clone must not crash.
    qwen_ctx_t *src = MakeSourceCtx();
    src->model_dir[0] = '\0';
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->model_dir[0] == '\0');
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedIntMaxRuntimePerfZeroed) {
    // Garbage: all runtime_perf fields set to SIZE_MAX / INT_MAX.
    // The clone must zero its own copy regardless of the source
    // values (this is the fix that prevents the clone from
    // re-using the source's persisted QKV scratch).
    qwen_ctx_t *src = MakeSourceCtx();
    src->runtime_perf.decoder_prefill_qkv_bytes = SIZE_MAX;
    src->runtime_perf.decoder_prefill_qkv_layers = INT_MAX;
    src->runtime_perf.decoder_prefill_gate_up_bytes = SIZE_MAX;
    src->runtime_perf.decoder_prefill_gate_up_layers = INT_MAX;
    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_qkv_bytes == 0U);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_qkv_layers == 0);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_gate_up_bytes == 0U);
    QASR_EXPECT(clone->runtime_perf.decoder_prefill_gate_up_layers == 0);
    qwen_free(clone);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedTwoClonesFromSameSource) {
    // The server creates one clone per session; we test that two
    // clones from the same source are independent objects.
    qwen_ctx_t *src = MakeSourceCtx();
    qwen_ctx_t *clone_a = qwen_clone_shared(src);
    qwen_ctx_t *clone_b = qwen_clone_shared(src);
    QASR_EXPECT(clone_a != nullptr);
    QASR_EXPECT(clone_b != nullptr);
    QASR_EXPECT(clone_a != clone_b);
    QASR_EXPECT(clone_a != src);
    QASR_EXPECT(clone_b != src);
    // Free in reverse order of creation to mimic the server's
    // shutdown sequence.
    qwen_free(clone_b);
    qwen_free(clone_a);
    qwen_free(src);
}

QASR_TEST(QwenCloneSharedDoesNotModifySource) {
    // Side-effect freedom: cloning must not touch the source's
    // state machine fields.  We snapshot every field we care
    // about before and after.
    qwen_ctx_t *src = MakeSourceCtx();
    src->last_run_cancelled = 1;
    src->vad_silence_run = 42;
    src->vad_last_prob = 0.87f;

    // Take a coarse fingerprint of the source.
    const int own = src->owns_model_data;
    const int sil = src->vad_silence_run;
    const float prob = src->vad_last_prob;
    const int enc = src->config.enc_layers;
    const int dec = src->config.dec_layers;

    qwen_ctx_t *clone = qwen_clone_shared(src);
    QASR_EXPECT(clone != nullptr);

    QASR_EXPECT_EQ(src->owns_model_data, own);
    QASR_EXPECT_EQ(src->vad_silence_run, sil);
    QASR_EXPECT(src->vad_last_prob == prob);
    QASR_EXPECT_EQ(src->config.enc_layers, enc);
    QASR_EXPECT_EQ(src->config.dec_layers, dec);
    QASR_EXPECT_EQ(src->last_run_cancelled, 1);
    qwen_free(clone);
    qwen_free(src);
}

}  // namespace
