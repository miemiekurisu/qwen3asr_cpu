#include "qasr/inference/aligner_client.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// C backend
extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_audio.h"
}

namespace qasr {

// ========================================================================
// Validation
// ========================================================================

Status ValidateAlignerConfig(const AlignerConfig & config) {
    if (config.model_dir.empty()) {
        return Status(StatusCode::kInvalidArgument, "aligner model_dir must not be empty");
    }
    return OkStatus();
}

// ========================================================================
// ForcedAligner::Impl
// ========================================================================

struct ForcedAligner::Impl {
    qwen_ctx_t * ctx = nullptr;
};

ForcedAligner::ForcedAligner() = default;

ForcedAligner::~ForcedAligner() {
    Unload();
}

ForcedAligner::ForcedAligner(ForcedAligner && other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

ForcedAligner & ForcedAligner::operator=(ForcedAligner && other) noexcept {
    if (this != &other) {
        Unload();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

Status ForcedAligner::Load(const AlignerConfig & config) {
    Status s = ValidateAlignerConfig(config);
    if (!s.ok()) return s;

    if (impl_) {
        return Status(StatusCode::kFailedPrecondition, "aligner already loaded");
    }

    qwen_ctx_t * ctx = qwen_load(config.model_dir.c_str());
    if (!ctx) {
        return Status(StatusCode::kInternal,
                      "failed to load ForcedAligner model from " + config.model_dir);
    }

    if (ctx->config.classify_num <= 0) {
        qwen_free(ctx);
        return Status(StatusCode::kInvalidArgument,
                      "model at " + config.model_dir + " is not a ForcedAligner (no lm_head/classify_num)");
    }

    impl_ = new Impl();
    impl_->ctx = ctx;
    return OkStatus();
}

void ForcedAligner::Unload() {
    if (impl_) {
        if (impl_->ctx) {
            qwen_free(impl_->ctx);
            impl_->ctx = nullptr;
        }
        delete impl_;
        impl_ = nullptr;
    }
}

bool ForcedAligner::IsLoaded() const noexcept {
    return impl_ && impl_->ctx;
}

Status ForcedAligner::Align(const std::string & audio_path,
                             const std::string & text,
                             const std::string & language,
                             AlignResult * result) {
    if (!result) {
        return Status(StatusCode::kInvalidArgument, "result must not be null");
    }
    if (!IsLoaded()) {
        return Status(StatusCode::kFailedPrecondition, "aligner not loaded");
    }

    // Load WAV audio
    int n_samples = 0;
    float * samples = qwen_load_wav(audio_path.c_str(), &n_samples);
    if (!samples || n_samples <= 0) {
        return Status(StatusCode::kInternal, "failed to load WAV: " + audio_path);
    }

    // Run forced alignment
    qwen_align_result_t * ar = qwen_forced_align(
        impl_->ctx, samples, n_samples, text.c_str(), language.c_str());
    free(samples);

    if (!ar) {
        return Status(StatusCode::kInternal, "forced alignment failed");
    }

    // Convert C result to C++ AlignResult
    result->words.clear();
    result->words.reserve(static_cast<std::size_t>(ar->n_words));
    for (int i = 0; i < ar->n_words; i++) {
        AlignedWord w;
        w.text = ar->words[i].text ? ar->words[i].text : "";
        w.start_sec = static_cast<double>(ar->words[i].start_sec);
        w.end_sec = static_cast<double>(ar->words[i].end_sec);
        result->words.push_back(std::move(w));
    }

    qwen_align_result_free(ar);
    return OkStatus();
}

Status ForcedAligner::AlignSamples(const float * samples, int n_samples,
                                    const std::string & text,
                                    const std::string & language,
                                    AlignResult * result) {
    if (!result) {
        return Status(StatusCode::kInvalidArgument, "result must not be null");
    }
    if (!IsLoaded()) {
        return Status(StatusCode::kFailedPrecondition, "aligner not loaded");
    }
    if (!samples || n_samples <= 0) {
        return Status(StatusCode::kInvalidArgument, "no audio samples");
    }

    qwen_align_result_t * ar = qwen_forced_align(
        impl_->ctx, samples, n_samples,
        text.c_str(), language.c_str());

    if (!ar) {
        return Status(StatusCode::kInternal, "forced alignment failed");
    }

    result->words.clear();
    result->words.reserve(static_cast<std::size_t>(ar->n_words));
    for (int i = 0; i < ar->n_words; i++) {
        AlignedWord w;
        w.text = ar->words[i].text ? ar->words[i].text : "";
        w.start_sec = static_cast<double>(ar->words[i].start_sec);
        w.end_sec = static_cast<double>(ar->words[i].end_sec);
        result->words.push_back(std::move(w));
    }

    qwen_align_result_free(ar);
    return OkStatus();
}

}  // namespace qasr

