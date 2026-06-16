#include "qasr/engine/cuda_asr_engine.h"
#include "qasr/backend/cpu_backend.h"
#include "qasr/backend/cuda_backend.h"
#include "audio_segmentation.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#ifdef QASR_CUDA_BACKEND_ENABLED
#include <cuda_runtime.h>
#endif

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
}

namespace qasr {

#ifdef QASR_CUDA_BACKEND_ENABLED
/* ---- Single-segment GPU transcribe helper ---- */

static std::string transcribe_segment_gpu(
        CudaBackend *cuda_backend,
        CudaSessionState *sess_state,
        const V2EngineConfig &config,
        const std::string &language,
        const std::string &prompt,
        const float *seg_samples,
        int seg_sample_count,
        int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;

    /* Mel spectrogram (CPU) */
    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(seg_samples, seg_sample_count, &mel_frames);
    if (!mel) return "";

    /* GPU encoder forward */
    cudaEvent_t enc_start, enc_stop;
    cudaEventCreate(&enc_start);
    cudaEventCreate(&enc_stop);
    cudaEventRecord(enc_start);
    int enc_seq_len = 0;
    auto status = cuda_backend->EncoderForward(sess_state, mel, mel_frames, &enc_seq_len);
    std::free(mel);
    if (!status.ok() || enc_seq_len <= 0) {
        cudaEventDestroy(enc_start);
        cudaEventDestroy(enc_stop);
        return "";
    }
    cudaEventRecord(enc_stop);
    cudaEventSynchronize(enc_stop);
    float enc_ms = 0;
    cudaEventElapsedTime(&enc_ms, enc_start, enc_stop);
    cudaEventDestroy(enc_start);
    cudaEventDestroy(enc_stop);

    /* Load tokenizer */
    std::string vocab_path = config.model_dir + "/vocab.json";
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(vocab_path.c_str());
    if (!tokenizer) {
        return "";
    }

    /* Prepare forced language tokens */
    int *force_prompt_tokens = nullptr;
    int n_force_prompt_tokens = 0;
    if (!language.empty()) {
        char force_text[256];
        snprintf(force_text, sizeof(force_text), "language %s", language.c_str());
        int n_lang_txt = 0;
        int *lang_txt_tokens = qwen_tokenizer_encode(tokenizer, force_text, &n_lang_txt);
        n_force_prompt_tokens = n_lang_txt + 1;
        force_prompt_tokens = (int *)malloc((size_t)n_force_prompt_tokens * sizeof(int));
        if (lang_txt_tokens) {
            memcpy(force_prompt_tokens, lang_txt_tokens, (size_t)n_lang_txt * sizeof(int));
            std::free(lang_txt_tokens);
        }
        force_prompt_tokens[n_lang_txt] = QWEN_TOKEN_ASR_TEXT;
    }

    /* Prepare prompt tokens */
    int *prompt_tokens = nullptr;
    int n_prompt_tokens = 0;
    if (!prompt.empty()) {
        prompt_tokens = qwen_tokenizer_encode(tokenizer, prompt.c_str(), &n_prompt_tokens);
    }

    /* Build input token sequence */
    static const int PROMPT_PREFIX_HEAD[] = {QWEN_TOKEN_IM_START, 8948, 198};
    static const int PROMPT_PREFIX_TAIL[] = {QWEN_TOKEN_IM_END, 198, QWEN_TOKEN_IM_START, 872, 198, QWEN_TOKEN_AUDIO_START};
    static const int PROMPT_SUFFIX_BASE[] = {QWEN_TOKEN_AUDIO_END, QWEN_TOKEN_IM_END, 198, QWEN_TOKEN_IM_START, 77091, 198};

    int prefix_len = 3 + n_prompt_tokens + 6;
    int suffix_len = 6 + n_force_prompt_tokens;
    int total_tokens = prefix_len + enc_seq_len + suffix_len;

    std::vector<std::int32_t> tokens(static_cast<size_t>(total_tokens));
    int off = 0;
    for (int i = 0; i < 3; i++) tokens[off++] = PROMPT_PREFIX_HEAD[i];
    for (int i = 0; i < n_prompt_tokens; i++) tokens[off++] = prompt_tokens[i];
    for (int i = 0; i < 6; i++) tokens[off++] = PROMPT_PREFIX_TAIL[i];
    for (int i = 0; i < enc_seq_len; i++) tokens[off++] = QWEN_TOKEN_AUDIO_PAD;
    for (int i = 0; i < 6; i++) tokens[off++] = PROMPT_SUFFIX_BASE[i];
    for (int i = 0; i < n_force_prompt_tokens; i++) tokens[off++] = force_prompt_tokens[i];

    /* Reset KV cache for this segment (match CPU: kv_cache_len = 0) */
    sess_state->current_seq_len = 0;

    /* DecoderPrefill */
    cudaEvent_t prefill_start, prefill_stop;
    cudaEventCreate(&prefill_start);
    cudaEventCreate(&prefill_stop);
    cudaEventRecord(prefill_start);
    status = cuda_backend->DecoderPrefill(sess_state,
                                           static_cast<float *>(sess_state->enc_output.data()),
                                           enc_seq_len,
                                           tokens.data(), total_tokens);
    if (!status.ok()) {
        cudaEventDestroy(prefill_start);
        cudaEventDestroy(prefill_stop);
        qwen_tokenizer_free(tokenizer);
        if (force_prompt_tokens) std::free(force_prompt_tokens);
        if (prompt_tokens) std::free(prompt_tokens);
        return "";
    }
    cudaEventRecord(prefill_stop);
    cudaEventSynchronize(prefill_stop);
    float prefill_ms = 0;
    cudaEventElapsedTime(&prefill_ms, prefill_start, prefill_stop);
    cudaEventDestroy(prefill_start);
    cudaEventDestroy(prefill_stop);

    /* Autoregressive decode loop */
    sess_state->prev_token = tokens[total_tokens - 1];
    std::string output_text;
    float seg_sec = (float)seg_sample_count / (float)QASR_AUDIO_SAMPLE_RATE;
    int max_tokens = std::max(256, std::min(2048, (int)(seg_sec * 15.0f)));
    int past_asr_text = (n_force_prompt_tokens > 0) ? 1 : 0;

    cudaEvent_t dec_start, dec_stop;
    cudaEventCreate(&dec_start);
    cudaEventCreate(&dec_stop);
    float decode_ms = 0;
    cudaEventRecord(dec_start);

    for (int n_gen = 0; n_gen < max_tokens; n_gen++) {
        std::int32_t token = 0;
        status = cuda_backend->DecodeStep(sess_state, token);
        if (!status.ok()) break;
        if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;
        if (token == QWEN_TOKEN_ASR_TEXT) {
            past_asr_text = 1;
        } else if (past_asr_text) {
            const char *piece = qwen_tokenizer_decode(tokenizer, token);
            if (piece) output_text += piece;
        }
        if (out_n_tokens) (*out_n_tokens)++;
    }

    cudaEventRecord(dec_stop);
    cudaEventSynchronize(dec_stop);
    cudaEventElapsedTime(&decode_ms, dec_start, dec_stop);
    cudaEventDestroy(dec_start);
    cudaEventDestroy(dec_stop);

    qwen_tokenizer_free(tokenizer);
    if (force_prompt_tokens) std::free(force_prompt_tokens);
    if (prompt_tokens) std::free(prompt_tokens);

    /* Timing breakdown */
    if (config.verbosity >= 1) {
        fprintf(stderr, "  enc=%.1fms prefill=%.1fms decode=%.1fms tokens=%d\n",
                enc_ms, prefill_ms, decode_ms, out_n_tokens ? *out_n_tokens : 0);
    }

    return output_text;
}

/* Insert a space between two adjacent segment results when needed. */
static bool should_insert_boundary_space(int prev_ch, int next_ch) {
    if (prev_ch < 0 || next_ch < 0) return false;
    if (std::isspace((unsigned char)prev_ch) || std::isspace((unsigned char)next_ch))
        return false;
    if (std::isalnum((unsigned char)prev_ch) && std::isalnum((unsigned char)next_ch))
        return false;
    return true;
}
#endif

Status CudaAsrEngine::LoadModel(const V2EngineConfig & config) {
    config_ = config;

    cuda_backend_ = std::make_shared<CudaBackend>();
    auto status = cuda_backend_->Initialize();
    if (!status.ok()) {
        fprintf(stderr, "CUDA init failed, falling back to CPU: %s\n", status.message().c_str());
        if (!config_.allow_backend_fallback) {
            return status;
        }
        cuda_backend_.reset();
        cpu_fallback_ = std::make_shared<CpuBackend>();
        return cpu_fallback_->PrepareWeights(config_.model_dir);
    }

    model_ = std::make_shared<QwenModel>();
    status = cuda_backend_->PrepareWeights(config_.model_dir);
    if (!status.ok()) {
        fprintf(stderr, "CUDA prepare failed, falling back to CPU: %s\n", status.message().c_str());
        if (!config_.allow_backend_fallback) {
            return status;
        }
        cuda_backend_.reset();
        cpu_fallback_ = std::make_shared<CpuBackend>();
        return cpu_fallback_->PrepareWeights(config_.model_dir);
    }

    return OkStatus();
}

Status CudaAsrEngine::CreateSession(const SessionOptions & opts, std::uint64_t & out_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int>(sessions_.size()) >= config_.max_sessions) {
        return Status(StatusCode::kResourceExhausted,
                      "max sessions reached: " + std::to_string(config_.max_sessions));
    }
    out_id = next_session_id_++;
    auto session = std::make_shared<QwenSession>();
    session->session_id = out_id;
    session->model = model_;
    session->realtime = opts.realtime;
    session->language = opts.language;
    session->prompt = opts.prompt;
    session->priority = 0;

    /* Initialize session backend (CPU fallback or CUDA per-session state) */
    if (cpu_fallback_) {
        session->backend = std::make_unique<CpuBackend>();
    } else if (cuda_backend_) {
        session->backend = std::make_unique<CudaBackend>();
    } else {
        session->backend = std::make_unique<CpuBackend>();
    }

    sessions_[out_id] = session;
    return OkStatus();
}

Status CudaAsrEngine::CloseSession(std::uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        sessions_.erase(it);
    }
    return OkStatus();
}

AsrSegmentResult CudaAsrEngine::TranscribeSegment(std::uint64_t session_id,
                                                   const std::vector<float> & samples,
                                                   int64_t sample_rate,
                                                   TokenCallback on_token,
                                                   CancelCallback on_cancel) {
    (void)sample_rate;
    (void)on_token;
    (void)on_cancel;
    AsrSegmentResult result;

    /* Find session */
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        result.status = Status(StatusCode::kNotFound, "session not found: " + std::to_string(session_id));
        return result;
    }
    auto * session = it->second.get();
    if (!session) {
        result.status = Status(StatusCode::kFailedPrecondition, "session is null");
        return result;
    }

    if (cpu_fallback_) {
        /* Pure CPU fallback path */
        auto * base = reinterpret_cast<qwen_ctx_t *>(cpu_fallback_->base_ctx());
        if (!base) {
            result.status = Status(StatusCode::kFailedPrecondition, "CPU fallback model not loaded");
            return result;
        }

        qwen_set_force_language(base, session->language.empty() ? nullptr : session->language.c_str());
        qwen_set_prompt(base, session->prompt.empty() ? nullptr : session->prompt.c_str());
        if (config_.temperature >= 0.0f) {
            base->decode_temperature = config_.temperature;
        }

        char * raw = qwen_transcribe_audio(base, samples.data(),
                                              static_cast<int>(samples.size()));
        if (!raw) {
            result.status = Status(StatusCode::kInternal, "transcription failed");
            return result;
        }

        result.text = raw;
        std::free(raw);
        result.total_ms = base->perf_total_ms;
        result.audio_ms = base->perf_audio_ms;
        result.text_tokens = base->perf_text_tokens;
        result.encode_ms = base->perf_encode_ms;
        result.decode_ms = base->perf_decode_ms;
        result.status = OkStatus();
        return result;
    }

 #ifdef QASR_CUDA_BACKEND_ENABLED
    if (!cuda_backend_ || !cuda_backend_->cuda_weights() ||
        !cuda_backend_->cuda_weights()->decoder_ready) {
        result.status = Status(StatusCode::kFailedPrecondition, "no backend available");
        return result;
    }

    /* === GPU PIPELINE: multi-segment === */
    auto t0 = std::chrono::steady_clock::now();
    auto * cuda_backend = cuda_backend_.get();

    /* Allocate session state if needed */
    if (!session->backend_workspace) {
        auto sess_state = std::make_unique<CudaSessionState>();
        auto alloc_status = cuda_backend->AllocateSession(sess_state.get(), 4096);
        if (!alloc_status.ok()) {
            result.status = alloc_status;
            return result;
        }
        session->backend_workspace = sess_state.release();
    }
    auto * sess_state = static_cast<CudaSessionState *>(session->backend_workspace);

    /* Step 1: Silence compaction (CPU) */
    int compacted_count = 0;
    float * compacted = qasr_compact_silence(samples.data(),
                                               static_cast<int>(samples.size()),
                                               &compacted_count);
    if (compacted) {
        /* compaction succeeded */
    } else {
        compacted = const_cast<float *>(samples.data());
        compacted_count = static_cast<int>(samples.size());
    }

    /* Step 2: Determine if splitting is needed */
    int n_samples = compacted_count;
    float effective_segment_sec = 0.0f;
    if (n_samples > 30 * QASR_AUDIO_SAMPLE_RATE) {
        effective_segment_sec = 30.0f;
    }

    int target_samples = (int)(effective_segment_sec * QASR_AUDIO_SAMPLE_RATE);
    float search_sec = effective_segment_sec / 2.0f;
    int margin_samples = (int)(search_sec * QASR_AUDIO_SAMPLE_RATE);

    /* Single segment path: no splitting or audio fits in one segment */
    if (effective_segment_sec <= 0 ||
        n_samples <= target_samples + margin_samples) {
        int min_samples = QASR_AUDIO_SAMPLE_RATE / 2;
        float * seg_buf = compacted;
        bool owns_buf = false;

        if (n_samples < min_samples) {
            seg_buf = (float *)calloc(min_samples, sizeof(float));
            memcpy(seg_buf, compacted, (size_t)n_samples * sizeof(float));
            owns_buf = true;
        }

        int seg_token_count = 0;
        std::string text = transcribe_segment_gpu(cuda_backend, sess_state, config_,
                                                   session->language, session->prompt,
                                                   seg_buf, owns_buf ? min_samples : n_samples,
                                                   &seg_token_count);

        if (owns_buf) std::free(seg_buf);
        if (compacted != samples.data()) std::free(compacted);

        auto t1 = std::chrono::steady_clock::now();
        result.text = text;
        result.text_tokens = seg_token_count;
        result.audio_ms = 1000.0 * (double)samples.size() / (double)QASR_AUDIO_SAMPLE_RATE;
        result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.status = OkStatus();
        return result;
    }

    /* Step 3: Multi-segment path */
    /* Build split points */
    int splits[128];
    int n_splits = 0;
    splits[n_splits++] = 0;

    int split_pos = 0;
    while (split_pos + target_samples + margin_samples < n_samples) {
        int split = qasr_find_split_point(compacted, n_samples,
                                           split_pos + target_samples, search_sec);
        splits[n_splits++] = split;
        split_pos = split;
        if (n_splits >= 127) break;
    }
    splits[n_splits] = n_samples;

    if (config_.verbosity >= 2)
        fprintf(stderr, "CUDA: splitting into %d segments\n", n_splits);

    /* Transcribe each segment and concatenate */
    std::ostringstream result_text;
    int min_samples = QASR_AUDIO_SAMPLE_RATE / 2;
    int total_tokens = 0;

    for (int s = 0; s < n_splits; s++) {
        int seg_start = splits[s];
        int seg_end = splits[s + 1];
        int seg_samples_count = seg_end - seg_start;

        if (config_.verbosity >= 2)
            fprintf(stderr, "CUDA segment %d/%d: %.1f-%.1fs (%d samples)\n",
                    s + 1, n_splits,
                    (float)seg_start / QASR_AUDIO_SAMPLE_RATE,
                    (float)seg_end / QASR_AUDIO_SAMPLE_RATE,
                    seg_samples_count);

        /* Pad short segments */
        float * seg_buf = nullptr;
        const float * seg_ptr = compacted + seg_start;
        int use_count = seg_samples_count;

        if (seg_samples_count < min_samples) {
            seg_buf = (float *)calloc(min_samples, sizeof(float));
            memcpy(seg_buf, compacted + seg_start,
                   (size_t)seg_samples_count * sizeof(float));
            seg_ptr = seg_buf;
            use_count = min_samples;
        }

      int seg_n_tokens = 0;
        std::string seg_text = transcribe_segment_gpu(cuda_backend, sess_state,
                                                        config_, session->language,
                                                        session->prompt, seg_ptr, use_count,
                                                        &seg_n_tokens);

        if (seg_buf) std::free(seg_buf);
        if (seg_text.empty()) continue;

        /* Trim leading whitespace for boundary cleanup */
        size_t cut = 0;
        while (seg_text[cut] && std::isspace((unsigned char)seg_text[cut])) cut++;
        if (seg_text[cut] == '\0') continue;

        /* Insert boundary space if needed */
        std::string accumulated = result_text.str();
        if (!accumulated.empty() &&
            should_insert_boundary_space((int)(unsigned char)accumulated.back(),
                                          (int)(unsigned char)seg_text[cut])) {
            result_text << ' ';
        }
        result_text << (seg_text.data() + cut);
        total_tokens += seg_n_tokens;
    }

    if (compacted != samples.data()) std::free(compacted);

    auto t1 = std::chrono::steady_clock::now();
    result.text = result_text.str();
    result.text_tokens = total_tokens;
    result.audio_ms = 1000.0 * (double)samples.size() / (double)QASR_AUDIO_SAMPLE_RATE;
    result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.status = OkStatus();
    return result;
#else
    result.status = Status(StatusCode::kUnimplemented,
                           "CUDA not available — use CPU backend or rebuild with -DQASR_ENABLE_CUDA_BACKEND=ON");
    return result;
#endif
}

int CudaAsrEngine::ActiveSessionCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(sessions_.size());
}

std::unique_ptr<SessionHandle> CudaAsrEngine::CreateRealtimeSession(
    const SessionOptions & opts) {
    (void)opts;
    // CUDA realtime session: nativeCtx() returns nullptr, inference goes
    // through AsrEngine::TranscribeSegment.
    return nullptr;
}

void CudaAsrEngine::CloseSessionHandle(std::unique_ptr<SessionHandle>) {
    // No-op for CUDA path (no native ctx to free)
}

void *CudaAsrEngine::getVadHandle() const {
    // VAD is CPU-only; CUDA backend doesn't own a VAD handle.
    return nullptr;
}

}  // namespace qasr
