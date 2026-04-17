#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Abstract interface for an ASR inference backend.
///
/// Backends implement model loading, audio encoding, decoder prefill, and
/// autoregressive token generation.  The built-in CPU backend wraps the C
/// kernel library; future backends (GPU, ONNX Runtime, etc.) implement this
/// same interface.
///
/// Thread-safety: NOT thread-safe.  Each session should own its own instance.
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /// Load model weights from a directory.
    /// @param model_dir  Path containing config.json, *.safetensors, vocab, etc.
    /// @param threads    Number of compute threads (0 = auto-detect).
    virtual Status Load(const std::string & model_dir, int threads) = 0;

    /// Run the audio encoder on a mel spectrogram.
    /// @param mel          [mel_bins, mel_frames] input features.
    /// @param mel_frames   Number of time frames.
    /// @param output       Receives [out_seq_len, encoder_output_dim] hidden states.
    /// @param out_seq_len  Receives the resulting sequence length.
    virtual Status Encode(const float * mel, std::int32_t mel_frames,
                          std::vector<float> * output,
                          std::int32_t * out_seq_len) = 0;

    /// Multi-token decoder prefill.
    /// @param embeddings  [seq_len, hidden_dim] input embeddings.
    /// @param seq_len     Number of tokens to prefill.
    virtual Status Prefill(const float * embeddings, std::int32_t seq_len) = 0;

    /// Single-token autoregressive decode step.
    /// @param embed         [1, hidden_dim] embedding for the current token.
    /// @param out_token_id  Receives the greedy-selected token ID.
    virtual Status DecodeStep(const float * embed,
                              std::int32_t * out_token_id) = 0;

    /// Reset decoder state (KV cache) for a new utterance.
    virtual void ResetDecoder() = 0;

    /// @return true once Load() has succeeded.
    virtual bool IsLoaded() const noexcept = 0;

    /// Encoder output dimension (valid after Load).
    virtual std::int32_t EncoderOutputDim() const noexcept = 0;

    /// Decoder hidden dimension (valid after Load).
    virtual std::int32_t DecoderHiddenDim() const noexcept = 0;
};

/// Create the built-in CPU inference backend.
/// Returns nullptr when the CPU backend was not compiled in.
std::unique_ptr<InferenceBackend> CreateCpuBackend();

}  // namespace qasr
