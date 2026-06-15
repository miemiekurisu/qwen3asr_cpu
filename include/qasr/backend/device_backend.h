#pragma once

#include "qasr/engine/types.h"
#include "qasr/engine/config.h"
#include "qasr/core/status.h"
#include <memory>
#include <string>
#include <cstdint>
#include <vector>

namespace qasr {

/* Operator-level virtual functions that each backend must implement.
 * These are the building blocks for the decoder forward pass.
 * CPU backend calls CPU kernels, CUDA backend calls CUDA kernels,
 * MLX backend will call MLX primitives. */
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    /* --- Backend identity & lifecycle --- */
    virtual BackendKind kind() const = 0;
    virtual Status Initialize() = 0;
    virtual Status Shutdown() = 0;
    virtual Status PrepareWeights(const std::string & model_dir) = 0;

    /* --- High-level inference entry points --- */
    virtual Status EncodeMel(void * workspace,
                              const float * mel_features,
                              int mel_frames,
                              float * output,
                              int & out_tokens) = 0;

    virtual Status DecoderPrefill(void * workspace,
                                   const float * encoder_output,
                                   int encoder_tokens,
                                   std::int32_t * input_tokens,
                                   int n_tokens) = 0;

   virtual Status DecodeStep(void * workspace,
                                std::int32_t & out_token) = 0;

    virtual Status ResetDecoder(void * workspace) = 0;
    virtual size_t WorkspaceBytes(const V2EngineConfig & config) const = 0;

    /* --- Operator-level primitives (for fine-grained execution) --- */

    /* RMSNorm: out[seq, hidden] = x[seq, hidden] * rms_scale * weight[hidden]
     * Each row normalized independently. */
    virtual Status RmsNorm(float * out,
                            const float * x,
                            const float * weight,
                            int seq_len,
                            int hidden,
                            float eps) = 0;

    /* Per-head RMSNorm: in-place on x[seq, n_heads * head_dim]
     * Each [head_dim] slice normalized independently with shared weight[head_dim]. */
    virtual Status RmsNormPerHead(float * x,
                                   const float * weight,
                                   int seq_len,
                                   int n_heads,
                                   int head_dim,
                                   float eps) = 0;

    /* NeoX split-half RoPE: in-place rotation of x[seq, n_heads * head_dim]
     * cos_vals[seq, head_dim], sin_vals[seq, head_dim] */
    virtual Status ApplyRoPE(float * x,
                              const float * cos_vals,
                              const float * sin_vals,
                              int seq,
                              int n_heads,
                              int head_dim) = 0;

    /* SwiGLU activation: gate_up[seq, 2*intermediate] -> out[seq, intermediate]
     * out[j] = SiLU(g[2j]) * u[2j+1], interleaved [g0,u0,g1,u1,...] */
    virtual Status SwiGLU(float * out,
                           const float * gate_up,
                           int seq_len,
                           int intermediate) = 0;

    /* Argmax over logits[vocab_size], returns index in *out_idx.
     * Optional *out_val for the max value. */
    virtual Status ArgMax(const float * logits,
                           int vocab_size,
                           int * out_idx,
                           float * out_val = nullptr) = 0;
};

/* Factory: create a CPU backend instance. */
std::unique_ptr<DeviceBackend> CreateCpuBackend();

}  // namespace qasr
