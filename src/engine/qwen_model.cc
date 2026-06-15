#include "qasr/engine/qwen_model.h"
#include "qasr/backend/cpu_backend.h"
#include "qasr/backend/cuda_backend.h"

extern "C" {
#include "qwen_asr.h"
}

namespace qasr {

ModelConfig QwenModel::ParseConfig(const std::string & model_dir) {
    ModelConfig cfg;
    cfg.model_dir = model_dir;
    void * ctx = qwen_load(model_dir.c_str());
    if (!ctx) {
        return cfg;
    }
    auto * qwen_ctx = static_cast<qwen_ctx_t *>(ctx);
    cfg.num_layers = static_cast<int>(qwen_ctx->config.dec_layers);
    cfg.num_heads = static_cast<int>(qwen_ctx->config.dec_heads);
    cfg.hidden_size = static_cast<int>(qwen_ctx->config.dec_hidden);
    cfg.intermediate_size = static_cast<int>(qwen_ctx->config.dec_intermediate);
    cfg.vocab_size = static_cast<int>(qwen_ctx->config.vocab_size);
    cfg.max_seq_len = 0;
    cfg.mel_dim = QWEN_MEL_BINS;
    cfg.encoder_dim = static_cast<int>(qwen_ctx->config.enc_d_model);
    qwen_free(qwen_ctx);
    return cfg;
}

Status QwenModel::LoadCpuWeights(const V2EngineConfig & engine_config) {
    cpu_weights = std::make_shared<CpuWeights>();
    cpu_weights->model_dir = engine_config.model_dir;
    cpu_weights->ctx = qwen_load(engine_config.model_dir.c_str());
    if (!cpu_weights->ctx) {
        cpu_weights.reset();
        return Status(StatusCode::kInternal,
                      "qwen_load failed for " + engine_config.model_dir);
    }
    config = ParseConfig(engine_config.model_dir);
    return OkStatus();
}

Status QwenModel::LoadCudaWeights(const V2EngineConfig & engine_config) {
    cuda_weights = std::make_shared<CudaWeights>();
    config = ParseConfig(engine_config.model_dir);
    return Status(StatusCode::kUnimplemented,
                  "CUDA decoder residency not yet implemented (CUDA-2 stage)");
}

}  // namespace qasr
