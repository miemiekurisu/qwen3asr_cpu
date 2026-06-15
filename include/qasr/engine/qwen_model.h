#pragma once

#include "qasr/engine/types.h"
#include "qasr/backend/cpu_backend.h"
#include "qasr/backend/cuda_backend.h"
#include <memory>
#include <string>

namespace qasr {

struct ModelConfig {
    std::string model_dir;
    int num_layers = 0;
    int num_heads = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    int mel_dim = 0;
    int encoder_dim = 0;
};

class QwenModel {
public:
    ModelConfig config;
    std::shared_ptr<CpuWeights> cpu_weights;
    std::shared_ptr<CudaWeights> cuda_weights;

    Status LoadCpuWeights(const V2EngineConfig & engine_config);
    Status LoadCudaWeights(const V2EngineConfig & engine_config);
    ModelConfig ParseConfig(const std::string & model_dir);
};

}  // namespace qasr
