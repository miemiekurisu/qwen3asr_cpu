# 二参考剖析

核验日期：2026-04-10。

官方外部来源：

- Qwen 官方模型卡：<https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- vLLM Qwen3-ASR recipe：<https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html>
- OpenAI Speech-to-Text 指南：<https://platform.openai.com/docs/guides/speech-to-text>
- OpenAI Realtime Transcription 指南：<https://platform.openai.com/docs/guides/realtime-transcription>

本地参考：

- `qwen-asr-learn/`
- `whisper.cpp/`

## 总断

结论仅二句：

1. `qwen-asr-learn` 强于“模型正确性与 CPU 算子路径”；其音频前端、提示模板、编码器窗口、解码器前缀回滚、流式提交逻辑，皆可作一等算法参考。
2. `whisper.cpp` 强于“工程化外壳”；其 ABI、context/state 分离、CMake、examples、server、stream、tests、跨平台构建法，皆可作一等框架参考。

故新工程之策：

- 算法面，取 `qwen-asr-learn` 之思路，然以 C++ 重写，不改其源。
- 工程面，取 `whisper.cpp` 之组织法，然不引其模型实现以替 Qwen3-ASR。

## `qwen-asr-learn` 可用处

### 文件级

| 文件 | 角色 | 可采度 | 评语 |
|---|---|---:|---|
| `qwen_asr_audio.c` | WAV 读入、重采样、mel、stdin live buffer | 高 | CPU 首务，且无外依赖 |
| `qwen_asr_safetensors.c` | safetensors 多分片 mmap | 高 | 生产必需；后续应补 checksum、版本校验、页对齐与异常路径 |
| `qwen_asr_tokenizer.c` | Qwen2/Qwen3 BPE | 高 | 可作 tokenizer 正确性参考 |
| `qwen_asr_encoder.c` | 音频编码器权重装载与前向 | 高 | 模型结构关键信息齐 |
| `qwen_asr_decoder.c` | Qwen3 解码器、KV cache、prefill/step | 高 | 流式 CPU 引擎核心参考 |
| `qwen_asr_kernels.c` | BLAS、SIMD、线程池、注意力、norm、conv | 高 | 首批 CPU 优化基线 |
| `qwen_asr.c` | prompt 装配、分段、流式提交、去重、恢复 | 极高 | 当前最有价值之“完整推理流程”参考 |
| `main.c` | CLI | 中 | 可借参数面，不足以作生产服务 |
| `README.md` / `MODEL.md` | 架构说明 | 高 | 已足以反推模型 |
| `asr_regression.py` | 回归脚本 | 中高 | 可借判分思路，后续须改为本项目制式 |

### 函数级

#### 音频层

| 函数 | 义 |
|---|---|
| `qwen_load_wav` | 读 WAV，转单声道 float |
| `qwen_parse_wav_buffer` | 从内存块解析 WAV |
| `qwen_read_pcm_stdin` | stdin 读 WAV 或 raw s16le |
| `qwen_mel_spectrogram` | 产 128-bin log-mel |
| `qwen_live_audio_start_stdin` | 起 reader thread，供实时流式 |
| `qwen_live_audio_free` | 回收 live buffer |

#### 模型装载与推理主线

| 函数 | 义 |
|---|---|
| `qwen_load` | 开 safetensors，多分片 mmap，识别模型规格，装 encoder/decoder |
| `qwen_free` | 回收上下文与缓存 |
| `qwen_set_prompt` | 设 system prompt |
| `qwen_set_force_language` | 强制语言 |
| `qwen_transcribe` | 文件离线转写 |
| `qwen_transcribe_audio` | PCM 离线转写，含静音裁剪、分段、条件重试 |
| `qwen_transcribe_stdin` | stdin 离线 |
| `qwen_transcribe_stream` | 文件流式 |
| `qwen_transcribe_stream_live` | live buffer 流式 |

#### 编码器/解码器

| 函数 | 义 |
|---|---|
| `qwen_encoder_load` | 装 encoder 权重 |
| `qwen_encoder_forward` | Conv2D -> chunk PE -> window attention -> projector |
| `qwen_decoder_load` | 装 decoder 权重，融合 gate/up |
| `qwen_decoder_prefill` | 多 token 预填充，写 KV |
| `qwen_decoder_forward` | 单 token step decode |

#### 算子层

| 族 | 义 |
|---|---|
| `qwen_linear*`, `qwen_matmul_t*` | GEMM / BF16 matvec |
| `qwen_conv2d` | Conv stem |
| `qwen_layer_norm`, `qwen_rms_norm*` | 归一化 |
| `qwen_bidirectional_attention` | encoder attention |
| `qwen_causal_attention` | decoder attention |
| `qwen_compute_rope_neox`, `qwen_apply_rope_neox` | RoPE |
| `qwen_set_threads`, `qwen_get_num_cpus` | 线程池与核数 |

### 工程缺口

| 面 | 现状 | 缺口 |
|---|---|---|
| 语言 | 纯 C | 可移植，然模块边界弱，异常管理弱 |
| 构建 | Makefile | 不足以管 Windows；Linux 仅假定 OpenBLAS 路径 |
| 服务 | 无 | 无异步任务、无 session、无队列、无 backpressure |
| 协议 | CLI | 无 OpenAI/vLLM 兼容 |
| timestamp | 仅文本 | 无官方 ForcedAligner 集成 |
| 观测 | 简单 stderr | 无 metrics、trace、structured log |
| 测试 | 回归脚本 | 无单测、无 sanitizer 策略、无 CI 矩阵 |
| 内存 | 手工 malloc/free | 可控，然缺统一 ownership 规则与 leak gate |

## `whisper.cpp` 可用处

### 模块级

| 模块 | 角色 | 可采度 | 评语 |
|---|---|---:|---|
| `include/whisper.h` | C ABI 与参数面 | 极高 | context/state 分离，回调、segment/token/timestamp API 完整 |
| `src/whisper.cpp` | 核心实现 | 中 | 模型不同，不可直接借算法；可借内存与 state 组织 |
| `examples/cli` | CLI | 高 | 参数体系成熟 |
| `examples/stream` | 实时流式样例 | 高 | 可借 event loop 与音频获取结构 |
| `examples/server` | HTTP 服务样例 | 极高 | 路由、格式、参数、输出格式、多服务面参考价值大 |
| `tests/` | 单测与 ctest 组织 | 高 | 可借测试矩阵与 golden-style 校验 |
| `ggml/` | backend 与跨平台构建 | 中高 | 本项目不必照搬，但其 backend 抽象值得学 |

### API 族

| API 族 | 义 |
|---|---|
| `whisper_init_*`, `whisper_free_*` | context/state 生命周期 |
| `whisper_pcm_to_mel`, `whisper_encode`, `whisper_decode` | 分段推理 API |
| `whisper_full*` | 一体化全流程 API |
| `whisper_full_get_segment_*`, `whisper_full_get_token_*` | segment/token 结果抽取 |
| callbacks | progress/new_segment/encoder_begin/logits_filter |
| `whisper_vad_*` | VAD 独立能力 |
| logging/timings | 系统信息与 profiling |

### 工程优点

| 点 | 义 |
|---|---|
| context/state 分离 | 便于共享模型，多会话并发 |
| examples 完整 | CLI、stream、server 并列，接口自然稳定 |
| CMake 完整 | Windows/Linux/macOS 构建较成熟 |
| tests 接入 CTest | 易入 CI |
| 输出接口完善 | segment、token、timestamp、回调皆齐 |

### 不可直接移植处

| 点 | 因 |
|---|---|
| Whisper 模型结构 | 与 Qwen3-ASR 不同 |
| ggml graph | 引入代价大；本项目先以 CPU 定制核为先 |
| Whisper 时间戳算法 | 仅供工程参考，不可替代 Qwen 官方 ForcedAligner |

## 官方能力与二参考之差

### Qwen 官方明示能力

官方模型卡与仓库明言：

- 同一模型支持 offline / streaming。
- 官方工具包支持 vLLM batch inference、asynchronous serving、streaming inference、timestamp prediction。
- timestamp 主要依赖 `Qwen3-ForcedAligner-0.6B`。

### 二参考未达之处

| 能力 | `qwen-asr-learn` | `whisper.cpp` | 新工程应做 |
|---|---|---|---|
| Qwen3-ASR 正确推理 | 有 | 无 | 取 `qwen-asr-learn` 参考重写 |
| vLLM 兼容 | 无 | 无 | 新作协议层 |
| OpenAI 兼容 | 无 | server 仅 whisper 样式 | 新作协议层 |
| 异步服务 | 无 | 有样例，无生产队列 | 新作 runtime |
| timestamp | 无 | Whisper 自身可出，非 Qwen | 集成 ForcedAligner 路 |
| 多平台生产构建 | 弱 | 强 | 取其法，重立 CMake |

## 取舍表

| 项 | 取否 | 理由 |
|---|---|---|
| Qwen prompt 模板 | 取 | 正确性根基 |
| per-chunk conv 与 encoder 窗口注意力 | 取 | 模型行为关键 |
| prefix rollback 与 stable frontier commit | 取 | 流式关键 |
| 自制 pthread 线程池 | 不直接取 | Windows 不便；新工程改 `std::thread`/executor |
| `whisper.h` 风格 ABI | 取其神 | 公开 API 与 state 分离值得学 |
| `whisper.cpp` server 代码 | 取其路由组织，不抄模型逻辑 | 工程法可借 |
| `ggml` 作为本项目必选依赖 | 暂不取 | 首务 CPU 定制引擎，后再评 |

## 第一阶段结论

第一阶段不做“又一份 C 代码拼装”。应做三件事：

1. 先立 C++ 核心 ABI、状态机、协议层、测试基座。
2. 再把 `qwen-asr-learn` 之算法逐模块迁入：audio -> tokenizer -> safetensors -> encoder -> decoder -> stream。
3. 同时按 `whisper.cpp` 之法立 CLI / stream / server 三面，且始终保单测、集成、回归、文档同进。
