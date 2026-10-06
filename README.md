# qwen3asr_cpu

Qwen3-ASR 的 CPU + GPU 推理服务与命令行工具，使用 C/C++17 实现。提供本地离线转写、字幕输出、HTTP API、内置 Web UI。

支持 Qwen3-ASR 0.6B / 1.7B safetensors 模型：
- **CPU**: OpenBLAS (win/linux) / Accelerate (macOS)
- **GPU (CUDA)**: DGX Spark / GB10 (sm_121)，自定义 CUDA kernel + cuBLAS

**当前版本: v1.0.0**

> ⚠️ 命令行参数以 `qasr_cli --help` / `qasr_server --help` / `qasr_cpu_bench --help` 输出为权威；HTTP API 以 `src/service/server.cc` 的路由注册为权威；环境变量以 `src/backend/qwen_cpu/qwen_asr_perf.c` / `scripts/run_linux_server.sh` 的解析为权威。

## 功能

- **离线转写**: 单文件音频 → 文本 / SRT / VTT / JSON
- **HTTP API**: OpenAI 兼容 `/v1/audio/transcriptions`、Chat `/v1/chat/completions`、异步 `/api/transcriptions/async`
- **实时转写**: WebSocket `/api/realtime`、SSE 流式输出、VAD 分段
- **实时翻译 (POC)**: 每段 ASR 结果自动翻译，集成 MTranServer 侧车服务
- **Web UI**: 内置浏览器界面，支持离线/实时转写、实时翻译、词汇表、导出
- **字幕对齐**: 使用 Qwen3-ForcedAligner 生成词级时间戳
- **流式分段**: 长音频流式推理，分段输出
- **VAD 段式批量**: Silero VAD 自动分段 + 静音检测 + softcap 截止
- **热词/自定义词表**: 离线 + 实时均支持。两条机制可叠加:
  - `--prompt` 上下文提示(官方 Qwen3-ASR context 机制,写进 system prompt)
  - `--hotwords` token 概率偏置(对正在匹配的热词前缀的下一个 token 加 logit 偏置,不改模型)

## 构建

### Linux

```bash
scripts/build_linux.sh                       # 默认 clean + build + test
scripts/build_linux.sh --incremental         # 增量编译
scripts/build_linux.sh --model-dir /data/Qwen3-ASR-0.6B
```

### Linux / DGX Spark — CUDA 后端

```bash
./build_cuda.sh                              # 一键构建 + short 音频测试
./build_cuda.sh --long                       # + long 音频测试
./build_cuda.sh --clean                      # 全量重建
```

CPU/CUDA 输出对比验证：
```bash
./build-dgx/qasr_v2_test <model_dir> <audio.wav> verify
```

### macOS

```bash
brew install cmake ninja ffmpeg
cmake --preset macos-accelerate
cmake --build build/macos-accelerate -j"$(sysctl -n hw.ncpu)"
```

### Windows

```powershell
build_all.ps1                                # 一键 clean + configure + compile
build_all.ps1 --incremental                  # 增量编译
build_all.ps1 --openblas-dir D:\dev\OpenBLAS
```

## 模型

| Model | HuggingFace | ModelScope |
|---|---|---|
| Qwen3-ASR-0.6B | https://huggingface.co/Qwen/Qwen3-ASR-0.6B | https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B |
| Qwen3-ASR-1.7B | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B |
| Qwen3-ForcedAligner-0.6B | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B | https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B |

建议：
- 0.6B：实时、近实时、Web UI、低延迟
- 1.7B：离线批处理、长音频转写、字幕生产

## 快速使用

### Server (一键启动)

```bash
export QASR_MODEL_DIR=$HOME/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/<rev>

scripts/run_linux_server.sh --detach                     # HTTP
scripts/run_linux_server.sh --detach --https             # HTTP + HTTPS (浏览器 mic 需要)
scripts/run_linux_server.sh --detach --https --backend cuda  # GPU 后端
scripts/run_linux_server.sh --status                     # 健康检查
scripts/run_linux_server.sh --stop                       # 停止
```

### 实时翻译 (POC)

Web UI 支持将每段 ASR 结果通过 MTranServer 翻译为目标语言。

**前置条件**: 本地运行 MTranServer 实例（端口默认 8989）。

```bash
# 环境变量配置（可选）
export QASR_TRANSLATION_ENDPOINT=http://127.0.0.1:8989  # 翻译服务地址
export QASR_TRANSLATION_SOURCE_LANG=auto                  # 源语言
export QASR_TRANSLATION_TARGET_LANG=en                    # 目标语言
export QASR_TRANSLATION_TIMEOUT_MS=3000                   # 翻译超时(ms)
```

> ⚠️ 翻译功能为 POC，临时集成 MTranServer 独立服务。后续计划完善翻译管线：
> - 支持更多翻译后端（本地模型、云端 API）
> - 翻译结果缓存与批量处理
> - Windows / Apple Silicon 平台对齐

### CLI 转写

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio meeting.mp3 --language Chinese --threads 8
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 --output-format srt --output movie.srt

# 热词:prompt 上下文方式(官方机制,流式同样生效)
qasr_cli --model-dir ... --audio a.wav --stream --prompt "Zentrix, RayRemote, vLLM"
# 热词:token 概率偏置方式(--hotword-bias 为概率倍率,默认 2.0)
qasr_cli --model-dir ... --audio a.wav --stream --hotwords "Zentrix,RayRemote,vLLM" --hotword-bias 5
```

### HTTP API

OpenAI 兼容：
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions -F file=@audio.wav -F model=qwen3-asr
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-asr","messages":[{"role":"user","content":[{"type":"text","text":"Transcribe."},{"type":"audio_url","audio_url":{"url":"file:///path/audio.wav"}}]}]}'
```

完整 API 端点见 [`docs/API.md`](docs/API.md)。

## 热词 (Hotwords)

两种机制,可同时使用,**离线和实时(流式)都生效**:

| 机制 | 入口 | 原理 | 特点 |
|---|---|---|---|
| prompt 上下文 | `--prompt` / 会话 `prompt`/`instructions` 字段 | 文本注入 chat template 的 system 段(官方 Qwen3-ASR context) | 模型自行理解词汇语义,效果最好;专有名词拼写修正强 |
| logit 偏置 | `--hotwords` CSV / 会话 `hotwords` 字段 + `--hotword-bias` 倍率 | 解码时对"正在匹配中的热词序列"的下一 token 加 `log(倍率)` 偏置(KMP 前缀匹配) | 不依赖模型理解,确定性提升;普通词不会误触发,建议只放稀有词 |

Realtime 会话级用法:

```bash
# POST /api/realtime/start
{"prompt": "Zentrix, RayRemote, vLLM", "hotwords": "Zentrix,RayRemote"}
# POST /v1/realtime (OpenAI 兼容)
{"type":"session.create","session":{"prompt":"...","hotwords":"..."}}
# POST /api/capture/start (本机采集)
{"prompt":"...","hotwords":"..."}
```

经验值(0.6B, Apple Silicon 实测):`--hotword-bias` 2 对"高置信误识别"够用(如 Ray Remote→RayRemote);5~20 才压得动模型确信的读法;即使 20 也不会污染正常语音(机制只抬升热词序列的下一个 token)。混合大小写热词(如 "RayRemote")若 tokenizer 拆开,偏置无法合并输出成连写——此时 prompt 机制更合适。

### 推荐用法:按"词的常见度"选通道

- **常见词、或含常见子词的词 → 放 prompt**(`Pro`、`AI`、`Ray Remote` 这类):prompt 靠语义理解起作用,词越常见模型越认识;而 logit 偏置给常见词全力推会误伤正常语音。
- **冷门词/生造词 → 放 hotwords**(`Zentrix`、`秘塔搜索`、自造品牌名):模型本来就猜不到,偏置把它抬出来,确定性强、不依赖理解。
- 两者可叠加,互不干扰。词表大(几百词)时 hotwords 零开销;prompt 超过几百词会让每段解码多花 ~1.5s prefill 且效果稀释,精简到最重要的几十个。
- 判断一个词"常见还是冷门"的快速办法:看它 tokenize 成几个 token、首 token 编号大不大——单 token 且编号 < ~30000 的基本是常见词(如 `" Pro"=1298`);多 token 或编号 >60000 的偏冷门(`"秘"=99780`)。
- 偏置机制内置保护:单 token 低编号词的首 token 偏置按 `id/60000` 自动衰减(常见词 ≈ 没推),且每个 token 最多只补到刚好超过当前第一名(margin cap)。词表大/脏时建议 bias ≤ 20。

### 热词对延时的影响(0.6B, Apple M4 实测, `--stream`)

| 配置 | decode 总耗时(4.8s 音频) | 说明 |
|---|---|---|
| baseline | 623 ms | — |
| prompt 3 词 / hotwords 3 词 | 647 / 625 ms | 噪声内,无感 |
| prompt 300 词 | 2406 ms(3.9x) | 每段重 prefill ~600 token |
| hotwords 300 词 | 693 ms | +11%,仍零感 |

15s 多句音频同样趋势:prompt 300 词 decode 5.1s(2.9x),hotwords 300 词与 baseline 相同。感知延迟 ≈ 每句话说完到出字的时间,prompt 大词表每段约多 1s——**一般热词表只有几十个词,两种机制在流式下延时都感知不到,可以放心叠用**。

## 配置

### qasr_server

| Flag | 默认 | 说明 |
|---|---|---|
| `--model-dir` | (必填) | ASR 模型目录 |
| `--realtime-model-dir` | 同 `--model-dir` | realtime 模型；空 = 共享 |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8080` | HTTP 端口 |
| `--ui-dir` | `ui` | UI 静态资源 |
| `--threads` | 0=auto | 推理线程 |
| `--temperature` | -1.0=auto | 采样温度 |
| `--prompt` | (空) | realtime/capture 会话默认系统提示词(热词上下文) |
| `--hotwords` | (空) | 热词列表,逗号分隔(logit 偏置机制) |
| `--hotword-bias` | 2.0 | 热词概率倍率 |
| `--verbosity` | 0 | 日志级别 |

### 环境变量

| Env | 默认 | 说明 |
|---|---|---|
| `QASR_MODEL_DIR` | auto | ASR 模型目录 |
| `QASR_REALTIME_MODEL_DIR` | (空) | realtime 模型；空 = 与 batch 共享 |
| `QASR_PORT` | `19991` | HTTP 端口 |
| `QASR_HTTPS_PORT` | `19992` | HTTPS 端口 |
| `QASR_THREADS` | 0=auto | 推理线程 |
| `QASR_VERBOSITY` | 0 | 日志级别 |
| `QASR_VAD_MODEL` | auto | Silero VAD ONNX 模型路径 |
| `OPENBLAS_NUM_THREADS` | 0=auto | OpenBLAS 线程数 |
| `QWEN_RUNTIME_PROFILE` | `balanced` | `balanced` / `realtime` / `offline` / `edge_lowmem` |
| `QWEN_DEC_PREFILL_QKV_PERSIST` | 0 | 1=QKV 权重常驻内存 |
| `QWEN_DEC_PREFILL_QKV_BUDGET_MB` | 512 | QKV 预分配上限 |
| `QWEN_BF16_CACHE_MB` | 0=off | encoder BF16 权重缓存 |
| `QWEN_SILERO_VAD_MODEL` | (空) | VAD ONNX 路径 |

## 性能

### CPU (i7-14700KF, 0.6B)

| 配置 | RTF |
|------|-----|
| OpenBLAS 8 线程, balanced | ~0.3-0.5 |
| OPENBLAS_NUM_THREADS=8, QWEN_RUNTIME_PROFILE=balanced | 推荐 |

### CUDA (DGX Spark / GB10, sm_121)

| 测试 | CPU | CUDA | 加速比 |
|------|-----|------|--------|
| 0.6B short (3s) | 1437 ms | 862 ms | 1.67x |
| 1.7B short (3s) | 2190 ms | 1479 ms | 1.48x |
| 0.6B long (28.8min) | — | 165 s (RTF 10.5x) | ~10x |

## 项目结构

```
app/                    CLI, server, benchmark, v2 test entry points
include/qasr/           Public C++ headers (backend, engine, scheduler)
src/backend/qwen_cpu/   Internal C CPU backend and kernels
src/backend/*.cu        CUDA kernels
src/engine/             V2 engine (CPU + CUDA engine adapters)
src/scheduler/          GPU job scheduler
src/service/            HTTP server and realtime session handling
src/runtime/            Model bridge, tasks, sessions, queues
src/protocol/           OpenAI/vLLM request validation
src/audio/              WAV parsing, resampling, ffmpeg helpers
src/subtitle/           SRT/VTT/JSON subtitle writers
tests/                  Unit and regression tests
ui/                     Browser UI
scripts/                Build, benchmark, and utility scripts
docs/                   API reference
build_cuda.sh           One-click CUDA build & test script
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
