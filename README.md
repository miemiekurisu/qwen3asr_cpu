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
- **Web UI**: 内置浏览器界面，支持离线/实时转写、词汇表、导出
- **字幕对齐**: 使用 Qwen3-ForcedAligner 生成词级时间戳
- **流式分段**: 长音频流式推理，分段输出
- **VAD 段式批量**: Silero VAD 自动分段 + 静音检测 + softcap 截止

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

### CLI 转写

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio meeting.mp3 --language Chinese --threads 8
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 --output-format srt --output movie.srt
```

### HTTP API

OpenAI 兼容：
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions -F file=@audio.wav -F model=qwen3-asr
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-asr","messages":[{"role":"user","content":[{"type":"text","text":"Transcribe."},{"type":"audio_url","audio_url":{"url":"file:///path/audio.wav"}}]}]}'
```

完整 API 端点见 [`docs/API.md`](docs/API.md)。

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
