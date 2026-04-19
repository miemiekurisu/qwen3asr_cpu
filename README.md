# qwen3asr_cpu

[English](#english) | [中文](#中文)

---

<a name="english"></a>

## English

A high-performance C/C++ inference server for **Qwen3-ASR**, optimized for CPU-only real-time streaming speech recognition. Supports both **0.6B** and **1.7B** models. Features an OpenAI-compatible REST API, WebSocket streaming, and a built-in web UI — no GPU required.

### Usage

> **Windows note:** The executables depend on OpenBLAS at runtime. Either add the OpenBLAS `bin/` directory to your `PATH`, or copy `libopenblas.dll` into the same folder as the executables.
>
> ```powershell
> # Option A: add to PATH (current session)
> $env:PATH = "C:\path\to\OpenBLAS\bin;$env:PATH"
>
> # Option B: copy DLL next to the binary
> copy C:\path\to\OpenBLAS\bin\libopenblas.dll <qasr_cli.exe dir>
> ```

#### CLI — Batch Transcription

`--audio` accepts any format (WAV, MP3, M4A, FLAC, …). Non-WAV files are automatically converted via ffmpeg.

```bash
# Transcribe an audio file to plain text
qasr_cli --model-dir <asr-model-dir> --audio <input-audio>

# Output as SRT subtitles (requires both ASR model + ForcedAligner model)
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir>

# Write to a specific output file
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir> \
  --output output.srt

# Use INT8 decoder for faster decoding
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir> \
  --decoder-int8 --threads 16
```

#### Server — Web UI & API

```bash
# Start the server (opens web UI at http://localhost:8080)
qasr_server --model-dir <asr-model-dir> --port 8080 --threads 8

# With INT8 acceleration
qasr_server --model-dir <asr-model-dir> --port 8080 --decoder-int8 --threads 16

# Specify the built-in web UI directory (auto-detected if ui/ exists next to the binary)
qasr_server --model-dir <asr-model-dir> --port 8080 --ui-dir <ui-dir>
```

The web UI supports real-time microphone streaming and file upload transcription in the browser.

#### Subtitle Generation with ForcedAligner

For word-level subtitle timestamps, use `--align` with a ForcedAligner model alongside the ASR model.

**Download both models:**

| Model | Link |
|-------|------|
| Qwen3-ASR-0.6B (or 1.7B) | [HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ForcedAligner-0.6B | [HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B) |

```bash
qasr_cli \
  --model-dir <asr-model-dir> \
  --audio movie.mp3 \
  --output-format srt \
  --align \
  --aligner-model-dir <aligner-model-dir> \
  --output movie.srt
```

> **Tip:** Use `--output-format vtt` for WebVTT, or `--output-format json` for structured JSON with timestamps.

### Features

- **Real-time Streaming ASR**: Chunk-based incremental decoding with stable/partial text output via WebSocket
- **OpenAI-compatible API**: Drop-in replacement for `/v1/audio/transcriptions` and Realtime Sessions
- **vLLM-compatible API**: `/v1/chat/completions` endpoint for batch transcription
- **INT8 Decoder**: All decoder projections quantized to u8×s8→f32 via oneDNN (~30% faster decode)
- **GEMM-based Attention**: Batched `cblas_sgemm` attention replacing per-head dot-product loops (~2× attention speedup)
- **AVX2 SIMD Kernels**: Vectorized RMS norm, SiLU, element-wise operations on x86-64
- **KV Cache Shift**: Efficient context window management for long audio streams
- **Arena Memory Allocation**: Pre-allocated inference buffers, zero malloc in hot path
- **Safetensors Direct Loading**: Load HuggingFace model weights directly — no conversion step
- **Built-in Web UI**: Real-time microphone streaming and offline file upload in the browser
- **Docker Ready**: Multi-stage Dockerfile for one-command deployment
- **Cross-platform**: Windows (MSVC), Linux (GCC/Clang), macOS (Accelerate)
- **Pure C17/C++17**: No Python runtime required

### Supported Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| Qwen3-ASR-0.6B ([HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) \| [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)) | ~0.6B | **Recommended for realtime streaming** — fast, low memory |
| Qwen3-ASR-1.7B ([HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) \| [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)) | ~1.7B | Higher accuracy, better for offline batch transcription |

### Performance

Benchmark on 109-second Mandarin audio, Intel 16-thread CPU (Windows, OpenBLAS 0.3.28 + oneDNN 3.7):

| Stage | Time |
|-------|------|
| Audio encoding (Whisper encoder) | ~6,300 ms |
| Text decoding — prefill (28 layers × INT8) | ~31,200 ms |
| Text decoding — per-round steady state | 3.5–4.2 s |
| **Real-time Factor (RTF)** | **< 1.0** |

**Streaming Latency**: ~3.5–4.2 seconds per 0.5s audio chunk (coalesced to ~4s windows), delivering partial results in real time.

#### Key Optimizations

- **INT8 Decoder (oneDNN)**: u8×s8→f32 quantized matmul for all 6 projection types × 28 layers. Online calibration with per-channel scales.
- **GEMM-based Attention**: Replaces per-head scalar dot-product with batched `cblas_sgemm` calls for Q×K and Score×V. ~2× speedup on prefill attention.
- **KV Cache Shift**: On context window overflow, shifts the KV cache by evicting the oldest entries instead of recomputing. Enables unbounded audio streams.
- **Arena Pre-allocation**: Attention score buffers, intermediate tensors allocated once from a contiguous arena. Eliminates per-layer malloc/free overhead.
- **Custom Thread Pool**: `parallel_for` with configurable thread count. OpenBLAS thread count synced to avoid oversubscription.
- **Selective Logits**: Only the last token's logits are computed for the LM head during both prefill and decode.

### Dependencies

| Dependency | Version | Required | How to get | Notes |
|-----------|---------|----------|-----------|-------|
| CMake | 3.21+ | Yes | [cmake.org](https://cmake.org/download/) | Ninja generator recommended |
| C/C++ Compiler | C17+C++17 | Yes | MSVC 2022+ / GCC 8+ / Clang 7+ | |
| Ninja | 1.10+ | Yes | [ninja-build.org](https://ninja-build.org/) | Ships with VS 2022 |
| OpenBLAS | 0.3.x | Yes* | [github.com/OpenMathLib/OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/releases) | *macOS uses Accelerate instead |
| oneDNN | 3.x | No | Auto-downloaded by CMake | Enables INT8 decoder (~30% speedup) |
| Model | — | Yes | See [Supported Models](#supported-models) | safetensors format; 0.6B ~1.2 GB, 1.7B ~3.4 GB |

> **oneDNN** is automatically downloaded and compiled from source during CMake configure if not found on the system. No manual installation needed. Set `QASR_ENABLE_ONEDNN=OFF` to disable.

#### Installing OpenBLAS

**Windows**: Download the prebuilt release from [OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases) (e.g. `OpenBLAS-0.3.28-x64.zip`). Extract it and set `OPENBLAS_DIR` or `OPENBLAS_ROOT` to the extracted directory.

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install libopenblas-dev
```

**macOS**: No installation needed — Apple Accelerate is used automatically.

### Building

#### Windows (MSVC + OpenBLAS)

Option A — Use the build script (recommended, auto-detects Visual Studio and OpenBLAS):

```powershell
.\tools\build_windows_openblas.ps1 -OpenBlasDir "D:\dev\OpenBLAS"
```

Option B — Manual CMake:

```powershell
# From Visual Studio Developer PowerShell
cmake --preset windows-openblas -DOpenBLAS_DIR="D:\dev\OpenBLAS\lib\cmake\openblas"
cmake --build build/windows-openblas
```

> The build script (`tools/build_windows_openblas.ps1`) automatically finds Visual Studio, sets up the environment, locates OpenBLAS, builds, and runs tests. Run with `-SkipTests` to skip testing.

#### Linux (GCC + OpenBLAS)

```bash
sudo apt-get install build-essential cmake ninja-build libopenblas-dev
cmake --preset linux-openblas
cmake --build build/linux-openblas -j$(nproc)
```

#### macOS (Clang + Accelerate)

```bash
cmake --preset macos-accelerate
cmake --build build/macos-accelerate -j$(sysctl -n hw.ncpu)
```

#### Docker

```bash
docker build -t qasr .
docker run -p 8080:8080 -v /path/to/Qwen3-ASR-0.6B:/models/qwen3-asr qasr
```

### API Endpoints

#### OpenAI-compatible Transcription API

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

#### Real-time WebSocket Streaming

Connect to `ws://localhost:8080/v1/realtime` and send base64-encoded PCM16LE audio chunks. The server returns incremental transcription events:

```json
{"type": "response.audio_transcript.delta", "delta": "你好"}
{"type": "response.audio_transcript.done", "transcript": "你好世界"}
```

### Project Structure

```
qwen3asr_cpu/
├── app/
│   ├── qasr_cli.cc              # CLI entry point
│   └── qasr_server.cc           # Server entry point
├── include/qasr/
│   ├── core/                    # Status, audio types, timestamps, state machine
│   ├── inference/               # Encoder, decoder, streaming policy interfaces
│   ├── protocol/                # OpenAI & vLLM API validation
│   ├── runtime/                 # Engine, config, session management, task queue
│   └── service/                 # HTTP server, realtime WebSocket service
├── src/
│   ├── backend/qwen_cpu/        # Pure C CPU inference backend
│   │   ├── qwen_asr.c           # Main model lifecycle & streaming strategy
│   │   ├── qwen_asr_decoder.c   # Decoder prefill/decode with INT8 + GEMM attention
│   │   ├── qwen_asr_encoder.c   # Whisper-style audio encoder
│   │   ├── qwen_asr_kernels.c   # BLAS-accelerated attention, matmul dispatch
│   │   ├── qwen_asr_kernels_avx.c  # AVX2 SIMD kernels
│   │   └── qwen_asr_onednn.c    # oneDNN INT8 matmul wrapper
│   └── ...                      # Service layer (C++)
├── tests/                       # 438 unit tests
├── ui/                          # Built-in web UI (HTML/CSS/JS)
├── Dockerfile                   # Multi-stage Docker build
└── CMakeLists.txt
```

### Supported Languages

Qwen3-ASR-0.6B supports 30+ languages including:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese (Mandarin) | zh | English | en |
| Cantonese | yue | Japanese | ja |
| Korean | ko | German | de |
| French | fr | Spanish | es |
| Italian | it | Portuguese | pt |
| Russian | ru | Arabic | ar |
| Hindi | hi | Thai | th |
| Vietnamese | vi | Indonesian | id |

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Acknowledgments

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — Original model by Alibaba Cloud
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — Optimized BLAS library
- [oneDNN](https://github.com/oneapi-src/oneDNN) — Intel Deep Neural Network Library

---

<a name="中文"></a>

## 中文

高性能 C/C++ 实现的 **Qwen3-ASR** 推理服务器，专为纯 CPU 实时流式语音识别优化。支持 **0.6B** 和 **1.7B** 两种模型。内置 OpenAI 兼容 REST API、WebSocket 流式接口和 Web UI — 无需 GPU。

### 使用方法

> **Windows 提示：** 可执行文件运行时依赖 OpenBLAS。请将 OpenBLAS `bin/` 目录添加到 `PATH`，或将 `libopenblas.dll` 复制到可执行文件同目录下。
>
> ```powershell
> # 方式 A：添加到 PATH（当前会话）
> $env:PATH = "C:\path\to\OpenBLAS\bin;$env:PATH"
>
> # 方式 B：复制 DLL 到程序目录
> copy C:\path\to\OpenBLAS\bin\libopenblas.dll <qasr_cli.exe 所在目录>
> ```

#### CLI — 批量转写

`--audio` 接受任意格式（WAV、MP3、M4A、FLAC……），非 WAV 文件会自动通过 ffmpeg 转换。

```bash
# 转写音频为纯文本
qasr_cli --model-dir <asr-model-dir> --audio <input-audio>

# 输出 SRT 字幕（需要 ASR 模型 + ForcedAligner 模型）
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir>

# 写入指定输出文件
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir> \
  --output output.srt

# 使用 INT8 解码器加速
qasr_cli --model-dir <asr-model-dir> --audio <input-audio> \
  --output-format srt --align --aligner-model-dir <aligner-model-dir> \
  --decoder-int8 --threads 16
```

#### 服务器 — Web UI 与 API

```bash
# 启动服务器（Web UI 地址 http://localhost:8080）
qasr_server --model-dir <asr-model-dir> --port 8080 --threads 8

# 启用 INT8 加速
qasr_server --model-dir <asr-model-dir> --port 8080 --decoder-int8 --threads 16

# 指定内置 Web UI 目录（如果 ui/ 在程序同目录下会自动检测）
qasr_server --model-dir <asr-model-dir> --port 8080 --ui-dir <ui-dir>
```

Web UI 支持浏览器内实时麦克风流式识别和文件上传转写。

#### 使用 ForcedAligner 生成字幕

如需词级字幕时间戳，请配合 ForcedAligner 模型使用 `--align` 选项。

**下载两个模型：**

| 模型 | 链接 |
|------|------|
| Qwen3-ASR-0.6B (或 1.7B) | [HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ForcedAligner-0.6B | [HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B) |

```bash
qasr_cli \
  --model-dir <asr-model-dir> \
  --audio movie.mp3 \
  --output-format srt \
  --align \
  --aligner-model-dir <aligner-model-dir> \
  --output movie.srt
```

> **提示：** 使用 `--output-format vtt` 输出 WebVTT 格式，或 `--output-format json` 输出带时间戳的结构化 JSON。

### 特性

- **实时流式 ASR**：基于分块的增量解码，通过 WebSocket 输出稳定/部分文本
- **OpenAI 兼容 API**：支持 `/v1/audio/transcriptions` 和 Realtime Sessions 接口
- **vLLM 兼容 API**：`/v1/chat/completions` 端点用于批量转写
- **INT8 解码器**：通过 oneDNN 将所有解码器投影量化为 u8×s8→f32（解码速度提升 ~30%）
- **GEMM 注意力**：使用批量 `cblas_sgemm` 替代逐头点积循环（注意力速度提升 ~2 倍）
- **AVX2 SIMD 内核**：向量化的 RMS norm、SiLU、逐元素运算（x86-64）
- **KV Cache 滑动**：长音频流的高效上下文窗口管理
- **Arena 内存预分配**：预分配推理缓冲区，热路径零 malloc
- **Safetensors 直接加载**：直接加载 HuggingFace 模型权重，无需转换
- **内置 Web UI**：浏览器内实时麦克风流式和离线文件上传
- **Docker 就绪**：多阶段 Dockerfile，一键部署
- **跨平台**：Windows (MSVC)、Linux (GCC/Clang)、macOS (Accelerate)
- **纯 C17/C++17**：不依赖 Python 运行时

### 支持的模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| Qwen3-ASR-0.6B ([HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) \| [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)) | ~6 亿 | **推荐用于实时流式** — 速度快、内存占用低 |
| Qwen3-ASR-1.7B ([HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) \| [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)) | ~17 亿 | 精度更高，适合离线批量转写 |

### 性能

基准测试：109 秒普通话音频，Intel 16 线程 CPU (Windows, OpenBLAS 0.3.28 + oneDNN 3.7)：

| 阶段 | 耗时 |
|------|------|
| 音频编码 (Whisper encoder) | ~6,300 ms |
| 文本解码 — prefill (28 层 × INT8) | ~31,200 ms |
| 文本解码 — 稳态每轮 | 3.5–4.2 s |
| **实时率 (RTF)** | **< 1.0** |

**流式延迟**：每 0.5 秒音频块约 3.5–4.2 秒处理（合并为 ~4 秒窗口），实时输出部分结果。

#### 关键优化

- **INT8 解码器 (oneDNN)**：所有 6 种投影类型 × 28 层使用 u8×s8→f32 量化矩阵乘法，在线校准逐通道缩放
- **GEMM 注意力**：用批量 `cblas_sgemm` 替代逐头标量点积计算 Q×K 和 Score×V，prefill 注意力提速 ~2 倍
- **KV Cache 滑动**：上下文窗口溢出时滑动 KV Cache 驱逐最旧条目，而非重新计算，支持无限长音频流
- **Arena 预分配**：注意力分数缓冲区、中间张量从连续内存池一次分配，消除逐层 malloc/free 开销
- **自定义线程池**：`parallel_for` 可配置线程数，与 OpenBLAS 线程数同步避免过度订阅
- **选择性 Logits**：prefill 和 decode 阶段仅计算最后一个 token 的 LM head logits

### 依赖

| 依赖 | 版本 | 必需 | 获取方式 | 备注 |
|------|------|------|---------|------|
| CMake | 3.21+ | 是 | [cmake.org](https://cmake.org/download/) | 推荐使用 Ninja 生成器 |
| C/C++ 编译器 | C17+C++17 | 是 | MSVC 2022+ / GCC 8+ / Clang 7+ | |
| Ninja | 1.10+ | 是 | [ninja-build.org](https://ninja-build.org/) | VS 2022 自带 |
| OpenBLAS | 0.3.x | 是* | [github.com/OpenMathLib/OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/releases) | *macOS 使用 Accelerate 替代 |
| oneDNN | 3.x | 否 | CMake 自动下载 | 启用 INT8 解码器（提速 ~30%） |
| 模型 | — | 是 | 见[支持的模型](#支持的模型) | safetensors 格式；0.6B ~1.2 GB，1.7B ~3.4 GB |

> **oneDNN** 会在 CMake 配置阶段自动下载并编译源码，无需手动安装。设置 `QASR_ENABLE_ONEDNN=OFF` 可禁用。

#### 安装 OpenBLAS

**Windows**：从 [OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases) 下载预编译包（如 `OpenBLAS-0.3.28-x64.zip`），解压后设置 `OPENBLAS_DIR` 或 `OPENBLAS_ROOT` 环境变量指向解压目录。

**Linux (Debian/Ubuntu)**：
```bash
sudo apt-get install libopenblas-dev
```

**macOS**：无需安装 — 自动使用 Apple Accelerate。

### 构建

#### Windows (MSVC + OpenBLAS)

方式一 — 使用构建脚本（推荐，自动检测 Visual Studio 和 OpenBLAS）：

```powershell
.\tools\build_windows_openblas.ps1 -OpenBlasDir "D:\dev\OpenBLAS"
```

方式二 — 手动 CMake：

```powershell
# 在 Visual Studio 开发者 PowerShell 中
cmake --preset windows-openblas -DOpenBLAS_DIR="D:\dev\OpenBLAS\lib\cmake\openblas"
cmake --build build/windows-openblas
```

> 构建脚本 (`tools/build_windows_openblas.ps1`) 自动查找 Visual Studio、设置环境、定位 OpenBLAS、编译并运行测试。使用 `-SkipTests` 跳过测试。

#### Linux (GCC + OpenBLAS)

```bash
sudo apt-get install build-essential cmake ninja-build libopenblas-dev
cmake --preset linux-openblas
cmake --build build/linux-openblas -j$(nproc)
```

#### macOS (Clang + Accelerate)

```bash
cmake --preset macos-accelerate
cmake --build build/macos-accelerate -j$(sysctl -n hw.ncpu)
```

#### Docker

```bash
docker build -t qasr .
docker run -p 8080:8080 -v /path/to/Qwen3-ASR-0.6B:/models/qwen3-asr qasr
```

### 快速开始

#### 1. 服务器模式（推荐）

启动服务器：

```bash
./qasr_server --model-dir /path/to/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8080 --threads 8
```

启用 INT8 解码器加速：

```bash
./qasr_server --model-dir /path/to/Qwen3-ASR-0.6B --decoder-int8 --threads 16
```

然后在浏览器打开 `http://localhost:8080` 使用 Web UI。

#### 2. OpenAI 兼容转写 API

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

#### 3. 实时 WebSocket 流式

连接 `ws://localhost:8080/v1/realtime`，发送 base64 编码的 PCM16LE 音频块。服务器返回增量转写事件：

```json
{"type": "response.audio_transcript.delta", "delta": "你好"}
{"type": "response.audio_transcript.done", "transcript": "你好世界"}
```

#### 4. CLI 模式

```bash
# 基本转写
./qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B -f audio.wav

# 流式模式 + INT8
./qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B -f audio.wav --stream --decoder-int8

# 详细输出带计时
./qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B -f audio.wav --verbosity 3
```

### 音频要求

- **格式**：WAV (PCM)
- **采样率**：16 kHz
- **声道**：单声道
- **位深**：16-bit

使用 ffmpeg 转换：

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### 项目结构

```
qwen3asr_cpu/
├── app/
│   ├── qasr_cli.cc              # CLI 入口
│   └── qasr_server.cc           # 服务器入口
├── include/qasr/
│   ├── core/                    # 状态码、音频类型、时间戳、状态机
│   ├── inference/               # 编码器、解码器、流式策略接口
│   ├── protocol/                # OpenAI 与 vLLM API 校验
│   ├── runtime/                 # 引擎、配置、会话管理、任务队列
│   └── service/                 # HTTP 服务器、实时 WebSocket 服务
├── src/
│   ├── backend/qwen_cpu/        # 纯 C CPU 推理后端
│   │   ├── qwen_asr.c           # 模型生命周期 & 流式策略
│   │   ├── qwen_asr_decoder.c   # 解码器 prefill/decode (INT8 + GEMM attention)
│   │   ├── qwen_asr_encoder.c   # Whisper 风格音频编码器
│   │   ├── qwen_asr_kernels.c   # BLAS 加速的注意力与矩阵乘法分发
│   │   ├── qwen_asr_kernels_avx.c  # AVX2 SIMD 内核
│   │   └── qwen_asr_onednn.c    # oneDNN INT8 矩阵乘法封装
│   └── ...                      # 服务层 (C++)
├── tests/                       # 438 个单元测试
├── ui/                          # 内置 Web UI (HTML/CSS/JS)
├── Dockerfile                   # 多阶段 Docker 构建
└── CMakeLists.txt
```

### 支持的语言

Qwen3-ASR-0.6B 支持 30+ 语言，包括：

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| 中文（普通话） | zh | 英语 | en |
| 粤语 | yue | 日语 | ja |
| 韩语 | ko | 德语 | de |
| 法语 | fr | 西班牙语 | es |
| 意大利语 | it | 葡萄牙语 | pt |
| 俄语 | ru | 阿拉伯语 | ar |
| 印地语 | hi | 泰语 | th |
| 越南语 | vi | 印尼语 | id |

### 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。

### 致谢

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — 阿里云开源的原始模型
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — 优化的 BLAS 库
- [oneDNN](https://github.com/oneapi-src/oneDNN) — Intel 深度神经网络库
