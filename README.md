# qwen3asr_cpu

Qwen3-ASR 的 CPU 推理服务与命令行工具，使用 C/C++17 实现。项目提供本地离线转写、字幕输出、HTTP API、内置 Web UI，以及面向 Windows / Linux / macOS 的 CPU 构建路径。

支持 Qwen3-ASR 0.6B / 1.7B safetensors 模型；Windows 和 Linux 使用 OpenBLAS，macOS 使用 Accelerate。可选 oneDNN INT8 路径用于 encoder / decoder 加速。

## 快速开始

### Windows

推荐使用根目录的 `build.bat`：

```bat
build.bat
```

它会转发到 `tools\build_windows_openblas.ps1`，自动导入 MSVC 构建环境、查找 OpenBLAS，然后执行 clean + configure + compile。脚本不会自动下载 OpenBLAS；如果没有从 `OPENBLAS_DIR`、已有 build cache 或常见依赖目录中找到它，请显式传 `--openblas-dir`。

按需追加：

```bat
build.bat --test
build.bat --benchmark
build.bat --test --benchmark
build.bat --openblas-dir D:\dev\OpenBLAS
```

运行时需要让 OpenBLAS DLL 可见：

```powershell
$env:PATH = "D:\dev\OpenBLAS\bin;$env:PATH"
```

### Linux

一键入口是 `tools/build_linux.sh`，**仅支持 Debian 系**（Debian/Ubuntu/Kali/Linux Mint/Pop!_OS/elementary/Raspbian/Zorin/MX/deepin/Parrot）。脚本会按顺序：

1. 校验系统与编译工具链（g++ ≥ 10 / cmake ≥ 3.21 / ninja / pkg-config / ffmpeg / git / curl），缺失会提示安装命令并退出。
2. 检查 OpenBLAS：先看 `${QASR_DEPS_DIR:-/opt/qasr-deps}` 与系统 pkg-config；没有则尝试 `apt-get install -y libopenblas-dev`；再没有则从 GitHub 拉 `${QASR_OPENBLAS_TAG:-v0.3.30}` 源码编译到 `${QASR_DEPS_DIR}`；都失败会打印手动步骤并退出。
3. 检查 ONNX Runtime（Silero VAD 依赖，preset `linux-openblas` 默认启用）：先看 `$QASR_ONNXRUNTIME_ROOT` / `${QASR_DEPS_DIR}/onnxruntime`；没有则复用相邻 `paddle_on_cpu/third_party/onnxruntime-linux-x64-*/`；再没有则从 GitHub releases 下 `v${QASR_ONNXRUNTIME_VERSION:-1.20.1}` 的预编译 `.tgz` 解到 `${QASR_DEPS_DIR}/onnxruntime`。找不到时 **不会**中断构建——Silero VAD 退化为 stub 模式（VAD 段式仍工作但不会自动 commit, 只在 40s 强制 cap 或 `eof` 时 commit）。
4. 探测 `Qwen3-ASR-0.6B` 模型（`$QASR_MODEL_DIR` → `--model-dir` → `~/.cache/huggingface/.../snapshots/*/model.safetensors` → `./models/<repo>`），缺失会提示三种下载方式。
5. 探测 `testfile/*.wav`（缺失提示 `tools/aishell_fetch.py`）。
6. 默认 `clean` 删除 `build/`，然后 `cmake -S/-B/-G Ninja` + `cmake --build` + `ctest`。

```bash
tools/build_linux.sh                       # 默认 clean + build + test
tools/build_linux.sh --incremental         # 增量编译（不删 build/）
tools/build_linux.sh --clean-only          # 只清不编
tools/build_linux.sh --asan                # 用 linux-openblas-asan preset
tools/build_linux.sh --no-test -j 4        # 跳过 ctest, 4 并发
tools/build_linux.sh --model-dir /data/Qwen3-ASR-0.6B
tools/build_linux.sh --no-dep --no-apt     # 离线：禁止自动装包/下载
```

常用环境变量：见 [`docs/CLI.md`](docs/CLI.md) §"tools/build_linux.sh" 节。`tools/build_linux.sh --help` 同步列出。

手动等价流程（脚本不可用时）：

```bash
sudo apt-get install build-essential cmake ninja-build libopenblas-dev ffmpeg
cmake --preset linux-openblas
cmake --build build/linux-openblas -j"$(nproc)"
ctest --test-dir build/linux-openblas --output-on-failure
```

#### 启动 server (一键)

编译完可以用 `tools/run_linux_server.sh` 一键起后台 server（HTTP + 可选 HTTPS + 状态查询），避免手动管 PID / log / 反代：

```bash
export QASR_MODEL_DIR=$HOME/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/<rev>

tools/run_linux_server.sh --detach                  # 后台 HTTP (API/curl 用)
tools/run_linux_server.sh --detach --https          # 后台 HTTP + HTTPS (浏览器用, 推荐, 浏览器需要 https 才能拿 mic 权限)
tools/run_linux_server.sh --status                  # /api/health 检查
tools/run_linux_server.sh --stop                    # 停 --detach 起的 server / proxy
```

HTTPS 反代每次启动 `mktemp -d` 生成自签 cert（退出时自动清，仓库卫生 + 临时安全）；想跨重启复用 cert 设 `QASR_TLS_CERT_DIR=/path/to/cert` 即可。完整环境变量 + flags 见 [`docs/CLI.md`](docs/CLI.md) §"tools/run_linux_server.sh" 节。

### macOS

macOS 没有提供一键脚本，请按下面的手动流程编译：

```bash
brew install cmake ninja ffmpeg
cmake --preset macos-accelerate
cmake --build build/macos-accelerate -j"$(sysctl -n hw.ncpu)"
ctest --test-dir build/macos-accelerate --output-on-failure
```

## 模型

下载 Qwen3-ASR 模型目录后直接传给 `--model-dir`：

| Model | HuggingFace | ModelScope |
|---|---|---|
| Qwen3-ASR-0.6B | <https://huggingface.co/Qwen/Qwen3-ASR-0.6B> | <https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B> |
| Qwen3-ASR-1.7B | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> | <https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B> |
| Qwen3-ForcedAligner-0.6B | <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | <https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B> |

CLI、server 和 API 都通过 `--model-dir` / `model` 指定 ASR 模型，不强制绑定 0.6B 或 1.7B。按 CPU 使用体验建议：

- `Qwen3-ASR-0.6B`：优先用于实时、近实时、Web UI 和低延迟服务。
- `Qwen3-ASR-1.7B`：优先用于离线批处理、长音频转写和字幕生产。
- `Qwen3-ForcedAligner-0.6B`：用于时间轴对齐。它不是替代 ASR 的转写模型，而是在 ASR 得到文本后，把文本和音频对齐成更细的字幕时间。

在 CPU 上不建议把 1.7B 当严格实时模型使用；入口上可以运行，但延迟通常会很高。

## CLI 示例

> 完整参数表 (含默认值 + 影响) 见 [`docs/CLI.md`](docs/CLI.md) §"`qasr_cli` — 离线 CLI 转写"。本节只列常用例子。

基本转写：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav
```

MP3 / M4A / FLAC 等非 WAV 输入会自动通过 `ffmpeg` 转成 16 kHz mono WAV：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio meeting.mp3
```

指定语言、线程数和提示词：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav \
  --language Chinese \
  --prompt "会议记录，包含技术术语" \
  --threads 8
```

输出 SRT 字幕：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 \
  --output-format srt \
  --output movie.srt
```

使用 ForcedAligner 生成更细的字幕时间轴：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 \
  --output-format srt \
  --align \
  --aligner-model-dir /path/to/Qwen3-ForcedAligner-0.6B \
  --output movie.srt
```

输出 WebVTT：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio lecture.wav \
  --output-format vtt \
  --output lecture.vtt
```

输出 JSON 段落：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio interview.wav \
  --output-format json \
  --output interview.json
```

启用可选 INT8 路径（仅 encoder，decoder INT8 已移除 — 见 [`docs/INCIDENTS.md`](docs/INCIDENTS.md) C8 条目）：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav \
  --encoder-int8 \
  --threads 16
```

流式分段推理：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio long.wav \
  --stream \
  --emit-segments \
  --stream-max-new-tokens 32
```

查看完整参数：

```bash
qasr_cli --help
```

## Server 示例

> 完整参数表见 [`docs/CLI.md`](docs/CLI.md) §"`qasr_server` — HTTP / WebSocket 服务"。本节只列常用例子。

启动 Web UI 和 HTTP API：

```bash
qasr_server --model-dir /path/to/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port 8080 \
  --ui-dir ui \
  --threads 8
```

打开：

```text
http://127.0.0.1:8080/
```

启用 INT8：

```bash
qasr_server --model-dir /path/to/Qwen3-ASR-0.6B \
  --port 8080 \
  --encoder-int8 \
  --threads 16
```

查看服务帮助：

```bash
qasr_server --help
```

## HTTP API 示例

OpenAI-style transcription：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

返回纯文本：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=text
```

返回 verbose JSON 和 segment 时间戳：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```

Chat-style audio transcription：

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-asr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "audio_url", "audio_url": {"url": "file:///absolute/path/audio.wav"}}
      ]
    }]
  }'
```

私有异步上传接口：

```bash
curl -X POST http://localhost:8080/api/transcriptions/async \
  -F audio=@audio.wav
```

查询 job：

```bash
curl http://localhost:8080/api/jobs/<job-id>
```

## Docker

```bash
docker build -t qasr .
docker run --rm -p 8080:8080 \
  -v /path/to/Qwen3-ASR-0.6B:/models/qwen3-asr \
  qasr
```

## 常用环境变量

```bash
OPENBLAS_NUM_THREADS=8
QWEN_RUNTIME_PROFILE=balanced
QWEN_DEC_PREFILL_QKV_PERSIST=1
QWEN_DEC_PREFILL_QKV_BUDGET_MB=512
QWEN_ENC_QKV_POLICY=best
```

## 项目结构

```text
app/                    CLI, server, benchmark entry points
include/qasr/           Public C++ headers
src/backend/qwen_cpu/   Internal C CPU backend and kernels
src/service/            HTTP server and realtime session handling
src/runtime/            Model bridge, tasks, sessions, queues
src/protocol/           OpenAI/vLLM request validation
src/audio/              WAV parsing, resampling, ffmpeg conversion helpers
src/subtitle/           SRT/VTT/JSON subtitle writers
tests/                  Unit and regression tests
ui/                     Browser UI
tools/                  Build, benchmark, Docker helper scripts
docs/                   Design notes and internal references
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

## 进一步阅读

- [`docs/CLI.md`](docs/CLI.md) — 全套启动参数 single source of truth (C++ flags / env vars / shell tools)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 模块边界、数据流、UI 状态机、并发模型、god function 拆分计划
- [`docs/API.md`](docs/API.md) — 全部 HTTP 端点、错误码、SSE 事件
- [`docs/SECURITY.md`](docs/SECURITY.md) — 信任边界、威胁清单、生产部署 checklist
- [`docs/AUDIT_C1.md`](docs/AUDIT_C1.md) — C1 审计报告 (死代码 / UAF / OOM / god function)
- [`docs/INCIDENTS.md`](docs/INCIDENTS.md) — 历史事故 + 修复 (150s crash / UI 状态机 / OOM 风险 / decoder-int8 移除)
- [`docs/BLAS_COMPARISON.md`](docs/BLAS_COMPARISON.md) — OpenBLAS / Accelerate / oneDNN 对比

### 高级特性

#### Per-feature 模型 (`--realtime-model-dir`)

`tools/run_linux_server.sh` 启动时, batch 和 realtime 可独立指定模型:

```bash
export QASR_MODEL_DIR=.../Qwen3-ASR-1.7B/...   # batch (高质量)
export QASR_REALTIME_MODEL_DIR=.../Qwen3-ASR-0.6B/...  # realtime (低延迟)
tools/run_linux_server.sh --detach --https
```

同路径时共享 `SharedAsrModel` 实例 (省 1.2 GB); 不同路径各自 `qwen_load` (总计 4.6 GB)。日志会打印 "2 个独立实例" 或 "0 额外内存"。

#### VAD 段式批量转写

`POST /api/transcriptions/async` 现在走 VAD 段式 (`kBatchVadSilenceFrames=16` = 500ms 静音, 40s 强制 cap)。28.77 min 长音频从单次全段改为 ~40 个段, 单段 RTF 0.16-0.18, 总 wall time ~ RTF 1.3-1.5x (因 VAD sweep + 段提交 overhead)。`long.mp3` (6.9 MB, 28.77 min) 端到端 ~600s 跑通。

#### UI 状态机 4 态

`ui/app.js` 实现 4 态按钮机 (idle/starting/live/stopping)。`realtimeStarting` + `realtimeStopping` 双 flag + `updateControlAvailability()` 统一管控件。旧文字保留 (Stop→Start 在下方追加), 启停期间控件全置灰, 杜绝重入。详见 `docs/INCIDENTS.md` (2026-06-05 entry) + `docs/ARCHITECTURE.md` 状态机图。

#### 测试

```bash
# C++ 单测 (665 cases)
ctest --test-dir build/linux-openblas

# JS 状态机纯函数 (24 cases)
node tests/state_pure_test.js

# Bash 烟雾测试 (18 cases)
bash tools/smoke_test.sh
```
