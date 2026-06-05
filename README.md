# qwen3asr_cpu

Qwen3-ASR 的 CPU 推理服务与命令行工具，使用 C/C++17 实现。项目提供本地离线转写、字幕输出、HTTP API、内置 Web UI，以及面向 Windows / Linux / macOS 的 CPU 构建路径。

支持 Qwen3-ASR 0.6B / 1.7B safetensors 模型；Windows 和 Linux 使用 OpenBLAS，macOS 使用 Accelerate。可选 oneDNN INT8 路径用于 encoder / decoder 加速。

> ⚠️ **本文档由 AI 根据代码编写,可能存在疏漏**。请以实际代码为准: 命令行参数以 `qasr_cli --help` / `qasr_server --help` / `qasr_cpu_bench --help` 输出为权威; HTTP API 以 `src/service/server.cc` 的路由注册为权威; 环境变量以 `src/backend/qwen_cpu/qwen_asr_perf.c` / `tools/run_linux_server.sh` 的解析为权威。

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
build.bat --openblas-dir <path-to-openblas>   :: 例如 C:\dev\OpenBLAS
```

运行时需要让 OpenBLAS DLL 可见：

```powershell
$env:PATH = "<path-to-openblas>\bin;$env:PATH"   :: 例如 C:\dev\OpenBLAS\bin
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

**注**: `--encoder-int8` 选项当前**入口暂禁用**（接受参数但不生效，见 [已知问题](#已知问题)），
decoder INT8 已在 C8 移除（见 [`docs/INCIDENTS.md`](docs/INCIDENTS.md) C8 条目）。如需最高质量，直接用
默认 FP16 路径；如需最高吞吐，关闭 `--encoder-int8` 并配 `--threads 8`+。

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

**注**: `--encoder-int8` 入口暂禁用（见 [已知问题](#已知问题)），传参不报错但不生效。
生产配置：

```bash
qasr_server --model-dir /path/to/Qwen3-ASR-0.6B \
  --host 0.0.0.0 --port 8080 --ui-dir ui \
  --threads 8
```

查看服务帮助：

```bash
qasr_server --help
```

## 参数参考

完整参数同步在 `qasr_cli --help` / `qasr_server --help` / `tools/build_linux.sh --help` 输出里。
本节是 single source of truth（与上述 help 输出保持同步，CI 不会自动校验但 review 时强制要求）。

### `qasr_cli` — 离线 CLI 转写

单文件音频 → 文本 / SRT / VTT / JSON。

**必需**

| Flag | 说明 |
|---|---|
| `--model-dir <dir>` | ASR 模型目录 (Qwen3-ASR-0.6B / 1.7B) |
| `--audio <file>` | 音频文件 (WAV / MP3 / FLAC / AAC 等；非 WAV 自动 ffmpeg 转码) |

**输出**

| Flag | 默认 | 说明 |
|---|---|---|
| `--output-format <fmt>` | `text` | `text` / `srt` / `vtt` / `json` |
| `--output <path>` | stdout | 输出文件 (字幕格式自动写 `<audio>.<ext>`) |

**对齐 (字幕时间戳)**

| Flag | 说明 |
|---|---|
| `--align` | 启用词级强制对齐 (会拉 Qwen3-ForcedAligner) |
| `--aligner-model-dir <dir>` | ForcedAligner 模型目录 (`--align` 时必填) |

**推理**

| Flag | 默认 | 说明 |
|---|---|---|
| `--threads <n>` | 0=自动 | 推理线程数 (设给 OpenBLAS) |
| `--language <lang>` | (auto) | 强制语言 (`Chinese` / `English` / ...) |
| `--prompt <text>` | (无) | 提示文本 (引导识别风格) |
| `--temperature <float>` | -1.0=auto | 采样温度；`0`=贪心, `>0`=采样 |
| `--encoder-int8` | off | **暂禁用** (入口屏蔽, no-op; 见 [已知问题](#已知问题)) |

> 没有 `--decoder-int8`：C8 整套移除 (语言一致性 / code-switch / 幻觉风险, 见 [`docs/INCIDENTS.md`](docs/INCIDENTS.md) C8 条目)。

**高级 / 流式**

| Flag | 默认 | 说明 |
|---|---|---|
| `--stream` | off | 流式分段推理 |
| `--stream-max-new-tokens <n>` | 32 (max 128) | 流式每段最大 token 数 |
| `--emit-tokens` | off | 逐 token 输出到 stdout |
| `--emit-segments` | off | 按段输出到 stdout |
| `--segment-max-codepoints <n>` | 48 | 每段最大字符数 (决定 flush 阈值) |
| `--verbosity <n>` | 0 | 日志级别 (0=静默, 1=commit+summary, 2=per-poll, 3=raw) |

**帮助**: `-h`, `--help`。

### `qasr_server` — HTTP / WebSocket 服务

模型长驻，提供 OpenAI 兼容 `/v1/audio/transcriptions`、异步 `/api/transcribe`、WebSocket `/api/realtime` / `/v1/realtime`、静态 UI 等端点。完整 HTTP 端点见 [`docs/API.md`](docs/API.md)。

**必需**

| Flag | 说明 |
|---|---|
| `--model-dir <dir>` | batch 转写模型 (必填) |

**模型**

| Flag | 默认 | 说明 |
|---|---|---|
| `--realtime-model-dir <dir>` | 同 `--model-dir` | realtime 模型。空 = 与 batch 共享一份 `SharedAsrModel` (省 ~1.2 GB)；不同 = 加载第二份 (典型: realtime 用 0.6B 省延迟, batch 用 1.7B 拼质量) |

**网络 / 静态资源**

| Flag | 默认 | 说明 |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | 监听地址 (生产 `0.0.0.0`) |
| `--port <n>` | `8080` | HTTP 端口 |
| `--ui-dir <dir>` | `ui` | UI 静态资源目录 |

**推理**

| Flag | 默认 | 说明 |
|---|---|---|
| `--threads <n>` | 0=自动 | 推理线程数 (设给 OpenBLAS) |
| `--temperature <float>` | -1.0=auto | 采样温度；`0`=贪心, `>0`=采样 |
| `--encoder-int8` | off | **暂禁用** (见 [已知问题](#已知问题)) |

> 没有 `--decoder-int8` / `--realtime-decoder-int8`：同 C8 移除。

**日志 / 帮助**

| Flag | 默认 | 说明 |
|---|---|---|
| `--verbosity <n>` | 0 | 0=silent (生产推荐) / 1=commit+summary / 2=per-poll / 3=raw |
| `--quiet`, `-q` | — | 等价 `--verbosity 0` |
| `-h`, `--help` | — | 打印 usage |

### `tools/run_linux_server.sh`

封装 `qasr_server` 的生命周期 (前台 / 后台 / HTTPS / 状态查询 / 停止)。完整脚本逻辑在脚本内。

**Flags**: `--detach` (后台 + PID 文件) / `--https` (起 Python HTTPS 反代, 浏览器拿 mic 权限需要) / `--https-info` (打印当前 cert 路径+指纹) / `--status` (调 `/api/health` 探活) / `--stop` (停 `--detach` 起来的 server/proxy) / `--verbose` (覆盖 `QASR_VERBOSITY=3`)。

### `tools/qasr_supervisor.sh`

进程守护：server 死了自动拉起, 直到显式 kill。用于 systemd-less 环境 / 容器内。

**Flags**: `--no-loop` (单次启动, 死了不拉, 调试用)。

### `tools/build_linux.sh`

一站式编译：探测依赖 (OpenBLAS / ONNX Runtime / HF 模型) → cmake → build → ctest。详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §"Build" 节。

**行为 Flags**: `--clean` (✓) / `--incremental` `--no-clean` / `--clean-only` / `--asan` (用 `linux-openblas-asan` preset) / `--no-test` / `--no-dep` / `--no-model` / `--no-audio` / `--bench` (跑 `qasr_cpu_bench`) / `--compare-blas` (跑 `tools/compare_blas.sh`) / `--no-apt` / `-h` `--help`。

**路径 Flags**: `--blas NAME` (openblas / blis / mkl / auto / ref) / `--model-dir DIR` / `--deps-dir DIR` (默认 `/opt/qasr-deps`) / `--build-dir DIR` (默认 `build/linux-openblas`) / `--openblas-tag TAG` (默认 `v0.3.30`)。

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

### Shell 脚本变量 (`QASR_*`)

| Env | 用在 | 默认 | 说明 |
|---|---|---|---|
| `QASR_MODEL_DIR` | build / run / supervisor | auto | ASR 模型目录 (Qwen3-ASR-0.6B / 1.7B) |
| `QASR_REALTIME_MODEL_DIR` | run | (空) | realtime 模型；空 = 与 batch 共享 (省 1.2 GB) |
| `QASR_HOST` | run | `0.0.0.0` | 监听地址 (HTTPS 模式) |
| `QASR_PORT` | run | `19991` | HTTP 端口 (HTTPS 模式) |
| `QASR_HTTPS_PORT` | run | `19992` | HTTPS 端口 (`--https` 时) |
| `QASR_UI_DIR` | run | `$PROJECT_ROOT/ui` | UI 目录 |
| `QASR_THREADS` | run | `0`=auto | 推理线程 (透传给 `qasr_server --threads`) |
| `QASR_VERBOSITY` | run | `0` | 日志级别 |
| `QASR_VAD_MODEL` | run | `$PROJECT_ROOT/models/silero_vad/silero_vad.onnx` | Silero VAD ONNX 模型 (由 `run_linux_server.sh` 透传给 server 作为 `QWEN_SILERO_VAD_MODEL`) |
| `QASR_TLS_CERT_DIR` | run | mktemp -d | 持久化 cert 目录 (默认启动时 `mktemp -d`, 退出时删) |
| `QASR_PROJECT_ROOT` | run | auto | 项目根 |
| `QASR_BUILD_DIR` | run / build | `build/linux-openblas` | 编译输出 |
| `QASR_LOG_FILE` | run | `/tmp/qasr_server.log` | server 日志 |
| `QASR_PID_FILE` | run | `/tmp/qasr_server.pid` | server PID |
| `QASR_PROXY_SCRIPT` | run | `$SCRIPT_DIR/https_proxy.py` | HTTPS 反代脚本 |
| `QASR_PROXY_LOG` | run | `/tmp/qasr_proxy.log` | proxy 日志 (`--https`) |
| `QASR_PROXY_PID` | run | `/tmp/qasr_proxy.pid` | proxy PID (`--https`) |
| `QASR_DEPS_DIR` | build | `/opt/qasr-deps` | OpenBLAS / ONNX 装哪 |
| `QASR_OPENBLAS_TAG` | build | `v0.3.30` | OpenBLAS 版本 |
| `QASR_PYTHON` | build | auto | python3 路径 |
| `QASR_JOBS` | build | `nproc` | 编译并发 |
| `QASR_APT_MIRROR` | build | 系统默认 | apt 源 (留空用系统默认) |
| `QASR_APT_RETRIES` | docker | `3` | (仅 `tools/docker_linux_openblas.sh`) `apt-get install` 重试次数 |
| `QASR_APT_TIMEOUT` | docker | `20` | (仅 `tools/docker_linux_openblas.sh`) `apt-get install` 单次超时 (秒) |
| `QASR_HF_CACHE` | build | `~/.cache/huggingface` | HF 缓存根 |
| `QASR_HF_REPO` | build | `Qwen/Qwen3-ASR-0.6B` | 模型仓库 (探测失败时下载) |
| `QASR_ONNXRUNTIME_ROOT` | build | auto | ONNX runtime 安装路径 (手装时指定) |
| `QASR_ONNXRUNTIME_VERSION` | build | `1.20.1` | ONNX runtime 版本 (自动下载时用) |

HF / 包下载代理 (内网环境，按需启用)：

```bash
# 替换 <your-proxy-host> / <your-lan-cidr> 为实际代理地址和局域网段
export https_proxy=http://<your-proxy-host>:8117
export no_proxy=127.0.0.1,localhost,<your-lan-cidr>,10.0.0.0/8,172.16.0.0/12
```

### 推理性能微调 (`QWEN_*`)

Qwen3-ASR 内部 C 后端用这些 env var 调线程 / 内存：

| Env | 默认 | 说明 |
|---|---|---|
| `OPENBLAS_NUM_THREADS` | 0=auto | OpenBLAS 线程数 (强烈建议显式设, i7-14700KF 用 `8`-`12`) |
| `QWEN_RUNTIME_PROFILE` | `balanced` | `balanced` / `realtime` / `offline` / `edge_lowmem` — 预调线程+内存 |
| `QWEN_DEC_PREFILL_QKV_PERSIST` | 0 | 1=解码 prefill QKV 权重常驻内存 (省 alloc, ~1 GB RSS) |
| `QWEN_DEC_PREFILL_QKV_BUDGET_MB` | 512 | QKV 预分配上限 (MB), 超了降级非持久 |
| `QWEN_DEC_PREFILL_GATE_UP_PERSIST` | 0 | 1=MLP gate_up 持久 |
| `QWEN_DEC_PREFILL_GATE_UP_BUDGET_MB` | 0 | MLP gate_up 预算 |
| `QWEN_ENC_QKV_POLICY` | `best` | `best` / `force_separate` / `force_packed` / `shape_auto` |
| `QWEN_ENC_QKV_PACK_MIN_SEQ` | 4 | `shape_auto` 启用 packed 的最小 seq_len |
| `QWEN_ENC_QKV_SHAPE_AUTO_LARGE_SEQ` | 96 | `shape_auto` 视为"大 seq"阈值 |
| `QWEN_ENC_QKV_SHAPE_AUTO_LARGE_DMODEL` | 1024 | `shape_auto` 视为"大 dmodel"阈值 |
| `QWEN_ENC_QKV_SHAPE_AUTO_MAX_SEPARATE_THREADS` | 8 | `shape_auto` 用 separate 的最大线程数 |
| `QWEN_PREFILL_THREADS` | 0=auto | encoder QKV prefill 线程数 (覆盖 `shape_auto` 推断) |
| `QWEN_DECODE_THREADS` | 0=auto | decoder 线程数 |
| `QWEN_BF16_CACHE_MB` | 0=off | encoder BF16 权重缓存上限 (MB); 0=不缓存; 设了之后会按需 keep 住已量化权重 |
| `QWEN_STREAM_NO_ENC_CACHE` | 0=cache on | 非 0=禁 encoder 缓存 (流式节省内存但每次重算); 注意: realtime 模式强制开启缓存, 此 env 不生效 |
| `QWEN_DEC_LAYER_TIMING` | 0=off | 非 0=打印每层 decoder 时间 (debug 用, 影响性能) |

#### Silero VAD 路径查找 (`QWEN_*`)

C 端 `qwen_silero_vad_resolve_path` 按下列顺序查找 ONNX 文件。`run_linux_server.sh`
已把 `$QASR_VAD_MODEL` 透传给 `QWEN_SILERO_VAD_MODEL`, 普通用户无需设这些:

| Env | 默认 | 说明 |
|---|---|---|
| `QWEN_SILERO_VAD_MODEL` | (空) | 直接路径: `/path/to/silero_vad.onnx` |
| `QWEN_SILERO_VAD_DIR` | (空) | 目录: `$DIR/silero_vad.onnx` |
| `QWEN_MODEL_DIR` | (空) | 兜底: `$QWEN_MODEL_DIR/silero_vad.onnx` (与 ASR 模型同目录) |

实战 (i7-14700KF, 0.6B)：

```bash
export OPENBLAS_NUM_THREADS=8
export QWEN_RUNTIME_PROFILE=balanced
# 长跑实时 (24h) 不爆 RSS, 关键是把 prefill 预算砍半:
export QWEN_DEC_PREFILL_QKV_PERSIST=1
export QWEN_DEC_PREFILL_QKV_BUDGET_MB=256
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

## 已知问题

影响用户的最近两个变更（用户视角摘要，完整回滚步骤见 [`docs/INCIDENTS.md`](docs/INCIDENTS.md)）：

### 2026-06-05: `--encoder-int8` 入口暂禁用 (c5b67cd)

**现象**: `--encoder-int8` 仍然接受, **不报错但 no-op**, `--help` 已隐藏该行。
**原因**: 用户报告偶发转写质量退化 (Whisper-style conv 在某些声学分布上 INT8 仍掉点),
encoder 内存只占 ~20%, 风险/收益不划算。代码 (5 处调用 + C API + 量化路径) **完整保留**,
后续要恢复只需 `git revert c5b67cd`。

### 2026-06-05: UI `offlineStop` 5xx 静默覆盖修 (c5b67cd)

**现象**: 修复前, 点 batch 任务"停止"按钮若服务端 cancel 返回 5xx, 300ms 后状态行
被 poll 循环静默改回"转写中: 0.6s", 用户以为 Stop 没生效。修复后状态行保留"停止失败: <reason>",
Stop 按钮解禁可重试。

## 进一步阅读

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 模块边界、数据流、UI 状态机、并发模型、god function 拆分计划
- [`docs/API.md`](docs/API.md) — 全部 HTTP 端点、错误码、SSE 事件
- [`docs/SECURITY.md`](docs/SECURITY.md) — 信任边界、威胁清单、生产部署 checklist
- [`docs/AUDIT_C1.md`](docs/AUDIT_C1.md) — C1 审计报告 (死代码 / UAF / OOM / god function)
- [`docs/INCIDENTS.md`](docs/INCIDENTS.md) — 历史事故 + 修复 (150s crash / UI 状态机 / OOM 风险 / decoder-int8 移除 / OOM 风险 god function)
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
# C++ 单测 (665 cases, 全部 PASS, ASAN build 654 cases)
ctest --test-dir build/linux-openblas --output-on-failure

# JS 状态机纯函数 (47 cases — 6 个纯函数: 重置/确认/降采样/PCM16/字符数/导出名)
node tests/state_pure_test.js

# JS UI 4 态机集成 (12 cases — jsdom 模拟, 验证按钮 + 文案)
node tests/ui_state_machine_test.js

# JS UI 异步流 (7 cases — jsdom + queue-fetch mock, 验证 health/export/stop 错误)
node tests/ui_async_test.js

# Bash 烟雾测试 (18 cases — server 启停 + build + 健康检查)
bash tools/smoke_test.sh
```
