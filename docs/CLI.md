# CLI / 启动参数参考

**这是 qasr_cpu 全套启动参数的 single source of truth**。三处使用:
1. C++ 端的 `--help` 输出 (`BuildCliUsage` / `BuildServerUsage`) — 这些函数输出的字符串与本文档保持同步
2. shell 脚本 `--help` 输出 (`run_linux_server.sh` / `qasr_supervisor.sh` / `build_linux.sh`) — 三者格式统一为 `Usage:` / `Flags:` / `Env vars:` / `Examples:`
3. README / API.md / ARCHITECTURE.md / SECURITY.md — 一律引用本文档而非重复列举

如有新参数或语义变更，**先改本文档**，再同步到 C++ / shell / 其他 markdown。

---

## 目录

- [`qasr_cli` — 离线 CLI 转写](#qasr_cli--离线-cli-转写)
- [`qasr_server` — HTTP / WebSocket 服务](#qasr_server--http--websocket-服务)
- [`qasr_align` — 强制对齐 (词级时间戳)](#qasr_align--强制对齐-词级时间戳) *(如有)*
- [`tools/run_linux_server.sh`](#toolsrun_linux_serversh)
- [`tools/qasr_supervisor.sh`](#toolsqasr_supervisorsh)
- [`tools/build_linux.sh`](#toolsbuild_linuxsh)
- [跨工具共用的 `QASR_*` 环境变量](#跨工具共用的-qasr_-环境变量)

---

## `qasr_cli` — 离线 CLI 转写

单文件音频 → 文本 / SRT / VTT / JSON。

### 必需参数

| Flag | 说明 |
|---|---|
| `--model-dir <dir>` | ASR 模型目录 (Qwen3-ASR-0.6B / 1.7B 任意, 不绑定) |
| `--audio <file>` | 音频文件 (WAV / MP3 / FLAC / AAC 等; 非 WAV 自动调 ffmpeg 转码) |

### 输出

| Flag | 默认 | 说明 |
|---|---|---|
| `--output-format <fmt>` | `text` | `text` / `srt` / `vtt` / `json` |
| `--output <path>` | stdout (字幕格式自动写 `<audio>.<ext>`) | 输出文件路径 |

### 对齐 (生成更精确的字幕时间戳)

| Flag | 说明 |
|---|---|
| `--align` | 启用词级强制对齐 |
| `--aligner-model-dir <dir>` | ForcedAligner 模型目录 (`--align` 时必填) |

### 推理

| Flag | 默认 | 说明 |
|---|---|---|
| `--threads <n>` | 0=自动 | CPU 线程数 |
| `--language <lang>` | (auto) | 强制语言 (`Chinese` / `English` / ...) |
| `--prompt <text>` | (无) | 提示文本 (引导识别风格) |
| `--temperature <float>` | -1.0=auto | 采样温度; `0`=贪心, `>0`=采样 |
| `--encoder-int8` | off | Encoder INT8 量化 (Whisper-style conv, 风险小, 收益明显) |

> ❌ **没有 `--decoder-int8`**: decoder INT8 在 C8 移除, 显著降低识别质量 (语言一致性 / code-switch / 幻觉). 见 `docs/INCIDENTS.md` 2026-06-05 条目.

### 高级 / 流式

| Flag | 默认 | 说明 |
|---|---|---|
| `--stream` | off | 流式分段推理 |
| `--stream-max-new-tokens <n>` | 32 (max 128) | 流式每段最大 token 数 |
| `--emit-tokens` | off | 逐 token 输出到 stdout |
| `--emit-segments` | off | 按段输出到 stdout |
| `--segment-max-codepoints <n>` | 48 | 每段最大字符数 (决定 flush 阈值) |
| `--verbosity <n>` | 0 | 日志详细级别 (0=静默, 1=详细) |

### 帮助

| Flag | 说明 |
|---|---|
| `-h`, `--help` | 打印 usage |

### 常用例子

```bash
# 基本
qasr_cli --model-dir $HF/Qwen3-ASR-0.6B --audio meeting.wav

# 中文会议记录 + 8 线程
qasr_cli --model-dir $HF/Qwen3-ASR-0.6B --audio meeting.mp3 \
  --language Chinese --prompt "会议记录, 包含技术术语" --threads 8

# 1.7B 出 SRT 字幕
qasr_cli --model-dir $HF/Qwen3-ASR-1.7B --audio movie.mp3 \
  --output-format srt --output movie.srt

# 内存紧: 只开 encoder INT8
qasr_cli --model-dir $HF/Qwen3-ASR-0.6B --audio long.wav \
  --encoder-int8 --threads 8
```

---

## `qasr_server` — HTTP / WebSocket 服务

模型长驻, 提供 `/v1/audio/transcriptions` (OpenAI 兼容), `/api/transcribe` (异步), `/api/realtime` / `/v1/realtime` (WebSocket 流式), 静态 UI 等端点. 完整 HTTP API 见 [`docs/API.md`](API.md).

### 必需参数

| Flag | 说明 |
|---|---|
| `--model-dir <dir>` | batch 转写模型目录 (必填) |

### 模型

| Flag | 默认 | 说明 |
|---|---|---|
| `--realtime-model-dir <dir>` | 同 `--model-dir` | realtime / host-capture worker 模型. 空 = 与 batch 共享一份 `SharedAsrModel` (省 6 GB). 不同 = 加载第二份 (典型: realtime 用 0.6B 省延迟, batch 用 1.7B 拼质量) |

### 网络 / 静态资源

| Flag | 默认 | 说明 |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | 监听地址 |
| `--port <n>` | `8080` | HTTP 端口 |
| `--ui-dir <dir>` | `ui` | UI 静态资源目录 |

### 推理

| Flag | 默认 | 说明 |
|---|---|---|
| `--threads <n>` | 0=自动 | 推理线程数 (设给 OpenBLAS) |
| `--temperature <float>` | -1.0=auto | 采样温度; `0`=贪心, `>0`=采样 |
| `--encoder-int8` | off | Encoder INT8 (realtime clone 自动沿用) |

> ❌ **没有 `--decoder-int8` / `--realtime-decoder-int8`**: 同上, C8 移除.

### 日志

| Flag | 默认 | 说明 |
|---|---|---|
| `--verbosity <n>` | 0 | 0=silent (推荐生产) / 1=commit+summary / 2=per-poll / 3=raw |
| `--quiet`, `-q` | — | 等价 `--verbosity 0` |
| `-h`, `--help` | — | 打印 usage |

### 常用例子

```bash
# 开发: 前台 + 日志
qasr_server --model-dir $HF/Qwen3-ASR-0.6B --host 0.0.0.0 --port 19991 --verbosity 2

# 生产: 静默 + detached, 用 run_linux_server.sh 启
tools/run_linux_server.sh --detach

# 双模型: batch 1.7B, realtime 0.6B
qasr_server --model-dir $HF/Qwen3-ASR-1.7B \
            --realtime-model-dir $HF/Qwen3-ASR-0.6B \
            --encoder-int8
```

---

## `tools/run_linux_server.sh`

封装 `qasr_server` 的生命周期 (前台 / 后台 / HTTPS / 状态查询 / 停止). 完整脚本逻辑见脚本内.

### Flags

| Flag | 说明 |
|---|---|
| `--detach` | 后台启动, 写 PID 到 `$QASR_PID_FILE` (默认 `/tmp/qasr_server.pid`) |
| `--https` | 同时起 Python HTTPS 反代 (浏览器拿 mic 权限必须 https) |
| `--https-info` | 打印当前 HTTPS cert / proxy 信息 (mktemp 路径 / 自签 cert 指纹) |
| `--status` | 调 `/api/health` 探活 |
| `--stop` | 停 `--detach` 起来的 server / proxy |
| `--verbose` | 覆盖 `QASR_VERBOSITY=3` (开发用, 一行一 poll) |

### Env vars (完整列表)

| Env | 默认 | 说明 |
|---|---|---|
| `QASR_MODEL_DIR` | (无, 必填) | Qwen3-ASR-0.6B 目录, 须含 `model.safetensors` |
| `QASR_REALTIME_MODEL_DIR` | (空) | realtime 模型; 空 = 与 batch 共享 |
| `QASR_HOST` | `0.0.0.0` | 监听地址 |
| `QASR_PORT` | `19991` | HTTP 端口 |
| `QASR_HTTPS_PORT` | `19992` | HTTPS 端口 (仅 `--https` 时生效) |
| `QASR_UI_DIR` | `$PROJECT_ROOT/ui` | UI 目录 |
| `QASR_THREADS` | `0`=auto | 推理线程 |
| `QASR_VERBOSITY` | `0` | 日志级别 |
| `QASR_VAD_MODEL` | `$PROJECT_ROOT/models/silero_vad/silero_vad.onnx` | Silero VAD ONNX 模型 |
| `QASR_TLS_CERT_DIR` | mktemp -d | 持久化 cert 目录 (默认启动时 `mktemp -d`, 退出时删) |
| `QASR_PROJECT_ROOT` | (auto) | 项目根 |
| `QASR_BUILD_DIR` | `build/linux-openblas` | 二进制目录 |
| `QASR_LOG_FILE` | `/tmp/qasr_server.log` | server 日志 |
| `QASR_PID_FILE` | `/tmp/qasr_server.pid` | server PID |
| `QASR_PROXY_SCRIPT` | `$SCRIPT_DIR/https_proxy.py` | HTTPS 反代脚本 |
| `QASR_PROXY_LOG` | `/tmp/qasr_proxy.log` | proxy 日志 (`--https` 时) |
| `QASR_PROXY_PID` | `/tmp/qasr_proxy.pid` | proxy PID (`--https` 时) |

### 例子

```bash
# 后台 + HTTPS, 默认端口
QASR_MODEL_DIR=$HF/Qwen3-ASR-0.6B tools/run_linux_server.sh --detach --https

# 状态 / 停
tools/run_linux_server.sh --status
tools/run_linux_server.sh --stop

# 持久化 cert (跨重启复用)
QASR_TLS_CERT_DIR=/etc/qasr/tls QASR_MODEL_DIR=... tools/run_linux_server.sh --detach --https
```

---

## `tools/qasr_supervisor.sh`

进程守护: server 死了自动拉起, 直到显式 kill. 用于 systemd-less 环境 / 容器内.

### Flags

| Flag | 说明 |
|---|---|
| `--no-loop` | 单次启动, 死了不拉 (调试用) |

### Env vars

| Env | 默认 | 说明 |
|---|---|---|
| `QASR_MODEL_DIR` | (无, 必填) | 透传给 `run_linux_server.sh` |

### 例子

```bash
# 永远循环, 死了拉起
QASR_MODEL_DIR=$HF/Qwen3-ASR-0.6B tools/qasr_supervisor.sh

# 单次启动, 死了不拉
QASR_MODEL_DIR=$HF/Qwen3-ASR-0.6B tools/qasr_supervisor.sh --no-loop
```

---

## `tools/build_linux.sh`

一站式编译: 探测依赖 (OpenBLAS / ONNX Runtime / HF 模型) → cmake → build → ctest. 见 `docs/ARCHITECTURE.md` §"Build" 节.

### Flags

#### 行为

| Flag | 默认 | 说明 |
|---|---|---|
| `--clean` | ✓ | 删 build 目录后从头构 |
| `--incremental` / `--no-clean` | — | 增量编译, 不删 build/ |
| `--clean-only` | — | 只清不编 |
| `--asan` | — | 用 `linux-openblas-asan` preset (AddressSanitizer) |
| `--no-test` | — | 跳 ctest |
| `--no-dep` | — | 跳 OpenBLAS 检查/构建 |
| `--no-model` | — | 跳模型探测 |
| `--no-audio` | — | 跳测试音频探测 |
| `--bench` | — | 编完跑 `qasr_cpu_bench` |
| `--compare-blas` | — | 编完跑 `tools/compare_blas.sh` (需装好 OpenBLAS+BLIS+MKL) |
| `--no-apt` | — | 不调 apt-get (即使缺包) |
| `-h`, `--help` | — | 打印 usage |

#### 路径 / 依赖

| Flag | 默认 | 说明 |
|---|---|---|
| `--blas NAME` | preset 默认 | BLAS 后端: `openblas` / `blis` / `mkl` / `auto` / `ref` (传给 `-DQASR_BLAS=NAME`) |
| `--model-dir DIR` | `$QASR_MODEL_DIR` 或 auto | 模型目录 (覆盖 env) |
| `--deps-dir DIR` | `/opt/qasr-deps` | OpenBLAS / ONNX 安装位置 |
| `--build-dir DIR` | `build/linux-openblas` | 编译输出目录 |
| `--openblas-tag TAG` | `v0.3.30` | 源码下载版本 |

### Env vars

| Env | 默认 | 说明 |
|---|---|---|
| `QASR_DEPS_DIR` | `/opt/qasr-deps` | OpenBLAS / ONNX 装哪 |
| `QASR_BUILD_DIR` | `build/linux-openblas` | 编译输出 |
| `QASR_MODEL_DIR` | auto 探测 | 模型目录 (优先 env, 再 HF cache, 再 `./models/`) |
| `QASR_OPENBLAS_TAG` | `v0.3.30` | OpenBLAS 版本 |
| `QASR_PYTHON` | auto | python3 路径 |
| `QASR_JOBS` | `nproc` | 编译并发 |
| `QASR_APT_MIRROR` | 系统默认 | apt 源 (留空用系统默认) |
| `QASR_HF_CACHE` | `~/.cache/huggingface` | HF 缓存根 |
| `QASR_HF_REPO` | `Qwen/Qwen3-ASR-0.6B` | 模型仓库 |
| `QASR_ONNXRUNTIME_ROOT` | auto | ONNX runtime 安装路径 (手装时指定) |
| `QASR_ONNXRUNTIME_VERSION` | `1.20.1` | ONNX runtime 版本 (自动下载时用) |

### 例子

```bash
# 默认: clean + configure + build + ctest
tools/build_linux.sh

# 增量 + ASAN
tools/build_linux.sh --incremental --asan

# 离线: 禁止自动装包/下载
tools/build_linux.sh --no-dep --no-apt

# 试 BLIS 后端
tools/build_linux.sh --blas blis

# 装自定义 ONNX Runtime
QASR_ONNXRUNTIME_ROOT=/opt/onnxruntime-1.20.1 tools/build_linux.sh
```

---

## 跨工具共用的 `QASR_*` 环境变量

| Env | 用在 | 默认 | 说明 |
|---|---|---|---|
| `QASR_MODEL_DIR` | build / run / supervisor | auto | ASR 模型目录 |
| `QASR_REALTIME_MODEL_DIR` | run | 空 | realtime 模型 (空=共享) |
| `QASR_HOST` | run | `0.0.0.0` | 监听地址 |
| `QASR_PORT` | run | `19991` | HTTP 端口 |
| `QASR_HTTPS_PORT` | run | `19992` | HTTPS 端口 |
| `QASR_UI_DIR` | run | `$PROJECT_ROOT/ui` | UI 目录 |
| `QASR_THREADS` | run | `0`=auto | 推理线程 |
| `QASR_VERBOSITY` | run | `0` | 日志级别 |
| `QASR_VAD_MODEL` | run | `$PROJECT_ROOT/models/silero_vad/silero_vad.onnx` | Silero VAD 模型 |
| `QASR_TLS_CERT_DIR` | run | mktemp -d | 持久化 cert |
| `QASR_PROJECT_ROOT` | run | auto | 项目根 |
| `QASR_BUILD_DIR` | run / build | `build/linux-openblas` | 编译输出 |
| `QASR_LOG_FILE` | run | `/tmp/qasr_server.log` | server 日志 |
| `QASR_PID_FILE` | run | `/tmp/qasr_server.pid` | server PID |
| `QASR_PROXY_SCRIPT` | run | `$SCRIPT_DIR/https_proxy.py` | HTTPS 反代脚本 |
| `QASR_PROXY_LOG` | run | `/tmp/qasr_proxy.log` | proxy 日志 |
| `QASR_PROXY_PID` | run | `/tmp/qasr_proxy.pid` | proxy PID |
| `QASR_DEPS_DIR` | build | `/opt/qasr-deps` | OpenBLAS / ONNX 装哪 |
| `QASR_OPENBLAS_TAG` | build | `v0.3.30` | OpenBLAS 版本 |
| `QASR_PYTHON` | build | auto | python3 路径 |
| `QASR_JOBS` | build | `nproc` | 编译并发 |
| `QASR_APT_MIRROR` | build | 系统默认 | apt 源 |
| `QASR_HF_CACHE` | build | `~/.cache/huggingface` | HF 缓存 |
| `QASR_HF_REPO` | build | `Qwen/Qwen3-ASR-0.6B` | 模型仓库 |
| `QASR_ONNXRUNTIME_ROOT` | build | auto | ONNX runtime 路径 |
| `QASR_ONNXRUNTIME_VERSION` | build | `1.20.1` | ONNX runtime 版本 |

---

## 维护规则

1. **新增 CLI flag / env var**: 先在本文档加, 再改 C++ / shell 解析. CI 不会自动校验, 但 PR review 时强制要求同步.
2. **删除 flag / env var**: 标 `--deprecated` 在本文档留 1 个 release 周期, 然后删. C++ / shell 解析同步删.
3. **flag 命名**: 单字用连字符 (`--output-format`), 不用下划线 (那是字段名风格).
4. **help 字符串三处同步**: C++ 端 `BuildCliUsage` / `BuildServerUsage` 是权威; 三个 `.sh` 的 help block 引用同字段.
5. **bug fix 与 doc 同步**: 修 bug 时若影响参数语义, 必同步本文档; 反之亦然.
