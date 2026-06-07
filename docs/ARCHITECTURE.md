# Architecture

qwen3asr_cpu 的模块边界、数据流和状态机。

## 顶层结构

```
┌─────────────────────────────────────────────────────────────┐
│                       HTTP / Web UI                          │
│   (ui/* — index.html, app.js, live_monitor.js, state_pure)  │
└────────────────┬───────────────────────────┬────────────────┘
                 │ JSON / multipart            │ SSE
                 ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    HTTP Server (cpp-httplib)                 │
│  src/base/http_server.cc                                    │
└────────────────┬───────────────────────────┬────────────────┘
                 │ routes registered by       │
                 ▼                             │
┌─────────────────────────────────────────────────────────────┐
│              src/service/server.cc (RunServer)              │
│  - routes Get/Post/GetStream                                 │
│  - realtime session lifecycle                                │
│  - VAD-segmented worker                                      │
└────────┬──────────────────┬──────────────────┬───────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ src/runtime/    │  │ src/audio/      │  │ src/protocol/    │
│  - job_pool     │  │  - wav parser   │  │  - OpenAI compat │
│  - task queue   │  │  - resampler    │  │  - request parse │
│  - sessions     │  │  - ffmpeg argv  │  │                  │
│  - model bridge │  │                 │  │                  │
└────────┬────────┘  └────────┬────────┘  └──────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│      src/backend/qwen_cpu/  (C99, model inference)           │
│  - qwen_asr.c      (qwen_load / qwen_clone_shared / free)   │
│  - qwen_asr_encoder.c, qwen_asr_decoder.c                   │
│  - qwen_asr_kernels*.c   (AVX2 / generic SIMD)              │
│  - qwen_asr_onednn.c     (INT8 path, optional)              │
│  - qwen_asr_audio.c      (audio chunk ingest)               │
│  - qwen_silero_vad.c     (ONNX VAD integration)             │
│  - qwen_asr_safetensors.c / _tokenizer.c / _utf8.c         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
            OpenBLAS / Accelerate / oneDNN (BLAS + INT8)
```

## 模块职责

### `src/backend/qwen_cpu/` — C99 CPU 后端

- **qwen_asr.c**: ctx 生命周期 (`qwen_load` / `qwen_clone_shared` / `qwen_free`)。`qwen_clone_shared` 是浅 struct copy + `owns_model_data=0`，让多 session 共享权重。
- **qwen_asr_encoder.c / qwen_asr_decoder.c**: 编码器 (Whisper-style conv) 和解码器 (autoregressive LM)。
- **qwen_asr_kernels_*.c**: GEMV / softmax / RoPE / sample，按 ISA 分发 (AVX2 / NEON / generic)。
- **qwen_asr_onednn.c**: 可选 INT8 加速，仅在 `--encoder-int8` 启用时调用。Decoder INT8 已移除 (C8) — 显著降低识别质量 (语言一致性 / code-switch / 幻觉) 且 ASan 跑出大量越界 (走 quantize + oneDNN matmul 的 fast path 难调试)。Encoder INT8 风险小、收益明显，保留。
- **qwen_silero_vad.c**: Silero VAD v5 ONNX runtime 集成，用于实时流 VAD 段式。
- **qwen_asr_audio.c**: live audio chunk 累积、VAD 状态维护。

### `src/runtime/` — C++ 运行时

- **model_bridge**: 把 `qwen_ctx_t *` 包成可被多 session 共享的 `SharedAsrModel` 引用计数对象。
- **session_manager / realtime_session**: session 状态机 (idle/starting/live/stopping/finalized/error)。
- **job_pool / task_queue / task**: 异步转写任务的并发原语。
- **realtime_session.h**: realtime policy (chunk sec, max new tokens, rollback, idle flush)。

### `src/service/` — HTTP + 实时

- **server.cc (RunServer)**: 1784 行 god function，注册所有 HTTP 路由 + 启动 worker 线程 + 维护 VAD 段式 worker。**已知 god function，按 C1 审计 §5.1 标 TODO 待拆**。
- **server.h**: `ServerConfig` 数据类 + `ValidateServerConfig` / `ParseServerArguments` / `BuildServerUsage` 等独立函数 (已单测)。

### `src/protocol/` — OpenAI / vLLM 兼容

请求解析 (`/v1/audio/transcriptions` `/v1/chat/completions` `/v1/realtime`)，多部分表单和 JSON body 校验。

### `src/audio/` — 音频输入

WAV 解析 (PCM16/24/32 bit)、重采样 (任意 SR → 16 kHz)、ffmpeg 进程构建 (`BuildFfmpegArgv` 防 shell 注入)。

### `ui/` — 浏览器端

- **index.html**: 单页面，含 3 个 script tag (`live_monitor.js` → `state_pure.js` → `app.js`)。
- **app.js** (1019 行): 全部 UI 状态、控件、终端渲染、4 态按钮机。
- **state_pure.js**: 抽离的两个 DOM-free 纯函数 (`computeConfirmedRealtimeText` / `computeSoftResetLines`)，被 app.js 调用并被 Node 单测覆盖。
- **live_monitor.js**: 音频电平 meter (Web Audio API)。

### `tools/` — 运维脚本

- **build_linux.sh / build_windows_openblas.ps1 / build.bat**: 平台编译入口。
- **run_linux_server.sh**: 后台启动 server (+ 可选 HTTPS 反代)，自带 PID / log / 状态查询 / 端口预检。
- **qasr_supervisor.sh**: 长时间运行的 supervisor 循环，崩溃自动重启。
- **smoke_test.sh**: 18 个 bash 烟雾测试。

## 关键数据流

### 1. 异步批量转写 (`POST /api/transcriptions/async`)

```
client POST multipart  ──►  server.cc:3616
                              │
                              ▼
                    TranscribeFileVadSegmentedImpl
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ffmpeg → WAV 16k    VAD sweep (Silero ONNX)   qwen_transcribe
        │                     │                     │
        ▼                     ▼                     ▼
  AppendManualLiveAudio   segment commit       inference_ms
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                       JSON job result
                              │
                              ▼
       client GET /api/jobs/:id ◄── progress poll
```

`TranscribeFileVadSegmentedImpl` (`src/service/server.cc:2337`) 是 1.7B 离线转写的核心，复用 realtime 的 VAD 段式模式 (10 帧 = 500ms 静音阈值 / 40s 强制 cap)。

### 2. 实时 mic → ASR (`POST /api/realtime/start` + chunk/eof/stop)

```
mic ──►  MediaRecorder (16 kHz PCM16)  ──►  app.js sendChunk
                                              │
                                              ▼
                              POST /api/realtime/chunk
                                              │
                                              ▼
                              server.cc:4076  AppendManualLiveAudio
                                              │
                                              ▼
                              live worker thread (pthread)
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                          Silero VAD sweep         qwen_transcribe
                                  │                       │
                                  ▼                       ▼
                          segment commit            stable/partial text
                                  │                       │
                                  └───────────┬───────────┘
                                              ▼
                          client SSE /api/realtime/stream
                                              │
                                              ▼
                                          app.js terminal render
```

### 3. UI 状态机 (4 态)

```
          click Start                  click Stop
   ┌─────────────┐  start  ┌────────────┐  stop  ┌────────────┐
   │   idle      │ ──────► │  starting  │ ─────►│   live     │
   │ (无活)      │         │ (updating  │        │ (recording)│
   │ Start ✓     │         │  flag)     │        │ Stop ✓     │
   │ Stop ✗      │         │  Start ✗   │        │ Start ✗    │
   │ Clear ✓     │         │  Stop ✗    │        │ Stop ✓     │
   │ Export ✓    │         │  Clear ✗   │        │ Clear ✗    │
   └─────────────┘         │  Export ✗  │        │ Export ✗   │
        ▲                  └────────────┘        └─────┬──────┘
        │                                               │
        │ click Stop (after Stop on live)              │
        │                                               │
        │                  ┌─────────────┐              │
        │                  │  stopping   │ ◄────────────┘
        │                  │ (waiting    │  click Stop on live
        │                  │  for ack)   │
        │                  │  Start ✗    │  (回 idle 通过软重置:
        │                  │  Stop ✗     │   done 行保留,
        │                  │  Clear ✗    │   typing→done, 加新 cursor)
        │                  │  Export ✗   │
        │                  └──────┬──────┘
        │                         │ ack received
        └─────────────────────────┘
```

实现: `ui/app.js:294 updateControlAvailability()` 读 `realtimeStarting` / `realtimeStopping` / `activeFeature` 算出按钮 enabled 状态，单点 source of truth。

## 并发模型

### 单进程单 session (默认)

- `qasr_server` 单进程。
- realtime worker 走 pthread，按 `kMaxRealtimeSessions=64` 上限。
- batch 异步任务走 `job_pool` 线程池 (默认 = CPU 核数)。

### 模型共享

- `qwen_load()` 加载一次模型 (~1.2 GB for 0.6B, ~3.4 GB for 1.7B)。
- `qwen_clone_shared()` 浅 struct copy，`owns_model_data=0`，让多 session 共享权重指针。
- free 由源 ctx 负责，clone 不释放权重（150s crash 修复的契约）。

### per-feature 模型 (--realtime-model-dir)

- batch 和 realtime 可独立指定模型目录 (典型: 1.7B batch + 0.6B realtime)。
- 同路径共享 `SharedAsrModel` 实例 (省 ~1.2 GB 内存)。
- 不同路径各自 `qwen_load` 完整拷贝 (~6 GB 总计)。

## 已知 god function / 拆分计划

详见 `docs/AUDIT_C1.md §5.1` 和 `docs/INCIDENTS.md` (2026-06-05 god function entry)。
