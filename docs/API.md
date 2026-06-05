# HTTP API Reference

`qasr_server` 暴露的完整 HTTP 端点。基础 URL 假设是 `http://127.0.0.1:19991` (默认) 或 `https://127.0.0.1:19992` (HTTPS 反代)。

**约定**:
- 成功响应: 200 / 201 + JSON (除非显式说明)。
- 错误响应: 4xx / 5xx + JSON `{"error": {"code": "...", "message": "..."}}`。
- SSE 流 (`/api/realtime/stream`): `Content-Type: text/event-stream`，事件名 `event: chunk` / `event: final` / `event: error`。

---

## 健康与诊断

### `GET /api/health`

服务存活检查。

**响应**:
```json
{"status": "ok"}
```

HTTP 200 即代表 server 已加载模型可接受请求。HTTP 502 / 503 代表模型未就绪或启动失败。

### `GET /api/metrics`

Prometheus 风格的运行指标。

**响应** (节选):
```text
# HELP qasr_realtime_active_sessions Active realtime session count
# TYPE qasr_realtime_active_sessions gauge
qasr_realtime_active_sessions 0
# HELP qasr_inference_total Total inference calls
# TYPE qasr_inference_total counter
qasr_inference_total 17
# HELP qasr_inference_ms_total Cumulative inference wall-clock ms
# TYPE qasr_inference_ms_total counter
qasr_inference_ms_total 10234
```

### `GET /health`

`/api/health` 的无前缀别名，行为完全一致。

### `GET /api/debug/get`

返回内部 session 状态 (调试用，无认证)。

**响应**:
```json
{
  "active_sessions": [...],
  "queue_depth": 0,
  "model_loaded": true
}
```

### `POST /api/debug/state`

更新内部调试状态 (开发用)。

**请求体**:
```json
{"verbosity": 3, "force_oom_test": false}
```

---

## 模型信息

### `GET /v1/models`

OpenAI 兼容的模型列表。

**响应**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-asr",
      "object": "model",
      "created": 1234567890,
      "owned_by": "qasr"
    }
  ]
}
```

---

## 异步转写 (推荐用于 > 10s 音频)

### `POST /api/transcriptions/async`

提交音频文件，返回 job ID。Audio 由 server 端 ffmpeg 转码为 16 kHz mono WAV，**不再依赖客户端提供正确格式**。

**请求**:
- `audio`: 必填，multipart file 字段。WAV / MP3 / M4A / FLAC / OGG 都接受。
- `model`: 可选，默认 `qwen3-asr`。
- `language`: 可选 (`Chinese` / `English` / `Japanese` / `Korean` / `auto`)。
- `response_format`: 可选 (`json` / `text` / `verbose_json` / `srt` / `vtt`)，默认 `json`。
- `temperature`: 可选，默认 -1.0 (模型自适应)。
- `timestamp_granularities[]`: 可选 (`segment` / `word`)，需配合 `response_format=verbose_json`。

**响应** (HTTP 202):
```json
{
  "id": "job_2026-06-05T12-34-56_a1b2c3d4",
  "status": "queued",
  "created_at": 1749128096
}
```

### `GET /api/jobs/:id`

查询 job 状态。

**状态值**: `queued` / `running` / `succeeded` / `failed` / `cancelled`。

**响应** (succeeded):
```json
{
  "id": "job_...",
  "status": "succeeded",
  "text": "Hello world.",
  "segments": [{"start": 0.0, "end": 1.2, "text": "Hello world."}],
  "inference_ms": 845,
  "audio_duration_sec": 5.2,
  "rtf": 0.16
}
```

**响应** (failed):
```json
{
  "id": "job_...",
  "status": "failed",
  "error": {
    "code": "kInvalidArgument",
    "message": "audio file is empty"
  }
}
```

### `POST /api/jobs/:id/cancel`

取消排队中或运行中的 job。已完成的 job 返回 409。

**响应**:
```json
{"status": "cancelled"}
```

---

## 同步转写 (用于 < 10s 短音频)

### `POST /api/transcriptions`

同步等待转写完成再返回。

**请求**: 同 `/api/transcriptions/async`。
**响应**: 同 `/api/jobs/:id` 的 succeeded 字段直接平铺。

⚠️ 长音频会超时 (默认 30s)，请用 async 端点。

---

## OpenAI 兼容端点

### `POST /v1/audio/transcriptions`

OpenAI Whisper API 兼容。

**请求**:
```bash
curl -X POST http://127.0.0.1:19991/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```

**响应**: 同 OpenAI Whisper verbose_json 格式。

### `POST /v1/chat/completions`

OpenAI Chat Completions 兼容的音频转写 (multimodal messages)。

**请求**:
```json
{
  "model": "qwen3-asr",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe this audio."},
      {"type": "audio_url", "audio_url": {"url": "file:///abs/path.wav"}}
    ]
  }]
}
```

**响应**:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello world."},
    "finish_reason": "stop"
  }]
}
```

### `POST /v1/realtime`

OpenAI Realtime 兼容入口 (session create / input audio buffer append / commit)。

**请求** (session create):
```json
{
  "action": "session_create",
  "session_id": "sess_001",
  "model": "qwen3-asr",
  "language": "auto"
}
```

**请求** (append):
```json
{
  "action": "input_audio_buffer_append",
  "session_id": "sess_001",
  "audio": "<base64 PCM16 LE>"
}
```

**响应** (SSE): 同 `/api/realtime/stream`。

---

## 实时 mic (私有端点，供 Web UI 使用)

### `POST /api/realtime/start`

启动 realtime session。

**请求体**: 空 (session_id 由 server 生成)。
**响应**:
```json
{
  "session_id": "rt_2026-06-05T12-34-56_e5f6g7h8",
  "model": "qwen3-asr",
  "realtime_model": "qwen3-asr-0.6B"
}
```

### `POST /api/realtime/chunk?session_id=...`

追加一段 PCM16 LE 16 kHz mono 音频 (binary body, `Content-Type: application/octet-stream`)。
**响应**: 200 + 空 body。

### `POST /api/realtime/eof?session_id=...`

标记流结束，触发 VAD flush + 最终段提交。
**响应**: 200 + 空 body。

### `POST /api/realtime/stop?session_id=...`

停止 session，释放资源。
**响应**: 200 + `{"finalized": true, "text": "..."}`。

### `GET /api/realtime/status?session_id=...`

查询 session 当前状态 (stable_text, partial_text, sample_count, ...)。

**响应**:
```json
{
  "session_id": "rt_...",
  "stable_text": "Hello",
  "partial_text": "world",
  "sample_count": 32000,
  "decoded_samples": 24000,
  "retained_sample_count": 8000,
  "retained_sample_offset": 24000,
  "decoded": true,
  "finalized": false,
  "supported": true
}
```

### `GET /api/realtime/stream?session_id=...`

SSE 流，订阅 session 状态更新 (Web UI 用)。

**事件**:
- `event: chunk` data: `{"stable_text": "...", "partial_text": "...", "decoded": true}`
- `event: final` data: `{"stable_text": "...", "finalized": true}`
- `event: error` data: `{"error": {"code": "...", "message": "..."}}`

### `GET /api/realtime/audio_diag?session_id=...`

音频 ingress 诊断 (debug 用)，返回最近 N 个 chunk 的 RMS / peak / VAD 概率。

---

## 主机 capture (CLI / 工具调用)

### `POST /api/capture/start`

启动本地音频 capture (需要 server 端有 pulseaudio / wasapi 支持，通常用于嵌入工具)。
**请求**:
```json
{"device": "default", "sample_rate": 16000}
```

### `POST /api/capture/stop`

停止 capture。
### `GET /api/capture/status`

查询 capture 状态。

---

## 静态资源

### `GET /` → `ui/index.html`
### `GET /app.js` → `ui/app.js`
### `GET /live_monitor.js` → `ui/live_monitor.js`
### `GET /state_pure.js` → `ui/state_pure.js` (新增于 C4)
### `GET /style.css` → `ui/style.css`

---

## 错误码

| HTTP | 含义 |
|------|------|
| 200 | 成功 |
| 202 | 已接受 (async job 排队成功) |
| 400 | 请求体格式错 (`kInvalidArgument`) |
| 404 | 资源不存在 (job_id 找不到 / 路径不对) |
| 409 | 状态冲突 (cancel 已完成 job) |
| 413 | 音频文件过大 (默认 > 64 MB) |
| 422 | 音频解码失败 (ffmpeg 报错) |
| 500 | server 内部错误 |
| 503 | 模型未就绪 (启动中) |

`{"error": {"code": "<StatusCode enum>", "message": "..."}}` 是统一错误格式。
