# 设计细则

日期：2026-04-10

## 一、首期代码骨架

### `qasr/core/status.h`

| 实体 | 职责 |
|---|---|
| `StatusCode` | 错误码枚举 |
| `Status` | 统一错误对象 |
| `OkStatus` | 产 OK |
| `StatusCodeName` | 码转文 |

约定：

- 一切 public 函数皆回 `Status` 或显式可判错误值。
- 字符串消息须短，供日志拼装，不作终端长文。

### `qasr/core/audio_types.h`

| 实体 | 职责 |
|---|---|
| `AudioSpan` | 音频视图；不持有内存 |
| `ValidateAudioSpan` | 校输入 |
| `IsMono16kAudio` | 判是否单声道 16k |
| `AudioDurationMs` | 求时长 |

### `qasr/core/timestamp.h`

| 实体 | 职责 |
|---|---|
| `TimestampRange` | 起止毫秒 |
| `ValidateTimestampRange` | 校区间 |
| `SamplesToMilliseconds` | 样本数转毫秒 |
| `FormatSrtTimestamp` | SRT 时间串 |
| `FormatJsonTimestamp` | JSON 时间串 |

### `qasr/runtime/blas.h`

| 实体 | 职责 |
|---|---|
| `BlasBackend` | 编译期 BLAS 类型 |
| `CompiledBlasBackend` | 读当前编译结果 |
| `BlasBackendName` | 枚举转文 |
| `ValidateBlasPolicy` | 校平台 BLAS 律 |

### `qasr/runtime/config.h`

| 实体 | 职责 |
|---|---|
| `EngineConfig` | 进程级配置 |
| `ValidateEngineConfig` | 校配置 |
| `HasAnyProtocolSurface` | 判是否至少开一协议面 |

### `qasr/runtime/task.h`

| 实体 | 职责 |
|---|---|
| `TaskMode` | 离线 / 流式 |
| `TimestampMode` | 无 / segment / word |
| `DecodeRequestOptions` | 单请求参数 |
| `ValidateDecodeRequestOptions` | 校参数 |
| `TimestampModeSupported` | 校任务与时间戳组合 |
| `MakeDeterministicRequestId` | 造稳定 request id |

### `qasr/runtime/engine.h`

| 实体 | 职责 |
|---|---|
| `BootstrapPlan` | 进程启动计划 |
| `ValidateBootstrapInputs` | 校配置与请求组合 |
| `BuildBootstrapPlan` | 由配置推启动面 |

### `qasr/runtime/model_bridge.h`

| 实体 | 职责 |
|---|---|
| `AsrRunOptions` | 单次 ASR 推理参数 |
| `AsrRunResult` | 单次 ASR 推理结果与性能计数 |
| `CpuBackendAvailable` | 判当前平台能否启 CPU 后端 |
| `ValidateModelDirectory` | 校模型目录与分片齐备性 |
| `ValidateAsrRunOptions` | 校推理入参 |
| `ShouldFlushAsrSegment` | CLI 分段输出刷新判定 |
| `RunAsr` | 调 CPU 后端，完成离线/流式转写 |

CLI 长音频输出：

- `--emit-tokens`：逐 token 输出。
- `--emit-segments`：按标点或长度聚合输出。
- `--segment-max-codepoints <n>`：无标点时强制分段阈值，默认 `48`。

### `src/backend/qwen_cpu/qwen_asr_stream.h`

| 实体 | 职责 |
|---|---|
| `qwen_stream_skip_recent_duplicate_prefix` | 流式 append-only 输出前，跳过近邻已提交 token span |

### `qasr/cli/options.h`

| 实体 | 职责 |
|---|---|
| `CliOptions` | CLI 参数承载 |
| `ParseCliArguments` | 解析并校 CLI 参数 |
| `BuildCliUsage` | 生成用法文本 |

### `app/qasr_cli.cc`

| 实体 | 职责 |
|---|---|
| `main` | CLI 入口；调参解、桥接推理、打印结果与性能 |

### `qasr/protocol/openai.h`

| 实体 | 职责 |
|---|---|
| `OpenAiEndpoint` | 协议面枚举 |
| `OpenAiEndpointPath` | 取路径 |
| `IsOpenAiPathSupported` | 判路径 |
| `ValidateOpenAiRequest` | 校首期契约 |

### `qasr/protocol/vllm.h`

| 实体 | 职责 |
|---|---|
| `VllmChatCompletionsPath` | 取 vLLM path |
| `ValidateVllmRequest` | 校首期契约 |

## 二、未来模块设计

以下尚未实现，但已定职责边界。

### `qasr/storage/safetensors_loader`

| 类/函数 | 义 |
|---|---|
| `MappedFile` | 跨平台 mmap/MapViewOfFile 封装 |
| `SafeTensorIndex` | 单文件 tensor 索引 |
| `ShardRegistry` | 多分片一致性 |
| `LoadTensorView` | 返回只读 tensor view |
| `ValidateShardChecksums` | 校验文件完整性 |

### `qasr/model/tokenizer`

| 类/函数 | 义 |
|---|---|
| `Tokenizer` | encode/decode |
| `LoadVocabJson` | 装词表 |
| `LoadMergesTxt` | 装 merge |
| `EncodeUtf8` | 编码 |
| `DecodeIds` | 解码 |

### `qasr/audio/frontend`

| 类/函数 | 义 |
|---|---|
| `WaveReader` | WAV 读入 |
| `PcmResampler` | 重采样 |
| `MelExtractor` | mel |
| `SilenceCompactor` | 静音压缩 |
| `StreamingAudioRing` | 流式环形缓冲 |

### `qasr/inference/encoder`

| 类/函数 | 义 |
|---|---|
| `EncoderWeights` | encoder 权重视图 |
| `EncoderWindowPlan` | chunk / window 规划 |
| `EncodeChunk` | 单 chunk 编码 |
| `EncodeAudio` | 全音频编码 |
| `ConcatEncoderWindows` | 组装窗口输出 |

### `qasr/inference/decoder`

| 类/函数 | 义 |
|---|---|
| `DecoderWeights` | decoder 权重视图 |
| `KvCache` | KV cache |
| `Prefill` | 多 token 预填 |
| `DecodeStep` | 单 token 步进 |
| `BuildPromptEmbeddings` | prompt + audio embedding 组装 |

### `qasr/inference/streaming_policy`

| 类/函数 | 义 |
|---|---|
| `StreamPolicyConfig` | rollback 等参数 |
| `StreamChunkPlanner` | 定时取 chunk / overlap |
| `EncoderCache` | encoder completed windows |
| `RunPartialDecode` | 跑本轮候选文本 |
| `CommitFrontier` | stable frontier 提交 |
| `LongestCommonStablePrefix` | 连续候选求稳定前缀 |
| `DetectDegenerateTail` | 尾部重复检测 |
| `ForceFreezeAgedSuffix` | 未决尾巴超龄冻结 |
| `ReanchorContext` | 恢复性重锚 |
| `EvictOldHistory` | 裁剪 encoder / decoder 历史 |

约束：

- 不得以停顿为必须条件。
- VAD 只能作 hint，不可作唯一 commit 依据。
- `stable_prefix` 只增不减。

### `qasr/inference/timestamp_provider`

| 类/函数 | 义 |
|---|---|
| `TimestampProvider` | 统一接口 |
| `ForcedAlignerProvider` | Qwen3-ForcedAligner 实现 |
| `AlignTranscript` | 文本对齐音频 |
| `BuildSegmentTimestamps` | 生成 segment 时间戳 |
| `BuildWordTimestamps` | 生成词级时间戳 |

### `qasr/runtime/session_manager`

| 类/函数 | 义 |
|---|---|
| `SessionManager` | session 生命周期 |
| `CreateSession` | 开 session |
| `CloseSession` | 关 session |
| `LookupSession` | 取 session |
| `SweepExpiredSessions` | 清理超时 |

### `qasr/runtime/realtime_session`

| 类/函数 | 义 |
|---|---|
| `RealtimeSession` | 单会话流式状态 |
| `AppendAudio` | 收音频块 |
| `TickDecode` | 定周期触发滑窗 decode |
| `BuildPartialDelta` | 产不稳定尾巴 |
| `CommitStableText` | 提交稳定前缀 |
| `FlushTail` | stop / close 时清尾 |
| `SnapshotMetrics` | 取时延、尾长、内存指标 |

### `qasr/runtime/task_queue`

| 类/函数 | 义 |
|---|---|
| `TaskQueue` | 有界队列 |
| `Enqueue` | 入队 |
| `TryDequeue` | 出队 |
| `RejectOverload` | 背压 |
| `CancelTask` | 取消 |

### `qasr/protocol/openai_server`

| 类/函数 | 义 |
|---|---|
| `HandleChatCompletions` | 处理 `/v1/chat/completions` |
| `HandleAudioTranscriptions` | 处理 `/v1/audio/transcriptions` |
| `HandleModels` | 处理 `/v1/models` |
| `HandleMetrics` | 处理 `/api/metrics` |
| `HandleRealtimeSession` | 处理 `/v1/realtime`，二期 |
| `WriteSseDelta` | SSE 输出 delta |
| `FormatVerboseJson` | 产 verbose_json |

当前 HTTP 服务防护：

- task queue 有界，默认上限 `64`
- worker 数随 CPU，最小回退 `4`
- payload 上限 `64MiB`
- read / write timeout `30s`
- keepalive timeout `5s`
- keepalive max count `100`
- realtime 活跃会话上限 `64`

### `qasr/protocol/vllm_server`

| 类/函数 | 义 |
|---|---|
| `HandleVllmChatCompletions` | vLLM 兼容 chat 面 |
| `ValidateVllmStreamingContract` | 校 streaming 限制 |
| `BuildVllmResponse` | 构响应 |

## 三、关键状态机

### Session

`Created -> Warmed -> Running -> Flushing -> Closed`

### Request

`Accepted -> Queued -> Running -> Streaming -> Succeeded/Failed/Cancelled`

### Stream chunk

`Ingested -> Encoded -> Prefilled -> Decoded -> Committed`

### Realtime text

`Unseen -> Partial -> Stable -> Final`

## 四、错误码原则

| 码 | 用途 |
|---|---|
| `kInvalidArgument` | 请求字段错 |
| `kOutOfRange` | 时间/索引越界 |
| `kFailedPrecondition` | 模式组合非法、资源未就绪 |
| `kNotFound` | session/model/artifact 不存在 |
| `kInternal` | 不应出现之内部错 |
| `kUnimplemented` | 预留能力尚未落 |

## 五、协议契约，首期定稿

### OpenAI

| 面 | 首期定稿 |
|---|---|
| chat/completions | 支持文本转写；timestamp 暂不走此面 |
| audio/transcriptions | 支持离线与 timestamp |
| realtime | 私有 `/api/realtime/*` 先行；标准 `/v1/realtime` 二期 |

### vLLM

| 面 | 首期定稿 |
|---|---|
| chat/completions | 支持 |
| streaming | 支持 |
| batch + streaming | 不支持 |
| streaming + timestamp | 不支持 |

## 六、近实时无停顿细则

### 输入律

- 浏览器或本机设备先产 `20ms` 帧。
- 传输聚合为 `160~320ms`。
- 服务端每 `640~960ms` 固定 `tick`，不等停顿。
- 长流只保近窗音频，累计样本数另存，免内存线性增长。

### 滑窗律

- 本轮 decode 主 chunk `2s`。
- encoder window `8s`。
- encoder 历史上限 `32s`。
- 服务层 retained PCM 上限 `32s`。
- decoder prefix 仅留最近 `150 token`。

### 提交律

| 项 | 规则 |
|---|---|
| `partial` | 本轮全文减去已稳定前缀 |
| `stable` | 只向前增长；已提交前缀内的模型修正不再外发 |
| `final` | stop / flush / align 完成后 |

### 失败保护

- 未决尾巴超过 `12~15s`，强制冻结一段。
- 连续多轮空转或重复尾巴，触发 `ReanchorContext`。
- 每次外发前扫描近邻已提交 token，若候选前缀已出现，则跳过，防 rollback/reanchor 重放旧句。
- 若 session 堆积，则先降刷新率，再拒绝新会话。

### 时间戳律

- 流式主路只给 provisional segment。
- word/char timestamp 不阻主路，由 aligner 异步补。

## 七、测试设计

首批代码皆配单测，且覆盖三类值：

- 正常值
- 极值
- 错值 / 随机值

后续模型层再加：

- golden sample
- long-form regression
- streaming stability regression
- timestamp consistency regression
- no-pause continuous-speech regression
- realtime memory-flat regression
