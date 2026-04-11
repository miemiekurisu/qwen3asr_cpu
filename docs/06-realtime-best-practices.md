# 无停顿近实时 ASR 设计

日期：2026-04-10

## 一、问题

目标非“等一句说完再出字”，乃：

- 人几乎不停顿时，仍持续出字
- 模型上下文有界，不能无限长
- 用户观感近实时字幕
- CPU 优先，跨平台不破

故不可把“停顿”当主判据；停顿至多作辅证。

## 二、外部事实

### 1. Qwen 官方约束

- 官方模型卡明言：Qwen3-ASR 支持 streaming，但 streaming 目前只在 vLLM backend 上提供，且 streaming 不支持 batch，不支持 timestamps。
- 官方模型卡又明言：词级/字级时间戳应交 `Qwen3-ForcedAligner-0.6B`。

推论：

- 流式与精细时间戳，宜分两路。
- 我们之 CPU 引擎若做实时，也不宜承诺“流式词级时间戳即刻准确”。

源：

- [Qwen3-ASR 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

### 2. vLLM 官方会话语义

- vLLM OpenAI 兼容服务已定义 Realtime API。
- 音频格式为 `16kHz mono PCM16`。
- 基本序列为：`append -> commit -> transcription.delta -> transcription.done`。

推论：

- 我们自研服务面，宜内核采此状态机。
- HTTP 私有面可先实现，后再映射到 WebSocket `/v1/realtime`。

源：

- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)

### 3. vLLM Qwen3-ASR realtime buffer 之保守默认

- vLLM `qwen3_asr_realtime` buffer 默认 `segment_duration_s = 5.0`。
- 其义是“够段才处理”。

推论：

- `5s` 对“能跑通”足矣；对字幕观感偏慢。
- 我们需更短内核 cadence，但仍保 bounded context。

源：

- [vLLM qwen3_asr_realtime API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_asr_realtime/)

### 4. `qwen-asr-learn` 已给本项目最可行基线

- `--stream` 以 `2s` chunk 运作。
- encoder 默认 `8s` window。
- rollback 默认 `5 token`。
- `unfixed_chunks` 默认 `2`。
- `max_new_tokens` 默认 `32`。
- `max_new_tokens` 最大 `128`；更大值会拖慢每个实时 chunk，长音频应走离线分段而非流式路径。
- 长流下会自动裁 `encoder history` 与 `decoder prefix`，使算量与内存有界。

推论：

- 本项目首版实时核，应先复制此策略，再在服务层补稳定提交与 UX。

源：

- `qwen-asr-learn/README.md`
- `qwen-asr-learn/qwen_asr.h`

### 5. 云服务之共同做法：稳定前缀

- AWS 明言实时转写会先给 partial，后给 final。
- 启用 partial-result stabilization 后，只允许末尾少数词改写。
- AWS 以 `Stable` 字段标记稳定词，并指出此法较“长时间无字，忽然整句突发”更利字幕体验。

推论：

- UI 必须分稳定前缀与不稳定尾巴。
- 传输层宜显式传 `stable` 语义，而非只传整句覆盖。

源：

- [Amazon Transcribe: Streaming and partial results](https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html)
- [Amazon Transcribe Item](https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_Item.html)

### 6. 学界强证：局部一致优于等停顿

- Whisper-Streaming 以 `local agreement policy` + `self-adaptive latency` 实现长语流实时转写。
- 其在未分句长音频上报告 `3.3s latency`。

推论：

- “连续几轮候选之公共前缀即可提交”乃成熟法。
- 不必等待句末停顿，亦能稳步前行。

源：

- [Turning Whisper into Real-Time Transcription System](https://arxiv.org/abs/2307.14743)

### 7. 学界次证：边界截断须专治

- Simul-Whisper 指出 chunk 边界截断词会显著伤流式结果。
- 其以 attention time alignment + truncation detection，在 `1s` chunk 下仍仅有较小绝对 WER 劣化。

推论：

- 固定短 chunk 可行，但边界治理不可省。
- 我们现阶段可先以 rollback + unstable tail 规避；后续再加专门截断检测。

源：

- [Simul-Whisper](https://arxiv.org/abs/2406.10052)

### 8. 新近方向：双通道/双阶段

- 新论文以 `CTC partial decoder + attention rerank` 做 streaming two-pass。
- 其义是：快路保时延，慢路保质量。

推论：

- 本项目长期路线可采“双路”：先给 partial，再由更稳路修正。
- 但此法通常需训练或额外头部；首版不宜阻塞于此。

源：

- [Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding](https://arxiv.org/abs/2506.12154)

## 三、结论

最佳实践非单招，实为六律并用：

1. 不以停顿为主判据。
2. 以定长滑窗保无限流可行。
3. 以 rollback / overlap 解边界截断。
4. 以 local agreement / stable frontier 提交前缀。
5. 以 partial / stable / final 三态分离 UI 与协议。
6. 以 forced aligner 迟后求精细时间戳。

## 四、本项目采用法

### 1. 会话状态

`append -> tick -> partial -> stable_commit -> flush -> final`

其义：

- `append`：写入 ring，不立刻整段推理
- `tick`：按固定周期尝试解码
- `partial`：给可改写尾巴
- `stable_commit`：提交不可回滚前缀
- `flush`：stop / 强制冻结 / 结束时清尾
- `final`：必要时交 aligner，再出终稿

### 2. 三轨

| 轨 | 目的 | 约束 |
|---|---|---|
| `partial lane` | 最快出字 | 允许改写尾部 |
| `stable lane` | 保字幕不乱跳 | 只交连续多轮公共前缀 |
| `final lane` | 保最终正确 | stop 后可再细修与对齐 |

### 3. 初始参数

| 项 | 初值 | 理由 |
|---|---|---|
| 浏览器采样块 | `20 ms` | 音频采集常值 |
| 上传聚合块 | `160~320 ms` | 降 HTTP/WS 开销 |
| decode tick | `640~960 ms` | 保 UI 连续更新 |
| 流式 chunk | `2.0 s` | 低延迟与上下文折中 |
| encoder window | `8.0 s` | 局部注意力窗口 |
| encoder 保留历史 | `32 s` | 有界 |
| 服务端 PCM 保留 | `32 s` | 长流内存有界 |
| decoder rollback | `5 token` | 防尾部抖动 |
| unfixed chunks | `2` | 冷启容错 |
| prefix cap | `150 token` | 有界 |
| max_new_tokens | `32` | 有界 |
| 强制冻结尾巴 | `12~15 s` | 禁尾巴无限长 |

### 4. 提交规则

首版算法：

1. 每轮产新候选串。
2. 与前一轮求最长公共前缀。
3. 再扣去末尾 `rollback_tail_guard` 若干 token。
4. 若该前缀已连续两轮一致，则并入 `stable_prefix`。
5. 余下文本记为 `unstable_suffix`。
6. `unstable_suffix` 存活超阈值，或未决音频超 `12~15s`，则强制冻结一段。

补律：

- VAD 仅可帮助“提早 flush”。
- 即便完全无停顿，系统亦须按 `tick` 推进。

### 5. 时间戳规则

首版：

- `partial` 仅给 segment provisional range
- `stable` 给稳定 segment range
- `final` 才允 word/char 对齐

故：

- 流式字幕先准文本与节奏
- 精细词级时间戳后补

### 6. UI 规则

- 稳定前缀：正常字色
- 不稳定尾巴：浅色或虚线底
- 只重绘尾巴，不重绘整段
- 每次仅保 `1~2` 行字幕
- 若设备性能不足，宁减刷新频率，不可让文本左右跳闪

### 7. 服务协议

私有面先行：

- `/api/realtime/start`
- `/api/realtime/chunk`
- `/api/realtime/stop`

当前增量响应字段：

- 兼容稿：`stable_text`、`partial_text`、`text`、`finalized`
- 主显示稿：`recent_segments`、`live_stable_text`、`live_partial_text`、`live_text`、`display_text`
- 其中 `recent_segments` 只保近段；`text` 留作终稿与全量记录

二期映射：

- `/v1/realtime` WebSocket
- 事件语义贴近 vLLM：
  - `session.created`
  - `input_audio_buffer.append`
  - `input_audio_buffer.commit`
  - `transcription.delta`
  - `transcription.done`

## 五、实现次序

1. 已完成：现有 `/api/realtime/*` 走“定周期滑窗 + 稳定前缀提交 + `recent_segments/live_tail`”。
2. 已完成：测试 UI 主视图改为“近段 + 活尾”，不再实时累加全文。
3. 下一步：补服务端 session worker、bounded queue、指标。
4. 再补 WebSocket `/v1/realtime`。
5. 最后再看 forced aligner 流式后处理。

## 六、测试

必须新增：

- 无停顿连续朗读 `10min+`
- 极短 chunk 与乱序 chunk
- chunk 边界截断词回归
- 稳定前缀单调不回退
- 长流内存平台化，不线性涨
- stop / flush / reconnect
- Docker Linux host audio device 路径

指标：

- `first_partial_ms`
- `first_stable_ms`
- `commit_lag_ms`
- `unstable_tail_ms`
- `realtime_factor`
- `rss_mb`
- `cpu_user_sys_pct`

## 七、对本项目之明确决断

今后实时 ASR 一律遵下列硬规则：

1. 不得假设人会停顿。
2. 不得让 decoder 历史无限长。
3. 不得整句反复覆写，须分稳定与不稳定。
4. 不得把词级时间戳绑死在流式主路。
5. 不得只以“模型能跑”充“字幕可用”。
