# 实时流式管线审计：漂移分析与多线程机会

日期：2026-04-11

---

## 0. 结论先行

当前实时管线存在 **三类可消除的滞后源**，合计在 0.6B 模型上约 1.5-3 秒观感延迟，在 1.7B 上更不可用。核心根因是 **整条管线串行运行在单线程上**：音频等待 → encoder → prefill → decode → 文本提交，都在 `stream_impl()` 的 while 循环中顺序执行，没有任何流水线并行。

业界成熟方案（sherpa-onnx、whisper.cpp、vLLM realtime）全部采用某种形式的 **encoder/decoder 分离** 或 **双缓冲流水线**。当前项目有充分的条件做同样的改造，且不需要改模型。

**可实施的三级改造路线**：

1. **立即可做（无架构变更）**：UI 发送间隔 800ms→400ms、轮询间隔 250ms→150ms、chunk_sec 2.0→1.0，约可减少 500-800ms 观感滞后
2. **中期（拆分 encoder 线程）**：encoder 在独立线程上持续处理新窗口，decoder 线程只做 prefill+decode，两者通过队列对接——这是最大的一次提速
3. **远期（双 ctx 乒乓）**：两个 clone ctx 交替 decode，当 ctx_A 在解码第 N 轮时 ctx_B 已在准备第 N+1 轮的 prefill

---

## 1. 当前管线时序分析

### 1.1 端到端数据流（单轮）

```
浏览器                    服务端 HTTP               stream_impl (live worker)
  │                          │                              │
  │─── PCM chunk ──────────>│                              │
  │    (每800ms一批)         │─ AppendManualLiveAudio ────>│
  │                          │                              │
  │                          │     [等 chunk_samples 凑够]  │
  │                          │                              │
  │                          │     [encoder: 当前窗口]    ← 占 CPU
  │                          │     [prefill: 构造嵌入]    ← 占 CPU
  │                          │     [decode: 逐 token]     ← 占 CPU
  │                          │     [commit: token→text]     │
  │                          │     [token_cb → session.mu]  │
  │                          │                              │
  │<── poll /status (250ms) ─│                              │
```

### 1.2 关键时间锚点（0.6B, 10s 音频, 2s chunk）

| 阶段 | 耗时（实测 benchmark） | 说明 |
|------|----------------------|------|
| 浏览器到服务端 | ~800ms (sendTimer 间隔) | 即使音频已有，最快也要等 800ms 才发送 |
| 凑够 chunk | 2000ms (chunk_sec=2.0) | `LA_WAIT` 阻塞直至 2s 音频到齐 |
| encoder (缓存命中) | ~50-80ms | 仅编码 partial tail Window |
| encoder (新 window) | ~200-400ms | 完整 8s window 编码 |
| prefill (delta) | ~100-300ms | 取决于 KV reuse 比例 |
| decode (32 token) | ~200-600ms | ~10-20ms/token |
| 文本提交 + mutex | <1ms | 轻量 |
| poll 到达 UI | ~125ms (平均) | 250ms 间隔的期望值 |

**总计单轮最大观感延迟 = 800 + 2000 + 400 + 300 + 600 + 125 ≈ 4.2 秒**

其中:
- **可消除的工程浪费**: 发送间隔 (800ms)、chunk 凑齐等待 (2000ms)、轮询间隔 (125ms) = 约 2.9 秒
- **不可消除的计算**: encoder + prefill + decode ≈ 0.5-1.3 秒

### 1.3 "待下轮" 现象的根因

用户观察到 UI 显示 "待下轮" 而非 "已解码"：这源于 **UI 与 decode 的节奏不对齐**。poll 到达时 live worker 正在 `LA_WAIT` 或计算中，`session->last_decode_ran` 还未置位。这不是 bug，但说明 decode 周期与 UI 刷新周期存在明显错位。

---

## 2. 漂移分析

### 2.1 定义

"漂移" = 用户说话到屏幕出字的延迟随时间增加。

### 2.2 当前是否存在漂移？

**轻微漂移，但被周期性 reset 限制住了。**

理由：

- `stream_impl` 在每 `QWEN_STREAM_RESET_INTERVAL_CHUNKS`（45 轮 = 90 秒）做一次 reanchor，清除 encoder cache 和 decoder prefix，重建 KV 从头开始。此时 prefill 无法复用旧 KV，一轮 prefill 时间会跳到 300-500ms。
- 在 45 轮间隔内，随着音频推进：
  - encoder cache 上限 4 个 window = 32 秒，恒定
  - prefix token 上限 150，恒定
  - KV 复用率逐渐提高（prefill delta 趋小）
  - **decoder 输出长度是变量**：如果模型输出的 text token 越来越多但 max_new_tokens 限制住了，会导致 stagnant_chunks，触发 recovery_reset
- 真正的漂移只会发生在 **encoder 处理赶不上实时** 的场景——在 0.6B 上 Processing RTF≈0.55，encoder 本身不漂移；在 1.7B 上 RTF≈2.6，**严重漂移**

### 2.3 非最优解的证据

1. **chunk_sec = 2.0 是保守默认**：qwen-asr-learn 的 `--stream` 就用 2s，但那是做"能跑"的基线。对字幕场景 1.0-1.5s chunk 更优，encoder cache 可以吸收重复编码开销。
2. **encode → prefill → decode 全串行**：encoder 在等 chunk 凑够的那 2 秒，CPU 完全空闲。这段时间完全可以用来预计算下一轮 encoder 或其他工作。
3. **server 层的 `SharedAsrModel` 带全局 mutex**：即使有 clone ctx (live worker 已用 clone)，`TranscribeRealtime` 和 `TranscribeFile` 共享同一把 `mu_` 锁，意味着批量请求与实时请求互斥。但 live worker 路径已绕过了此锁，只是离线批量会阻塞 clone 创建。

---

## 3. 业界多线程方案对比

### 3.1 sherpa-onnx (k2-fsa)

```
特点：
- OnlineStream 持有独立的 encoder state（无状态 encoder 不需要 KV）
- DecodeStreams(OnlineStream **ss, int n) 接受 N 个流同时解码
- Transducer 架构天然支持 encoder/joiner 分离
- encoder 和 decoder 在同一线程但 batched
```

核心设计：**流级别隔离** + **批量 decode**。多个流共享 model weight 但各持有独立 state，一次 `DecodeStreams` 处理全部活动流。

### 3.2 whisper.cpp

```
特点：
- whisper_state 与 whisper_context 分离（#523）
- whisper_full_parallel() 将音频切段后开 N 个 state 并行处理
- 每个 state 独立持有 KV cache
- encoder/decoder 通过 n_threads 参数控制 BLAS/ggml 并行
- stream 场景没有原生支持，但 examples/stream 通过外部循环 + 滑窗实现
```

核心设计：**context/state 分离** + **段级并行处理**。流式场景通过 `whisper_full()` 反复调用 + 滑窗实现，没有 encoder/decoder pipeline 分离。

### 3.3 vLLM qwen3_asr_realtime

```
特点：
- segment_duration_s = 5.0 的 buffer 收集策略
- encoder 在 GPU 上瞬时完成，不是瓶颈
- 流式输出通过 SSE (Server-Sent Events) 推送
- 真正的并行来自 GPU batch + continuous batching
```

核心设计：**GPU 驱动，encoder 不是瓶颈**。CPU 场景不可直接照搬其策略。

### 3.4 对本项目的适用性分析

本项目的 Qwen3-ASR 在 CPU 上运行，encoder-decoder 都是 transformer，且 **encoder 和 decoder 使用各自独立的权重和 KV cache**。这意味着：

✅ encoder 和 decoder 可以在不同线程上并行——它们不共享可变状态
✅ `qwen_clone_shared()` 已存在且在 live worker 中已使用——clone ctx 共享 weight, 独立 KV
✅ encoder cache (sliding window) 已实现——encoder 输出可以异步生产

---

## 4. 可实施的多线程改造方案

### 方案 A：encoder-decoder 流水线分离（推荐首选）

```
       AudioThread (existing)         EncoderThread (new)        DecoderThread (renamed from live worker)
              │                              │                              │
    收麦克风 PCM ──────────>  ring buffer ──>│                              │
              │                              │─ encode window ──> enc_queue │
              │                              │─ encode window ──> enc_queue │
              │                              │                    ─────────>│
              │                              │                    prefill+decode
              │                              │─ encode partial ─>           │
              │                              │                    ─────────>│
              │                              │                    prefill+decode
              │                              │                              │─> token_cb
```

**关键约束**：
- encoder 只读 audio samples + model weight (shared, immutable)
- decoder 只读 encoder output + decoder weight (shared) + 自己的 KV cache (owned)
- 两者之间通过 `enc_output_queue` (SPSC 无锁队列) 传递
- decoder 无需等 chunk 凑齐——encoder 有新 window 就推

**收益估算**：
- encoder 提前完成窗口编码，decoder 到达时直接取用
- 单轮延迟从 `wait + enc + prefill + dec` 变为 `max(enc_pipeline_latency, prefill + dec)`
- 0.6B 实测：encoder 200ms, prefill+decode 500ms → 原本 700ms 变为约 500ms (因为 encoder 已提前完成)
- 更重要的: **chunk_sec 可以降到 1.0s 甚至 0.5s**，因为 encoder 在后台持续跑，decode 可以更频繁触发

**实现难度**：中等。需要：
1. 将 `stream_impl()` 拆为 `encoder_loop()` 和 `decoder_loop()` 两个函数
2. 添加一个线程安全的 encoder output queue
3. decoder loop 从 queue 取已编码窗口，拼接后做 prefill + decode
4. 音频 ring buffer 双方共享（encoder 读、audio 写，已有 `qwen_live_audio_t` 机制）

### 方案 B：双 ctx 乒乓（进阶）

```
         chunk N              chunk N+1
    ┌──────────────┐     ┌──────────────┐
    │ ctx_A decode │     │ ctx_B decode │
    │ (prefill+AR) │     │ (prefill+AR) │
    └──────────────┘     └──────────────┘
                    ↕ overlap ↕
```

**原理**：两个 clone ctx 交替工作。当 ctx_A 在做第 N 轮 decode 时，ctx_B 可以提前构建第 N+1 轮的 input_embeds 并开始 prefill（但需要 encoder 输出已就绪）。

**约束**：
- 两个 ctx 各自独立 KV cache，需要 2× KV 内存
- 模型权重共享（`qwen_clone_shared` 已就绪）
- 需要维护两个 ctx 之间的 text state 同步

**收益**：理论上可以进一步把 prefill 时间也吃掉，但对 0.6B 来说边际收益不大（prefill 只有 100-300ms），对 1.7B 可能更有意义。

**实现难度**：高。涉及交错调度和 state 同步逻辑。建议先做方案 A，验证收益后再考虑。

### 方案 C：立即可做的非架构改进

不改线程模型，仅调参数：

| 参数 | 当前值 | 建议值 | 效果 |
|------|--------|--------|------|
| UI sendTimer | 800ms | 400ms | 减少音频发送延迟 |
| UI pollTimer | 250ms | 150ms | 更快获取新结果 |
| chunk_sec | 2.0s | 1.0s | live worker 更频繁解码 |
| max_new_tokens | 32 | 24 (chunk_sec=1.0时) | 已有自适应逻辑 |

**预计效果**：观感延迟从 ~4s 降到 ~2.5s，零代码改动。

---

## 5. 为什么 "既然一段时间之外都认为无关的，充分利用多线程" 是对的

用户观察到了一个关键事实：当前 `QWEN_STREAM_MAX_ENC_WINDOWS = 4`（32s）和 `QWEN_STREAM_MAX_PREFIX_TOKENS = 150` 已经把上下文限制在有限窗口内。这意味着：

1. **encoder 处理的是滑动窗口**，旧 window 会被 evict。既然只保留最近 4 个 window，encoder 可以在音频到达后立即开始编码新窗口，不需要等 decoder 完成当前轮。
2. **decoder 的 prefix 有上限**，不会无限增长。prefill 的增量 delta 通常很小（KV reuse 率实测很高），所以 decoder 的计算量是可预测且有界的。
3. **两者之间没有状态耦合**：encoder 不读 decoder 状态，decoder 只读 encoder 的输出（不可变）。

这正是经典的 **生产者-消费者** 模式，天然适合多线程。当前把它们串行塞在一个 while 循环里，等于强制 CPU 在一半时间里闲置（等音频凑够 / 等 encoder 完成才开始 prefill）。

---

## 6. 具体代码改造指引

### 6.1 拆分 stream_impl（方案 A 详细步骤）

**Step 1**: 新增 `stream_enc_queue_t`

```c
typedef struct {
    float *enc_output;     // encoder 输出 [seq_len, dim]
    int seq_len;
    int64_t start_sample;  // 此 window 对应的音频起点
    int is_partial;        // 是否为非完整 window (尾部)
    int is_eof;            // 最终标记
} stream_enc_item_t;

typedef struct {
    stream_enc_item_t *items;
    int capacity;
    int head;              // consumer reads
    int tail;              // producer writes
    // platform mutex + condvar
} stream_enc_queue_t;
```

**Step 2**: 拆出 `encoder_thread_fn`

从 `stream_impl()` 的 while 循环中提取 encoder 部分（line 1735-1870）：
- 等音频（从 live audio 或 ring buffer）
- 执行 `stream_encode_span()`
- 将结果推入 `enc_queue`
- 管理 encoder cache (`enc_cache[]` 数组)

**Step 3**: 拆出 `decoder_thread_fn`

从 `stream_impl()` 保留 prefill + decode + commit 部分：
- 从 `enc_queue` 取已编码窗口
- 拼接 encoder output
- 构建 input_embeds
- prefill + decode
- token commit + callback

**Step 4**: 修改 `qwen_transcribe_stream_live` 启动两个线程

```c
char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live) {
    stream_enc_queue_t queue;
    init_enc_queue(&queue, 16);
    
    // encoder 线程：读 live audio → 编码 → 入队
    thread_t enc_thread;
    thread_create(&enc_thread, encoder_thread_fn, &enc_args);
    
    // decoder 在当前线程运行：出队 → prefill → decode → commit
    char *result = decoder_loop(ctx, &queue, ...);
    
    thread_join(&enc_thread);
    destroy_enc_queue(&queue);
    return result;
}
```

### 6.2 UI 层调参（方案 C 详细改动）

文件 `ui/app.js` line 717-718:

```javascript
// 当前
sendTimer: window.setInterval(() => flushRealtimeChunk(false), 800),
pollTimer: window.setInterval(() => pollRealtimeStatus(), 250),

// 改为
sendTimer: window.setInterval(() => flushRealtimeChunk(false), 400),
pollTimer: window.setInterval(() => pollRealtimeStatus(), 150),
```

服务端 `stream_chunk_sec` 默认值改为 1.0：

文件 `src/service/server.cc` 的 `RealtimeStreamChunkSeconds()`:
```cpp
float RealtimeStreamChunkSeconds(const RealtimePolicyConfig & policy) noexcept {
    // 当前返回 min_decode_interval_ms 转换，默认 800ms → ~0.8s
    // 但 ctx->stream_chunk_sec 默认是 2.0
}
```

需要将 `qwen_asr.h` 中 `stream_chunk_sec` 的默认初始化从 `2.0f` 改为 `1.0f`。

---

## 7. 风险与兜底

| 风险 | 概率 | 缓解 |
|------|------|------|
| 多线程引入 race condition | 中 | encoder 与 decoder 无共享可变状态，queue 是唯一同步点，用标准 mutex+condvar |
| encoder 线程跑太快占满内存 | 低 | queue 设容量上限（如 8 个 window），满则 encoder 阻塞等 decoder 消费 |
| 1.0s chunk 的 decoder max_new_tokens 不够 | 低 | 已有自适应逻辑 `RealtimeStreamMaxNewTokens` 会根据 chunk_sec 调整 |
| 双线程 BLAS 线程争抢 | 中 | encoder 用 N/2 线程，decoder 用 N/2 线程，通过 `openblas_set_num_threads` 动态切换 |
| decode 还没完成 encoder 又推了新 window | 无 | 这正是期望行为——encoder 提前完成是好事 |
| ACCESS_VIOLATION 长时间运行崩溃 | 高 | 已知 issue，独立于多线程改造，需另查 KV cache / buffer 越界 |

---

## 8. 实施优先级

```
Week 1:  方案 C（调 UI 和 chunk_sec 参数）→ 立即验证观感改善
Week 2:  方案 A Step 1-2（定义 queue、拆 encoder 线程）
Week 3:  方案 A Step 3-4（拆 decoder loop、集成测试）
Week 4:  benchmark 0.6B + 1.7B × 新旧管线对比
Future:  方案 B（双 ctx 乒乓，仅 1.7B 需要时）
```

---

## 9. 附录：当前 stream_impl 关键代码位置

| 代码段 | 文件 | 行号 | 说明 |
|--------|------|------|------|
| stream_impl 主函数 | qwen_asr.c | 1437-2407 | 近 1000 行的巨型函数 |
| live audio wait | qwen_asr.c | 1662-1730 | `LA_WAIT` 阻塞等音频凑够 |
| encoder cache 逻辑 | qwen_asr.c | 1775-1870 | 滑窗缓存 + evict |
| prefill + KV reuse | qwen_asr.c | 1990-2030 | memcmp 找 reuse 前缀 |
| decode loop | qwen_asr.c | 2050-2090 | 逐 token forward |
| text commit | qwen_asr.c | 2170-2320 | 稳定前缀 + 去重 + 提交 |
| token callback | server.cc | 2175-2180 | session.mu 锁下追加 stable_text |
| UI send | app.js | 717 | 800ms setInterval |
| UI poll | app.js | 718 | 250ms setInterval |
