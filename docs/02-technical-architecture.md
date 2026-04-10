# 技术架构

日期：2026-04-10

## 架构总图

自下而上凡九层：

1. `core`
2. `platform`
3. `storage`
4. `audio`
5. `model`
6. `inference`
7. `runtime`
8. `protocol`
9. `apps`

## 分层

### `core`

义：

- `Status`
- 时间戳
- 音频基础类型
- 小型无依赖工具

戒：

- 不许知 HTTP
- 不许知 BLAS 具体实现
- 不许知模型细节

### `platform`

义：

- OS / CPU / SIMD 探测
- BLAS 选择
- 文件映射
- 时钟、线程、NUMA 预留

平台律：

- macOS: Accelerate
- Linux: OpenBLAS
- Windows: OpenBLAS

### `storage`

义：

- safetensors mmap
- vocab / config / metadata 装载
- checksum / file existence / shard consistency

### `audio`

义：

- WAV / stdin / future url ingress
- resample
- mel
- silence compaction
- streaming ring buffer

### `model`

义：

- Qwen3-ASR model spec
- tokenizer
- weight registry
- prompt template

### `inference`

子模块：

- `encoder`
- `decoder`
- `kv_cache`
- `streaming_policy`
- `realtime_scheduler`
- `timestamp_provider`

其要：

- encoder 采 per-chunk conv 与 window attention
- decoder 采 Q/K RMSNorm、RoPE、GQA、prefill / step
- streaming 采 prefix rollback 与 stable frontier commit
- realtime 不以停顿为前提；以定长滑窗、局部一致、前缀提交为主
- timestamp 不混入 decoder；以 provider 抽象接 ForcedAligner

### `runtime`

义：

- model pool
- session state
- request queue
- async executor
- backpressure
- cancellation
- metrics
- realtime session loop

并发法：

- `ModelContext`: 只读，共享
- `SessionState`: 可变，独占
- `RequestTask`: 短生
- `WorkerPool`: 长生

### `protocol`

二面并立：

- `openai`
- `vllm`

首版目标：

- OpenAI `/v1/chat/completions`
- OpenAI `/v1/audio/transcriptions`
- OpenAI `/v1/models`
- vLLM-compatible `/v1/chat/completions`
- SSE 流式

后版预留：

- OpenAI `/v1/realtime`

现已实现私有测试面：

- `/api/transcriptions`
- `/api/transcriptions/async`
- `/api/jobs/:id`
- `/api/metrics`
- `/api/realtime/start|chunk|stop`
- `/api/capture/start|stop|status`

### `apps`

三类入口：

- `qasr-cli`
- `qasr-stream`
- `qasr-server`

## 请求流

### 离线

1. 收音频
2. normalize + resample + mel
3. encoder
4. prompt assemble
5. decoder prefill / generate
6. optional aligner
7. format result

### 流式

1. 收 chunk
2. 写 ring buffer
3. 定周期 planner 取滚动窗
4. encoder window cache
5. prefix rollback + bounded decode
6. local agreement / stable frontier commit
7. 分离 `partial` / `stable` / `final`
8. SSE / WS 推送 delta

注：

- VAD 只作 hint，不得作为唯一切段依据。
- 连续无停顿语流，仍须靠滑窗与前缀提交前行。

### timestamp

1. ASR 出 text
2. 流式先给 provisional segment range
3. 若请求 word timestamp，则交 ForcedAligner
4. 统一转 segment / word JSON

## 协议映射

### OpenAI 面

| 路径 | 用途 | 首版策略 |
|---|---|---|
| `/v1/chat/completions` | 多模态音频输入、流式文本 | 支持 |
| `/v1/audio/transcriptions` | 离线转写、timestamp 输出 | 支持 |
| `/v1/models` | 模型发现 | 支持 |
| `/v1/realtime` | 实时转写会话 | 预留；先以私有 `/api/realtime/*` 验证 |

### vLLM 面

| 路径 | 用途 | 首版策略 |
|---|---|---|
| `/v1/chat/completions` | 与官方 Qwen 文档一致 | 支持 |

约束：

- 依官方 Qwen 文档，vLLM streaming 不支持 batch，不支持 timestamps。
- 依官方 vLLM 文档，Realtime 走 `append/commit -> delta/done` 语义。

## 近实时字幕架构

三轨并行：

1. `partial lane`
2. `stable lane`
3. `final lane`

其义：

- `partial lane` 追最低感知时延；可改写末尾少量字。
- `stable lane` 只提交连续两轮以上一致之公共前缀。
- `final lane` 在 stop / flush / 对齐完成后给最终文本与时间戳。

初始参数律：

| 项 | 初值 |
|---|---|
| 输入聚合粒度 | `160~320 ms` |
| 解码触发周期 | `640~960 ms` |
| 流式 chunk | `2.0 s` |
| encoder window | `8.0 s` |
| encoder 历史上限 | `32 s` |
| 服务端 PCM 保留上限 | `32 s` |
| decoder rollback | `5 token` |
| decoder prefix cap | `150 token` |
| 冷启 unfixed chunks | `2` |
| 每步新生 token 上限 | `32` |
| 强制冻结上限 | `12~15 s` 未决尾巴不得再长 |

这些值取自二源折中：

- `qwen-asr-learn` 之 `2s + 8s + rollback 5 + cap 150`
- vLLM realtime 之 `append/commit` 会话义

后续须由基准压测再调。

## CPU 优化路线

必做：

- BF16 权重 mmap
- prefill / decode 分路
- encoder window cache
- decoder prefix cap
- OpenBLAS / Accelerate GEMM
- x86 AVX2/AVX512
- ARM NEON

次做：

- cache-aware blocking
- pinned arena
- per-thread scratch
- zero-copy protocol buffers

不先做：

- GPU first path
- 复杂 graph runtime

## 内存所有权

| 对象 | 所有者 | 释放时机 |
|---|---|---|
| model weights mmap | `ModelContext` | model unload |
| tokenizer vocab | `ModelContext` | model unload |
| session KV cache | `SessionState` | session close |
| streaming audio ring | `SessionState` | session close |
| request payload | `RequestTask` | response finish |
| aligner context | `TimestampProvider` | provider unload |

律：

- 不裸露 `new/delete`
- allocator 接口统一
- 任一缓冲区须写“谁分配、谁释放、何时释放”

## 构建矩阵

| 平台 | 编译器 | BLAS | 验证法 |
|---|---|---|---|
| macOS arm64 | Apple Clang | Accelerate | 本机直编 |
| Linux arm64 | Clang/GCC | OpenBLAS | Docker `linux/arm64` |
| Linux amd64 | Clang/GCC | OpenBLAS | Docker `linux/amd64` |
| Windows amd64 | MSVC/clang-cl | OpenBLAS | 原生或 CI runner |

## 第一阶段落地

本日先落：

- CMake 与平台 BLAS 规则
- `core` / `runtime` / `protocol` 骨架
- 单测基座
- Docker Linux/OpenBLAS 验证脚本

未落：

- 真模型推理核
- ForcedAligner 接入

已落：

- CLI 真转写
- HTTP 服务
- OpenAI / vLLM 兼容基础面
- async job
- 本地 UI
- Linux host capture backend 骨架
