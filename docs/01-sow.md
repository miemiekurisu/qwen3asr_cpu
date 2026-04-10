# SOW

版本：v0.1
日期：2026-04-10

## 目标

以 C++17 写一套生产级 Qwen3-ASR CPU 推理框架。首务如下：

- 跨平台：Windows / Linux / macOS。
- BLAS：Windows、Linux 用 OpenBLAS；macOS 用 Accelerate。
- 能力：离线、流式、异步服务、OpenAI 兼容、vLLM 兼容、timestamp 输出。
- 约束：`qwen-asr-learn` 与 `whisper.cpp` 仅供参考，不可改。

## 非目标

本阶段不以 GPU 为先，不以 GUI 为先，不以移动端为先。

## 范围

### P0 文档与骨架

交付：

- 参考剖析文档
- 设计文档
- 技术架构文档
- 开发规范文档
- 经验文档
- 新工程骨架与测试基座

验收：

- 文档齐
- 工程可配置、可编、可测
- 二参考目录不受触动

### P1 基础运行时

交付：

- 错误模型
- 配置模型
- 任务模型
- session / queue / async executor 骨架
- OpenAI / vLLM 协议映射骨架

验收：

- 单测全过
- sanitizer 首轮可跑
- API 边界定稿

### P2 基础模型面

交付：

- safetensors mmap loader
- tokenizer
- audio frontend
- VAD / silence compaction

验收：

- 与参考输出对齐
- 回归样本通过
- 内存泄漏为零

### P3 推理核

交付：

- encoder
- decoder
- KV cache
- prefill / step decode
- CPU kernel 与 BLAS 路由

验收：

- 0.6B、1.7B 皆可跑
- 正确率达参考基线
- CPU 资源占用稳定

### P4 流式与 timestamp

交付：

- chunked streaming
- prefix rollback
- stable frontier commit
- ForcedAligner 接口
- segment / word timestamp

验收：

- 流式文本稳定
- offline timestamp 可用
- streaming 与 timestamp 约束清楚

### P5 服务化

交付：

- OpenAI `/v1/chat/completions`
- OpenAI `/v1/audio/transcriptions`
- Realtime 预留面
- vLLM 兼容 chat/completions 面
- async queue / backpressure / metrics

验收：

- 协议样例通过
- SSE 流式可用
- 高并发不崩

### P6 生产化

交付：

- Docker 验证脚本
- Linux amd64 / arm64 验证
- Windows OpenBLAS 验证
- 文档漂移检查

验收：

- 构建矩阵通过
- 回归稳定
- 文档与实现一致

## 里程碑验收线

| 里程碑 | 必达线 |
|---|---|
| M0 | 文档齐，骨架可编可测 |
| M1 | tokenizer/audio/safetensors 正确 |
| M2 | offline 推理正确 |
| M3 | streaming 正确且稳定 |
| M4 | timestamp 可交付 |
| M5 | OpenAI / vLLM 服务可交付 |
| M6 | 三平台验证齐 |

## 风险

| 风险 | 说明 | 对策 |
|---|---|---|
| 官方文档漂移 | vLLM recipe 与模型卡已见不一致 | 以官方模型卡为主，recipe 为辅，落经验文档 |
| OpenBLAS 安装差异 | Windows / Linux 路径各异 | CMake 明确约束，Docker 做 Linux 验证 |
| CPU 性能不足 | 1.7B 对长音频压强大 | 先优化 0.6B；流式、分段、cache 全上 |
| 时间戳能力分裂 | ASR 与 ForcedAligner 为两模型 | 抽统一 timestamp provider |

## 甲乙边界

此处仅记工程边界：

- 参考项目不改。
- 若官方协议有新变，先改文档，再改实现，再回归。
- 任一模块未过单测、集成、回归，不入主线。
