# Qwen-ASR 共性优化方案：最佳实践与可实施路线（对比实施版）

> 目的：整理一份**对不同参数体量模型都成立**、且兼容**实时流 / 离线批处理 / 单机 PC / 工业服务器 / 边缘板 / GGUF 导出**的优化路线图。本文优先覆盖**全局都可受益**的优化，再列出可分场景启用的增强路线，便于逐项实验对比。

---

## 0. 结论先行

如果目标是先做“全局都可受益”的优化，优先级应当是：

1. **统一层描述、权重准备、scratch/arena 与 profiling 框架**
2. **把 prefill 热层的 BF16→F32 展开移出热路径，前移到加载期或首次使用时**
3. **先做少量热点层（优先 QKV，再看 Gate/Up）的持久化表示，而不是一上来全模型持久化**
4. **对 Intel 平台，优先评估 oneDNN 的 MatMul + pre-pack/reorder 能力；OpenBLAS 继续作为跨平台基线**
5. **量化先从翻译侧开始，再做 ASR 的 prefill-only weight-only INT8/Q8**
6. **GGUF 作为导出/边缘部署格式或对照格式，而不是当前热路径优化的第一抓手**

如果只允许先做一轮投入最小、收益最稳的改造，建议顺序是：

- `LinearSpec + PreparedWeight + RuntimeProfile + ScratchArena`
- `QKV prefill 持久化`
- `GateUp prefill 持久化`
- `translator int8`
- `ASR prefill-only INT8/Q8`

---

## 1. 当前项目中已观察到的关键现状（基于更新后的仓库）

以下观察来自当前上传仓库的源码结构：

- decoder prefill 已接入 `qwen_linear_nobias_bf16_qkv_prefill(...)`，QKV 的调用次数问题已部分解决。
- 但 `qwen_linear_nobias_bf16_qkv_prefill(...)` 仍在热路径里：
  - 把 `Wq/Wk/Wv` 转成一个新的 `W_fused`
  - 用 `cblas_sgemm(...)` 执行 GEMM
  - 再拆分输出
- `qwen_linear_bf16(...)` / `qwen_linear_nobias_bf16(...)` 在 `seq_len > 1` 时仍通过 `bf16_get_f32_view(...)` 获取 F32 视图。
- `QWEN_BF16_CACHE_MB` 默认仍关闭，当前 cache 本质仍是“展开缓存”，不是“加载期一次性预处理、推理期全程复用”的持久化权重。
- `qwen_asr_kernels.c` 中仍存在静态全局 `qkv_out` scratch；`encoder` 中也存在 `malloc/free` 型临时 `qkv_out`。
- OpenBLAS 路径已经接通，项目中也已有性能统计与 OpenBLAS 条件编译入口。

这意味着：

- 第一阶段“减少调用次数”的工作已经起步；
- 当前主瓶颈已转向**固定权重重复变换**与**中间表示生命周期管理**；
- 下一步最有价值的不是继续做细碎 kernel micro-opt，而是升级到**统一权重准备机制**与**持久化运行时表示**。

---

## 2. 总体设计原则

### 2.1 同核异策

离线批处理与实时数据流应共享同一套核心执行框架，但在以下方面允许 profile 级差异：

- 权重准备范围
- 权重表示格式
- 内存预算
- 调度策略
- 线程策略
- 量化策略

### 2.2 框架必须与模型体量解耦

不能写出只适配某个固定 hidden size / intermediate size / 层数的优化路径。正确做法是：

- 框架层通用
- 模型差异体现在元数据
- 实时 / 离线 / 边缘 / 服务器差异体现在 profile

### 2.3 先消除“工程浪费”，再做“数值近似”

当前最值得先做的，是消除：

- 热路径 BF16→F32 重复展开
- 每轮重新构造 fused 权重
- 静态全局 scratch
- 没有统一 arena / profiling 的问题

这些改造在**不引入精度变化**的前提下就能对所有模型受益。

---

## 3. 公开资料中最值得参考的几条最佳实践路线

## 路线 A：统一“层描述 + 权重准备 + profile”框架（最先做）

### 核心思想

建立与模型体量无关的统一抽象：

- `LinearSpec`
- `PreparedWeight`
- `RuntimeProfile`
- `PrefillPlan / DecodePlan`
- `ScratchArena`

### 为什么它是第一优先级

因为后续所有优化——包括持久化 F32、INT8/Q8、GGUF 导出、边缘板低内存版、服务器多实例——都需要统一入口，否则代码会很快按场景分叉。

### 建议最小结构

```c
// 线性层角色
QKV_FUSED / WO / GATE_UP_FUSED / DOWN / OTHER

// 运行时权重格式
RAW_BF16 / PERSIST_F32 / PACKED_INT8 / PACKED_Q8 / BACKEND_NATIVE

// profile
realtime / offline / server_balanced / edge_lowmem
```

### 预期收益

- 直接收益：中等
- 中长期收益：最高
- 风险：低
- 通用性：最高

### 适用场景

- 全部场景
- 全部模型体量

---

## 路线 B：加载期或首次使用时，准备 prefill 热层的持久化表示（第二优先级）

### 核心思想

把 prefill 热层的 BF16→F32 展开从“每个 chunk 热路径”迁出，改成：

- 加载期一次性准备
- 或首次使用时懒准备 + 后续复用

### 第一阶段建议只做哪些层

优先顺序：

1. `QKV fused`
2. `GateUp fused`
3. 视内存与收益再考虑 `WO / DOWN`

### 为什么这条路重要

你当前的热点损耗主要来自固定权重反复变换，而不是“算子根本没有融合”。

### 注意点

不要一开始就对**全部热层**做永久 F32 副本；这会把“工程优化”变成“粗暴用内存换时间”。

建议按档位推进：

- **档位 1**：只做 `QKV fused`
- **档位 2**：加 `GateUp fused`
- **档位 3**：再考虑 `WO / DOWN`

### 预期收益

- 对 0.6B：通常是最有希望拿到明显收益的第一刀
- 对 1.7B：同样有效，但要严格控制额外内存

### 风险

- 单机 PC：中等，可控
- 多实例服务器：需要预算
- 边缘板：应谨慎，通常不适合作为 F32 主方案

---

## 路线 C：Intel 平台用 oneDNN 做 MatMul + 权重预打包（开源，优先于闭源 BLAS 特化）

### 为什么值得认真评估

oneDNN 是开源、跨平台的深度学习加速库，支持 CPU/GPU，自动 ISA 选择，并把 memory、primitive、reorder 作为一等概念暴露出来。Intel 的 oneDNN 文档明确建议：

- 对可复用输入使用 `format_tag::any`
- `create primitive once, use multiple times`
- 通过 reorder / pre-pack 让库选择最优布局
- 对压缩权重 matmul 提供显式示例

### 这意味着什么

如果你的目标是解决：

- 固定权重重复展开
- 后端友好的 packed weight
- 将来 prefill-only INT8/Q8

那么 **oneDNN 比纯 CBLAS 更接近你真正需要的接口层级**。

### 最适合切入的部分

不要一上来替换全部 BLAS 路径。建议只切最有价值的子集：

1. `QKV fused prefill`
2. `GateUp fused prefill`
3. 视情况扩到 `WO / DOWN`

decode 继续保持当前自写 AVX2 / BF16 matvec 核心，不必急着迁移。

### 适用范围

- Intel PC：高价值
- Intel 服务器：高价值
- macOS / ARM：不作为主路
- 边缘板：通常不作为第一优先

### 风险

- 工程复杂度明显高于“持久化 F32”
- 但这是中期最正确的开源 Intel 路线

---

## 路线 D：跨平台基线继续保留 OpenBLAS，但要按最佳实践构建和使用

### OpenBLAS 侧的公开最佳实践

OpenBLAS 官方建议在面向多硬件分发时启用：

- `DYNAMIC_ARCH=1`
- CBLAS 接口
- 合理的线程配置

运行时建议重点关注：

- `OPENBLAS_NUM_THREADS`
- `OPENBLAS_VERBOSE=2`
- `OPENBLAS_CORETYPE`
- 避免过度线程数

### OpenBLAS 在本项目中的合理定位

- 继续作为**跨平台基线后端**
- 继续作为**fallback 路径**
- 对 Apple/ARM 和非 Intel 服务器仍有价值

但要明确：

- OpenBLAS 适合作为“当前生产基线”
- 不应被视为解决持久化 packed weight 问题的最终方案

### 结论

保持 OpenBLAS 是对的，但不要把所有长期优化目标都压在 OpenBLAS 的 CBLAS API 上。

---

## 路线 E：量化先从翻译侧开始，再做 ASR prefill-only 量化

### 结论

如果要求“精度只允许有限下降”，最佳实践顺序不是先量化 ASR 主链路，而是：

1. **翻译模型先做 INT8**
2. **ASR 先做 prefill-only weight-only INT8 / Q8**
3. **decode 保守，尽量保持 BF16 / 高质量路径**

### 为什么翻译侧先做

翻译侧通常：

- 模型更小
- 句段更短
- 量化后的质量损失更可控
- 对端到端延时贡献更容易压到很小

CTranslate2 官方建议在 CPU 上优先使用 `int8`，并给出 CPU 线程、beam、batch 的调优建议。它支持 MarianMT、M2M100、NLLB 等翻译模型。对于 x86-64，预编译包会在 Intel 上优先使用 MKL，因此如果坚持纯开源运行时，应考虑自编译或直接使用 Marian 原生方案。

### ASR 为什么只建议先动 prefill

因为：

- prefill 更像大块计算，量化收益更直接
- decode 是逐 token 小矩阵 / matvec，数值敏感性和调度敏感性更高
- 先做 prefill-only 量化，更容易在不明显伤精度的前提下获得收益

### 推荐顺序

- `translator int8`
- `ASR: QKV prefill int8/q8`
- `ASR: GateUp prefill int8/q8`
- 再视精度和收益考虑是否扩到 `WO / DOWN`

### 不建议第一阶段就做的事

- 不建议把 Q4 当作工业主线首选
- 不建议先做全模型统一低比特

---

## 路线 F：翻译模型选择——生产级 CPU 最佳实践

### 推荐优先级

#### 方案 1：Marian / OPUS-MT（热点方向在线主翻译）

Marian 是纯 C++、MIT 许可，官方明确说明其已用于 Microsoft Translator 与其他公司/研究项目的生产翻译服务。对工业 CPU 场景，这是最稳妥的在线翻译主引擎之一。

适合：

- 中↔英
- 日↔英
- 英↔法
- 英↔西
- 英↔德

注意：OPUS-MT 的**模型许可并不统一**，发布前必须按方向审计。示例：

- `opus-mt-ja-en`：Apache-2.0
- `opus-mt-en-zh`：Apache-2.0
- `opus-mt-zh-en`：CC-BY-4.0

#### 方案 2：M2M100 418M（多语回退）

M2M100 418M 许可为 MIT，支持 100 种语言、9900 个翻译方向。它更适合：

- 热点方向之外的回退
- 运维简化
- 单模型覆盖多语

但在 CPU 上，它通常不应作为所有在线请求的默认翻译器，尤其在资源紧张设备上。

### 实用建议

- **实时主翻译**：热点方向优先 Marian/OPUS-MT
- **长尾回退**：M2M100 418M
- **边缘板**：优先单方向 Marian/OPUS-MT，不建议默认 M2M100

---

## 路线 G：Qwen3-ASR 的 ONNX / GGUF 路线如何定位

### ONNX CPU 路线

已有社区 ONNX CPU 版本表明：

- Qwen3-ASR-0.6B 的 encoder + decoder 都可跑在 ONNX Runtime
- decoder 可做 INT8
- 在 Intel N100 这类低功耗 CPU 上可做到接近实时或实时

这条路线的意义是：

- 它证明了 0.6B 的 CPU 压缩部署是现实的
- 它可作为你当前 C/OpenBLAS 路线的重要对照组
- 若某些算子难以继续在手写运行时里优化，可考虑借鉴 ONNX 路线的量化边界与模型拆分方式

### GGUF 路线

Qwen3-ASR-0.6B 已有社区 GGUF 版，说明：

- GGUF 适合做导出格式
- 适合做边缘/实验/对照部署
- 适合接入 ggml / 自定义小运行时

但当前阶段不建议把 GGUF 当作“热路径优化主线”。

因为：

- 导出 GGUF 本身不会自动加速你当前运行时
- 若运行时仍然在热路径中重新展开权重，则 GGUF 只能改善文件体积和存储，不会从根上解决关键瓶颈

### 建议定位

- **ONNX**：对照路线、边缘 CPU 路线、量化参考路线
- **GGUF**：导出格式、边缘/实验路线、对照路线
- **当前主线**：仍应以现有 C/OpenBLAS 框架的“统一权重准备 + 持久化运行时表示”优化为主

---

## 路线 H：实时流与离线批处理应如何分化（但仍共享同一框架）

### `realtime_profile`

目标：

- 低首包延时
- 低句段滞后
- 优先热层
- 内存预算保守

建议策略：

- 默认只准备 `QKV fused`
- 视内存再开 `GateUp fused`
- 翻译只处理稳定短句
- 翻译优先 int8
- decode 侧量化保守

### `offline_profile`

目标：

- 最大吞吐
- 可接受更高内存
- 可接受更长初始化

建议策略：

- `QKV + GateUp + Wo + Down` 都可进入持久化准备候选
- 允许更大的 prepared-weight 预算
- 支持更激进的 prefill-only INT8/Q8
- 允许句段合并 / micro-batch

### `server_balanced_profile`

目标：

- 适合多实例部署
- 控制每实例额外内存
- 保持端到端延时稳定

建议策略：

- 每实例只准备最热点层
- worker-local prepared weights
- worker-local scratch arena
- 多 session 靠多实例，不靠单实例极限并发

### `edge_lowmem_profile`

目标：

- 严格控制 RAM
- 允许更高句段延时
- 优先低内存表示

建议策略：

- 不做大规模 F32 持久化
- 优先 INT8/Q8 packed
- translator 只放一个方向
- 不做长尾多语全覆盖

---

## 路线 I：更前沿但第二阶段再考虑的增强项

以下路线有价值，但不应作为第一阶段主线：

### 1. LP-GEMM / 布局传播

2026 年的 LP-GEMM 工作指出：顺序 GEMM 中重复 pack/unpack 会造成明显浪费，并在 x86 上报告了相对 OpenBLAS 的显著加速。这条路线说明：你当前“把 fused 结果仍然恢复成标准布局，再下一层重新打包”的问题，长期是可以继续深挖的。

### 2. SparAMX / 稀疏 + AMX

适合 Intel 新平台、适合 decode 侧更激进的 CPU 优化。但这属于第二阶段研究路线，不是当前这条框架优化的第一抓手。

### 3. FBGEMM 风格的 shape-specific INT8 kernel

FBGEMM 公开资料与文档都强调：

- 低比特服务器推理
- shape-specific runtime codegen
- 原生 tensor format
- runtime-specific kernels

这是第二阶段“prefill-only INT8/Q8”的重要参考方向，但不建议一开始就把整个项目切换到这一体系。

---

## 4. 推荐实施顺序（建议直接照此做实验）

## Phase 1：搭框架，不改数值

### 目标

先把“全局都可受益”的抽象搭起来。

### 建议实施项

1. 新增 `LinearSpec`
2. 新增 `PreparedWeight`
3. 新增 `RuntimeProfile`
4. 新增 `ScratchArena`
5. 统一 profiling 输出

### 成功标志

- 0.6B / 1.7B 都能加载与运行
- 实时 / 离线只切 profile，不切核心代码路径

---

## Phase 2：先做最小高收益持久化

### 目标

先把最大且最共性的热路径浪费消掉。

### 建议实施项

1. `QKV fused -> PERSIST_F32`
2. prefill 时直接吃 prepared weight
3. 不再在热路径重新构造 `W_fused`
4. `qkv_out` 改成 arena 管理

### 成功标志

- 0.6B 与 1.7B 都能明显减少 prefill 中 BF16→F32 展开时间
- 代码结构不出现按模型体量分叉

---

## Phase 3：扩到 GateUp，并引入翻译侧 int8

### 建议实施项

1. `GateUp fused -> PERSIST_F32`
2. translator 切到 Marian/OPUS-MT 或自编译 CTranslate2 的 int8 路线
3. 翻译只处理 stable segment

### 成功标志

- 0.6B 主链路接近准实时翻译
- translator 不再是主瓶颈

---

## Phase 4：引入 prefill-only INT8 / Q8

### 建议实施项

1. `QKV fused -> PACKED_INT8/Q8`
2. `GateUp fused -> PACKED_INT8/Q8`
3. 先只在 prefill 使用
4. decode 继续走 BF16 / 高质量路径

### 成功标志

- 0.6B 在句段级实时翻译上明显更稳
- 1.7B 在离线/异步精修上吞吐改善

---

## Phase 5：再考虑 oneDNN / ONNX / GGUF 深度分支

### 目标

在核心框架稳定后，再探索更强 backend 或更强导出格式。

### 可选路线

- Intel 专属：oneDNN MatMul + reorder/prepack
- 对照部署：Qwen3-ASR ONNX CPU
- 边缘/实验：GGUF 导出
- 高级 INT8：FBGEMM / shape-specific kernel

---

## 5. 建议做的对比实验矩阵

## A. 权重准备策略对比

| 实验项 | 描述 | 目标 |
|---|---|---|
| baseline | 当前实现 | 基线 |
| QKV persist | 仅 QKV fused 持久化 F32 | 观察最大直接收益 |
| QKV + GateUp persist | 扩大热点层覆盖 | 观察边际收益 |
| all hot layers persist | 含 Wo/Down | 判断内存换时间上限 |

## B. 运行时后端对比

| 实验项 | 描述 | 目标 |
|---|---|---|
| OpenBLAS baseline | 当前跨平台基线 | 基准 |
| oneDNN-QKV | 仅 QKV prefill 使用 oneDNN | 验证 Intel 预打包收益 |
| oneDNN-QKV+MLP | QKV + GateUp 用 oneDNN | 评估迁移成本 |

## C. 量化对比

| 实验项 | 描述 | 目标 |
|---|---|---|
| translator int8 | 仅翻译模型量化 | 快速降低端到端延时 |
| ASR prefill int8-qkv | 仅 QKV prefill 量化 | 低风险切入 |
| ASR prefill int8-qkv+gateup | 扩到 MLP | 观察进一步收益 |
| full-model low-bit | 暂不作为第一阶段主线 | 只做边界测试 |

## D. 部署 profile 对比

| profile | 重点 | 核心关注 |
|---|---|---|
| realtime | 低延时 | TTFT / 句段滞后 |
| offline | 高吞吐 | RTF / throughput |
| server_balanced | 多实例稳定 | per-instance RSS / latency jitter |
| edge_lowmem | 内存受限 | RSS / OOM 风险 / 句段稳定性 |

---

## 6. 不建议优先投入的方向

1. **先写固定 0.6B 尺寸的特化 fast path**
2. **先做全模型低比特量化**
3. **先做 GGUF 主线化**
4. **先为边缘板牺牲主框架结构**
5. **先全面迁移所有 BLAS 调用到新后端**

这些路线要么过早特化，要么会把当前最关键的工程浪费问题掩盖掉。

---

## 7. 我的最终建议

如果你只打算先做一轮最稳、最通用、全局都受益的优化，请按下面的最小闭环实施：

1. **统一抽象**：`LinearSpec + PreparedWeight + RuntimeProfile + ScratchArena`
2. **统一 profiling**：层级 / 模型级 / 内存级指标
3. **QKV 持久化**：先做 `QKV fused -> PERSIST_F32`
4. **GateUp 持久化**：第二步再加
5. **translator int8**：先压翻译侧
6. **ASR prefill-only INT8/Q8**：第二阶段再做

这条路线的最大优点是：

- 先服务所有模型体量
- 先服务所有场景
- 不引入明显精度变化
- 后续扩展到 oneDNN / ONNX / GGUF / 边缘板时都不会推倒重来

---

## 8. 参考资料（公开资料，建议实现前逐条核对）

### Qwen3-ASR

- Qwen3-ASR-0.6B 模型卡：
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Qwen3-ASR 0.6B GGUF（社区转换）：
  - https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF
- Qwen3-ASR 0.6B ONNX CPU（社区方案）：
  - https://huggingface.co/wolfofbackstreet/Qwen3-ASR-0.6B-ONNX-CPU

### oneDNN / Intel 开源路线

- oneDNN 总览：
  - https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html
- oneDNN MatMul primitive：
  - https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.0-rev-3/elements/onednn/source/primitives/matmul
- oneDNN MatMul weights decompression / pre-packing 教程：
  - https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-1/matmul-tutorial-weights-decompression.html

### OpenBLAS

- OpenBLAS 运行时变量：
  - https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/
- OpenBLAS 分发与 `DYNAMIC_ARCH` 建议：
  - https://www.openmathlib.org/OpenBLAS/docs/distributing/
- OpenBLAS 用户手册：
  - https://www.openmathlib.org/OpenBLAS/docs/user_manual/

### 翻译与运行时

- Marian 官方站点：
  - https://marian-nmt.github.io/
- Marian 文档：
  - https://marian-nmt.github.io/docs/
- CTranslate2 支持的模型：
  - https://opennmt.net/CTranslate2/guides/transformers.html
- CTranslate2 性能建议：
  - https://opennmt.net/CTranslate2/performance.html
- CTranslate2 量化文档：
  - https://opennmt.net/CTranslate2/quantization.html
- CTranslate2 硬件支持：
  - https://opennmt.net/CTranslate2/hardware_support.html
- M2M100 418M：
  - https://huggingface.co/facebook/m2m100_418M
- OPUS-MT（示例）：
  - https://huggingface.co/Helsinki-NLP/opus-mt-ja-en
  - https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
  - https://huggingface.co/Helsinki-NLP/opus-mt-zh-en

### 前沿参考（第二阶段）

- LP-GEMM：
  - https://researchtrend.ai/papers/2604.04599
- Efficient LLM Inference on CPUs：
  - https://huggingface.co/papers/2311.00502
- SparAMX：
  - https://huggingface.co/papers/2502.12444
- FBGEMM 文档：
  - https://docs.pytorch.org/FBGEMM/fbgemm/index.html
- FBGEMM 工程文章：
  - https://engineering.fb.com/2018/11/07/ml-applications/fbgemm/

---

## 9. 给实现者的简短执行建议

如果你只想先做一轮，不要犹豫，直接按这个最小闭环来：

1. `LinearSpec`
2. `PreparedWeight`
3. `RuntimeProfile`
4. `ScratchArena`
5. `QKV persist`
6. `GateUp persist`
7. `translator int8`
8. 再决定是否走 `ASR prefill-only INT8/Q8`

先把这几项做完，再讨论“GGUF 主线”“边缘板主线”“全模型低比特主线”，工程风险会低很多。
