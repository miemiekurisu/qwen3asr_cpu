# CUDA 接入规划（仅规划，暂不改码）

日期：2026-04-11

关联文档：

- [audit20260411.md](audit20260411.md)
- [audit20260411-realtime-pipeline.md](audit20260411-realtime-pipeline.md)
- [04-development-standard.md](04-development-standard.md)

---

## 1. 文档目的

本文仅用于落地 CUDA 接入规划，不包含任何代码改动，也不主张在当前阶段直接开始重构。

本规划基于对当前仓库的行级阅读，目标是：

1. 在**不破坏现有 CPU 跨平台主路径**的前提下，为 CUDA 后端预留正式入口。
2. 保持 **CPU 仍为默认路径**，CUDA 为**编译时可选**能力。
3. 尽量减少对现有实时链路、显示语义与 CPU 特化优化的冲击。
4. 避免“表面抽象、实际仍耦合在 CPU 内核里”的伪改造。

---

## 2. 当前项目的真实现状

## 2.1 现有项目并不是“多后端架构”

虽然仓库中已有 `runtime`、`inference`、`service` 等分层，但当前真正可运行的 ASR 主路径仍然主要落在 `src/backend/qwen_cpu/`。

直接证据：

- `CMakeLists.txt` 当前只有 `QASR_ENABLE_CPU_BACKEND`
- `cmake/QasrBlas.cmake` 只处理 Accelerate / OpenBLAS
- `src/runtime/model_bridge.cc` 直接调用 `qwen_load` / `qwen_transcribe` / `qwen_transcribe_stream`
- `src/service/server.cc` 直接持有 `qwen_ctx_t*`

结论：

- 当前并没有稳定的 backend 抽象层
- C++ 层并未真正把 CPU 后端隔离出去
- 如果直接把 CUDA 逻辑塞入现有 CPU C 核，会迅速扩大改动面

## 2.2 C++ inference 子层目前不是主执行路径

仓库中已有：

- `src/inference/encoder.cc`
- `src/inference/decoder.cc`
- `src/inference/streaming_policy.cc`

但从当前代码路径看，这几层仍偏“骨架/占位”，而非线上主推理链。

因此：

- 不宜把 CUDA 首次接入建立在这些占位层之上
- 否则会形成“文档上分层了，实际服务仍绕过它”的双轨结构

## 2.3 实时显示语义与算核并非同一层

实时文本展示语义主要在：

- `src/service/realtime.cc`
- `src/service/server.cc`

这里负责：

- `partial / stable / final` 区分
- 稳定前缀推进
- 强制冻结
- display snapshot 组织
- recent segments + live tail 的展示拼装

这部分是用户感知最直接的实时体验层。

结论：

- CUDA 接入不应优先改动 `realtime.cc`
- 如果只做吞吐加速，但破坏 token 级稳定提交语义，实时体验会退化

## 2.4 当前实时流式内核已包含大量 CPU 特化策略

`src/backend/qwen_cpu/` 当前不只是“纯推理代码”，还包含大量围绕 CPU 做的专项处理：

- encoder 权重 BF16 -> F32 的加载期转换
- encoder QKV 打包
- decoder prefill prepared F32 快路
- Gate/Up 融合权重
- BF16 展开缓存
- 自建线程池
- AVX2 / NEON / generic 多分支核
- streaming encoder cache
- prefix rollback / reuse prefill / reanchor / periodic reset
- 背景 encoder 线程与 decode overlap

这意味着：

- CUDA 接入不能只理解“矩阵乘搬到 GPU”
- 必须先区分哪些是“CPU 专用优化”，哪些是“流式语义本身”

## 2.5 文档与源码已有轻度漂移

现有 `audit20260411-realtime-pipeline.md` 中把“背景 encoder 线程”描述为后续方向，但源码中实际已经有部分实现。

因此在 CUDA 规划阶段，需要坚持：

1. 以源码现状为准
2. 文档只作参考，不作事实来源
3. 改造前先校正“文档—代码—测试”三方一致性

---

## 3. 本次 CUDA 规划的边界

## 3.1 必须满足

1. **CPU 默认不变**
2. **CUDA 编译时可选**
3. **无 CUDA 环境时可完整构建与运行 CPU 版**
4. **实时显示语义不退化**
5. **尽量不动现有 UI / 协议层**
6. **尽量不破坏现有 CPU hack / trick 的收益**

## 3.2 明确不做

首轮规划不主张：

1. 直接把整个 `qwen_ctx_t` 改造成 CPU/GPU 混合巨型结构
2. 直接重写整个 streaming 算法
3. 直接上“全模型全链路 GPU 化”
4. 直接把 CUDA 硬塞进当前 placeholder 的 C++ inference 骨架
5. 为了支持 CUDA 反向污染现有 CPU-only 构建

---

## 4. 建议的总体路线

总体原则：

**先抽象，再接入；先保守落地，再逐步深化。**

推荐分为三层推进：

1. 构建层
2. 运行时后端抽象层
3. CUDA 后端实现层

---

## 5. 构建层规划

## 5.1 新增编译选项

建议新增：

```cmake
option(QASR_ENABLE_CUDA_BACKEND "Enable CUDA backend" OFF)
```

保留现有：

```cmake
option(QASR_ENABLE_CPU_BACKEND "Enable CPU backend" ON)
```

原则：

- CPU 后端默认开启
- CUDA 后端默认关闭
- 二者允许同时编译
- 无 CUDA 时不影响现有 CPU 构建

## 5.2 CMake 探测策略

建议通过 `find_package(CUDAToolkit)` 接入，而不是把本机路径硬编码到主构建逻辑中。

本机可参考路径：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2
```

但仓库内的规划应只写成：

- 优先使用 `CUDAToolkit_ROOT`
- 否则依赖标准 `find_package(CUDAToolkit)` 搜索路径

## 5.3 Preset 规划

建议补充：

- `windows-cuda`
- 如有需要，再补 `linux-cuda`

同时保留：

- `windows-openblas`
- `linux-openblas`
- `macos-accelerate`

原则：

- 不改变现有 CPU 预设
- CUDA 预设单独存在
- 让 CI 和本地调试都能明确区分 CPU / CUDA 构建物

---

## 6. 运行时抽象层规划

## 6.1 为什么必须先做这一层

当前 `model_bridge` 与 `server` 直接耦合 `qwen_ctx_t*`。  
若不先抽象，后续所有 CUDA 代码都会继续依附到 CPU 专用上下文上，导致：

- 类型继续膨胀
- 生命周期继续混乱
- clone / share 语义继续只围绕 CPU 设计
- service 层无法干净选择后端

因此首个必要改造不是 CUDA kernel，而是**后端边界**。

## 6.2 建议新增的核心概念

建议引入以下概念：

- `AsrBackendKind`
  - `cpu`
  - `cuda`

- `CompiledBackendSet`
  - 当前编译进来的后端集合

- `BackendCapabilities`
  - 是否支持 realtime
  - 是否支持 streaming clone
  - 是否支持 timestamps
  - 是否支持 token callback
  - 是否支持 shared model

- `BackendModel`
  - 持有共享权重与设备资源

- `BackendSession`
  - 单次离线推理会话

- `BackendStreamSession`
  - 实时流式会话

## 6.3 抽象边界建议

建议把当前 `server` 中以下直接依赖替换为抽象接口：

- 模型加载
- 离线转写
- 实时转写
- realtime clone
- 性能信息查询

不建议把实时文本状态机一起塞进后端层。  
实时文本状态机仍应留在 service 层，后端只负责提供：

- 原始文本
- 稳定 token 回调
- 时间戳（如支持）
- 性能计数

这样可以保住现有 UI / API 语义层不被算核改造波及。

---

## 7. 对 CPU 后端的处理原则

## 7.1 首轮不要拆散现有 CPU 核

`src/backend/qwen_cpu/` 当前已经沉淀了很多对 CPU 有效的优化。  
在没有建立可靠回归体系前，不适合一边抽象一边深拆这些热路径。

建议做法：

1. 先把现有 CPU 后端整体包成 adapter
2. 对外暴露统一 backend 接口
3. 内部实现尽量保持原样

这样做的价值：

- CPU 行为更稳定
- 后端抽象更容易验证“零行为变化”
- 给 CUDA 留出正式入口，而不要求 CPU 先重写一遍

## 7.2 `qwen_ctx_t` 不宜直接承载 GPU 资源

当前 `qwen_ctx_t` 已经过重。  
如果继续向其中追加：

- device buffer
- CUDA stream
- cuBLAS handle
- host-pinned staging buffer

会导致上下文语义更混乱。

建议：

- CPU 保持 `qwen_ctx_t`
- CUDA 另立自己的 model/session/stream session 结构
- 共享的仅是统一接口，而不是共享内部结构

---

## 8. CUDA 后端的首轮落点

## 8.1 不建议首轮直接做“全链路 GPU”

原因：

1. 当前实时链路高度依赖 token 级增量提交
2. decoder 单 token 路径与 KV 管理耦合很深
3. 一次性把 decode、KV、streaming reset 全搬到 GPU，风险过高

## 8.2 建议首轮只攻“大算子、长序列、收益确定”的部分

优先顺序建议：

1. encoder 主体计算
2. decoder prefill 大 GEMM
3. 保持 decode 单 token 路径先在 CPU

这样做的原因：

- encoder / prefill 更接近 GPU 擅长的批量矩阵计算
- decode 单 token 在小 batch、小步进场景下，GPU 首轮未必最优
- 可以在保持现有流式控制逻辑的前提下先获得一部分收益

## 8.3 实时首轮目标应是“保语义加速”，不是“另写一套流式协议”

CUDA 首轮实时目标建议定义为：

- 保留现有 `partial / stable / final` 语义
- 保留 token callback
- 保留 prefix rollback / reset / overlap / cache 逻辑
- 仅把部分算力热点迁到 GPU

不建议首轮目标定义成：

- 新写一套实时输出语义
- 为了适配 GPU 修改前端展示逻辑
- 为了吞吐牺牲稳定文本体验

---

## 9. 推荐分期

## P0：文档与事实校准

目标：

- 校正文档和源码漂移
- 明确哪些路径是真主路径
- 明确实时语义不可动边界

产物：

- 本规划文档
- 必要的审计补记

## P1：构建层接缝

目标：

- 增加 `QASR_ENABLE_CUDA_BACKEND`
- 增加 CUDA preset
- 增加编译时 backend 枚举与查询接口

验收：

- CPU-only 构建零回归
- 打开 CUDA 选项后能完成后端编译探测
- 无 CUDA 环境时错误信息清晰

## P2：后端抽象落地

目标：

- `server` 不再直持 `qwen_ctx_t*`
- `model_bridge` 不再只暴露 CPU 语义
- CPU 后端完成 adapter 化

验收：

- CPU 离线结果与当前一致
- CPU realtime 行为与当前一致
- server 测试不因抽象层重写而改变语义

## P3：CUDA 离线首版

目标：

- 支持加载 CUDA 后端
- 先打通离线路径
- 先实现 encoder / prefill 加速

验收：

- CPU / CUDA 结果在可接受误差内一致
- 无 GPU 时运行时能回退或报明确信息
- 不影响 CPU-only 包产物

## P4：CUDA 实时首版

目标：

- 接通 realtime stream session
- 保留 token callback 与稳定提交语义
- 验证 encoder cache / overlap / reset 在 CUDA 路径下仍成立

验收：

- 实时 UI 不退化
- `partial / stable / final` golden 不变
- 无 pause 连续说话场景仍可稳定推进

## P5：CUDA 深化

可选项：

- decode on GPU
- KV cache 常驻 GPU
- host/device pipeline 更深度重叠
- 多 session batching

该阶段不应在首轮立项中强行捆绑。

---

## 10. 最小侵入的代码切口

建议优先改造以下位置：

### 10.1 构建层

- `CMakeLists.txt`
- `cmake/QasrBlas.cmake`
- `CMakePresets.json`

### 10.2 运行时桥接层

- `src/runtime/model_bridge.h`
- `src/runtime/model_bridge.cc`

### 10.3 服务层与模型持有方式

- `src/service/server.cc`

重点改动点：

- `SharedAsrModel`
- realtime clone 创建逻辑
- backend 选择逻辑

### 10.4 新增统一 backend 目录

建议新增类似目录：

```text
src/backend/common/
src/backend/qwen_cuda/
```

其中：

- `common/` 放接口、公共枚举、能力描述
- `qwen_cpu/` 保持现状，逐步适配
- `qwen_cuda/` 独立承载 CUDA model/session 实现

## 10.5 首轮明确不建议优先动的部分

- `src/service/realtime.cc`
- 前端 UI
- 现有 HTTP / OpenAI 协议面
- 现有 placeholder 风格的 `src/inference/*`

---

## 11. 主要风险

## 11.1 结构风险

### 风险 1：后端抽象只做表面包装

如果只是把 `qwen_ctx_t*` 换个 typedef 或套一层薄 wrapper，实际上还是 CPU 内核直通，那么：

- CUDA 逻辑仍会继续侵入 service
- clone / session / device 生命周期无法理顺
- 后续维护成本更高

### 风险 2：把 CPU / GPU 资源强行塞进同一上下文

这样会让 `qwen_ctx_t` 继续膨胀，最终演变成无法测试、无法隔离、无法回归的超级结构体。

### 风险 3：抽象过度，首轮就试图重做整个推理架构

如果在没有 CPU 适配回归的前提下同时做：

- 后端抽象
- CUDA 接入
- 实时链重写

那么失败概率很高。

## 11.2 实时风险

### 风险 4：实时输出从“稳定增量”退化成“整段重吐”

这会直接影响：

- 用户体感
- display snapshot 语义
- `stable_text` / `partial_text` 的一致性

### 风险 5：CUDA 路径不复现 reset / reanchor / overlap 策略

这会让长时间实时场景出现：

- 文本老化
- 尾巴重复
- 漂移累积
- 稳定前缀推进异常

## 11.3 构建与部署风险

### 风险 6：Windows CUDA 构建污染现有 CPU-only 产线

必须确保：

- CUDA 是显式选择
- CPU-only 构建脚本不被破坏
- 无 NVCC 时错误在 configure 阶段即清楚暴露

### 风险 7：CPU 线程策略与 GPU 并行资源争抢

当前项目已有：

- OpenBLAS / Accelerate
- 自建线程池
- 背景 encoder 线程

引入 CUDA 后，需要重新评估：

- CPU 线程数是否降档
- host 预处理是否与 GPU 计算重叠
- 是否需要 pinned memory / staging buffer

---

## 12. 测试与验收矩阵建议

## 12.1 构建测试

至少覆盖：

1. Windows CPU-only
2. Windows CPU+CUDA
3. Linux CPU-only
4. Linux CPU+CUDA（如后续支持）
5. macOS CPU-only

## 12.2 功能一致性测试

至少覆盖：

1. 离线短音频 CPU vs CUDA 文本一致性
2. 离线长音频 CPU vs CUDA 文本一致性
3. streaming CPU vs CUDA `partial / stable / final` 语义一致性
4. token callback 单调追加一致性
5. timestamps 一致性（如支持）

## 12.3 回归测试

必须补齐：

1. backend 选择测试
2. 无 CUDA 环境下的回退或报错测试
3. realtime clone 生命周期测试
4. prefix rollback / reset / overlap 行为测试
5. 长时无停顿流式测试

## 12.4 性能测试

至少记录：

1. 首 token 延迟
2. 实时轮次耗时
3. encoder 耗时
4. prefill 耗时
5. decode 耗时
6. GPU 显存占用
7. CPU 占用变化

原则：

- 不只看吞吐
- 实时场景优先看稳定延迟与显示体验

---

## 13. 建议的首轮技术决策

为了降低首轮复杂度，建议默认采用：

1. **CUDA + cuBLAS** 作为首个 GPU 基线
2. 不在首轮引入 TensorRT / ONNX Runtime / 第三方大框架
3. 不在首轮追求跨设备多后端统一算子层
4. 先做 encoder / prefill 加速，再评估 decode 是否值得 GPU 化

理由：

- 当前项目已有大量自定义流式控制逻辑
- 先保现有主链最重要
- 首轮目标应是“建立可靠后端接缝”，不是“引入另一整套执行框架”

---

## 14. 建议的后续执行顺序

建议下一步按以下顺序推进：

1. 先补一版“后端抽象设计草案”
2. 再列“最小文件改动清单”
3. 再做 CPU adapter 化
4. 最后再开 CUDA 首版开发

如果跳过步骤 1 和 2，直接开始改码，极可能在中途演变成大面积返工。

---

## 15. 本规划的结论

当前项目接入 CUDA 是可行的，但不适合以“给现有 CPU 核临时贴一层 GPU 调用”方式推进。

更稳妥的路线是：

1. **先承认当前真实主路径仍在 `qwen_cpu`**
2. **先建立正式 backend 抽象**
3. **先把 CPU 包装成稳定适配器**
4. **再以可选编译的方式接入 CUDA**
5. **先做 encoder / prefill，后做 decode 深化**
6. **始终把实时稳定语义视为硬约束**

只有这样，才能做到：

- 对现有 CPU 主路径影响最小
- 对实时体验风险可控
- 对后续 CUDA 深化留有空间

