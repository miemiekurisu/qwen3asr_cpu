我重新整理成一份**可开源发布、跨平台、且只依赖 macOS 系统库 Accelerate 与 Intel/Linux 上 OpenBLAS** 的 CPU 优化方案。

先给结论：

**主线方案应该是：**

* **Apple M 系列**：`Accelerate(BLAS/vDSP)` 作为 prefill 主后端，保留你自己的 `NEON/BF16 matvec` 作为 decode 主后端。
* **Intel CPU**：`OpenBLAS` 作为 prefill 主后端，保留你自己的 `AVX2/AVX-512 matvec` 作为 decode 主后端。
* **统一设计原则**：不要追求“一套 kernel 同时吃掉 prefill 和 decode”，而是明确分成两类执行计划；同时把优化重点放在 **QKV 融合、权重预打包、shape-aware dispatch**，而不是继续在“BF16→F32 临时展开”这一条线上做小修小补。近年的 CPU LLM 研究也明确指出，prefill 和 decode 的工作负载不同，分开优化通常更合理；而顺序 GEMM 的 layout/pacing 冗余也确实能成为显著瓶颈。([arXiv][1])

你这份代码本身已经给了我们一个很清楚的出发点：
`seq_len == 1` 时，`qwen_linear_nobias_bf16` / `qwen_linear_bf16` 走自写 BF16 matvec；`seq_len > 1` 时，则通过 `bf16_get_f32_view()` 把权重展开成 F32，再进入 `qwen_linear()` 的 `cblas_sgemm`。另外，decoder prefill 里 `wq/wk/wv` 仍然是三次独立投影，而不是一次 fused QKV；CMake 也只给 `qasr_cpu_c` 加了 `-O3 -ffast-math`，没有系统性地做 `-march=native` 或多 ISA 版本分发。这几件事决定了：**你现在最值得做的不是换库，而是先把后端结构理顺。**

---

## 一、设计目标

这份方案只追求三件事：

1. **源码与发布许可干净**
2. **同一份包能在 macOS M 系列和 Intel/Linux 上工作**
3. **把收益最大的 CPU 优化先做掉**

从依赖角度看，OpenBLAS 是 BSD-3-Clause 许可；它支持多目标运行时检测 `DYNAMIC_ARCH=1`，并且当前版本已经把 x86_64 的运行时目标覆盖到 Haswell、Zen、SkylakeX、Cooper Lake、Sapphire Rapids 等。Apple 的 Accelerate 则是系统框架，官方说明其 BLAS/LAPACK 会在运行时根据 CPU 能力选择合适实现。工程上，这正好适合做“同一套抽象接口，不同平台接不同后端”。([GitHub][2])

---

## 二、我建议的总体架构

### 1. 统一抽象层

把后端接口拆成四类：

* `gemm_fp32()`：大中型 prefill GEMM
* `gemv_bf16()`：decode / `seq_len==1` 路径
* `vec_ops()`：RMSNorm、RoPE 前后的逐元素运算
* `pack_weight()`：权重预打包与缓存

这样做的原因很直接：
在 Apple 上，Accelerate 很适合承担 `gemm_fp32` 和部分向量计算；在 Intel 上，OpenBLAS 适合承担 `gemm_fp32`；而 `gemv_bf16` 和一些轻量向量算子，仍然由你自己的 SIMD 核心实现更可控。Accelerate 官方也把 BLAS、vDSP、BNNS 分成不同子库，分别面向线性代数、DSP/向量运算和机器学习推理。([Apple Developer][3])

### 2. 两套执行计划，而不是一套

* **Prefill plan**：面向矩阵乘，追求吞吐
* **Decode plan**：面向 matvec / 小矩阵，追求单 token 延迟

这不是“理论上更优”，而是已经有近期公开研究明确支持：prefill 与 decode 在 CPU 上的资源特征不同，统一执行计划通常是次优。([arXiv][1])

---

## 三、主线推荐：真正值得做的优化

### A. 两个平台都该做的公共优化

#### A1. 先做 prefill 的 QKV 融合

这是我认为收益最大的第一项。

你当前 decoder prefill 里：

* `wq` 一次
* `wk` 一次
* `wv` 一次

这是三次独立 GEMM。
建议在加载模型时直接构造 `Wqkv_fused = [Wq; Wk; Wv]`，prefill 只打一发 GEMM，输出后切分成 Q/K/V。

这样做的收益来自三方面：

* 少一次到两次 BLAS 调用开销
* 少读两遍权重
* 为后续的权重预打包创造条件

这条思路和最新的 LP-GEMM 研究高度一致：在顺序 GEMM 中，重复 pack / unpack 会浪费大量资源，允许布局在连续 GEMM 之间传播，能显著超过 OpenBLAS 的“单次 GEMM 最优、整体链路未必最优”模式。论文在 x86 上报告了顺序 GEMM 相对 OpenBLAS 的平均 2.25× 提升。([arXiv][4])

#### A2. 把“F32 视图缓存”升级成“持久化 packed weight”

你现在的 `QWEN_BF16_CACHE_MB` 本质上更像“展开后缓存”，不是“后端友好的 packed 布局缓存”。

更值得做的是：

* 模型加载时或首次命中时预打包
* 按层缓存 `packed_wqkv`, `packed_wo`, `packed_gate_up`, `packed_down`
* 不在热路径里做裸展开
* prefill 直接消费 packed 权重

因为对于连续出现的固定形状 GEMM，真正值钱的不是“是否缓存 F32”，而是“是否缓存成后端最容易吃的形状”。LP-GEMM 的结果已经说明，顺序 GEMM 的布局管理本身就是一类性能瓶颈。([arXiv][4])

#### A3. 明确做 shape-aware dispatch

不要再让所有 `seq_len > 1` 的情况都统一落到同一条 `SGEMM` 路上。

至少分成：

* `seq_len == 1`：专用 matvec
* `2 <= seq_len <= small_threshold`：small/skinny GEMM 路径
* `seq_len > small_threshold`：BLAS GEMM 路径

原因是，传统 BLAS 的优势通常出现在“大而规整”的 GEMM；而 decode、小批量 prefill、固定形状投影，往往更接近 small/skinny GEMM 的世界。你当前 workload 明显不是“全程一个大 GEMM 模式”。

---

### B. Apple M 系列 + Accelerate：值得尝试的方案

#### B1. 把 Accelerate 固定为 **prefill 主后端**

这是 Apple 路线的主线，不建议动摇。

理由有两个：

第一，Apple 官方明确说明 Accelerate 的 BLAS/LAPACK 会抽象底层 CPU 处理能力，代码在运行时会自动执行适合当前处理器的实现。第二，Accelerate 还同时提供 vDSP 和其他高性能向量库，这很适合把 prefill 的大矩阵乘和前后向量处理放在同一系统框架内。([Apple Developer][3])

对你当前程序而言，这意味着：

* 保留 `cblas_sgemm` 作为 Apple prefill 的第一基线
* 但不要继续围绕“是否要临时 BF16→F32 scratch”做过多花样
* 先把 QKV 融合、packed weights、small-shape dispatch 做完，再评估是否还要改算子级细节

#### B2. decode 继续保留你自己的 NEON/BF16 matvec

Apple 路线上，我不建议把 decode 也硬塞给 BLAS。

原因很简单：

* decode 是 `m=1` 或极小 `m`
* 你已经有专用 BF16 matvec 路径
* 在这类 shape 下，通用 BLAS 未必是最佳解

所以 Apple 路线最合理的形态是：

* **Prefill：Accelerate**
* **Decode：自写 NEON/BF16**

这比“全部换成统一库调用”更贴合你当前程序结构。

#### B3. 向量类算子尽量切到 vDSP / simd

Apple 官方把 vDSP 定位为高优化的 DSP/大数组算子集合，把 `simd` 定位为小向量/小矩阵运算工具。对你的代码来说，以下几类值得评估是否交给系统向量库：

* `add_inplace`
* `scale`
* `mul`
* reduction / max
* 某些小规模 RoPE / RMSNorm 辅助步骤

但我会强调一句：
**只迁那些“明显是纯向量、内存访问规整、没有复杂控制流”的部分。**
attention 主体、KV 写回、QKV 分裂不要为此大改。Apple 官方对 vDSP / vForce / simd 的定位也是面向向量与小矩阵、通用数值运算，而不是完整 Transformer 图执行。([Apple Developer][3])

#### B4. BNNS / BNNSGraph：只列为 Apple-only 实验，不列为主线

Apple 2024 明确推出了 BNNS Graph，强调：

* 面向 CPU 推理
* 图级优化
* 权重 repacking
* 可在实时场景中做到无运行时内存分配、支持单线程执行。([Apple Developer][5])

但我不建议把它作为你的主线方案，原因不是它不好，而是它和你的当前代码形态不匹配：

* 它更偏“整图编译/执行”
* 通常要走 Core ML / graph compile 这套流程
* 会让 Apple 后端和 Intel 后端分叉得过早

所以我的建议是：

* **主线**：Accelerate BLAS/vDSP + 自写 kernel
* **实验支线**：如果以后单独做 Apple-only 版本，再研究 BNNSGraph 是否能把 encoder 或某些固定子图整段吞掉

---

### C. Intel CPU + OpenBLAS：值得尝试的方案

#### C1. OpenBLAS 作为 prefill 主后端，且必须启用 `DYNAMIC_ARCH`

这是 Intel 侧的主线。

OpenBLAS 官方文档明确建议，在面向多种 CPU 分发时启用 `DYNAMIC_ARCH=1`，这样库会在运行时自动选择合适的目标内核；其 x86_64 动态目标列表目前覆盖从 Haswell 到 Sapphire Rapids 等多代处理器。OpenBLAS 本身也是 BSD-3-Clause。([openmathlib.github.io][6])

所以 Intel 侧我建议：

* 发布包时使用动态 OpenBLAS 或 vendored OpenBLAS
* 构建时启用 `DYNAMIC_ARCH=1`
* 在基准时打开 `OPENBLAS_VERBOSE=2`，确认实际选中的 core type
* 用 `OPENBLAS_NUM_THREADS` 控制线程数，而不是假设默认值就是最佳值。OpenBLAS 也公开了这些运行时变量。([openmathlib.github.io][7])

#### C2. decode 继续走自写 AVX2 / AVX-512 路，而不是 OpenBLAS

和 Apple 一样，我不建议把 decode 主路径交给 OpenBLAS。

原因：

* 你的 decode 天然是 matvec / 极小 GEMM
* 这类形状常常不是通用 BLAS 的甜点区
* 你已经有 `qwen_asr_kernels_avx.c`

但这里有一个关键前提：
**你得先把 x86 ISA 构建修正。**

你当前 CMake 没有系统性加 `-mavx2/-mfma/-mavx512*`，也没有做 per-ISA 多版本 dispatch。
所以在 Intel 侧真正值得做的是：

* 产出 `generic / avx2 / avx512` 多版本对象
* 运行时 CPUID 选择
* 而不是让整个库依赖一次性全局编译宏

OpenBLAS 自己就是这么做动态目标分发的；你自己的 decode kernel 也应该采用类似思想。([GitHub][2])

#### C3. OpenBLAS 的 BF16 新能力可以跟踪，但不作为当前主线依赖

这一点需要说清楚。

OpenBLAS 近两个版本在低精度这条线上是有进展的。0.3.31 已加入 `BGEMM` / `BGEMV` 等 bfloat16 扩展和批量接口；当前文档也已经把 bfloat16 相关功能列为扩展的一部分。([GitHub][8])

但对你当前项目，我的建议是：

* **关注**
* **可以做实验分支**
* **但先不要把主线押在 OpenBLAS BF16 API 上**

原因不是 OpenBLAS 不行，而是你的当前项目还没把“QKV 融合、packed weights、shape-aware dispatch”这些大块做好。
在这些结构性问题没解决之前，过早追 OpenBLAS BF16 API，容易把时间花在边角收益上。

---

## 四、我建议你明确放弃或延后的东西

### 1. 不把 BNNSGraph 作为跨平台主线

理由前面说过：Apple-only，且会强行引入图编译流。

### 2. 不把 OpenBLAS 当 decode 主后端

OpenBLAS 主战场仍然更适合 prefill / medium-large GEMM。

### 3. 不继续把主要精力放在“BF16→F32 scratch 再细调”

这条路不是不能做，而是优先级已经下降。
你现在更大的收益来自：

* fused QKV
* packed weights
* dispatch
* threading

### 4. 不做 Apple 与 Intel 的“完全相同内核实现”

这在工程上不划算。
真正该统一的是**抽象层和调度策略**，不是每个微内核都强行统一。

---

## 五、最终可执行的 CPU 优化方案

下面这份是我给你的正式版本。

### 第一层：公共抽象

保留现有 C/C++ 结构，新增 `cpu_backend.h`：

* `gemm_fp32(...)`
* `gemv_bf16(...)`
* `vec_add/vec_mul/vec_scale/...`
* `pack_linear_weight(...)`
* `free_packed_weight(...)`

### 第二层：平台后端

#### Apple backend

* prefill GEMM：Accelerate BLAS
* 向量算子：vDSP / 必要时 simd
* decode matvec：自写 NEON/BF16
* 可选实验：BNNSGraph，仅 Apple-only feature flag

#### Intel backend

* prefill GEMM：OpenBLAS
* decode matvec：自写 AVX2/AVX-512
* 线程与 CPU 选择：OpenBLAS runtime variables + 自己的 CPUID dispatch
* 构建：OpenBLAS `DYNAMIC_ARCH=1`

### 第三层：模型层优化

按顺序做：

1. **prefill QKV 融合**
2. **gate_up / 其余连续投影继续融合与整理**
3. **packed weight 缓存**
4. **small/medium/large shape dispatch**
5. **prefill / decode 分别调线程数**

### 第四层：基准矩阵

每个平台至少测四项：

* prefill 毫秒
* decode 毫秒/词元
* 总体实时率
* RSS / 峰值内存

Apple 和 Intel 都分别测：

* 旧版路径
* 只做 QKV 融合
* 融合 + packed weights
* 融合 + packed weights + shape dispatch

---

## 六、按投入产出比排序

### 必做

1. 修 x86 ISA dispatch
2. prefill QKV 融合
3. packed weight
4. prefill / decode 分治
5. Apple: Accelerate + 自写 NEON
6. Intel: OpenBLAS + 自写 AVX

### 值得试，但放第二阶段

1. Apple 上把部分纯向量算子切到 vDSP
2. Intel 上评估 OpenBLAS 的 BF16 新接口
3. Apple-only BNNSGraph 小范围试点

### 暂不建议投入主线

1. Apple-only 整图重写
2. 继续深挖 scratch F32 临时展开的小修补
3. 让 OpenBLAS 负责 decode 主路径

---

## 七、最简洁的一句话版本

**跨平台开源 CPU 包的最佳主线，不是追求“统一一个库吃掉全部”，而是：Apple 用 Accelerate 做 prefill、Intel 用 OpenBLAS 做 prefill，两边都保留你自己的 SIMD decode kernel；然后把主要工程资源投入到 QKV 融合、权重预打包、shape-aware dispatch 和 prefill/decode 分治。** 这条路线最符合你当前代码形态，也最符合公开文档和近期 CPU 推理研究给出的方向。([Apple Developer][3])

[1]: https://arxiv.org/abs/2507.18454?utm_source=chatgpt.com "Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving"
[2]: https://github.com/OpenMathLib/OpenBLAS "GitHub - OpenMathLib/OpenBLAS: OpenBLAS is an optimized BLAS library based on GotoBLAS2 1.13 BSD version. · GitHub"
[3]: https://developer.apple.com/accelerate/ "Accelerate Overview - Apple Developer"
[4]: https://arxiv.org/abs/2604.04599?utm_source=chatgpt.com "LP-GEMM: Integrating Layout Propagation into GEMM Operations"
[5]: https://developer.apple.com/wwdc24/10211 "Support real-time ML inference on the CPU - WWDC24 - Videos - Apple Developer"
[6]: https://www.openmathlib.org/OpenBLAS/docs/distributing/?utm_source=chatgpt.com "Redistributing OpenBLAS - openmathlib.github.io"
[7]: https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/?utm_source=chatgpt.com "Runtime variables - OpenBLAS - openmathlib.github.io"
[8]: https://github.com/xianyi/OpenBLAS/blob/develop/Changelog.txt?utm_source=chatgpt.com "OpenBLAS/Changelog.txt at develop"
