# 经验录

## 2026-04-10

### 1. 二参考之分工极明

- `qwen-asr-learn` 宜学“算法全链”
- `whisper.cpp` 宜学“工程外壳”

若二者混读不分层，易误入两端皆半。

### 2. `qwen-asr-learn` 最值钱者，不仅 encoder/decoder

最值钱处实为 `qwen_asr.c`：

- prompt 装配
- segment 切分
- silence compaction
- streaming rollback
- stable frontier commit
- 恢复性 re-anchor

此文件近乎“官方推理策略简化版”。

### 3. `whisper.cpp` 之价值，在 ABI 与 examples

最该学：

- context/state 分离
- `include/whisper.h` 风格
- `examples/server`
- `examples/stream`
- `tests/` + CTest

不该误学：

- 直接搬 Whisper 模型路径

### 4. 官方文档有漂移迹象

截至 2026-04-10：

- 本地 `MODEL_CARD_OFFICIAL.md` 与 Hugging Face 官方模型卡均述 Qwen3-ASR 支持 30 语 + 22 方言，且官方工具包支持 vLLM、async serving、streaming、timestamp。
- vLLM 的 Qwen3-ASR recipe 页面却写 11 语。

此类漂移，后续须以官方模型卡与官方仓库为主，recipe 为辅。

### 5. 当前环境事实

- 宿主：macOS / arm64
- 编译器：Apple clang 17.0.0
- CMake：4.2.3
- 本机未装 OpenBLAS
- Docker 可用
- Docker daemon 架构：`linux aarch64`

故：

- 本机可先验 macOS/Accelerate
- Linux/OpenBLAS 可先验 `linux/arm64`
- `linux/amd64` 须用 `docker run --platform linux/amd64`

### 6. 先立规约，再迁模型

若先搬模型代码，再补 runtime/protocol/tests，后患大。
正序应为：

1. 文档
2. 骨架
3. 测试基座
4. 模型迁移
5. 服务化

### 7. Docker 双架构 OpenBLAS 已验

于 2026-04-10 已实跑：

- `linux/arm64 + OpenBLAS`
- `linux/amd64 + OpenBLAS`

结果：

- CMake 配置通过
- 编译通过
- 单测通过

故当前“平台/BLAS 约束 + 基础骨架”已具三路实证：

- macOS/Accelerate
- Linux arm64/OpenBLAS
- Linux amd64/OpenBLAS

### 8. ModelScope 本地模型目录名会转义

用户口述路径为：

- `Qwen3-ASR-1.7B`

然本机实际目录为：

- `Qwen3-ASR-1___7B`

故模型发现逻辑不可只信“显示名”，须校：

- 目录存在
- `config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors.index.json`
- index 所列分片齐全

### 9. Vendor 后端可作首个可运行基线

为先达“主程序可启且可跑 ASR”，今采本仓 vendor CPU 后端：

- 不改 `qwen-asr-learn`
- 不从参考目录 include / link
- 以 CMake 只编本仓 `src/backend/qwen_cpu/`
- 由 `qasr/runtime/model_bridge.*` 封之
- 由 `qasr_cli` 调之

### 10. 本轮修复留档

2026-04-10，按开发守则补齐了：

- `CMakePresets.json` 增加 ASan/UBSan 预设
- `BuildServerUsage()` 单测补齐
- `RunServer()` 单测补齐
- `RunAsr()` 非正常值单测补齐
- 采用 `cmake` + `ctest` 本机验证通过


此法可先保：

- 本工程可独立构建
- 主程序可独立启动
- 后续可逐模块替换 vendor 后端

### 11. 本机已实跑 `qasr_cli`

于 2026-04-10，macOS / arm64 / Accelerate，模型：

- `/Users/kurisu/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1___7B`

样音：

- `qwen-asr-learn/samples/test_speech.wav`
- `qwen-asr-learn/samples/jfk.wav`

结果：

- 载模成功
- 转写成功
- `jfk.wav` 输出为：`And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.`

此时“主程序可启且可跑 ASR”已成。

### 12. 官方实时与时间戳，本就分路

外部事实：

- Qwen 官方模型卡称 streaming 仅 vLLM backend 支持
- 且 streaming 不支持 batch，不支持 timestamps
- 精细时间戳交 `Qwen3-ForcedAligner-0.6B`

故：

- 主实时路先求稳定字幕
- 词级时间戳宜异步后补

### 13. “无停顿”之核心，不是 VAD，而是稳定前缀

AWS 文档与 Whisper-Streaming 皆指向同一结论：

- partial 可先出
- stable prefix 后提交
- 尾巴允许少量改写

此比“等停顿再整句冒出”更合字幕 UX。

### 14. `qwen-asr-learn` 已隐含正确方向

其 `--stream` 默认：

- `2s` chunk
- `8s` encoder window
- `rollback 5`
- `unfixed_chunks 2`
- `max_new_tokens 32`
- 长流自动裁历史

这说明参考实现并未依赖停顿；其主法本就是“滑窗 + rollback + frontier commit”。

### 15. vLLM 官方 realtime 默认分段偏保守

vLLM `qwen3_asr_realtime` buffer 默认 `5s` 才吐一段。

此足以保正确，不足以保字幕观感。
若求“看似准实时”，须在我们自研 runtime 内把 cadence 压到亚秒级 tick，而非直接照搬 `5s`。

### 16. 双路法值记，但不宜阻首版

2025 年论文已示：

- 快路 partial
- 慢路 rerank / refine

然其多赖额外训练或额外头。
本项目首版当先把“无停顿滑窗 + 稳定前缀”做稳，再议双路增强。

### 17. 实时接口第一版已改成三态文本

现 `/api/realtime/*` 与 `/api/capture/*` 已不再只回一段 `partial_text`，而回：

- `stable_text`
- `partial_text`
- `text`
- `finalized`

此使前端后续可做“稳定前缀不重绘”。

## 2026-04-11

本日经验改按主题归并，不再续接全局编号；旧的重复编号、重复日期段与重复结论已合并如下。

### Windows / OpenBLAS 优化

- Windows/Intel 当前正确主线已明确：encoder QKV 走“加载期预打包 + 运行期单 GEMM + 拆分 q/k/v”。
- 先前的低风险过渡方案是“运行期临时把 Q/K/V 拷到连续缓冲区再打一发 GEMM”；后续实测已证伪，此法不应再回到热路径。
- `tools/build_windows_openblas.cmd` / `.ps1` 已作为 Windows 一键构建入口：
   - 不写死本机路径
   - 先检 `-OpenBlasDir`
   - 再检 `OPENBLAS_DIR`
   - 再检现有 `build/<preset>/CMakeCache.txt`
   - 最后按仓内相对目录约定检索 OpenBLAS
   - 若当前 shell 未带 MSVC 环境，则用 `vswhere` + `VsDevCmd` 自动补齐

本轮 Intel benchmark 环境：

- OpenBLAS `OPENBLAS_VERBOSE=2` 报告核心类型：`SkylakeX`
- Windows 主机：`16 logical CPUs`
- 基准程序：`qasr_cpu_bench`
- 对比对象：`separate` / `fused_copy` / `fused_packed`

Windows + OpenBLAS + SkylakeX + 16 logical CPUs，`qasr_cpu_bench` 实测：

- 8 线程：
   - `seq4 d1024`：`fused_packed = 1.43x separate`
   - `seq13 d1024`：`1.07x`
   - `seq52 d1024`：`1.08x`
   - `seq104 d1024`：`0.93x`
   - `seq13 d896`：`1.47x`
- 12 线程：
   - `seq4 d1024`：`1.67x`
   - `seq13 d1024`：`1.15x`
   - `seq52 d1024`：`1.14x`
   - `seq104 d1024`：`1.20x`
   - `seq13 d896`：`1.34x`
- `fused_copy` 即“每次调用临时拼接后再单 GEMM”，在全部测点都慢于 `separate`。

当前结论：

- Intel/Windows 保留 `fused_packed`
- 禁止把 `fused_copy` 回灌到热路径
- 线程数 8 与 12 皆可用，但最优值随 shape 变，不可假设 12 恒优

新增一轮“重方案”落地后，当前状态进一步更新：

- 已新增独立策略模块 `qwen_asr_perf.{h,c}`，把实验策略与 runtime CPU 特征检测从 encoder / kernels 主体里拆开，便于整体回退。
- encoder QKV 现支持四种策略名：`best` / `force_separate` / `force_packed` / `shape_auto`。
- 当前**生产默认**仍是 `best`：
   - `seq < 8` 走 `separate`
   - 其余已验证 shape 继续走 `packed`
- `shape_auto` 目前仅保留为**显式实验策略**，通过 `QWEN_ENC_QKV_POLICY=shape_auto` 启用；未改成默认，原因是它在大 shape 上有收益，但在部分中小 shape 上还不够稳定。
- x86 decode 热核已从“编译期宏碰运气”改成“独立 AVX2 源文件 + runtime CPUID 选择”：
   - CMake 在 x86 下仅给 `qwen_asr_kernels_avx.c` 加 `/arch:AVX2` 或 `-mavx2 -mfma`
   - 运行时若 CPU 支持 `AVX2 + FMA`，则自动启用 AVX2 热核
   - 不支持时自动回退 generic
   - 本轮 Windows/OpenBLAS 实机 bench 已确认 `runtime_kernel_backend=avx2`

进一步的 benchmark 结论必须明确写清：

- 若看**相对 speedup**，12 线程常优于 8 线程，因为 `separate` 在 12 线程下退化更明显。
- 若看**`fused_packed` 的绝对延迟**，8 线程在多数测点反而更低：
   - `seq4 d1024`：`0.444ms @8` vs `0.447ms @12`
   - `seq52 d1024`：`1.080ms @8` vs `1.504ms @12`
   - `seq104 d1024`：`1.897ms @8` vs `1.927ms @12`
   - `seq13 d896`：`0.415ms @8` vs `0.484ms @12`
- 故线程评估不可只看“加速比”，必须同时看**最终绝对时延**。
- 当前样本上，8 线程更像稳态低延迟候选；12 线程更像特定 shape 的吞吐候选。
- `seq104 d1024 @8` 上 `fused_packed` 仍略慢于 `separate`，说明 fused QKV 不是对所有大 shape 都绝对占优，后续仍需 shape-aware dispatch 来做最后分流。

本轮新增 `shape_auto` benchmark（仍基于 Windows + OpenBLAS + SkylakeX）后，结论更具体：

- `shape_auto` 当前逻辑是在 `seq>=96 && d_model>=1024` 时回退 `separate`，其它已打包 shape 继续走 `packed`。
- 8 线程下：
   - `seq104 d1024`：`shape_auto = 1.822ms`，优于 `fused_packed = 2.007ms`
   - `seq13 d896`：`shape_auto = 0.430ms`，优于 `separate = 0.710ms`
   - 但 `seq13 d1024` 这一类中小 shape 结果波动仍偏大，尚不足以证明 `shape_auto` 全面优于当前默认
- 12 线程下：
   - `seq104 d1024`：`shape_auto = 1.768ms`，仍优于 `separate = 1.911ms`，但低于 `fused_packed = 1.544ms`
- 因此当前判断应保持保守：
   - **decoder runtime AVX2 dispatch：成立，已纳入默认路径**
   - **encoder `shape_auto`：方向成立，但只保留为实验开关，不直接替换生产默认**
- 硬性约束“不可降低准确率”本轮通过新增 BF16 参考测试守住：
   - `qwen_linear_bf16(seq=1)` 与 reference 对齐
   - `qwen_argmax_matvec_bf16` 与 reference 对齐

Apple 旧实验仍保留一条参考：

- 在 M3 Pro + `long_test.wav` + `864 chunks` 上，prefill avg `263.6ms -> 208.6ms`，总推理 `~917ms/chunk -> ~861ms/chunk`，而 decode avg `22.2 -> 22.4 ms/tok` 基本不变。
- 这说明 Apple/prefill 的主要收益来自“多线程 BF16->F32 转换”而不只是“减少 BLAS 次数”；结构优化与执行级优化必须拆开看。
- 此结论只作 Apple/prefill 参考，不可直接套为 Intel/Windows 的主线判断。

实验性探索账本：

- `[已验证]` encoder QKV 加载期预打包：成立，现为 Windows/Intel 保留方案。
- `[已证伪]` encoder QKV 运行期临时拼接：不成立，全部测点慢于 `separate`。
- `[已验证]` 8 / 12 线程对比：线程最优值依赖 shape，且“相对 speedup”与“绝对时延”可能给出相反结论。
- `[已验证]` x86 decode 热核 runtime AVX2 dispatch：成立；Windows/OpenBLAS bench 已实测落到 `runtime_kernel_backend=avx2`。
- `[试验中]` encoder `shape_auto`：已做成显式策略开关，`seq104 d1024` 有收益，但尚未达到可替换默认策略的证据标准。
- `[已确认现状]` decoder prefill QKV 原本就已 fused，不应重复投入同一路优化。
- `[已确认现状]` decoder gate/up 原本就已 fused，不应重复投入同一路优化。
- `[已确认现状]` 当前 BF16 cache 只是 F32 view/cache，不等于真正后端友好的 packed weight cache。
- `[部分补齐]` x86 runtime dispatch：AVX2+FMA 热核已 runtime 化，但 `qwen_asr_kernels.c` 中仍有不少 AVX 分支停留在编译期宏，后续若继续做 full ISA dispatch，需要继续拆源文件。
- `[已确认缺口]` 当前只做 `seq==1` 与 `seq>1` 分流，尚无 `small / medium / large` shape-aware dispatch。
- `[已延后]` decoder 侧持久 packed weight cache：方向成立，但需先权衡内存占用后再做。
- `[已延后]` OpenBLAS BF16 API 实验：可跟踪，但在 packed weight 与 shape dispatch 之前优先级不高。

重复提醒：VS Code 的 CMake Tools 在当前环境下仍可能因未注入完整 MSVC 标准库环境而报 `stddef.h` / `cstdint` / `functional` 缺失；Windows 真正验证请优先走 `tools/build_windows_openblas.cmd` / `.ps1`。

### 流式服务与资源边界

- decode cadence 不宜随包即跑，现已统一到最小 `800ms` 门槛；浏览器亦按同 cadence flush。
- stop 不可只回最后一轮 partial；现已在 stop 时再跑一次非流式 full-audio decode，并强制 flush 尾巴。
- `RunServer` 的 handler 不得复用外层共享 `Status`；并发请求下极易互踩，必须每个 handler 自持局部变量。
- 实时长流必须限内存：
   - `max_decode_window_ms = 32000`
   - session 只保近 `32s` PCM
   - `sample_count` 记累计样本
   - `retained_sample_count` 记当前保留样本
   - stop 时若旧 PCM 已裁，不得用窗口结果覆盖既有全文
- HTTP 服务须先限队列：
   - httplib task queue 上限 `64`
   - read / write timeout `30s`
   - keepalive timeout `5s`
   - realtime 活跃会话上限 `64`

### 上传与前端路径

- 大 multipart 上传不是“带宽问题”而已，而是至少 `2x body` 服务端内存 + `1x tmp` 磁盘成本。
- `208MB WAV -> Failed to fetch` 说明：仅把上传上限从 `64MiB` 提到更大，不是正解。
- 当前离线 WAV 正解是浏览器端先转成 `16k mono PCM16` 再按小块推送到 `/api/realtime/*`：
   - 浏览器解析 WAV 头
   - 按块读帧
   - 先下混 mono
   - 若非 `16k` 则浏览器端重采样
   - 结果按小块推送
- 这样可同时降低网络峰值、服务端内存峰值与 tmp 文件写入。
- 当前边界：浏览器端仅支持 `PCM 16-bit WAV`；其它编码仍需回退整包上传或后续补 upload session。
- `/api/realtime/chunk` 的 PCM chunk 必须显式 `Content-Type: application/octet-stream`，否则可能被按表单体处理并触发 `413`。

### 边界、命名与许可

- `qwen-asr-learn/` 与 `whisper.cpp/` 只能参考，不能形成编译、include 或 link 依赖。
- 当前 CPU 核源码已内化到 `src/backend/qwen_cpu/`，构建路径不再指向参考目录。
- 命名也会形成依赖错觉，边界名须反映本项目所有权，不得反映来源：
   - `legacy_bridge` -> `model_bridge`
   - `LegacyAsr*` -> `AsrRun*`
   - `QASR_LEGACY_BRIDGE_ENABLED` -> `QASR_CPU_BACKEND_ENABLED`
   - `qasr_legacy_c` -> `qasr_cpu_c`
- CPU 后端可内化，但许可须随行：
   - `src/backend/qwen_cpu/LICENSE.upstream`
   - `NOTICE.md`
   - `docs/07-license-compliance.md`

### 长音频与流式稳定性

- 中文长音频 CLI 不能只等终稿；当前先用 token callback 在 C++ 桥接层聚合出段：
   - `--emit-segments`
   - `--segment-max-codepoints`
   - 标点触发 flush
   - 无标点达阈值亦 flush
- 长音频流式不可使用超大 `stream_max_new_tokens`；现已将 CLI / bridge / C 后端统一硬限到 `<= 128`。
- 流式重复并非多线程竞态：同一中文 30 秒样本在非流式 `threads=1/4` 均无重复，而流式 `threads=1/4` 均重复，根因是 append-only 提交策略而非 BLAS 竞态。
- 当前修法：
   - 已提交 frontier 只前进，不回退
   - 已提交前缀内的模型修正不再外发
   - 外发前扫近邻已提交 token，跳过 rollback/reanchor 的旧 span

### Docker 与环境

- Docker 验收依赖 daemon；`docker` CLI 存在不代表 daemon 已起。
- 本轮 `linux/arm64` 容器验收失败时，根因是 daemon socket 缺失，而非代码失败。
- 用户后续已证实：`linux/amd64` 虚拟可行；若 OpenBLAS Docker 构建失败，多半是 apt 源网络，而非架构虚拟失败。
- apt 源策略必须可禁用与限时：
   - `QASR_APT_MIRROR=` 可禁镜像替换
   - `QASR_APT_RETRIES` 默认 `3`
   - `QASR_APT_TIMEOUT` 默认 `20`

### C++ 抽象层补齐

- 一次性补齐了 10 个头文件、10 个实现文件、10 个测试文件，约新增 120 个测试用例。
- 该层当前可作为设计规格的外壳与契约层，但有两点必须明示：
   - encoder/decoder `.cc` 仍是 stub，待继续接入 C 后端真实权重加载与推理
   - tokenizer BPE 仍是简化版，待接入真正的 merge 优先级逻辑

### 受控测试样音

- `testfile/` 中的中文样音可入库作覆盖补充，但须受控：
   - 可提交 `testfile/*.mp3`
   - 不提交临时转码 WAV/PCM
   - 不提交模型文件
   - 新增大体积样音前须先说明用途

### 重复错误（已标记）

- `[重复文档]` 同一日期采用 append-only 记录，曾导致重复编号、重复日期段与重复结论；今后同日经验改按主题归并，不再续接旧编号流。
- `[重复实现]` “每次调用临时拼接 Q/K/V 再单 GEMM”曾被当作可行优化反复尝试；Windows/Intel 基准已证伪，后续视为禁回归路线。
- `[重复环境]` Windows 重建时若 `qasr_server.exe` 仍在运行，会反复触发 `LNK1168`；重编前须先停占用进程。
- `[重复兼容]` Win32 `max` 宏会破坏 `std::max`；Windows 代码路径必须坚持 `NOMINMAX` 或 `(std::max)` 规避。
- `[重复误判]` 大文件上传失败不应再靠“单纯提高 multipart 上限”处理；后续一律优先流式上传或浏览器端转码。
