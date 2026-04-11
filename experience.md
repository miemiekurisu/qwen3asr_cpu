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

### 17. 实时接口已由“三态文本”进到“近段 + 活尾”

现 `/api/realtime/*` 与 `/api/capture/*`：

- 仍兼回：`stable_text`、`partial_text`、`text`、`finalized`
- 新主显示字段：`recent_segments`、`live_stable_text`、`live_partial_text`、`live_text`、`display_text`
- `recent_segments` 只保近段，不再把全文累加态当实时主结构

此使前端可只绘近段与活尾，终态再落全量稿。

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
   - `seq < 4` 走 `separate`
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
- 进一步把本轮新增方向的结论写死如下：
   - **encoder packed QKV 默认门槛下调到 `seq>=4`：成立，已纳入默认路径**
   - 理由：在本轮 `qasr_cpu_bench` 中，`seq4 d1024` 的 `fused_packed` 在 `8` / `12` 线程都稳定快于 `separate`
   - 实测点：`seq4 d1024 @8` 为 `0.447ms vs 0.517ms`，`@12` 为 `0.383ms vs 0.569ms`
   - **prefill/decode split thread policy：不成立为生产默认**
   - 理由：在 `--threads 12` 下，`decode_threads=1` 与 `decode_threads=4` 都明显慢于 `decode_threads=12`；当前机器上 decoder 单 token 仍然受益于较高线程数
   - 实测点：`dec_decode_qkv_h2048` 为 `0.541ms @12`，`0.861ms @4`，`2.744ms @1`；`dec_decode_gate_up_h2048_i6144` 为 `1.943ms @12`，`2.696ms @4`，`8.503ms @1`
   - **decoder persistent packed F32 cache：不成立为生产默认**
   - 理由：decoder prefill QKV / gate-up 的 packed F32 microbench 虽有约 `1.4x` 到 `1.9x` 提升，但单层额外 cache 已达 `16MB` / `32MB` / `24MB` / `96MB` 量级，累计到多层后内存代价不可接受
   - 实测点：`dec_prefill_qkv_seq64_h2048` 为 `6.072ms -> 3.564ms`，但单层 cache `32MB`；`dec_gate_up_seq64_h2048_i6144` 为 `18.530ms -> 10.201ms`，但单层 cache `96MB`
- 硬性约束“不可降低准确率”本轮通过新增 BF16 参考测试守住：
   - `qwen_linear_bf16(seq=1)` 与 reference 对齐
   - `qwen_argmax_matvec_bf16` 与 reference 对齐

Apple 旧实验仍保留一条参考：

- 在 M3 Pro + `long_test.wav` + `864 chunks` 上，prefill avg `263.6ms -> 208.6ms`，总推理 `~917ms/chunk -> ~861ms/chunk`，而 decode avg `22.2 -> 22.4 ms/tok` 基本不变。
- 这说明 Apple/prefill 的主要收益来自“多线程 BF16->F32 转换”而不只是“减少 BLAS 次数”；结构优化与执行级优化必须拆开看。
- 此结论只作 Apple/prefill 参考，不可直接套为 Intel/Windows 的主线判断。

### 服务并发与清理

- 先前 `server.cc` 以单一 `active_workload` 把 offline / realtime / host capture 串成一把总锁；此与设计中“realtime 会话上限 64”相悖。
- 现已去除此总锁：
   - realtime 仍由会话表限流
   - host capture 仍只允 1 路
   - offline / chat / realtime 不再互相以 409 硬阻
- async job 先前只增不删；现已补后台维护线程，按固定周期清理终态 job，免长跑服务内存线性涨。

实验性探索账本：

- `[已验证]` encoder QKV 加载期预打包：成立，现为 Windows/Intel 保留方案。
- `[已证伪]` encoder QKV 运行期临时拼接：不成立，全部测点慢于 `separate`。
- `[已验证]` 8 / 12 线程对比：线程最优值依赖 shape，且“相对 speedup”与“绝对时延”可能给出相反结论。
- `[已验证]` x86 decode 热核 runtime AVX2 dispatch：成立；Windows/OpenBLAS bench 已实测落到 `runtime_kernel_backend=avx2`。
- `[试验中]` encoder `shape_auto`：已做成显式策略开关，`seq104 d1024` 有收益，但尚未达到可替换默认策略的证据标准。
- `[已验证]` encoder packed QKV 默认门槛下调到 `seq>=4`：成立；`seq4 d1024` 在 8 / 12 线程都稳定优于 `separate`，已并入默认路径。
- `[已确认现状]` decoder prefill QKV 原本就已 fused，不应重复投入同一路优化。
- `[已确认现状]` decoder gate/up 原本就已 fused，不应重复投入同一路优化。
- `[已确认现状]` 当前 BF16 cache 只是 F32 view/cache，不等于真正后端友好的 packed weight cache。
- `[部分补齐]` x86 runtime dispatch：AVX2+FMA 热核已 runtime 化，但 `qwen_asr_kernels.c` 中仍有不少 AVX 分支停留在编译期宏，后续若继续做 full ISA dispatch，需要继续拆源文件。
- `[已确认缺口]` 当前只做 `seq==1` 与 `seq>1` 分流，尚无 `small / medium / large` shape-aware dispatch。
- `[已证伪为默认方案]` prefill/decode split thread policy：本机 12 线程下 decoder 降线程明显变慢，不作为生产默认。
- `[已证伪为默认方案]` decoder 侧持久 packed weight cache：microbench 有收益，但额外 cache 内存不可接受，不进入默认路径。
- `[已延后]` OpenBLAS BF16 API 实验：可跟踪，但在 packed weight 与 shape dispatch 之前优先级不高。

围绕 `docs/enhancement2.md` 再做一轮后，现结论再细化如下：

- 本轮已落三件基础设施：
   - `RuntimeProfile`：统一控制 realtime / balanced / offline / lowmem 档位
   - `PreparedWeight`：decoder layer 增 `prefill_qkv_prepared`
   - `ScratchArena`：decoder prefill QKV 改走 `ctx->prefill_scratch`
- 本轮又续补两层：
   - decoder layer 再增 `prefill_gate_up_prepared`
   - decoder prefill 的 `WO / GateUp / Down` 在 `seq_len>1` 时改走显式 `BF16->F32` scratch，不再借全局静态 scratch
- 当前生产默认仍只固化 **decoder prefill QKV**；GateUp 虽已具同型 prepared 路，但默认预算仍关。
- 现默认预算：`QWEN_DEC_PREFILL_QKV_BUDGET_MB=512`。
   - 0.6B 形状：`(2048 + 2*1024) * 1024 * 4 * 28 = 448MB`，可落预算，故默认启 prepared QKV。
   - 1.7B 形状：同式为 `896MB`，超预算，故默认仍走 BF16 热路。
- 这样做之意不在“所有模型都强上 persistent F32”，而在“只把 0.6B 这类实时档位收益稳、内存尚可受者固化”。

1.7B 真模启动实测亦已补：

- 默认：`QWEN_RUNTIME_PROFILE=balanced`，server 可正常起；无 prepared QKV 日志，符合 `512MB` 预算挡下 1.7B 之预期。
- 强开：`QWEN_RUNTIME_PROFILE=realtime` + `QWEN_DEC_PREFILL_QKV_BUDGET_MB=1024`
   - `qasr_server` 成功起服
   - 日志：`decoder: prepared prefill QKV layers=28 bytes=896.0 MB profile=realtime ms=360.132`
- 结论：
   - 1.7B prepared QKV 在当前机可真实载入，不止停留于 shape bench
   - 其代价约额外 `896MB` 常驻与约 `360ms` 启动准备时间
   - 故仍不宜作默认；但对单实例低延时档，可作为显式 runtime profile 选项

Windows + OpenBLAS + SkylakeX，`qasr_cpu_bench` 新一轮数据：

- 12 线程：
   - `dec_prefill_qkv_seq32_h1024`：`2.108ms -> 1.243ms`，约 `1.696x`
   - `dec_prefill_qkv_seq64_h2048`：`6.820ms -> 3.793ms`，约 `1.798x`
- 4 线程：
   - `dec_prefill_qkv_seq32_h1024`：`2.846ms -> 2.358ms`，约 `1.207x`
   - `dec_prefill_qkv_seq64_h2048`：`8.162ms -> 6.080ms`，约 `1.342x`
- 结论：
   - 0.6B 形状收益成立，且在 4 线程与 12 线程皆为正收益。
   - 1.7B 形状 microbench 亦有益，但因总 prepared cache 约 `896MB`，默认不固化。
   - 故“QKV prepared + budget gating”成立；“全模型无差别持久化”不成立。

本轮尚未固化者：

- GateUp prepared F32：
   - 12 线程 `dec_gate_up_seq32_h1024_i3072` 为 `3.575ms -> 1.479ms`，约 `2.417x`
   - 但 0.6B 全层 cache 约 `672MB`，与 QKV 叠加后过重，故不进默认
- 1.7B GateUp 若全层 prepared，约 `2.6GB+`，风险过高，本轮未强行真机尝试。
- 通用 BF16 热路仍有旧 `bf16_scratch/cache`；但 decoder prefill 主热路已进一步收口：`QKV + WO + GateUp + Down` 在 `seq_len>1` 时已不再依赖全局静态 scratch。其余 encoder / 通用 fallback 后续再收。

验证与边界：

- 本轮新增单测覆盖：
   - `qwen_linear_nobias_qkv_f32_packed`
   - `qwen_linear_nobias_bf16_qkv_prefill` 新 scratch 语义
   - `qwen_linear_nobias_bf16_scratch`
   - `qwen_should_prepare_decoder_prefill_qkv/gate_up`
   - `qwen_decoder_prepare_runtime`
   - `qwen_float_arena_*`
   - `qwen_perf_now_ms`
- Windows/OpenBLAS 全量单测已过。
- 当前 workspace 仍无 `Qwen3-ASR-0.6B` 真模型目录，故本轮“0.6B realtime 提升”证据仍是 **0.6B shape microbench 代理**，不是端到端 live run；此限制须留档，不可写成已完成 0.6B 实流实测。

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

### Benchmark、PowerShell 与 realtime 指标

- `[已成功]` 已补 `tools/run_benchmark.ps1` 作为统一入口，可串起 `0.6B / 1.7B × realtime / batch` 四组 benchmark：自动发现 ModelScope 模型目录、自动选取 `testfile/` 中最大 `.wav`、自动起停 `qasr_server`、自动汇总 JSON 与 Markdown 报表。
- `[已成功]` realtime benchmark 现采用“长音频裁固定时长 + 抖动 chunk + 实时 pacing”法：默认截取长音频前 `120s`，chunk 在 `200-600ms` 间随机，按累计音频时长限速发送。若目标是测吞吐、lag 与稳定前缀，这比人为设计复杂分布更稳，也更便于复现。
- `[已确认环境事实]` `testfile/` 中文长音频在 Windows 实盘文件名可能是 URL 编码形式；脚本不可写死展示名，应按目录扫描结果取真实路径。
- `[已确认环境事实]` 当前长样音已是 `16kHz / mono / PCM16`；对此类输入应走“直接复制 PCM”快路，不应在 PowerShell 内逐采样重采样，否则 benchmark 会先被预处理本身拖慢。
- `[已踩坑]` PowerShell 5.1 兼容性比 PowerShell 7 更严格；benchmark 脚本若包含较激进的内联表达式或依赖较新的库行为，容易在 Windows 实机翻车。Windows 路径上的工具必须按 PowerShell 5.1 实测，而不能只在新版本 shell 上看起来可跑。
- `[已证伪]` PowerShell 5.1 下直接用 `System.Net.Http.HttpClient` 组 multipart 上传，不足以稳定满足本项目 `cpp-httplib` 对文件字段的识别；服务端会报 `multipart field 'audio' or 'file' is required`。改为手工拼 `multipart/form-data` boundary 并走 `Invoke-WebRequest` 后，batch benchmark 才稳定通过。
- `[已确认修正]` realtime 报表原先把 `wall / audio_duration` 直接记作 `RTF`，此值天然包含实时 pacing 的等待时间，必然 `>= 1`，不能与 batch 直接比较。现应明确区分：`Processing RTF = lag / audio_duration`，`E2E RTF = wall / audio_duration`。
- `[已成功]` 2026-04-11 的 10s 快速验证结果如下：
   - `Qwen3-ASR-0.6B`：realtime `Processing RTF = 0.551`，batch `RTF = 0.36`
   - `Qwen3-ASR-1.7B`：realtime `Processing RTF = 2.64`，batch `RTF = 0.782`
- `[当前判断]` 0.6B 在当前 Windows / OpenBLAS / 16 logical CPUs 环境上，realtime 额外成本主要来自 chunk-by-chunk 的反复 encode/decode、session 管理与 polling 开销；batch 明显更快，但 realtime 仍处于可讨论优化区间。1.7B 则已明显算力不足，无法在当前机型上跟住严格实时输入。
- `[已成功]` 当前 benchmark 主脚本已支持“realtime 崩后重启 server，再继续跑 batch”，这样即使流式链路仍不稳，也能把 batch 结果保住，避免四组 benchmark 全部中断。
- `[已失败但必须留档]` 0.6B 在 Windows 上跑 `180s` 级 realtime 长流时，`qasr_server` 仍可能在 `/api/realtime/stop` 前后触发 `0xC0000005 (ACCESS_VIOLATION)`。`12s` smoke 可过，但长流不稳；故在 backend 未修稳之前，不宜把 `120s` 或 `180s` realtime 结果直接当作最终结论。
- `[已尝试但未根治]` 在 `qwen_clone_shared()` 中把 prepared prefill cache 指针清空，可排除一类 clone 共享可变状态问题，且对短时 smoke 有帮助；但它未能根治长时 realtime crash，说明尚有第二条崩溃路径，后续应优先检查 live session 的 finalize / cleanup、长窗口状态裁剪以及 stop 路径。
- `[当前建议]` 后续讨论模型性能时，应优先并列观察：batch `RTF`、realtime `Processing RTF`、`first_stable_wall_ms`、`final_lag_ms`。若只拿 realtime 的 `wall / audio` 去对比 batch，结论会系统性偏悲观且失真。
