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

### 10. 本机已实跑 `qasr_cli`

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

### 11. 官方实时与时间戳，本就分路

外部事实：

- Qwen 官方模型卡称 streaming 仅 vLLM backend 支持
- 且 streaming 不支持 batch，不支持 timestamps
- 精细时间戳交 `Qwen3-ForcedAligner-0.6B`

故：

- 主实时路先求稳定字幕
- 词级时间戳宜异步后补

### 12. “无停顿”之核心，不是 VAD，而是稳定前缀

AWS 文档与 Whisper-Streaming 皆指向同一结论：

- partial 可先出
- stable prefix 后提交
- 尾巴允许少量改写

此比“等停顿再整句冒出”更合字幕 UX。

### 13. `qwen-asr-learn` 已隐含正确方向

其 `--stream` 默认：

- `2s` chunk
- `8s` encoder window
- `rollback 5`
- `unfixed_chunks 2`
- `max_new_tokens 32`
- 长流自动裁历史

这说明参考实现并未依赖停顿；其主法本就是“滑窗 + rollback + frontier commit”。

### 14. vLLM 官方 realtime 默认分段偏保守

vLLM `qwen3_asr_realtime` buffer 默认 `5s` 才吐一段。

此足以保正确，不足以保字幕观感。
若求“看似准实时”，须在我们自研 runtime 内把 cadence 压到亚秒级 tick，而非直接照搬 `5s`。

### 15. 双路法值记，但不宜阻首版

2025 年论文已示：

- 快路 partial
- 慢路 rerank / refine

然其多赖额外训练或额外头。
本项目首版当先把“无停顿滑窗 + 稳定前缀”做稳，再议双路增强。

### 16. 实时接口第一版已改成三态文本

现 `/api/realtime/*` 与 `/api/capture/*` 已不再只回一段 `partial_text`，而回：

- `stable_text`
- `partial_text`
- `text`
- `finalized`

此使前端后续可做“稳定前缀不重绘”。

## 2026-04-11

### 17. Windows/OpenBLAS 先落低风险 encoder QKV 融合

- encoder 自注意 Q/K/V 原为三次独立 F32 线性层
- 现仅在较长序列时，临时拼成一次 GEMM，再拆回三路输出
- 不改模型权重所有权，不引入持久重复缓存
- 若 scratch 申请失败或序列太短，则自动回退旧路径

### 18. Windows 一键构建脚本不得写死本机路径

- 构建脚本须先按仓内相对目录约定检索 OpenBLAS
- 若当前 shell 未带 MSVC 环境，则用 `vswhere` + `VsDevCmd` 自动补齐
- configure/build/test 均走 CMake preset，避免脚本复制构建参数

### 19. Windows / SkylakeX 上，QKV 只该做“预打包融合”，不该做“每次临时拼接”

2026-04-11 于 Windows + OpenBLAS + SkylakeX + 16 logical CPUs，实测 `qasr_cpu_bench`：

- 8 线程下，`fused_packed` 相对 `separate`：
   `seq4 d1024 = 1.43x`，`seq13 d1024 = 1.07x`，`seq52 d1024 = 1.08x`，`seq104 d1024 = 0.93x`，`seq13 d896 = 1.47x`
- 12 线程下，`fused_packed` 相对 `separate`：
   `seq4 d1024 = 1.67x`，`seq13 d1024 = 1.15x`，`seq52 d1024 = 1.14x`，`seq104 d1024 = 1.20x`，`seq13 d896 = 1.34x`
- `fused_copy` 即“每次调用把 Q/K/V 拷成临时连续块再打一发 GEMM”在全部测点都比 `separate` 更慢

故：

- runtime 应保留“加载期预打包 + 运行期单 GEMM”
- 不应把“每次临时拼接”留在热路径里
- 线程数 8 与 12 皆可用，但最优值会随 shape 变，不可假设 12 恒优


### 17. decode cadence 不宜随包即跑

原先每来一包即整段重解，浪费甚大。
今改为固定最小样本门槛：

- 默认 `800ms`

故浏览器亦改 `800ms` flush 一次，与服务 cadence 对齐。

### 18. stop 时应再跑一次终态 refine

若 stop 只回最后一轮 partial，往往欠尾字。
今 `stop` 改再跑一次非流式 full-audio decode，并强制 flush 尾巴，终稿更稳。

### 19. “参考”与“依赖”须硬分

用户已明令：

- `qwen-asr-learn/`
- `whisper.cpp/`

只能参考，不得成编译或 include 依赖。

故今已改：

- 现用 C 核源码已内化入 `src/backend/qwen_cpu/`
- `httplib.h` 与 `json.hpp` 复制入 `vendor/third_party/`
- CMake 只指向 `vendor/`，不再指向参考目录

后续仍须继续：

- 从“vendor 快照”再走向“自研替换”

### 20. 共享局部变量入 handler，极易生并发暗伤

`RunServer` 中若以一枚外层 `Status status` 供多 handler 复用，则多请求并发时可互踩。
今已尽改为各 handler 各自局部 `Status`，此类变量不得再上提到共享作用域。

### 21. 命名也会形成依赖错觉

即使构建已只指向 `vendor/`，若接口仍名 `legacy_bridge`，会误导后续开发继续按参考工程思路走。

今已改：

- `legacy_bridge` -> `model_bridge`
- `LegacyAsr*` -> `AsrRun*`
- `QASR_LEGACY_BRIDGE_ENABLED` -> `QASR_CPU_BACKEND_ENABLED`
- `qasr_legacy_c` -> `qasr_cpu_c`

经验：边界名须反映本项目所有权，不得反映来源。

### 21.1 CPU 后端可内化，但许可须随行

`vendor/qasr_cpu/` 已改入 `src/backend/qwen_cpu/`。这样构建路径更像主项目后端，不再给人“临时 vendor 依赖”错觉。

必须保：

- `src/backend/qwen_cpu/LICENSE.upstream`
- `NOTICE.md`
- `docs/07-license-compliance.md`

### 22. 实时长流必须限内存

实时 ASR 不可假设用户会停，也不可令 `samples` 无限增长。

今已改：

- `RealtimePolicyConfig::max_decode_window_ms = 32000`
- session 只保近 `32s` PCM
- `sample_count` 记累计样本
- `retained_sample_count` 记当前保留样本
- stop 时若已裁旧 PCM，不以窗口结果覆盖既有全文

经验：计时与解码窗口须分离；否则一裁样本，cadence 与稳定前缀都会漂。

### 23. PCM chunk 必须明示二进制类型

用 `curl --data-binary` 若不设 `Content-Type`，httplib 可能按表单体处理，触发 `413`。

今后 `/api/realtime/chunk` 冒烟一律加：

- `Content-Type: application/octet-stream`

UI 已如此设置。

### 24. Docker 验收依赖 daemon

本轮执行 `tools/docker_linux_openblas.sh linux/arm64` 时，Docker API socket 不存在：

- `/Users/kurisu/.docker/run/docker.sock`

故 Linux/OpenBLAS 容器验收未能启动。
此非代码失败；后续跑 Docker 前须先确认 Docker Desktop / daemon 已起。

本机现状：

- Docker CLI 存在：`/usr/local/bin/docker`
- Docker client：`29.1.3`
- 本机架构：`darwin/arm64`
- context：`desktop-linux`
- buildx 插件存在
- daemon socket 缺：`/Users/kurisu/.docker/run/docker.sock`

按脚本设计，`tools/docker_linux_openblas.sh linux/amd64` 可触发 amd64 Linux 容器；但须 daemon 起后才能确认 QEMU/binfmt 或 Docker Desktop emulation 实际可用。

用户启动 Docker 后已验：

- `docker buildx ls` 显示 `linux/amd64`
- `docker run --rm --platform linux/amd64 ubuntu:24.04 uname -m` 输出 `x86_64`

故 amd64 Linux 虚拟可行。
后续 OpenBLAS 构建若失败，多半是 apt 源网络，不是架构虚拟失败。

### 25. HTTP 服务须先限队列

ASR 请求耗时远大于普通 HTTP。若 server 允许无限排队，CPU 慢时会堆内存并拖垮延迟。

今已设：

- httplib task queue 上限 `64`
- read / write timeout `30s`
- keepalive timeout `5s`
- realtime 活跃会话上限 `64`

后续还须给 async job 做 TTL 清理与硬上限。

### 26. 中文测试音频可入库，但须受控

`testfile/` 存中文 ASR 测试样音，用户已准许入库作为覆盖补充。

规则：

- 可提交 `testfile/*.mp3`
- 不提交临时转码 WAV/PCM
- 不提交模型文件
- 新增大体积样音前须先说明用途

### 27. 长音频 CLI 不能只等终稿

用户用中文长音频测试时，默认 CLI 只在末尾吐全文，观感差，也不利于观察进度。

参考 whisper.cpp 后确认：

- 内部生成 segment 后立刻回调
- CLI 只打印新增 segment
- 最终文件输出再遍历全部 segment

本项目当前 CPU 后端尚无 segment callback，先用 token callback 在 C++ 桥接层做段聚合：

- `--emit-segments`
- `--segment-max-codepoints`
- 标点触发 flush
- 无标点达阈值亦 flush

后续更优解：在 `src/backend/qwen_cpu` 增原生 segment callback，带近似时间窗。

### 28. 流式重复非多线程竞态

同一 30 秒中文样本矩阵复验：

- 非流式 `threads=1`：无重复
- 非流式 `threads=4`：无重复
- 流式 `threads=1`：重复
- 流式 `threads=4`：重复

结论：

- 不是模型文件问题
- 不是 BLAS/OpenMP 多线程竞态
- 是流式 append-only 提交策略问题

根因：

- 模型后续 chunk 会修正已提交前缀
- 旧逻辑从首个差异点继续外发
- 外部没有 delete/update 通道，故修正文会表现为重复文本

修法：

- 已提交 frontier 只前进，不回退
- 已提交前缀内的模型修正不外发
- 外发前用 `qwen_stream_skip_recent_duplicate_prefix` 扫近邻已提交 token，跳过 rollback/reanchor 回放旧 span

验收：

- `ctest --test-dir build/macos-accelerate-cli --output-on-failure` 通过
- 中文 30 秒流式 `threads=1/4` 均无重复

### 29. Docker apt 源须可禁用与限时

amd64 Docker/OpenBLAS 验收卡在 apt：

- Aliyun 源：`noble/universe amd64 Packages` 连接失败
- Ubuntu 官方源：`noble InRelease` / `noble-backports InRelease` 返回 `503`

已改脚本：

- `QASR_APT_MIRROR=` 可禁用镜像替换
- `QASR_APT_RETRIES` 控重试次数，默认 `3`
- `QASR_APT_TIMEOUT` 控 HTTP/HTTPS timeout，默认 `20`

本轮 Docker 未进入 CMake 构建，不能作为代码失败判据。

### 30. 长音频不可用超大 stream max_new_tokens

`%E9%A1%BE%E5%90%9B%E5%AD%90%EF%BC%8801%EF%BC%89.mp3` 时长 `1726.457s`，约 `864` 个 2 秒流式 chunk。

用户命令设：

- `--stream`
- `--emit-tokens`
- `--stream-max-new-tokens 512`

这会使每个 chunk 最多解 `512` token，长时间无输出，看似卡死。此非 ffmpeg 死锁。

处置：

- CLI / bridge 限 `stream_max_new_tokens <= 128`
- C 后端也硬限 `QWEN_STREAM_MAX_NEW_TOKENS_LIMIT=128`
- 长文件离线测试应优先用非流式 `--emit-segments`

### 31. 全模块补齐：C++ 抽象层实现

审计发现设计规格 9 层架构中约 82% 的 C++ 组件缺失（仅有 core/status、audio_types、timestamp、service/realtime、runtime 五个底层模块）。本次一次性补齐所有缺失模块：

**新增头文件 10 个**（`include/qasr/`）：

| 层           | 文件                                  | 主要类型/函数                                  |
|:-------------|:--------------------------------------|:-----------------------------------------------|
| core         | `core/state_machine.h`                | SessionState, RequestState, RealtimeTextLane, StreamChunkState 枚举 + 转换验证 |
| storage      | `storage/safetensors_loader.h`        | MappedFile (RAII mmap), SafeTensorIndex, ShardRegistry, TensorView |
| model        | `model/tokenizer.h`                   | Tokenizer (BPE), LoadVocabJson, LoadMergesTxt  |
| audio        | `audio/frontend.h`                    | ReadWav, ParseWavBuffer, Resample, ComputeMelSpectrogram, CompactSilence, StreamingAudioRing |
| inference    | `inference/encoder.h`                 | EncoderWeights, EncoderWindowPlan, EncodeChunk, EncodeAudio |
| inference    | `inference/decoder.h`                 | DecoderWeights, KvCache, Prefill, DecodeStep, BuildPromptEmbeddings |
| inference    | `inference/streaming_policy.h`        | StreamPolicyConfig, StreamChunkPlanner, EncoderCache, CommitFrontier, DetectDegenerateTail |
| runtime      | `runtime/session_manager.h`           | SessionManager (线程安全, mutex 保护)          |
| runtime      | `runtime/task_queue.h`                | TaskQueue (有界队列 + 背压 + 取消)             |
| runtime      | `runtime/realtime_session.h`          | RealtimeSession (per-session 流式状态)         |

**新增实现文件 10 个**（`src/`）：state_machine.cc, safetensors_loader.cc, tokenizer.cc, frontend.cc, encoder.cc, decoder.cc, streaming_policy.cc, session_manager.cc, task_queue.cc, realtime_session.cc

**新增测试文件 10 个**（`tests/`）：覆盖正常/极端/错误场景，共新增 ~120 个测试用例，全部 217 个测试通过。

教训：

- encoder/decoder `.cc` 实现为 stub（返回零值或 EOS），待接入 C 后端真正的权重加载和推理
- tokenizer 的 BPE 编码实现为简化版（逐字符匹配），待接真正的 merge 优先级算法
- 所有模块遵循开发手册：Pre/Post/Thread-safe 契约注释、RAII 资源管理、Status 错误传播

## 2026-04-11

### Enhancement 方案落地（QKV 融合 + 多线程 BF16→F32 转换）

基于 enhancement.md 的分析，实施了两项 CPU prefill 优化：

1. **Prefill QKV 投影融合**
   - 原来每层 prefill 分三次调用 `qwen_linear_nobias_bf16`（wq/wk/wv），即三次 BF16→F32 转换 + 三次 cblas_sgemm
   - 新增 `qwen_linear_nobias_bf16_qkv_prefill()`：一次转换到连续 F32 缓冲区（[4096, 2048]），一次 cblas_sgemm，再拆分输出
   - decode 路径（seq=1）不受影响，仍走 `qwen_linear_nobias_bf16_qkv` 的 BF16 NEON matvec

2. **多线程 BF16→F32 转换**
   - 发现 `bf16_to_f32_buf` 原为单线程 —— 对 gate_up_fused（25M 值）单次~1.5ms，28 层累计 ~140ms，占 prefill 的 ~55%
   - 新增 `bf16_to_f32_buf_threaded()`（复用现有 pthread pool 的 parallel_for）
   - QKV 融合路径用自定义 `qkv_cvt_worker` 单次 parallel_for 完成三段权重转换（避免三次 thread dispatch 开销）
   - `bf16_get_f32_view` 也改用多线程路径（影响 wo/gate_up/down 投影）

#### 基准数据（long_test.wav, 864 chunks, M3 Pro）

| 指标 | 原始 | 优化后 | 变化 |
|:-----|:-----|:-------|:-----|
| Prefill avg | 263.6 ms | 208.6 ms | **−20.9%** |
| Decode avg | 22.2 ms/tok | 22.4 ms/tok | ~0%（噪声） |
| 总推理时间 | ~917 ms/chunk | ~861 ms/chunk | **−6.1%** |

#### 关键经验

- **enhancement.md 方向正确但实现优先级不同**：文档首推 QKV fusion，实测单独收益仅 ~3.5%，真正大头是多线程 BF16→F32 转换（~17%）。一个结构性优化（减 BLAS 调用）不如一个执行级优化（消除单线程瓶颈）
- **Apple Accelerate BLAS 自管线程**：BLAS sgemm 用 GCD 自行并行，不用我们的 pthread pool。转换和 BLAS 是顺序关系，不会线程竞争
- **转换阈值**：小于 65536 个值时走单线程，避免 thread dispatch 开销（~40μs）超过计算本身
- **内存 layout 注意**：fused QKV 转换的 worker 须正确映射"逻辑偏移 → 三段不连续 BF16 源"，原始实现有指针负偏移 UB，已修正为 local_pos 计算
- **识别准确率不变**：short_test.wav 和 long_test.wav 转录结果与优化前一致

#### 未实施项目及原因

| 方案 | 结论 | 理由 |
|:-----|:-----|:-----|
| 持久化 F32 权重缓存 | 延后 | +5.5GB 内存，且多线程转换已将转换开销压至 ~30ms（原 ~140ms） |
| Shape-aware dispatch | 延后 | 实测 delta_prefill 最小为 32，BLAS+AMX 在此规模仍优于 per-token matvec |
| vDSP 向量算子替换 | 延后 | RMSNorm/RoPE 等非瓶颈，优先级低 |
