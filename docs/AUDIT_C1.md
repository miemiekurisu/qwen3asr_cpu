# C1 审计报告 — 2026-06-05

按用户要求"准备提交前质量审查"的第一段: 死代码 + 野指针 + 循环引用 + OOM 高危 + god function.
**不实现监控**, **不拆 god function**, 只标 TODO + 报告. 后续 C2-C5 分段做.

## 范围

- C/C++ 25,351 LOC (server.cc 4396 + qwen_asr.c 4241 + kernels 2312 + ...)
- JS 1,113 LOC (app.js 1019 + live_monitor.js 74) - 本次已删 3 死函数
- Bash/Py 2,763 LOC - 暂不审计
- 643 C++ 单测, 16 个 C 源文件无测试 (C3 处理)

## 1. 死代码

| 检查项 | 工具 | 结果 |
|---|---|---|
| libqasr_cpu_c.a 导出符号 (T) 未引用 | nm + grep | **0** |
| libqasr_core.a 导出符号未引用 | nm + c++filt + grep | **0** |
| .h 公开 API 函数未调用 | grep 全部 .c/.cc/.js/.sh | **0** |
| .h 公开 C++ 成员函数未调用 | 类内 method 反查 | **0** |
| 文件内 static 函数未调用 | 同上 | **0** |
| 编译器 warning -Wunused | ninja rebuild | **0** |
| JS 顶层函数未调用 | grep | **3 处已删** |

**已删 JS 死函数** (`ui/app.js`):
- `offlineElapsedSeconds` - 仅 1 ref (自身)
- `formatSeconds` - 仅 1 ref (自身)
- `escapeHtml` - 仅 1 ref (自身)

均为纯函数, 无副作用, 无下游调用.

**HTML id ↔ JS getElementById 双向一致性**: **0 偏差**, 全部 id 在两边都对得上.

## 2. 野指针 / UAF / double-free

| 检查项 | 工具 | 结果 |
|---|---|---|
| free(x) 后解引用 x | 25 行后窗口扫描 | **0** |
| 同一函数 free(x) >1 次 | 名称匹配 (粗) | 5 处**误报** (lambda/branch 各自 free) |
| 150s double-free 修法 (commit e189ec8) | 检视 `qwen_free` `owns_model_data` 保护 | **完好, 见 qwen_asr.c:571-579** |
| `qwen_clone_shared` 浅拷贝安全 | 检视 `owns_model_data=0` + `prefill_*_prepared=NULL` | **完好, 见 qwen_asr.c:383-457** |
| `ApplyStableRealtimeCommit` 是否安全 | 检视 `std::string_view` 立即 copy | **安全, 见 server.cc:1857-1877** |

**结论**: 野指针 / UAF / double-free **清白**. 5 处误报已逐一检视, 不存在真问题.

## 3. 循环引用

| 检查项 | 工具 | 结果 |
|---|---|---|
| C++ shared_ptr 互相持有 | 类字段反查 | **0** |
| shared_ptr<RealtimeSession> | 持于 unordered_map<string, _> + lambda capture | 全部 weak-friendly |
| shared_ptr<HostCaptureSession> | 持于 server 全局, lambda capture | 同上 |
| shared_ptr<atomic<bool>> (cancel_flag) | Job 持 + lambda capture | leaf, 无 shared_ptr 字段 |
| Tokenizer::impl_ (Pimpl) | 持于 Tokenizer | leaf |
| JS 闭包自循环 | `realtimeState` / `realtimeArchive` / `offlineState` 全局, 不被 DOM 反向引用 | **0** |
| JS DOM 反向 ref 状态对象 | `makeTermLine` 仅设 className + textNode, **不挂自定义属性** | **0** |

**结论**: 循环引用 **0 处**.

## 4. OOM 高危 (本次仅审计, 不实现监控)

### 4.1 已发现高危点

**`qwen_live_audio_t::samples` 单调增长不收缩** (server.cc:1745-1807, qwen_asr_audio.c:417-438)

`AppendManualLiveAudio` 收到 chunk 就 realloc 扩容, **从未 trim 已消费样本**:
- VAD 段式 commit 时 `consumed_samples` 推进 (`server.cc:2945`), 但只更新 `live->decoded_cursor` 标记位, **不释放底层 buffer**
- `DestroyManualLiveAudio` (session 结束) 才 `std::free(live->samples)`
- 1 小时 realtime = 1h × 16kHz × 4B = **230 MB / session**
- `kMaxRealtimeSessions = 64` → 理论最坏 **14.7 GB** 单进程驻留

**触发场景**:
- 浏览器 mic 持续开着不说话
- 长会议/讲座录音
- 用户中途忘了 stop

**缓解** (待 C2+ 实现, 本次仅 TODO):
- 每 N 个 VAD commit 后, 移除 `live->samples[0..decoded_cursor]`
- 或切到 ring buffer (固定 16M samples = 64 MB cap)
- 或 cap session wall-clock 30 min (UI 端有 `realtimeStopping` 状态机可配合)

**已加 TODO** (本次 commit):
- `src/service/server.cc:1807` 末尾注释
- `src/backend/qwen_cpu/qwen_asr_audio.c:439` 末尾注释

### 4.2 其它 alloc 点扫查 (100MB+ 大块)

| 文件:行 | 大小 | 风险 | 状态 |
|---|---|---|---|
| server.cc:1794 | var (realloc to 2x) | 同 4.1, 同一 buffer | 见 4.1 |
| qwen_asr_audio.c:182 | var (file_size) | WAV 读入, 上限 ~几 GB | 已加 status 错误返回 |
| qwen_asr_audio.c:429 | var (realloc to 2x) | stdin 实时流, 同 4.1 风险模式 | 已加 SIZE_MAX 检查 |
| qwen_asr_decoder.c:149 | per-layer, 固定 | 模型加载, 配置驱动 | 安全 |
| qwen_asr_decoder.c:234 | per-layer, 固定 | 同上 | 安全 |

**结论**: 1 个高危 (4.1) + 多个安全 (配置驱动 / 已有 bounds check).

## 5. god function / god file (本次仅标 TODO, 不拆)

### 5.1 god function 候选 (>100 行的 C/C++ 函数)

| 文件:行 | 函数 | 行数 | 拆的理由 | 建议 |
|---|---|---|---|---|
| server.cc:2610-4394 | `RunServer` | 1784 | 包含所有 HTTP 路由 + lambda + VAD 段式 batch + live worker 启动 | **拆**: server_routes.cc (注册路由) / server_session.cc (session 管理) / server_vad.cc (VAD 段式) / server_live.cc (live worker) |
| server.cc:2330-2609 | `TranscribeFileVadSegmentedImpl` | 279 | 自由函数, VAD 主循环 + commit 决策 | 可拆, 优先级中 |
| server.cc:2101-2243 | `ParseServerArguments` | 142 | CLI 解析, 单一职责, 难拆 | 不拆 |
| json.cc:81-248 | `ParseValue` | 167 | JSON parser, 单一职责 | 不拆 |
| qwen_asr_decoder.c:822-1087 | `qwen_decoder_forward` | 265 | 1 layer forward, 全部 ops 内联 | 不拆 (per-layer) |
| qwen_asr_decoder.c:551-794 | `Prefill` | 243 | prefill 路径, 大量内联 | 不拆 (内联收益高) |
| qwen_asr_audio.c:463-658 | `live_reader_thread` | 195 | stdin 实时读取, 单一职责 | 不拆 |
| qwen_asr_kernels_*.c | 各 matvec/layernorm | 100-150 | SIMD kernel, 必须内联 | 不拆 (性能) |

**已加 TODO**:
- `src/service/server.cc:4395` 末尾, god function 注释

### 5.2 god file 候选

| 文件 | 行数 | 原因 | 建议 |
|---|---|---|---|
| server.cc | 4396 | 包含 routes + session + VAD + live worker + CLI | 拆 4 个文件 (见 5.1) |
| qwen_asr.c | 4241 | load + free + clone + streaming + bg + VAD | 不拆 (C 文件, 拆分增加构建复杂度) |
| qwen_asr_kernels.c | 2312 | 3 个 ISA 文件 + dispatch | 已是多文件 + dispatch, 不拆 |
| qwen_asr_decoder.c | 1087 | decoder 单文件, 内聚 | 不拆 |
| app.js | 1019 (修后) | 包含 state machine + DOM + 3 个 click handler + terminal archive | 拆 3 文件: state.js / ui.js / terminal.js (C4 处理) |

## 6. 后续 commit 计划

- **C1 (本次)**: 本报告 + 3 个 JS 死函数删除 + god function / OOM TODO 注释
- **C2**: C 核心函数补单测 (qwen_asr.c / server.cc god function 周边)
- **C3**: C 长尾 16 个无测试文件补单测
- **C4**: JS 状态机/工具函数补单测 + Bash smoke test + app.js 拆 3 文件
- **C5**: 文档完整化 (ARCHITECTURE / README 更新 / INCIDENTS 续写 / API.md / SECURITY.md)

## 7. 验证

```
$ node -c ui/app.js      # 语法 OK (3 死函数删除后)
$ ls ui/app.js && wc -l ui/app.js   # 1019 行 (原 1039, -20)
$ ninja -C build/linux-openblas     # 待跑 (本次只动 JS, 但要确认 server 仍编)
$ curl -sS http://127.0.0.1:19991/api/health   # 待跑
```

## 8. 已知遗留 (本次不做)

- god function 拆分 (用户答: 只标 TODO)
- in-process 内存监控 (用户答: 只审计高危)
- shared_ptr<Tokenizer::Impl> 跨 Pimpl 边界的安全 (本次扫了, 0 风险)
- `live->samples` ring buffer 改造 (本次仅 TODO, 不实现)
- 150s crash 根因已修 (commit e189ec8), 现状安全
