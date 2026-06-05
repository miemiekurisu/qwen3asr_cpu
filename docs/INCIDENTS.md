# Incident Reports

事后调查报告。每条记录一次故障 / 修复，附复现命令、根因、回滚方案。

---

## 2026-06-04: 150s `double free or corruption (!prev)` 崩溃

**症状**: `qasr_server` 在反复 start/stop 实时 session（90 s audio，1 session/6 s 节奏）约 150 s 后，dmesg 报 `double free or corruption (!prev)` 并 `SIGABRT` 退出。100% 复现：`tools/e2e_realtime_chain.py` 循环 ~25 个 session 必崩。

**根因**（两步链）：

1. `qwen_clone_shared()` 用 `ctx->decoder = src->decoder;`（浅 struct copy）共享所有 decoder 字段指针，包括 `tok_embed_suffix_max` 和 `lm_head_suffix_max`。
2. `qwen_free()` 中这两个 free 写在了 `if (ctx->owns_model_data) { ... }` 块**外面**，所以每个 clone 退出时都会释放共享 chunk。源 ctx 留下悬空指针，下个 clone 的 `decoder = src->decoder` 又复制这个悬空指针；下个 clone 退出时二次 free → glibc `!prev` 检测到堆破坏 abort。

辅助观察：dmesg 现场常报 `qwen_dot_f32_avx` / `qwen_softmax_causal_generic` / `dnnl::impl::matmul_desc_init` 的 SEGV。这是因为 freed chunk 被 malloc 重分配后，残留数据被当作注意力分数 / 权重读出脏值，最终在 encoder prefill 的 matvec 里崩。

**复现命令**（修复前）：

```bash
export QASR_MODEL_DIR=$HOME/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/<rev>
QASR_MODEL_DIR=$QASR_MODEL_DIR tools/run_linux_server.sh --detach
python3 - <<'PY'
import json, time, urllib.request, wave, numpy as np
w = wave.open("testfile/english_60s.wav","rb"); raw = w.readframes(w.getnframes()); w.close()
audio = np.frombuffer(raw, dtype=np.int16)[:5*16000]
audio = np.concatenate([audio, np.zeros(16000, dtype=np.int16)]).astype(np.int16)
def start(): return json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:19991/api/realtime/start", b"", method="POST", headers={"Content-Type":"application/octet-stream"}), timeout=30).read())["session_id"]
def post(p, b=b""): 
    r = urllib.request.Request("http://127.0.0.1:19991"+p, data=b, method="POST", headers={"Content-Type":"application/octet-stream"})
    try: return urllib.request.urlopen(r, timeout=30).status
    except: return 0
for _ in range(30):
    sid = start()
    for j in range(0, len(audio), 3200):
        post(f"/api/realtime/chunk?session_id={sid}", audio[j:j+3200].tobytes()); time.sleep(0.05)
    post(f"/api/realtime/eof?session_id={sid}", b""); time.sleep(2.5)
    post(f"/api/realtime/stop?session_id={sid}", b"")
PY
dmesg | tail -3    # 看 double free
```

**修复**（`src/backend/qwen_cpu/qwen_asr.c:570-583`）：

把 `tok_embed_suffix_max` / `lm_head_suffix_max` 的 free 移到 `if (ctx->owns_model_data) { ... }` 块**内**，确保只有源 ctx 释放它们；clone 留下只读共享指针。

**辅助修复**（`src/backend/qwen_cpu/qwen_asr_kernels.c`）：

加 `parallel_dispatch_lock`/`unlock` 串行化 `parallel_for()`。原因：全局 thread pool 的 `tp.fn/arg/n_done/generation` 是单 struct，并发 `parallel_for` 会让 worker 跑到第二个 caller 的 `fn(arg)` 上，第一个 caller 的 `n_done` 被第二个 caller 复位。**单 session 场景下不会触发**（live_worker 是唯一调用 `parallel_for` 的线程，bg encoder 走 TLS `tl_bg_thread=1` 的 inline 路径），但作为防御性修法保留。性能影响：单 session 锁永远不争用，开销 ~1 ms / 9.5 s ≈ 0.01%。

**验证**：

| 场景 | 命令 | 结果 |
|---|---|---|
| 60 sequential sessions | `python3` 循环 start/stop × 60 | 0 错 |
| 5 min 满载（70 realtime + 4 CLI 0.6B）| `qasr_cli` × 4 + python loop | 0 错，server 健康 |
| ASAN build × 33 sessions | `build/asan/qasr_server` | 0 错 |
| 4 并发 realtime sessions | python threading × 4 | wall 4.80 s vs 单 4.53 s（+6%） |
| CLI 0.6B english_60s 回归 | `qasr_cli --audio testfile/english_60s.wav` | 9.48 s, 文本正确 |
| ctest `qasr_unit_tests` | `ctest --test-dir build/linux-openblas` | 100% PASS |

**回滚方案**：

```bash
git revert <commit>           # 直接回滚
# 或手动: 把 qwen_asr.c:570-583 还原到 owns_model_data 块外
```

---

## 排查过程（简短复盘）

1. **方向 1 - 内存泄漏**：30 iter / 245 s RSS 稳定 2759 MB。**否定**。
2. **方向 2 - VAD 边界 / 异常音频**：12 边界 case（0.5 s / 50 ms / 1.0 s / 1.001 s 静音、39.99 / 40.0 / 40.01 s 强制切、截断 0.5/1.0/1.5/2.0 s 真实音频、0 字节）全过。**否定**。
3. **方向 3 - 并发争抢**：8 并发 batch 稳定；4 CLI + 1 realtime 满载 5 min 必崩。**复现**。
4. **dmesg 现场**：6 个 crash 全部 `qasr_server[PID]`，全部在 encoder prefill / attention 数值计算（`qwen_dot_f32_avx` / `qwen_softmax_causal_generic` / `dnnl::impl::matmul_desc_init`）。强烈指向**堆破坏**。
5. **ASAN 复现**：`-DQASR_ENABLE_ASAN=ON` 重建 `build/asan`，2 sessions 后 ASAN allocator 在第一个 `free()` 自己 SEGV（chunk header 已被破坏）。**信号**：被 free 的指针 `ctx->decoder.tok_embed_suffix_max` 在堆破坏时已不可信。
6. **MALLOC_CHECK_=3 复现**：`build/silero-test/qasr_server` + `MALLOC_CHECK_=3`，25 sessions 后打印 `double free or corruption (!prev)`。**根因浮现**。
7. **代码审计**：`qwen_clone_shared` 第 390 行 `ctx->decoder = src->decoder;` 浅拷贝 + `qwen_free` 第 573-574 行 free 写在 `owns_model_data` 块**外** → 共享指针被多次释放。**根因锁定**。

工具用过的：valgrind（10× 慢，3 min 才跑完 1 session，弃用）、ASAN（chunk header 已被破坏后自身 SEGV）、MALLOC_CHECK_=3（glibc 准确报告）。**最终** MALLOC_CHECK_=3 是性价比最高的工具。

---

## 2026-06-05: UI 实时启动按钮逻辑错误（"按第二次没反应" / 旧信息丢失）

**症状**: 用户在浏览器里按 Stop 后再按 Start，第二次 Start 无响应；或允许在直播中点 Clear/Export 误操作；或 Start 后旧文字被整块清空，违反"旧信息保留"原则。

**根因**（三点叠加）:

1. `startRealtime` click handler 的 `finally` 块强制 `startRealtime.disabled = false`，覆盖了 `updateControlAvailability()` 算出的 `true`（在启动/停止中应当灰化）。
2. 没有 `realtimeStopping` 标志位。Stop 后 `activeFeature` 还在等后台识别延时清理（有时 200-500ms），用户在这个窗口里再点 Start 命中 `hasRealtimeSession()`，早返回静默。
3. 第二次 Start 调用 `resetRealtimeArchive()` 整块清掉所有 `done` 行，违反"第二次 Start 应在旧文字下方追加"的需求。

**复现**（修复前）:

1. 打开 Web UI，点 Start → 允许麦克风 → 说话。
2. 等出现 2-3 句文字 → 点 Stop。
3. 立即点 Start（无延迟）。现象: 按钮看起来亮但没反应（console 无日志）。
4. 等待 5s + 再次点 Start → 旧文字整块消失，新文字从头开始。

**修复**（`ui/app.js`）:

- 新增 `realtimeStopping` 标志，`stopRealtime` click handler 入口置 true，`finally` 调 `updateControlAvailability()` 解禁（不再硬置 disabled）。
- 新增 `softResetRealtimeArchive(newSessionId)`：保 `done` 行 → 把 `typing` 行（如果有文字）提交 `done` → 删空 cursor → 加新空 cursor。
- 重写 `updateControlAvailability()` 统一管 4 态（idle/starting/live/stopping）:

  | 状态 | Start | Stop | Clear | Export | AudioMeter |
  |------|-------|------|-------|--------|------------|
  | idle (无活) | ✅ | ❌ | ✅ | ✅(有字) | 隐 |
  | starting | ❌ | ❌ | 隐 | ❌ | 隐 |
  | live | ❌ | ✅ | 隐 | ❌ | 显 |
  | stopping | ❌ | ❌ | 隐 | ❌ | 隐 |

- `clearRealtime` / `exportRealtimeText/Json` 在 idle 之外一律 disabled。
- `startRealtime` click handler 入口早返 `if (realtimeStarting || realtimeStopping) return;` + `finally { realtimeStarting = false; updateControlAvailability(); }`，3 道闸确保不重入。

**验证**（手工）:

| 场景 | 结果 |
|------|------|
| Start → Stop → 文字保留 | ✅ |
| Stop 立即再 Start → 旧文字下追加 | ✅ |
| Start 期间点 Start → 按钮置灰无反应 | ✅ |
| Live 期间点 Clear → 按钮 disabled | ✅ |
| Live 期间 Export → 按钮 disabled | ✅ |
| Stopping 期间点 Start → 按钮 disabled | ✅ |
| Idle 期间 Export(有字) → 可下载 | ✅ |

**回归**:
- `node -c ui/app.js` PASS
- `node tests/state_pure_test.js` 24/24 PASS（`computeSoftResetLines` 是 `softResetRealtimeArchive` 的纯逻辑镜像）
- `bash tools/smoke_test.sh` 18/18 PASS

**回滚方案**:
```bash
git revert <commit>  # 直接回滚 UI 改动
```

---

## 2026-06-05: OOM 风险 — `qwen_live_audio_t::samples` 单调增长

**症状**: 长跑实时 session（≥1h），RSS 持续上涨。审计发现 `qwen_live_audio_t::samples` 在整个 session 生命周期内只 realloc 不 trim，单 session 1h 累积 ~230 MB；按 `kMaxRealtimeSessions=64` 上限理论可吃 14.7 GB。当前未触发 OOM 因为单 session 短跑（<5 min）测试覆盖不到，**这是 C1 审计发现的高危风险点**。

**根因**:

`AppendManualLiveAudio`（`src/service/server.cc:1764`）调用 `realloc()` 扩展 `live->samples` 容量，但不释放已消费的前缀；`DestroyManualLiveAudio`（`src/service/server.cc:1745`）在 session 退出时才整体 `free()`。`live_audio_append`（`src/backend/qwen_cpu/qwen_asr_audio.c:417`）同模式。

VAD commit 路径在 `ApplyStableRealtimeCommit` 之后更新 `live->decoded_cursor`，但 cursor 之前的样本从未被 trim。

**当前缓解**（已加 TODO 注释）:

`src/service/server.cc:1808-1813` 和 `src/backend/qwen_cpu/qwen_asr_audio.c:439` 已加 TODO 注释指向 `docs/AUDIT_C1.md §4.1`。

**待实施**（任选其一）:

| 方案 | 改动量 | 内存上限 | 复杂度 |
|------|--------|----------|--------|
| Ring buffer 固定 cap 64 MB | 中 | 64 MB × 64 sessions = 4 GB | 中（需改 audio append 逻辑） |
| 周期 trim `samples[0..decoded_cursor]` | 小 | 1h session = 230 MB → ~10 MB 稳态 | 小（VAD commit 时 memmove） |
| Session wall-clock cap（≥1h 强制结束） | 小 | 0 额外（只限制时长） | 小 |

**当前策略**: 暂不修，因为实际用户场景为短跑（5-10 min）。待真实长跑需求出现时再做 C5.2+。

**回滚方案**: TODO 注释无运行时影响，无需回滚。

---

## 2026-06-05: God function `RunServer` (1784 行)

**症状**: `RunServer` 函数 1784 行（`src/service/server.cc:2610-4394`），单文件 4241 行，难以单测、code review、扩展。

**当前状态**: 仅 `TODO(god-function-audit-C1)` 注释，按 C1 审计策略"只标注 TODO，不拆分"暂不动。

**待拆**（`docs/AUDIT_C1.md §5.1` 建议）:

| 子模块 | 行数估计 | 职责 |
|--------|----------|------|
| `RoutesRegistration` | ~150 | 注册 HTTP 路由表 |
| `SessionLifecycle` | ~400 | session start/chunk/stop/eof/stream |
| `VadSegmentedWorker` | ~600 | VAD 段式解码 + 段提交 |
| `LiveWorker` | ~500 | 实时 worker 主循环 |
| `CliDispatcher` | ~100 | 解析 HTTP 路径参数 |

**回滚方案**: 不动则无需回滚。

---

## 2026-06-05: `--decoder-int8` 移除 (C8)

**症状**: `--decoder-int8` 在 C1 审计时就已标记"显著降低识别质量":
- 语言一致性下降 (Chinese 音频里出 English 片段)
- 中英混杂泄漏 (code-switch)
- 低置信度音频上更易产生幻觉

加 `--realtime-decoder-int8` 也救不了 — 解码器 INT8 量化的是自回归 Qwen3 LM 的权重, 不是缓存. 选项本身有毒.

**修复** (C8): 整套移除.
- `qwen_set_decoder_int8` C API 删除 (无调用方)
- `decoder_int8` / `int8_dec_layers` / `n_int8_dec_layers` 字段从 `qwen_ctx_t` 删除
- `qwen_decoder_prepare_int8` / `qwen_decoder_free_int8` (oneDNN path + stub) 删除
- `qwen_int8_dec_layer_t` struct 删除
- `--decoder-int8` / `--realtime-decoder-int8` 从 `qasr_cli` / `qasr_server` 删除
- `qwen_asr_decoder.c` 两个 QKV/wo/gate_up/down 处的 `if (il) { qwen_int8_matvec(...) } else { ... }` 三元块全部塌缩为 `else` 路径
- `QwenCloneSharedDisablesInt8` → `QwenCloneSharedDisablesEncoderInt8` (只测 encoder)
- `onednn_int8_test.cc` 删 group 3 (8 个 CLI 测试)

**保留**:
- `--encoder-int8`: encoder 是 Whisper-style conv, INT8 风险小, 收益明显
- `qwen_int8_enc_layer_t` + `qwen_encoder_prepare_int8` + `qwen_encoder_free_int8` 全保留
- `qwen_asr_encoder.c` INT8 路径保留
- `onednn_int8_test.cc` group 1+2 (encoder-relevant) 全部保留

**验证**:
- C++ 665/665 PASS (linux-openblas)
- ASAN: 654 PASS, 0 ASan/UBSan errors

**回滚方案**: git revert afb70d1..HEAD. 但强烈建议不回滚, 理由如上.

## 2026-06-05: `--encoder-int8` 入口暂禁用 (post-C8)

**症状**: `--encoder-int8` 在 C8 当时被标"保留", 但 post-C8 用户进一步报告: 偶发转写质量退化 (Whisper-style conv 在某些声学分布上 INT8 仍掉点), 而风险/收益不划算 (encoder 内存只占 ~20%, 收益有限). 暂禁用, **代码保留** 方便后续恢复.

**修复** (c5b67cd): 入口屏蔽, 不是删除.
- C++ 入口: `qasr_cli` / `qasr_server` 的 `--encoder-int8` 解析仍接受参数, 打 stderr warning, **不调** `qwen_set_encoder_int8`
- C++ 调用点: 5 处 `qwen_set_encoder_int8` 调用 (server startup 1 + server clone 1 + model_bridge 3) 改 no-op block + 注释
- `qasr_server` / `qasr_cli` 的 `--help` 输出删 `--encoder-int8` 行 (用户看不到, 老脚本传参不报错)
- C 层 API 完整保留: `qwen_set_encoder_int8` / `qwen_encoder_prepare_int8` / `qwen_encoder_free_int8` / `qwen_int8_enc_layer_t` / `qwen_asr_encoder.c` INT8 路径 — 全部不删

**测试**:
- 已有测试不变 (没有 `RunServerFailsIfEncoderInt8Broken` 之类), 因为"暂禁用"是从用户视角屏蔽, 内部 API 行为不变
- `tests/ui_async_test.js` (新增 7 测试) 不需要为这个改

**回滚方案**:
1. `git revert c5b67cd` (解 5 处 no-op block, 恢复 --help 行)
2. 不需要重新编译 C 层 (代码一直在)

**为什么用"屏蔽"不"删除"**:
- 用户对"完全删除"敏感, 担心后续想用时找不到
- 屏蔽的语义清楚: "**选项不工作**, 但你仍可以传, 不会报错"
- 后续如需恢复, 改 5 处 no-op block 即可, 1 行 C++ + 1 行 help

## 2026-06-05: UI `offlineStop` 5xx 静默被覆盖 (c5b67cd 修复)

**症状**: 用户点击 batch 任务的 "停止" 按钮, 服务端 cancel 返回 5xx (例如 `job already gone` 或 transient DB 错). UI 显示:
1. 状态文本瞬间变 "停止中..." (同步前缀)
2. 30ms 后 catch 把状态改成 "停止失败: job already gone" (用户能看到)
3. **300ms 后下一次 poll 又把状态改回 "转写中: 0.6s"** ← bug: 错误信息被静默吞掉
4. 用户以为"停止按钮没生效", 实际 cancel 早失败了, 但 UI 假装一切正常

**根因**: `submitOfflineViaAsync` 的 poll 循环 (300ms 一次) 调 `updateOfflineAsyncStatus(job, startTime)`, 无条件写 `offlineStatus.textContent = "转写中: 0.6s"`. 当用户已点 Stop, 状态行已经属于 stop 流, 不该被 poll 覆盖.

**修复** (c5b67cd): 加 `offlineState.stopError` 字段.
- 5xx catch: 设 `stopError = error.message`, 状态行写 "停止失败: ..."
- poll 循环: `if (!offlineState.stopRequested && !offlineState.stopError) updateOfflineAsyncStatus(...)` — 错误显示期间不刷新
- 终态分支 (`cancelled` / `failed` / `completed`): 清 `stopError`, 让后续行为正常
- `stopRequested` 在成功 + 失败两条路径都清, 用户能重试 Stop

**测试**: `tests/ui_async_test.js` 7 测试之一 (`offlineStop: 5xx cancel response surfaces an error and re-enables Stop retry`) 覆盖. 验证 (a) 状态行显示 "停止失败: ..." (b) Stop 按钮在错误后仍可点击 (重试).

**回滚方案**: `git revert c5b67cd`. 但回滚后 5xx 错误会再次被静默覆盖.
