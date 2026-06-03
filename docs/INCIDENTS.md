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
