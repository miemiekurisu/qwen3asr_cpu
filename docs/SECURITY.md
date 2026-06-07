# Security Model

qwen3asr_cpu 的安全模型、威胁清单、缓解措施。本地推理服务，**不联网**，不收集 telemetry，但暴露 HTTP 端口时仍需考虑 LAN / 公网威胁。

## 信任边界

```
┌─────────────────────────────────────────────────────────────┐
│  TRUSTED (本地用户)                                          │
│    - 操作 server 的用户                                       │
│    - Web UI 浏览器 (via HTTPS)                                │
│    - 局域网内 curl / python 客户端                             │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP / HTTPS (单端口)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  UNTRUSTED INPUT                                             │
│    - multipart upload: 任意文件 (PDF/EXE/...)                │
│    - JSON body: 任意字段                                     │
│    - SSE query string: session_id                            │
│    - URL path: /api/jobs/:id                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │ 解析 + 校验
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  qasr_server (C++ 进程)                                      │
│    - 拒绝畸形输入 (Status code → 4xx)                         │
│    - ffmpeg 子进程 (沙箱: 显式 argv, 不通过 shell)             │
│    - 文件路径: 限制在 --ui-dir 和 --model-dir 下              │
│    - 内存: 64 MB 上限 / chunk, kMaxRealtimeSessions=64       │
└─────────────────────────────────────────────────────────────┘
```

## 威胁清单

### 1. Shell 注入 (历史: 2026-06-04 修复)

**威胁**: 客户端提供 `--output /tmp/foo; rm -rf /` 之类的路径, server 用 `system()` 拼命令会执行任意命令。

**现状**: ✅ **已修**. `BuildFfmpegArgv()` (src/audio/ffmpeg_argv.cc) 用 `posix_spawnp` + 显式 argv，不通过 shell。`tests/ffmpeg_argv_test.cc` 24 个 case 覆盖反引号 / `$()` / `;` / `&` / `|` / 引号字符在 input/output 路径中不执行。

**回归测试**:
```bash
ctest --test-dir build/linux-openblas -R ffmpeg_argv
```

### 2. 路径穿越

**威胁**: `GET /api/jobs/../../etc/passwd` 之类，server 路径处理不当读任意文件。

**现状**: ✅ **已修**. HTTP router 用 cpp-httplib 的 path 参数匹配，:id 只接受 `job_*` / `rt_*` 前缀。UI 静态资源路径写死为 `/app.js` `/style.css` `/index.html` 等固定列表 (server.cc:3471-3500)。

### 3. 任意文件上传 → RCE

**威胁**: 上传恶意 ELF/EXE 当音频，server 用 ffmpeg 转换时如果 ffmpeg 有 CVE，可能执行任意代码。

**现状**: ⚠️ **部分缓解**:
- ffmpeg 进程用受限的 argv 调用，不接受 shell。
- 文件大小限 64 MB (server.cc 上限, `MAX_ASYNC_UPLOAD_BYTES`)。
- 不在 user-writable 目录工作 (ffmpeg 用临时目录)。
- **但** ffmpeg 自身 CVE 不在项目控制范围内。建议:
  - 定期 `apt update && apt upgrade ffmpeg`。
  - 公网暴露时用 reverse proxy + 文件类型白名单 (audio/wav, audio/mpeg, audio/mp4, audio/flac, audio/ogg) 限制。

### 4. 资源耗尽 (DoS)

**威胁**: 攻击者并发上传大文件 / 创建大量 realtime session，吃光 CPU/内存/磁盘。

**现状**: ⚠️ **部分缓解**:
- `kMaxRealtimeSessions=64` 硬上限 (server.cc:377)。
- 音频文件大小限 64 MB (`MAX_ASYNC_UPLOAD_BYTES`)。
- async job 池有界 (`job_pool` 的 max threads = nproc)。
- **未实现**: 单 IP 速率限制, 客户端并发限制, 临时文件清理 cron。

**建议生产部署**:
- 反代层 (caddy / nginx) 加 rate limit: `rate_limit 10r/s`。
- systemd unit 加 `MemoryMax=8G` 触发 OOM killer 优雅重启。

### 5. mic 权限劫持 (Web UI)

**威胁**: 用户访问 `http://<server-lan-ip>:8080` (HTTP), 浏览器拒绝 mic 权限, 用户在地址栏改成 `https://...` 但 DNS 劫持到攻击者 IP。

**现状**: ✅ **已修**:
- 默认 `tools/run_linux_server.sh --https` 启 HTTPS 反代。
- 自签 cert 每次 `mktemp -d` 生成，**不**持久 (避免误信过期 cert)。
- 想持久 cert: `export QASR_TLS_CERT_DIR=/path/to/cert`，自己管 trust store。
- 启动 log 显式打印 "Using ephemeral self-signed cert, browser will warn"。

**已知陷阱**: 浏览器对自签 cert 弹安全警告，需用户手动 "Advanced → Proceed"。

### 6. 模型 / 权重篡改

**威胁**: 攻击者改 `model.safetensors` 让模型输出恶意内容 (jailbreak / 数据外泄)。

**现状**: ✅ **缓解**:
- safetensors 格式本身防任意代码执行 (与 pickle 不同)。
- model_dir 路径在启动时校验存在性 + 含 `model.safetensors` 或分片文件。
- **未实现**: safetensors 文件 SHA256 校验, 模型签名验证 (Qwen 官方未提供签名)。

**建议**: 从 HuggingFace 官方仓库下载模型，验证 `models--Qwen--Qwen3-ASR-0.6B/blobs/<sha256>` 与官方一致。

### 7. 信息泄露

**威胁**: server log 包含用户音频路径 / 文本, log 文件被未授权访问读到隐私。

**现状**: ⚠️ **部分缓解**:
- 默认 log 在 `/tmp/qasr_server.log` (仅 root 可读)。
- `QASR_LOG_FILE` 可改路径, 但**不**自动 chmod 600。
- realtime SSE 流**不**写 log (除错误)。

**建议**: 生产环境:
- `chmod 600 /var/log/qasr_server.log`。
- logrotate 配 `create 0600 root root`。
- 转写文本含 PII 时, 在反代层加 redaction。

### 8. C 层内存安全 (CWE-416, CWE-415)

**威胁**: C99 后端用裸 `malloc`/`free`，可能 UAF / double-free / buffer overflow。

**现状**: ✅ **积极审计**:
- C1 审计 (`docs/AUDIT_C1.md`) 全 376 导出符号 + 文件内 static 函数扫一遍。
- 150s double-free 修复 (2026-06-04, INCIDENTS.md)。
- `qwen_free()` 用 `FREE0(p) do { free(p); p = NULL; } while (0)` 宏防裸 free。
- `qwen_clone_shared` + `owns_model_data=0` 让多 session 共享权重不重复 free。
- 24 个回归测试 (`tests/qwen_clone_shared_test.cc`) 覆盖 clone/free 契约。

**未审计**: 第三方依赖 (cpp-httplib, jsoncpp, onnxruntime, openblas) — 跟随上游 release。

### 9. C++ 异常 / abort

**威胁**: C++ 异常未捕获 abort, 攻击者构造触发 abort 的输入。

**现状**: ✅ **缓解**:
- 所有 HTTP handler 用 `try/catch` 包裹, 异常 → 500 + JSON。
- realtime worker 在主循环 catch, 标记 session error, 不影响其他 session。

## 安全配置 checklist (生产部署)

```bash
# 1. HTTPS 必开
tools/run_linux_server.sh --detach --https

# 2. bind 到 LAN IP, 不暴露 0.0.0.0 给公网
export QASR_HOST=<server-lan-ip>   # 例如 192.168.x.x / 10.x.x.x
export QASR_PORT=19991
export QASR_HTTPS_PORT=19992

# 3. log 文件权限
sudo install -d -m 750 -o qasr -g qasr /var/log/qasr
export QASR_LOG_FILE=/var/log/qasr/server.log

# 4. systemd unit
[Service]
MemoryMax=8G
Restart=on-failure
RestartSec=5
User=qasr
Group=qasr
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/qasr /tmp

# 5. 反代层 (caddy)
example.com {
    reverse_proxy 127.0.0.1:19992
    rate_limit 10r/s
    basicauth {
        user $2a$14$...
    }
}
```

## 不在 scope

- 多用户 / RBAC / OAuth: 本地推理服务，信任局域网用户。
- 端到端加密音频: 不支持，依赖 TLS。
- 模型水印: Qwen3-ASR 不内置，第三方方案未集成。
- 输入音频 PII 标记: 不做, 留给上层应用。
