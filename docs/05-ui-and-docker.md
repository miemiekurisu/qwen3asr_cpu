# UI 与 Docker

日期：2026-04-10

## 一、本地启动

```bash
cmake -S . -B build/macos-accelerate-cli -G "Unix Makefiles" -DQASR_ENABLE_TESTS=ON
cmake --build build/macos-accelerate-cli -j4
./build/macos-accelerate-cli/qasr_server \
  --model-dir /Users/kurisu/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1___7B \
  --ui-dir ui \
  --host 127.0.0.1 \
  --port 3458 \
  --threads 4
```

浏览器开：

- `http://127.0.0.1:3458/`
- `http://127.0.0.1:3458/api/metrics`

当前 UI 三路：

- 离线：WAV 上传
- 实时：浏览器麦克风分块 PCM16 推送
- Host Audio：Linux / Docker 宿主设备直采

## 二、Docker Linux 启动

先构镜像：

```bash
docker build -t qasr-ui:latest .
```

再跑：

```bash
./tools/docker_run_ui_linux.sh /path/to/Qwen3-ASR-1___7B qasr-ui:latest
```

其义：

- 映射 `8080`
- 挂模型目录到 `/models/qwen3-asr`
- 若宿主有 `/dev/snd`，则透传之
- 若宿主启 PulseAudio socket，亦透传之

浏览器开：

- `http://127.0.0.1:8080/`
- `http://127.0.0.1:8080/api/metrics`

## 三、现阶段约束

- 实时 UI 现走“浏览器麦克风 -> PCM16 分块 -> 服务端累积流式转写”
- `/api/realtime/chunk` 已回 `stable_text`、`partial_text`、`text`
- `/api/realtime/stop` 已做一次终态 flush，回 `finalized=true`
- 此路本地与 Docker 皆可测
- Linux 容器音频设备映射已备好启动脚本
- Linux 宿主设备服务端采集 backend 已落 `arecord` / `parec` 骨架
- macOS 下 `Host Audio` 面仅示 unsupported
- 容器内“真接宿主声卡并稳定取流”仍须实体设备回归

## 四、后续 UI 计划

- UI 视觉层再显式区分 `partial` / `stable` / `final`
- 稳定前缀不重绘；仅更新尾巴
- Host Audio 与浏览器麦克风共用同一会话语义
