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

当前 UI 两路：

- 离线：浏览器端读取 WAV，并优先转成 `16k mono PCM16` 小块推送
- 实时：浏览器麦克风分块 PCM16 推送

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

## 三、离线上传策略

当前离线面板不再默认走整包 multipart 上传，而改为：

1. 浏览器先解析 WAV 头
2. 若为 `PCM + 16-bit`，则按块读取音频帧
3. 浏览器端先下混为 mono，若采样率非 `16k` 则同步重采样到 `16k`
4. 结果以 `PCM16` 小块推送到 `/api/realtime/start|chunk|stop`
5. 前端优先消费 realtime 返回的 `recent_segments + live_stable_text + live_partial_text`

此法的收益：

- 避免单个大 multipart 请求触发 `64MiB` 上传上限
- 避免服务端先整包入内存、再复制 multipart part、再落 tmp 文件
- 对立体声 `16k` WAV，可在浏览器侧先降到 mono，网络流量约减半
- 上传过程中即可增量显示转写结果

当前回退策略：

- 若浏览器端无法解析或转换该 WAV，且文件不超过 `64MiB`，则回退到 `/api/transcriptions/async`
- 若文件已超过 `64MiB` 且又不满足前端分块条件，则前端直接报错，提示先转为 `16-bit PCM WAV`

## 四、现阶段约束

- 实时 UI 现走“浏览器麦克风 -> PCM16 分块 -> 服务端累积流式转写”
- 离线 UI 现走“浏览器 WAV -> mono/resample -> PCM16 分块 -> 服务端 realtime 会话”
- `/api/realtime/chunk` 兼回旧字段 `stable_text`、`partial_text`、`text`
- `/api/realtime/chunk` 主显示字段改为：`recent_segments`、`live_stable_text`、`live_partial_text`、`live_text`、`display_text`
- `/api/realtime/stop` 已做一次终态 flush，回 `finalized=true`
- UI 实时主视图只示近段与活尾；另设“已确定文本”区保留 `stable_text`/终稿
- UI 已可把“已确定文本”导出为 `TXT` 或 `JSON`
- 本地与 Docker 皆可测
- 浏览器端分块目前仅覆盖 `PCM 16-bit WAV`
- 若要支持压缩 WAV / float WAV / 其它音频容器，仍需浏览器端额外转码或服务端 chunk upload 会话

## 五、后续 UI 计划

- UI 视觉层已显式区分“近段 / 稳定尾 / 不稳定尾 / 终稿”
- 实时状态下不再以全文单串作主视图；只更新近段与尾巴
- 已确定文本单独留存于页面，可在 stop 前后导出
- 若后续补服务端 upload session，可把“非 PCM16 WAV”也改为分块上传，彻底去掉大 multipart 回退
