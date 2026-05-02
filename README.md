# qwen3asr_cpu

Qwen3-ASR 的 CPU 推理服务与命令行工具，使用 C/C++17 实现。项目提供本地离线转写、字幕输出、HTTP API、内置 Web UI，以及面向 Windows / Linux / macOS 的 CPU 构建路径。

支持 Qwen3-ASR 0.6B / 1.7B safetensors 模型；Windows 和 Linux 使用 OpenBLAS，macOS 使用 Accelerate。可选 oneDNN INT8 路径用于 encoder / decoder 加速。

## 快速开始

### Windows

推荐使用根目录的 `build.bat`：

```bat
build.bat
```

默认行为是 clean + configure + compile。按需追加：

```bat
build.bat --test
build.bat --benchmark
build.bat --test --benchmark
build.bat --openblas-dir D:\dev\OpenBLAS
```

运行时需要让 OpenBLAS DLL 可见：

```powershell
$env:PATH = "D:\dev\OpenBLAS\bin;$env:PATH"
```

### Linux

```bash
sudo apt-get install build-essential cmake ninja-build libopenblas-dev ffmpeg
cmake --preset linux-openblas
cmake --build build/linux-openblas -j"$(nproc)"
ctest --test-dir build/linux-openblas --output-on-failure
```

### macOS

```bash
cmake --preset macos-accelerate
cmake --build build/macos-accelerate -j"$(sysctl -n hw.ncpu)"
ctest --test-dir build/macos-accelerate --output-on-failure
```

## 模型

下载 Qwen3-ASR 模型目录后直接传给 `--model-dir`：

| Model | HuggingFace | ModelScope |
|---|---|---|
| Qwen3-ASR-0.6B | <https://huggingface.co/Qwen/Qwen3-ASR-0.6B> | <https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B> |
| Qwen3-ASR-1.7B | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> | <https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B> |
| Qwen3-ForcedAligner-0.6B | <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | <https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B> |

CLI、server 和 API 都通过 `--model-dir` / `model` 指定 ASR 模型，不强制绑定 0.6B 或 1.7B。按 CPU 使用体验建议：

- `Qwen3-ASR-0.6B`：优先用于实时、近实时、Web UI 和低延迟服务。
- `Qwen3-ASR-1.7B`：优先用于离线批处理、长音频转写和字幕生产。
- `Qwen3-ForcedAligner-0.6B`：用于时间轴对齐。它不是替代 ASR 的转写模型，而是在 ASR 得到文本后，把文本和音频对齐成更细的字幕时间。

在 CPU 上不建议把 1.7B 当严格实时模型使用；入口上可以运行，但延迟通常会很高。

## CLI 示例

基本转写：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav
```

MP3 / M4A / FLAC 等非 WAV 输入会自动通过 `ffmpeg` 转成 16 kHz mono WAV：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio meeting.mp3
```

指定语言、线程数和提示词：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav \
  --language Chinese \
  --prompt "会议记录，包含技术术语" \
  --threads 8
```

输出 SRT 字幕：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 \
  --output-format srt \
  --output movie.srt
```

使用 ForcedAligner 生成更细的字幕时间轴：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio movie.mp3 \
  --output-format srt \
  --align \
  --aligner-model-dir /path/to/Qwen3-ForcedAligner-0.6B \
  --output movie.srt
```

输出 WebVTT：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio lecture.wav \
  --output-format vtt \
  --output lecture.vtt
```

输出 JSON 段落：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-1.7B --audio interview.wav \
  --output-format json \
  --output interview.json
```

启用可选 INT8 路径：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio audio.wav \
  --decoder-int8 \
  --encoder-int8 \
  --threads 16
```

> ⚠️ **`--decoder-int8` 会显著降低识别质量**：语言一致性下降、中英文混杂泄漏、低置信度音频上更易产生幻觉。仅在内存严重受限时启用；批量转写优先只开 `--encoder-int8`。实时（`/api/realtime`、`/v1/realtime`）会话默认 **不会** 沿用 `--decoder-int8`，需显式 `--realtime-decoder-int8` 才会启用。

流式分段推理：

```bash
qasr_cli --model-dir /path/to/Qwen3-ASR-0.6B --audio long.wav \
  --stream \
  --emit-segments \
  --stream-max-new-tokens 32
```

查看完整参数：

```bash
qasr_cli --help
```

## Server 示例

启动 Web UI 和 HTTP API：

```bash
qasr_server --model-dir /path/to/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port 8080 \
  --ui-dir ui \
  --threads 8
```

打开：

```text
http://127.0.0.1:8080/
```

启用 INT8：

```bash
qasr_server --model-dir /path/to/Qwen3-ASR-0.6B \
  --port 8080 \
  --decoder-int8 \
  --encoder-int8 \
  --threads 16
```

> ⚠️ 同上：`--decoder-int8` 主要节省内存，但识别质量明显下降，且不会自动应用到实时会话。如确需实时也使用 INT8 解码器，请同时加 `--realtime-decoder-int8`。

查看服务帮助：

```bash
qasr_server --help
```

## HTTP API 示例

OpenAI-style transcription：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

返回纯文本：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=text
```

返回 verbose JSON 和 segment 时间戳：

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```

Chat-style audio transcription：

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-asr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "audio_url", "audio_url": {"url": "file:///absolute/path/audio.wav"}}
      ]
    }]
  }'
```

私有异步上传接口：

```bash
curl -X POST http://localhost:8080/api/transcriptions/async \
  -F audio=@audio.wav
```

查询 job：

```bash
curl http://localhost:8080/api/jobs/<job-id>
```

## Docker

```bash
docker build -t qasr .
docker run --rm -p 8080:8080 \
  -v /path/to/Qwen3-ASR-0.6B:/models/qwen3-asr \
  qasr
```

## 常用环境变量

```bash
OPENBLAS_NUM_THREADS=8
QWEN_RUNTIME_PROFILE=balanced
QWEN_DEC_PREFILL_QKV_PERSIST=1
QWEN_DEC_PREFILL_QKV_BUDGET_MB=512
QWEN_ENC_QKV_POLICY=best
```

## 项目结构

```text
app/                    CLI, server, benchmark entry points
include/qasr/           Public C++ headers
src/backend/qwen_cpu/   Internal C CPU backend and kernels
src/service/            HTTP server and realtime session handling
src/runtime/            Model bridge, tasks, sessions, queues
src/protocol/           OpenAI/vLLM request validation
src/audio/              WAV parsing, resampling, ffmpeg conversion helpers
src/subtitle/           SRT/VTT/JSON subtitle writers
tests/                  Unit and regression tests
ui/                     Browser UI
tools/                  Build, benchmark, Docker helper scripts
docs/                   Design notes and internal references
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
