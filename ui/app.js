const healthBadge = document.getElementById("healthBadge");
const runtimeHint = document.getElementById("runtimeHint");
const uploadForm = document.getElementById("uploadForm");
const audioFile = document.getElementById("audioFile");
const offlineResult = document.getElementById("offlineResult");
const startRealtime = document.getElementById("startRealtime");
const stopRealtime = document.getElementById("stopRealtime");
const realtimeResult = document.getElementById("realtimeResult");
const realtimeStatus = document.getElementById("realtimeStatus");
const hostBackend = document.getElementById("hostBackend");
const hostDevice = document.getElementById("hostDevice");
const startHostCapture = document.getElementById("startHostCapture");
const stopHostCapture = document.getElementById("stopHostCapture");
const hostCaptureStatus = document.getElementById("hostCaptureStatus");
const hostCaptureResult = document.getElementById("hostCaptureResult");

let realtimeState = {
  audioContext: null,
  source: null,
  processor: null,
  mediaStream: null,
  sessionId: "",
  sendTimer: null,
  sending: false,
  pending: [],
  sampleRate: 0,
};

let hostCaptureState = {
  active: false,
  supported: true,
  pollTimer: null,
};

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderTranscript(element, data, fallback) {
  const stable = data?.stable_text || "";
  const partial = data?.partial_text || "";
  const text = data?.text || "";
  if (!stable && !partial && !text) {
    element.textContent = fallback;
    return;
  }
  if (data?.finalized) {
    element.innerHTML = `<span class=\"final\">${escapeHtml(text || stable)}</span>`;
    return;
  }
  element.innerHTML =
    `<span class=\"stable\">${escapeHtml(stable)}</span>` +
    `<span class=\"partial\">${escapeHtml(partial || text)}</span>`;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.status === "ok") {
      healthBadge.textContent = "已就绪";
      healthBadge.classList.add("ok");
      runtimeHint.textContent = "离线上传、浏览器麦克风与宿主采音接口可测";
      await refreshHostCaptureStatus();
      return;
    }
  } catch (error) {
    runtimeHint.textContent = error.message;
  }
  healthBadge.textContent = "未就绪";
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!audioFile.files[0]) {
    offlineResult.textContent = "请先选择 WAV 文件。";
    return;
  }

  offlineResult.textContent = "转写中...";
  const form = new FormData();
  form.append("audio", audioFile.files[0]);

  try {
    const response = await fetch("/api/transcriptions", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error.message);
    }
    offlineResult.textContent = `${data.text}\n\n推理 ${data.inference_ms.toFixed(0)} ms`;
  } catch (error) {
    offlineResult.textContent = `失败：${error.message}`;
  }
});

function setHostCaptureButtons(active, supported) {
  startHostCapture.disabled = !supported || active;
  stopHostCapture.disabled = !supported || !active;
}

function stopHostCapturePoll() {
  if (hostCaptureState.pollTimer !== null) {
    window.clearInterval(hostCaptureState.pollTimer);
    hostCaptureState.pollTimer = null;
  }
}

function ensureHostCapturePoll() {
  if (hostCaptureState.pollTimer !== null) {
    return;
  }
  hostCaptureState.pollTimer = window.setInterval(() => {
    refreshHostCaptureStatus().catch((error) => {
      hostCaptureStatus.textContent = `查询失败：${error.message}`;
    });
  }, 2000);
}

async function refreshHostCaptureStatus() {
  const response = await fetch("/api/capture/status");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error.message);
  }

  const supported = data.supported !== false;
  const active = Boolean(data.active);
  hostCaptureState.active = active;
  hostCaptureState.supported = supported;
  setHostCaptureButtons(active, supported);

  if (!supported) {
    stopHostCapturePoll();
    hostCaptureStatus.textContent = "当前服务无可用采音后端；请安装 ffmpeg。";
    hostCaptureResult.textContent = "尚无结果";
    return;
  }

  if (!active) {
    stopHostCapturePoll();
    hostCaptureStatus.textContent = "宿主采音未开始";
    if (data.error) {
      hostCaptureResult.textContent = data.error;
    }
    return;
  }

  ensureHostCapturePoll();
  hostCaptureStatus.textContent = `${data.backend || "auto"} 已启动，累计 ${data.sample_count || 0} 样本`;
  renderTranscript(hostCaptureResult, data, "采集中...");
  if (data.error) {
    hostCaptureStatus.textContent = `采音异常：${data.error}`;
  }
}

async function startLinuxHostCapture() {
  hostCaptureStatus.textContent = "宿主采音启动中...";
  const response = await fetch("/api/capture/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      backend: hostBackend.value,
      device: hostDevice.value.trim(),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error.message);
  }
  renderTranscript(hostCaptureResult, null, "采集中...");
  hostCaptureStatus.textContent = `${data.backend} 已启动`;
  ensureHostCapturePoll();
  await refreshHostCaptureStatus();
}

async function stopLinuxHostCapture() {
  stopHostCapturePoll();
  const response = await fetch("/api/capture/stop", {
    method: "POST",
    body: "",
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error.message);
  }
  hostCaptureState.active = false;
  setHostCaptureButtons(false, true);
  hostCaptureStatus.textContent = `${data.backend || "capture"} 已停止`;
  if (data.error && !(data.text || data.stable_text || data.partial_text)) {
    hostCaptureResult.textContent = data.error;
    return;
  }
  renderTranscript(hostCaptureResult, data, "尚无结果");
}

function downsampleTo16k(input, inputRate) {
  if (inputRate === 16000) {
    return input;
  }
  const ratio = inputRate / 16000;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(input.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let i = start; i < end; i += 1) {
      sum += input[i];
      count += 1;
    }
    output[index] = count > 0 ? sum / count : 0;
  }
  return output;
}

function floatToPcm16(input) {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return output;
}

async function flushRealtimeChunk(force) {
  if (!realtimeState.sessionId || realtimeState.sending) {
    return;
  }
  if (!force && realtimeState.pending.length === 0) {
    return;
  }
  const buffer = new Int16Array(realtimeState.pending);
  realtimeState.pending = [];
  if (buffer.length === 0) {
    return;
  }

  realtimeState.sending = true;
  try {
    const response = await fetch(`/api/realtime/chunk?session_id=${encodeURIComponent(realtimeState.sessionId)}`, {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream"},
      body: buffer.buffer,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error.message);
    }
    renderTranscript(realtimeResult, data, "尚无结果");
    const decodeLabel = data.decoded ? "已解码" : "待下轮";
    realtimeStatus.textContent = `会话 ${realtimeState.sessionId}，累计 ${data.sample_count} 样本，${decodeLabel}，稳 ${data.stable_text.length} / 尾 ${data.partial_text.length}`;
  } catch (error) {
    realtimeStatus.textContent = `失败：${error.message}`;
  } finally {
    realtimeState.sending = false;
  }
}

async function startRealtimeCapture() {
  const sessionResponse = await fetch("/api/realtime/start", {method: "POST", body: ""});
  const sessionData = await sessionResponse.json();
  if (!sessionResponse.ok) {
    throw new Error(sessionData.error.message);
  }

  const mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(mediaStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  realtimeState = {
    audioContext,
    source,
    processor,
    mediaStream,
    sessionId: sessionData.session_id,
    sendTimer: window.setInterval(() => flushRealtimeChunk(false), 800),
    sending: false,
    pending: [],
    sampleRate: audioContext.sampleRate,
  };

  processor.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0);
    const downsampled = downsampleTo16k(channel, realtimeState.sampleRate);
    const pcm = floatToPcm16(downsampled);
    realtimeState.pending.push(...pcm);
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
  startRealtime.disabled = true;
  stopRealtime.disabled = false;
  renderTranscript(realtimeResult, null, "实时转写中...");
  realtimeStatus.textContent = `会话 ${realtimeState.sessionId} 已启动`;
}

async function stopRealtimeCapture() {
  if (!realtimeState.sessionId) {
    return;
  }
  window.clearInterval(realtimeState.sendTimer);
  await flushRealtimeChunk(true);

  realtimeState.processor.disconnect();
  realtimeState.source.disconnect();
  realtimeState.mediaStream.getTracks().forEach((track) => track.stop());
  await realtimeState.audioContext.close();

  const response = await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(realtimeState.sessionId)}`, {
    method: "POST",
    body: "",
  });
  const data = await response.json();
  if (response.ok && data.text) {
    renderTranscript(realtimeResult, data, "尚无结果");
    realtimeStatus.textContent = `会话 ${realtimeState.sessionId} 已停止，终稿已出`;
  } else {
    realtimeStatus.textContent = data.error ? data.error.message : "停止失败";
  }

  realtimeState = {
    audioContext: null,
    source: null,
    processor: null,
    mediaStream: null,
    sessionId: "",
    sendTimer: null,
    sending: false,
    pending: [],
    sampleRate: 0,
  };
  startRealtime.disabled = false;
  stopRealtime.disabled = true;
}

startRealtime.addEventListener("click", async () => {
  try {
    await startRealtimeCapture();
  } catch (error) {
    realtimeStatus.textContent = `启动失败：${error.message}`;
  }
});

stopRealtime.addEventListener("click", async () => {
  try {
    await stopRealtimeCapture();
  } catch (error) {
    realtimeStatus.textContent = `停止失败：${error.message}`;
  }
});

startHostCapture.addEventListener("click", async () => {
  try {
    await startLinuxHostCapture();
  } catch (error) {
    hostCaptureStatus.textContent = `启动失败：${error.message}`;
  }
});

stopHostCapture.addEventListener("click", async () => {
  try {
    await stopLinuxHostCapture();
  } catch (error) {
    hostCaptureStatus.textContent = `停止失败：${error.message}`;
  }
});

checkHealth();
