const healthBadge = document.getElementById("healthBadge");
const runtimeHint = document.getElementById("runtimeHint");
const uploadForm = document.getElementById("uploadForm");
const audioFile = document.getElementById("audioFile");
const offlineSubmit = document.getElementById("offlineSubmit");
const offlineStop = document.getElementById("offlineStop");
const offlineStatus = document.getElementById("offlineStatus");
const offlineResult = document.getElementById("offlineResult");
const startRealtime = document.getElementById("startRealtime");
const stopRealtime = document.getElementById("stopRealtime");
const clearRealtime = document.getElementById("clearRealtime");
const realtimeResult = document.getElementById("realtimeResult");
const realtimeStatus = document.getElementById("realtimeStatus");


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
  startedAt: 0,
};

let offlineState = {
  jobId: "",
  stopRequested: false,
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

function hasOfflineJob() {
  return offlineState.jobId !== "";
}

function hasRealtimeSession() {
  return realtimeState.sessionId !== "";
}

function updateControlAvailability() {
  const offlineActive = hasOfflineJob();
  const realtimeActive = hasRealtimeSession();
  audioFile.disabled = offlineActive || realtimeActive;
  offlineSubmit.disabled = offlineActive || realtimeActive;
  offlineStop.disabled = !offlineActive || offlineState.stopRequested || realtimeActive;
  startRealtime.disabled = offlineActive || realtimeActive;
  stopRealtime.disabled = !realtimeActive;
}

function resetOfflineState() {
  offlineState = {
    jobId: "",
    stopRequested: false,
  };
  updateControlAvailability();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.status === "ok") {
      healthBadge.textContent = "已就绪";
      healthBadge.classList.add("ok");
      runtimeHint.textContent = "离线上传与浏览器麦克风实时转写可用";
      return;
    }
  } catch (error) {
    runtimeHint.textContent = error.message;
  }
  healthBadge.textContent = "未就绪";
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (hasRealtimeSession()) {
    offlineStatus.textContent = "实时转写进行中，请先停止。";
    return;
  }
  if (!audioFile.files[0]) {
    offlineResult.textContent = "请先选择 WAV 文件。";
    return;
  }

  offlineResult.textContent = "";
  offlineStatus.textContent = "提交中...";
  const startTime = performance.now();

  try {
    const form = new FormData();
    form.append("audio", audioFile.files[0]);
    const submitRes = await fetch("/api/transcriptions/async", {
      method: "POST",
      body: form,
    });
    const submitData = await submitRes.json();
    if (!submitRes.ok) {
      throw new Error(submitData.error?.message || "提交失败");
    }
    const jobId = submitData.id;
    offlineState.jobId = jobId;
    offlineState.stopRequested = false;
    updateControlAvailability();
    offlineStatus.textContent = "转写中...";

    let lastTextLen = 0;
    while (true) {
      await new Promise((r) => setTimeout(r, 300));
      const pollRes = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      const job = await pollRes.json();
      if (!pollRes.ok) {
        throw new Error(job.error?.message || "查询失败");
      }

      if (job.state === "running" || job.state === "queued" || job.state === "cancelling") {
        const text = job.text || "";
        if (text.length > lastTextLen) {
          lastTextLen = text.length;
          offlineResult.innerHTML =
            `<span class="stable">${escapeHtml(text)}</span>`;
        }
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        const chars = [...text].length;
        const prefix = job.state === "cancelling" ? "停止中..." : "转写中...";
        offlineStatus.textContent = chars > 0
          ? `${prefix} 已识别 ${chars} 字 / ${elapsed}s`
          : `${prefix} ${elapsed}s`;
        continue;
      }

      if (job.state === "cancelled") {
        if (job.text) {
          offlineResult.innerHTML =
            `<span class="stable">${escapeHtml(job.text)}</span>`;
        } else {
          offlineResult.textContent = "已停止";
        }
        offlineStatus.textContent = "已停止";
        break;
      }

      if (job.state === "failed") {
        throw new Error(job.error || "转写失败");
      }

      offlineResult.innerHTML =
        `<span class="final">${escapeHtml(job.text)}</span>`;
      const audioDur = (job.audio_ms / 1000).toFixed(1);
      const infMs = job.inference_ms.toFixed(0);
      const rtf = (job.inference_ms / job.audio_ms).toFixed(2);
      offlineStatus.textContent =
        `音频 ${audioDur}s / 推理 ${infMs}ms / RTF ${rtf} / ${job.tokens} tokens`;
      break;
    }
  } catch (error) {
    offlineResult.textContent = `失败：${error.message}`;
    offlineStatus.textContent = "";
  } finally {
    resetOfflineState();
  }
});

offlineStop.addEventListener("click", async () => {
  if (!hasOfflineJob()) {
    return;
  }

  offlineState.stopRequested = true;
  updateControlAvailability();
  offlineStatus.textContent = "停止中...";

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(offlineState.jobId)}/cancel`, {
      method: "POST",
      body: "",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error?.message || "停止失败");
    }
  } catch (error) {
    offlineState.stopRequested = false;
    updateControlAvailability();
    offlineStatus.textContent = `停止失败：${error.message}`;
  }
});



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
    const audioDur = (data.sample_count / 16000).toFixed(1);
    const wallElapsed = ((performance.now() - realtimeState.startedAt) / 1000).toFixed(1);
    const lag = (wallElapsed - audioDur).toFixed(1);
    const infMs = data.inference_ms !== undefined ? data.inference_ms.toFixed(0) : "-";
    const decodeLabel = data.decoded ? "已解码" : "待下轮";
    realtimeStatus.textContent = `音频 ${audioDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel}`;
  } catch (error) {
    realtimeStatus.textContent = `失败：${error.message}`;
  } finally {
    realtimeState.sending = false;
  }
}

async function startRealtimeCapture() {
  if (hasOfflineJob()) {
    throw new Error("离线转写进行中，请先停止");
  }

  const mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
  const sessionResponse = await fetch("/api/realtime/start", {method: "POST", body: ""});
  const sessionData = await sessionResponse.json();
  if (!sessionResponse.ok) {
    mediaStream.getTracks().forEach((track) => track.stop());
    throw new Error(sessionData.error.message);
  }

  let audioContext = null;
  try {
    audioContext = new AudioContext();
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
      startedAt: performance.now(),
    };

    processor.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(channel, realtimeState.sampleRate);
      const pcm = floatToPcm16(downsampled);
      realtimeState.pending.push(...pcm);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    updateControlAvailability();
    clearRealtime.style.display = "none";
    renderTranscript(realtimeResult, null, "实时转写中...");
    realtimeStatus.textContent = `会话 ${realtimeState.sessionId} 已启动`;
  } catch (error) {
    mediaStream.getTracks().forEach((track) => track.stop());
    if (audioContext) {
      await audioContext.close();
    }
    try {
      await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionData.session_id)}`, {
        method: "POST",
        body: "",
      });
    } catch (_cleanupError) {
      // Best effort only; the original startup error is more important to surface.
    }
    throw error;
  }
}

async function stopRealtimeCapture() {
  if (!realtimeState.sessionId) {
    return;
  }
  const sessionId = realtimeState.sessionId;
  window.clearInterval(realtimeState.sendTimer);
  await flushRealtimeChunk(true);

  realtimeState.processor.disconnect();
  realtimeState.source.disconnect();
  realtimeState.mediaStream.getTracks().forEach((track) => track.stop());
  await realtimeState.audioContext.close();

  try {
    const response = await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: "",
    });
    const data = await response.json();
    if (response.ok && data.text) {
      renderTranscript(realtimeResult, data, "尚无结果");
      realtimeStatus.textContent = `会话 ${sessionId} 已停止，终稿已出`;
    } else {
      realtimeStatus.textContent = data.error ? data.error.message : "停止失败";
    }
  } finally {
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
      startedAt: 0,
    };
    updateControlAvailability();
    clearRealtime.style.display = "";
  }
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

clearRealtime.addEventListener("click", () => {
  realtimeResult.textContent = "尚无结果";
  realtimeStatus.textContent = "未开始";
  clearRealtime.style.display = "none";
});

updateControlAvailability();
checkHealth();
