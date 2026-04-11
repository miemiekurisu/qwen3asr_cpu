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
const exportRealtimeText = document.getElementById("exportRealtimeText");
const exportRealtimeJson = document.getElementById("exportRealtimeJson");
const realtimeResult = document.getElementById("realtimeResult");
const realtimeConfirmed = document.getElementById("realtimeConfirmed");
const realtimeArchiveHint = document.getElementById("realtimeArchiveHint");
const realtimeStatus = document.getElementById("realtimeStatus");

const wavUpload = globalThis.QasrWavUpload;
const MAX_ASYNC_UPLOAD_BYTES = 64 * 1024 * 1024;

let realtimeState = {
  audioContext: null,
  source: null,
  processor: null,
  mediaStream: null,
  sessionId: "",
  sendTimer: null,
  pollTimer: null,
  sending: false,
  pending: [],
  sampleRate: 0,
  startedAt: 0,
};

let realtimeArchive = {
  sessionId: "",
  confirmedText: "",
  lastPayload: null,
  finalized: false,
  updatedAt: "",
};

let offlineState = {
  mode: "",
  jobId: "",
  sessionId: "",
  stopRequested: false,
  startedAt: 0,
  sourceSampleRate: 0,
  totalSourceFrames: 0,
  uploadedSourceFrames: 0,
};

function ensureTranscriptFrame(element) {
  if (element._transcriptFrame) {
    return element._transcriptFrame;
  }

  element.textContent = "";
  const finalLine = document.createElement("span");
  finalLine.className = "transcript-block final";

  const historyLine = document.createElement("span");
  historyLine.className = "transcript-block history";

  const liveLine = document.createElement("span");
  liveLine.className = "transcript-block live";
  const stableLine = document.createElement("span");
  stableLine.className = "stable";
  const partialLine = document.createElement("span");
  partialLine.className = "partial";
  liveLine.append(stableLine, partialLine);

  element.append(finalLine, historyLine, liveLine);
  element._transcriptFrame = {
    finalLine,
    historyLine,
    liveLine,
    stableLine,
    partialLine,
  };
  return element._transcriptFrame;
}

function resetTranscriptFrame(element, fallback) {
  element._transcriptFrame = null;
  element.textContent = fallback;
}

function renderTranscript(element, data, fallback) {
  const recentSegments = Array.isArray(data?.recent_segments)
    ? data.recent_segments.filter((segment) => typeof segment === "string" && segment)
    : [];
  const liveStable = data?.live_stable_text || "";
  const livePartial = data?.live_partial_text || "";
  const stable = data?.stable_text || "";
  const partial = data?.partial_text || "";
  const text = data?.text || "";
  const hasSegmentView = recentSegments.length > 0 || liveStable || livePartial;
  if (!hasSegmentView && !stable && !partial && !text) {
    resetTranscriptFrame(element, fallback);
    return;
  }

  const frame = ensureTranscriptFrame(element);
  if (data?.finalized) {
    frame.finalLine.textContent = text || stable || recentSegments.join("\n");
    frame.finalLine.style.display = "block";
    frame.historyLine.textContent = "";
    frame.historyLine.style.display = "none";
    frame.stableLine.textContent = "";
    frame.partialLine.textContent = "";
    frame.liveLine.style.display = "none";
    return;
  }

  const fallbackStable = hasSegmentView ? "" : stable;
  const fallbackPartial = hasSegmentView ? "" : (partial || text);
  frame.finalLine.textContent = "";
  frame.finalLine.style.display = "none";
  frame.historyLine.textContent = recentSegments.join("\n");
  frame.historyLine.style.display = recentSegments.length > 0 ? "block" : "none";
  frame.stableLine.textContent = liveStable || fallbackStable;
  frame.partialLine.textContent = livePartial || fallbackPartial;
  frame.liveLine.style.display = (frame.stableLine.textContent || frame.partialLine.textContent) ? "block" : "none";
}

function hasOfflineJob() {
  return offlineState.mode !== "";
}

function hasRealtimeSession() {
  return realtimeState.sessionId !== "";
}

function updateControlAvailability() {
  const offlineActive = hasOfflineJob();
  const realtimeActive = hasRealtimeSession();
  const canStopOffline =
    (offlineState.mode === "async" && offlineState.jobId !== "") ||
    (offlineState.mode === "stream" && offlineState.sessionId !== "");
  audioFile.disabled = offlineActive || realtimeActive;
  offlineSubmit.disabled = offlineActive || realtimeActive;
  offlineStop.disabled = !canStopOffline || offlineState.stopRequested || realtimeActive;
  startRealtime.disabled = offlineActive || realtimeActive;
  stopRealtime.disabled = !realtimeActive;
}

function resetOfflineState() {
  offlineState = {
    mode: "",
    jobId: "",
    sessionId: "",
    stopRequested: false,
    startedAt: 0,
    sourceSampleRate: 0,
    totalSourceFrames: 0,
    uploadedSourceFrames: 0,
  };
  updateControlAvailability();
}

function offlineElapsedSeconds() {
  if (!offlineState.startedAt) {
    return 0;
  }
  return (performance.now() - offlineState.startedAt) / 1000;
}

function formatSeconds(value) {
  return value.toFixed(1);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function countCodepoints(text) {
  return Array.from(text || "").length;
}

function extractConfirmedRealtimeText(data) {
  if (!data || typeof data !== "object") {
    return "";
  }
  if (data.finalized && typeof data.text === "string") {
    return data.text;
  }
  if (typeof data.stable_text === "string") {
    return data.stable_text;
  }
  return "";
}

function updateRealtimeExportAvailability() {
  const hasConfirmedText = Boolean(realtimeArchive.confirmedText.trim());
  exportRealtimeText.disabled = !hasConfirmedText;
  exportRealtimeJson.disabled = !hasConfirmedText;
}

function renderRealtimeArchive(fallback) {
  const confirmedText = realtimeArchive.confirmedText;
  if (!confirmedText) {
    realtimeConfirmed.textContent = fallback;
    realtimeArchiveHint.textContent = realtimeState.sessionId
      ? "已确定文本会随着稳定前缀推进保存在此处。"
      : "已确定文本会保存在此处，停止后可导出。";
    updateRealtimeExportAvailability();
    return;
  }

  realtimeConfirmed.textContent = confirmedText;
  const chars = countCodepoints(confirmedText);
  realtimeArchiveHint.textContent = realtimeArchive.finalized
    ? `已保留终稿 ${chars} 字，可导出 TXT 或 JSON。`
    : `已保留已确定文本 ${chars} 字，实时主视图仍只显示近段与活尾。`;
  updateRealtimeExportAvailability();
}

function resetRealtimeArchive(fallback = "尚无已确定文本") {
  realtimeArchive = {
    sessionId: "",
    confirmedText: "",
    lastPayload: null,
    finalized: false,
    updatedAt: "",
  };
  renderRealtimeArchive(fallback);
}

function syncRealtimeArchive(data) {
  realtimeArchive.sessionId = data?.session_id || realtimeState.sessionId || realtimeArchive.sessionId;
  realtimeArchive.confirmedText = extractConfirmedRealtimeText(data);
  realtimeArchive.lastPayload = data || null;
  realtimeArchive.finalized = Boolean(data?.finalized);
  realtimeArchive.updatedAt = new Date().toISOString();
  renderRealtimeArchive("尚无已确定文本");
}

function buildRealtimeExportName(ext) {
  const sessionId = (realtimeArchive.sessionId || "session").replace(/[^a-zA-Z0-9_-]+/g, "-");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `qasr-realtime-${sessionId}-${stamp}.${ext}`;
}

function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], {type: mimeType});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportRealtimeTranscript(format) {
  if (!realtimeArchive.confirmedText.trim()) {
    realtimeStatus.textContent = "暂无可导出的已确定文本";
    return;
  }

  if (format === "txt") {
    triggerDownload(
      buildRealtimeExportName("txt"),
      realtimeArchive.confirmedText,
      "text/plain;charset=utf-8",
    );
    realtimeStatus.textContent = "已导出 TXT";
    return;
  }

  const payload = {
    exported_at: new Date().toISOString(),
    session_id: realtimeArchive.sessionId,
    finalized: realtimeArchive.finalized,
    confirmed_text: realtimeArchive.confirmedText,
    latest_response: realtimeArchive.lastPayload,
  };
  triggerDownload(
    buildRealtimeExportName("json"),
    `${JSON.stringify(payload, null, 2)}\n`,
    "application/json;charset=utf-8",
  );
  realtimeStatus.textContent = "已导出 JSON";
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.status === "ok") {
      healthBadge.textContent = "已就绪";
      healthBadge.classList.add("ok");
      runtimeHint.textContent = "离线 WAV 分块上传与浏览器麦克风实时转写可用";
      return;
    }
  } catch (error) {
    runtimeHint.textContent = error.message;
  }
  healthBadge.textContent = "未就绪";
}

async function inspectOfflineUploadFile(file) {
  if (!wavUpload || typeof wavUpload.parseWavHeader !== "function") {
    return {
      supported: false,
      reason: "浏览器端 WAV 分块模块未加载",
    };
  }

  const probeBytes = await file.slice(0, Math.min(file.size, 1024 * 1024)).arrayBuffer();
  try {
    const format = wavUpload.parseWavHeader(probeBytes);
    if (format.frameCount <= 0) {
      throw new Error("WAV data 为空");
    }
    if (format.dataOffset + format.dataSize > file.size) {
      throw new Error("WAV data chunk 超出文件大小");
    }
    return {supported: true, format};
  } catch (error) {
    return {
      supported: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

function updateOfflineAsyncStatus(job, startTime) {
  const text = job.text || "";
  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  const chars = [...text].length;
  const prefix = job.state === "cancelling" ? "停止中..." : "转写中...";
  offlineStatus.textContent = chars > 0
    ? `${prefix} 已识别 ${chars} 字 / ${elapsed}s`
    : `${prefix} ${elapsed}s`;
}

async function submitOfflineViaAsync(file, startTime) {
  const form = new FormData();
  form.append("audio", file);
  const submitRes = await fetch("/api/transcriptions/async", {
    method: "POST",
    body: form,
  });
  const submitData = await submitRes.json();
  if (!submitRes.ok) {
    throw new Error(submitData.error?.message || "提交失败");
  }

  offlineState.mode = "async";
  offlineState.jobId = submitData.id;
  offlineState.stopRequested = false;
  updateControlAvailability();
  offlineStatus.textContent = "转写中...";

  let lastTextLen = 0;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const pollRes = await fetch(`/api/jobs/${encodeURIComponent(submitData.id)}`);
    const job = await pollRes.json();
    if (!pollRes.ok) {
      throw new Error(job.error?.message || "查询失败");
    }

    if (job.state === "running" || job.state === "queued" || job.state === "cancelling") {
      const text = job.text || "";
      if (text.length > lastTextLen) {
        lastTextLen = text.length;
        offlineResult.innerHTML = `<span class=\"stable\">${escapeHtml(text)}</span>`;
      }
      updateOfflineAsyncStatus(job, startTime);
      continue;
    }

    if (job.state === "cancelled") {
      if (job.text) {
        offlineResult.innerHTML = `<span class=\"stable\">${escapeHtml(job.text)}</span>`;
      } else {
        offlineResult.textContent = "已停止";
      }
      offlineStatus.textContent = "已停止";
      break;
    }

    if (job.state === "failed") {
      throw new Error(job.error || "转写失败");
    }

    offlineResult.innerHTML = `<span class=\"final\">${escapeHtml(job.text)}</span>`;
    const audioDur = (job.audio_ms / 1000).toFixed(1);
    const infMs = job.inference_ms.toFixed(0);
    const rtf = (job.inference_ms / job.audio_ms).toFixed(2);
    offlineStatus.textContent =
      `音频 ${audioDur}s / 推理 ${infMs}ms / RTF ${rtf} / ${job.tokens} tokens`;
    break;
  }
}

async function startOfflineStreamSession() {
  const response = await fetch("/api/realtime/start", {
    method: "POST",
    body: "",
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || "无法创建离线流式会话");
  }
  return data;
}

async function sendOfflineStreamChunk(sessionId, pcmChunk) {
  const response = await fetch(`/api/realtime/chunk?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    headers: {"Content-Type": "application/octet-stream"},
    body: pcmChunk.buffer,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || "离线分块上传失败");
  }
  return data;
}

function updateOfflineStreamStatus(data, format, finalized) {
  const uploadedSeconds = formatSeconds(offlineState.uploadedSourceFrames / format.sampleRate);
  const totalSeconds = formatSeconds(format.durationSeconds);
  const transcribedSeconds = formatSeconds(data.sample_count / 16000);
  const elapsed = formatSeconds(offlineElapsedSeconds());
  const infMs = data.inference_ms !== undefined ? data.inference_ms.toFixed(0) : "-";
  if (finalized) {
    const prefix = offlineState.stopRequested ? "已停止" : "已完成";
    offlineStatus.textContent =
      `${prefix}：已上传 ${uploadedSeconds}/${totalSeconds}s / 已转写 ${transcribedSeconds}s / 推理 ${infMs}ms / 耗时 ${elapsed}s`;
    return;
  }
  const decodeLabel = data.decoded ? "已解码" : "待下轮";
  const prefix = offlineState.stopRequested ? "停止中..." : "上传转写中...";
  offlineStatus.textContent =
    `${prefix} 已上传 ${uploadedSeconds}/${totalSeconds}s / 已转写 ${transcribedSeconds}s / 推理 ${infMs}ms / ${decodeLabel}`;
}

async function finalizeOfflineStreamSession(format) {
  if (!offlineState.sessionId) {
    return null;
  }
  const sessionId = offlineState.sessionId;
  offlineState.sessionId = "";
  updateControlAvailability();
  const response = await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    body: "",
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || "离线流式会话停止失败");
  }
  renderTranscript(offlineResult, data, "尚无结果");
  updateOfflineStreamStatus(data, format, true);
  return data;
}

async function releaseOfflineStreamSession() {
  if (!offlineState.sessionId) {
    return;
  }
  const sessionId = offlineState.sessionId;
  offlineState.sessionId = "";
  updateControlAvailability();
  try {
    await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: "",
    });
  } catch (_error) {
    // Best effort to release server-side exclusivity.
  }
}

async function submitOfflineViaStream(file, startTime, format) {
  const session = await startOfflineStreamSession();
  offlineState.mode = "stream";
  offlineState.sessionId = session.session_id;
  offlineState.stopRequested = false;
  offlineState.sourceSampleRate = format.sampleRate;
  offlineState.totalSourceFrames = format.frameCount;
  offlineState.uploadedSourceFrames = 0;
  updateControlAvailability();

  const transformer = wavUpload.createMonoPcm16Transformer(format, 16000);
  const sourceFramesPerChunk = Math.max(format.sampleRate, Math.floor(format.sampleRate * 2));
  renderTranscript(offlineResult, null, "离线音频处理中...");
  offlineStatus.textContent = `上传转写中... 已上传 0.0/${formatSeconds(format.durationSeconds)}s`;

  let finalized = false;
  try {
    for (let frameOffset = 0; frameOffset < format.frameCount; frameOffset += sourceFramesPerChunk) {
      if (offlineState.stopRequested) {
        break;
      }
      const frameCount = Math.min(sourceFramesPerChunk, format.frameCount - frameOffset);
      const range = wavUpload.getChunkByteRange(format, frameOffset, frameCount);
      const chunkBuffer = await file.slice(range.start, range.end).arrayBuffer();
      const isLastChunk = frameOffset + frameCount >= format.frameCount;
      const pcmChunk = wavUpload.convertChunkToMonoPcm16(chunkBuffer, transformer, isLastChunk);
      offlineState.uploadedSourceFrames = frameOffset + frameCount;
      if (pcmChunk.length === 0) {
        continue;
      }
      const data = await sendOfflineStreamChunk(offlineState.sessionId, pcmChunk);
      renderTranscript(offlineResult, data, "尚无结果");
      updateOfflineStreamStatus(data, format, false);
    }
    await finalizeOfflineStreamSession(format);
    finalized = true;
  } finally {
    if (!finalized) {
      await releaseOfflineStreamSession();
    }
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (hasRealtimeSession()) {
    offlineStatus.textContent = "实时转写进行中，请先停止。";
    return;
  }
  const file = audioFile.files[0];
  if (!file) {
    offlineResult.textContent = "请先选择 WAV 文件。";
    return;
  }

  offlineState.mode = "preparing";
  offlineState.startedAt = performance.now();
  offlineState.stopRequested = false;
  updateControlAvailability();
  offlineResult.textContent = "";
  offlineStatus.textContent = "检查音频中...";

  try {
    const inspection = await inspectOfflineUploadFile(file);
    if (inspection.supported) {
      await submitOfflineViaStream(file, offlineState.startedAt, inspection.format);
      return;
    }

    if (file.size > MAX_ASYNC_UPLOAD_BYTES) {
      throw new Error(`${inspection.reason}；且文件超过 64MB，无法回退到整包上传，请先转成 16-bit PCM WAV`);
    }

    offlineStatus.textContent = "当前文件不适合前端分块，回退到整包上传...";
    await submitOfflineViaAsync(file, offlineState.startedAt);
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

  if (offlineState.mode !== "async" || !offlineState.jobId) {
    return;
  }

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
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      sum += input[sampleIndex];
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
    syncRealtimeArchive(data);
    const audioDur = (data.sample_count / 16000).toFixed(1);
    const decodedDur = (data.decoded_samples / 16000).toFixed(1);
    const wallElapsed = ((performance.now() - realtimeState.startedAt) / 1000).toFixed(1);
    const lag = (wallElapsed - decodedDur).toFixed(1);
    const infMs = data.inference_ms !== undefined ? data.inference_ms.toFixed(0) : "-";
    const decodeLabel = data.decoded ? "已解码" : "待下轮";
    realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel}`;
  } catch (error) {
    realtimeStatus.textContent = `失败：${error.message}`;
  } finally {
    realtimeState.sending = false;
  }
}

async function pollRealtimeStatus() {
  if (!realtimeState.sessionId || realtimeState.sending) {
    return;
  }
  try {
    const response = await fetch(`/api/realtime/status?session_id=${encodeURIComponent(realtimeState.sessionId)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error?.message || "查询实时状态失败");
    }
    renderTranscript(realtimeResult, data, "尚无结果");
    syncRealtimeArchive(data);
    const audioDur = (data.sample_count / 16000).toFixed(1);
    const decodedDur = (data.decoded_samples / 16000).toFixed(1);
    const wallElapsed = ((performance.now() - realtimeState.startedAt) / 1000).toFixed(1);
    const lag = (wallElapsed - decodedDur).toFixed(1);
    const infMs = data.inference_ms !== undefined ? data.inference_ms.toFixed(0) : "-";
    const decodeLabel = data.decoded ? "已解码" : "待下轮";
    realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel}`;
  } catch (error) {
    realtimeStatus.textContent = `失败：${error.message}`;
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
      sendTimer: window.setInterval(() => flushRealtimeChunk(false), 400),
      pollTimer: window.setInterval(() => pollRealtimeStatus(), 150),
      sending: false,
      pending: [],
      sampleRate: audioContext.sampleRate,
      startedAt: performance.now(),
    };
    resetRealtimeArchive("实时转写中，已确定文本会保存在此处。");
    realtimeArchive.sessionId = sessionData.session_id;
    updateRealtimeExportAvailability();

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
  window.clearInterval(realtimeState.pollTimer);
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
    if (response.ok) {
      renderTranscript(realtimeResult, data, "尚无结果");
      syncRealtimeArchive(data);
      realtimeStatus.textContent = data.text
        ? `会话 ${sessionId} 已停止，终稿已出`
        : `会话 ${sessionId} 已停止`;
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
      pollTimer: null,
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
  resetTranscriptFrame(realtimeResult, "尚无结果");
  resetRealtimeArchive();
  realtimeStatus.textContent = "未开始";
  clearRealtime.style.display = "none";
});

exportRealtimeText.addEventListener("click", () => {
  exportRealtimeTranscript("txt");
});

exportRealtimeJson.addEventListener("click", () => {
  exportRealtimeTranscript("json");
});

updateControlAvailability();
resetRealtimeArchive();
checkHealth();
