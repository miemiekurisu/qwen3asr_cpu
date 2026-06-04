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
const realtimeStatus = document.getElementById("realtimeStatus");
const audioMeter = document.getElementById("audioMeter");
const meterPre = document.getElementById("meterPre");
const meterPost = document.getElementById("meterPost");
const meterSrv = document.getElementById("meterSrv");

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
  /* Terminal-state machine for the live display.
   * Each element: { state: 'done'|'typing'|'cursor', el, text }
   * `done` lines are committed; `typing` is animating; `cursor`
   * is the empty line that blinks while waiting. */
  lines: [],
  typewriterTimer: null,
  lastSegmentCount: 0,
  finalized: false,
  updatedAt: "",
};

/* Pick up the initial cursor line rendered by index.html so the
 * terminal state machine has a consistent starting point. */
{
  const initial = realtimeResult.querySelector(".term-line.cursor");
  if (initial) {
    realtimeArchive.lines.push({ state: "cursor", el: initial, text: "" });
  }
}

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
  /* Terminal state lives in `realtimeArchive` (a closure-friendly
   * global), not on the element — there is only ONE terminal display
   * and it owns its state.  We still return a tiny frame object for
   * the legacy apply/render plumbing to call into. */
  return {
    pendingData: null,
    pendingFallback: null,
    renderScheduled: false,
  };
}

function resetTranscriptFrame(element, fallback) {
  /* The terminal display owns its own state machine.  For the realtime
   * element, we reset the machine; for everything else (e.g. the
   * offline result), we just clear the DOM. */
  if (element === realtimeResult) {
    if (realtimeArchive.typewriterTimer !== null) {
      clearInterval(realtimeArchive.typewriterTimer);
      realtimeArchive.typewriterTimer = null;
    }
    realtimeArchive.lines = [];
    realtimeArchive.lastSegmentCount = 0;
    realtimeArchive.finalized = false;
    element.innerHTML = "";
    const cursorLine = makeTermLine("cursor", "");
    element.appendChild(cursorLine);
    realtimeArchive.lines.push({ state: "cursor", el: cursorLine, text: "" });
  } else {
    element.textContent = fallback;
  }
}

function makeTermLine(state, text) {
  const div = document.createElement("div");
  div.className = "term-line " + state;
  if (state === "cursor" && !text) {
    div.classList.add("empty");
    const c = document.createElement("span");
    c.className = "cursor-blink";
    div.appendChild(c);
  } else {
    div.appendChild(document.createTextNode(text));
  }
  return div;
}

function renderTranscript(element, data, fallback) {
  /* Coalesce: keep the latest data, schedule exactly one rAF if not
   * already scheduled.  Multiple back-to-back renderTranscript() calls
   * (one per HTTP response within 150-400 ms) all land on the same
   * paint frame, eliminating visible stutter. */
  if (data === null || data === undefined) {
    resetTranscriptFrame(element, fallback);
    return;
  }
  const frame = ensureTranscriptFrame(element);
  frame.pendingData = data;
  frame.pendingFallback = fallback;
  if (frame.renderScheduled) {
    return;
  }
  frame.renderScheduled = true;
  requestAnimationFrame(() => {
    frame.renderScheduled = false;
    applyTranscriptRender(element, frame.pendingData, frame.pendingFallback);
  });
}

function applyTranscriptRender(element, data, fallback) {
  const segments = Array.isArray(data?.segments)
    ? data.segments.filter((s) => typeof s === "string" && s)
    : [];

  /* On finalize, freeze the cursor line (no more typewriter) but keep
   * the lines as-is so the user can read the final transcript. */
  if (data?.finalized) {
    if (realtimeArchive.typewriterTimer !== null) {
      clearInterval(realtimeArchive.typewriterTimer);
      realtimeArchive.typewriterTimer = null;
    }
    realtimeArchive.finalized = true;
    realtimeArchive.lastSegmentCount = segments.length;
    return;
  }

  /* New segment committed by the server?  Animate the new text into
   * the live line via the typewriter, then commit it to a `done`
   * line and start a fresh `cursor` line. */
  if (segments.length > realtimeArchive.lastSegmentCount) {
    const newText = segments[segments.length - 1];
    if (newText) {
      animateNewSegment(element, newText);
    }
    realtimeArchive.lastSegmentCount = segments.length;
  }

  /* Auto-scroll to keep the live line in view. */
  element.scrollTop = element.scrollHeight;
}

function animateNewSegment(element, text) {
  /* Stop any in-flight typewriter (segmentation events should be
   * rare; this is a safety net). */
  if (realtimeArchive.typewriterTimer !== null) {
    clearInterval(realtimeArchive.typewriterTimer);
    realtimeArchive.typewriterTimer = null;
  }

  /* Find the bottom-most line.  If it's the cursor (blinking,
   * empty), promote it to a typing line — the user has spoken and
   * we're filling in the text.  If a previous segment's typing
   * animation was somehow interrupted mid-stream, fall through and
   * just append a fresh typing line. */
  let typingEntry = null;
  const lastLine = realtimeArchive.lines[realtimeArchive.lines.length - 1];
  if (lastLine && lastLine.state === "cursor") {
    /* Convert the cursor line into a typing line: strip the
     * cursor-blink span, drop the "cursor"/"empty" classes, add
     * "typing". */
    const blink = lastLine.el.querySelector(".cursor-blink");
    if (blink) blink.remove();
    lastLine.el.classList.remove("cursor", "empty");
    lastLine.el.classList.add("typing");
    lastLine.state = "typing";
    lastLine.el.textContent = "";
    typingEntry = lastLine;
  } else {
    /* No cursor to convert (we were mid-typing, or somehow there
     * was no cursor).  Append a fresh typing line. */
    const typingLine = makeTermLine("typing", "");
    element.appendChild(typingLine);
    typingEntry = { state: "typing", el: typingLine, text: "" };
    realtimeArchive.lines.push(typingEntry);
  }

  /* Typewriter: reveal one char at a time.  Speed adapts to length
   * so short sentences aren't annoying and long ones are watchable:
   *   <= 8 chars: 10ms/char (snappy)
   *   8-30 chars: 18ms/char
   *   >= 30 chars: 14ms/char (capped so a 60-char sentence still
   *   finishes in under a second). */
  const perCharMs = text.length <= 8 ? 10 : text.length <= 30 ? 18 : 14;
  let i = 0;
  const tick = () => {
    if (i >= text.length) {
      clearInterval(realtimeArchive.typewriterTimer);
      realtimeArchive.typewriterTimer = null;
      /* Typewriter complete: freeze the line as "done" and add a
       * single new cursor line underneath.  This cursor is the
       * ONLY place the blink animation is active — there is no
       * cursor under an in-progress typing line. */
      typingEntry.el.classList.remove("typing");
      typingEntry.el.classList.add("done");
      typingEntry.state = "done";
      typingEntry.text = text;
      const cursorLine = makeTermLine("cursor", "");
      element.appendChild(cursorLine);
      realtimeArchive.lines.push({ state: "cursor", el: cursorLine, text: "" });
      element.scrollTop = element.scrollHeight;
      return;
    }
    i += 1;
    typingEntry.el.textContent = text.slice(0, i);
    typingEntry.text = text.slice(0, i);
  };
  realtimeArchive.typewriterTimer = setInterval(tick, perCharMs);
  /* Show the first character immediately so the user doesn't wait
   * one tick to see anything. */
  tick();
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

function extractConfirmedRealtimeText() {
  /* Archive = locked terminal lines (state 'done').  The currently-
   * typing line is excluded because it isn't a final commitment yet
   * — it could still be replaced if a new segment commit arrives. */
  return realtimeArchive.lines
    .filter((l) => l.state === "done" && l.text && l.text.trim())
    .map((l) => l.text)
    .join(" ");
}

function updateRealtimeExportAvailability() {
  const hasConfirmedText = Boolean(extractConfirmedRealtimeText().trim());
  exportRealtimeText.disabled = !hasConfirmedText;
  exportRealtimeJson.disabled = !hasConfirmedText;
}

function resetRealtimeArchive(fallback = "尚无已确定文本") {
  realtimeArchive = {
    sessionId: "",
    lines: [],
    typewriterTimer: null,
    lastSegmentCount: 0,
    finalized: false,
    updatedAt: "",
  };
  /* Mirror the terminal reset on the DOM. */
  realtimeResult.innerHTML = "";
  const cursorLine = makeTermLine("cursor", "");
  realtimeResult.appendChild(cursorLine);
  realtimeArchive.lines.push({ state: "cursor", el: cursorLine, text: "" });
  updateRealtimeExportAvailability();
}

function syncRealtimeArchive(data) {
  realtimeArchive.sessionId = data?.session_id || realtimeState.sessionId || realtimeArchive.sessionId;
  realtimeArchive.finalized = Boolean(data?.finalized);
  realtimeArchive.updatedAt = new Date().toISOString();
  updateRealtimeExportAvailability();
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
  const text = extractConfirmedRealtimeText();
  if (!text.trim()) {
    realtimeStatus.textContent = "暂无可导出的已确定文本";
    return;
  }

  if (format === "txt") {
    triggerDownload(
      buildRealtimeExportName("txt"),
      text,
      "text/plain;charset=utf-8",
    );
    realtimeStatus.textContent = "已导出 TXT";
    return;
  }

  const payload = {
    exported_at: new Date().toISOString(),
    session_id: realtimeArchive.sessionId,
    finalized: realtimeArchive.finalized,
    confirmed_text: text,
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

  // Signal end-of-audio (non-blocking).
  const eofPrefix = offlineState.stopRequested ? "停止中..." : "后处理中...";
  offlineStatus.textContent = eofPrefix;
  try {
    await fetch(`/api/realtime/eof?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: "",
    });
  } catch (_err) {
    // Best-effort; the stream will still work if eof was already set.
  }

  // Open SSE stream for progressive updates while worker finishes.
  const streamResult = await new Promise((resolve, reject) => {
    const url = `/api/realtime/stream?session_id=${encodeURIComponent(sessionId)}`;
    const es = new EventSource(url);
    let lastData = null;

    es.onmessage = (event) => {
      if (event.data === "[DONE]") {
        es.close();
        resolve(lastData);
        return;
      }
      try {
        const data = JSON.parse(event.data);
        lastData = data;
        // Offline batch: show full accumulated stable_text, not windowed
        // recent_segments. The user needs to see the entire transcript
        // growing.  offlineResult is a <pre>, so we collapse final/history/
        // stable/partial into a single textContent (escaped implicitly).
        const fullStable = data.stable_text || "";
        const trailing = data.live_partial_text || data.partial_text || "";
        if (data.finalized) {
          offlineResult.textContent = data.text || fullStable;
        } else {
          offlineResult.textContent = trailing
            ? (fullStable ? fullStable + "\n" + trailing : trailing)
            : fullStable;
        }
        updateOfflineStreamStatus(data, format, false);
      } catch (_parseErr) {
        // Ignore malformed events.
      }
    };

    es.onerror = () => {
      es.close();
      if (lastData) {
        resolve(lastData);
      } else {
        reject(new Error("streaming connection lost"));
      }
    };
  });

  // Final cleanup: join worker thread, release session.
  offlineState.sessionId = "";
  updateControlAvailability();
  try {
    const response = await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: "",
    });
    const data = await response.json();
    if (response.ok) {
      // Show full final text, not windowed view.
      const finalText = data.text || data.stable_text || "";
      offlineResult.textContent = finalText;
      updateOfflineStreamStatus(data, format, true);
      return data;
    }
  } catch (_err) {
    // Best-effort cleanup.
  }

  // Fallback: use last streamed snapshot.
  if (streamResult) {
    updateOfflineStreamStatus(streamResult, format, true);
  }
  return streamResult;
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
      // Offline batch: show full accumulated stable_text, not windowed
      // view.  offlineResult is a <pre>, so collapse to a single
      // textContent (trailing partial appended on its own line).
      const fullStable = data.stable_text || "";
      const trailing = data.live_partial_text || data.partial_text || "";
      offlineResult.textContent = trailing
        ? (fullStable ? fullStable + "\n" + trailing : trailing)
        : fullStable;
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
  // Decimate, not average-pool!  Average of N consecutive samples
  // cancels the audio signal (speech oscillates around zero, summing
  // N samples → ~0).  We must *pick* one sample per window, not
  // average them.  Use the center of each window for a stable pick.
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const center = (index + 0.5) * ratio;
    output[index] = input[Math.min(input.length - 1, Math.floor(center))];
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
  // realtimeState.pending is a list of Int16Array chunks. Concatenate into a
  // single Int16Array via direct typed-array copies (no per-sample boxing).
  const chunks = realtimeState.pending;
  realtimeState.pending = [];
  let total = 0;
  for (let i = 0; i < chunks.length; i++) {
    total += chunks[i].length;
  }
  if (total === 0) {
    return;
  }
  const buffer = new Int16Array(total);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    buffer.set(chunks[i], offset);
    offset += chunks[i].length;
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
    realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel} | mic 峰 ${(realtimeState.prePeak||0).toFixed(3)} → 16k 峰 ${(realtimeState.postPeak||0).toFixed(3)}`;
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
    realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel} | mic 峰 ${(realtimeState.prePeak||0).toFixed(3)} → 16k 峰 ${(realtimeState.postPeak||0).toFixed(3)}`;
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
      meterTimer: window.setInterval(() => updateAudioMeter(), 100),
      sending: false,
      pending: [],
      prePeak: 0,
      preRms: 0,
      postPeak: 0,
      postRms: 0,
      sampleRate: audioContext.sampleRate,
      startedAt: performance.now(),
    };
    audioMeter.style.display = "block";
    resetRealtimeArchive("实时转写中，已确定文本会保存在此处。");
    realtimeArchive.sessionId = sessionData.session_id;
    updateRealtimeExportAvailability();

    processor.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0);
      /* Diagnostic: track peak/RMS at capture time, pre-downsample.
       * If pre_peak is non-zero but post_peak is zero, the bug is in
       * downsampleTo16k.  If both are zero, the bug is upstream
       * (mic muted, wrong device, ScriptProcessor quirk). */
      let pre_peak = 0;
      let pre_sum = 0;
      for (let i = 0; i < channel.length; i++) {
        const a = channel[i] < 0 ? -channel[i] : channel[i];
        if (a > pre_peak) pre_peak = a;
        pre_sum += channel[i] * channel[i];
      }
      realtimeState.prePeak = Math.max(realtimeState.prePeak || 0, pre_peak);
      realtimeState.preRms = Math.sqrt(pre_sum / channel.length);
      const downsampled = downsampleTo16k(channel, realtimeState.sampleRate);
      const pcm = floatToPcm16(downsampled);
      let post_peak = 0;
      let post_sum = 0;
      for (let i = 0; i < downsampled.length; i++) {
        const a = downsampled[i] < 0 ? -downsampled[i] : downsampled[i];
        if (a > post_peak) post_peak = a;
        post_sum += downsampled[i] * downsampled[i];
      }
      realtimeState.postPeak = Math.max(realtimeState.postPeak || 0, post_peak);
      realtimeState.postRms = Math.sqrt(post_sum / downsampled.length);
      // Push the typed-array chunk as-is. Spreading (push(...pcm)) would box
      // every Int16 sample into a JS Number, churning the GC at audio rate.
      realtimeState.pending.push(pcm);
    };

    function updateAudioMeter() {
      if (!realtimeState) return;
      const pre = realtimeState.prePeak || 0;
      const post = realtimeState.postPeak || 0;
      meterPre.textContent = pre.toFixed(3);
      meterPost.textContent = post.toFixed(3);
      meterPre.style.color = pre > 0.01 ? "#0a0" : "#c00";
      meterPost.style.color = post > 0.01 ? "#0a0" : "#c00";
      /* Poll the server-side ingress diag endpoint.  Use a fire-and-
       * forget fetch; failure is fine (server may have restarted).
       * max_peak 是全程最大, 不会因为末尾静音就掉到 0. */
      if (realtimeState.sessionId) {
        fetch(`/api/realtime/audio_diag?session_id=${encodeURIComponent(realtimeState.sessionId)}`)
          .then(r => r.json())
          .then(d => {
            const p = typeof d.max_peak === "number" ? d.max_peak : d.peak;
            if (typeof p === "number") {
              meterSrv.textContent = p.toFixed(3);
              meterSrv.style.color = p > 0.01 ? "#0a0" : "#c00";
            }
          })
          .catch(() => {});
      }
    }

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
    const prevStats = realtimeStatus.textContent;
    if (response.ok) {
      renderTranscript(realtimeResult, data, "尚无结果");
      syncRealtimeArchive(data);
      const stopLabel = data.text ? "已停止，终稿已出" : "已停止";
      realtimeStatus.textContent = prevStats
        ? `${prevStats} / ${stopLabel}`
        : `会话 ${sessionId} ${stopLabel}`;
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
