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

/* Batch and Realtime are TWO INDEPENDENT feature lines on the
 * client and on the server.  Batch is "upload complete audio file
 * → server ffmpeg → offline transcribe" via /api/transcriptions/async.
 * Realtime is "mic stream → VAD-segmented worker" via /api/realtime/*.
 * They must not share a session and must not be active at the same
 * time; the UI enforces this in updateControlAvailability(). */
const BATCH_FEATURE = "batch";
const REALTIME_FEATURE = "realtime";
const MAX_ASYNC_UPLOAD_BYTES = 64 * 1024 * 1024;

let activeFeature = "";
let realtimeStarting = false;
let realtimeStopping = false;
/* Synchronous in-flight guard for startRealtimeCapture.  Unlike
 * `realtimeStarting` (which is set by the click handler) and
 * `activeFeature` (which is set only after the awaits complete),
 * this flag is set at the entry of startRealtimeCapture itself,
 * before any await, so a second concurrent invocation of
 * startRealtimeCapture cannot race past the guard at the top. */
let realtimeCapturing = false;

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
  jobId: "",
  stopRequested: false,
  /* Non-empty when a stop attempt failed and the error message
   * is currently displayed.  The poll loop refuses to overwrite
   * the status formatter (which would say "转写中: …") while
   * this is set, so the user actually sees "停止失败: …" instead
   * of the error being silently replaced 300ms later. */
  stopError: "",
  startedAt: 0,
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
    /* If the terminal already has done lines above this cursor,
     * the placeholder text should be terse ("continue speaking")
     * rather than the long "waiting for first input" message —
     * the user knows this is the transcript area by then.  Cheap
     * O(n) scan; n is the number of committed segments which is
     * small in practice. */
    if (realtimeArchive.lines.some((l) => l.state === "done")) {
      div.classList.add("has-done");
    }
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

  /* New segment(s) committed by the server?  When the server pushes
   * multiple commits in a single SSE event (fast speech, or
   * condition-variable wakeup coalescing), segments[] may jump by
   * N>1 at once.  Animate ONLY the last one via the typewriter (so
   * the user sees live transcription) but render the EARLIER ones
   * as instant `done` lines (no typewriter for them — they were
   * already "spoken" before this event arrived).  Without this,
   * middle segments were silently dropped on screen, making the UI
   * show fewer done lines than the server has in segments_text.
   *
   * IMPORTANT: process new segments BEFORE the finalize check.  The
   * server's final flush (triggered by /stop) commits a final segment
   * at the same time as data.finalized=true.  If the finalize check
   * runs first, it returns early and the final segment is dropped
   * on screen — symptom: "stop button shows 终稿已出 but result is
   * empty".  Symptom reproduced in browser E2E test, session 12,
   * 4.86s push + immediate stop, server returned segments=1 /
   * finalized=true, but the UI never displayed the text.
   *
   * Order: (1) render any new segments, (2) freeze on finalize.
   *
   * On finalize, skip the typewriter — show the full text immediately.
   * Without this, animateNewSegment() starts a typewriter (showing
   * the first char 'T' synchronously) and the finalize check below
   * immediately clearInterval()s it, leaving the user with just the
   * first character.  Symptom: "result shows only T".
   *
   * Order: (1) render any new segments (typewriter if streaming,
   * instant if finalized), (2) freeze on finalize. */
  if (segments.length > realtimeArchive.lastSegmentCount) {
    const newCount = segments.length - realtimeArchive.lastSegmentCount;
    if (newCount === 1) {
      /* Single new segment: animate it (or render final if stopping). */
      const newText = segments[segments.length - 1];
      if (newText) {
        if (data?.finalized) {
          renderFinalizedSegment(element, newText);
        } else {
          animateNewSegment(element, newText);
        }
      }
    } else {
      /* Multiple new segments in one event (SSE batch).  Render all
       * but the last one as instant done lines (no typewriter — they
       * were committed before this event, the user is catching up),
       * then animate the last one if streaming, or render it final. */
      for (let i = realtimeArchive.lastSegmentCount; i < segments.length - 1; i++) {
        const t = segments[i];
        if (t) renderFinalizedSegment(element, t);
      }
      const newText = segments[segments.length - 1];
      if (newText) {
        if (data?.finalized) {
          renderFinalizedSegment(element, newText);
        } else {
          animateNewSegment(element, newText);
        }
      }
    }
    realtimeArchive.lastSegmentCount = segments.length;
  }

  /* On finalize, freeze the cursor line (no more typewriter) but keep
   * the lines as-is so the user can read the final transcript. */
  if (data?.finalized) {
    if (realtimeArchive.typewriterTimer !== null) {
      clearInterval(realtimeArchive.typewriterTimer);
      realtimeArchive.typewriterTimer = null;
    }
    realtimeArchive.finalized = true;
    return;
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

/* Like animateNewSegment but skips the typewriter and renders the
 * full text immediately.  Used when the session is finalized: the
 * user has clicked Stop, the final text is available, and the slow
 * typewriter reveal is pure friction (would be cut off after 1
 * char by the finalize freeze).  Also creates a fresh cursor line
 * underneath, matching the post-typewriter state. */
function renderFinalizedSegment(element, text) {
  /* Stop any in-flight typewriter (e.g. from a prior render that
   * raced with finalize) so we don't end up with two timers. */
  if (realtimeArchive.typewriterTimer !== null) {
    clearInterval(realtimeArchive.typewriterTimer);
    realtimeArchive.typewriterTimer = null;
  }
  /* Find the bottom-most line.  Promote the cursor line to a
   * done line carrying the full text.  If there's no cursor (we
   * were mid-typing from a previous render, or the array is
   * empty), append a fresh done line. */
  const lastLine = realtimeArchive.lines[realtimeArchive.lines.length - 1];
  if (lastLine && lastLine.state === "cursor") {
    const blink = lastLine.el.querySelector(".cursor-blink");
    if (blink) blink.remove();
    lastLine.el.classList.remove("cursor", "empty");
    lastLine.el.classList.add("done");
    lastLine.state = "done";
    lastLine.text = text;
    lastLine.el.textContent = text;
  } else {
    const doneLine = makeTermLine("done", text);
    element.appendChild(doneLine);
    realtimeArchive.lines.push({ state: "done", el: doneLine, text });
  }
  element.scrollTop = element.scrollHeight;
}

function hasOfflineJob() {
  return activeFeature === BATCH_FEATURE;
}

function hasRealtimeSession() {
  return activeFeature === REALTIME_FEATURE && realtimeState.sessionId !== "";
}

function updateControlAvailability() {
  const offlineActive = hasOfflineJob();
  const realtimeActive = hasRealtimeSession();
  const realtimeBusy = realtimeStarting || realtimeStopping || realtimeActive;
  const canStopOffline = offlineActive && offlineState.jobId !== "" && !offlineState.stopRequested;
  const hasConfirmedText = Boolean(extractConfirmedRealtimeText().trim());

  audioFile.disabled = offlineActive || realtimeActive;
  offlineSubmit.disabled = offlineActive || realtimeActive;
  offlineStop.disabled = !canStopOffline;

  startRealtime.disabled = offlineActive || realtimeBusy;
  stopRealtime.disabled = !realtimeActive || realtimeStopping;
  /* Clear is allowed only in idle (not live, not starting, not stopping).
   * Hide it during the in-flight windows and during a live session. */
  clearRealtime.style.display = realtimeBusy || offlineActive ? "none" : "";
  /* Export follows the same rule as clear: only after stop, never
   * while a session is live or being set up/torn down. */
  exportRealtimeText.disabled = realtimeBusy || offlineActive || !hasConfirmedText;
  exportRealtimeJson.disabled = realtimeBusy || offlineActive || !hasConfirmedText;

  /* Audio meter is only useful when audio is actually flowing. */
  if (audioMeter) {
    audioMeter.style.display = realtimeActive ? "block" : "none";
  }
}

function resetOfflineState() {
  offlineState = {
    jobId: "",
    stopRequested: false,
    stopError: "",
    startedAt: 0,
  };
  if (activeFeature === BATCH_FEATURE) {
    activeFeature = "";
  }
  updateControlAvailability();
}

function countCodepoints(text) {
  return QasrStatePure.countCodepoints(text);
}

function extractConfirmedRealtimeText() {
  /* Archive = locked terminal lines (state 'done').  The currently-
   * typing line is excluded because it isn't a final commitment yet
   * — it could still be replaced if a new segment commit arrives. */
  return QasrStatePure.computeConfirmedRealtimeText(realtimeArchive.lines);
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

/* Soft reset for "Start a new session after Stop".  Keeps the
 * already-committed `done` lines visible (the "old info preserved"
 * rule the operator asked for), stops the typewriter, and ensures
 * the terminal ends with a fresh empty cursor so the next session's
 * segments animate into a clean line. */
function softResetRealtimeArchive(newSessionId) {
  if (realtimeArchive.typewriterTimer !== null) {
    clearInterval(realtimeArchive.typewriterTimer);
    realtimeArchive.typewriterTimer = null;
  }
  realtimeArchive.sessionId = newSessionId;
  realtimeArchive.lastSegmentCount = 0;
  realtimeArchive.finalized = false;
  realtimeArchive.updatedAt = new Date().toISOString();

  /* If the previous session ended on an in-flight typing line, decide
   * whether to keep its text (commit to done) or drop it (empty). */
  const lastLine = realtimeArchive.lines[realtimeArchive.lines.length - 1];
  if (lastLine && lastLine.state === "typing") {
    if (lastLine.text) {
      lastLine.state = "done";
      lastLine.el.classList.remove("typing");
      lastLine.el.classList.add("done");
    } else {
      realtimeArchive.lines.pop();
      if (lastLine.el && lastLine.el.parentNode) {
        lastLine.el.parentNode.removeChild(lastLine.el);
      }
    }
  }

  /* Ensure the terminal ends with a fresh empty cursor. */
  const tail = realtimeArchive.lines[realtimeArchive.lines.length - 1];
  if (!tail || tail.state !== "cursor" || tail.text) {
    const cursorLine = makeTermLine("cursor", "");
    realtimeResult.appendChild(cursorLine);
    realtimeArchive.lines.push({ state: "cursor", el: cursorLine, text: "" });
  }

  updateRealtimeExportAvailability();
}

function syncRealtimeArchive(data) {
  realtimeArchive.sessionId = data?.session_id || realtimeState.sessionId || realtimeArchive.sessionId;
  realtimeArchive.finalized = Boolean(data?.finalized);
  realtimeArchive.updatedAt = new Date().toISOString();
  updateRealtimeExportAvailability();
}

function buildRealtimeExportName(ext) {
  return QasrStatePure.buildRealtimeExportName(realtimeArchive.sessionId, ext);
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
      runtimeHint.textContent = "音频文件上传转写 (server ffmpeg) 与浏览器麦克风实时转写可用";
      return;
    }
  } catch (error) {
    runtimeHint.textContent = error.message;
  }
  healthBadge.textContent = "未就绪";
}

function updateOfflineAsyncStatus(job, startTime) {
  const text = job.text || "";
  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  const chars = countCodepoints(text);
  const prefix = job.state === "cancelling" ? "停止中" : "转写中";
  const tokenLabel = job.token_count > 0 ? ` / ${job.token_count} tokens` : "";
  if (chars > 0) {
    offlineStatus.textContent = `${prefix}: 已识别 ${chars} 字 / ${elapsed}s${tokenLabel}`;
  } else {
    offlineStatus.textContent = `${prefix}: ${elapsed}s${tokenLabel}`;
  }
}

function showOfflineError(message) {
  offlineResult.textContent = `失败: ${message}`;
}

async function submitOfflineViaAsync(file, startTime) {
  if (file.size > MAX_ASYNC_UPLOAD_BYTES) {
    throw new Error(`文件 ${(file.size / 1024 / 1024).toFixed(1)}MB 超过 64MB 限制 (server multipart 上限)`);
  }
  if (file.size <= 0) {
    throw new Error("文件为空");
  }

  offlineResult.textContent = "上传到服务器...";
  const form = new FormData();
  form.append("audio", file);
  const submitRes = await fetch("/api/transcriptions/async", {
    method: "POST",
    body: form,
  });
  const submitData = await submitRes.json();
  if (!submitRes.ok) {
    throw new Error(submitData.error?.message || `HTTP ${submitRes.status}`);
  }

  activeFeature = BATCH_FEATURE;
  offlineState.jobId = submitData.id;
  offlineState.stopRequested = false;
  updateControlAvailability();
  offlineStatus.textContent = "提交成功, 转写中...";

  let lastTextLen = 0;
  let pollAttempts = 0;
  /* Long-audio VAD-segmented batch can take 10-15 min for a 28 min
   * file.  6000 × 300ms = 30 min safety cap.  The server's per-segment
   * text callback fills job.text in real time, so the user sees
   * progress even within this window. */
  const MAX_POLL_ATTEMPTS = 6000;
  while (pollAttempts < MAX_POLL_ATTEMPTS) {
    pollAttempts += 1;
    await new Promise((resolve) => setTimeout(resolve, 300));
    const pollRes = await fetch(`/api/jobs/${encodeURIComponent(submitData.id)}`);
    const job = await pollRes.json();
    if (!pollRes.ok) {
      throw new Error(job.error?.message || `HTTP ${pollRes.status}`);
    }

    if (job.state === "running" || job.state === "queued" || job.state === "cancelling") {
      const text = job.text || "";
      if (text.length > lastTextLen) {
        lastTextLen = text.length;
        offlineResult.textContent = text;
      }
      /* Don't overwrite the status line while a stop attempt is
       * in flight OR a stop error is currently displayed.  The
       * running formatter ("转写中: 0.6s") would otherwise replace
       * the user's stop signal within 300ms.  We keep refreshing
       * `offlineResult` (text content) because that is
       * append-only, but the status line belongs to the stop
       * flow until it terminates (state=cancelled/completed/
       * failed, which the branches below clear stopError on). */
      if (!offlineState.stopRequested && !offlineState.stopError) {
        updateOfflineAsyncStatus(job, startTime);
      }
      continue;
    }

    if (job.state === "cancelled") {
      offlineState.stopError = "";
      offlineResult.textContent = job.text || "已停止";
      offlineStatus.textContent = "已停止";
      return;
    }

    if (job.state === "failed") {
      offlineState.stopError = "";
      throw new Error(job.error || "转写失败");
    }

    // completed
    offlineState.stopError = "";
    offlineResult.textContent = job.text || "";
    const audioDur = (job.audio_ms / 1000).toFixed(1);
    const infMs = (job.inference_ms || 0).toFixed(0);
    const rtf = job.audio_ms > 0 ? (job.inference_ms / job.audio_ms).toFixed(2) : "-";
    const tokens = job.tokens || 0;
    offlineStatus.textContent =
      `音频 ${audioDur}s / 推理 ${infMs}ms / RTF ${rtf} / ${tokens} tokens`;
    return;
  }
  throw new Error("转写超时 (>30min), 请检查音频长度或服务端状态");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (hasRealtimeSession()) {
    showOfflineError("实时转写进行中, 请先停止");
    return;
  }
  if (hasOfflineJob()) {
    return;
  }
  const file = audioFile.files[0];
  if (!file) {
    showOfflineError("请先选择音频文件");
    return;
  }

  activeFeature = BATCH_FEATURE;
  offlineState.jobId = "";
  offlineState.startedAt = performance.now();
  offlineState.stopRequested = false;
  updateControlAvailability();
  offlineResult.textContent = "上传到服务器...";
  offlineStatus.textContent = "上传中...";

  try {
    await submitOfflineViaAsync(file, offlineState.startedAt);
  } catch (error) {
    showOfflineError(error.message);
    offlineStatus.textContent = "";
  } finally {
    resetOfflineState();
  }
});

offlineStop.addEventListener("click", async () => {
  if (!hasOfflineJob() || !offlineState.jobId) {
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
    /* Cancel accepted: clear stopRequested so the poll loop can
     * pick up state="cancelled" and surface "已停止" when the
     * server acknowledges. */
    offlineState.stopRequested = false;
  } catch (error) {
    /* Cancel failed: clear stopRequested so the user can retry,
     * but record the error so the poll loop leaves the visible
     * "停止失败: …" status alone (don't replace it with the
     * "转写中: …" formatter on the next 300ms poll). */
    offlineState.stopRequested = false;
    offlineState.stopError = error.message;
    updateControlAvailability();
    offlineStatus.textContent = `停止失败: ${error.message}`;
  }
});

function downsampleTo16k(input, inputRate) {
  return QasrStatePure.downsampleTo16k(input, inputRate);
}

function floatToPcm16(input) {
  return QasrStatePure.floatToPcm16(input);
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
    /* Cache the server-side max_peak for updateAudioMeter to read
     * without issuing a separate /audio_diag fetch.  Both fields
     * are piggybacked on /status now. */
    if (typeof data.max_ingress_peak === "number") {
      realtimeState.maxSrvPeak = data.max_ingress_peak;
    } else if (typeof data.max_peak === "number") {
      realtimeState.maxSrvPeak = data.max_peak;
    }
    realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s / 耗时 ${wallElapsed}s / 滞后 ${lag}s / 推理 ${infMs}ms / ${decodeLabel} | mic 峰 ${(realtimeState.prePeak||0).toFixed(3)} → 16k 峰 ${(realtimeState.postPeak||0).toFixed(3)}`;
  } catch (error) {
    realtimeStatus.textContent = `失败：${error.message}`;
  } finally {
    realtimeState.sending = false;
  }
}

/* Subscribe to /api/realtime/stream (Server-Sent Events) for the
 * active session.  The server pushes one full snapshot on connect,
 * then a compact "update" event every time the ASR worker commits a
 * new segment.  This replaces the previous 300ms /status poll loop
 * (saves ~3.3 req/s) and lets the UI update instantly when a
 * segment is committed, instead of waiting up to 300ms.
 *
 * Lifecycle:
 *   - Called from startRealtimeCapture after sessionId is set
 *   - Auto-closes when server sends data: [DONE] (finalize)
 *   - Manually closed in stopRealtimeCapture before sending /stop
 *   - The EventSource is stored in realtimeState.sse so the stop
 *     path can call .close() on it.
 */
function openSseStream(sessionId) {
  if (!realtimeState || realtimeState.sessionId !== sessionId) {
    return;
  }
  /* Update the meter/display from a snapshot.  Used both for the
   * initial SSE snapshot and for incremental updates. */
  const applyUpdate = (data) => {
    if (!realtimeState || realtimeState.sessionId !== sessionId) return;
    if (typeof data.max_ingress_peak === "number") {
      realtimeState.maxSrvPeak = data.max_ingress_peak;
    } else if (typeof data.max_peak === "number") {
      realtimeState.maxSrvPeak = data.max_peak;
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
  };

  let es;
  try {
    es = new EventSource(`/api/realtime/stream?session_id=${encodeURIComponent(sessionId)}`);
  } catch (err) {
    realtimeStatus.textContent = `SSE 失败：${err.message}`;
    return;
  }
  realtimeState.sse = es;
  /* Maintain a mirror of segments_text in the client so the
   * "update" events (which only carry the latest segment) can be
   * merged into a full snapshot for renderTranscript. */
  realtimeState.sseSegments = [];
  realtimeState.sseLastFull = null;

  es.onmessage = (ev) => {
    if (!realtimeState || realtimeState.sessionId !== sessionId) return;
    if (ev.data === "[DONE]") {
      es.close();
      if (realtimeState) realtimeState.sse = null;
      return;
    }
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    if (data.type === "update") {
      /* Merge: take the last full snapshot and append the new
       * segments to its segments array.  The server sends a
       * `new_segments` array containing ALL segments committed
       * since the last SSE event (could be 0, 1, or many — fast
       * speech can produce 2-3 commits between SSE wakeups).
       * Previously we only pushed the latest text, so middle
       * segments were dropped on the client (symptom: "fast
       * speech loses words"). */
      if (!realtimeState.sseLastFull) return;
      if (Array.isArray(data.new_segments) && data.new_segments.length > 0) {
        for (const seg of data.new_segments) {
          if (typeof seg === "string" && seg.length > 0) {
            realtimeState.sseSegments.push(seg);
          }
        }
      }
      const latestText = realtimeState.sseSegments.length > 0
        ? realtimeState.sseSegments[realtimeState.sseSegments.length - 1]
        : "";
      const merged = Object.assign({}, realtimeState.sseLastFull, {
         sample_count: data.total_samples,
         decoded_samples: data.decoded_samples,
         inference_ms: data.last_inference_ms,
         last_decode_ran: data.last_decode_ran,
         max_ingress_peak: data.max_ingress_peak,
         last_ingress_peak: data.last_ingress_peak,
         ingress_chunks: data.ingress_chunks,
         segments: realtimeState.sseSegments.slice(),
         text: data.text || latestText,
         stable_text: data.stable_text || latestText,
         partial_text: data.partial_text || "",
         live_stable_text: data.live_stable_text || "",
         live_partial_text: data.live_partial_text || "",
         live_text: data.live_text || "",
         display_text: data.display_text || "",
       });
       applyUpdate(merged);
      if (data.finalized) {
        /* The next message from the server is data: [DONE]; the
         * onmessage handler will close the connection. */
      }
    } else {
      /* Initial full snapshot. */
      realtimeState.sseLastFull = data;
      const segs = Array.isArray(data.segments) ? data.segments.slice() : [];
      realtimeState.sseSegments = segs;
      applyUpdate(data);
    }
  };
  es.onerror = () => {
    /* EventSource auto-reconnects; we don't need to do anything
     * here.  On finalize, the server sends [DONE] and we close
     * explicitly.  If the connection is lost before [DONE], the
     * next reconnect attempt will pick up where we left off (the
     * server is stateless w.r.t. SSE — it always sends the current
     * snapshot on connect). */
  };
}

async function startRealtimeCapture() {
  if (hasOfflineJob()) {
    throw new Error("离线转写进行中, 请先停止");
  }
  if (activeFeature === REALTIME_FEATURE) {
    /* Already live; re-entry is a no-op so the button click is
     * idempotent.  Without this, a fast double-click after Stop
     * would race with the cleanup of the previous session. */
    return;
  }
  /* Synchronous in-flight guard.  The previous guard above only
   * checks `activeFeature`, but `activeFeature = REALTIME_FEATURE`
   * is set AFTER the awaits below.  A second concurrent invocation
   * (e.g. fast double-click, programmatic dispatch, or any re-entry
   * path that bypasses the click handler's `realtimeStarting` flag)
   * would also pass the guard and create a second server session.
   * This flag is set synchronously here so the next concurrent call
   * hits the guard at the top and bails.  Cleared in the catch and
   * finally below. */
  if (realtimeCapturing) {
    return;
  }
  realtimeCapturing = true;

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
      /* SSE replaces the previous 300ms /status poll.  See
       * openSseStream.  Saves ~3.3 req/s per session. */
      sse: null,
      meterTimer: window.setInterval(() => updateAudioMeter(), 200),
      sending: false,
      pending: [],
      prePeak: 0,
      preRms: 0,
      postPeak: 0,
      postRms: 0,
      sampleRate: audioContext.sampleRate,
      startedAt: performance.now(),
    };
    /* Only flip activeFeature AFTER realtimeState is fully populated.
     * If anything between sessionData validation and this point throws
     * (AudioContext failure, ScriptProcessor constructor, etc.) the
     * catch below must clear activeFeature — otherwise the mutex
     * thinks realtime is live but sessionId is empty, and the UI
     * gets stuck. */
    activeFeature = REALTIME_FEATURE;
    /* Soft reset: keep the prior session's done lines (the
     * "old info preserved, new info appends" rule), stop the
     * typewriter, and append a fresh cursor for the new session. */
    softResetRealtimeArchive(sessionData.session_id);
    /* Open the SSE stream AFTER realtimeState.sessionId is set so
     * the onmessage handler can verify the session still matches
     * before touching UI state.  This drives all segment updates
     * and the meter display — no more /status polling. */
    openSseStream(sessionData.session_id);

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
      /* The server-side max_peak is now piggybacked on every
       * /api/realtime/status response (pollRealtimeStatus).  No more
       * separate /audio_diag fetch — saves ~10 req/s.  We update
       * meterSrv from the latest poll result via realtimeState.maxSrvPeak. */
      const srv = realtimeState.maxSrvPeak;
      if (typeof srv === "number") {
        meterSrv.textContent = srv.toFixed(3);
        meterSrv.style.color = srv > 0.01 ? "#0a0" : "#c00";
      }
    }

    source.connect(processor);
    processor.connect(audioContext.destination);
    updateControlAvailability();
    realtimeStatus.textContent = `会话 ${realtimeState.sessionId} 已启动`;
  } catch (error) {
    mediaStream.getTracks().forEach((track) => track.stop());
    if (audioContext) {
      await audioContext.close();
    }
    /* Cleanup: send stop to server only if we actually got a real
     * session_id.  Before that point (getUserMedia rejected, fetch
     * rejected, response not ok) the server has no session for us
     * and a stop with an undefined / error-payload session_id would
     * either 404 (best case) or create a phantom server-side entry
     * in the worst case. */
    if (sessionData && sessionResponse && sessionResponse.ok && sessionData.session_id) {
      try {
        await fetch(`/api/realtime/stop?session_id=${encodeURIComponent(sessionData.session_id)}`, {
          method: "POST",
          body: "",
        });
      } catch (_cleanupError) {
        // Best effort only; the original startup error is more important to surface.
      }
    }
    /* If we got as far as setting activeFeature = REALTIME_FEATURE
     * (just after realtimeState was populated), clear it now so the
     * mutex returns to a consistent state.  This is the only place
     * the catch can leak state — everything below is wired up. */
    if (activeFeature === REALTIME_FEATURE) {
      activeFeature = "";
    }
    realtimeCapturing = false;
    updateControlAvailability();
    throw error;
  } finally {
    /* Belt and suspenders: if the success path forgot to clear
     * realtimeCapturing (it doesn't, but defensive), this finally
     * guarantees it's released for the next click.  The catch above
     * also sets it false; this is a no-op in the error path and
     * the only release in the success path. */
    realtimeCapturing = false;
  }
}

async function stopRealtimeCapture() {
  if (!realtimeState.sessionId) {
    return;
  }
  const sessionId = realtimeState.sessionId;
  /* Tear down the audio graph first, regardless of what the stop
   * fetch does.  If any of these disconnects / close() throws
   * (e.g. partial Web Audio implementation, double-stop race),
   * we MUST still reset realtimeState and activeFeature below —
   * otherwise the user is stuck with a UI that thinks a session
   * is live when the server has already cleaned up.  Wrap the
   * whole "disconnect + fetch" block in a single try/finally. */
  try {
    window.clearInterval(realtimeState.sendTimer);
    /* Close the SSE stream so the server breaks out of its
     * wait_for and we don't keep an idle connection open.  Do
     * this before sending /stop — the server's wait predicate
     * checks `finalized` OR new segment count, but it doesn't
     * know about the client going away.  EventSource.close()
     * sends a request to the server to close. */
    if (realtimeState.sse) {
      try { realtimeState.sse.close(); } catch {}
      realtimeState.sse = null;
    }
    await flushRealtimeChunk(true);

    realtimeState.processor.disconnect();
    realtimeState.source.disconnect();
    realtimeState.mediaStream.getTracks().forEach((track) => track.stop());
    await realtimeState.audioContext.close();

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
      sending: false,
      pending: [],
      sampleRate: 0,
      startedAt: 0,
    };
    if (activeFeature === REALTIME_FEATURE) {
      activeFeature = "";
    }
    updateControlAvailability();
  }
}

startRealtime.addEventListener("click", async () => {
  /* Defensive: each guard also lives inside startRealtimeCapture, but
   * we want immediate visible feedback (button stays disabled) rather
   * than a silent no-op that the user has to debug. */
  if (realtimeStarting) {
    return;
  }
  if (realtimeStopping) {
    realtimeStatus.textContent = "正在停止后台转写, 请稍候...";
    return;
  }
  if (activeFeature === REALTIME_FEATURE) {
    return;
  }
  if (hasOfflineJob()) {
    realtimeStatus.textContent = "离线转写进行中, 请先停止";
    return;
  }
  realtimeStarting = true;
  updateControlAvailability();
  try {
    await startRealtimeCapture();
  } catch (error) {
    realtimeStatus.textContent = `启动失败：${error.message}`;
  } finally {
    realtimeStarting = false;
    /* Defer the button state to updateControlAvailability() so we
     * never re-enable a button that the state machine says should
     * stay disabled (e.g. after a successful start, activeFeature
     * is REALTIME_FEATURE so Start should remain disabled). */
    updateControlAvailability();
  }
});

stopRealtime.addEventListener("click", async () => {
  if (realtimeStopping) {
    return;
  }
  if (activeFeature !== REALTIME_FEATURE || !realtimeState.sessionId) {
    return;
  }
  realtimeStopping = true;
  /* Lock the UI immediately so a second Start click is visibly
   * disabled while /api/realtime/stop is in flight (the "wait for
   * backend inference delay" rule). */
  updateControlAvailability();
  realtimeStatus.textContent = "正在停止后台转写, 请稍候...";
  try {
    await stopRealtimeCapture();
  } catch (error) {
    realtimeStatus.textContent = `停止失败：${error.message}`;
  } finally {
    realtimeStopping = false;
    updateControlAvailability();
  }
});

clearRealtime.addEventListener("click", () => {
  /* Clear is only meaningful in the idle state.  Block the click
   * (with a status hint) if a session is live or a transition is
   * in flight, otherwise the user could lose the audio pipeline
   * state. */
  if (realtimeStarting || realtimeStopping) {
    realtimeStatus.textContent = "正在启动/停止, 请稍候...";
    return;
  }
  if (activeFeature === REALTIME_FEATURE) {
    realtimeStatus.textContent = "实时转写进行中, 请先停止";
    return;
  }
  if (hasOfflineJob()) {
    realtimeStatus.textContent = "离线转写进行中, 请先停止";
    return;
  }
  resetTranscriptFrame(realtimeResult, "尚无结果");
  resetRealtimeArchive();
  realtimeStatus.textContent = "未开始";
  updateControlAvailability();
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
