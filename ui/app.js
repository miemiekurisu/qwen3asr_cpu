// ui/app.js — QASR UI: DOM + events + rendering
// Depends on: live_monitor.js, state_pure.js, state.js, terminal.js

(function () {
  'use strict';

  // ───── DOM References ─────
  var healthBadge = document.getElementById('healthBadge');
  var runtimeHint = document.getElementById('runtimeHint');
  var uploadForm = document.getElementById('uploadForm');
  var audioFile = document.getElementById('audioFile');
  var offlineSubmit = document.getElementById('offlineSubmit');
  var offlineStop = document.getElementById('offlineStop');
  var offlineStatus = document.getElementById('offlineStatus');
  var offlineResult = document.getElementById('offlineResult');
  var startRealtime = document.getElementById('startRealtime');
  var stopRealtime = document.getElementById('stopRealtime');
  var clearRealtime = document.getElementById('clearRealtime');
  var exportRealtimeText = document.getElementById('exportRealtimeText');
  var exportRealtimeJson = document.getElementById('exportRealtimeJson');
  var exportSrt = document.getElementById('exportSrt');
  var realtimeResult = document.getElementById('realtimeResult');
  var realtimeStatus = document.getElementById('realtimeStatus');
  var audioMeter = document.getElementById('audioMeter');
  var meterPre = document.getElementById('meterPre');
  var meterPost = document.getElementById('meterPost');
  var meterSrv = document.getElementById('meterSrv');
 // New elements
   var liveCaption = document.getElementById('liveCaption');
  var captionLines = document.getElementById('captionLines');
  var transcriptBody = document.getElementById('transcriptBody');
  var domainSelect = document.getElementById('domainSelect');
  var styleSelect = document.getElementById('styleSelect');
  var glossaryBody = document.getElementById('glossaryBody');
  var addGlossaryBtn = document.getElementById('addGlossaryBtn');
  var customPrompt = document.getElementById('customPrompt');

  // ───── State (using QasrState) ─────
  var uiState = QasrState.createUIState();
  var offlineState = QasrState.createOfflineState();
  var realtimeState = QasrState.createRealtimeState();
  var archiveState = QasrState.createArchiveState();
  var glossaryState = QasrState.createGlossaryState();

  // Terminal archive (kept for backward compat with terminal.js)
  var terminalArchive = {
    lines: [],
    typewriterTimer: null,
  };

 // ───── Live Caption ─────

  function renderLiveCaption(data) {
    if (!captionLines) return;
    var segments = [];
    if (data && data.segments) {
      for (var i = 0; i < data.segments.length; i++) {
        if (typeof data.segments[i] === 'string' && data.segments[i]) {
          segments.push(data.segments[i]);
        }
      }
    }
    // Show last 3
    var start = Math.max(0, segments.length - 3);
    var html = '';
    for (var j = start; j < segments.length; j++) {
      var status = (j === segments.length - 1 && !data.finalized) ? '翻译中' : '已确认';
      html += '<span class="caption-line"><span class="source">' + escapeHtml(segments[j]) +
              '</span><span class="status-tag">[' + status + ']</span></span>';
    }
    captionLines.innerHTML = html;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ───── 3. Transcript Panel ─────

  function renderTranscriptPanel(data) {
    if (!transcriptBody) return;
    var segments = [];
    if (data && data.segments) {
      for (var i = 0; i < data.segments.length; i++) {
        if (typeof data.segments[i] === 'string' && data.segments[i]) {
          segments.push(data.segments[i]);
        }
      }
    }
    var html = '';
    var sampleRate = 16000;
    var cumulativeMs = 0;
    for (var j = 0; j < segments.length; j++) {
      var ts = QasrState.msToTimestamp(cumulativeMs);
      var statusClass = (data.finalized) ? 'status-done' : 'status-typing';
      var statusText = (data.finalized) ? '已确认' : '转写中';
      html += '<div class="transcript-row" data-seg="' + j + '">';
      html += '<span class="col-time">' + ts + '</span>';
      html += '<span class="col-text">' + escapeHtml(segments[j]) + '</span>';
      html += '<span class="col-status ' + statusClass + '">' + statusText + '</span>';
      html += '</div>';
      cumulativeMs += 3000; // Estimate 3s per segment
    }
    transcriptBody.innerHTML = html;
    // Auto-scroll
    var container = document.getElementById('transcriptContainer');
    if (container) container.scrollTop = container.scrollHeight;
  }

  // ───── 4. Glossary ─────

  function renderGlossary() {
    if (!glossaryBody) return;
    var html = '';
    for (var i = 0; i < glossaryState.entries.length; i++) {
      var e = glossaryState.entries[i];
      html += '<div class="glossary-row">';
      html += '<span>' + escapeHtml(e.source) + '</span>';
      html += '<span>' + escapeHtml(e.target) + '</span>';
      html += '<span>' + (e.lang || '-') + '</span>';
      html += '</div>';
    }
    glossaryBody.innerHTML = html;
  }

  // ───── 5. Export ─────

  function extractConfirmedText() {
    return QasrStatePure.computeConfirmedRealtimeText(terminalArchive.lines);
  }

  function exportTranscript(format) {
    var text = extractConfirmedText();
    if (!text.trim()) {
      realtimeStatus.textContent = '暂无可导出的已确定文本';
      return;
    }

    var filename = QasrState.buildExportName(realtimeState.sessionId, format);

    if (format === 'txt') {
      triggerDownload(filename, text, 'text/plain;charset=utf-8');
      realtimeStatus.textContent = '已导出 TXT';
      return;
    }

    if (format === 'json') {
      var payload = JSON.stringify({
        exported_at: new Date().toISOString(),
        session_id: realtimeState.sessionId,
        finalized: archiveState.finalized,
        confirmed_text: text,
      }, null, 2) + '\n';
      triggerDownload(filename, payload, 'application/json;charset=utf-8');
      realtimeStatus.textContent = '已导出 JSON';
      return;
    }

    if (format === 'srt') {
      var segs = [];
      for (var i = 0; i < terminalArchive.lines.length; i++) {
        if (terminalArchive.lines[i].state === 'done' && terminalArchive.lines[i].text) {
          segs.push({ text: terminalArchive.lines[i].text, sample_count: 48000 });
        }
      }
      var srt = QasrState.buildSrtFromSegments(segs, 16000);
      triggerDownload(filename, srt, 'text/plain;charset=utf-8');
      realtimeStatus.textContent = '已导出 SRT';
      return;
    }
  }

  function triggerDownload(filename, content, mimeType) {
    var blob = new Blob([content], {type: mimeType});
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 0);
  }

  function updateExportAvailability() {
    var hasText = Boolean(extractConfirmedText().trim());
    exportRealtimeText.disabled = !hasText;
    exportRealtimeJson.disabled = !hasText;
    exportSrt.disabled = !hasText;
  }

  // ───── Terminal (using QasrTerminal) ─────

  function renderTranscript(element, data, fallback) {
    if (data === null || data === undefined) {
      resetTerminal(element, fallback);
      return;
    }
    var frame = element._transcriptFrame || {};
    frame.pendingData = data;
    frame.pendingFallback = fallback;
    element._transcriptFrame = frame;
    if (frame.renderScheduled) return;
    frame.renderScheduled = true;
    requestAnimationFrame(function () {
      frame.renderScheduled = false;
      applyTranscriptRender(element, frame.pendingData, frame.pendingFallback);
    });
  }

  function applyTranscriptRender(element, data, fallback) {
    var segments = [];
    if (Array.isArray(data.segments)) {
      for (var i = 0; i < data.segments.length; i++) {
        if (typeof data.segments[i] === 'string' && data.segments[i]) {
          segments.push(data.segments[i]);
        }
      }
    }

    if (segments.length > archiveState.lastSegmentCount) {
      var newCount = segments.length - archiveState.lastSegmentCount;
      if (newCount === 1) {
        var newText = segments[segments.length - 1];
        if (newText) {
          if (data.finalized) {
            QasrTerminal.renderFinalizedSegment(element, newText, terminalArchive);
          } else {
            QasrTerminal.animateSegment(element, newText, terminalArchive);
          }
        }
      } else {
        for (var j = archiveState.lastSegmentCount; j < segments.length - 1; j++) {
          if (segments[j]) QasrTerminal.renderFinalizedSegment(element, segments[j], terminalArchive);
        }
        var lastText = segments[segments.length - 1];
        if (lastText) {
          if (data.finalized) {
            QasrTerminal.renderFinalizedSegment(element, lastText, terminalArchive);
          } else {
            QasrTerminal.animateSegment(element, lastText, terminalArchive);
          }
        }
      }
      archiveState.lastSegmentCount = segments.length;
    }

    if (data.finalized) {
      if (terminalArchive.typewriterTimer !== null) {
        clearInterval(terminalArchive.typewriterTimer);
        terminalArchive.typewriterTimer = null;
      }
      archiveState.finalized = true;
    }

    element.scrollTop = element.scrollHeight;
    // Update new panels
    renderLiveCaption(data);
    renderTranscriptPanel(data);
    updateExportAvailability();
  }

  function resetTerminal(element, fallback) {
    if (terminalArchive.typewriterTimer !== null) {
      clearInterval(terminalArchive.typewriterTimer);
      terminalArchive.typewriterTimer = null;
    }
    terminalArchive.lines = [];
    archiveState.lastSegmentCount = 0;
    archiveState.finalized = false;
    element.innerHTML = '';
    var cursorLine = QasrTerminal.makeTermLine('cursor', '', []);
    element.appendChild(cursorLine);
    terminalArchive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
    // Clear new panels
    if (captionLines) captionLines.innerHTML = '';
    if (transcriptBody) transcriptBody.innerHTML = '';
  }

  function softResetArchive(newSessionId) {
    if (terminalArchive.typewriterTimer !== null) {
      clearInterval(terminalArchive.typewriterTimer);
      terminalArchive.typewriterTimer = null;
    }
    archiveState.sessionId = newSessionId;
    archiveState.lastSegmentCount = 0;
    archiveState.finalized = false;
    archiveState.updatedAt = new Date().toISOString();

    var lastLine = terminalArchive.lines[terminalArchive.lines.length - 1];
    if (lastLine && lastLine.state === 'typing') {
      if (lastLine.text) {
        lastLine.state = 'done';
        lastLine.el.classList.remove('typing');
        lastLine.el.classList.add('done');
      } else {
        terminalArchive.lines.pop();
        if (lastLine.el && lastLine.el.parentNode) {
          lastLine.el.parentNode.removeChild(lastLine.el);
        }
      }
    }

    var tail = terminalArchive.lines[terminalArchive.lines.length - 1];
    if (!tail || tail.state !== 'cursor' || tail.text) {
      var cursorLine = QasrTerminal.makeTermLine('cursor', '', terminalArchive.lines);
      realtimeResult.appendChild(cursorLine);
      terminalArchive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
    }
    updateExportAvailability();
  }

  function resetArchive(fallback) {
    terminalArchive.lines = [];
    terminalArchive.typewriterTimer = null;
    archiveState.lastSegmentCount = 0;
    archiveState.finalized = false;
    realtimeResult.innerHTML = '';
    var cursorLine = QasrTerminal.makeTermLine('cursor', '', []);
    realtimeResult.appendChild(cursorLine);
    terminalArchive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
    if (captionLines) captionLines.innerHTML = '';
    if (transcriptBody) transcriptBody.innerHTML = '';
    updateExportAvailability();
  }

  function syncArchive(data) {
    archiveState.sessionId = (data && data.session_id) || realtimeState.sessionId || archiveState.sessionId;
    archiveState.finalized = Boolean(data && data.finalized);
    archiveState.updatedAt = new Date().toISOString();
    updateExportAvailability();
  }

  // ───── Control Availability ─────

  function hasOfflineJob() { return uiState.activeFeature === QasrState.BATCH_FEATURE; }
  function hasRealtimeSession() { return uiState.activeFeature === QasrState.REALTIME_FEATURE && realtimeState.sessionId; }

  function updateControlAvailability() {
    var offlineActive = hasOfflineJob();
    var realtimeActive = hasRealtimeSession();
    var realtimeBusy = uiState.realtimeStarting || uiState.realtimeStopping || realtimeActive;
    var canStopOffline = offlineActive && offlineState.jobId && !offlineState.stopRequested;
    var hasText = Boolean(extractConfirmedText().trim());

    audioFile.disabled = offlineActive || realtimeActive;
    offlineSubmit.disabled = offlineActive || realtimeActive;
    offlineStop.disabled = !canStopOffline;
    startRealtime.disabled = offlineActive || realtimeBusy;
    stopRealtime.disabled = !realtimeActive || uiState.realtimeStopping;
    clearRealtime.style.display = (realtimeBusy || offlineActive) ? 'none' : '';
    exportRealtimeText.disabled = realtimeBusy || offlineActive || !hasText;
    exportRealtimeJson.disabled = realtimeBusy || offlineActive || !hasText;
    exportSrt.disabled = realtimeBusy || offlineActive || !hasText;
    if (audioMeter) audioMeter.style.display = realtimeActive ? 'block' : 'none';
  }

  function resetOfflineState() {
    offlineState = QasrState.createOfflineState();
    if (uiState.activeFeature === QasrState.BATCH_FEATURE) uiState.activeFeature = '';
    updateControlAvailability();
  }

  // ───── Health ─────

  async function checkHealth() {
    try {
      var response = await fetch('/api/health');
      var data = await response.json();
      if (data.status === 'ok') {
        healthBadge.textContent = '已就绪';
        healthBadge.classList.add('ok');
        runtimeHint.textContent = '音频文件上传转写 (server ffmpeg) 与浏览器麦克风实时转写可用';
        return;
      }
    } catch (error) {
      runtimeHint.textContent = error.message;
    }
    healthBadge.textContent = '未就绪';
  }

  // ───── Offline Upload ─────

  function countCodepoints(text) { return QasrStatePure.countCodepoints(text); }

  function updateOfflineStatus(job, startTime) {
    var text = job.text || '';
    var elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    var chars = countCodepoints(text);
    var prefix = job.state === 'cancelling' ? '停止中' : '转写中';
    var tokenLabel = job.token_count > 0 ? ' / ' + job.token_count + ' tokens' : '';
    offlineStatus.textContent = chars > 0
      ? prefix + ': 已识别 ' + chars + ' 字 / ' + elapsed + 's' + tokenLabel
      : prefix + ': ' + elapsed + 's' + tokenLabel;
  }

  function showOfflineError(message) { offlineResult.textContent = '失败: ' + message; }

  async function submitOfflineViaAsync(file, startTime) {
    if (file.size > QasrState.MAX_ASYNC_UPLOAD_BYTES) {
      throw new Error('文件 ' + (file.size / 1024 / 1024).toFixed(1) + 'MB 超过 64MB 限制');
    }
    if (file.size <= 0) throw new Error('文件为空');

    offlineResult.textContent = '上传到服务器...';
    var form = new FormData();
    form.append('audio', file);
    var submitRes = await fetch('/api/transcriptions/async', { method: 'POST', body: form });
    var submitData = await submitRes.json();
    if (!submitRes.ok) throw new Error(submitData.error ? submitData.error.message : 'HTTP ' + submitRes.status);

    uiState.activeFeature = QasrState.BATCH_FEATURE;
    offlineState.jobId = submitData.id;
    offlineState.stopRequested = false;
    updateControlAvailability();
    offlineStatus.textContent = '提交成功, 转写中...';

    var lastTextLen = 0;
    var pollAttempts = 0;
    while (pollAttempts < 6000) {
      pollAttempts++;
      await new Promise(function (r) { setTimeout(r, 300); });
      var pollRes = await fetch('/api/jobs/' + encodeURIComponent(submitData.id));
      var job = await pollRes.json();
      if (!pollRes.ok) throw new Error(job.error ? job.error.message : 'HTTP ' + pollRes.status);

      if (job.state === 'running' || job.state === 'queued' || job.state === 'cancelling') {
        var text = job.text || '';
        if (text.length > lastTextLen) { lastTextLen = text.length; offlineResult.textContent = text; }
        if (!offlineState.stopRequested && !offlineState.stopError) {
          updateOfflineStatus(job, startTime);
        }
        continue;
      }
      if (job.state === 'cancelled') {
        offlineState.stopError = '';
        offlineResult.textContent = job.text || '已停止';
        offlineStatus.textContent = '已停止';
        return;
      }
      if (job.state === 'failed') {
        offlineState.stopError = '';
        throw new Error(job.error || '转写失败');
      }
      // completed
      offlineState.stopError = '';
      offlineResult.textContent = job.text || '';
      var audioDur = (job.audio_ms / 1000).toFixed(1);
      var infMs = (job.inference_ms || 0).toFixed(0);
      var rtf = job.audio_ms > 0 ? (job.inference_ms / job.audio_ms).toFixed(2) : '-';
      offlineStatus.textContent = '音频 ' + audioDur + 's / 推理 ' + infMs + 'ms / RTF ' + rtf + ' / ' + (job.tokens || 0) + ' tokens';
      return;
    }
    throw new Error('转写超时 (>30min)');
  }

  // ───── Realtime Capture ─────

  async function flushRealtimeChunk(force) {
    if (!realtimeState.sessionId || realtimeState.sending) return;
    if (!force && (!realtimeState.pending || realtimeState.pending.length === 0)) return;

    var chunks = realtimeState.pending;
    realtimeState.pending = [];
    var total = 0;
    for (var i = 0; i < chunks.length; i++) total += chunks[i].length;
    if (total === 0) return;

    var buffer = new Int16Array(total);
    var offset = 0;
    for (var j = 0; j < chunks.length; j++) {
      buffer.set(chunks[j], offset);
      offset += chunks[j].length;
    }

    realtimeState.sending = true;
    try {
      var response = await fetch('/api/realtime/chunk?session_id=' + encodeURIComponent(realtimeState.sessionId), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: buffer.buffer,
      });
      var data = await response.json();
      if (!response.ok) throw new Error(data.error ? data.error.message : 'HTTP ' + response.status);
      renderTranscript(realtimeResult, data, '尚无结果');
      syncArchive(data);
      if (typeof data.max_ingress_peak === 'number') realtimeState.maxSrvPeak = data.max_ingress_peak;
      else if (typeof data.max_peak === 'number') realtimeState.maxSrvPeak = data.max_peak;
      var audioDur = (data.sample_count / 16000).toFixed(1);
      var decodedDur = (data.decoded_samples / 16000).toFixed(1);
      var wallElapsed = ((performance.now() - realtimeState.startedAt) / 1000).toFixed(1);
      var lag = (wallElapsed - decodedDur).toFixed(1);
      var infMs = (data.inference_ms !== undefined) ? data.inference_ms.toFixed(0) : '-';
      var decodeLabel = data.decoded ? '已解码' : '待下轮';
      realtimeStatus.textContent = '音频 ' + audioDur + 's / 已解码 ' + decodedDur + 's / 耗时 ' + wallElapsed + 's / 滞后 ' + lag + 's / 推理 ' + infMs + 'ms / ' + decodeLabel +
        ' | mic 峰 ' + (realtimeState.prePeak || 0).toFixed(3) + ' → 16k 峰 ' + (realtimeState.postPeak || 0).toFixed(3);
    } catch (error) {
      realtimeStatus.textContent = '失败：' + error.message;
    } finally {
      realtimeState.sending = false;
    }
  }

  // ───── SSE Stream ─────

  function openSseStream(sessionId) {
    if (!realtimeState || realtimeState.sessionId !== sessionId) return;

    function applyUpdate(data) {
      if (!realtimeState || realtimeState.sessionId !== sessionId) return;
      if (typeof data.max_ingress_peak === 'number') realtimeState.maxSrvPeak = data.max_ingress_peak;
      else if (typeof data.max_peak === 'number') realtimeState.maxSrvPeak = data.max_peak;
      renderTranscript(realtimeResult, data, '尚无结果');
      syncArchive(data);
      var audioDur = (data.sample_count / 16000).toFixed(1);
      var decodedDur = (data.decoded_samples / 16000).toFixed(1);
      var wallElapsed = ((performance.now() - realtimeState.startedAt) / 1000).toFixed(1);
      var lag = (wallElapsed - decodedDur).toFixed(1);
      var infMs = (data.inference_ms !== undefined) ? data.inference_ms.toFixed(0) : '-';
      var decodeLabel = data.decoded ? '已解码' : '待下轮';
      realtimeStatus.textContent = '音频 ' + audioDur + 's / 已解码 ' + decodedDur + 's / 耗时 ' + wallElapsed + 's / 滞后 ' + lag + 's / 推理 ' + infMs + 'ms / ' + decodeLabel +
        ' | mic 峰 ' + (realtimeState.prePeak || 0).toFixed(3) + ' → 16k 峰 ' + (realtimeState.postPeak || 0).toFixed(3);
    }

    var es;
    try { es = new EventSource('/api/realtime/stream?session_id=' + encodeURIComponent(sessionId)); }
    catch (err) { realtimeStatus.textContent = 'SSE 失败：' + err.message; return; }
    realtimeState.sse = es;
    realtimeState.sseSegments = [];
    realtimeState.sseLastFull = null;

    es.onmessage = function (ev) {
      if (!realtimeState || realtimeState.sessionId !== sessionId) return;
      if (ev.data === '[DONE]') { es.close(); if (realtimeState) realtimeState.sse = null; return; }
      var data;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'update') {
        if (!realtimeState.sseLastFull) return;
        if (Array.isArray(data.new_segments) && data.new_segments.length > 0) {
          for (var i = 0; i < data.new_segments.length; i++) {
            if (typeof data.new_segments[i] === 'string' && data.new_segments[i].length > 0) {
              realtimeState.sseSegments.push(data.new_segments[i]);
            }
          }
        }
        var latest = realtimeState.sseSegments.length > 0 ? realtimeState.sseSegments[realtimeState.sseSegments.length - 1] : '';
        var merged = Object.assign({}, realtimeState.sseLastFull, {
          sample_count: data.total_samples, decoded_samples: data.decoded_samples,
          inference_ms: data.last_inference_ms, segments: realtimeState.sseSegments.slice(),
          text: data.text || latest, stable_text: data.stable_text || latest,
          partial_text: data.partial_text || '',
        });
        applyUpdate(merged);
      } else {
        realtimeState.sseLastFull = data;
        realtimeState.sseSegments = Array.isArray(data.segments) ? data.segments.slice() : [];
        applyUpdate(data);
      }
    };
    es.onerror = function () {}; // Auto-reconnect
  }

  async function startRealtimeCapture() {
    if (hasOfflineJob()) throw new Error('离线转写进行中, 请先停止');
    if (uiState.activeFeature === QasrState.REALTIME_FEATURE) return;
    if (uiState.realtimeCapturing) return;
    uiState.realtimeCapturing = true;

    var mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    var sessionResponse = await fetch('/api/realtime/start', { method: 'POST', body: '' });
    var sessionData = await sessionResponse.json();
    if (!sessionResponse.ok) {
      mediaStream.getTracks().forEach(function (t) { t.stop(); });
      throw new Error(sessionData.error ? sessionData.error.message : '启动失败');
    }

    var audioContext = null;
    try {
      audioContext = new AudioContext();
      var source = audioContext.createMediaStreamSource(mediaStream);
      var processor = audioContext.createScriptProcessor(4096, 1, 1);

      realtimeState = QasrState.createRealtimeState();
      realtimeState.audioContext = audioContext;
      realtimeState.source = source;
      realtimeState.processor = processor;
      realtimeState.mediaStream = mediaStream;
      realtimeState.sessionId = sessionData.session_id;
      realtimeState.sendTimer = window.setInterval(function () { flushRealtimeChunk(false); }, 400);
      realtimeState.sse = null;
      realtimeState.meterTimer = window.setInterval(function () { updateAudioMeter(); }, 200);
      realtimeState.sending = false;
      realtimeState.pending = [];
      realtimeState.sampleRate = audioContext.sampleRate;
      realtimeState.startedAt = performance.now();

      uiState.activeFeature = QasrState.REALTIME_FEATURE;
      softResetArchive(sessionData.session_id);
      openSseStream(sessionData.session_id);

      processor.onaudioprocess = function (event) {
        var channel = event.inputBuffer.getChannelData(0);
        var pre_peak = 0, pre_sum = 0;
        for (var i = 0; i < channel.length; i++) {
          var a = channel[i] < 0 ? -channel[i] : channel[i];
          if (a > pre_peak) pre_peak = a;
          pre_sum += channel[i] * channel[i];
        }
        realtimeState.prePeak = Math.max(realtimeState.prePeak || 0, pre_peak);
        realtimeState.preRms = Math.sqrt(pre_sum / channel.length);
        var downsampled = QasrStatePure.downsampleTo16k(channel, realtimeState.sampleRate);
        var pcm = QasrStatePure.floatToPcm16(downsampled);
        var post_peak = 0, post_sum = 0;
        for (var j = 0; j < downsampled.length; j++) {
          var b = downsampled[j] < 0 ? -downsampled[j] : downsampled[j];
          if (b > post_peak) post_peak = b;
          post_sum += downsampled[j] * downsampled[j];
        }
        realtimeState.postPeak = Math.max(realtimeState.postPeak || 0, post_peak);
        realtimeState.postRms = Math.sqrt(post_sum / downsampled.length);
        realtimeState.pending.push(pcm);
      };

      function updateAudioMeter() {
        if (!realtimeState) return;
        var pre = realtimeState.prePeak || 0;
        var post = realtimeState.postPeak || 0;
        if (meterPre) { meterPre.textContent = pre.toFixed(3); meterPre.style.color = pre > 0.01 ? '#0a0' : '#c00'; }
        if (meterPost) { meterPost.textContent = post.toFixed(3); meterPost.style.color = post > 0.01 ? '#0a0' : '#c00'; }
        var srv = realtimeState.maxSrvPeak;
        if (meterSrv && typeof srv === 'number') { meterSrv.textContent = srv.toFixed(3); meterSrv.style.color = srv > 0.01 ? '#0a0' : '#c00'; }
      }

      source.connect(processor);
      processor.connect(audioContext.destination);
      updateControlAvailability();
      realtimeStatus.textContent = '会话 ' + realtimeState.sessionId + ' 已启动';
    } catch (error) {
      mediaStream.getTracks().forEach(function (t) { t.stop(); });
      if (audioContext) await audioContext.close();
      if (sessionData && sessionData.session_id) {
        try { await fetch('/api/realtime/stop?session_id=' + encodeURIComponent(sessionData.session_id), { method: 'POST', body: '' }); } catch {}
      }
      if (uiState.activeFeature === QasrState.REALTIME_FEATURE) uiState.activeFeature = '';
      uiState.realtimeCapturing = false;
      updateControlAvailability();
      throw error;
    } finally {
      uiState.realtimeCapturing = false;
    }
  }

  async function stopRealtimeCapture() {
    if (!realtimeState.sessionId) return;
    var sessionId = realtimeState.sessionId;
    try {
      window.clearInterval(realtimeState.sendTimer);
      if (realtimeState.sse) { try { realtimeState.sse.close(); } catch {} realtimeState.sse = null; }
      await flushRealtimeChunk(true);
      realtimeState.processor.disconnect();
      realtimeState.source.disconnect();
      realtimeState.mediaStream.getTracks().forEach(function (t) { t.stop(); });
      await realtimeState.audioContext.close();
      var response = await fetch('/api/realtime/stop?session_id=' + encodeURIComponent(sessionId), { method: 'POST', body: '' });
      var data = await response.json();
      var prevStats = realtimeStatus.textContent;
      if (response.ok) {
        renderTranscript(realtimeResult, data, '尚无结果');
        syncArchive(data);
        var stopLabel = data.text ? '已停止，终稿已出' : '已停止';
        realtimeStatus.textContent = prevStats ? (prevStats + ' / ' + stopLabel) : ('会话 ' + sessionId + ' ' + stopLabel);
      } else {
        realtimeStatus.textContent = (data.error ? data.error.message : '停止失败');
      }
    } finally {
      realtimeState = QasrState.createRealtimeState();
      if (uiState.activeFeature === QasrState.REALTIME_FEATURE) uiState.activeFeature = '';
      archiveState.sessionId = '';
      updateControlAvailability();
    }
  }

  // ───── Event Handlers ─────

  uploadForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    if (hasRealtimeSession()) { showOfflineError('实时转写进行中, 请先停止'); return; }
    if (hasOfflineJob()) return;
    var file = audioFile.files[0];
    if (!file) { showOfflineError('请先选择音频文件'); return; }
    uiState.activeFeature = QasrState.BATCH_FEATURE;
    offlineState.startedAt = performance.now();
    offlineState.stopRequested = false;
    updateControlAvailability();
    offlineResult.textContent = '上传到服务器...';
    offlineStatus.textContent = '上传中...';
    try { await submitOfflineViaAsync(file, offlineState.startedAt); }
    catch (error) { showOfflineError(error.message); offlineStatus.textContent = ''; }
    finally { resetOfflineState(); }
  });

  offlineStop.addEventListener('click', async function () {
    if (!hasOfflineJob() || !offlineState.jobId) return;
    offlineState.stopRequested = true;
    updateControlAvailability();
    offlineStatus.textContent = '停止中...';
    try {
      var response = await fetch('/api/jobs/' + encodeURIComponent(offlineState.jobId) + '/cancel', { method: 'POST', body: '' });
      var data = await response.json();
      if (!response.ok) throw new Error(data.error ? data.error.message : '停止失败');
      offlineState.stopRequested = false;
    } catch (error) {
      offlineState.stopRequested = false;
      offlineState.stopError = error.message;
      offlineStatus.textContent = '停止失败: ' + error.message;
    }
  });

  startRealtime.addEventListener('click', async function () {
    if (uiState.realtimeStarting || uiState.realtimeStopping) return;
    if (uiState.activeFeature === QasrState.REALTIME_FEATURE) return;
    if (hasOfflineJob()) { realtimeStatus.textContent = '离线转写进行中, 请先停止'; return; }
    uiState.realtimeStarting = true;
    updateControlAvailability();
    try { await startRealtimeCapture(); }
    catch (error) { realtimeStatus.textContent = '启动失败：' + error.message; }
    finally { uiState.realtimeStarting = false; updateControlAvailability(); }
  });

  stopRealtime.addEventListener('click', async function () {
    if (uiState.realtimeStopping) return;
    if (uiState.activeFeature !== QasrState.REALTIME_FEATURE || !realtimeState.sessionId) return;
    uiState.realtimeStopping = true;
    updateControlAvailability();
    realtimeStatus.textContent = '正在停止后台转写, 请稍候...';
    try { await stopRealtimeCapture(); }
    catch (error) { realtimeStatus.textContent = '停止失败：' + error.message; }
    finally { uiState.realtimeStopping = false; updateControlAvailability(); }
  });

  clearRealtime.addEventListener('click', function () {
    if (uiState.realtimeStarting || uiState.realtimeStopping) { realtimeStatus.textContent = '正在启动/停止, 请稍候...'; return; }
    if (uiState.activeFeature === QasrState.REALTIME_FEATURE) { realtimeStatus.textContent = '实时转写进行中, 请先停止'; return; }
    if (hasOfflineJob()) { realtimeStatus.textContent = '离线转写进行中, 请先停止'; return; }
    resetTerminal(realtimeResult, '尚无结果');
    resetArchive();
    realtimeStatus.textContent = '未开始';
    updateControlAvailability();
  });

  exportRealtimeText.addEventListener('click', function () { exportTranscript('txt'); });
  exportRealtimeJson.addEventListener('click', function () { exportTranscript('json'); });
  exportSrt.addEventListener('click', function () { exportTranscript('srt'); });

  domainSelect.addEventListener('change', function () {
    glossaryState.domain = domainSelect.value;
  });

  customPrompt.addEventListener('change', function () {
    glossaryState.customPrompt = customPrompt.value;
  });

  // ───── Init ─────

  async function loadServerInfo() {
    try {
      var resp = await fetch('/api/metrics');
      var data = await resp.json();
      if (data.backend) uiState.serverBackend = data.backend;
    } catch (e) {
      uiState.serverBackend = 'cpu';
    }
  }

  // Pick up initial cursor line
  var initial = realtimeResult.querySelector('.term-line.cursor');
  if (initial) terminalArchive.lines.push({ state: 'cursor', el: initial, text: '' });

  loadServerInfo();
  updateControlAvailability();
  resetArchive();
  renderGlossary();
  checkHealth();

})();
