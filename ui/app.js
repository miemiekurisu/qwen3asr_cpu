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
   var transcriptBody = document.getElementById('transcriptBody');
  // Translation elements
  var translationSourceLang = document.getElementById('translationSourceLang');
  var translationTargetLang = document.getElementById('translationTargetLang');
  var translationStatus = document.getElementById('translationStatus');
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

  // ───── Translation State ─────
  var runtimeConfig = window.QASR_RUNTIME_CONFIG || {};
  var translationConfig = runtimeConfig.translation || {};
  var translationState = {
    enabled: translationConfig.enabled !== false,
    endpoint:
      translationConfig.endpoint ||
      localStorage.getItem('qasrTranslationEndpoint') ||
      'http://127.0.0.1:8989',
    sourceLang:
      translationConfig.sourceLang ||
      localStorage.getItem('qasrTranslationSourceLang') ||
      'auto',
    targetLang:
      translationConfig.targetLang ||
      localStorage.getItem('qasrTranslationTargetLang') ||
      'en',
    timeoutMs: translationConfig.timeoutMs || 3000,
    byIndex: {},
    pending: {},
  };

  // Terminal archive (kept for backward compat with terminal.js)
  var terminalArchive = {
    lines: [],
    typewriterTimer: null,
  };

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/'/g, '&#39;').replace(/"/g, '&quot;');
  }

  function safeFixed(value, decimals) {
    return (typeof value === 'number' && !isNaN(value)) ? value.toFixed(decimals) : '-';
  }

  // ───── 3. Transcript Panel ─────
  var transcriptCache = '';

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
    var candidates = [];
    if (data && data.candidates) {
      for (var i = 0; i < data.candidates.length; i++) {
        if (typeof data.candidates[i] === 'string' && data.candidates[i]) {
          candidates.push(data.candidates[i]);
        }
      }
    }
    var bodyHtml = '';
    var sampleRate = 16000;
    var segmentSamples = [];
    if (data && Array.isArray(data.segmentSamples)) {
      segmentSamples = data.segmentSamples;
    }
    var isFinalSession = data && (data.finalized || data.event_type === 'transcript.final');
  /* Each committed segment is a separate row with timestamp from
      * real audio sample position (tracked per-segment at SSE receive time).
      * Render in reverse chronological order: latest segment first. */
    for (var j = segments.length - 1; j >= 0; j--) {
      var samplePos = segmentSamples[j] || 0;
      var ts = QasrState.msToTimestamp(samplePos * 1000 / sampleRate);

      /* Build text column: original + translation (if available). */
      var tr = translationState.byIndex[j];
      var textHtml = '';
      textHtml += '<div class="translation-original">' + escapeHtml(segments[j]) + '</div>';

      var statusText = '已确认';
      var statusClass = 'status-done';

      if (tr && tr.ok) {
        textHtml += '<div class="translation-text">'
          + escapeHtml(tr.target || '')
          + ': '
          + escapeHtml(tr.text || '')
          + '</div>';
        statusText = '已翻译';
      } else if (tr && !tr.ok) {
        textHtml += '<div class="translation-error">翻译失败：'
          + escapeHtml(tr.error || '')
          + '</div>';
        statusText = '翻译失败';
        statusClass = 'status-typing';
      } else {
        textHtml += '<div class="translation-pending">译文等待中…</div>';
        statusText = '翻译中';
        statusClass = 'status-typing';
      }

      bodyHtml += '<div class="transcript-row done">';
      bodyHtml += '<span class="col-time">' + ts + '</span>';
      bodyHtml += '<span class="col-text">' + textHtml + '</span>';
      bodyHtml += '<span class="col-status ' + statusClass + '">' + statusText + '</span>';
      bodyHtml += '</div>';
    }
    if (candidates.length > 0) {
      var candSamplePos = segmentSamples.length > 0 ? segmentSamples[segmentSamples.length - 1] : 0;
      var candTs = QasrState.msToTimestamp(candSamplePos * 1000 / sampleRate);
      bodyHtml += '<div class="transcript-row typing">';
      bodyHtml += '<span class="col-time">' + candTs + '</span>';
      bodyHtml += '<span class="col-text">' + escapeHtml(candidates.join('')) + '</span>';
      bodyHtml += '<span class="col-status status-typing">识别中</span>';
      bodyHtml += '</div>';
    }
    if (bodyHtml !== transcriptCache) {
      transcriptCache = bodyHtml;
      transcriptBody.innerHTML = bodyHtml;
      var container = document.getElementById('transcriptContainer');
      if (container) container.scrollTop = container.scrollHeight;
    }
  }

  // ───── 4. Live Caption (top panel) ─────

  var captionText = document.getElementById('captionText');

  function renderLiveCaption(data) {
    if (!captionText) return;
    if (!data) { captionText.textContent = ''; return; }
    var text = data.live_text || data.text || data.stable_text || '';
    if (text) {
      captionText.textContent = text;
    }
  }

  // ───── 4.5 Translation ─────

  /* Core translate call — no dedup, no status updates.
     Used by stop handler for guaranteed await-able translation. */
  async function doTranslateOne(index, text) {
    var source = translationState.sourceLang || 'auto';
    var target = translationState.targetLang || 'en';

    /* Same source and target — no translation needed. */
    if (source !== 'auto' && source === target) {
      translationState.byIndex[index] = {
        ok: true, text: text,
        source: source, target: target, latencyMs: 0
      };
      return;
    }

    var started = performance.now();

    try {
      var controller = null;
      var timeoutId = null;
      if (typeof AbortController !== 'undefined') {
        controller = new AbortController();
        timeoutId = setTimeout(function () { controller.abort(); }, translationState.timeoutMs || 3000);
      }

      var response = await fetch('/api/translation/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller ? controller.signal : undefined,
        body: JSON.stringify({
          from: source, to: target, text: text, html: false
        })
      });

      if (timeoutId !== null) clearTimeout(timeoutId);
      var bodyText = await response.text();

      if (!response.ok) {
        throw new Error('HTTP ' + response.status + ': ' + bodyText.slice(0, 160));
      }

      var parsed;
      try { parsed = JSON.parse(bodyText); } catch (_e) { parsed = bodyText; }

      var translated = '';
      if (typeof parsed === 'string') {
        translated = parsed;
      } else if (parsed) {
        translated = parsed.result || parsed.text || parsed.translation ||
                     parsed.translated_text || parsed.translatedText || '';
      }
      if (!translated) translated = bodyText;

      translationState.byIndex[index] = {
        ok: true, text: translated,
        source: source, target: target,
        latencyMs: performance.now() - started
      };
    } catch (error) {
      translationState.byIndex[index] = {
        ok: false, text: '', error: error.message,
        source: source, target: target,
        latencyMs: performance.now() - started
      };
    }
  }

  /* SSE-path translation with dedup + status updates.
     Calls doTranslateOne for the actual fetch; adds pending dedup
     and re-renders the transcript panel. */
  async function scheduleTranslationForSegment(index, text) {
    if (!translationState.enabled) return;
    if (!text || !text.trim()) return;

    var source = translationState.sourceLang || 'auto';
    var target = translationState.targetLang || 'en';

    /* Same source and target — no translation needed. */
    if (source !== 'auto' && source === target) {
      translationState.byIndex[index] = {
        ok: true, text: text,
        source: source, target: target, latencyMs: 0
      };
      return;
    }

    var key = index + ':' + source + ':' + target + ':' + text;
    if (translationState.pending[key]) return;
    translationState.pending[key] = true;

    if (translationStatus) {
      translationStatus.textContent = '翻译：请求中';
    }

    try {
      await doTranslateOne(index, text);
      if (translationStatus && translationState.byIndex[index] && translationState.byIndex[index].ok) {
        translationStatus.textContent = '翻译：已连接 MTranServer';
      } else if (translationStatus) {
        translationStatus.textContent = '翻译：失败';
      }
    } finally {
      delete translationState.pending[key];
      /* Re-render transcript panel with updated translation. */
      transcriptCache = '';
      if (realtimeState && realtimeState.sseSegments) {
        renderTranscriptPanel({
          segments: realtimeState.sseSegments.slice(),
          candidates: realtimeState.sseCandidates ? realtimeState.sseCandidates.slice() : [],
          segmentSamples: realtimeState.sseSegmentSamples ? realtimeState.sseSegmentSamples.slice() : [],
          event_type: 'translation.final'
        });
      }
    }
 }

  function clearTranslationCacheAndRerender() {
    translationState.byIndex = {};
    translationState.pending = {};
    transcriptCache = '';

    if (realtimeState && Array.isArray(realtimeState.sseSegments)) {
      for (var i = 0; i < realtimeState.sseSegments.length; i++) {
        scheduleTranslationForSegment(i, realtimeState.sseSegments[i]);
      }
      renderTranscriptPanel({
        segments: realtimeState.sseSegments.slice(),
        candidates: realtimeState.sseCandidates ? realtimeState.sseCandidates.slice() : [],
        segmentSamples: realtimeState.sseSegmentSamples ? realtimeState.sseSegmentSamples.slice() : [],
        event_type: 'translation.refresh'
      });
    }
  }

  // ───── 5. Glossary ─────

  function renderGlossary() {
    if (!glossaryBody) return;
    var html = '';
    for (var i = 0; i < glossaryState.entries.length; i++) {
      var e = glossaryState.entries[i];
      html += '<div class="glossary-row">';
      html += '<span>' + escapeHtml(e.source) + '</span>';
      html += '<span>' + escapeHtml(e.target) + '</span>';
      html += '<span>' + escapeHtml(e.lang || '-') + '</span>';
      html += '</div>';
    }
    glossaryBody.innerHTML = html;
  }

  // ───── 5. Export ─────

  function extractConfirmedText() {
    /* P1: Only export confirmed (finalized) segments, not tentative candidates.
     * Use the SSE state's sseSegments which only contains confirmed text. */
    if (realtimeState && Array.isArray(realtimeState.sseSegments)) {
      var confirmed = realtimeState.sseSegments.filter(function(s) {
        return typeof s === 'string' && s.trim();
      });
      if (confirmed.length > 0) return confirmed.join(' ');
    }
    /* Fallback: use terminalArchive for compatibility with legacy sessions. */
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
      /* P1: Only export confirmed segments. */
      if (realtimeState && Array.isArray(realtimeState.sseSegments)) {
        for (var i = 0; i < realtimeState.sseSegments.length; i++) {
          if (typeof realtimeState.sseSegments[i] === 'string' && realtimeState.sseSegments[i]) {
            segs.push({ text: realtimeState.sseSegments[i], sample_count: 48000 });
          }
        }
      } else {
        for (var i = 0; i < terminalArchive.lines.length; i++) {
          if (terminalArchive.lines[i].state === 'done' && terminalArchive.lines[i].text) {
            segs.push({ text: terminalArchive.lines[i].text, sample_count: 48000 });
          }
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
    if (!element) return;
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
      try {
        frame.renderScheduled = false;
        applyTranscriptRender(element, frame.pendingData, frame.pendingFallback);
      } catch (_e) { /* rAF 回调内异常不杀死后续 rAF */ }
    });
  }

  function applyTranscriptRender(element, data, fallback) {
    if (!element || !element._transcriptFrame) return;
    /* Combine confirmed segments (final) with tentative candidates.
     * Confirmed segments are rendered as final; candidates animated. */
    var segments = [];
    if (Array.isArray(data.segments)) {
      for (var i = 0; i < data.segments.length; i++) {
        if (typeof data.segments[i] === 'string' && data.segments[i]) {
          segments.push(data.segments[i]);
        }
      }
    }
    var candidates = [];
    if (Array.isArray(data.candidates)) {
      for (var i = 0; i < data.candidates.length; i++) {
        if (typeof data.candidates[i] === 'string' && data.candidates[i]) {
          candidates.push(data.candidates[i]);
        }
      }
    }
    var allSegments = segments.concat(candidates);

    /* Retranscription (reconciled): replace the entire terminal
     * display with the new high-quality result.  The old VAD
     * segments and candidates are stale and discarded. */
    if (data.reconciled) {
      resetTerminal(element, fallback);
      /* Reset lastSegmentCount so the < 1-segment render below
       * treats the reconciled segments as new. */
      archiveState.lastSegmentCount = 0;
      /* Fall through: the segment rendering loop will render the
       * reconciled text as finalized segments. */
    }

    if (allSegments.length > archiveState.lastSegmentCount) {
      var newCount = allSegments.length - archiveState.lastSegmentCount;
      var isFinal = data.finalized || data.event_type === 'transcript.final';
      if (newCount === 1) {
        var newText = allSegments[allSegments.length - 1];
        if (newText) {
          /* Determine if this segment is confirmed or tentative.
           * Segments from data.segments are confirmed; candidates are tentative. */
          var isConfirmed = (allSegments.length - 1) < segments.length;
          if (isFinal || isConfirmed) {
            QasrTerminal.renderFinalizedSegment(element, newText, terminalArchive);
          } else {
            QasrTerminal.animateSegment(element, newText, terminalArchive);
          }
        }
      } else {
        for (var j = archiveState.lastSegmentCount; j < allSegments.length - 1; j++) {
          if (allSegments[j]) {
            if (j < segments.length || isFinal) {
              QasrTerminal.renderFinalizedSegment(element, allSegments[j], terminalArchive);
            } else {
              QasrTerminal.animateSegment(element, allSegments[j], terminalArchive);
            }
          }
        }
        var lastText = allSegments[allSegments.length - 1];
        if (lastText) {
          var lastIsConfirmed = (allSegments.length - 1) < segments.length;
          if (isFinal || lastIsConfirmed) {
            QasrTerminal.renderFinalizedSegment(element, lastText, terminalArchive);
          } else {
            QasrTerminal.animateSegment(element, lastText, terminalArchive);
          }
        }
      }
      archiveState.lastSegmentCount = allSegments.length;
    }

    if (data.finalized) {
      if (terminalArchive.typewriterTimer !== null) {
        clearInterval(terminalArchive.typewriterTimer);
        terminalArchive.typewriterTimer = null;
      }
      archiveState.finalized = true;
      /* Session finalized: promote any remaining candidate lines to done
       * so they become eligible for export.  This matches the server's
       * behavior of force-finalizing candidates into segments_text. */
      QasrTerminal.promoteCandidateToDone(terminalArchive);
    }

    element.scrollTop = element.scrollHeight;
    // Update new panels
    renderLiveCaption(data);
    renderTranscriptPanel(data);
  }

  function resetTerminal(element, fallback) {
    if (!element) return;
    element._transcriptFrame = null;
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
    transcriptCache = '';
    if (transcriptBody) transcriptBody.innerHTML = '';
  }

  function softResetArchive(newSessionId) {
    if (!realtimeResult) return;
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
    if (!realtimeResult) return;
    terminalArchive.lines = [];
    terminalArchive.typewriterTimer = null;
    archiveState.lastSegmentCount = 0;
    archiveState.finalized = false;
    realtimeResult.innerHTML = '';
    var cursorLine = QasrTerminal.makeTermLine('cursor', '', []);
    realtimeResult.appendChild(cursorLine);
    terminalArchive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
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
      var audioDur = safeFixed(job.audio_ms / 1000, 1);
      var infMs = (job.inference_ms || 0).toFixed(0);
      var rtf = (typeof job.audio_ms === 'number' && job.audio_ms > 0) ? safeFixed(job.inference_ms / job.audio_ms, 2) : '-';
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
    var sessionIdAtSend = realtimeState.sessionId;
    try {
      var response = await fetch('/api/realtime/chunk?session_id=' + encodeURIComponent(sessionIdAtSend), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: buffer.buffer,
      });
      var data = await response.json();
      if (!response.ok) throw new Error(data.error ? data.error.message : 'HTTP ' + response.status);
      if (realtimeState.sessionId !== sessionIdAtSend) return;
      renderTranscript(realtimeResult, data, '尚无结果');
      syncArchive(data);
      if (typeof data.max_ingress_peak === 'number') realtimeState.maxSrvPeak = data.max_ingress_peak;
      else if (typeof data.max_peak === 'number') realtimeState.maxSrvPeak = data.max_peak;
      var audioDur = safeFixed(data.sample_count / 16000, 1);
      var decodedDur = safeFixed(data.decoded_samples / 16000, 1);
      var wallElapsed = safeFixed((performance.now() - realtimeState.startedAt) / 1000, 1);
      var lag = safeFixed(
        typeof data.decoded_samples === 'number' ? (performance.now() - realtimeState.startedAt) / 1000 - data.decoded_samples / 16000 : undefined,
        1
      );
      var infMs = (data.inference_ms !== undefined) ? safeFixed(data.inference_ms, 0) : '-';
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
      var audioDur = safeFixed(data.sample_count / 16000, 1);
      var decodedDur = safeFixed(data.decoded_samples / 16000, 1);
      var wallElapsed = safeFixed((performance.now() - realtimeState.startedAt) / 1000, 1);
      var lag = safeFixed(
        typeof data.decoded_samples === 'number' ? (performance.now() - realtimeState.startedAt) / 1000 - data.decoded_samples / 16000 : undefined,
        1
      );
      var infMs = (data.inference_ms !== undefined) ? safeFixed(data.inference_ms, 0) : '-';
      var decodeLabel = data.decoded ? '已解码' : '待下轮';
      realtimeStatus.textContent = '音频 ' + audioDur + 's / 已解码 ' + decodedDur + 's / 耗时 ' + wallElapsed + 's / 滞后 ' + lag + 's / 推理 ' + infMs + 'ms / ' + decodeLabel +
        ' | mic 峰 ' + (realtimeState.prePeak || 0).toFixed(3) + ' → 16k 峰 ' + (realtimeState.postPeak || 0).toFixed(3);
    }

    if (realtimeState.sse) { try { realtimeState.sse.close(); } catch {} realtimeState.sse = null; }
    var es;
    try { es = new EventSource('/api/realtime/stream?session_id=' + encodeURIComponent(sessionId)); }
    catch (err) { realtimeStatus.textContent = 'SSE 失败：' + err.message; return; }
    realtimeState.sse = es;
    realtimeState.sseSegments = [];
    realtimeState.sseCandidates = [];
    realtimeState.sseSegmentSamples = [];
    realtimeState.sseLastFull = null;

    es.onmessage = function (ev) {
      if (!realtimeState || realtimeState.sessionId !== sessionId) return;
      if (ev.data === '[DONE]') { es.close(); if (realtimeState) realtimeState.sse = null; return; }
      var data;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'update') {
        if (!realtimeState.sseLastFull) return;
      if (data.reconciled) {
           /* Retranscription result: REPLACE everything with the new
            * single high-quality segment.  Old candidates and segments
            * are stale — the batch decoder re-decoded the full audio. */
           realtimeState.sseSegments = Array.isArray(data.new_segments)
             ? data.new_segments.slice() : [];
           realtimeState.sseCandidates = [];
           /* Re-translate reconciled segments. */
           translationState.byIndex = {};
           translationState.pending = {};
           for (var ri = 0; ri < realtimeState.sseSegments.length; ri++) {
             scheduleTranslationForSegment(ri, realtimeState.sseSegments[ri]);
           }
          } else if (data.partial_version && !data.new_segments && !data.new_candidates) {
            /* Live partial update (no new VAD segment yet).  Show the
             * current ASR live text in the caption panel only.  Don't
             * touch the transcript panel — it updates only on VAD
             * segment events (transcript.candidate / transcript.final). */
            var live = data.live_text || '';
            if (live && typeof renderLiveCaption === 'function') {
              var partialData = {
                segments: realtimeState.sseSegments.concat(realtimeState.sseCandidates).concat([live]),
                text: live,
                stable_text: data.live_stable_text || '',
                partial_text: data.live_partial_text || '',
                finalized: false,
                event_type: 'transcript.candidate',
              };
              renderLiveCaption(partialData);
            }
          return;
        } else if (data.event_type === 'transcript.candidate' &&
                   Array.isArray(data.new_candidates) && data.new_candidates.length > 0) {
          /* P1: VAD candidate — tentative, shown with animation. */
          for (var i = 0; i < data.new_candidates.length; i++) {
            if (typeof data.new_candidates[i] === 'string' && data.new_candidates[i].length > 0) {
              realtimeState.sseCandidates.push(data.new_candidates[i]);
            }
          }
       } else if (data.event_type === 'transcript.final' &&
                    Array.isArray(data.new_segments) && data.new_segments.length > 0) {
           /* P1: Two-pass final — confirmed text replaces candidates.
            * Use per-segment sample positions from server when available. */
           var positions = Array.isArray(data.new_segment_positions)
             ? data.new_segment_positions : null;
           for (var i = 0; i < data.new_segments.length; i++) {
             if (typeof data.new_segments[i] === 'string' && data.new_segments[i].length > 0) {
               var pos = positions && i < positions.length
                 ? positions[i] : (data.total_samples || data.decoded_samples || 0);
               var finalIndex = realtimeState.sseSegments.length;
               var finalText = data.new_segments[i];
               realtimeState.sseSegments.push(finalText);
               realtimeState.sseSegmentSamples.push(pos);
               /* Trigger translation for this newly confirmed segment. */
               scheduleTranslationForSegment(finalIndex, finalText);
             }
           }
           /* Remove newly-finalized candidates (first N oldest). */
           var toRemove = Math.min(data.new_segments.length, realtimeState.sseCandidates.length);
           if (toRemove > 0) realtimeState.sseCandidates.splice(0, toRemove);
     } else if (Array.isArray(data.new_segments) && data.new_segments.length > 0) {
           /* Fallback: legacy behavior for compatibility. */
           var positions = Array.isArray(data.new_segment_positions)
             ? data.new_segment_positions : null;
           var currentSamples = data.total_samples || data.decoded_samples || 0;
           for (var i = 0; i < data.new_segments.length; i++) {
             if (typeof data.new_segments[i] === 'string' && data.new_segments[i].length > 0) {
               var pos = positions && i < positions.length
                 ? positions[i] : currentSamples;
               var fallbackIdx = realtimeState.sseSegments.length;
               var fallbackText = data.new_segments[i];
               realtimeState.sseSegments.push(fallbackText);
               realtimeState.sseSegmentSamples.push(pos);
               scheduleTranslationForSegment(fallbackIdx, fallbackText);
             }
           }
        }
        /* Merge: candidates are tentative, segments are confirmed. */
        var allText = realtimeState.sseSegments.concat(realtimeState.sseCandidates);
        var latest = allText.length > 0 ? allText[allText.length - 1] : '';
        var merged = Object.assign({}, realtimeState.sseLastFull, {
          sample_count: data.total_samples, decoded_samples: data.decoded_samples,
          inference_ms: data.last_inference_ms,
          segments: realtimeState.sseSegments.slice(),
          candidates: realtimeState.sseCandidates.slice(),
          segmentSamples: realtimeState.sseSegmentSamples.slice(),
          text: data.text || data.live_text || latest,
          stable_text: data.stable_text || data.live_text || latest,
          partial_text: data.partial_text || '',
          live_text: data.live_text || '',
          live_stable_text: data.live_stable_text || '',
          live_partial_text: data.live_partial_text || '',
          finalized: data.finalized,
        });
        if (data.event_type) merged.event_type = data.event_type;
        if (data.reconciled) merged.reconciled = true;
        applyUpdate(merged);
      } else {
        realtimeState.sseLastFull = data;
        realtimeState.sseSegments = Array.isArray(data.segments) ? data.segments.slice() : [];
          realtimeState.sseCandidates = Array.isArray(data.candidates) ? data.candidates.slice() : [];
          /* Initialize segment sample positions from the full snapshot.
           * Use server-provided per-segment positions when available,
           * otherwise fall back to decoded_samples for all segments. */
          realtimeState.sseSegmentSamples = Array.isArray(data.segmentSamples)
            ? data.segmentSamples.slice() : [];
          if (realtimeState.sseSegmentSamples.length === 0) {
            var initSamples = data.decoded_samples || data.total_samples || 0;
            for (var i = 0; i < realtimeState.sseSegments.length; i++) {
              realtimeState.sseSegmentSamples.push(initSamples);
            }
          }
       applyUpdate(data);
        /* Translate segments from the initial snapshot. */
        translationState.byIndex = {};
        translationState.pending = {};
        for (var si = 0; si < realtimeState.sseSegments.length; si++) {
          scheduleTranslationForSegment(si, realtimeState.sseSegments[si]);
        }
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
      if (realtimeState.sendTimer) { window.clearInterval(realtimeState.sendTimer); realtimeState.sendTimer = null; }
      if (realtimeState.meterTimer) { window.clearInterval(realtimeState.meterTimer); realtimeState.meterTimer = null; }
      if (realtimeState.sse) { try { realtimeState.sse.close(); } catch {} realtimeState.sse = null; }
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
      /* Flush remaining audio so the server has the full last chunk. */
      await flushRealtimeChunk(true);
      /* Stop mic BEFORE closing SSE so server can push final events. */
      realtimeState.processor.disconnect();
      realtimeState.source.disconnect();
      realtimeState.mediaStream.getTracks().forEach(function (t) { t.stop(); });
      await realtimeState.audioContext.close();
    /* Send stop AFTER mic is closed and audio flushed. */
       var response = await fetch('/api/realtime/stop?session_id=' + encodeURIComponent(sessionId), { method: 'POST', body: '' });
       var data = await response.json();
       /* Sync stop response segments into SSE state and collect any
        * that still lack a translation result.  The stop response is
        * the authoritative snapshot — it may contain segments that SSE
        * never pushed (connection closed during finalization). */
       var pendingTranslations = [];
       if (response.ok && Array.isArray(data.segments)) {
         var stopSegments = data.segments.slice();
         var stopSamples = (Array.isArray(data.segmentSamples) && data.segmentSamples.length > 0)
           ? data.segmentSamples.slice() : [];
         var totalSamples = data.total_samples || data.decoded_samples || 0;

         /* Extend SSE state with segments from stop response. */
         while (realtimeState.sseSegments.length < stopSegments.length) {
           var idx = realtimeState.sseSegments.length;
           realtimeState.sseSegments.push(stopSegments[idx]);
           realtimeState.sseSegmentSamples.push(
             (stopSamples.length > 0)
               ? (stopSamples[idx] || stopSamples[stopSamples.length - 1])
               : totalSamples
           );
         }

         /* For every segment, if there's no translation yet, queue it. */
         for (var si = 0; si < stopSegments.length; si++) {
           if (!translationState.byIndex[si] && stopSegments[si]) {
             pendingTranslations.push({ index: si, text: stopSegments[si] });
           }
         }
       }
       /* Now SSE can be safely closed — server has finished and pushed [DONE]. */
       if (realtimeState.sse) { try { realtimeState.sse.close(); } catch {} realtimeState.sse = null; }
       var prevStats = realtimeStatus.textContent;
       if (response.ok) {
         renderTranscript(realtimeResult, data, '尚无结果');
         /* Also render the transcript panel so new segments from
          * the stop response are visible immediately (before translation). */
         renderTranscriptPanel({
           segments: realtimeState.sseSegments.slice(),
           candidates: realtimeState.sseCandidates ? realtimeState.sseCandidates.slice() : [],
           segmentSamples: realtimeState.sseSegmentSamples ? realtimeState.sseSegmentSamples.slice() : [],
           event_type: 'transcript.final'
         });
         syncArchive(data);
         var stopLabel = data.text ? '已停止，终稿已出' : '已停止';
         realtimeStatus.textContent = prevStats ? (prevStats + ' / ' + stopLabel) : ('会话 ' + sessionId + ' ' + stopLabel);
      } else {
        realtimeStatus.textContent = (data.error ? data.error.message : '停止失败');
      }
    } finally {
       /* Minimal cleanup — defer full state reset until translations complete. */
       archiveState.sessionId = '';
    }

    /* Await translation of any segments added during stop before
     * resetting state and re-enabling the start button.  We do NOT
     * use scheduleTranslationForSegment here — it has SSE-side
     * pending dedup that would skip already-in-flight translations.
     * Instead, for each segment without a translation result, we
     * directly call the translation API. */
    if (pendingTranslations.length > 0) {
      var stopTransPromises = pendingTranslations.map(function (pt) {
        return doTranslateOne(pt.index, pt.text);
      });
      await Promise.all(stopTransPromises);
    }

    /* Final render after translations — ensures the transcript panel
     * shows the latest translation results before we reset state. */
    renderTranscriptPanel({
      segments: realtimeState.sseSegments ? realtimeState.sseSegments.slice() : [],
      candidates: realtimeState.sseCandidates ? realtimeState.sseCandidates.slice() : [],
      segmentSamples: realtimeState.sseSegmentSamples ? realtimeState.sseSegmentSamples.slice() : [],
      event_type: 'transcript.final'
    });

    /* Now safe to fully reset. */
    realtimeState = QasrState.createRealtimeState();
    if (uiState.activeFeature === QasrState.REALTIME_FEATURE) uiState.activeFeature = '';
    updateControlAvailability();
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
     /* Also clear the live caption panel. */
     if (captionText) captionText.textContent = '';
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

  /* Translation language selectors. */
  if (translationSourceLang) {
    translationSourceLang.value = translationState.sourceLang;
    translationSourceLang.addEventListener('change', function () {
      translationState.sourceLang = translationSourceLang.value || 'auto';
      localStorage.setItem('qasrTranslationSourceLang', translationState.sourceLang);
      clearTranslationCacheAndRerender();
    });
  }

  if (translationTargetLang) {
    translationTargetLang.value = translationState.targetLang;
    translationTargetLang.addEventListener('change', function () {
      translationState.targetLang = translationTargetLang.value || 'en';
      localStorage.setItem('qasrTranslationTargetLang', translationState.targetLang);
      clearTranslationCacheAndRerender();
    });
  }

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

  loadServerInfo();
  updateControlAvailability();
  resetArchive();
  renderGlossary();
  checkHealth();

})();
