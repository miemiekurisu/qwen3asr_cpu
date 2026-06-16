// ui/state.js — Pure state management for QASR UI
// No DOM access. All state is plain data.
(function (root) {
  'use strict';

  var BATCH_FEATURE = 'batch';
  var REALTIME_FEATURE = 'realtime';
  var MAX_ASYNC_UPLOAD_BYTES = 64 * 1024 * 1024;

  // ───── Session State ─────

  function createSessionState() {
    return {
      id: '',
      status: 'idle', // idle | listening | asr | idle
      sourceLang: 'auto',
      asrLatencyMs: 0,
      queueDepth: 0,
      backend: 'CPU',
      text: '',
      segments: [],
      createdAt: 0,
    };
  }

  // ───── Offline State ─────

  function createOfflineState() {
    return {
      jobId: '',
      stopRequested: false,
      stopError: '',
      startedAt: 0,
    };
  }

  // ───── Realtime State ─────

  function createRealtimeState() {
    return {
      sessionId: '',
      sampleRate: 0,
      startedAt: 0,
      prePeak: 0,
      preRms: 0,
      postPeak: 0,
      postRms: 0,
      maxSrvPeak: 0,
      // SSE state
      sseSegments: [],
      sseLastFull: null,
    };
  }

  // ───── Archive State ─────

  function createArchiveState() {
    return {
      sessionId: '',
      lastSegmentCount: 0,
      finalized: false,
      updatedAt: '',
    };
  }

  // ───── Glossary State ─────

  function createGlossaryState() {
    return {
      domain: 'general',
      style: 'subtitle',
      entries: [], // { source, target, lang, locked }
      customPrompt: '',
    };
  }

  // ───── UI State (global) ─────

  function createUIState() {
    return {
      activeFeature: '',
      realtimeStarting: false,
      realtimeStopping: false,
      realtimeCapturing: false,
      serverBackend: 'cpu',
      sessions: [],
      currentSessionId: '',
    };
  }

  // ───── Export helpers ─────

  function buildExportName(sessionId, ext, now) {
    var safeId = (sessionId || 'session').replace(/[^a-zA-Z0-9_-]+/g, '-');
    var stamp = (now || new Date()).toISOString().replace(/[:.]/g, '-');
    return 'qasr-realtime-' + safeId + '-' + stamp + '.' + ext;
  }

  function buildSrtFromSegments(segments, sampleRate) {
    var srt = '';
    var sr = sampleRate || 16000;
    var cumulative = 0;
    for (var i = 0; i < segments.length; i++) {
      var seg = segments[i];
      var dur = seg.sample_count ? Math.round(seg.sample_count / sr * 1000) : 3000;
      var start = cumulative;
      var end = cumulative + dur;
      var startMs = padTimestamp(start);
      var endMs = padTimestamp(end);
      srt += (i + 1) + '\n';
      srt += startMs + ' --> ' + endMs + '\n';
      srt += (seg.text || '') + '\n\n';
      cumulative = end;
    }
    return srt;
  }

  function padTimestamp(ms) {
    var h = Math.floor(ms / 3600000);
    var m = Math.floor((ms % 3600000) / 60000);
    var s = Math.floor((ms % 60000) / 1000);
    var f = Math.floor(ms % 1000);
    return String(h).padStart(2, '0') + ':' +
           String(m).padStart(2, '0') + ':' +
           String(s).padStart(2, '0') + ',' +
           String(f).padStart(3, '0');
  }

  function msToTimestamp(ms) {
    var m = Math.floor(ms / 60000);
    var s = Math.floor((ms % 60000) / 1000);
    var f = Math.floor((ms % 1000) / 100);
    return String(m).padStart(2, '0') + ':' +
           String(s).padStart(2, '0') + '.' + f;
  }

  // ───── Export ─────
  var api = {
    BATCH_FEATURE: BATCH_FEATURE,
    REALTIME_FEATURE: REALTIME_FEATURE,
    MAX_ASYNC_UPLOAD_BYTES: MAX_ASYNC_UPLOAD_BYTES,
    createSessionState: createSessionState,
    createOfflineState: createOfflineState,
    createRealtimeState: createRealtimeState,
    createArchiveState: createArchiveState,
    createGlossaryState: createGlossaryState,
    createUIState: createUIState,
    buildExportName: buildExportName,
    buildSrtFromSegments: buildSrtFromSegments,
    msToTimestamp: msToTimestamp,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.QasrState = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
