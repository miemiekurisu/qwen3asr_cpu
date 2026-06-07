// ui/state_pure.js
//
// Pure helpers extracted from ui/app.js so they can be unit-tested
// from Node without a browser.  These functions operate on plain
// data structures (the "lines" array the realtime terminal keeps),
// not on DOM elements.  The DOM-touching logic stays in app.js; the
// pure part is here.
//
// Exposed as globals (no ES6 modules) so the existing <script> tag
// wiring in index.html continues to work without a bundler.  The
// Node test file loads this script via `vm` so it sees the same
// globals the browser does.
//
// Test framework: vanilla Node assert + a tiny `test(name, fn)`
// runner.  No external dependencies, no Jest, no Mocha.

(function (root) {
  'use strict';

  /**
   * Given a list of terminal line records, return a NEW list
   * representing the state after a "soft reset" (i.e. the user
   * clicked Stop and then Start a new session).  The DOM-update
   * side effects (classList, appendChild, removeChild) stay in
   * app.js; this function only computes the new array.
   *
   * Rules (see app.js softResetRealtimeArchive for the canonical
   * implementation this mirrors):
   *   1. If the LAST entry is `state === "typing"`:
   *        - if its `text` is non-empty: convert to `state === "done"`
   *          (the user's last words get committed; they don't
   *          disappear when a new session starts)
   *        - if its `text` is empty: drop it entirely (the cursor
   *          was empty; nothing to keep)
   *   2. Ensure the list ends with a fresh empty cursor.  If the
   *      tail is not a cursor (or is a cursor with non-empty text),
   *      append a new `{ state: "cursor", el: null, text: "" }`.
   *
   * The function does NOT mutate the input array; it returns a
   * new array.  Each output entry is a shallow copy of the input
   * entry (so callers can flip `state` without affecting the
   * original).
   *
   * @param {Array<{state: string, el: any, text: string}>} lines
   * @returns {Array<{state: string, el: any, text: string}>}
   */
  function computeSoftResetLines(lines) {
    if (!Array.isArray(lines)) {
      throw new TypeError('computeSoftResetLines: lines must be an array');
    }
    /* Shallow-copy each entry, preserving any extra fields the
     * caller has attached (id, segmentId, …).  Only `state` is
     * potentially mutated; the rest are pass-through. */
    const out = lines.map(function (l) {
      const copy = {};
      for (const k in l) {
        if (Object.prototype.hasOwnProperty.call(l, k)) {
          copy[k] = l[k];
        }
      }
      return copy;
    });
    if (out.length > 0) {
      const last = out[out.length - 1];
      if (last.state === 'typing') {
        if (last.text && last.text.length > 0) {
          last.state = 'done';
        } else {
          out.pop();
        }
      }
    }
    const tail = out[out.length - 1];
    if (!tail || tail.state !== 'cursor' || (tail.text && tail.text.length > 0)) {
      out.push({ state: 'cursor', el: null, text: '' });
    }
    return out;
  }

  /**
   * Compute the confirmed realtime text from an archive of lines.
   * Mirrors extractConfirmedRealtimeText() in app.js.  Lines with
   * state !== "done" or empty text are skipped; the rest are
   * joined with a single space.
   *
   * @param {Array<{state: string, text: string}>} lines
   * @returns {string}
   */
  function computeConfirmedRealtimeText(lines) {
    if (!Array.isArray(lines)) {
      return '';
    }
    return lines
      .filter(function (l) {
        return l && l.state === 'done' && l.text && l.text.trim();
      })
      .map(function (l) { return l.text; })
      .join(' ');
  }

  /**
   * Count Unicode code points, not UTF-16 code units.  A CJK
   * character is one code point (one code unit too, in that case),
   * but a surrogate pair (emoji, certain historic scripts) is two
   * code units and one code point.  `Array.from` walks by code
   * point.  null / undefined are treated as the empty string.
   *
   * @param {string} text
   * @returns {number}
   */
  function countCodepoints(text) {
    return Array.from(text || '').length;
  }

  /**
   * Decimate a Float32Array of audio samples from `inputRate` to
   * 16 kHz by picking one sample per output window (center of the
   * window for a stable pick), NOT by averaging.  Average-of-N
   * would cancel the audio signal because speech oscillates
   * around zero and summing N samples tends to 0.
   *
   * If `inputRate === 16000`, returns `input` unchanged (no copy).
   *
   * @param {Float32Array} input
   * @param {number} inputRate
   * @returns {Float32Array}
   */
  function downsampleTo16k(input, inputRate) {
    if (inputRate === 16000) {
      return input;
    }
    const ratio = inputRate / 16000;
    const outputLength = Math.floor(input.length / ratio);
    const output = new Float32Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      const center = (index + 0.5) * ratio;
      output[index] = input[Math.min(input.length - 1, Math.floor(center))];
    }
    return output;
  }

  /**
   * Convert Float32 audio samples ([-1, 1]) to Int16 PCM.
   *
   * Asymmetric int16 range: -1.0 -> -32768 (not -32767) because
   * two's-complement int16 spans [-32768, 32767].  Using -32767
   * for both ends would waste one code point.  Input is clamped
   * to [-1, 1] first so out-of-range values (e.g. 1.5 from a hot
   * signal) don't wrap to the wrong sign.
   *
   * @param {Float32Array} input
   * @returns {Int16Array}
   */
  function floatToPcm16(input) {
    const output = new Int16Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      output[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    return output;
  }

  /**
   * Build a download filename for a realtime transcript.  Used by
   * the Export TXT / Export JSON click handlers.  The session id
   * is sanitized to [a-zA-Z0-9_-] (anything else collapses to a
   * single dash) so the filename is safe across filesystems and
   * shells.  The timestamp uses ISO 8601 with `:` and `.` replaced
   * by `-` (Windows / FAT32 cannot store those characters in
   * filenames).
   *
   * @param {string} sessionId   raw session id, may contain unsafe chars
   * @param {string} ext         file extension without the dot (e.g. "txt")
   * @param {Date}   [now]       optional Date for testability
   * @returns {string}           e.g. "qasr-realtime-rt-abc-2026-06-05T12-00-00-000Z.txt"
   */
  function buildRealtimeExportName(sessionId, ext, now) {
    const safeId = (sessionId || 'session').replace(/[^a-zA-Z0-9_-]+/g, '-');
    const stamp = (now || new Date()).toISOString().replace(/[:.]/g, '-');
    return 'qasr-realtime-' + safeId + '-' + stamp + '.' + ext;
  }

  // ───── export ─────
  const api = {
    computeSoftResetLines: computeSoftResetLines,
    computeConfirmedRealtimeText: computeConfirmedRealtimeText,
    countCodepoints: countCodepoints,
    downsampleTo16k: downsampleTo16k,
    floatToPcm16: floatToPcm16,
    buildRealtimeExportName: buildRealtimeExportName,
  };
  if (typeof module !== 'undefined' && module.exports) {
    // Node (test runner).
    module.exports = api;
  } else {
    // Browser.
    root.QasrStatePure = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
