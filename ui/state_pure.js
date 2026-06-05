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

  // ───── export ─────
  const api = {
    computeSoftResetLines: computeSoftResetLines,
    computeConfirmedRealtimeText: computeConfirmedRealtimeText,
  };
  if (typeof module !== 'undefined' && module.exports) {
    // Node (test runner).
    module.exports = api;
  } else {
    // Browser.
    root.QasrStatePure = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
