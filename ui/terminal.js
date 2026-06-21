// ui/terminal.js — Terminal rendering helpers
// Exposes pure rendering logic. DOM updates are done by calling functions.
(function (root) {
  'use strict';

  /**
   * Create a terminal line DOM element.
   * @param {'done'|'typing'|'cursor'} state
   * @param {string} text
   * @param {Array} allLines — full archive lines array (for has-done check)
   * @returns {HTMLElement}
   */
  function makeTermLine(state, text, allLines) {
    var div = document.createElement('div');
    div.className = 'term-line ' + state;
    if (state === 'cursor' && !text) {
      div.classList.add('empty');
      var hasDone = false;
      for (var i = 0; i < (allLines || []).length; i++) {
        if (allLines[i].state === 'done') { hasDone = true; break; }
      }
      if (hasDone) div.classList.add('has-done');
      var c = document.createElement('span');
      c.className = 'cursor-blink';
      div.appendChild(c);
    } else {
      div.appendChild(document.createTextNode(text));
    }
    return div;
  }

  /**
   * Animate a new segment with typewriter effect.
   * @param {HTMLElement} element — the terminal container
   * @param {string} text — the text to type out
   * @param {Object} archive — { lines, typewriterTimer }
   * @returns {number} — the setInterval handle
   */
  function animateSegment(element, text, archive) {
    if (archive.typewriterTimer !== null) {
      clearInterval(archive.typewriterTimer);
      archive.typewriterTimer = null;
    }

    // Merge new text into existing typing line instead of creating a new
    // line per segment.  This prevents VAD segment boundaries from
    // appearing as visual line breaks in the terminal.
    var typingEntry = null;
    var lastLine = archive.lines[archive.lines.length - 1];
    var prevText = '';
    if (lastLine && lastLine.state === 'cursor') {
      // Cursor → typing
      var blink = lastLine.el.querySelector('.cursor-blink');
      if (blink) blink.remove();
      lastLine.el.classList.remove('cursor', 'empty');
      lastLine.el.classList.add('typing');
      lastLine.state = 'typing';
      lastLine.el.textContent = '';
      typingEntry = lastLine;
    } else if (lastLine && (lastLine.state === 'typing' || lastLine.state === 'candidate')) {
      // Append to existing typing/candidate line.  When the model
      // re-decodces the next segment with tail context, the new text
      // extends the previous sentence — don't break the visual flow.
      prevText = lastLine.text || '';
      if (lastLine.state === 'candidate') {
        lastLine.el.classList.remove('candidate');
        lastLine.el.classList.add('typing');
      }
      lastLine.state = 'typing';
      typingEntry = lastLine;
      // Strip longest common prefix to avoid duplication when the
      // model re-decodes the same audio with slight differences
      // (e.g. "几日未见" vs "今日未见").
      var commonLen = 0;
      var maxLen = Math.min(prevText.length, text.length);
      while (commonLen < maxLen && prevText.charCodeAt(commonLen) === text.charCodeAt(commonLen)) {
        commonLen++;
      }
      if (commonLen > 0 && commonLen > prevText.length / 2) {
        text = text.slice(commonLen);
      }
      typingEntry.el.textContent = prevText;
      typingEntry.text = prevText;
    } else {
      // First segment — fresh typing line
      var typingLine = makeTermLine('typing', '', archive.lines);
      element.appendChild(typingLine);
      typingEntry = { state: 'typing', el: typingLine, text: '' };
      archive.lines.push(typingEntry);
    }

    var textChars = Array.from(text);
    var perCharMs = textChars.length <= 8 ? 10 : textChars.length <= 30 ? 18 : 14;
    var i = 0;
    function tick() {
      if (i >= textChars.length) {
        clearInterval(archive.typewriterTimer);
        archive.typewriterTimer = null;
        typingEntry.el.classList.remove('typing');
        typingEntry.el.classList.add('candidate');
        typingEntry.state = 'candidate';
        typingEntry.text = prevText + text;
        var cursorLine = makeTermLine('cursor', '', archive.lines);
        element.appendChild(cursorLine);
        archive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
        element.scrollTop = element.scrollHeight;
        return;
      }
      i += 1;
      var visible = textChars.slice(0, i).join('');
      typingEntry.el.textContent = prevText + visible;
      typingEntry.text = prevText + visible;
    }
    archive.typewriterTimer = setInterval(tick, perCharMs);
    tick(); // Show first char immediately
    return archive.typewriterTimer;
  }

  /**
   * Render a finalized segment (instant, no typewriter).
   * @param {HTMLElement} element
   * @param {string} text
   * @param {Object} archive
   */
  function renderFinalizedSegment(element, text, archive) {
    if (archive.typewriterTimer !== null) {
      clearInterval(archive.typewriterTimer);
      archive.typewriterTimer = null;
    }
    var lastLine = archive.lines[archive.lines.length - 1];
    if (lastLine && lastLine.state === 'cursor') {
      /* Convert cursor line → done. */
      var blink = lastLine.el.querySelector('.cursor-blink');
      if (blink) blink.remove();
      lastLine.el.classList.remove('cursor', 'empty');
      lastLine.el.classList.add('done');
      lastLine.state = 'done';
      lastLine.text = text;
      lastLine.el.textContent = text;
    } else if (lastLine && (lastLine.state === 'candidate' || lastLine.state === 'typing')) {
      /* Promote existing candidate/typing line to done instead of
       * creating a new line — prevents text duplication when a
       * previously-shown candidate is confirmed by the server. */
      lastLine.el.classList.remove('candidate', 'typing');
      lastLine.el.classList.add('done');
      lastLine.state = 'done';
      lastLine.text = text;
      lastLine.el.textContent = text;
    } else {
      /* Fresh confirmed line (no prior candidate). */
      var doneLine = makeTermLine('done', text, archive.lines);
      element.appendChild(doneLine);
      archive.lines.push({ state: 'done', el: doneLine, text: text });
    }
    element.scrollTop = element.scrollHeight;
  }

  /**
   * Promote candidate lines to done state when the finalizer confirms them.
   * This makes the text eligible for export.
   * @param {Object} archive — { lines }
   */
  function promoteCandidateToDone(archive) {
    if (!archive || !Array.isArray(archive.lines)) return;
    for (var i = 0; i < archive.lines.length; i++) {
      if (archive.lines[i].state === 'candidate') {
        archive.lines[i].state = 'done';
        var el = archive.lines[i].el;
        if (el) {
          el.classList.remove('candidate');
          el.classList.add('done');
        }
      }
    }
  }

  // ───── Export ─────
  var api = {
    makeTermLine: makeTermLine,
    animateSegment: animateSegment,
    renderFinalizedSegment: renderFinalizedSegment,
    promoteCandidateToDone: promoteCandidateToDone,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.QasrTerminal = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
