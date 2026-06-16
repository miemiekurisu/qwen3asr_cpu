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

    // Find or create typing line
    var typingEntry = null;
    var lastLine = archive.lines[archive.lines.length - 1];
    if (lastLine && lastLine.state === 'cursor') {
      var blink = lastLine.el.querySelector('.cursor-blink');
      if (blink) blink.remove();
      lastLine.el.classList.remove('cursor', 'empty');
      lastLine.el.classList.add('typing');
      lastLine.state = 'typing';
      lastLine.el.textContent = '';
      typingEntry = lastLine;
    } else if (lastLine && lastLine.state === 'typing') {
      // Complete previous typing
      lastLine.el.textContent = lastLine.text;
      lastLine.el.classList.remove('typing');
      lastLine.el.classList.add('done');
      lastLine.state = 'done';
      // Create new cursor -> typing
      var cursorLine = makeTermLine('cursor', '', archive.lines);
      element.appendChild(cursorLine);
      archive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
      var blink2 = cursorLine.querySelector('.cursor-blink');
      if (blink2) blink2.remove();
      cursorLine.classList.remove('cursor', 'empty');
      cursorLine.classList.add('typing');
      cursorLine.textContent = '';
      typingEntry = { state: 'typing', el: cursorLine, text: '' };
      archive.lines[archive.lines.length - 1] = typingEntry;
    } else {
      var typingLine = makeTermLine('typing', '', archive.lines);
      element.appendChild(typingLine);
      typingEntry = { state: 'typing', el: typingLine, text: '' };
      archive.lines.push(typingEntry);
    }

    var perCharMs = text.length <= 8 ? 10 : text.length <= 30 ? 18 : 14;
    var i = 0;
    function tick() {
      if (i >= text.length) {
        clearInterval(archive.typewriterTimer);
        archive.typewriterTimer = null;
        typingEntry.el.classList.remove('typing');
        typingEntry.el.classList.add('done');
        typingEntry.state = 'done';
        typingEntry.text = text;
        var cursorLine = makeTermLine('cursor', '', archive.lines);
        element.appendChild(cursorLine);
        archive.lines.push({ state: 'cursor', el: cursorLine, text: '' });
        element.scrollTop = element.scrollHeight;
        return;
      }
      i += 1;
      typingEntry.el.textContent = text.slice(0, i);
      typingEntry.text = text.slice(0, i);
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
      var blink = lastLine.el.querySelector('.cursor-blink');
      if (blink) blink.remove();
      lastLine.el.classList.remove('cursor', 'empty');
      lastLine.el.classList.add('done');
      lastLine.state = 'done';
      lastLine.text = text;
      lastLine.el.textContent = text;
    } else {
      var doneLine = makeTermLine('done', text, archive.lines);
      element.appendChild(doneLine);
      archive.lines.push({ state: 'done', el: doneLine, text: text });
    }
    element.scrollTop = element.scrollHeight;
  }

  // ───── Export ─────
  var api = {
    makeTermLine: makeTermLine,
    animateSegment: animateSegment,
    renderFinalizedSegment: renderFinalizedSegment,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.QasrTerminal = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
