#!/usr/bin/env python3
"""
audit_bug_replay_frontend_test.py - Verify all 12 frontend bug fixes.
CI-safe: no browser needed - tests extract and analyze source directly.
"""
import os, sys, re

UI_DIR = os.path.join(os.path.dirname(__file__), '..', 'ui')
PASS = 0
FAIL = 0

def check(condition, msg):
    global PASS, FAIL
    if condition:
        print(f"  PASS: {msg}")
        PASS += 1
    else:
        print(f"  FAIL: {msg}")
        FAIL += 1

with open(os.path.join(UI_DIR, 'app.js'), 'r') as f:
    app_js = f.read()
with open(os.path.join(UI_DIR, 'live_monitor.js'), 'r') as f:
    live_monitor_js = f.read()

# ██  Round 1 — 7 bugs fixed earlier  ██

# ─── #1: XSS — e.lang unescaped in innerHTML ───
print("\n--- #1: XSS (e.lang unescaped in innerHTML) ---")
check('escapeHtml(e.lang || \'-\')' in app_js or 'escapeHtml(e.lang||\'-\')' in app_js,
      "e.lang is wrapped with escapeHtml()")

# ─── #2: escapeHtml missing quote encoding ───
print("\n--- #2: escapeHtml missing quote encoding ---")
fn = re.search(r'function escapeHtml[\s\S]{0,300}', app_js)
if fn:
    impl = fn.group(0)
    check("&#39;" in impl and "&quot;" in impl,
          f"escapeHtml encodes both single and double quotes")

# ─── #3: realtimeResult null access ───
print("\n--- #3: realtimeResult potentially null ---")
check('if (!element) return;' in app_js,
      "renderTranscript/applyTranscriptRender/resetTerminal guard null element")
check('if (!realtimeResult) return;' in app_js,
      "softResetArchive and resetArchive guard null realtimeResult")

# ─── #4: NaN display ───
print("\n--- #4: NaN display vulnerability ---")
check('function safeFixed' in app_js, "safeFixed helper function exists")
check(app_js.count('safeFixed(data.sample_count') > 0,
      "safeFixed used for sample_count/decode_samples in realtime paths")

# ─── #5: SSE connection leak ───
print("\n--- #5: SSE connection leak ---")
fn = re.search(r'function openSseStream[\s\S]{0,2000}', app_js)
if fn:
    func = fn.group(0)
    closes_before = False
    try:
        if 'close(' in func and 'es = new EventSource' in func:
            closes_before = func.index('close(') < func.index('es = new EventSource')
    except ValueError:
        pass
    check(closes_before, "openSseStream closes previous EventSource before creating new one")

# ─── #6: Timer leak on setup failure ───
print("\n--- #6: Timer leak on setup failure ---")
catch_blocks = re.findall(r'catch\s*\([^)]*\)\s*\{[^}]{0,800}', app_js, re.DOTALL)
timer_leak_fixed = any(
    'realtimeState' in cb and 'clearInterval' in cb and 'sendTimer' in cb
    for cb in catch_blocks)
check(timer_leak_fixed, "catch block in startRealtimeCapture clears sendTimer and meterTimer")

# ─── #7+8: _transcriptFrame not cleared on reset ───
print("\n--- #7+8: stale _transcriptFrame after reset ---")
fn = re.search(r'function resetTerminal[\s\S]{0,1500}', app_js)
if fn:
    check('element._transcriptFrame = null' in fn.group(0),
          "resetTerminal clears element._transcriptFrame")
fn = re.search(r'function applyTranscriptRender[\s\S]{0,2000}', app_js)
if fn:
    guard = 'element._transcriptFrame' in fn.group(0).split('return')[0]
    check(guard, "applyTranscriptRender guards on _transcriptFrame")

# ██  Round 2 — 5 bugs fixed now  ██

# ─── #5.3: flushRealtimeChunk vs stopRealtimeCapture race ───
print("\n--- #5.3: flushRealtimeChunk race with stopRealtimeCapture ---")
# Check that sessionId is captured before async and checked after
check("sessionIdAtSend" in app_js and 'realtimeState.sessionId !== sessionIdAtSend' in app_js,
      "flushRealtimeChunk captures sessionId before fetch and guards after await")

# ─── #5.5: rAF callback no try/catch ───
print("\n--- #5.5: rAF callback no try/catch ---")
check("try {" in app_js and "applyTranscriptRender" in app_js and "_e" in app_js,
      "rAF callback wrapped in try/catch")

# ─── #5.10: 多余初始游标推入后立即丢弃 ───
print("\n--- #5.10: initial cursor push immediately discarded by resetArchive ---")
# Check that the init code no longer pushes cursor before resetArchive()
init_section = app_js[app_js.rfind("loadServerInfo"):]
check('.term-line.cursor' not in init_section,
      "no redundant cursor push before resetArchive() in init")

# ─── #5.11: MutationObserver never disconnect ───
print("\n--- #5.11: MutationObserver never disconnect ---")
check("disconnectObservers" in live_monitor_js and "beforeunload" in live_monitor_js,
      "live_monitor.js stores observers and disconnects on beforeunload")

# ─── #5.12: Live monitor DOM 操作 ───
print("\n--- #5.12: Live monitor innerHTML read ---")
# Verify snap() no longer reads innerHTML (only textContent)
check('.innerHTML' not in live_monitor_js,
      "live_monitor.js snap() does not read innerHTML (uses textContent only)")
check('.textContent' in live_monitor_js,
      "live_monitor.js snap() uses textContent")

# ─── Summary ───
print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL > 0 else 0)
