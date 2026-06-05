// tests/state_pure_test.js
//
// Node-side unit tests for ui/state_pure.js.  No browser, no DOM,
// no Jest.  Run with:
//   node tests/state_pure_test.js
//
// Exits 0 if all tests pass, 1 otherwise.

'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const pure = require(path.join(__dirname, '..', 'ui', 'state_pure.js'));

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  [PASS] ${name}`);
  } catch (e) {
    failed += 1;
    failures.push({ name, error: e });
    console.log(`  [FAIL] ${name}`);
    console.log(`         ${e && e.message ? e.message : e}`);
    if (e && e.stack) {
      const lines = e.stack.split('\n').slice(1, 4);
      for (const l of lines) {
        console.log(`         ${l.trim()}`);
      }
    }
  }
}

// ───── computeConfirmedRealtimeText ─────

test('confirmed: empty list returns empty string', () => {
  assert.equal(pure.computeConfirmedRealtimeText([]), '');
});

test('confirmed: null input returns empty string', () => {
  assert.equal(pure.computeConfirmedRealtimeText(null), '');
});

test('confirmed: undefined input returns empty string', () => {
  assert.equal(pure.computeConfirmedRealtimeText(undefined), '');
});

test('confirmed: non-array input returns empty string', () => {
  assert.equal(pure.computeConfirmedRealtimeText('hello'), '');
});

test('confirmed: only done lines with text are joined', () => {
  const lines = [
    { state: 'done', text: 'hello' },
    { state: 'done', text: 'world' },
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), 'hello world');
});

test('confirmed: typing lines are excluded', () => {
  const lines = [
    { state: 'done', text: 'committed' },
    { state: 'typing', text: 'tentative' },
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), 'committed');
});

test('confirmed: cursor lines are excluded', () => {
  const lines = [
    { state: 'done', text: 'a' },
    { state: 'cursor', text: '' },
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), 'a');
});

test('confirmed: whitespace-only done lines are excluded', () => {
  const lines = [
    { state: 'done', text: '   ' },
    { state: 'done', text: '\t\n' },
    { state: 'done', text: '' },
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), '');
});

test('confirmed: mixed done and empty-text done', () => {
  const lines = [
    { state: 'done', text: 'first' },
    { state: 'done', text: '' },
    { state: 'done', text: 'third' },
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), 'first third');
});

test('confirmed: null entries in the array are skipped', () => {
  const lines = [
    null,
    { state: 'done', text: 'a' },
    null,
  ];
  assert.equal(pure.computeConfirmedRealtimeText(lines), 'a');
});

test('confirmed: long stream of done lines preserves order', () => {
  const lines = [];
  for (let i = 0; i < 50; ++i) {
    lines.push({ state: 'done', text: `seg${i}` });
  }
  const out = pure.computeConfirmedRealtimeText(lines);
  const parts = out.split(' ');
  assert.equal(parts.length, 50);
  assert.equal(parts[0], 'seg0');
  assert.equal(parts[49], 'seg49');
});

// ───── computeSoftResetLines ─────

test('soft reset: empty input yields a single empty cursor', () => {
  const out = pure.computeSoftResetLines([]);
  assert.equal(out.length, 1);
  assert.equal(out[0].state, 'cursor');
  assert.equal(out[0].text, '');
  assert.equal(out[0].el, null);
});

test('soft reset: non-array input throws TypeError', () => {
  assert.throws(
    () => pure.computeSoftResetLines(null),
    TypeError,
  );
  assert.throws(
    () => pure.computeSoftResetLines('not-an-array'),
    TypeError,
  );
});

test('soft reset: typing tail with text is committed to done', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'typing', text: 'partial', el: 'elB' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out.length, 3);
  assert.equal(out[0].state, 'done');
  assert.equal(out[0].text, 'a');
  assert.equal(out[1].state, 'done');
  assert.equal(out[1].text, 'partial');
  assert.equal(out[2].state, 'cursor');
  assert.equal(out[2].text, '');
});

test('soft reset: typing tail with empty text is dropped', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'typing', text: '', el: 'elB' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out.length, 2);
  assert.equal(out[0].state, 'done');
  assert.equal(out[0].text, 'a');
  assert.equal(out[1].state, 'cursor');
});

test('soft reset: tail is already a cursor (empty), no extra append', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'cursor', text: '', el: 'elCursor' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out.length, 2);
  assert.equal(out[1].state, 'cursor');
  assert.equal(out[1].text, '');
  assert.equal(out[1].el, 'elCursor');
});

test('soft reset: tail is cursor with non-empty text → append fresh cursor', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'cursor', text: 'leftover', el: 'elCursor' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out.length, 3);
  assert.equal(out[0].state, 'done');
  assert.equal(out[1].state, 'cursor');
  assert.equal(out[1].text, 'leftover');
  assert.equal(out[2].state, 'cursor');
  assert.equal(out[2].text, '');
  assert.equal(out[2].el, null);
});

test('soft reset: tail is done → append fresh cursor', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'done', text: 'b', el: 'elB' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out.length, 3);
  assert.equal(out[0].text, 'a');
  assert.equal(out[1].text, 'b');
  assert.equal(out[2].state, 'cursor');
  assert.equal(out[2].text, '');
});

test('soft reset: does not mutate the input array', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA' },
    { state: 'typing', text: 'partial', el: 'elB' },
  ];
  const snapshot = JSON.parse(JSON.stringify(lines));
  pure.computeSoftResetLines(lines);
  assert.equal(lines.length, snapshot.length);
  assert.equal(lines[0].state, snapshot[0].state);
  assert.equal(lines[1].state, snapshot[1].state);
  assert.equal(lines[1].text, snapshot[1].text);
});

test('soft reset: output entries are shallow copies (safe to mutate)', () => {
  const original = { state: 'done', text: 'a', el: 'elA' };
  const out = pure.computeSoftResetLines([original]);
  out[0].text = 'mutated';
  assert.equal(original.text, 'a', 'input must not change when output is mutated');
});

test('soft reset: "old info preserved" — 5 done lines survive', () => {
  const lines = [
    { state: 'done', text: 'one', el: 1 },
    { state: 'done', text: 'two', el: 2 },
    { state: 'done', text: 'three', el: 3 },
    { state: 'done', text: 'four', el: 4 },
    { state: 'done', text: 'five', el: 5 },
    { state: 'typing', text: 'partial', el: 6 },
  ];
  const out = pure.computeSoftResetLines(lines);
  // 5 done + 1 done (committed from typing) + 1 cursor
  assert.equal(out.length, 7);
  for (let i = 0; i < 6; ++i) {
    assert.equal(out[i].state, 'done');
  }
  assert.equal(out[6].state, 'cursor');
  assert.equal(out[0].text, 'one');
  assert.equal(out[5].text, 'partial');
});

test('soft reset: only typing line, with text, gets done + cursor', () => {
  const out = pure.computeSoftResetLines([
    { state: 'typing', text: 'last words', el: 'el' },
  ]);
  assert.equal(out.length, 2);
  assert.equal(out[0].state, 'done');
  assert.equal(out[0].text, 'last words');
  assert.equal(out[1].state, 'cursor');
});

test('soft reset: only typing line, empty, drops it + adds cursor', () => {
  const out = pure.computeSoftResetLines([
    { state: 'typing', text: '', el: 'el' },
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].state, 'cursor');
  assert.equal(out[0].text, '');
});

test('soft reset: id is preserved through the helper', () => {
  const lines = [
    { state: 'done', text: 'a', el: 'elA', id: 'L1' },
    { state: 'typing', text: 'b', el: 'elB', id: 'L2' },
  ];
  const out = pure.computeSoftResetLines(lines);
  assert.equal(out[0].id, 'L1');
  assert.equal(out[1].id, 'L2');
});

// ───── countCodepoints ─────

test('countCodepoints: empty string returns 0', () => {
  assert.equal(pure.countCodepoints(''), 0);
});

test('countCodepoints: null and undefined return 0', () => {
  assert.equal(pure.countCodepoints(null), 0);
  assert.equal(pure.countCodepoints(undefined), 0);
});

test('countCodepoints: ASCII is one code point per char', () => {
  assert.equal(pure.countCodepoints('hello'), 5);
});

test('countCodepoints: CJK is one code point per char', () => {
  assert.equal(pure.countCodepoints('你好世界'), 4);
});

test('countCodepoints: emoji surrogate pair is ONE code point (not 2)', () => {
  /* "😀" is U+1F600 GRINNING FACE — encoded in UTF-16 as the
   * surrogate pair 0xD83D 0xDE00 (2 code units, 1 code point).
   * .length would return 2; Array.from returns 1. */
  assert.equal('😀'.length, 2, 'sanity: JS .length is 2 (proves the test is meaningful)');
  assert.equal(pure.countCodepoints('😀'), 1);
});

test('countCodepoints: mixed ASCII + emoji + CJK counts code points only', () => {
  /* "a😀中" = a (1) + 😀 (1) + 中 (1) = 3 code points
   *         = a (1) + 😀 (2 units) + 中 (1) = 4 UTF-16 code units */
  assert.equal('a😀中'.length, 4, 'sanity: JS .length is 4');
  assert.equal(pure.countCodepoints('a😀中'), 3);
});

// ───── downsampleTo16k ─────

test('downsampleTo16k: same rate (16000) returns the input unchanged (no copy)', () => {
  const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const out = pure.downsampleTo16k(input, 16000);
  assert.equal(out, input, 'must return the same reference, not a copy');
});

test('downsampleTo16k: 48000 -> 16000 picks 1 of every 3 samples (center)', () => {
  /* 6 samples @ 48k, ratio=3.  Output windows are [0..3) [3..6).
   *  Center of [0..3) is at index 1.5 -> pick index 1.
   *  Center of [3..6) is at index 4.5 -> pick index 4.
   *  Use integer-valued samples (1.0, 2.0, …) to avoid Float32
   *  precision noise (0.1 stored as ~0.10000000149…). */
  const input = new Float32Array([1, 2, 3, 4, 5, 6]);
  const out = pure.downsampleTo16k(input, 48000);
  assert.equal(out.length, 2);
  assert.equal(out[0], 2);
  assert.equal(out[1], 5);
});

test('downsampleTo16k: 44100 -> 16000 uses non-integer ratio', () => {
  /* ratio = 44100/16000 = 2.75625.  Pick indices (9 samples → 3 outputs):
   *  index 0: center=1.378 -> floor=1
   *  index 1: center=4.134 -> floor=4
   *  index 2: center=6.890 -> floor=6
   *  floor(9 / 2.75625) = floor(3.265) = 3. */
  const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const out = pure.downsampleTo16k(input, 44100);
  assert.equal(out.length, 3);
  assert.equal(out[0], 2);
  assert.equal(out[1], 5);
  assert.equal(out[2], 7);
});

test('downsampleTo16k: empty input returns empty Float32Array', () => {
  const input = new Float32Array([]);
  const out = pure.downsampleTo16k(input, 48000);
  assert.equal(out.length, 0);
  assert.ok(out instanceof Float32Array, 'must be a Float32Array');
});

test('downsampleTo16k: last window is partial (input length not a multiple of ratio)', () => {
  /* 4 samples @ 48k, ratio=3.  floor(4/3) = 1 output.
   *  Center of [0..3) = 1.5 -> pick index 1. */
  const input = new Float32Array([1, 2, 3, 4]);
  const out = pure.downsampleTo16k(input, 48000);
  assert.equal(out.length, 1);
  assert.equal(out[0], 2);
});

// ───── floatToPcm16 ─────

test('floatToPcm16: zero in -> zero out', () => {
  const input = new Float32Array([0, 0, 0]);
  const out = pure.floatToPcm16(input);
  assert.equal(out.length, 3);
  for (let i = 0; i < 3; i += 1) {
    assert.equal(out[i], 0);
  }
});

test('floatToPcm16: -1.0 maps to -32768 (asymmetric int16 range)', () => {
  const out = pure.floatToPcm16(new Float32Array([-1.0]));
  assert.equal(out[0], -32768, 'asymmetric: -1.0 must hit the bottom of int16 range');
});

test('floatToPcm16: 1.0 maps to 32767 (not 32768)', () => {
  const out = pure.floatToPcm16(new Float32Array([1.0]));
  assert.equal(out[0], 32767, 'asymmetric: 1.0 must hit 32767, not 32768');
});

test('floatToPcm16: 0.5 maps to 16383 (Int16Array truncates 16383.5)', () => {
  /* 0.5 * 32767 = 16383.5.  Int16Array assignment truncates the
   * fractional part (it does NOT round), so the result is 16383,
   * not 16384.  This is intentional PCM behaviour — clipping
   * the half-LSB is standard. */
  const out = pure.floatToPcm16(new Float32Array([0.5]));
  assert.equal(out[0], 16383);
});

test('floatToPcm16: -0.5 maps to -16384 (negative half, using -32768 base)', () => {
  /* -0.5 * 32768 = -16384, exact.  Critical: this is NOT
   * -16383.5 (which is what -0.5 * 32767 would give).  The
   * asymmetric range is what makes the magnitude symmetric
   * around zero. */
  const out = pure.floatToPcm16(new Float32Array([-0.5]));
  assert.equal(out[0], -16384);
});

test('floatToPcm16: clamps out-of-range values to [-1, 1]', () => {
  const input = new Float32Array([1.5, -1.5, 2.0, -2.0]);
  const out = pure.floatToPcm16(input);
  assert.equal(out[0], 32767, '1.5 must clamp to 1.0 -> 32767');
  assert.equal(out[1], -32768, '-1.5 must clamp to -1.0 -> -32768');
  assert.equal(out[2], 32767);
  assert.equal(out[3], -32768);
});

test('floatToPcm16: empty input returns empty Int16Array', () => {
  const out = pure.floatToPcm16(new Float32Array([]));
  assert.equal(out.length, 0);
  assert.ok(out instanceof Int16Array);
});

// ───── buildRealtimeExportName ─────

test('buildRealtimeExportName: typical session id + txt extension', () => {
  const name = pure.buildRealtimeExportName('rt_abc-123', 'txt', new Date('2026-06-05T12:00:00.000Z'));
  assert.equal(name, 'qasr-realtime-rt_abc-123-2026-06-05T12-00-00-000Z.txt');
});

test('buildRealtimeExportName: sanitizes unsafe session id chars', () => {
  /* Slashes, spaces, dots, colons in session id must collapse
   * to single dashes so the filename is safe. */
  const name = pure.buildRealtimeExportName('a/b c.d:e', 'txt', new Date('2026-06-05T12:00:00.000Z'));
  assert.equal(name, 'qasr-realtime-a-b-c-d-e-2026-06-05T12-00-00-000Z.txt');
});

test('buildRealtimeExportName: empty session id falls back to "session"', () => {
  const name = pure.buildRealtimeExportName('', 'json', new Date('2026-06-05T12:00:00.000Z'));
  assert.equal(name, 'qasr-realtime-session-2026-06-05T12-00-00-000Z.json');
});

test('buildRealtimeExportName: null session id falls back to "session"', () => {
  const name = pure.buildRealtimeExportName(null, 'json', new Date('2026-06-05T12:00:00.000Z'));
  assert.equal(name, 'qasr-realtime-session-2026-06-05T12-00-00-000Z.json');
});

test('buildRealtimeExportName: timestamp is ISO with : and . replaced by -', () => {
  /* Windows / FAT32 cannot store ":" or "." in filenames, so
   * the timestamp has those replaced by "-".  Critical for
   * cross-platform download. */
  const name = pure.buildRealtimeExportName('s', 'txt', new Date('2026-01-02T03:04:05.678Z'));
  assert.ok(!name.includes(':'), 'no colons in filename: ' + name);
  assert.ok(name.match(/^qasr-realtime-s-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z\.txt$/),
    'expected format, got: ' + name);
});

// ───── summary ─────

console.log('');
console.log('────────────────────────────────────');
console.log(`  state_pure: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
process.exit(0);
