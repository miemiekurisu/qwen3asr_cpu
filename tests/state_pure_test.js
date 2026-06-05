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

// ───── summary ─────

console.log('');
console.log('────────────────────────────────────');
console.log(`  state_pure: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
process.exit(0);
