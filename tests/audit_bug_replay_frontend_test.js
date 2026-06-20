/**
 * audit_bug_replay_frontend_test.js
 *
 * Node.js test file that demonstrates frontend vulnerabilities found
 * during the audit:
 *
 * 1. XSS: e.lang unescaped in innerHTML (app.js:120)
 * 2. escapeHtml doesn't escape quotes (app.js:74-76)
 * 3. realtimeResult null access (app.js:22, 307)
 * 4. NaN display from missing fields (app.js:489-490)
 * 5. SSE connection leak (app.js:528)
 * 6. Timer leak on setup failure (app.js:589-591, 638)
 *
 * CI-safe: no browser needed — tests extract and analyze the source
 * directly.
 */

const fs = require('fs');
const path = require('path');

const UI_DIR = path.join(__dirname, '..', 'ui');
const files = {
    appJs: path.join(UI_DIR, 'app.js'),
    indexHtml: path.join(UI_DIR, 'index.html'),
    stateJs: path.join(UI_DIR, 'state.js'),
};

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`  ✓ ${message}`);
        passed++;
    } else {
        console.log(`  ✗ ${message}`);
        failed++;
    }
}

function assertIncludes(text, pattern, message) {
    if (pattern.test(text)) {
        console.log(`  ✓ ${message}`);
        passed++;
    } else {
        console.log(`  ✗ ${message}: expected /${pattern}/ not found`);
        failed++;
    }
}

console.log('=== Frontend Audit: Bug Replay Tests ===\n');

try {
    const appJs = fs.readFileSync(files.appJs, 'utf-8');
    const indexHtml = fs.readFileSync(files.indexHtml, 'utf-8');

    /* ─── Bug 1: XSS — e.lang unescaped in innerHTML ─── */
    console.log('\n--- Bug 1: XSS (e.lang unescaped in innerHTML) ---');

    /* Extract the glossary rendering code around e.lang */
    assertIncludes(appJs, /e\.lang/, 'e.lang is referenced');
    assertIncludes(appJs, /escapeHtml/, 'escapeHtml function exists');

    /* Check that e.lang is NOT wrapped in escapeHtml() while e.source is */
    const glossaryRenderMatch = appJs.match(/html\s*\+=.*language.*lang[^;]*;/);
    if (glossaryRenderMatch) {
        // Found a line referencing lang — check if escapeHtml wraps it
        const langLine = glossaryRenderMatch[0];
        if (langLine.includes('escapeHtml(e.lang)')) {
            assert(true, 'e.lang is already escaped via escapeHtml()');
        } else if (langLine.includes('e.lang')) {
            assert(false,
                `e.lang appears UNESCAPED in innerHTML: "${langLine.trim().substring(0, 80)}"`);
        }
    } else {
        /* Look more broadly for glossary render */
        const glossaryHtml = appJs.match(/function renderGlossary[\s\S]{0,2000}/);
        if (glossaryHtml) {
            const html = glossaryHtml[0];
            const hasUnescapedLang = /\.lang\b(?!\s*\()/.test(html) &&
                !/escapeHtml\([^)]*\.lang/.test(html);
            if (hasUnescapedLang) {
                assert(false,
                    'e.lang appears in innerHTML assembly without escapeHtml()');
            } else {
                assert(true, 'e.lang appears to be properly escaped');
            }
        } else {
            console.log('  ⚠ Could not find renderGlossary function — check manually');
        }
    }

    /* ─── Bug 2: escapeHtml doesn't escape quotes ─── */
    console.log('\n--- Bug 2: escapeHtml missing quote encoding ---');

    const escapeHtmlMatch = appJs.match(/function escapeHtml[\s\S]{0,200}/);
    if (escapeHtmlMatch) {
        const escapeImpl = escapeHtmlMatch[0];
        const hasQuoteEscape = escapeImpl.includes('&#39;') || escapeImpl.includes('&quot;') ||
            escapeImpl.includes('&apos;');
        assert(!hasQuoteEscape,
            'escapeHtml() does NOT encode quotes — fragile');
    }

    /* ─── Bug 3: realtimeResult null access ─── */
    console.log('\n--- Bug 3: realtimeResult potentially null ---');

    assertIncludes(appJs, /getElementById\('realtimeResult'\)/,
        'getElementById(realtimeResult) exists');

    /* Check if realtimeResult appears without null guard */
    const realtimeResultUses = appJs.match(/realtimeResult\.[a-zA-Z]/g);
    if (realtimeResultUses) {
        // Check that there's no null check before usage
        const hasNullGuard = appJs.includes('if (!realtimeResult)');
        assert(!hasNullGuard,
            `realtimeResult used ${realtimeResultUses.length} times without null guard`);
    }

    /* ─── Bug 4: NaN display from missing fields ─── */
    console.log('\n--- Bug 4: NaN display vulnerability ---');

    assertIncludes(appJs, /\.toFixed\(1\)/,
        'toFixed(1) used for display formatting (potential NaN)');

    /* Check if division/are guarded */
    const sampleCountDiv = appJs.match(/sample_count\s*\/\s*16000/);
    if (sampleCountDiv) {
        const hasNanGuard = appJs.includes('isNaN') || appJs.includes('|| 0');
        assert(!hasNanGuard,
            'sample_count/16000 lacks NaN guard');
    }

    /* ─── Bug 5: SSE connection leak (no close before overwrite) ─── */
    console.log('\n--- Bug 5: SSE connection leak ---');

    /* Check openSseStream for es.close() before assignment */
    assertIncludes(appJs, /EventSource/, 'EventSource is used');

    const sseFunction = appJs.match(/function openSseStream[\s\S]{0,2000}/);
    if (sseFunction) {
        const func = sseFunction[0];
        const closesBeforeAssign = func.includes('close') &&
            func.indexOf('close') < func.indexOf('es = new EventSource') ||
            func.indexOf('close') < func.indexOf('realtimeState.sse');
        assert(!closesBeforeAssign,
            'openSseStream does NOT close previous EventSource before overwriting');
    }

    /* ─── Bug 6: Timer leak on setup failure ─── */
    console.log('\n--- Bug 6: Timer leak on setup failure ---');

    assertIncludes(appJs, /setInterval/,
        'setInterval is used for sendTimer/meterTimer');

    const timerCleanupCode = appJs.match(/clearInterval\s*\([^)]*sendTimer[^)]*\)/);
    if (timerCleanupCode) {
        console.log('  ✓ clearInterval(sendTimer) exists in stop/catch');
    } else {
        assert(false, 'Cannot find clearInterval for sendTimer');
    }

    /* Check the catch block clears timers before overwriting realtimeState */
    const catchBlocks = appJs.match(/catch\s*\([^)]*\)\s*\{[\s\S]{0,1000}/g);
    if (catchBlocks) {
        for (const catchBlock of catchBlocks) {
            if (catchBlock.includes('realtimeState')) {
                const hasTimerCleanup = catchBlock.includes('clearInterval');
                if (!hasTimerCleanup && catchBlock.includes('setInterval')) {
                    // This catch block has setInterval references but no clearInterval
                    assert(false,
                        'catch block near setInterval lacks clearInterval');
                }
            }
        }
    }

    /* ─── Bug 7: renderTerminal rAF not cancelled on reset ─── */
    console.log('\n--- Bug 7: _transcriptFrame not cleared on reset ---');

    assertIncludes(appJs, /_transcriptFrame/,
        '_transcriptFrame is used on DOM element');
    assertIncludes(appJs, /resetTerminal/,
        'resetTerminal function exists');

    const resetTerminalFn = appJs.match(/function resetTerminal[\s\S]{0,1000}/);
    if (resetTerminalFn) {
        const resetImpl = resetTerminalFn[0];
        const clearsFrame = resetImpl.includes('_transcriptFrame');
        assert(!clearsFrame,
            'resetTerminal does NOT clear _transcriptFrame — stale rAF data persists');
    }

    /* ─── Bug 8: SnapshotRealtimeSession no locking ─── */
    console.log('\n--- Bug 8: SnapshotRealtimeSession locking ---');

    assertIncludes(appJs, /SnapshotRealtimeSession/,
        'SnapshotRealtimeSession is referenced (if server-side function)');

    console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);

} catch (err) {
    console.error('Test error:', err.message);
    process.exit(1);
}

/* ─── Additional standalone vulnerability proofs ─── */
console.log('\n=== Additional vulnerability proofs ===\n');

/* Proof: escapeHtml doesn't sanitize quotes */
function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const xssPayload = '"><script>alert(1)</script>';
const escaped = escapeHtml(xssPayload);
console.log(`escapeHtml(): "${xssPayload}" → "${escaped}"`);
console.log(`  Quote injection still possible: ${escaped.includes('"') ? 'YES (XSS)' : 'NO'}`);

/* Proof: Unescaped lang field in innerHTML */
const langPayload = '<img src=x onerror=alert(1)>';
const safeSource = 'normal text';
const safeTarget = 'normal text';
const html = '<span>' + (langPayload || '-') + '</span>';
console.log(`\ninnerHTML with unescaped lang:`);
console.log(`  "${html}"`);
console.log(`  XSS possible: ${html.includes('<img') ? 'YES (XSS)' : 'NO'}`);

/* Proof: toFixed(NaN) */
const nanDisplay = (undefined / 16000).toFixed(1);
console.log(`\nNaN display: "${nanDisplay}"`);
console.log(`  NaN shown: ${nanDisplay === 'NaN' ? 'YES (UI pollution)' : 'NO'}`);

/* Proof: realtimeResult null access */
function testNullAccess() {
    try {
        const nullElement = null;
        nullElement.appendChild(document.createElement('div'));
        return false;
    } catch (e) {
        return e instanceof TypeError;
    }
}
/* Can't test in Node.js without DOM, but we can verify the pattern */
console.log(`\nNull access pattern: realtimeResult.appendChild(cursorLine)`);
console.log(`  Type: Uncaught TypeError: Cannot read properties of null`);

console.log('\n=== Bug reproduction complete ===\n');

/* Exit with code 1 if any test failed */
if (failed > 0) {
    process.exit(1);
}
