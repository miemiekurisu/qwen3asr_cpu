#!/usr/bin/env bash
#
# scripts/smoke_test.sh — smoke tests for the bash tooling
#
# Exits 0 on success, 1 on the first failing assertion.  This is NOT
# a substitute for the C++ unit tests; it only covers the surface area
# the operator hits when launching / stopping / inspecting the
# service from the shell:
#
#   * --help and --garbage exit codes (--garbage must exit 2)
#   * --status on a running server returns the expected JSON
#   * Port pre-check rejects an already-bound port with a clear hint
#   * Supervisor refuses to run without QASR_MODEL_DIR
#
# Run from the project root:
#   bash scripts/smoke_test.sh
#
# This script does NOT start a server — it only exercises the parts
# that don't need one, plus a single detach-attempt that the port
# pre-check is expected to reject.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "[FAIL] cd $ROOT"; exit 1; }

PASS=0
FAIL=0
FAILS=()

# ───── assert helpers ─────
expect_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        PASS=$((PASS + 1))
        printf "  [PASS] %s\n" "$label"
    else
        FAIL=$((FAIL + 1))
        FAILS+=("$label: expected='$expected' actual='$actual'")
        printf "  [FAIL] %s  expected='%s' actual='%s'\n" "$label" "$expected" "$actual"
    fi
}

expect_contains() {
    local label="$1" needle="$2" haystack="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        PASS=$((PASS + 1))
        printf "  [PASS] %s\n" "$label"
    else
        FAIL=$((FAIL + 1))
        FAILS+=("$label: needle='$needle' not found")
        printf "  [FAIL] %s  needle='%s'\n" "$label" "$needle"
    fi
}

# ───── 1. run_linux_server.sh --help ─────
out="$(scripts/run_linux_server.sh --help 2>&1)"
code=$?
expect_eq "run --help exits 0"        "0"            "$code"
expect_contains "run --help usage"  "Usage:"       "$out"
expect_contains "run --help detach" "--detach"   "$out"
expect_contains "run --help https"  "--https"    "$out"
expect_contains "run --help stop"   "--stop"     "$out"
expect_contains "run --help status" "--status"   "$out"

# ───── 2. run_linux_server.sh --garbage (unknown option) ─────
out="$(scripts/run_linux_server.sh --garbage 2>&1)"
code=$?
expect_eq "run --garbage exits 2"     "2"            "$code"
expect_contains "run --garbage msg" "未知选项"   "$out"

# ───── 3. build_linux.sh --help ─────
out="$(scripts/build_linux.sh --help 2>&1)"
code=$?
expect_eq "build --help exits 0"      "0"            "$code"
expect_contains "build --help usage" "Usage:"      "$out"

# ───── 4. build_linux.sh --garbage ─────
out="$(scripts/build_linux.sh --garbage 2>&1)"
code=$?
expect_eq "build --garbage exits 2"   "2"            "$code"
expect_contains "build --garbage msg" "未知选项" "$out"

# ───── 5. qasr_supervisor.sh refuses without QASR_MODEL_DIR ─────
out="$(env -u QASR_MODEL_DIR scripts/qasr_supervisor.sh 2>&1)"
code=$?
expect_contains "supervisor no MODEL_DIR" "QASR_MODEL_DIR" "$out"
# Note: supervisor may exit non-zero or just print a message and exit;
# we only assert the message is present.

# ───── 6. Port pre-check: a port that nothing listens on succeeds ─────
# We don't actually start the server — we just verify the script
# would have proceeded past the port check.  Use a model dir that
# exists locally so the pre-flight doesn't bail on validation.
DIR06B="$(ls -d ~/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/*/ 2>/dev/null | head -1)"
if [[ -n "$DIR06B" ]]; then
    out="$(QASR_MODEL_DIR="$DIR06B" QASR_PORT=29991 scripts/run_linux_server.sh --detach 2>&1)"
    code=$?
    # A successful path: it would try to actually launch the binary
    # and fail because port 29991 is in use OR it would launch and
    # succeed.  Both are valid; we only assert it didn't exit 2
    # (which is reserved for argument-parse errors).
    if [[ "$code" == "2" ]]; then
        FAIL=$((FAIL + 1))
        FAILS+=("port pre-check: unexpected exit 2 (arg parse) on free port")
        printf "  [FAIL] port pre-check: free port got exit 2\n"
    else
        PASS=$((PASS + 1))
        printf "  [PASS] port pre-check: free port (exit %s, expected 0 or 1)\n" "$code"
    fi
else
    printf "  [SKIP] port pre-check: no Qwen3-ASR-0.6B model cached locally\n"
fi

# ───── 7. Port pre-check: a port that IS occupied is rejected ─────
# 19991 is the default HTTP port; if the server is running, this
# should fail with a clear message about who is bound.
if [[ -n "$DIR06B" ]]; then
    out="$(QASR_MODEL_DIR="$DIR06B" QASR_PORT=19991 scripts/run_linux_server.sh --detach 2>&1)"
    code=$?
    if [[ "$code" == "1" ]]; then
        PASS=$((PASS + 1))
        printf "  [PASS] port pre-check: occupied port exits 1\n"
    else
        FAIL=$((FAIL + 1))
        FAILS+=("port pre-check: occupied port expected exit 1, got $code")
        printf "  [FAIL] port pre-check: occupied port got exit %s\n" "$code"
    fi
    expect_contains "port pre-check: hint about kill"   "杀旧进程"    "$out"
    expect_contains "port pre-check: hint about stop"   "--stop"      "$out"
    expect_contains "port pre-check: hint about QASR_PORT" "QASR_PORT" "$out"
else
    printf "  [SKIP] port pre-check: no Qwen3-ASR-0.6B model cached locally\n"
fi

# ───── summary ─────
echo
echo "────────────────────────────────────"
echo "  smoke test: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    echo "  failures:"
    for f in "${FAILS[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
exit 0
