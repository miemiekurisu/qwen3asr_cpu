#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# build_cuda.sh — One-click build & test for Qwen3-ASR CUDA backend
# Linux CUDA only (DGX Spark / sm_121)
#
# Usage:
#   ./build_cuda.sh                    # build + short audio test
#   ./build_cuda.sh --long             # build + short + long audio test
#   ./build_cuda.sh --clean            # full clean rebuild + short test
#   ./build_cuda.sh --clean --long     # full clean rebuild + both tests
#   ./build_cuda.sh --no-build         # skip build, only run tests
#   ./build_cuda.sh --serve            # start server (bind 0.0.0.0:19991)
#   ./build_cuda.sh --serve --https    # start server with HTTPS (port 443→19991)
#   ./build_cuda.sh --serve --port 8080 # custom port
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_DIR="${SCRIPT_DIR}/build-dgx"
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'

DO_LONG=false; DO_CLEAN=false; NO_BUILD=false; DO_SERVE=false
SERVE_PORT=19991; SERVE_HOST="0.0.0.0"; SERVE_BACKEND=""
for arg in "$@"; do
    case "$arg" in
        --long) DO_LONG=true ;;
        --clean) DO_CLEAN=true ;;
        --no-build) NO_BUILD=true ;;
        --serve) DO_SERVE=true ;;
        --port) shift; SERVE_PORT="$1" ;;
        --host) shift; SERVE_HOST="$1" ;;
        --backend) shift; SERVE_BACKEND="$1" ;;
        --help)
            echo "Usage: $0 [--clean] [--long] [--no-build] [--serve] [--port N] [--backend cpu|cuda]"
            echo "  --clean    Full clean rebuild"
            echo "  --long     Include long.mp3 test"
            echo "  --no-build Skip build, only run tests"
            echo "  --serve    Start server (bind 0.0.0.0:19991)"
            echo "  --port N   Server port (default: 19991)"
            echo "  --backend  Inference backend (default: cpu)"
            exit 0 ;;
        *) echo "Unknown option: $arg (use --help)"; exit 1 ;;
    esac
done

MODEL_0_6B="/home/wink/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
MODEL_1_7B="/home/wink/.cache/huggingface/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
TEST_SHORT="testfile/short.mp3"
TEST_LONG="testfile/long.mp3"

step()  { echo -e "${BOLD}[$1/$2] $3${NC}"; }
fail()  { echo -e "${RED}ERROR: $1${NC}"; exit 1; }
ok()    { echo -e "  ${GREEN}$1${NC}"; }
skip()  { echo -e "  ${YELLOW}skip: $1${NC}"; }

TOTAL_STEPS=7

echo ""
echo -e "${BOLD}=================================================${NC}"
echo -e "${BOLD}  Qwen3-ASR CUDA Build & Test${NC}"
echo -e "${BOLD}=================================================${NC}"

# ── 1. CUDA environment ─────────────────────────────────────
step 1 $TOTAL_STEPS "CUDA environment check"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"
if ! command -v "$NVCC" &>/dev/null; then
    # fallback
    CUDA_HOME="/usr/local/cuda-13.0"
    NVCC="${CUDA_HOME}/bin/nvcc"
    if ! command -v "$NVCC" &>/dev/null; then
        fail "nvcc not found. Set CUDA_HOME or install CUDA toolkit."
    fi
fi

CUBLAS=""
for f in "${CUDA_HOME}/lib64/libcublas.so" "${CUDA_HOME}/lib64/libcublas.so.12"; do
    [ -f "$f" ] && { CUBLAS="$f"; break; }
done
[ -n "$CUBLAS" ] || echo -e "  ${YELLOW}WARNING: libcublas not found${NC}"

DEVICE_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
echo "  nvcc     : $($NVCC --version | head -1)"
echo "  cuda_home: ${CUDA_HOME}"
echo "  cublas   : ${CUBLAS:-not found}"
echo "  gpu      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) (x${DEVICE_COUNT})"

if [ ! -f "$MODEL_0_6B"/*.safetensors 2>/dev/null ] && [ ! -f "$MODEL_0_6B"/model.safetensors 2>/dev/null ]; then
    fail "Model not found at ${MODEL_0_6B}"
fi
ok "CUDA environment OK"

# ── 2. CMake configuration ──────────────────────────────────
step 2 $TOTAL_STEPS "CMake configuration"

if [ "$NO_BUILD" = true ]; then
    skip "build skipped"
elif [ "$DO_CLEAN" = true ]; then
    echo "  Cleaning ${BUILD_DIR}..."
    rm -rf "$BUILD_DIR"
fi

if [ "$NO_BUILD" = false ]; then
    mkdir -p "$BUILD_DIR"
    cmake -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DQASR_ENABLE_TESTS=ON \
        -DQASR_ENABLE_CPU_BACKEND=ON \
        -DQASR_ENABLE_CUDA_BACKEND=ON \
        -DQASR_CUDA_PLATFORM=dgx_spark \
        -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DCMAKE_CUDA_COMPILER="${NVCC}" \
        -DQASR_ENABLE_SILERO_VAD=OFF
    ok "CMake done"
fi

# ── 3. Build ────────────────────────────────────────────────
step 3 $TOTAL_STEPS "Building"

if [ "$NO_BUILD" = true ]; then
    skip "build skipped"
else
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    ok "Build done"
fi

# ── 4. Unit tests (skip if --serve) ──────────────────────
if [ "$DO_SERVE" = false ]; then
step 4 $TOTAL_STEPS "Unit tests"

UNIT_BIN="${BUILD_DIR}/qasr_unit_tests"
[ -x "$UNIT_BIN" ] || fail "Binary not found: $UNIT_BIN"
"$UNIT_BIN" 2>&1 | grep "all tests passed" || fail "Unit tests failed"
ok "Unit tests passed"

# ── 5. Short audio: verify (CPU vs CUDA) ───────────────────
step 5 $TOTAL_STEPS "Short audio: output verification (0.6B)"

V2_BIN="${BUILD_DIR}/qasr_v2_test"
[ -x "$V2_BIN" ] || fail "Binary not found: $V2_BIN"
VERIFY_OUTPUT=$(timeout 180 "$V2_BIN" "$MODEL_0_6B" "$TEST_SHORT" verify 2>&1)

CPU_TEXT=$(echo "$VERIFY_OUTPUT" | sed -n '/VERIFY: CPU vs CUDA/,/CUDA --- ===/p' | grep "Text:" | head -1)
CUDA_TEXT=$(echo "$VERIFY_OUTPUT" | sed -n '/CUDA --- ===/,$p' | grep "Text:" | head -1)

if [ "$CPU_TEXT" != "$CUDA_TEXT" ]; then
    echo -e "${RED}  Output MISMATCH!${NC}"
    echo "  CPU: $CPU_TEXT"
    echo "  CUDA: $CUDA_TEXT"
    exit 1
fi
echo "  $CPU_TEXT"
CPU_TOTAL=$(echo "$VERIFY_OUTPUT" | sed -n '/VERIFY: CPU vs CUDA/,/CUDA --- ===/p' | grep "Total:" | head -1)
CUDA_TOTAL=$(echo "$VERIFY_OUTPUT" | sed -n '/CUDA --- ===/,$p' | grep "Total:" | head -1)
echo "  CPU: $CPU_TOTAL"
echo "  CUDA: $CUDA_TOTAL"
ok "Output matches CPU"

# ── 6. Short audio: timing breakdown ───────────────────────
step 6 $TOTAL_STEPS "Short audio: timing (0.6B + 1.7B)"

echo "── 0.6B short.mp3 (from verify) ──"
echo "$VERIFY_OUTPUT" \
    | sed -n '/CUDA --- ===/,$p' \
    | grep -E "(enc=|decode=|Text:|Tokens:|Total:)" \
    | sed 's/^/  /'

echo ""
echo "── 1.7B short.mp3 ──"
timeout 180 "${BUILD_DIR}/qasr_v2_test" "$MODEL_1_7B" "$TEST_SHORT" cuda 2>&1 \
    | grep -E "(enc=|decode=)" \
    | sed 's/^/  /'
"${BUILD_DIR}/qasr_v2_test" "$MODEL_1_7B" "$TEST_SHORT" cuda 2>&1 \
    | grep -E "(Text:|Tokens:|Total:)" \
    | sed 's/^/  /'

# ── 7. Long audio test ─────────────────────────────────────
step 7 $TOTAL_STEPS "Long audio test (0.6B)"

if [ "$DO_LONG" = true ]; then
    echo "── 0.6B long.mp3 (~28.8 min) ──"
    timeout 900 "${BUILD_DIR}/qasr_v2_test" "$MODEL_0_6B" "$TEST_LONG" cuda 2>&1 \
        | grep -E "(enc=|decode=|Text:|Tokens:|Total:)" \
        | tail -6 \
        | sed 's/^/  /'
else
    skip "use --long to test long.mp3"
fi
else
    skip "tests skipped (--serve)"
fi

# ── 8. Start server (--serve) ─────────────────────────────
if [ "$DO_SERVE" = true ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
    step $TOTAL_STEPS $TOTAL_STEPS "Starting server"

    SERVER_BIN="${BUILD_DIR}/qasr_server"
    [ -x "$SERVER_BIN" ] || fail "Server binary not found: $SERVER_BIN"

    # Kill existing
    if pgrep -f "qasr_server.*--port ${SERVE_PORT}" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}Existing server on port $SERVE_PORT, killing...${NC}"
        pkill -f "qasr_server.*--port ${SERVE_PORT}" || true
        sleep 2
    fi

    echo "  model:      $MODEL_0_6B"
    echo "  host:       $SERVE_HOST"
    echo "  port:       $SERVE_PORT"
    echo "  backend:    ${SERVE_BACKEND:-cpu}"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""

    SERVER_ARGS=(
        --model-dir "$MODEL_0_6B"
        --host "$SERVE_HOST"
        --port "$SERVE_PORT"
        --verbosity 1
    )
    [[ -n "$SERVE_BACKEND" ]] && SERVER_ARGS+=(--backend "$SERVE_BACKEND")
    exec "$SERVER_BIN" "${SERVER_ARGS[@]}"
fi

# ── Summary ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=================================================${NC}"
echo -e "${GREEN}  All done.${NC}"
echo ""
echo -e "  Reference (CPU, previous runs):"
echo -e "  0.6B short: ~1900-2000 ms  (I see someone here...)"
echo -e "  1.7B short: ~2676 ms       (我这边有人...)"
echo ""
echo -e "  Long audio RTF (real-time factor):"
echo -e "  1726 s audio  /  164 s  =  10.5x RTF  (0.6B)"
echo ""
echo -e "${BOLD}=================================================${NC}"
