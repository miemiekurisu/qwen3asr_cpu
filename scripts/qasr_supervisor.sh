#!/usr/bin/env bash
#
# qasr_supervisor.sh — Auto-restart qasr_server on crash.
#
# 启停参数全在 run_linux_server.sh 里, supervisor 不持参数.
# 完整 env / flags 参考 docs/CLI.md (single source of truth).

set -e

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_linux_server.sh"
LOG="/tmp/qasr_supervisor.log"

NO_LOOP=0
for arg in "$@"; do
    case "$arg" in
        --no-loop) NO_LOOP=1 ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [--no-loop]

Flags:
  --no-loop        单次启动, 死了不拉起 (调试用)
  -h, --help       打印本帮助

Env vars (透传给 run_linux_server.sh, 详见 docs/CLI.md):
  QASR_MODEL_DIR   Qwen3-ASR-0.6B 模型目录 (必填)

Examples:
  # 永远循环, 死了拉起
  QASR_MODEL_DIR=\$HF/Qwen3-ASR-0.6B $(basename "$0")

  # 单次启动, 死了不拉
  QASR_MODEL_DIR=\$HF/Qwen3-ASR-0.6B $(basename "$0") --no-loop

完整参数 + 其他工具: docs/CLI.md
EOF
            exit 0
            ;;
    esac
done

if [[ -z "${QASR_MODEL_DIR:-}" ]]; then
    echo "[supervisor] 错误: 未设置 \$QASR_MODEL_DIR" >&2
    exit 1
fi
if [[ ! -x "$RUNNER" ]]; then
    echo "[supervisor] 错误: 找不到 $RUNNER" >&2
    exit 1
fi

log() {
    printf "[supervisor %s] %s\n" "$(date '+%F %T')" "$*"
}

log "启动 (runner=$RUNNER model=$QASR_MODEL_DIR no_loop=$NO_LOOP)"

run_once() {
    pkill -9 qasr_server 2>/dev/null || true
    sleep 1
    # 不让 run_linux_server.sh 自己 fork 走 — 它前台跑, 我们这里捕获 exit
    "$RUNNER" --detach
    # 监控
    local pid_file=/tmp/qasr_server.pid
    if [[ -f "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file")"
        while kill -0 "$pid" 2>/dev/null; do
            sleep 1
        done
        log "PID $pid 已退出 (code=$?)"
        return "$?"
    fi
    return 1
}

if [[ $NO_LOOP -eq 1 ]]; then
    run_once
    exit $?
fi

# 守护循环
while true; do
    run_once || true
    log "5s 后拉起"
    sleep 5
done
