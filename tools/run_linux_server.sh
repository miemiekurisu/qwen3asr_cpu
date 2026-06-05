#!/usr/bin/env bash
#
# run_linux_server.sh — One-key Linux launcher for qwen3asr_cpu (server + optional HTTPS proxy).
#
# 用法:
#   tools/run_linux_server.sh                       # 前台 qasr_server (HTTP)
#   tools/run_linux_server.sh --detach              # 后台 qasr_server (HTTP)
#   tools/run_linux_server.sh --detach --https      # 后台 qasr_server + HTTPS proxy (推荐)
#   tools/run_linux_server.sh --stop                # 停 --detach 起的 server 和 proxy
#   tools/run_linux_server.sh --status              # 查 HTTP + HTTPS health
#   tools/run_linux_server.sh --https-info          # 显示 cert / proxy 状态
#
# 必填环境变量:
#   QASR_MODEL_DIR         Qwen3-ASR-0.6B 目录 (含 model.safetensors)
#                          例: export QASR_MODEL_DIR=$HOME/models/Qwen3-ASR-0.6B
#
# 可选环境变量 (有合理默认):
#   QASR_REALTIME_MODEL_DIR  实时/host-capture 模型目录 (默认 = 跟 batch 同 = 共享内存)
#                            推荐: 0.6B 跑 realtime, 1.7B 跑 batch
#                            例: 1.7B 离线高质量 + 0.6B 实时低延迟
#   QASR_HOST          监听地址 (默认 0.0.0.0)
#   QASR_PORT          监听端口 (默认 19991)
#   QASR_UI_DIR        UI 目录 (默认 $PROJECT_ROOT/ui)
#   QASR_THREADS       推理线程数 (默认 0 = 自动)
#   QASR_VERBOSITY     0=silent, 1=commit, 2=per-poll, 3=raw (默认 0)
#   QASR_VAD_MODEL     Silero VAD ONNX 模型路径
#                      (默认 $PROJECT_ROOT/models/silero_vad/silero_vad.onnx)
#   QASR_HTTPS_PORT    HTTPS 代理端口 (默认 19992, 仅 --https 时生效)
#
# HTTPS:
#   --https  同时启动 tools/https_proxy.py 反代, 自动 mktemp -d 生成自签 cert
#            (每次启动 cert 都不一样, 退出时自动删, 仓库卫生 + 临时安全)
#   浏览器访问 https://<host>:<https-port>/, 首次警告选"高级→继续"即可
#   想跨重启复用 cert: 设 QASR_TLS_CERT_DIR (例: /etc/qasr/tls), 配合 --https
#
# 路径:
#   QASR_PROJECT_ROOT  项目根 (默认自动探测 = 脚本 ../)
#   QASR_BUILD_DIR     编译输出 (默认 $PROJECT_ROOT/build/linux-openblas,
#                        对应 build_linux.sh 装到 build/linux-openblas/)
#   QASR_LOG_FILE      server 日志 (默认 /tmp/qasr_server.log, --detach 才用)
#   QASR_PID_FILE      server PID 文件 (默认 /tmp/qasr_server.pid)
#   QASR_PROXY_LOG     proxy  日志 (默认 /tmp/qasr_proxy.log, --https 才用)
#   QASR_PROXY_PID     proxy  PID 文件 (默认 /tmp/qasr_proxy.pid)
#   QASR_TLS_CERT_DIR  cert 目录 (默认 mktemp -d, 退出时删)

set -euo pipefail

# ─────────────── 颜色 ───────────────
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    C_RED="$(tput setaf 1)"; C_GRN="$(tput setaf 2)"; C_YEL="$(tput setaf 3)"
    C_BLU="$(tput setaf 4)"; C_DIM="$(tput dim)"; C_BLD="$(tput bold)"; C_RST="$(tput sgr0)"
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_DIM=""; C_BLD=""; C_RST=""
fi
log_info()  { printf "${C_BLU}[INFO]${C_RST}  %s\n" "$*"; }
log_ok()    { printf "${C_GRN}[OK]${C_RST}    %s\n" "$*"; }
log_warn()  { printf "${C_YEL}[WARN]${C_RST}  %s\n" "$*" >&2; }
log_err()   { printf "${C_RED}[ERROR]${C_RST} %s\n" "$*" >&2; }
log_step()  { printf "\n${C_BLD}${C_BLU}── %s ──${C_RST}\n" "$*"; }

# ─────────────── 路径 ───────────────
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

if [[ -n "${QASR_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$QASR_PROJECT_ROOT"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
[[ -d "$PROJECT_ROOT" ]] || { log_err "项目根不存在: $PROJECT_ROOT"; exit 1; }

# ─────────────── 默认值 ───────────────
HOST="${QASR_HOST:-0.0.0.0}"
PORT="${QASR_PORT:-19991}"
UI_DIR="${QASR_UI_DIR:-$PROJECT_ROOT/ui}"
THREADS="${QASR_THREADS:-0}"
VERBOSITY="${QASR_VERBOSITY:-0}"
VAD_MODEL="${QASR_VAD_MODEL:-$PROJECT_ROOT/models/silero_vad/silero_vad.onnx}"
BUILD_DIR="${QASR_BUILD_DIR:-$PROJECT_ROOT/build/linux-openblas}"
LOG_FILE="${QASR_LOG_FILE:-/tmp/qasr_server.log}"
PID_FILE="${QASR_PID_FILE:-/tmp/qasr_server.pid}"
PROXY_SCRIPT="${QASR_PROXY_SCRIPT:-$SCRIPT_DIR/https_proxy.py}"
PROXY_LOG="${QASR_PROXY_LOG:-/tmp/qasr_proxy.log}"
PROXY_PID="${QASR_PROXY_PID:-/tmp/qasr_proxy.pid}"
HTTPS_PORT="${QASR_HTTPS_PORT:-19992}"
TLS_CERT_DIR="${QASR_TLS_CERT_DIR:-}"
MODEL_DIR="${QASR_MODEL_DIR:-}"
REALTIME_MODEL_DIR="${QASR_REALTIME_MODEL_DIR:-}"
DETACHED=0
DO_STOP=0
DO_STATUS=0
DO_HTTPS_INFO=0
USE_HTTPS=0

# ─────────────── 参数解析 ───────────────
usage() {
    cat <<EOF
用法: $(basename "$0") [选项]

选项:
  --detach            后台启动 qasr_server, 写 PID 到 $PID_FILE
  --https             同时启动 HTTPS 反代 (推荐, 浏览器 mic 权限需要 https)
                      默认 mktemp -d 临时 cert (退出时删), 想持久: 设 QASR_TLS_CERT_DIR
  --stop              停掉 --detach 起的 server (和 proxy, 若有)
  --status            查 /api/health (HTTP + HTTPS, 若后者在跑)
  --https-info        显示 cert 目录 / proxy 状态
  --verbose           覆盖 \$QASR_VERBOSITY=3 (开发用, 一行一 poll)
  -h, --help          显示本帮助

必填环境变量:
  QASR_MODEL_DIR      Qwen3-ASR-0.6B 目录
                      例: export QASR_MODEL_DIR=\$HOME/models/Qwen3-ASR-0.6B

可选环境变量 (有默认):
  QASR_REALTIME_MODEL_DIR  realtime 模型 (默认 = 跟 batch 共享内存)
                           推荐: 1.7B batch + 0.6B realtime
  QASR_HOST=0.0.0.0   QASR_PORT=19991   QASR_HTTPS_PORT=19992
  QASR_THREADS=0      (0=自动)
  QASR_VERBOSITY=0    (0=silent, 1=commit, 2=per-poll, 3=raw)
  QASR_VAD_MODEL=...  (默认 \$PROJECT_ROOT/models/silero_vad/silero_vad.onnx)
  QASR_UI_DIR=...     (默认 \$PROJECT_ROOT/ui)
  QASR_LOG_FILE=...   (默认 /tmp/qasr_server.log)
  QASR_PID_FILE=...   (默认 /tmp/qasr_server.pid)
  QASR_PROXY_LOG=...  (默认 /tmp/qasr_proxy.log, --https 时)
  QASR_PROXY_PID=...  (默认 /tmp/qasr_proxy.pid, --https 时)
  QASR_TLS_CERT_DIR=... 持久化 cert 目录 (默认 mktemp -d, 退出时删)

示例:
  export QASR_MODEL_DIR=\$HOME/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/<rev>
  tools/run_linux_server.sh --detach                  # 后台 HTTP only (API/curl 用)
  tools/run_linux_server.sh --detach --https          # 后台 HTTP + HTTPS (浏览器用, 推荐)

  # 1.7B batch + 0.6B realtime (推荐: 离线要质量, 实时要速度)
  export QASR_MODEL_DIR=\$HOME/.../Qwen3-ASR-1.7B/snapshots/<rev>
  export QASR_REALTIME_MODEL_DIR=\$HOME/.../Qwen3-ASR-0.6B/snapshots/<rev>
  tools/run_linux_server.sh --detach --https

  tools/run_linux_server.sh --stop                    # 停
  tools/run_linux_server.sh --status                  # 健康检查
  tools/run_linux_server.sh --https-info              # cert / proxy 状态

HTTPS 方案对比:
  --https (本工具, 当前推荐)   Python 反代 + 每次新 cert, 0 配置, 临时
  A. qasr_server 自带 TLS        待 mbedTLS 集成
  C. Caddy / nginx 反代         需系统装, 适合生产
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach)         DETACHED=1; shift ;;
        --https)          USE_HTTPS=1; shift ;;
        --stop)           DO_STOP=1; shift ;;
        --status)         DO_STATUS=1; shift ;;
        --https-info)     DO_HTTPS_INFO=1; shift ;;
        --verbose)        VERBOSITY=3; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                log_err "未知选项: $1"; usage >&2; exit 2 ;;
    esac
done

# ─────────────── 校验 ───────────────
check_required() {
    log_step "参数校验"
    if [[ -z "$MODEL_DIR" ]]; then
        log_err "未设置 \$QASR_MODEL_DIR"
        log_err ""
        log_err "请先下载 Qwen3-ASR-0.6B 模型, 然后:"
        log_err "  export QASR_MODEL_DIR=/path/to/Qwen3-ASR-0.6B"
        log_err ""
        log_err "下载方式 (任选一种):"
        log_err "  pip install -U huggingface_hub"
        log_err "  python3 -c \"from huggingface_hub import snapshot_download; \\"
        log_err "      snapshot_download('Qwen/Qwen3-ASR-0.6B', cache_dir='\$HOME/.cache/huggingface')\""
        log_err ""
        log_err "  或 git lfs install && git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B"
        exit 1
    fi
    if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" && ! -f "$MODEL_DIR/model-00001-of-00003.safetensors" ]]; then
        log_err "QASR_MODEL_DIR=$MODEL_DIR 不含 model.safetensors 或分片 (model-00001-of-NNNN.safetensors)"
        log_err "  请检查路径, 或重新下载"
        exit 1
    fi
    log_ok "QASR_MODEL_DIR=$MODEL_DIR"

    # 可选: realtime 模型 (默认 = 跟 batch 共享)
    if [[ -n "$REALTIME_MODEL_DIR" ]]; then
        if [[ ! -d "$REALTIME_MODEL_DIR" ]]; then
            log_err "QASR_REALTIME_MODEL_DIR=$REALTIME_MODEL_DIR 目录不存在"
            exit 1
        fi
        if [[ ! -f "$REALTIME_MODEL_DIR/model.safetensors" && ! -f "$REALTIME_MODEL_DIR/model-00001-of-00002.safetensors" && ! -f "$REALTIME_MODEL_DIR/model-00001-of-00003.safetensors" ]]; then
            log_err "QASR_REALTIME_MODEL_DIR=$REALTIME_MODEL_DIR 不含 model.safetensors"
            exit 1
        fi
        log_ok "QASR_REALTIME_MODEL_DIR=$REALTIME_MODEL_DIR"
    else
        log_ok "QASR_REALTIME_MODEL_DIR=<unset>  (realtime 与 batch 共享内存)"
    fi

    if [[ ! -x "$BUILD_DIR/qasr_server" ]]; then
        log_err "找不到 $BUILD_DIR/qasr_server"
        log_err "  请先跑: tools/build_linux.sh"
        exit 1
    fi
    log_ok "binary:  $BUILD_DIR/qasr_server"

    if [[ ! -f "$VAD_MODEL" ]]; then
        log_warn "Silero VAD 模型不存在: $VAD_MODEL"
        log_warn "  实时 VAD 段式将退化为 40s 强制切段, 但仍可用"
        log_warn "  下载: curl -L -o $VAD_MODEL https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
        VAD_MODEL=""
    else
        log_ok "VAD:      $VAD_MODEL"
    fi

    if [[ $USE_HTTPS -eq 1 ]]; then
        if [[ ! -f "$PROXY_SCRIPT" ]]; then
            log_err "找不到 $PROXY_SCRIPT (--https 需要)"
            exit 1
        fi
        if ! command -v python3 >/dev/null 2>&1; then
            log_err "--https 需要 python3, 但 \$PATH 里没找到"
            exit 1
        fi
        log_ok "HTTPS:    enabled (proxy=$PROXY_SCRIPT, port=$HTTPS_PORT)"
    fi
}

# ─────────────── 停止 ───────────────
stop_pid_file() {
    local label="$1" pid_file="$2"
    if [[ ! -f "$pid_file" ]]; then return 0; fi
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        return 0
    fi
    log_info "停 $label PID $pid"
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        sleep 0.5
        if ! kill -0 "$pid" 2>/dev/null; then
            log_ok "  $label 已停"
            rm -f "$pid_file"
            return 0
        fi
    done
    log_warn "  $label 强杀"
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$pid_file"
}

do_stop() {
    log_step "停止"
    stop_pid_file "proxy"  "$PROXY_PID"
    stop_pid_file "server" "$PID_FILE"
    # 兜底
    pkill -9 -f "build/linux-openblas/qasr_server" 2>/dev/null || true
    pkill -9 -f "build/silero-test/qasr_server"   2>/dev/null || true
    pkill -9 -f "tools/https_proxy.py"            2>/dev/null || true
}

# ─────────────── 状态 ───────────────
do_status() {
    local http="http://127.0.0.1:$PORT/api/health"
    local code
    code="$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$http" 2>/dev/null || echo "000")"
    if [[ "$code" == "200" ]]; then
        log_ok "HTTP  $http  $(curl -s "$http" | head -c 80)"
    else
        log_err "HTTP  $http  code=$code (server 没起?)"
    fi

    if [[ -f "$PROXY_PID" ]] && kill -0 "$(cat "$PROXY_PID" 2>/dev/null)" 2>/dev/null; then
        local https="https://127.0.0.1:$HTTPS_PORT/api/health"
        code="$(curl -sk --max-time 3 -o /dev/null -w '%{http_code}' "$https" 2>/dev/null || echo "000")"
        if [[ "$code" == "200" ]]; then
            log_ok "HTTPS $https  $(curl -sk "$https" | head -c 80)"
        else
            log_err "HTTPS $https  code=$code (proxy 没起来?)"
        fi
    fi
}

# ─────────────── HTTPS info ───────────────
do_https_info() {
    if [[ ! -f "$PROXY_PID" ]]; then
        log_info "proxy 没在跑 (PID 文件不存在: $PROXY_PID)"
        return 0
    fi
    local pid
    pid="$(cat "$PROXY_PID" 2>/dev/null || true)"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        log_warn "proxy PID 文件过期"
        return 0
    fi
    log_ok "proxy PID $pid, log=$PROXY_LOG"
    # 找 cert 路径
    local cert_dir
    cert_dir="$(grep -oE "ephemeral cert dir: [^ ]+" "$PROXY_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')"
    if [[ -n "$cert_dir" && -d "$cert_dir" ]]; then
        log_ok "cert 目录 (ephemeral): $cert_dir"
        ls -la "$cert_dir"
    else
        log_info "cert 目录没找到 (可能持久或不在默认位置)"
        log_info "  看 log: tail $PROXY_LOG"
    fi
}

# ─────────────── 启动 ───────────────
# ─────────────── 端口预检 ───────────────
check_port_free() {
    local port="$1" label="$2"
    if command -v ss >/dev/null 2>&1; then
        if ss -tln "sport = :$port" 2>/dev/null | grep -q ":$port"; then
            local occupant
            occupant="$(ss -tlnp "sport = :$port" 2>/dev/null | grep ":$port" | head -1 | sed -E 's/.*users:\(\("([^,]+)".*/\1/' )"
            log_err "$label 端口 $port 已被占用: $occupant"
            log_err "  解决: 杀旧进程:    kill \$(cat $PID_FILE 2>/dev/null) 2>/dev/null"
            log_err "        或跑本脚本 --stop"
            log_err "        或换端口:    QASR_PORT=19993 tools/run_linux_server.sh --detach --https"
            return 1
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -iTCP:"$port" -sTCP:LISTEN -P -n 2>/dev/null | grep -q ":$port"; then
            local pid
            pid="$(lsof -iTCP:"$port" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1)"
            log_err "$label 端口 $port 已被占用 (PID $pid)"
            return 1
        fi
    fi
    return 0
}

do_start() {
    log_step "启动参数"
    log_info "binary:  $BUILD_DIR/qasr_server"
    log_info "model:   $MODEL_DIR"
    if [[ -n "$REALTIME_MODEL_DIR" ]]; then
        log_info "realtime-model: $REALTIME_MODEL_DIR  (2 个独立实例, 内存吃紧)"
    else
        log_info "realtime-model: <shared with batch>     (0 额外内存)"
    fi
    log_info "ui:      $UI_DIR"

    check_port_free "$PORT"     "server"  || exit 1
    [[ $USE_HTTPS -eq 1 ]] && check_port_free "$HTTPS_PORT" "proxy" || exit 1
    log_info "host:    $HOST"
    log_info "port:    $PORT  (HTTP)"
    [[ $USE_HTTPS -eq 1 ]] && log_info "         $HTTPS_PORT  (HTTPS, --https)"
    log_info "threads: $THREADS  (0=auto)"
    log_info "verbose: $VERBOSITY  (0=silent)"

    local args=(
        --model-dir "$MODEL_DIR"
        --ui-dir    "$UI_DIR"
        --host      "$HOST"
        --port      "$PORT"
        --threads   "$THREADS"
        --verbosity "$VERBOSITY"
    )
    [[ -n "$REALTIME_MODEL_DIR" ]] && args+=(--realtime-model-dir "$REALTIME_MODEL_DIR")

    if [[ $DETACHED -eq 1 ]]; then
        log_step "后台启动 server (日志 $LOG_FILE, PID $PID_FILE)"
        QASR_MODEL_DIR="$MODEL_DIR" \
        QWEN_SILERO_VAD_MODEL="${VAD_MODEL:-}" \
            nohup "$BUILD_DIR/qasr_server" "${args[@]}" > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        log_ok "已起, PID $pid"
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            log_err "启动后立即挂掉, 看日志: $LOG_FILE"
            tail -20 "$LOG_FILE" >&2 || true
            rm -f "$PID_FILE"
            exit 1
        fi

        # HTTPS proxy
        if [[ $USE_HTTPS -eq 1 ]]; then
            log_step "后台启动 HTTPS proxy (日志 $PROXY_LOG, PID $PROXY_PID)"
            local proxy_args=(
                --bind-host  "$HOST"
                --bind-port  "$HTTPS_PORT"
                --upstream   "http://127.0.0.1:$PORT"
            )
            if [[ -n "$TLS_CERT_DIR" ]]; then
                proxy_args+=(--cert-dir "$TLS_CERT_DIR" --reuse-cert)
            fi
            nohup python3 "$PROXY_SCRIPT" "${proxy_args[@]}" > "$PROXY_LOG" 2>&1 &
            local ppid=$!
            echo "$ppid" > "$PROXY_PID"
            log_ok "proxy PID $ppid"
            sleep 2
            if ! kill -0 "$ppid" 2>/dev/null; then
                log_err "proxy 启动后挂掉, 看日志: $PROXY_LOG"
                tail -10 "$PROXY_LOG" >&2 || true
                # 杀 server 一起清掉
                kill "$pid" 2>/dev/null || true
                rm -f "$PID_FILE" "$PROXY_PID"
                exit 1
            fi
        fi

        log_info "等 1-3s 让模型加载..."
        sleep 3
        do_status
        echo
        log_info "停止: $(basename "$0") --stop"
        if [[ $USE_HTTPS -eq 1 ]]; then
            log_info "HTTPS URL: https://<lan-ip>:$HTTPS_PORT/"
            log_info "  浏览器首次警告选'高级→继续'即可"
        fi
    else
        # 前台 + (可选) proxy 后台
        if [[ $USE_HTTPS -eq 1 ]]; then
            log_step "后台启动 HTTPS proxy (前台 Ctrl+C 时它继续跑, 用 --stop 杀)"
            local proxy_args=(
                --bind-host  "$HOST"
                --bind-port  "$HTTPS_PORT"
                --upstream   "http://127.0.0.1:$PORT"
            )
            if [[ -n "$TLS_CERT_DIR" ]]; then
                proxy_args+=(--cert-dir "$TLS_CERT_DIR" --reuse-cert)
            fi
            nohup python3 "$PROXY_SCRIPT" "${proxy_args[@]}" > "$PROXY_LOG" 2>&1 &
            local ppid=$!
            echo "$ppid" > "$PROXY_PID"
            log_ok "proxy PID $ppid, 端口 $HTTPS_PORT"
            trap 'kill "$ppid" 2>/dev/null; rm -f "$PROXY_PID"' EXIT
        fi
        log_step "前台启动 server (Ctrl+C 退出)"
        exec env QASR_MODEL_DIR="$MODEL_DIR" \
                QWEN_SILERO_VAD_MODEL="${VAD_MODEL:-}" \
                "$BUILD_DIR/qasr_server" "${args[@]}"
    fi
}

# ─────────────── 主流程 ───────────────
main() {
    if [[ $DO_STOP -eq 1 ]]; then
        do_stop
        exit $?
    fi
    if [[ $DO_STATUS -eq 1 ]]; then
        do_status
        exit $?
    fi
    if [[ $DO_HTTPS_INFO -eq 1 ]]; then
        do_https_info
        exit $?
    fi
    check_required
    do_start
}

main "$@"
