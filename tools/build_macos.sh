#!/usr/bin/env bash
#
# build_macos.sh — One-key macOS build script for qwen3asr_cpu
#
# 工作流程:
#   1. 校验系统 (macOS only)
#   2. 检查编译工具链 (clang++/cmake/ninja/ffmpeg/git/curl)
#   3. 检查 Accelerate 框架 (macOS 自带, 无需额外安装)
#   4. 探测模型 (HF 缓存 / $QASR_MODEL_DIR / ./models/...) → 缺失则提示下载
#   5. 探测测试音频 (testfile/*.wav) → 缺失则提示拉取
#   6. clean → cmake --preset macos-accelerate → build → ctest
#   7. (可选) --bench 跑 qasr_cpu_bench
#
# 用法:
#   tools/build_macos.sh [选项]
#
# 常用:
#   tools/build_macos.sh                          # 默认 clean + configure + build + test
#   tools/build_macos.sh --incremental            # 增量编译 (不删 build/)
#   tools/build_macos.sh --clean-only             # 只清, 不编
#   tools/build_macos.sh --asan                   # 用 macos-accelerate-asan preset
#   tools/build_macos.sh --no-test                # 跳过 ctest
#   tools/build_macos.sh --model-dir /path/Qwen3-ASR-0.6B
#
# 环境变量 (覆盖默认值):
#   QASR_BUILD_DIR      编译输出目录 (默认 build/macos-accelerate)
#   QASR_MODEL_DIR      模型目录 (优先于自动探测)
#   QASR_PYTHON         python3 路径 (默认自动探测)
#   QASR_JOBS           并发数 (默认 sysctl -n hw.ncpu)
#   QASR_HF_CACHE       HF 缓存根目录 (默认 $HOME/Library/Caches/huggingface)
#   QASR_HF_REPO        模型仓库 (默认 Qwen/Qwen3-ASR-0.6B)
#   QASR_ONNXRUNTIME_ROOT   ONNX runtime 路径 (默认自动探测)
#   QASR_ONNXRUNTIME_VERSION  ONNX runtime 版本 (默认 1.20.1)

set -euo pipefail

# ─────────────── 颜色 ───────────────
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    C_RED="$(tput setaf 1)"
    C_GRN="$(tput setaf 2)"
    C_YEL="$(tput setaf 3)"
    C_BLU="$(tput setaf 4)"
    C_DIM="$(tput dim)"
    C_BLD="$(tput bold)"
    C_RST="$(tput sgr0)"
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_DIM=""; C_BLD=""; C_RST=""
fi

log_info()  { printf "${C_BLU}[INFO]${C_RST}  %s\n" "$*"; }
log_ok()    { printf "${C_GRN}[OK]${C_RST}    %s\n" "$*"; }
log_warn()  { printf "${C_YEL}[WARN]${C_RST}  %s\n" "$*" >&2; }
log_err()   { printf "${C_RED}[ERROR]${C_RST} %s\n" "$*" >&2; }
log_step()  { printf "\n${C_BLD}${C_BLU}── %s ──${C_RST}\n" "$*"; }
log_cmd()   { printf "${C_DIM}\$ %s${C_RST}\n" "$*"; }

# ─────────────── 路径 ───────────────
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -z "$SCRIPT_PATH" ]]; then SCRIPT_PATH="$0"; fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─────────────── 默认值 ───────────────
BUILD_DIR="${QASR_BUILD_DIR:-build/macos-accelerate}"
PRESET="macos-accelerate"
HF_REPO="${QASR_HF_REPO:-Qwen/Qwen3-ASR-0.6B}"
HF_CACHE="${QASR_HF_CACHE:-$HOME/Library/Caches/huggingface}"
PYTHON_BIN="${QASR_PYTHON:-}"

JOBS="${QASR_JOBS:-}"
MODEL_DIR_OVERRIDE=""

DO_CLEAN=1            # 默认 clean
DO_CLEAN_ONLY=0
DO_TEST=1
DO_MODEL=1
DO_AUDIO=1
DO_BENCH=0
ASSUME_YES=0
EXTRA_CMAKE_DEFS=()
EXTRA_BUILD_TARGET=""

# 状态(被 check_* 填充,供 build 阶段使用)
DETECTED_MODEL_DIR=""
DETECTED_AUDIO=""
DETECTED_ONNXRUNTIME_ROOT=""  # 传给 cmake 的 -DQASR_ONNXRUNTIME_ROOT
ONNXRUNTIME_VERSION="${QASR_ONNXRUNTIME_VERSION:-1.20.1}"
DO_onnxruntime=0          # macOS preset 默认不开 VAD

# ─────────────── 帮助 ───────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [flags]

Flags:
  --clean              删除 build 目录后从头构建 (默认)
  --incremental        增量编译, 不删 build/
  --clean-only         只清理, 不配置/编译/测试
  --asan               用 macos-accelerate-asan preset (AddressSanitizer)
  -j, --jobs N         并发数 (默认 sysctl -n hw.ncpu)
  -t, --target NAME    只编指定 cmake target
  -D VAR=VALUE         额外 cmake 缓存变量 (可重复)
  --no-test            跳过 ctest
  --no-model           跳过模型探测
  --no-audio           跳过测试音频探测
  --bench              编译后跑 qasr_cpu_bench --threads 8 --warmup 5 --scale 3
  -y, --yes            非交互 (brew 装包自动 -y)

 路径/版本:
  --model-dir DIR      指定模型目录 (覆盖 \$QASR_MODEL_DIR)
  --build-dir DIR      编译输出目录 (默认 $BUILD_DIR)

帮助:
  -h, --help           显示本帮助
  -v, --version        显示脚本版本

Env vars:
  QASR_BUILD_DIR      编译输出目录 (默认 build/macos-accelerate)
  QASR_MODEL_DIR      模型目录 (优先于自动探测)
  QASR_PYTHON         python3 路径 (默认自动探测)
  QASR_JOBS           并发数 (默认 sysctl -n hw.ncpu)
  QASR_HF_CACHE       HF 缓存根目录 (默认 \$HOME/Library/Caches/huggingface)
  QASR_HF_REPO        模型仓库 (默认 Qwen/Qwen3-ASR-0.6B)
  QASR_ONNXRUNTIME_ROOT   ONNX runtime 路径 (默认自动探测)
  QASR_ONNXRUNTIME_VERSION  ONNX runtime 版本 (默认 1.20.1)

Examples:
  $(basename "$0")                              # clean + build + test (Accelerate)
  $(basename "$0") --incremental                # 增量
  $(basename "$0") --asan                       # ASan 构建
  $(basename "$0") --no-test -j 4               # 不跑测试, 4 并发
  $(basename "$0") --no-model                   # 跳过所有可选检查
  $(basename "$0") --model-dir /data/Qwen3-ASR-0.6B

 完整参数 + 其他工具: docs/CLI.md
EOF
}

# ─────────────── 解析参数 ───────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)         DO_CLEAN=1; shift ;;
        --incremental|--no-clean) DO_CLEAN=0; shift ;;
        --clean-only)    DO_CLEAN=1; DO_CLEAN_ONLY=1; shift ;;
        --asan)          PRESET="macos-accelerate-asan"; shift ;;
        --no-test)       DO_TEST=0; shift ;;
        --no-model)      DO_MODEL=0; shift ;;
        --no-audio)      DO_AUDIO=0; shift ;;
        --bench)         DO_BENCH=1; shift ;;
        -y|--yes)        ASSUME_YES=1; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j[0-9]*)        JOBS="${1#-j}"; shift ;;
        -t|--target)     EXTRA_BUILD_TARGET="$2"; shift 2 ;;
        -D)              EXTRA_CMAKE_DEFS+=("-D$2"); shift 2 ;;
        -D*)             EXTRA_CMAKE_DEFS+=("$1"); shift ;;
        --model-dir)     MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --build-dir)     BUILD_DIR="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        -v|--version)    printf "build_macos.sh 1.0.0 (2026-06-01)\n"; exit 0 ;;
        *)               log_err "未知选项: $1"; usage >&2; exit 2 ;;
    esac
done

# 相对路径转绝对 (相对项目根)
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR"; fi

# 默认 jobs
if [[ -z "$JOBS" ]]; then
    JOBS="$(sysctl -n hw.ncpu)"
fi

# python3 自动探测
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        PYTHON_BIN="python3"
    fi
fi

# brew 前缀
BREW_CMD=""
if command -v brew >/dev/null 2>&1; then
    BREW_CMD="brew"
fi

# ─────────────── 1. OS 检查 ───────────────
check_os() {
    log_step "系统检查"
    local uname_str
    uname_str="$(uname -s)"
    if [[ "$uname_str" != "Darwin" ]]; then
        log_err "当前系统: $uname_str (仅支持 macOS)"
        exit 1
    fi
    local mac_ver
    mac_ver="$(sw_vers -productVersion 2>/dev/null || echo "unknown")"
    local xcode_ver
    xcode_ver="$(xcodebuild -version 2>/dev/null | head -1 | awk '{print $2}' || echo "unknown")"
    log_ok "OS: macOS $mac_ver"
    log_ok "Xcode: $xcode_ver"
}

# ─────────────── 2. 工具链检查 ───────────────
check_toolchain() {
    log_step "编译工具链检查"
    local missing=()

    for t in cmake ninja git curl tar make; do
        if ! command -v "$t" >/dev/null 2>&1; then
            missing+=("$t")
        fi
    done

    # clang++ (macOS 自带)
    if command -v clang++ >/dev/null 2>&1; then
        local clang_ver
        clang_ver=$(clang++ --version 2>/dev/null | head -1 || echo "unknown")
        log_ok "clang++: $clang_ver"
    else
        missing+=("clang++")
    fi

    # cmake 版本 (>= 3.21, preset v3)
    if command -v cmake >/dev/null 2>&1; then
        local cm_ver
        cm_ver=$(cmake --version | head -1 | awk '{print $3}')
        local cm_major="${cm_ver%%.*}"
        if [[ "$cm_major" =~ ^[0-9]+$ ]] && [[ "$cm_major" -lt 3 ]]; then
            log_warn "cmake $cm_ver 较旧, 建议 >= 3.21"
        else
            log_ok "cmake $cm_ver"
        fi
    fi

    # ninja
    if command -v ninja >/dev/null 2>&1; then
        log_ok "ninja $(ninja --version)"
    fi

    # ffmpeg (可选, 用于音频处理)
    if command -v ffmpeg >/dev/null 2>&1; then
        log_ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
    else
        log_warn "ffmpeg 未找到 (音频处理可能需要, brew install ffmpeg)"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "缺少工具: ${missing[*]}"
        log_info "macOS 上可通过 Homebrew 安装:"
        log_info "  brew install cmake ninja git curl"
        log_info ""
        log_info "如未安装 Homebrew:"
        log_info "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    log_ok "工具链完整"
}

# ─────────────── 3. Accelerate 框架检查 ───────────────
check_accelerate() {
    log_step "Accelerate 框架检查"
    # macOS 自带 Accelerate, 通过 c++ -framework Accelerate 即可链接
    # 简单验证框架是否存在
    if system_profiler SPFrameworksDataType 2>/dev/null | grep -q "Accelerate"; then
        log_ok "Accelerate framework: 已安装 (macOS 自带)"
    else
        # system_profiler 可能很慢, fallback 检查头文件
        if xcrun --show-sdk-path 2>/dev/null | xargs -I{} test -d "{}/System/Library/Frameworks/Accelerate.framework"; then
            log_ok "Accelerate framework: 可用 (macOS 自带)"
        else
            log_warn "无法确认 Accelerate 框架 (请确保 Xcode Command Line Tools 已安装)"
            log_warn "安装命令: xcode-select --install"
        fi
    fi
}

# ─────────────── 3.5 ONNX Runtime (Silero VAD 依赖, 可选) ───────────────
probe_onnxruntime() {
    if [[ -n "${QASR_ONNXRUNTIME_ROOT:-}" ]] \
        && [[ -f "$QASR_ONNXRUNTIME_ROOT/include/onnxruntime_c_api.h" ]] \
        && [[ -f "$QASR_ONNXRUNTIME_ROOT/lib/libonnxruntime.dylib" \
            || -f "$QASR_ONNXRUNTIME_ROOT/lib/libonnxruntime.1.dylib" ]]; then
        log_ok "ONNX Runtime (env QASR_ONNXRUNTIME_ROOT): $QASR_ONNXRUNTIME_ROOT"
        DETECTED_ONNXRUNTIME_ROOT="$QASR_ONNXRUNTIME_ROOT"
        return 0
    fi
    # 尝试 brew 安装的 onnxruntime
    if command -v brew >/dev/null 2>&1; then
        local brew_or
        brew_or="$(brew --prefix onnxruntime 2>/dev/null || true)"
        if [[ -n "$brew_or" ]] && [[ -f "$brew_or/include/onnxruntime_c_api.h" ]]; then
            log_ok "ONNX Runtime (brew): $brew_or"
            DETECTED_ONNXRUNTIME_ROOT="$brew_or"
            return 0
        fi
    fi
    return 1
}

check_onnxruntime() {
    if [[ $DO_onnxruntime -ne 1 ]]; then
        return 0
    fi
    log_step "ONNX Runtime 依赖检查 (Silero VAD 需要)"
    if probe_onnxruntime; then
        return 0
    fi
    log_warn "未检测到 ONNX Runtime (VAD 可选依赖, 缺则 VAD 退化为 40s 强制 cap)"
    log_info "安装方式:"
    log_info "  brew install onnxruntime"
    log_info "  或: 从 https://github.com/microsoft/onnxruntime/releases 下载 macOS 版本"
    log_info "  设 QASR_ONNXRUNTIME_ROOT 指向安装目录"
}

# ─────────────── 4. 模型检查 ───────────────
probe_model() {
    local repo_basename
    repo_basename="$(basename "$HF_REPO")"   # Qwen3-ASR-0.6B

    if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
        if [[ -f "$MODEL_DIR_OVERRIDE/model.safetensors" ]]; then
            DETECTED_MODEL_DIR="$MODEL_DIR_OVERRIDE"
            log_ok "Model (--model-dir): $DETECTED_MODEL_DIR"
            return 0
        fi
        log_warn "--model-dir=$MODEL_DIR_OVERRIDE 不含 model.safetensors"
    fi
    if [[ -n "${QASR_MODEL_DIR:-}" ]]; then
        if [[ -f "$QASR_MODEL_DIR/model.safetensors" ]]; then
            DETECTED_MODEL_DIR="$QASR_MODEL_DIR"
            log_ok "Model (QASR_MODEL_DIR): $DETECTED_MODEL_DIR"
            return 0
        fi
        log_warn "QASR_MODEL_DIR=$QASR_MODEL_DIR 不含 model.safetensors"
    fi
    # HF 缓存
    local hf_path
    hf_path="$(find -L "$HF_CACHE" -path "*snapshots*" -name "model.safetensors" -type f 2>/dev/null | head -1 || true)"
    # 也检查 ~/.cache/huggingface (Linux 风格路径, 有些用户可能用)
    if [[ -z "$hf_path" ]]; then
        hf_path="$(find -L "$HOME/.cache/huggingface" -path "*snapshots*" -name "model.safetensors" -type f 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$hf_path" ]]; then
        DETECTED_MODEL_DIR="$(dirname "$hf_path")"
        log_ok "Model (HF cache): $DETECTED_MODEL_DIR"
        return 0
    fi
    # 项目本地
    for d in "$PROJECT_ROOT/models/$repo_basename" "$PROJECT_ROOT/$repo_basename"; do
        if [[ -f "$d/model.safetensors" ]]; then
            DETECTED_MODEL_DIR="$d"
            log_ok "Model (local): $DETECTED_MODEL_DIR"
            return 0
        fi
    done
    return 1
}

check_model() {
    log_step "模型检查"
    if probe_model; then
        return 0
    fi
    local repo_basename
    repo_basename="$(basename "$HF_REPO")"
    log_warn "未检测到模型 ($HF_REPO)"
    log_info "下载方式 (任选一种):"
    log_info "  1) pip: pip install -U huggingface_hub"
    log_info "     $PYTHON_BIN -c from huggingface_hub import snapshot_download;"
    log_info "         snapshot_download('$HF_REPO', cache_dir='$HF_CACHE')"
    log_info "  2) git: git lfs install && git clone https://huggingface.co/$HF_REPO $PROJECT_ROOT/$repo_basename"
    log_info "  3) 手动: 从 https://huggingface.co/$HF_REPO 下载 model.safetensors,"
    log_info "             放到 $PROJECT_ROOT/$repo_basename/"
    log_warn "build 仍将继续, 但 qasr_cli E2E 测试会失败 (--no-test 跳过 ctest 不影响 build)"
}

# ─────────────── 5. 测试音频检查 ───────────────
probe_audio() {
    local f
    f="$(find "$PROJECT_ROOT/testfile" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) 2>/dev/null | head -1 || true)"
    if [[ -n "$f" ]]; then
        DETECTED_AUDIO="$f"
        return 0
    fi
    return 1
}

check_audio() {
    log_step "测试音频检查"
    if probe_audio; then
        log_ok "Audio: $DETECTED_AUDIO"
        return 0
    fi
    log_warn "testfile/ 下无 .wav/.mp3/.flac (--no-audio 时忽略)"
    log_info "拉样音:"
    log_info "  $PYTHON_BIN tools/aishell_fetch.py --speaker S0002 --clips 18"
    log_info "  (需要 pip install huggingface_hub, 网络可达 huggingface.co)"
}

# ─────────────── 6. 清理 ───────────────
do_clean() {
    if [[ $DO_CLEAN -eq 0 ]]; then
        log_info "跳过 clean (--incremental)"
        return
    fi
    log_step "清理 $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    log_ok "已删除"
}

# ─────────────── 7. CMake configure ───────────────
do_configure() {
    log_step "CMake configure (preset=$PRESET)"
    cd "$PROJECT_ROOT"
    local args=(-S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja)
    if [[ -n "$DETECTED_ONNXRUNTIME_ROOT" ]]; then
        args+=("-DQASR_ONNXRUNTIME_ROOT=$DETECTED_ONNXRUNTIME_ROOT")
        args+=("-DQASR_ENABLE_SILERO_VAD=ON")
    fi
    if [[ ${#EXTRA_CMAKE_DEFS[@]} -gt 0 ]]; then
        args+=("${EXTRA_CMAKE_DEFS[@]}")
    fi
    log_cmd "cmake ${args[*]}"
    if ! cmake "${args[@]}" 2>&1 | tee /tmp/qasr_cmake.log; then
        log_err "cmake configure 失败 (日志 /tmp/qasr_cmake.log)"
        log_err "常见原因:"
        log_err "  - Xcode Command Line Tools 未安装 (xcode-select --install)"
        log_err "  - cmake 版本过低 (>= 3.21 推荐, brew upgrade cmake)"
        exit 1
    fi
    log_ok "configure 成功"
}

# ─────────────── 8. 编译 ───────────────
do_build() {
    log_step "编译 (jobs=$JOBS)"
    cd "$PROJECT_ROOT"
    local args=(--build "$BUILD_DIR" -j "$JOBS")
    if [[ -n "$EXTRA_BUILD_TARGET" ]]; then
        args+=(--target "$EXTRA_BUILD_TARGET")
    fi
    log_cmd "cmake ${args[*]}"
    if ! cmake "${args[@]}" 2>&1 | tee /tmp/qasr_build.log; then
        log_err "build 失败 (日志 /tmp/qasr_build.log)"
        exit 1
    fi
    log_ok "build 成功"
}

# ─────────────── 9. ctest ───────────────
do_test() {
    if [[ $DO_TEST -eq 0 ]]; then
        log_info "跳过 ctest (--no-test)"
        return
    fi
    log_step "单元测试 (qasr_unit_tests)"
    cd "$PROJECT_ROOT"
    log_cmd "ctest --test-dir $BUILD_DIR -R qasr_unit_tests --output-on-failure"
    if ! ctest --test-dir "$BUILD_DIR" -R qasr_unit_tests --output-on-failure 2>&1 \
            | tee /tmp/qasr_test.log; then
        log_err "ctest 失败 (日志 /tmp/qasr_test.log)"
        exit 1
    fi
    log_ok "ctest PASS"
}

# ─────────────── 10. (可选) benchmark ───────────────
do_bench() {
    if [[ $DO_BENCH -eq 0 ]]; then
        return
    fi
    local bench="$BUILD_DIR/qasr_cpu_bench"
    if [[ ! -x "$bench" ]]; then
        log_warn "找不到 $bench, 跳过 bench"
        return
    fi
    log_step "Benchmark (8 线程, 5 warmup, scale 3)"
    log_cmd "$bench --threads 8 --warmup 5 --scale 3"
    "$bench" --threads 8 --warmup 5 --scale 3 | tail -30
}

# ─────────────── 11. 总结 ───────────────
print_summary() {
    log_step "构建总结"
    log_ok "Build 目录: $BUILD_DIR"
    log_ok "Preset:     $PRESET"
    log_ok "BLAS:       Apple Accelerate (macOS 自带)"
    [[ -n "$DETECTED_MODEL_DIR" ]] && log_ok "Model:      $DETECTED_MODEL_DIR"
    [[ -n "$DETECTED_AUDIO" ]]     && log_ok "Audio:      $DETECTED_AUDIO"
    echo
    if [[ -x "$BUILD_DIR/qasr_cli" ]]; then
        log_info "E2E 转写:"
        local md="${DETECTED_MODEL_DIR:-(设置 \$QASR_MODEL_DIR 或 --model-dir)}"
        local ad="${DETECTED_AUDIO:-(testfile/aishell_S0002_limai_108s.wav)}"
        log_info "  $BUILD_DIR/qasr_cli --model-dir \"$md\" --audio \"$ad\" --language Chinese"
    fi
    if [[ -x "$BUILD_DIR/qasr_server" ]]; then
        log_info "启动 server:  $BUILD_DIR/qasr_server --model-dir \"$md\""
        log_info "或一键启动:   tools/run_macos_server.sh --detach --https"
    fi
    echo
    log_ok "✅ 构建完成"
}

# ─────────────── 主流程 ───────────────
main() {
    cd "$PROJECT_ROOT"
    log_info "qwen3asr_cpu one-key build (macOS / Accelerate)"
    log_info "项目: $PROJECT_ROOT"
    log_info "编译: $BUILD_DIR   preset: $PRESET   jobs: $JOBS"
    echo

    check_os
    check_toolchain
    check_accelerate
    check_onnxruntime
    [[ $DO_MODEL -eq 1 ]] && check_model
    [[ $DO_AUDIO -eq 1 ]] && check_audio

    if [[ $DO_CLEAN_ONLY -eq 1 ]]; then
        do_clean
        log_ok "clean-only 完成"
        exit 0
    fi

    do_clean
    do_configure
    do_build
    do_test
    do_bench
    print_summary
}

main "$@"
