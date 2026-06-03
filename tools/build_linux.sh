#!/usr/bin/env bash
#
# build_linux.sh — One-key Linux build script for qwen3asr_cpu (Debian-family)
#
# 工作流程:
#   1. 校验系统 (Debian/Ubuntu/...)
#   2. 检查编译工具链 (g++/cmake/ninja/pkg-config/ffmpeg/git/curl)
#   3. 检查并按需构建 OpenBLAS (系统包 → 源码下载 → 手动提示)
#       也可 --blas=blis|mkl|auto|ref 选其他后端 (MKL 走探测 oneAPI/conda)
#   4. 探测模型 (HF 缓存 / $QASR_MODEL_DIR / ./models/...) → 缺失则提示下载
#   5. 探测测试音频 (testfile/*.wav) → 缺失则提示拉取
#   6. clean → cmake --preset linux-openblas → build → ctest
#   7. (可选) --compare-blas 跑 OpenBLAS/BLIS/MKL 三家实测对比
#
# 用法:
#   tools/build_linux.sh [选项]
#
# 常用:
#   tools/build_linux.sh                         # 默认 clean + configure + build + test
#   tools/build_linux.sh --incremental           # 增量编译 (不删 build/)
#   tools/build_linux.sh --clean-only            # 只清, 不编
#   tools/build_linux.sh --asan                  # 用 linux-openblas-asan preset
#   tools/build_linux.sh --no-test               # 跳过 ctest
#   tools/build_linux.sh --model-dir /path/Qwen3-ASR-0.6B
#
# 环境变量 (覆盖默认值):
#   QASR_DEPS_DIR       OpenBLAS 源码安装位置 (默认 /opt/qasr-deps)
#   QASR_BUILD_DIR      编译输出目录 (默认 build/linux-openblas)
#   QASR_MODEL_DIR      模型目录 (优先于自动探测)
#   QASR_OPENBLAS_TAG   OpenBLAS 版本 (默认 v0.3.30, 用于源码下载)
#   QASR_PYTHON         python3 路径 (默认自动探测)
#   QASR_JOBS           并发数 (默认 nproc)
#   QASR_APT_MIRROR     apt 源 (留空用系统默认)
#   QASR_HF_CACHE       HF 缓存根目录 (默认 $HOME/.cache/huggingface)
#   QASR_HF_REPO        模型仓库 (默认 Qwen/Qwen3-ASR-0.6B)

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
# tolerate being sourced (BASH_SOURCE may be empty) — fall back to script arg
if [[ -z "$SCRIPT_PATH" ]]; then SCRIPT_PATH="$0"; fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─────────────── 默认值 ───────────────
DEPS_DIR="${QASR_DEPS_DIR:-/opt/qasr-deps}"
BUILD_DIR="${QASR_BUILD_DIR:-build/linux-openblas}"
PRESET="linux-openblas"
OPENBLAS_TAG="${QASR_OPENBLAS_TAG:-v0.3.30}"
HF_REPO="${QASR_HF_REPO:-Qwen/Qwen3-ASR-0.6B}"
HF_CACHE="${QASR_HF_CACHE:-$HOME/.cache/huggingface}"
PYTHON_BIN="${QASR_PYTHON:-}"

JOBS="${QASR_JOBS:-}"
MODEL_DIR_OVERRIDE=""
APT_MIRROR="${QASR_APT_MIRROR:-}"

DO_CLEAN=1            # 默认 clean
DO_CLEAN_ONLY=0
DO_TEST=1
DO_DEP=1
DO_MODEL=1
DO_AUDIO=1
DO_BENCH=0
DO_COMPARE_BLAS=0
QASR_BLAS_CHOICE=""   # 空 = 用 preset 默认
ASSUME_YES=0
SKIP_SYSTEM_PKG=0     # 不调 apt-get
EXTRA_CMAKE_DEFS=()
EXTRA_BUILD_TARGET=""

# 状态(被 check_* 填充,供 build 阶段使用)
DETECTED_OPENBLAS_DIR=""   # 传给 cmake 的 -DOpenBLAS_DIR
DETECTED_MODEL_DIR=""
DETECTED_AUDIO=""

# ─────────────── 帮助 ───────────────
usage() {
    cat <<EOF
用法: $(basename "$0") [选项]

编译选项:
  --clean              删除 build 目录后从头构建 (默认)
  --incremental        增量编译, 不删 build/
  --clean-only         只清理, 不配置/编译/测试
  --asan               用 linux-openblas-asan preset (AddressSanitizer)
  -j, --jobs N         并发数 (默认 nproc)
  -t, --target NAME    只编指定 cmake target
  -D VAR=VALUE         额外 cmake 缓存变量 (可重复)
  --no-test            跳过 ctest
  --no-dep             跳过 OpenBLAS 依赖检查/构建
  --no-model           跳过模型探测
  --no-audio           跳过测试音频探测
  --bench              编译后跑 qasr_cpu_bench --threads 8 --warmup 5 --scale 3
  --compare-blas       编译后跑 tools/compare_blas.sh (需装好 OpenBLAS+BLIS+MKL)
  -y, --yes            非交互 (apt-get 装包自动 -y)
  --no-apt             不调用 apt-get (即使缺包)

BLAS 后端 (覆盖 linux-openblas preset, 通过 -DQASR_BLAS 传给 cmake):
  --blas NAME          openblas|blis|mkl|auto|ref
                       openblas = 默认 (BSD-3, AVX-VNNI 内核)
                       blis     = 备选 (BSD-3, Zen/AVX-512 强;AVX2-only Intel 反而慢)
                       mkl      = oneAPI,专有,个人免费,Intel 上最快
                       详见 docs/BLAS_COMPARISON.md

路径/版本:
  --model-dir DIR      指定模型目录 (覆盖 \$QASR_MODEL_DIR)
  --deps-dir DIR       OpenBLAS 安装位置 (默认 $DEPS_DIR)
  --build-dir DIR      编译输出目录 (默认 $BUILD_DIR)
  --openblas-tag TAG   源码下载版本 (默认 $OPENBLAS_TAG)

帮助:
  -h, --help           显示本帮助
  -v, --version        显示脚本版本

示例:
  $(basename "$0")                              # clean + build + test (OpenBLAS)
  $(basename "$0") --incremental                # 增量
  $(basename "$0") --asan                       # ASan 构建
  $(basename "$0") --no-test -j 4               # 不跑测试, 4 并发
  $(basename "$0") --no-dep --no-model          # 跳过所有可选检查
  $(basename "$0") --model-dir /data/Qwen3-ASR-0.6B
  $(basename "$0") --blas blis --incremental    # 切到 BLIS (BSD-3 备选)
  $(basename "$0") --blas mkl --incremental     # 切到 oneAPI MKL (专有)
  $(basename "$0") --compare-blas               # 跑 OpenBLAS/BLIS/MKL 三家对比
EOF
}

# ─────────────── 解析参数 ───────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)         DO_CLEAN=1; shift ;;
        --incremental|--no-clean) DO_CLEAN=0; shift ;;
        --clean-only)    DO_CLEAN=1; DO_CLEAN_ONLY=1; shift ;;
        --asan)          PRESET="linux-openblas-asan"; shift ;;
        --no-test)       DO_TEST=0; shift ;;
        --no-dep)        DO_DEP=0; shift ;;
        --no-model)      DO_MODEL=0; shift ;;
        --no-audio)      DO_AUDIO=0; shift ;;
        --bench)         DO_BENCH=1; shift ;;
        --compare-blas)  DO_COMPARE_BLAS=1; shift ;;
        --blas)          QASR_BLAS_CHOICE="$2"; shift 2 ;;
        -y|--yes)        ASSUME_YES=1; shift ;;
        --no-apt)        SKIP_SYSTEM_PKG=1; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j[0-9]*)        JOBS="${1#-j}"; shift ;;
        -t|--target)     EXTRA_BUILD_TARGET="$2"; shift 2 ;;
        -D)              EXTRA_CMAKE_DEFS+=("-D$2"); shift 2 ;;
        -D*)             EXTRA_CMAKE_DEFS+=("$1"); shift ;;
        --model-dir)     MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --deps-dir)      DEPS_DIR="$2"; shift 2 ;;
        --build-dir)     BUILD_DIR="$2"; shift 2 ;;
        --openblas-tag)  OPENBLAS_TAG="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        -v|--version)    printf "build_linux.sh 1.0.0 (2026-06-01)\n"; exit 0 ;;
        *)               log_err "未知选项: $1"; usage >&2; exit 2 ;;
    esac
done

# 相对路径转绝对 (相对项目根)
if [[ "$BUILD_DIR" != /* ]]; then BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR"; fi
if [[ "$DEPS_DIR"  != /* ]]; then DEPS_DIR="$PROJECT_ROOT/$DEPS_DIR"; fi

# 默认 jobs
if [[ -z "$JOBS" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    else
        JOBS=2
    fi
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

# sudo 前缀
SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo "
fi

# ─────────────── 1. OS 检查 ───────────────
check_os() {
    log_step "系统检查"
    if [[ ! -r /etc/os-release ]]; then
        log_err "未找到 /etc/os-release, 仅支持 Debian 系 Linux"
        exit 1
    fi
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}" in
        debian|ubuntu|kali|linuxmint|pop|elementary|raspbian|zorin|mx|mxlinux|deepin|parrot|mxlinux)
            log_ok "OS: ${PRETTY_NAME:-$ID} (ID=$ID)"
            ;;
        *)
            log_err "OS '$ID' 不是 Debian/Ubuntu 系。本脚本只支持 Debian-family."
            log_err "其它发行版请参考:tools/build_windows_openblas.ps1 / cmake/QasrBlas.cmake"
            exit 1
            ;;
    esac
}

# ─────────────── 2. 工具链检查 ───────────────
check_toolchain() {
    log_step "编译工具链检查"
    local missing=()
    local have_gxx=0
    for t in g++ cmake ninja pkg-config git ffmpeg curl tar make; do
        if ! command -v "$t" >/dev/null 2>&1; then
            missing+=("$t")
        fi
    done

    # g++ 版本 (C++20)
    if command -v g++ >/dev/null 2>&1; then
        have_gxx=1
        local gxx_ver
        gxx_ver=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion 2>/dev/null || echo "0")
        # 取主版本
        local gxx_major="${gxx_ver%%.*}"
        if [[ "$gxx_major" =~ ^[0-9]+$ ]] && [[ "$gxx_major" -lt 10 ]]; then
            log_warn "g++ $gxx_ver 较旧, 建议 >= 10 (C++20 完整支持)"
        else
            log_ok "g++ $gxx_ver"
        fi
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

    # ffmpeg
    if command -v ffmpeg >/dev/null 2>&1; then
        log_ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "缺少工具: ${missing[*]}"
        log_info "请在 Debian/Ubuntu 上执行:"
        log_info "  ${SUDO}apt-get update"
        log_info "  ${SUDO}apt-get install -y build-essential cmake ninja-build pkg-config git curl ffmpeg libfftw3-dev"
        exit 1
    fi
    log_ok "工具链完整"
}

# ─────────────── 3. OpenBLAS ───────────────
# 探测顺序:
#   1. ${DEPS_DIR}/lib/cmake/openblas/openblasConfig.cmake
#   2. ${DEPS_DIR}/include/openblas.h (header only 提示)
#   3. 系统 pkg-config openblas
#   4. /usr/include/openblas.h 等
#   5. apt-get install libopenblas-dev (若允许)
#   6. 源码下载并构建到 ${DEPS_DIR}
#   7. 失败 → 提示手动安装
probe_openblas() {
    if [[ -f "$DEPS_DIR/lib/cmake/openblas/openBLASConfig.cmake" ]] \
        || [[ -f "$DEPS_DIR/lib/cmake/openblas/OpenBLASConfig.cmake" ]] \
        || [[ -f "$DEPS_DIR/lib/cmake/openblas/openblasConfig.cmake" ]]; then
        log_ok "OpenBLAS (cmake config): $DEPS_DIR"
        DETECTED_OPENBLAS_DIR="$DEPS_DIR/lib/cmake/openblas"
        return 0
    fi
    if [[ -f "$DEPS_DIR/include/openblas.h" ]] || [[ -f "$DEPS_DIR/include/x86_64-linux-gnu/openblas.h" ]]; then
        log_ok "OpenBLAS (headers): $DEPS_DIR/include"
        DETECTED_OPENBLAS_DIR="$DEPS_DIR/lib/cmake/openblas"  # 让 cmake 找不到,fallback 到 pkg-config 不可用时报错
        return 0
    fi
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas 2>/dev/null; then
        local ver; ver=$(pkg-config --modversion openblas 2>/dev/null || echo "?")
        log_ok "OpenBLAS (system pkg-config): $ver"
        DETECTED_OPENBLAS_DIR=""
        return 0
    fi
    if [[ -f /usr/include/openblas.h ]] \
        || [[ -f /usr/include/x86_64-linux-gnu/openblas.h ]] \
        || [[ -f /usr/local/include/openblas.h ]]; then
        log_ok "OpenBLAS (system headers): $(ls /usr/include/openblas.h /usr/include/x86_64-linux-gnu/openblas.h 2>/dev/null | head -1)"
        DETECTED_OPENBLAS_DIR=""
        return 0
    fi
    return 1
}

apt_install_openblas() {
    if [[ $SKIP_SYSTEM_PKG -eq 1 ]]; then
        log_warn "跳过 apt-get (--no-apt)"
        return 1
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        log_warn "未找到 apt-get, 跳过"
        return 1
    fi
    log_info "尝试 apt-get install libopenblas-dev ..."
    local apt_args=("-y" "install" "libopenblas-dev")
    if [[ $ASSUME_YES -eq 0 ]]; then
        apt_args=("-y" "install" "libopenblas-dev")  # 脚本默认 -y
    fi
    if [[ -n "$APT_MIRROR" ]]; then
        # 仅在不破坏系统源的前提下,通过 sources.list.d 临时加镜像;
        # 这里保守一点,只提示用户,不直接改源
        log_info "(使用 QASR_APT_MIRROR=$APT_MIRROR 仅作提示, 未自动改源)"
    fi
    if $SUDO apt-get "${apt_args[@]}" >/tmp/qasr_apt.log 2>&1; then
        log_ok "libopenblas-dev 安装成功"
        return 0
    fi
    log_warn "apt-get install libopenblas-dev 失败 (/tmp/qasr_apt.log)"
    return 1
}

build_openblas_from_source() {
    log_info "从源码下载并构建 OpenBLAS ${OPENBLAS_TAG} 到 ${DEPS_DIR}"
    log_warn "源码构建可能需要 5-10 分钟, 临时需要 $((JOBS+1)) 个进程, ~500MB 磁盘"
    local work; work="$(mktemp -d -t qasr-openblas-XXXXXX)"
    local url="https://github.com/OpenBLAS/OpenBLAS/archive/refs/tags/${OPENBLAS_TAG}.tar.gz"
    log_info "下载: $url"
    if ! curl -fsSL --retry 3 --connect-timeout 15 -o "$work/openblas.tar.gz" "$url"; then
        log_err "下载 OpenBLAS 失败"
        log_err "手动下载: $url"
        log_err "解压后:"
        log_err "  cd OpenBLAS-${OPENBLAS_TAG#v}"
        log_err "  make -j$JOBS PREFIX=$DEPS_DIR"
        log_err "  $SUDO make install PREFIX=$DEPS_DIR"
        rm -rf "$work"
        exit 1
    fi
    log_info "解压 ..."
    if ! tar -xzf "$work/openblas.tar.gz" -C "$work"; then
        log_err "解压失败"; rm -rf "$work"; exit 1
    fi
    local src_dir="$work/OpenBLAS-${OPENBLAS_TAG#v}"
    if [[ ! -d "$src_dir" ]]; then
        # 部分 tag 可能目录名是 OpenBLAS-0.3.30 (无 v)
        src_dir="$(find "$work" -maxdepth 1 -type d -name 'OpenBLAS-*' | head -1)"
    fi
    if [[ -z "$src_dir" || ! -d "$src_dir" ]]; then
        log_err "找不到 OpenBLAS 源码目录"; rm -rf "$work"; exit 1
    fi
    log_info "编译 ... (日志: /tmp/qasr_openblas_build.log)"
    if ! (cd "$src_dir" && make -j"$JOBS" PREFIX="$DEPS_DIR" \
            >/tmp/qasr_openblas_build.log 2>&1); then
        log_err "OpenBLAS 编译失败, 末 40 行:"
        tail -40 /tmp/qasr_openblas_build.log >&2 || true
        log_err "手动编译:"
        log_err "  cd $src_dir"
        log_err "  make -j$JOBS PREFIX=$DEPS_DIR"
        log_err "  $SUDO make install PREFIX=$DEPS_DIR"
        rm -rf "$work"
        exit 1
    fi
    log_info "安装到 $DEPS_DIR ..."
    if ! (cd "$src_dir" && $SUDO make install PREFIX="$DEPS_DIR" \
            >/tmp/qasr_openblas_install.log 2>&1); then
        log_err "OpenBLAS 安装失败 (/tmp/qasr_openblas_install.log)"
        tail -30 /tmp/qasr_openblas_install.log >&2 || true
        rm -rf "$work"
        exit 1
    fi
    rm -rf "$work"
    log_ok "OpenBLAS 已安装: $DEPS_DIR"
    if [[ -f "$DEPS_DIR/lib/cmake/openblas/openBLASConfig.cmake" ]] \
        || [[ -f "$DEPS_DIR/lib/cmake/openblas/OpenBLASConfig.cmake" ]]; then
        DETECTED_OPENBLAS_DIR="$DEPS_DIR/lib/cmake/openblas"
    fi
}

check_openblas() {
    log_step "OpenBLAS 依赖检查"
    if probe_openblas; then
        return 0
    fi
    log_warn "未检测到 OpenBLAS"
    if [[ $DO_DEP -eq 0 ]]; then
        log_err "OpenBLAS 缺失, 且 --no-dep 禁止自动安装"
        log_err "手动安装方式 (任选):"
        log_err "  ${SUDO}apt-get install -y libopenblas-dev"
        log_err "  或下载源码编译到 $DEPS_DIR (--openblas-tag 控制版本)"
        exit 1
    fi
    if apt_install_openblas; then
        if probe_openblas; then return 0; fi
    fi
    # 兜底:源码构建
    build_openblas_from_source
    if ! probe_openblas; then
        log_err "OpenBLAS 安装后仍探测不到, 请检查 $DEPS_DIR"
        exit 1
    fi
}

# ─────────────── 4. 模型检查 ───────────────
# 探测顺序:
#   $QASR_MODEL_DIR > --model-dir > $HF_CACHE/.../snapshots/*/model.safetensors
#   > $PROJECT_ROOT/models/<repo_basename>/ > $PROJECT_ROOT/<repo_basename>/
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
    # HF 缓存: <cache>/models--<owner>--<name>/snapshots/<rev>/model.safetensors
    # 注: snapshot 下是 symlink,需 -L 跟随;同时排除 .locks/ 等中间目录
    local hf_path
    hf_path="$(find -L "$HF_CACHE" -path "*snapshots*" -name "model.safetensors" -type f 2>/dev/null | head -1 || true)"
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
    log_info "     $PYTHON_BIN -c \"from huggingface_hub import snapshot_download; \\"
    log_info "         snapshot_download('$HF_REPO', cache_dir='$HF_CACHE')\""
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
    # 注:cmake --preset 模式对 -D 在 --preset 后追加的 cache 变量支持不稳,
    # 显式用 -S/-B/-G 调用更可靠。
    local args=(-S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja)
    if [[ -n "$DETECTED_OPENBLAS_DIR" ]]; then
        args+=("-DOpenBLAS_DIR=$DETECTED_OPENBLAS_DIR")
    fi
    if [[ -n "$QASR_BLAS_CHOICE" ]]; then
        args+=("-DQASR_BLAS=$QASR_BLAS_CHOICE")
    fi
    if [[ ${#EXTRA_CMAKE_DEFS[@]} -gt 0 ]]; then
        args+=("${EXTRA_CMAKE_DEFS[@]}")
    fi
    log_cmd "cmake ${args[*]}"
    if ! cmake "${args[@]}" 2>&1 | tee /tmp/qasr_cmake.log; then
        log_err "cmake configure 失败 (日志 /tmp/qasr_cmake.log)"
        log_err "常见原因:"
        log_err "  - OpenBLAS 未正确安装 (尝试 $SUDO apt-get install -y libopenblas-dev)"
        log_err "  - cmake 版本过低 (>= 3.21 推荐)"
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

# ─────────────── 10b. (可选) BLAS 三家对比 ───────────────
do_compare_blas() {
    if [[ $DO_COMPARE_BLAS -eq 0 ]]; then
        return
    fi
    if [[ -z "$DETECTED_MODEL_DIR" ]]; then
        log_warn "未找到模型目录,跳过 --compare-blas (指定 \$QASR_MODEL_DIR 或 --model-dir)"
        return
    fi
    log_step "BLAS 对比 (OpenBLAS / BLIS / MKL)  ──  详见 docs/BLAS_COMPARISON.md"
    if [[ -x "$PROJECT_ROOT/tools/compare_blas.sh" ]]; then
        "$PROJECT_ROOT/tools/compare_blas.sh" "$DETECTED_MODEL_DIR"
    else
        log_warn "找不到 tools/compare_blas.sh, 跳过"
    fi
}

# ─────────────── 11. 总结 ───────────────
print_summary() {
    log_step "构建总结"
    log_ok "Build 目录: $BUILD_DIR"
    log_ok "Preset:     $PRESET"
    [[ -n "$DETECTED_OPENBLAS_DIR" ]] && log_ok "OpenBLAS:   $DETECTED_OPENBLAS_DIR"
    [[ -n "$DETECTED_MODEL_DIR" ]]    && log_ok "Model:      $DETECTED_MODEL_DIR"
    [[ -n "$DETECTED_AUDIO" ]]        && log_ok "Audio:      $DETECTED_AUDIO"
    echo
    if [[ -x "$BUILD_DIR/qasr_cli" ]]; then
        log_info "E2E 转写:"
        local md="${DETECTED_MODEL_DIR:-(设置 \$QASR_MODEL_DIR 或 --model-dir)}"
        local ad="${DETECTED_AUDIO:-(testfile/aishell_S0002_limai_108s.wav)}"
        log_info "  $BUILD_DIR/qasr_cli --model-dir \"$md\" --audio \"$ad\" --language Chinese"
    fi
    if [[ -x "$BUILD_DIR/qasr_server" ]]; then
        log_info "启动 server:  $BUILD_DIR/qasr_server --model-dir \"$md\""
    fi
    echo
    log_ok "✅ 构建完成"
}

# ─────────────── 主流程 ───────────────
main() {
    cd "$PROJECT_ROOT"
    log_info "qwen3asr_cpu one-key build (Linux/Debian-family)"
    log_info "项目: $PROJECT_ROOT"
    log_info "编译: $BUILD_DIR   preset: $PRESET   jobs: $JOBS"
    echo

    check_os
    check_toolchain
    # 探测 OpenBLAS 总是要做(为了传 -DOpenBLAS_DIR),只是缺失时是否自动装取决于 DO_DEP
    check_openblas
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
    do_compare_blas
    print_summary
}

main "$@"
