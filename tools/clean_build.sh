#!/usr/bin/env bash
set -euo pipefail

# ── clean + configure + build + test in one shot ──
#
# Usage:
#   tools/clean_build.sh [OPTIONS] [PRESET]
#
# Examples:
#   tools/clean_build.sh                            # auto-detect preset from OS
#   tools/clean_build.sh macos-accelerate            # use CMake preset
#   tools/clean_build.sh -G "Unix Makefiles" macos-accelerate  # override generator
#   tools/clean_build.sh -b build/custom             # custom build dir, no preset
#   tools/clean_build.sh --no-test macos-accelerate  # skip ctest
#   tools/clean_build.sh --list                      # show available presets

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

build_dir=""
generator=""
extra_defs=()
jobs=""
target=""
run_tests=true
preset=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [PRESET]

Clean + configure + build + test using a CMake preset or custom config.

Options:
  -b DIR        Build directory (overrides preset default)
  -G GEN        CMake generator (overrides preset default)
  -D VAR=VAL    Extra CMake cache variable (repeatable)
  -j N          Parallel build jobs (default: auto)
  -t TARGET     Build specific target instead of all
  --no-test     Skip running tests after build
  --list        List available configure presets and exit
  -h, --help    Show this help

If no PRESET is given, auto-detects from OS (macOS → macos-accelerate, Linux → linux-openblas).
If -b is given without PRESET, uses manual cmake invocation.
Generator is auto-detected: uses preset's generator if available, otherwise falls back.
EOF
}

list_presets() {
    local presets_file="${PROJECT_DIR}/CMakePresets.json"
    if [[ ! -f "${presets_file}" ]]; then
        echo "No CMakePresets.json found in ${PROJECT_DIR}" >&2
        exit 1
    fi
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for p in data.get('configurePresets', []):
    if p.get('hidden'):
        continue
    desc = p.get('description', '')
    name = p['name']
    print(f'  {name:30s} {desc}')
" "${presets_file}"
}

# Query a preset field (binaryDir or generator) with inheritance resolution
query_preset_field() {
    local preset_name="$1"
    local field="$2"
    python3 -c "
import json, sys, os
name, field = sys.argv[1], sys.argv[2]
src = sys.argv[3]
with open(os.path.join(src, 'CMakePresets.json')) as f:
    data = json.load(f)
presets = {p['name']: p for p in data.get('configurePresets', [])}

def find_field(p, field):
    val = p.get(field, '')
    if val:
        return val
    parent = p.get('inherits')
    if parent:
        parents = [parent] if isinstance(parent, str) else parent
        for pname in parents:
            if pname in presets:
                result = find_field(presets[pname], field)
                if result:
                    return result
    return ''

if name not in presets:
    print(f'error: preset \"{name}\" not found', file=sys.stderr)
    sys.exit(1)
val = find_field(presets[name], field)
if field == 'binaryDir':
    val = val.replace('\${sourceDir}', src).replace('\${presetName}', name)
print(val)
" "${preset_name}" "${field}" "${PROJECT_DIR}"
}

# Check whether a generator's build tool is available
generator_available() {
    local gen="$1"
    case "${gen}" in
        Ninja*)       command -v ninja &>/dev/null ;;
        *Makefiles*)  command -v make &>/dev/null || command -v gmake &>/dev/null ;;
        Xcode)        command -v xcodebuild &>/dev/null ;;
        *)            return 0 ;;  # assume available for unknown generators
    esac
}

# Pick the first available generator
detect_fallback_generator() {
    if command -v ninja &>/dev/null; then
        echo "Ninja"
    elif command -v make &>/dev/null || command -v gmake &>/dev/null; then
        echo "Unix Makefiles"
    else
        echo ""
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b)       build_dir="$2"; shift 2 ;;
        -G)       generator="$2"; shift 2 ;;
        -D)       extra_defs+=("-D$2"); shift 2 ;;
        -D*)      extra_defs+=("$1"); shift ;;
        -j)       jobs="$2"; shift 2 ;;
        -t)       target="$2"; shift 2 ;;
        --no-test) run_tests=false; shift ;;
        --list)   list_presets; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        -*)       echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)        preset="$1"; shift ;;
    esac
done

# Auto-detect preset from OS when none specified
if [[ -z "${preset}" && -z "${build_dir}" ]]; then
    case "$(uname -s)" in
        Darwin) preset="macos-accelerate" ;;
        Linux)  preset="linux-openblas" ;;
        *)      echo "error: cannot auto-detect preset for $(uname -s); specify a PRESET or -b BUILD_DIR" >&2
                exit 1 ;;
    esac
    echo "── auto-detected preset: ${preset}"
fi

# Resolve build directory from preset
if [[ -n "${preset}" && -z "${build_dir}" ]]; then
    build_dir="$(query_preset_field "${preset}" binaryDir)"
    if [[ -z "${build_dir}" ]]; then
        echo "error: could not resolve build dir for preset '${preset}'" >&2
        exit 1
    fi
fi

# Make build_dir absolute
if [[ "${build_dir}" != /* ]]; then
    build_dir="${PROJECT_DIR}/${build_dir}"
fi

echo "── clean: ${build_dir}"
rm -rf "${build_dir}"

# Auto-detect generator if not explicitly set
if [[ -z "${generator}" && -n "${preset}" ]]; then
    preset_gen="$(query_preset_field "${preset}" generator)"
    if [[ -n "${preset_gen}" ]] && ! generator_available "${preset_gen}"; then
        generator="$(detect_fallback_generator)"
        if [[ -z "${generator}" ]]; then
            echo "error: preset needs '${preset_gen}' but no build tool found" >&2
            exit 1
        fi
        echo "── generator '${preset_gen}' not found, using '${generator}'"
    fi
fi

# Configure
configure_args=()
if [[ -n "${preset}" ]]; then
    if [[ -n "${generator}" ]]; then
        # --preset and -G cannot be combined; fall back to manual configuration
        # with the resolved build directory and cache variables from the preset.
        configure_args+=(-S "${PROJECT_DIR}" -B "${build_dir}" -G "${generator}")
    else
        configure_args+=(--preset "${preset}")
    fi
else
    configure_args+=(-S "${PROJECT_DIR}" -B "${build_dir}")
    if [[ -n "${generator}" ]]; then
        configure_args+=(-G "${generator}")
    fi
fi
configure_args+=("${extra_defs[@]+"${extra_defs[@]}"}")

echo "── configure: cmake ${configure_args[*]}"
cmake "${configure_args[@]}"

# Build
build_args=(--build "${build_dir}")
if [[ -n "${jobs}" ]]; then
    build_args+=(-j "${jobs}")
else
    # Auto-detect parallelism
    if command -v nproc &>/dev/null; then
        build_args+=(-j "$(nproc)")
    elif command -v sysctl &>/dev/null; then
        build_args+=(-j "$(sysctl -n hw.ncpu)")
    fi
fi
if [[ -n "${target}" ]]; then
    build_args+=(--target "${target}")
fi

echo "── build: cmake ${build_args[*]}"
cmake "${build_args[@]}"

# Test
if ${run_tests}; then
    echo "── test: ctest --test-dir ${build_dir} --output-on-failure"
    ctest --test-dir "${build_dir}" --output-on-failure
fi

echo "── done"
