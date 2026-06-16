#!/usr/bin/env bash
# scripts/compare_blas.sh — build qasr_blas_bench for each BLAS backend, run it
# against the project's long audio, and print a side-by-side comparison.
#
# Usage: scripts/compare_blas.sh <model-snapshot-dir> [audio.wav] [threads]
#   model-snapshot-dir  e.g. ~/.cache/huggingface/.../snapshots/<sha>
#   audio.wav           defaults to testfile/aishell_S0002_limai_108s.wav
#   threads             defaults to 8
#
# Skips a backend if its library is not installed; the script is *not* an
# installer (BLIS needs `apt-get install libblis-openmp-dev` or a source
# build, MKL needs oneAPI or the conda `mkl-devel` package).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${1:-}"
AUDIO="${2:-$REPO_ROOT/testfile/aishell_S0002_limai_108s.wav}"
THREADS="${3:-8}"

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    echo "Usage: $0 <model-snapshot-dir> [audio.wav] [threads]" >&2
    echo "  <model-snapshot-dir> must point to a Qwen3-ASR-0.6B snapshot" >&2
    exit 1
fi
if [[ ! -f "$AUDIO" ]]; then
    echo "audio not found: $AUDIO" >&2
    exit 1
fi

run_one() {
    local backend="$1"
    local build_dir="build/bench-$backend"
    local extra_cmake=()
    if [[ "$backend" != "openblas" ]]; then
        extra_cmake=("-DQASR_BLAS=$backend")
    fi
    {
        echo
        echo "==================== $backend ===================="
    } >&2
    cmake -S . -B "$build_dir" -G Ninja "${extra_cmake[@]}" \
        > "$build_dir.log" 2>&1 \
        || { echo "  [SKIP] $backend: cmake configure failed (see $build_dir.log)" >&2; return 1; }
    cmake --build "$build_dir" --target qasr_blas_bench -j "$(nproc)" \
        >> "$build_dir.log" 2>&1 \
        || { echo "  [SKIP] $backend: build failed (see $build_dir.log)" >&2; return 1; }
    "$build_dir/qasr_blas_bench" \
        --model-dir "$MODEL_DIR" \
        --audio "$AUDIO" \
        --threads "$THREADS" \
        --rounds 3 \
        2>/dev/null
}

declare -A RESULTS
for backend in openblas blis mkl; do
    if json="$(run_one "$backend")"; then
        RESULTS[$backend]="$json"
    fi
done

echo
echo "==================== summary ===================="
if [[ -z "${RESULTS[openblas]:-}" ]]; then
    echo "no successful run; cannot summarise" >&2
    exit 1
fi
# Pretty-print a markdown table sorted by best RTF
printf "| %-12s | %-10s | %-10s | %-10s |\n" "backend" "wall_best" "rtf_best" "speedup"
printf "| %-12s | %-10s | %-10s | %-10s |\n" "------------" "----------" "----------" "----------"

baseline_rtf="$(echo "${RESULTS[openblas]}" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["rtf_best"])')"
for backend in openblas blis mkl; do
    json="${RESULTS[$backend]:-}"
    if [[ -z "$json" ]]; then
        printf "| %-12s | %-10s | %-10s | %-10s |\n" "$backend" "(skipped)" "-" "-"
        continue
    fi
    python3 - "$json" "$baseline_rtf" "$backend" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
baseline = float(sys.argv[2])
backend = sys.argv[3]
speedup = row["rtf_best"] / baseline
print(f"| {backend:<12} | {row['wall_ms_best']:>8.0f}ms | {row['rtf_best']:>8.3f}x | {speedup:>8.2f}x |")
PY
done

echo
echo "JSON outputs:"
for backend in "${!RESULTS[@]}"; do
    echo "  $backend: ${RESULTS[$backend]}"
done
