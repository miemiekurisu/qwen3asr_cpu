#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model_dir> [image_tag]" >&2
  exit 1
fi

MODEL_DIR="$1"
IMAGE_TAG="${2:-qasr-ui:latest}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model_dir not found: $MODEL_DIR" >&2
  exit 1
fi

ARGS=(
  run --rm -it
  -p 8080:8080
  -v "$MODEL_DIR:/models/qwen3-asr:ro"
)

if [[ -e /dev/snd ]]; then
  ARGS+=(--device /dev/snd)
fi

if [[ -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/pulse/native" ]]; then
  ARGS+=(
    -e "PULSE_SERVER=unix:/tmp/pulse/native"
    -v "${XDG_RUNTIME_DIR}/pulse/native:/tmp/pulse/native"
  )
fi

ARGS+=("$IMAGE_TAG")

exec docker "${ARGS[@]}"
