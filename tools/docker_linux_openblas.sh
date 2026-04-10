#!/usr/bin/env bash
set -euo pipefail

platform="${1:-linux/arm64}"
image="${2:-ubuntu:24.04}"
build_dir="build/docker-${platform//\//-}"
apt_mirror="${QASR_APT_MIRROR:-http://mirrors.aliyun.com/ubuntu}"

docker run --rm \
  --platform "${platform}" \
  -v "${PWD}:/workspace" \
  -w /workspace \
  "${image}" \
  bash -lc "
    if [ -n '${apt_mirror}' ]; then
      find /etc/apt -type f \\( -name '*.list' -o -name '*.sources' \\) -print0 |
        xargs -0 sed -i -E 's#http://(archive|security).ubuntu.com/ubuntu#${apt_mirror}#g';
    fi &&
    apt-get -o Acquire::Retries=5 update &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake ninja-build pkg-config libopenblas-dev &&
    cmake -S . -B ${build_dir} -G Ninja -DQASR_ENABLE_TESTS=ON &&
    cmake --build ${build_dir} &&
    ctest --test-dir ${build_dir} --output-on-failure
  "
