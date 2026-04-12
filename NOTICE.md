# Notices

This repository contains first-party code and a copied third-party component.

## First-party project

- Name: qwen-asr-provider
- License: MIT
- Copyright: 2026 miemiekurisu
- License file: `LICENSE`

## CPU ASR backend snapshot

- Path: `src/backend/qwen_cpu/`
- License: MIT
- Upstream copyright: Copyright (c) 2026 Salvatore Sanfilippo
- Upstream license file: `src/backend/qwen_cpu/LICENSE.upstream`
- Status: copied and integrated into this repository as the current CPU backend.

## Self-implemented components

The HTTP server (`src/base/http_server.cc`) and JSON parser (`src/base/json.cc`)
are first-party implementations. No third-party single-header libraries
(cpp-httplib, nlohmann/json, etc.) are used.

## Third-party build-time dependencies

- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — BSD-3-Clause — linked dynamically at runtime
- [oneDNN](https://github.com/oneapi-src/oneDNN) — Apache-2.0 — optional, downloaded and built from source by CMake

## Model files

Model weights are not stored in this repository. Users must comply with the
license and terms of the model they mount or download, including Qwen/Qwen3-ASR
model terms.
