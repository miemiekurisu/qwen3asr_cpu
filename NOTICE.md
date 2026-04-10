# Notices

This repository contains first-party code and copied third-party components.

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

## cpp-httplib

- Path: `vendor/third_party/httplib.h`
- License: MIT
- Copyright: Copyright (c) 2025 Yuji Hirose. All rights reserved.
- Notice: license header is preserved in the copied single header.

## nlohmann/json

- Path: `vendor/third_party/json.hpp`
- License: MIT
- Main copyright: SPDX-FileCopyrightText: 2013-2022 Niels Lohmann
- Notice: SPDX and bundled attribution headers are preserved in the copied single header.

## Model files

Model weights are not stored in this repository. Users must comply with the
license and terms of the model they mount or download, including Qwen/Qwen3-ASR
model terms.

## Excluded reference directories

The local `qwen-asr-learn/` and `whisper.cpp/` directories are excluded from
this repository. They are references only and are not build, include, or link
dependencies of this project.
