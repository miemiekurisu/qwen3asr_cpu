# License Compliance

日期：2026-04-10

## 目标

本项目须能分发源码与二进制，同时保留上游版权、许可与模型义务边界。

## 当前许可

| 范围 | 路径 | 许可 | 义务 |
|---|---|---|---|
| 项目自有代码 | `app/`, `include/`, `src/`, `tests/`, `ui/`, `docs/` | MIT | 分发时带 `LICENSE` |
| CPU ASR 后端 | `src/backend/qwen_cpu/` | MIT | 分发时带 `src/backend/qwen_cpu/LICENSE.upstream` |
| cpp-httplib | `vendor/third_party/httplib.h` | MIT | 保留头部版权与许可 |
| nlohmann/json | `vendor/third_party/json.hpp` | MIT | 保留 SPDX 与头部版权 |
| 模型权重 | 外部挂载或用户缓存 | 模型自身条款 | 不入库，用户自守 |

## 禁止

- 不提交模型权重。
- 不提交参考项目目录。
- 不移除上游版权头。
- 不把第三方代码伪装成自有代码。

## 发布清单

发布源码或二进制时必须带：

- `LICENSE`
- `NOTICE.md`
- `src/backend/qwen_cpu/LICENSE.upstream`
- `vendor/third_party/httplib.h` 内原版权头
- `vendor/third_party/json.hpp` 内 SPDX 版权头

## 后续引入依赖规则

新增第三方代码前须记录：

- 来源
- 作者
- 版本或提交
- 许可
- 本仓路径
- 是否改动
- 分发义务

并同步更新：

- `NOTICE.md`
- 本文件
- `experience.md`
