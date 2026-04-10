# Vendor 说明

本目录只放“已复制入本仓”之第三方单文件依赖。

约束：

- 构建、包含、链接，只可指向本目录副本。
- 不得再直接依赖参考工程原目录。
- CPU 推理后端已内化到 `src/backend/qwen_cpu/`。

当前内容：

- `vendor/third_party/httplib.h`
- `vendor/third_party/json.hpp`
