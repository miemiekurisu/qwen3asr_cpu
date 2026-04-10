# Vendor 说明

本目录只放“已复制入本仓”之第三方或参考快照。

约束：

- 构建、包含、链接，只可指向本目录副本。
- 不得再直接依赖参考工程原目录。
- 若后续要替换为自研实现，先改本目录之消费点，再删副本。

当前内容：

- `vendor/qasr_cpu/`
  - CPU 推理后端快照
  - 后续以本仓接口逐步替换
- `vendor/third_party/httplib.h`
- `vendor/third_party/json.hpp`
