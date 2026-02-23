---
title: FAQ
description: NextRec 常见问题解答（GAUC、streaming、DataProcessor、ONNX 等）
---

# FAQ

## 1. `target`、`task`、`training_mode` 分别是什么？

- `target`：标签列名（单任务字符串，多任务列表）。
- `task`：任务类型（例如二分类 `binary`、回归 `regression`、召回 `matching`）。
- `training_mode`：训练范式（常见为 `pointwise`；部分模型/损失支持 pairwise/listwise）。

## 2. 为什么 GAUC / ranking@K 指标算不出来？

这类指标需要分组 ID（通常是 `user_id`）。

- Python API：在 `fit/evaluate` 里传 `user_id_column="user_id"`。
- CLI：在 `data` 段配置 `id_column: user_id`（或使用别名 `user_id_column`）。

## 3. 如何在大数据上训练（避免一次性读入内存）？

使用 `RecDataLoader` 的 streaming 模式，或 CLI 配置 `data.streaming: true` 并设置 `dataloader.chunk_size`。

参考：[API 文档 / RecDataLoader](apis/rec-dataloader.md)

## 4. `DataProcessor` 什么时候要用？

当你需要让训练/推理严格共享同一套预处理（数值缩放、hash/label 编码、序列 padding 等）时，建议用 `DataProcessor`：

- 训练前 `fit`
- 训练/推理 `transform`
- 保存/加载处理器用于线上一致性

参考：[API 文档 / DataProcessor](apis/data-processor.md)

## 5. 训练日志和产物保存在哪里？

默认在当前工作目录下的 `nextrec_logs/{session_id}/`。

- Python API：`session_id` 决定目录名；若不传会自动生成。
- CLI：在 train_config 的 `session.id` 与 `session.artifact_root` 控制。

参考：[API 文档 / Session 与日志](apis/session-logging.md)

## 6. 如何导出并用 ONNX 推理？

- 训练后调用 `export_onnx(...)` 导出
- 使用 `predict_onnx(...)` 对文件/数据进行推理

参考：[API 文档 / ONNX](apis/onnx.md)
