---
title: API 文档
description: NextRec 核心 API 导航
---

# API 文档

对于初上手使用NextRec的用户，我们准备了文档来帮助理解框架的各项API。对于一个完整的推荐算法模型工作流，通常会经历以下流程：
- 数据准备
- 数据预处理
- 训练模型/模型评估
- 模型加载和线上部署

在这份文档里，我们会按顺序介绍开发者需要了解的Python API接口。

## 核心

- [定义特征](features.md)
- [数据预处理](data-processor.md)
- [数据加载](dataloader.md)
- [基类模型的生命周期](base-model.md)

## 评估与工程化

- [损失函数](loss.md)
- [评估指标](metrics.md)
- [日志管理](session-logging.md)
