---
title: 训练多任务模型
description: 完整的多任务学习模型训练教程，包含 ESMM、MMOE、PLE 等模型
---

# 训练多任务模型

本教程将带你完成一个完整的多任务学习模型开发流程：

1. 什么是多任务学习
2. 多任务学习模型介绍
3. 数据准备与预处理
4. 模型训练
5. 模型评估与推理

## 什么是多任务学习？

多任务学习（Multi-Task Learning, MTL）是一种机器学习范式，通过同时学习多个相关任务来提升模型的泛化能力。在推荐系统中，常见的应用场景包括：

- **CTR + CVR 预估**：同时预测点击率和转化率
- **点击 + 收藏 + 购买**：多目标优化
- **短期 + 长期兴趣建模**：同时预测即时兴趣和长期偏好

### 多任务学习的优势

| 优势 | 说明 |
|-----|------|
| 共享表示 | 底层特征共享，减少过拟合 |
| 迁移学习 | 主任务帮助辅任务学习 |
| 样本效率 | 充分利用所有样本数据 |
| 隐式正则化 | 避免单任务过拟合 |

## 多任务学习模型

NextRec 支持多种多任务学习模型：

| 模型 | 说明 |
|-----|------|
| ESMM | 全空间多任务模型，解决样本选择偏差 |
| MMOE | 多门控混合专家网络 |
| PLE | 多层级专家网络 |
| AITM | 注意力级联多目标模型 |
| HMoE | 混合专家多任务模型 |
| ShareBottom | 基础共享底层模型 |
| CrossStitch | 跨任务特征融合 |
| POSO | 门控多任务模型 |

### ESMM 模型架构

```
┌─────────────────────────────────────────┐
│              ESMM 结构                   │
├─────────────────────────────────────────┤
│                                         │
│   Sparse Features                       │
│         │                               │
│         ▼                               │
│   ┌─────────────────┐                   │
│   │  Shared Embedding │                 │
│   └────────┬────────┘                   │
│            │                            │
│      ┌─────┴─────┐                      │
│      ▼           ▼                      │
│   ┌─────┐   ┌──────────┐                │
│   │ CTR │   │   CVR    │                │
│   │ 塔  │   │   塔     │                 │
│   └──┬──┘   └────┬─────┘                │
│      │            │                     │
│      ▼            │                     │
│   pCTR      pCVR × pCTR                 │
│                 │                       │
│                 ▼                       │
│            pCTCVR                       │
│                                         │
└─────────────────────────────────────────┘
```

## 环境准备

```python
import pandas as pd
import numpy as np
import torch

# 导入 NextRec 组件
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.models.multi_task.esmm import ESMM
from nextrec.data.preprocessor import DataProcessor

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

## 1. 数据准备与预处理

### 1.1 数据格式

多任务学习需要多个目标标签，示例数据格式如下：

| user_id | item_id | dense_0 | sparse_0 | sequence_0 | click | conversion |
|---------|---------|---------|----------|------------|-------|------------|
| 1       | 100     | 0.5     | 3        | [1,2,3]    | 1     | 0          |

- `click`: 点击标签 (二分类)
- `conversion`: 转化标签 (二分类)

### 1.2 数据预处理

```python
# 加载数据
df = pd.read_csv("dataset/multitask_task.csv")

# 将序列特征从字符串转换为列表
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# 定义任务标签
task_labels = ["click", "conversion"]

print(f"数据形状: {df.shape}")
print(f"任务标签: {task_labels}")
```

### 1.3 数据集划分

```python
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(valid_df)}")
```

## 2. 特征定义

### 2.1 Dense 特征

```python
# 稠密特征：连续数值
num_dense = len([col for col in df.columns if col.startswith("dense_")])
dense_features = [DenseFeature(f"dense_{i}") for i in range(num_dense)]
```

### 2.2 Sparse 特征

```python
# 稀疏特征：离散 ID
sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max() + 1), embedding_dim=32),
    SparseFeature("item_id", vocab_size=int(df["item_id"].max() + 1), embedding_dim=32),
]

# 添加其他稀疏特征
num_sparse = len([col for col in df.columns if col.startswith("sparse_")])
sparse_features.extend([
    SparseFeature(f"sparse_{i}", vocab_size=int(df[f"sparse_{i}"].max() + 1), embedding_dim=16)
    for i in range(num_sparse)
])
```

### 2.3 Sequence 特征

```python
# 序列特征：用户行为序列
sequence_cols = [col for col in df.columns if col.startswith("sequence_")]
sequence_features = [
    SequenceFeature(
        col,
        vocab_size=int(df[col].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1),
        embedding_dim=32,
        padding_idx=0,
    )
    for col in sequence_cols
]

print(f"Dense: {len(dense_features)}, Sparse: {len(sparse_features)}, Sequence: {len(sequence_features)}")
```

## 3. 模型创建与配置

### 3.1 创建 ESMM 模型

```python
# CTR 塔参数
ctr_params = {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4}

# CVR 塔参数
cvr_params = {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4}

# 创建 ESMM 模型
model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_mlp_params=ctr_params,
    cvr_mlp_params=cvr_params,
    target=task_labels,        # 多任务标签
    task=["binary", "binary"], # 两个二分类任务
    device=device,
    session_id="esmm_tutorial"
)
```

### 3.2 配置损失函数

NextRec 支持多种损失函数组合：

#### 选项 1：固定损失权重

```python
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    loss_weights=[0.3, 0.7],  # 手动设置权重
)
```

#### 选项 2：Focal Loss + GradNorm

```python
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["focal", "focal"],
    loss_params=[
        {"gamma": 2.0, "alpha": 0.25},  # CTR 任务
        {"gamma": 1.0, "alpha": 0.75},  # CVR 任务
    ],
    loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025},
)
```

### 3.3 损失权重说明

| 参数 | 说明 |
|-----|------|
| `loss` | 每个任务的损失函数 |
| `loss_weights` | 损失权重，支持固定权重或 GradNorm 动态调整 |
| `loss_params` | 损失函数的额外参数 |

## 4. 模型训练

```python
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "gauc", "logloss"],
    epochs=10,
    batch_size=512,
    shuffle=True,
    user_id_column="user_id",  # 用于计算 GAUC
)

print("训练完成!")
```

### 训练参数说明

| 参数 | 说明 |
|-----|------|
| `train_data` | 训练数据 |
| `valid_data` | 验证数据 |
| `metrics` | 评估指标列表 |
| `user_id_column` | 用户 ID 列名，用于 GAUC 计算 |

## 5. 模型评估

### 5.1 评估指标

```python
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss"],
    batch_size=512,
    user_id_column="user_id",
)

print("=== 验证集评估结果 ===")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

### 5.2 常见评估指标

| 指标 | 说明 | 理想值 |
|-----|------|-------|
| AUC | ROC 曲线下面积 | 越接近 1 越好 |
| GAUC | 分组 AUC | 越接近 1 越好 |
| LogLoss | 对数损失 | 越接近 0 越好 |

## 6. 模型推理

```python
# 批量预测
predictions = model.predict(
    valid_df,
    batch_size=512,
    return_dataframe=True
)

print(f"预测结果形状: {predictions.shape}")
print(predictions.head())
```

## 7. 完整代码

```python
"""
ESMM 多任务学习模型完整示例
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.models.multi_task.esmm import ESMM

# ========== 1. 数据准备 ==========
df = pd.read_csv("dataset/multitask_task.csv")

# 转换序列特征
for col in df.columns:
    if "sequence" in col:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

task_labels = ["click", "conversion"]

# ========== 2. 数据集划分 ==========
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

# ========== 3. 特征定义 ==========
num_dense = len([col for col in df.columns if col.startswith("dense_")])
dense_features = [DenseFeature(f"dense_{i}") for i in range(num_dense)]

sparse_features = [
    SparseFeature("user_id", vocab_size=int(df["user_id"].max() + 1), embedding_dim=32),
    SparseFeature("item_id", vocab_size=int(df["item_id"].max() + 1), embedding_dim=32),
]

num_sparse = len([col for col in df.columns if col.startswith("sparse_")])
sparse_features.extend([
    SparseFeature(f"sparse_{i}", vocab_size=int(df[f"sparse_{i}"].max() + 1), embedding_dim=16)
    for i in range(num_sparse)
])

sequence_cols = [col for col in df.columns if col.startswith("sequence_")]
sequence_features = [
    SequenceFeature(
        col,
        vocab_size=int(df[col].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1),
        embedding_dim=32,
        padding_idx=0,
    )
    for col in sequence_cols
]

# ========== 4. 创建模型 ==========
model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_mlp_params={"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4},
    cvr_mlp_params={"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.4},
    target=task_labels,
    task=["binary", "binary"],
    device="cpu"
)

# ========== 5. 编译模型 ==========
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    loss_weights=[0.3, 0.7],
)

# ========== 6. 训练 ==========
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "gauc", "logloss"],
    epochs=10,
    batch_size=512,
    shuffle=True,
    user_id_column="user_id",
)

# ========== 7. 评估 ==========
metrics = model.evaluate(valid_df, metrics=["auc", "gauc", "logloss"])

# ========== 8. 推理 ==========
predictions = model.predict(valid_df, return_dataframe=True)
print(predictions.head())
```

## 下一步

- [训练召回模型](./retrieval.md)
- [分布式训练](../apis/distributed-training.md)
