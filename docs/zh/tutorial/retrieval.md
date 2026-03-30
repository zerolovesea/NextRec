---
title: 训练召回模型
description: 完整的召回模型训练、评估、推理教程，包括双塔模型、向量化匹配
---

# 训练召回模型

本教程介绍推荐系统的召回阶段，演示如何使用 DSSM（Deep Structured Semantic Model）构建召回模型。

## 什么是召回模型？

推荐系统通常分为两个阶段：

1. **召回（Match）**：从海量候选池中快速筛选出候选集（几千到几万）
2. **排序（Ranking）**：对召回的候选集精细排序

```
┌─────────────────────────────────────────────────┐
│              推荐系统流程                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  用户 ──▶ 召回 ──▶ 排序 ──▶ 业务规则 ──▶ 推荐结果 │
│              │        │                         │
│         (海量→千级)  (千级→百级)                │
│                                                 │
└─────────────────────────────────────────────────┘
```

## DSSM 模型原理

DSSM（Deep Structured Semantic Model）也称为双塔模型：

- **用户塔**：将用户特征编码为向量
- **物品塔**：将物品特征编码为向量
- **相似度计算**：通过向量相似度召回用户和物品

```
┌────────────────────────────────────────────────┐
│              DSSM 双塔结构                      │
├────────────────────────────────────────────────┤
│                                                │
│   用户侧                      物品侧            │
│  ┌─────────┐               ┌─────────┐       │
│  │ User    │               │ Item    │       │
│  │ Features│               │ Features│       │
│  └────┬────┘               └────┬────┘       │
│       │                         │             │
│       ▼                         ▼             │
│  ┌─────────┐               ┌─────────┐       │
│  │ User    │               │ Item    │       │
│  │ Tower   │               │ Tower   │       │
│  │  DNN    │               │  DNN    │       │
│  └────┬────┘               └────┬────┘       │
│       │                         │             │
│       └────────┬────────────────┘             │
│                ▼                              │
│         ┌─────────┐                          │
│         │ Cosine  │  ──▶ 相似度             │
│         └─────────┘                          │
│                                                │
└────────────────────────────────────────────────┘
```

## 环境准备

```python
import pandas as pd
import numpy as np
import torch

from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.models.retrieval.dssm import DSSM
from nextrec.data.preprocessor import DataProcessor
```

## 1. 数据准备

### 1.1 匹配数据格式

召回模型需要用户-物品交互数据：

```python
# 用户特征
user_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "age": [25, 30, 35],
    "gender": ["M", "F", "M"],
    "hist_item_ids": ["101,102,103", "201,202", "301,302,303,304"]
})

# 物品特征
item_df = pd.DataFrame({
    "item_id": [101, 102, 103, 201, 202, 301, 302],
    "category": [1, 2, 1, 3, 3, 4, 4],
    "price": [99.0, 199.0, 59.0, 299.0, 159.0, 89.0, 129.0]
})

# 交互数据（训练标签）
train_df = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [101, 201, 102, 301, 103, 302],
    "label": [1, 0, 1, 0, 1, 0]  # 1: 正样本, 0: 负样本
})
```

### 1.2 数据预处理

```python
# 处理用户数据
user_processor = DataProcessor()
user_processor.add_numeric_feature("age", scaler="standard")
user_processor.add_sparse_feature("gender", encode_method="label")
user_processor.add_sequence_feature(
    "hist_item_ids",
    encode_method="hash",
    hash_size=10000,
    max_len=50,
    separator=","
)
user_processor.fit(user_df)

# 处理物品数据
item_processor = DataProcessor()
item_processor.add_numeric_feature("price", scaler="standard")
item_processor.add_sparse_feature("category", encode_method="label")
item_processor.fit(item_df)
```

## 2. 特征定义

### 2.1 用户侧特征

```python
# 用户特征
user_dense_features = [DenseFeature("age")]
user_sparse_features = [
    SparseFeature("gender", vocab_size=2, embedding_dim=4)
]
user_sequence_features = [
    SequenceFeature(
        name="hist_item_ids",
        vocab_size=10000,
        embedding_dim=32,
        max_len=50,
        combiner="mean"
    )
]
```

### 2.2 物品侧特征

```python
# 物品特征
item_dense_features = [DenseFeature("price")]
item_sparse_features = [
    SparseFeature("category", vocab_size=10, embedding_dim=8)
]
```

## 3. 模型创建

### 3.1 DSSM 模型

```python
# 创建 DSSM 召回模型
model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=user_sequence_features,
    item_dense_features=item_dense_features,
    item_sparse_features=item_sparse_features,
    user_tower_layers=[128, 64, 32],    # 用户塔 DNN 层
    item_tower_layers=[128, 64, 32],    # 物品塔 DNN 层
    temperature=0.1,                    # 温度参数，控制分布锐度
    task="binary",
    device="cuda"
)
```

### 3.2 配置训练

```python
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="bpr"  # 使用 BPR (Bayesian Personalized Ranking) 损失
)
```

## 4. 模型训练

### 4.1 准备训练数据

召回模型的训练需要特殊的数据格式：

```python
# 构建训练数据：用户特征 + 正负样本物品特征
# 格式: {user_features...}, {item_features...}, label
train_data = build_match_data(user_df, item_df, train_df)
```

### 4.2 训练

```python
model.fit(
    train_data=train_data,
    epochs=10,
    batch_size=256,
    metrics=["recall@10", "precision@10", "ndcg@10"]
)
```

## 5. 模型评估

### 5.1 召回评估指标

```python
# 评估召回效果
metrics = model.evaluate(
    valid_data,
    metrics=["recall@10", "precision@10", "ndcg@10", "mrr@10"]
)

print("召回评估结果:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
```

### 5.2 指标说明

| 指标 | 说明 |
|-----|------|
| Recall@K | 召回率，前 K 个推荐中包含正样本的比例 |
| Precision@K | 精确率 |
| NDCG@K | 归一化折损累计增益 |
| MRR@K | 平均倒数排名 |

## 6. 向量检索

### 6.1 提取用户和物品向量

```python
# 提取用户向量
user_embeddings = model.get_user_embedding(user_df)

# 提取物品向量
item_embeddings = model.get_item_embedding(item_df)

print(f"用户向量形状: {user_embeddings.shape}")
print(f"物品向量形状: {item_embeddings.shape}")
```

### 6.2 近似最近邻检索

```python
import faiss

# 构建 FAISS 索引
dimension = item_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # 内积相似度
index.add(item_embeddings)

# 检索 Top-K
top_k = 10
distances, indices = index.search(user_embeddings, top_k)

print("检索结果:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  用户0 - 物品{idx}: {dist:.4f}")
```

## 7. 完整代码

```python
"""
DSSM 召回模型完整示例
"""
import pandas as pd
import numpy as np

from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.models.retrieval.dssm import DSSM
from nextrec.data.preprocessor import DataProcessor

# ========== 1. 数据准备 ==========
# 用户数据
user_df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "age": [25, 30, 35, 28, 40],
    "gender": ["M", "F", "M", "F", "M"],
    "hist_item_ids": ["101,102,103", "201,202", "301,302,303", "401,402", "501,502,503,504"]
})

# 物品数据
item_df = pd.DataFrame({
    "item_id": [101, 102, 103, 201, 202, 301, 302, 401, 402, 501],
    "category": [1, 2, 1, 3, 3, 4, 4, 5, 5, 6],
    "price": [99.0, 199.0, 59.0, 299.0, 159.0, 89.0, 129.0, 199.0, 249.0, 399.0]
})

# ========== 2. 数据预处理 ==========
# 用户处理器
user_processor = DataProcessor()
user_processor.add_numeric_feature("age", scaler="standard")
user_processor.add_sparse_feature("gender", encode_method="label")
user_processor.add_sequence_feature("hist_item_ids", encode_method="hash", hash_size=10000, max_len=50, separator=",")
user_processor.fit(user_df)

# 物品处理器
item_processor = DataProcessor()
item_processor.add_numeric_feature("price", scaler="standard")
item_processor.add_sparse_feature("category", encode_method="label")
item_processor.fit(item_df)

# 转换数据
user_processed = user_processor.transform(user_df)
item_processed = item_processor.transform(item_df)

# ========== 3. 特征定义 ==========
# 用户特征
user_dense_features = [DenseFeature("age")]
user_sparse_features = [SparseFeature("gender", vocab_size=2, embedding_dim=4)]
user_sequence_features = [
    SequenceFeature("hist_item_ids", vocab_size=10000, embedding_dim=32, max_len=50, combiner="mean")
]

# 物品特征
item_dense_features = [DenseFeature("price")]
item_sparse_features = [SparseFeature("category", vocab_size=10, embedding_dim=8)]

# ========== 4. 创建模型 ==========
model = DSSM(
    user_dense_features=user_dense_features,
    user_sparse_features=user_sparse_features,
    user_sequence_features=user_sequence_features,
    item_dense_features=item_dense_features,
    item_sparse_features=item_sparse_features,
    user_tower_layers=[128, 64, 32],
    item_tower_layers=[128, 64, 32],
    temperature=0.1,
    task="binary",
    device="cuda"
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="bpr"
)

# ========== 5. 训练 ==========
# 构建训练数据（用户-物品交互）
train_data = {
    "user_age": user_processed["age"],
    "user_gender": user_processed["gender"],
    "user_hist_item_ids": user_processed["hist_item_ids"],
    "item_price": item_processed["price"],
    "item_category": item_processed["category"],
    "label": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 示例标签
}

model.fit(
    train_data=train_data,
    epochs=10,
    batch_size=32,
    metrics=["recall@10", "precision@10"]
)

# ========== 6. 向量检索 ==========
# 获取用户和物品向量
user_emb = model.get_user_embedding(user_processed)
item_emb = model.get_item_embedding(item_processed)

print(f"用户向量: {user_emb.shape}")
print(f"物品向量: {item_emb.shape}")

# ========== 7. 评估 ==========
metrics = model.evaluate(valid_data, metrics=["recall@10", "precision@10"])
print(f"Recall@10: {metrics['recall@10']:.4f}")
```

## 8. 其他召回模型

### YoutubeDNN

适合视频推荐的经典双塔模型：

```python
from nextrec.models.retrieval.youtube_dnn import YoutubeDNN

model = YoutubeDNN(
    user_features=...,
    item_features=...,
    item_tower_layers=[128, 64],
    ...
)
```

### MIND

多兴趣召回模型，适合用户有多个兴趣点：

```python
from nextrec.models.retrieval.mind import MIND

model = MIND(
    user_features=...,
    item_features=...,
    num_interests=4,  # 多个兴趣胶囊
    ...
)
```

## 下一步

- [训练精排模型](./ranking.md)
- [训练多任务模型](./multitask.md)
