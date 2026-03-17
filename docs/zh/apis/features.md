---
title: 特征定义
description: DenseFeature、SparseFeature、SequenceFeature 的完整参数说明
---

# 推荐系统的特征

特征是推荐系统里的重要概念，推荐模型通常需要同时处理多种信号，并将其转换为向量。NextRec 使用统一的特征对象描述模型输入，主要支持三种特征类型：

- **稠密特征**：连续或可序数化的数值（年龄、价格、时长等），常见做法是标准化/归一化或对数变换。
- **稀疏特征**：高基数离散字段（用户 ID、物品 ID、性别、职业、设备类型等），通常需要索引化后放入 embedding lookup 表里。
- **序列特征**：可变长历史行为（浏览/点击/购买列表），通常需要转化为ID后，截断、padding，再通过聚合（mean/sum/attention）这些ID的向量为一段定长的向量。

在初始化模型之前，我们需要对每个特征进行定义，让模型知道如何处理不同的特征。


## 稠密特征的定义

稠密特征适用于连续数值型数据，如年龄、收入、评分等，简单的示例如下：

```python
from nextrec.basic.features import DenseFeature

dense_features = [
    DenseFeature(
        name="age", 
        input_dim=1, 
        use_projection=False,
        trainable=True
        ),
    DenseFeature(
        name="price", 
        input_dim=8, 
        use_projection=True,
        proj_dim=1,
        projection_type="mlp",
        mlp_hidden_dims=[16, 8],
        mlp_activation="relu",
        mlp_dropout=0.0,
        trainable=True
        )    
    ]
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称，用作输入键 |
| `input_dim` | 特征的输入维度，对于单个数值为 1 |
| `use_projection` | 是否使用投影层将输入投影到更高维度 |
| `proj_dim` | 投影层维度，如果为 None 或 0，则不应用投影 |
| `projection_type` | 投影类型，支持 `"linear"`（默认）和 `"mlp"` |
| `mlp_hidden_dims` | 当 `projection_type="mlp"` 时使用的隐藏层维度列表 |
| `mlp_activation` | 当 `projection_type="mlp"` 时使用的激活函数 |
| `mlp_dropout` | 当 `projection_type="mlp"` 时使用的 Dropout |
| `trainable` | 投影层是否可训练 |
| `pretrained_weight` | 可选的预训练投影权重 |

### 多维 Dense 映射到 1 维示例

```python
DenseFeature(
    name="dense_vec",
    input_dim=12,            # 输入是 12 维连续特征
    use_projection=True,
    proj_dim=1,              # 输出映射为 1 维
    projection_type="mlp",   # 使用 MLP 投影
    mlp_hidden_dims=[32, 16],
)
```

---

## 稀疏特征

适用于离散 ID 或类别型数据，通过 Embedding 查表转换为稠密向量，简单的示例如下：

```python
from nextrec.basic.features import SparseFeature

sparse_features = [
    # 用户ID
    SparseFeature(
        name="user_id", 
        vocab_size=10000, 
        embedding_dim=32
        ),
    # 物品ID（使用权重共享）
    SparseFeature(
        name="item_id", 
        vocab_size=20000, 
        embedding_dim=32, 
        embedding_name="item"
        )
]
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称，用作输入键 |
| `vocab_size` | **[必需]** 唯一类别ID的数量（max_id+1 或 hash 空间大小） |
| `embedding_name` | 嵌入表名称，当两个特征共用一个embedding_name时共享词表，默认为 `name` |
| `embedding_dim` | 嵌入向量维度，设为 None 则自动调整大小为 `min(vocab_size ** 0.5, 64)` |
| `padding_idx` | 用于填充 token 的索引 |
| `init_type` | 嵌入向量的初始化类型：`"normal"`、`"uniform"`、`"xavier_uniform"` 等 |
| `init_params` | 初始化器参数，如 `{"mean": 0, "std": 0.01}` |
| `l1_reg` | 嵌入向量的 L1 正则化权重 |
| `l2_reg` | 嵌入向量的 L2 正则化权重 |
| `trainable` | 嵌入向量是否可训练 |
| `pretrained_weight` | 可选的预训练嵌入权重 |

## 序列特征

适用于用户行为序列，如点击序列、浏览序列、购买序列等。简单示例如下：

```python
from nextrec.basic.features import SequenceFeature

sequence_features = [
    # 简单序列：平均池化
    SequenceFeature(
        name="click_hist",
        vocab_size=100000,
        embedding_dim=32,
        max_len=50,
        combiner="mean"
    ),
    # 使用点积注意力
    SequenceFeature(
        name="browse_hist",
        vocab_size=100000,
        embedding_dim=32,
        max_len=50,
        combiner="dot_attention"
    ),
]
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称，用作输入键 |
| `vocab_size` | **[必需]** 序列词汇表中唯一ID的数量 |
| `max_len` | 用于填充/截断的最大序列长度 |
| `embedding_name` | 嵌入表名称，当两个特征共用一个embedding_name时共享词表 |
| `embedding_dim` | 嵌入向量维度，设为 None 则自动调整大小为 `min(vocab_size ** 0.5, 64)` |
| `combiner` | 序列嵌入的池化方法，可选项包含：`mean`,`sum`,`concat`,`dot_attention`,`self_attention` |
| `padding_idx` | 用于填充 token 的索引 |
| `init_type` | 嵌入向量初始化类型 |
| `init_params` | 初始化器参数 |
| `l1_reg` | 嵌入向量的 L1 正则化权重 |
| `l2_reg` | 嵌入向量的 L2 正则化权重 |
| `trainable` | 嵌入向量是否可训练 |
| `pretrained_weight` | 可选的预训练嵌入权重 |

---

## 下一步

- [基类模型的生命周期](./base-model.md) - 了解如何使用这些特征训练模型
- [数据预处理](./data-processor.md) - 了解如何处理原始数据
- [快速开始](../getting-started.md) - 完整的使用示例
