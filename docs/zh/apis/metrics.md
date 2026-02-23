---
title: 评估指标
description: 指标计算与配置说明（AUC/GAUC/KS 与 ranking@K）
---

# 评估指标

NextRec 支持丰富的评估指标，涵盖分类、排序、召回等场景。

## 支持的指标列表

### 分类/回归指标

| 指标名称 | 别名 | 描述 |
|:---------|:-----|:-----|
| `auc` | - | ROC AUC，衡量分类器区分正负样本能力 |
| `gauc` | - | 分组 AUC，按用户分组计算，更适合推荐场景 |
| `ks` | - | Kolmogorov-Smirnov 统计量 |
| `logloss` | - | 对数损失，衡量概率预测准确性 |
| `accuracy` | `acc` | 准确率 |
| `precision` | - | 精确率 |
| `recall` | - | 召回率 |
| `f1` | - | F1 分数 |
| `micro_f1` | - | Micro-F1 分数 |
| `macro_f1` | - | Macro-F1 分数 |
| `mse` | - | 均方误差 |
| `mae` | - | 平均绝对误差 |
| `rmse` | - | 均方根误差 |
| `r2` | - | R² 决定系数 |
| `mape` | - | 平均绝对百分比误差 |
| `msle` | - | 均方对数误差 |

### 排序/召回指标

| 指标名称 | 描述 |
|:---------|:-----|
| `precision@K` | Precision@K，前 K 个推荐中正样本比例 |
| `recall@K` | Recall@K，正样本被召回的比例 |
| `ndcg@K` | NDCG@K，归一化折损累计增益 |
| `mrr@K` | MRR@K，平均倒数排名 |
| `map@K` | MAP@K，平均精确度均值 |
| `hitrate@K` | HitRate@K，至少有一个正样本被推荐的覆盖率 |
| `topk_recall@K` | Top-K 召回率 |
| `topk_precision@K` | Top-K 精确率 |
| `lift@K` | Lift@K，Lift 指标 |
| `cosine` | 余弦相似度 |

> **注意**: K 支持的值通常为 5, 10, 20

---

## 二分类指标

### AUC（ROC AUC）

衡量分类器区分正负样本的能力，取值范围 [0, 1]，越接近 1 越好。

```python
metrics = model.evaluate(valid_df, metrics=["auc"])
# {'auc': 0.8532}
```

### LogLoss（对数损失）

衡量概率预测的准确性，越接近 0 越好。

```python
metrics = model.evaluate(valid_df, metrics=["logloss"])
# {'logloss': 0.3421}
```

### KS（Kolmogorov-Smirnov）

衡量正负样本分布的最大差距，取值范围 [0, 1]，越接近 1 越好。

```python
metrics = model.evaluate(valid_df, metrics=["ks"])
# {'ks': 0.4523}
```

### 其他分类指标

```python
metrics = model.evaluate(valid_df, metrics=["accuracy", "precision", "recall", "f1"])
# {'accuracy': 0.82, 'precision': 0.75, 'recall': 0.78, 'f1': 0.76}
```

---

## GAUC（分组 AUC）

GAUC 按用户分组计算 AUC，更适合推荐系统的评估场景。

### 为什么需要 GAUC？

- 普通 AUC 将所有样本混合计算，忽略了用户维度的差异
- GAUC 按每个用户计算 AUC，然后加权平均，更能反映用户体验

### 使用方法

```python
# 需要指定 user_id_column
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc"],
    user_id_column="user_id"
)

# {'auc': 0.85, 'gauc': 0.82}
```

**CLI 配置：**

```yaml
# train_config.yaml
train:
  metrics:
    - auc
    - gauc
  user_id_column: user_id  # 或 id_column: user_id
```

### 注意事项

如果不提供 `user_id_column`，`gauc` 会回退到普通 AUC 并打印警告：

```
Warning: gauc requires user_id_column, falling back to AUC
```

---

## Ranking@K 指标

推荐系统中常用 Top-K 排序指标来评估模型性能。

### Precision@K

前 K 个推荐中正样本的比例：

```python
metrics = model.evaluate(valid_df, metrics=["precision@10"], user_id_column="user_id")
# {'precision@10': 0.1234}
```

### Recall@K

正样本被召回的比例：

```python
metrics = model.evaluate(valid_df, metrics=["recall@10"], user_id_column="user_id")
# {'recall@10': 0.4567}
```

### NDCG@K（Normalized Discounted Cumulative Gain）

考虑排序位置的评估指标，越靠前越重要：

```python
metrics = model.evaluate(valid_df, metrics=["ndcg@10"], user_id_column="user_id")
# {'ndcg@10': 0.5678}
```

### MRR@K（Mean Reciprocal Rank）

第一个正样本位置的倒数均值：

```python
metrics = model.evaluate(valid_df, metrics=["mrr@10"], user_id_column="user_id")
# {'mrr@10': 0.4567}
```

### MAP@K（Mean Average Precision）

平均精确度的均值：

```python
metrics = model.evaluate(valid_df, metrics=["map@10"], user_id_column="user_id")
# {'map@10': 0.3456}
```

### HitRate@K

至少有一个正样本被推荐的覆盖率：

```python
metrics = model.evaluate(valid_df, metrics=["hitrate@10"], user_id_column="user_id")
# {'hitrate@10': 0.6789}
```

## 示例代码

```python
# 基础指标
model.fit(train_df, metrics=["auc", "logloss"])

# 带 GAUC
model.fit(train_df, metrics=["auc", "gauc"], user_id_column="user_id")

# 完整配置
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss", "ks", "precision@10", "recall@10", "ndcg@10"],
    user_id_column="user_id"
)
```

## NextRec CLI 配置

在命令行工具NextRec CLI中，通过修改训练配置文件中的metrics参数来进行调整。

```yaml
# train_config.yaml
train:
  metrics:
    - auc
    - gauc
    - logloss
    - ks
    - precision@10
    - recall@10
    - ndcg@10

# 数据配置中指定 user_id
data:
  id_column: user_id
```

---

## 下一步

- [日志管理](./session-logging.md) - 完整的训练流程
- [分布式训练](./distributed-training.md) - 实践指标配置
