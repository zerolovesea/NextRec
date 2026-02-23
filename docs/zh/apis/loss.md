---
title: 损失函数
description: 支持的损失函数
---

# 损失函数

NextRec支持10余种不同任务类型的损失函数。包括pointwise，pairwise，listwise。具体支持的损失函数如下：

## Pointwise 损失函数

| 损失函数名称 | 别名 | 描述 | 适用任务 |
|:------------|:-----|:-----|:---------|
| `bce` | `binary_crossentropy` | 二元交叉熵损失，用于二分类任务 | 二分类 |
| `weighted_bce` | - | 加权二元交叉熵损失，支持样本权重 | 二分类 |
| `focal` | `focal_loss` | 焦点损失，用于处理类别不平衡问题 | 二分类 |
| `cb_focal` | `class_balanced_focal` | 类别平衡焦点损失，需要 `class_counts` 参数 | 多分类/二分类 |
| `crossentropy` | `ce` | 交叉熵损失，用于多分类任务 | 多分类 |
| `mse` | - | 均方误差损失，用于回归任务 | 回归 |
| `mae` | - | 平均绝对误差损失，用于回归任务 | 回归 |

## Pairwise 损失函数

| 损失函数名称 | 描述 | 适用任务 |
|:------------|:-----|:---------|
| `bpr` | 贝叶斯个性化排序 (Bayesian Personalized Ranking) 损失 | 排序 |
| `hinge` | Hinge 损失 (SVM 风格) | 排序 |
| `triplet` | 三元组损失，用于学习 item 嵌入 | 表示学习 |

## Listwise 损失函数

| 损失函数名称 | 描述 | 适用任务 |
|:------------|:-----|:---------|
| `sampled_softmax` / `softmax` | 采样 Softmax 损失，用于大规模排序 | 排序 |
| `infonce` | InfoNCE 损失，对比学习常用 | 表示学习 |
| `listnet` | ListNet 损失，基于列表的排序 | 排序 |
| `listmle` | ListMLE 损失，最大似然估计方法 | 排序 |
| `approx_ndcg` | 近似 NDCG 损失，直接优化 NDCG 指标 | 排序 |

## 使用示例

在基类模型`BaseModel`的`compile`参数，通过修改该方法中的`loss`以及`loss_params`参数来配置损失函数。示例代码：

```python
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="binary_crossentropy", # 设置损失函数
)
```

## NextRec CLI 集成

在命令行工具NextRec CLI中，通过修改训练配置文件中的`loss`参数来进行调整。当多任务时，需要依次设置不同任务的损失函数。

```yaml
train:
  loss:
    - 'bce'
    - 'focal'
  loss_params:
    - {} 
    - alpha: 0.8
      gamma: 2.0
```

---

## 下一步

- [评估指标](./metrics.md) - 支持的评估指标
- [CLI 工具](../cli/nextrec-cli.md) - 命令行工具 NextRec CLI