---
title: 样本加权
description: 多任务的样本加权
---

# 样本加权

在多任务场景下，经常会遇到不同任务的需要的关注程度不同。例如某些场景下，对用户最后的转化率更关心，有的时候需要对用户的点击率/注册率更关心。这时需要对不同任务进行加权，来让模型进行适当的侧重学习。

多任务加权经过了一段时间的发展，从最早的手动自定义任务权重，到模型通过网络来学习动态权重。NextRec同样提供了自定义任务权重，以及基于Grad Norm的动态调参。

## 自定义任务权重

在前面的[章节](../apis/base-model.md)里，我们了解到基类模型`BaseModel`的`compile`参数，通过修改该方法中的`loss_weights`参数来配置不同任务的权重。

以一个简单的ESMM模型为例，拥有两个任务，我们设置`loss_weights=[0.3, 0.7]`，来让损失更侧重conversion任务。在每个epoch里，会为每个任务的loss乘以这个权重值。

```python
from nextrec.models.multi_task.esmm import ESMM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

model = ESMM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    ctr_params={"dims": [64, 32], "activation": "relu", "dropout": 0.4},
    cvr_params={"dims": [64, 32], "activation": "relu", "dropout": 0.4},
    target=task_labels,
    task=["click", "conversion"],
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    loss_weights=[0.3, 0.7] # 我们在这里赋予任务权重
)
```

## Grad Norm动态权重

自定义任务权重是非常粗浅的，基于先验知识的方法，显然并不符合数据驱动的宗旨。这当中有几个核心问题：

- 不同任务的loss的尺度是不一样的，尤其是出现在回归任务和分类任务时，mse和bce的尺度差很大，导致大的loss会主导整个训练任务。对于这个问题，工业界层用loss归一化来调整。
- 不同任务的loss收敛速度不一致，对于简单的任务，loss很快收敛，而难的任务则收敛的很慢。然而全局而言，模型只看到了整体的loss快速收敛，而忽视了困难任务。

在实际场景下，不同任务的损失权重，应该随着样本的变化，训练的变化，才能找到最佳的数值。对于这个问题，2018年发布的论文GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks中给出了作者的解决方案。

GradNorm的主要思想是，通过每个任务对共享参数的梯度强度，来更细致的了解哪个任务对参数的影响更大，进而调整弱势任务的权重。这比loss更近了一步，上升到了参数层面。

在NextRec 0.4.13版本中，加入了对grad norm的支持，只需要将之前comile里的loss_weights方法改为{"method": "grad_norm", "alpha": 1.5, "lr": 0.025}或"grad_norm"即可。

示例代码如下：

```python
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 5e-4, "weight_decay": 1e-4},
    loss=["bce", "bce"],
    loss_weights={"method": "grad_norm", "alpha": 1.5, "lr": 0.025}, # 或者可以直接写"grad_norm"
)
```

## NextRec CLI 集成

在命令行工具NextRec CLI中，通过修改训练配置文件的中的`train`参数，来调整样本权重的参数

```yaml
train:
    loss_weights: [0.3, 0.7] # 或{"method": "grad_norm", "alpha": 1.5, "lr": 0.025}
```

---

## 下一步

- [损失函数](./loss.md) - 支持的损失函数
- [CLI 工具](../cli/index.md) - 命令行工具 NextRec CLI
