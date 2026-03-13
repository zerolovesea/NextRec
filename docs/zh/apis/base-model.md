---
title: 基类模型的生命周期
description: BaseModel 的初始化、compile、fit、evaluate/predict 的完整参数说明
---

# 基类模型的生命周期

NextRec里几乎所有模型都继承自基类模型（`nextrec.basic.model.BaseModel`），它封装了模型训练/评估/推理等通用方法。要构建一个推荐算法模型，需要经过**初始化模型，模型定义，编译，训练，验证，推理环节**。


## 初始化

以下是一个简单的DeepFM模型的初始化定义流程，在此之前我们需要定义特征，然后初始化模型参数。

```python
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

# 定义特征
dense_features = [DenseFeature("age"), DenseFeature("income")]
sparse_features = [
    SparseFeature("user_id", vocab_size=1000000, embedding_dim=32),
    SparseFeature("item_id", vocab_size=2000000, embedding_dim=32),
]

# 创建模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={                    # DNN 部分的 MLP 参数
        "hidden_dims": [256, 128],  # 隐藏层维度
        "activation": "relu",       # 激活函数
        "dropout": 0.2,             # Dropout 比例
    },    
    target="click",
    task="binary",
    device="cpu",
    embedding_l2_reg=1e-5,
    dense_l2_reg=1e-4,
    session_id="deepfm_exp001",
)
```


### 参数说明

| 参数 | 说明 |
|-----|------|
| `dense_features` | 稠密连续特征列表 List[DenseFeature] |
| `sparse_features` | 稀疏特征列表 List[SparseFeature] |
| `sequence_features` | 序列特征列表 List[SequenceFeature] |
| `target` | 目标列名，如 `"click"` 或 `["click", "buy"]`（多任务） |
| `id_columns` | 用于计算 GAUC 或推理时透传的 ID 列 |
| `task` | 任务类型：`"binary"`、`"regression"`|
| `training_mode` | 训练模式：`"pointwise"`、`"pairwise"`、`"listwise"` |
| `embedding_l1_reg` | 嵌入向量参数 L1 正则化强度 |
| `dense_l1_reg` | 密集向量参数 L1 正则化强度 |
| `embedding_l2_reg` | 嵌入向量参数 L2 正则化强度，如 `1e-5` |
| `dense_l2_reg` | 密集向量参数 L2 正则化强度，如 `1e-4` |
| `device` | 计算设备：`"cpu"`、`"cuda:0"`、`"mps"` |
| `session_id` | 实验会话名，用于日志管理 |
| `distributed` | **[无需主动设置]** 是否启用 DistributedDataParallel |
| `rank` | **[无需主动设置]** 全局排名，默认从环境变量 `RANK` 获取 |
| `world_size` | **[无需主动设置]** 进程数，默认从环境变量 `WORLD_SIZE` 获取 |
| `local_rank` | **[无需主动设置]** 本地设备的RANK |
| `ddp_find_unused_parameters` | **[无需主动设置]** DDP 模型中是否存在未使用参数 |

## 配置训练参数

模型初始化以后，需要为训练配置一些必要的参数，例如优化器，学习率调度器，损失函数等。简单示例如下：

```python
# 单任务训练
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    loss="binary_crossentropy"
)

# 带学习率调度
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    scheduler="cosine",
    scheduler_params={"T_max": 10, "eta_min": 1e-6}
)

# 带warmup的学习率调度
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    scheduler="cosine",
    scheduler_params={"T_max": 10},
    warmup={"epochs": 2, "start_factor": 0.1, "end_factor": 1.0}
)

# 多任务训练（带损失权重）
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="binary_crossentropy",
    loss_weights={"click": 1.0, "buy": 0.5}
)

# 使用 GradNorm 动态权重
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="binary_crossentropy",
    loss_weights={"click": 1.0, "buy": 1.0, "method": "grad_norm", "alpha": 1.5}
)
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `optimizer` | 优化器：`"adam"`,`"sgd"`,`"adamw"`,`"adagrad"`,`"rmsprop"` 或pytorch优化器实例 |
| `optimizer_params` | 优化器参数，如 `{"lr": 0.001, "weight_decay": 1e-5}` |
| `scheduler` | 学习率调度器：`"step"`,`"cosine"`|
| `scheduler_params` | 调度器参数，如 `{"step_size": 10, "gamma": 0.1}` |
| `warmup` | 可选 warmup 配置，支持 `False/None/True/dict`。`True` 使用默认值：`{"epochs":1,"start_factor":0.1,"end_factor":1.0}`；`dict` 可显式配置 `enabled/epochs/start_factor/end_factor`。即使不配置 `scheduler` 也可单独使用 warmup（结束后保持 `base_lr`）。 |
| `loss` | 损失函数：`"bce"`,`"weighted_bce"`,`"mse"`,`"focal"`,`"bpr"`等 |
| `loss_params` | 损失函数参数，多任务时可为列表，其中为每个损失函数的参数 |
| `loss_weights` | 多任务损失权重，如 `{"click": 1.0, "buy": 0.5}`时代表click任务的样本权重为1，buy任务的样本权重为0.5；使用时 `{"method": "grad_norm", "alpha": 1.5, "lr": 0.025}` 启用 GradNorm 进行动态权重计算|
| `ignore_label` | 计算损失时忽略的标签值，默认 -1 意味着当模型看到标签为-1的样本时不会计算损失 |

## 模型训练

准备工作完成以后就可以开始训练。示例代码如下：

```python
# 简单训练
model.fit(
    train_data=train_df,
    epochs=10,
    batch_size=256,
    metrics=["auc", "logloss"]
)

# 带验证集和早停
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    epochs=10,
    batch_size=256,
    metrics=["auc", "gauc", "logloss"],
    user_id_column="user_id",
    early_stop_patience=5,
    early_stop_monitor_task="click"
)

# 使用 W&B
model.fit(
    train_data=train_df,
    epochs=10,
    use_wandb=True,
    wandb_api="your-api-key",
    wandb_kwargs={"project": "nextrec-exp"}
)
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `train_data` | 训练数据：DataFrame / Dict / DataLoader |
| `valid_data` | 验证数据：DataFrame / Dict / DataLoader |
| `metrics` | 评估指标，如 `['auc', 'logloss']` |
| `epochs` | 训练轮数 |
| `shuffle` | 是否打乱训练数据 |
| `batch_size` | 批次大小 |
| `user_id_column` | GAUC 计算所需的用户 ID 列 |
| `valid_split` | 当未输入valid_data时，从train_data进行划分出验证集的比例，如 `0.1` |
| `early_stop_patience` | 早停轮数 |
| `early_stop_monitor_task` | 早停参考的任务，会根据指定任务和 `metrics` 中第一个指标进行早停，例如 `metrics=['auc', ...]` 时监控 `val_auc_{task}` |
| `num_workers` | Pytorch DataLoader 进程数，提高以加速数据加载 |
| `use_tensorboard` | 是否启用 tensorboard 日志 |
| `use_wandb` | 是否启用 Weights & Biases 日志 |
| `use_swanlab` | 是否启用 SwanLab 日志 |
| `wandb_api` | W&B API 密钥 |
| `swanlab_api` | SwanLab API 密钥 |
| `wandb_kwargs` | wandb.init 的额外参数 |
| `swanlab_kwargs` | swanlab.init 的额外参数，例如`{"project": "nextrec", "name": nextrec_model}` |
| `auto_ddp_sampler` | **[无需主动设置]** 自动附加 DistributedSampler |
| `log_interval` | 每 N 轮记录验证指标 |
| `note` | 训练运行的备注 |
| `summary_sections` | 打印的摘要部分，如 `["feature", "model", "train", "data"]` |

说明：
- `early_stop_monitor_task` 仅在多任务训练下生效。
- 当前早停监控的指标固定取 `metrics` 中的第一个指标。

### 训练产物

开始训练后，会在命令执行的当前路径下生成`nextrec_logs`文件夹，并在其中的`session_id`
路径下生成一系列训练产物，包括日志等

```
nextrec_logs/
└── deepfm_exp001/
    ├── checkpoint_deepfm.pt          # 模型检查点
    ├── best_deepfm.pt                # 最佳模型
    ├── metrics.json                  # 训练指标
    ├── predictions/                  # 预测结果
    └── processor.pkl                 # 数据处理器
```

## 模型评估

在模型训练的每一轮里，都会计算评估指标，这是通过基类模型的evaluate方法实现的。在训练完成后，也可以单独调用这个方法，在其他数据集上进行评估。

```python
# 基础评估
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "logloss"]
)
print(metrics)  # {'auc': 0.8532, 'logloss': 0.3421}

# 带 GAUC
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "gauc", "logloss", "precision@10"],
    user_id_column="user_id"
)

# 多任务评估
metrics = model.evaluate(
    valid_df,
    metrics={
        "click": ["auc", "logloss"],
        "buy": ["auc", "logloss"]
    }
)

# 按场景分组评估
metrics = model.evaluate(
    valid_df,
    metrics=["auc", "precision", "recall"],
    group_by="product"
)
print(metrics["overall"])
print(metrics["grouped"])
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `data` | 评估数据：DataFrame / Dict / DataLoader |
| `metrics` | 指标名称，如 `['auc', 'logloss']` |
| `batch_size` | 批次大小 |
| `user_ids` | GAUC 计算的用户 ID 列名 |
| `user_id_column` | 用户 ID 列名 |
| `group_by` | 按列分组评估，可传单列或列名列表 |
| `num_workers` | Pytorch DataLoader 进程数，提高以加速数据加载 |
| `thresholds` | 二分类任务的阈值 |
| `show_data_summary` | 是否记录数据分布的统计摘要 |
| `show_confusion_matrix` | 是否输出混淆矩阵 |


## 模型推理

模型训练完后，可以调用模型的predict方法进行推理。示例代码如下：

```python
# 简单预测
predictions = model.predict(test_df)

# 预测并保存
predictions = model.predict(
    test_df,
    save_path="./predictions.csv",
    save_format="csv"
)

# 返回 numpy 数组
predictions = model.predict(test_df, return_dataframe=False)
# {'click': array([0.85, 0.12, ...])}
```

### 参数说明

| 参数 | 说明 |
|-----|------|
| `data` | 预测数据：文件路径 / 字典 / DataFrame / DataLoader |
| `batch_size` | 批次大小 |
| `save_path` | 保存预测结果的路径 |
| `save_format` | 保存格式：`"csv"` 或 `"parquet"` |
| `return_dataframe` | 是否返回 DataFrame，否则返回 NumPy 数组 |
| `stream_chunk_size` | 大数据集流式处理时每块数据的行数，需要根据机器内存压力酌情设置 |
| `num_workers` | Pytorch DataLoader 进程数，提高以加速数据加载 |
| `prefetch_factor` | Pytorch Dataloader预取数据的批次数，提高以加速数据加载 |
| `num_processes` | 流式文件推理的进程数，用于多进程推理 |
| `processor` | 可选的 DataProcessor 用于转换输入数据 |
| `profiler` | **[Bool]** 可选的管道阶段性能分析器，设置为True时会输出推理流程中各环节计算时间 |


## ONNX 导出与推理

ONNX（Open Neural Network Exchange）由Microsoft和Facebook共同发起，是一个开放的深度学习模型交换格式，用于在不同的深度学习框架之间导出、共享和部署模型。使用ONNX的官方推理引擎进行推理，会比原框架推理更快，适用于生产环境的部署。在NextRec中支持将模型导出为onnx格式并进行推理。

```python
# 导出 ONNX
onnx_path = model.export_onnx(
    save_path="./model.onnx",
    batch_size=1
)

# 使用 ONNX 推理
import nextrec.utils.onnx as onnx

predictions = onnx.predict_onnx(
    onnx_path="./model.onnx",
    data=test_df,
    batch_size=512,
    return_dataframe=True
)
```

---

## 完整示例

```python
import pandas as pd
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.basic.features import DenseFeature, SparseFeature

# 1. 准备数据
train_df = pd.read_csv("train.csv")

# 2. 定义特征
dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=100000, embedding_dim=32),
    SparseFeature("item_id", vocab_size=200000, embedding_dim=32),
]

# 3. 创建模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="click",
    task="binary",
    device="cuda:0"
)

# 4. 配置训练
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="binary_crossentropy"
)

# 5. 训练
model.fit(
    train_data=train_df,
    epochs=10,
    batch_size=256,
    metrics=["auc", "logloss"],
    early_stop_patience=5
)

# 6. 评估
metrics = model.evaluate(valid_df, metrics=["auc", "logloss"])
print(f"AUC: {metrics['auc']:.4f}")

# 7. 推理
predictions = model.predict(test_df)
```

---

## 下一步

- [数据处理](../apis/data-processor.md) - 使用数据预处理工具
- [教程与示例](../tutorial/index.md) - 了解如何处理原始数据
- [指标详解](./metrics.md) - 评估指标
