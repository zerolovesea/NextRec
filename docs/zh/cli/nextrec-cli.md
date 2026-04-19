---
title: NextRec CLI
description: 基于 YAML 的训练/预测/评估命令行工具概览与常用配置
---

# NextRec CLI

为了更方便的训练和部署模型，减少编写代码的负担，我们提供了命令行工具`NextRec CLI`。用户只需要配置对应的配置文件即可一键完成训练，推理，验证。

## 基本用法

- `train`：训练
- `predict`：推理
- `evaluate`：评估

示例：

```bash
nextrec --mode=train --train_config=train_config.yaml
nextrec --mode=predict --predict_config=predict_config.yaml
nextrec --mode=evaluate --evaluate_config=evaluate_config.yaml
```

## 配置文件

我们看到，在不同的模式都涉及对应的配置文件。依次来看一下：

### 训练配置

训练任务的主配置文件通常是`train_config.yaml`，以下是训练一个多任务模型的示例配置，在其中对所有支持的配置项进行了注释。

```yaml
# 实验id和日志保存路径
session:
  id: test_session                     
  artifact_root: nextrec_logs          

# 特征配置路径
feature_config: feature_config.yaml
# 模型配置文件路径
model_config: cross_stitch.yaml

# 数据相关配置
data:
  # 训练数据的路径
  path: /path/to/your/data 
  # 训练文件的格式
  format: parquet
  # 训练集目标列
  target:
    - label_apply
    - label_credit
  # 分组指标使用的用户/样本分组列
  group_id: phone
  # 验证集比例，也可以在valid_path下直配置验证集数据路径
  valid_ratio: 0.2
  # valid_path
  # 是否使用数据流加载数据，适用于大规模数据集
  streaming: false                                                               

# 数据加载器相关配置
dataloader:
  batch_size: 1024
  shuffle: true
  # 流式模式下，一个文件是一个batch，此时需要设置chunk_size控制
  # 每次从文件中读取多少行数据，也就是一个batch
  chunk_size: 20000                                                                
  num_workers: 6
  prefetch_factor: 2

# 训练相关配置
train:
  optimizer: adam   
  optimizer_params:  
    lr: 0.001
  # 可选学习率调度器：step / cosine
  scheduler: cosine
  scheduler_params:
    T_max: 10
  # 可选warmup（默认关闭）
  warmup:
    enabled: true
    epochs: 1
    start_factor: 0.1
    end_factor: 1.0
  # 损失函数，多任务时需要配置多个损失函数
  loss:                                                                         
    - 'bce'
    - 'focal' 
  loss_params:
    - {}                                                                   
    - alpha: 0.8
      gamma: 2.0
  # 当损失函数设置为weighted_bce时，可配置auto_balance
  # 来让模型根据正负样本的比例来赋权
  # - auto_balance: true
  # 可选的损失权重或GradNorm，权重损失示例 [1.0, 0.5]
  # GradNorm示例 {"method": "grad_norm", "alpha": 1.5, "lr": 0.025}
  loss_weights: [1.0, 0.5]
  # 忽略标签值，遮盖不需要计算损失的样本，例如未完件样本的授信标签
  # 可以设置为-1，此时多任务模型会忽略这些样本的授信任务损失计算
  ignore_label: -1 
  # 评估指标
  metrics: 
    - auc
    - ks
    - precision
    - recall
    - topk_recall@5
    - topk_recall@10
    - topk_precision@5
    - topk_precision@10
  epochs: 3
  early_stop_patience: 5
  # 多任务时可指定早停监控的任务名，监控指标仍取metrics中的第一个
  early_stop_monitor_task: label_register
  # 训练中的验证阶段按列分组展示指标
  valid_group_by: product
  shuffle: true
  device: cpu
  # 每N个batch记录一次日志，1表示每个batch都记录
  log_interval: 1
  # 是否使用SwanLab记录实验日志和模型
  use_swanlab: true  
  # 实验描述，会记录在SwanLab和训练日志中
  note: "NextRec test note"
  # SwanLab API Key
  swanlab_api: your swanlab api       
  swanlab_kwargs:
    # SwanLab项目名称
    project: nextrec
    # SwanLab实验名称 
    name: experiment-test 
# 导出ONNX
export_onnx:  
  enable: true
  batch_size: 1024                                                                 
       
# nextrec --mode train --train_config train_config.yaml
```

说明：
- `train.early_stop_patience` 控制早停轮数，设为 `0` 表示关闭早停。
- `train.early_stop_monitor_task` 仅在多任务训练下生效，用于指定早停监控哪个任务。
- 当前早停监控的指标仍然取 `train.metrics` 中的第一个指标，例如 `metrics: [auc, ...]` 时会监控 `val_auc_{task}`。
- `train.valid_group_by` 只作用在训练中的验证阶段，用于按列分组展示验证指标，不改变训练 loss，也不改变 early stop 监控口径。

Warmup 参数说明：

- `train.warmup` 支持 `null/bool/dict`。
- 默认关闭：不填或设为 `false`。
- 设为 `true` 时使用默认值：`epochs=1,start_factor=0.1,end_factor=1.0`。
- 设为 `dict` 时可自定义：
  - `enabled`：是否启用 warmup。
  - `epochs`：warmup 持续的 epoch 数。
  - `start_factor`：warmup 起始学习率系数，实际学习率为 `base_lr * start_factor`。
  - `end_factor`：warmup 结束学习率系数，通常设为 `1.0`。
- 不配置 `scheduler` 也可以单独使用 warmup，此时 warmup 完成后学习率保持在 `base_lr`。
- warmup 会先执行，再进入 `scheduler` 指定的主调度策略（例如 `cosine` 或 `step`）。

### 特征配置

在训练配置中，有一项`feature_config`配置，其中输入的是特征配置文件的路径。在特征配置文件里为模型需要的特征配置预处理和参数定义。示例配置文件如下，可以看到这里的配置和前面Python API的[预处理配置](../apis/data-processor.md)以及[特征配置](../apis/features.md)是一致的。

对于 `sparse` 和 `sequence` 特征，还可在 `processor_config` 中增加 `filter_value` 与 `keep_value` 进行样本过滤，并通过 `match_mode` 指定匹配模式（`exact`/`contains`/`regex`），详细规则见[数据预处理](../apis/data-processor.md)中的“sparse/sequence 样本过滤教程”。

```yaml
# 稠密特征（dense）：processor_config 使用 scaler
dense:
  price:
    processor_config:
      - {scaler: robust, fill_na: 0.0}   # 可选 scaler: standard/minmax/robust/maxabs/log/none
      - {scaler: [log, minmax], fill_na: 0.0}  # 支持顺序变换：先 log1p 再 minmax
    embedding_config:
      - {embedding_dim: 16, input_dim: 1, use_projection: true, proj_dim: 8}
      - {input_dim: 1, use_projection: true, proj_dim: 1, projection_type: mlp, mlp_hidden_dims: [8, 4], mlp_activation: relu, mlp_dropout: 0.0}

# 稀疏特征（sparse）：processor_config 使用 encode_method
sparse:
  user_id:
    processor_config:
      - {encode_method: label, min_freq: 2, fill_na: "<UNK>"}
    embedding_config:
      - {embedding_name: user_item_shared, embedding_dim: 32, padding_idx: 0, init_type: xavier_uniform}
  item_id:
    processor_config:
      - {encode_method: hash, hash_size: 200000, min_freq: 2, keep_value: ["1001", "1002"], filter_value: ["-1"], match_mode: exact}
    embedding_config:
      - {embedding_name: user_item_shared, embedding_dim: 32, padding_idx: 0}

# 序列特征（sequence）：支持 separator/max_len/pad_value/truncate
sequence:
  hist_item_seq:
    processor_config:
      - {
          encode_method: hash,
          hash_size: 200000,
          min_freq: 2,
          separator: ",",
          max_len: 50,
          pad_value: 0,
          truncate: pre,
          keep_value: ["1001", "1002"],
          filter_value: ["-1"],
          match_mode: contains
        }
    embedding_config:
      - {embedding_name: user_item_shared, embedding_dim: 32, max_len: 50, combiner: mean, padding_idx: 0}
```

说明：
- `dense` 类型没有 `encode_method`，请使用 `scaler`。
- `dense.processor_config.scaler` 支持字符串或列表（列表按顺序执行）。
- `dense.embedding_config` 支持 `projection_type/mlp_hidden_dims/mlp_activation/mlp_dropout`，可用于多维 dense 的 MLP 投影。
- `sparse` / `sequence` 类型没有 `type` 字段，请使用 `encode_method`。
- 样本过滤默认 `match_mode: exact`，可选 `contains`（子串匹配）和 `regex`（正则匹配）。
- 若不填写 `embedding_name`，默认使用该特征的输出名（例如 `item_id_hash`），即不与其他特征共享 embedding 表。

### 模型配置

由于每个模型都有各自自定义的配置，需要为每个模型单独配置。我们提供了模型的[默认配置参数](https://github.com/zerolovesea/NextRec/tree/main/nextrec_cli_preset/model_configs)，并附上了对应模型的背景介绍，以方便用户快速上手模型。

上述文件准备完成后，使用`nextrec --mode=train --train_config=train_config.yaml`即可开始训练。

### 推理配置

同样需要配置推理配置文件，示例如下：

```yaml
# 训练任务的session路径
checkpoint_path: nextrec_logs/test_session

# 模型配置路径
model_config: model_configs/cross_stitch.yaml

predict:
  # 推理集数据路径
  data_path: /path/to/your/data
  # 数据集格式
  source_data_format: csv
  # 推理结果中需要透传的 key 列
  key_column: phone
  # 推理结果文件名
  # 结果文件将保存至{checkpoint_path}/predictions/{name}.{save_data_format}
  name: pred 
  # 推理集保存格式，支持csv与parquet
  save_data_format: csv 
  # 推理前按候选值展开样本；每条样本会做笛卡尔积复制
  # 用于多场景任务下，每个样本需要根据不同的场景特征进行推理
  expand:
    custom_col: [custom_class_1, custom_class_2, custom_class_3]
  # 推理结束查看前N行数据
  preview_rows: 5
  batch_size: 512
  # Dataloader的多进程数
  num_workers: 4
  # 推理的多进程数
  num_processes: 1 
  device: cpu 
  # 是否流式推理
  streaming: true 
  chunk_size: 20000 

# nextrec --mode predict --predict_config predict_config.yaml

```

文件准备完成后，使用`nextrec --mode predict --predict_config predict_config.yaml`即可开始推理。

说明：
- `predict.expand` 为“列名 -> 候选值列表”的映射。
- 推理时会先对每条样本做笛卡尔积展开，再送入模型。
- 上述示例会把每条样本复制 3 次，并分别写入 `custom_class_1/custom_class_2/custom_class_3` 到 `custom_col` 列。
- 展开列会自动追加到预测结果中，便于区分不同场景。
- 流式推理会按 chunk 展开，不会先把整份数据一次性放大到内存中，但展开倍数越大，单个 chunk 的内存与写出量也会同步增长，需要根据用户的机器性能来选择合适的参数。

### 验证配置

通过验证配置，能够使用`nextrec --mode evaluate --evaluate_config=evaluate_config.yaml`来一键使用现有的模型在新数据集上执行验证。示例配置文件如下：

```yaml
# 训练任务的session路径
checkpoint_path: nextrec_logs/test_session

# 模型配置路径
model_config: model_configs/cross_stitch.yaml

evaluate:
  # 验证集路径，需要包含标签列
  data_path: /path/to/your/data
  # 验证集数据格式
  source_data_format: csv                 
  key_column: user_id
  group_id: user_id
  # 按自定义的场景列进行分组评估，可填单列
  group_by: custom_col
  # 目标列
  target:
    - click
    - conversion

  device: cpu
  streaming: false

  metrics:
    - auc
    - ks
    - precision
    - recall

  # 混淆矩阵
  confusion_matrix:
    enable: true
    # 混淆矩阵的阈值计算
    thresholds: [0.5, 0.9] 
                                         
# nextrec --mode evaluate --evaluate_config evaluate_config.yaml
```

说明：
- `evaluate.group_by` 用于按列分组统计指标，例如 `custom_col=custom_class_1/custom_class_2/...`。
- 配置后会同时输出整体指标和分组指标。
- 分组结果会保存到 `{checkpoint_path}/evaluate/{name}/metrics_by.csv` 与 `metrics_by.json`。
- `gauc` 等依赖分组的指标使用 `group_id` 作为用户 ID，`key_column` 只负责结果透传，`group_by` 只负责切分统计口径。

## 下一步

- [NextRec Studio](../cli/nextrec-studio.md) - 了解使用NextRec Studio，进一步降低NextRec CLI的上手成本
