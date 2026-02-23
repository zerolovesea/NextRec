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
  # 用户/样本ID列名称                                                             
  id_column: phone
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

### 特征配置

在训练配置中，有一项`feature_config`配置，其中输入的是特征配置文件的路径。在特征配置文件里为模型需要的特征配置预处理和参数定义。示例配置文件如下，可以看到这里的配置和前面Python API的[预处理配置](../apis/data-processor.md)以及[特征配置](../apis/features.md)是一致的。

```yaml
# 稠密特征
dense:
  user_active_days_7:
    # 预处理配置
    processor_config: {type: numeric, scaler: standard}
    # 嵌入配置
    embedding_config: {name: user_active_days_7, input_dim: 1, embedding_dim: 8, use_projection: false}
  user_ctr:
    processor_config: {type: numeric, scaler: standard}
    embedding_config: {name: user_ctr, input_dim: 1, embedding_dim: 8, use_projection: false}
  item_price:
    processor_config: {type: numeric, scaler: standard}
    embedding_config: {name: item_price, input_dim: 1, embedding_dim: 8, use_projection: false}
  item_popularity:
    processor_config: {type: numeric, scaler: standard}
    embedding_config: {name: item_popularity, input_dim: 1, embedding_dim: 8, use_projection: false}
  time_since_last_click:
    processor_config: {type: numeric, scaler: standard}
    embedding_config: {name: time_since_last_click, input_dim: 1, embedding_dim: 8, use_projection: false}

# 稀疏特征
sparse:
  user_id:
    # 预处理配置
    processor_config: {type: sparse, encode_method: hash, hash_size: 100000, min_freq: 1}
    # 嵌入配置
    embedding_config: {name: user_id, embedding_dim: 8, padding_idx: 0}
  item:
    processor_config: {type: sparse, encode_method: hash, hash_size: 5000, min_freq: 1}
    embedding_config: {name: item, embedding_dim: 8, padding_idx: 0}
  gender:
    processor_config: {type: sparse, encode_method: hash, hash_size: 10, min_freq: 1}
    embedding_config: {name: gender, embedding_dim: 8, padding_idx: 0}
  city:
    processor_config: {type: sparse, encode_method: hash, hash_size: 20, min_freq: 1}
    embedding_config: {name: city, embedding_dim: 8, padding_idx: 0}
  device:
    processor_config: {type: sparse, encode_method: hash, hash_size: 10, min_freq: 1}
    embedding_config: {name: device, embedding_dim: 8, padding_idx: 0}
  channel:
    processor_config: {type: sparse, encode_method: hash, hash_size: 10, min_freq: 1}
    embedding_config: {name: channel, embedding_dim: 8, padding_idx: 0}
  age_bucket:
    processor_config: {type: sparse, encode_method: hash, hash_size: 20, min_freq: 1}
    embedding_config: {name: age_bucket, embedding_dim: 8, padding_idx: 0}

# 序列特征
sequence:
  hist_item_seq:
    # 预处理配置
    processor_config: {type: sequence, encode_method: hash, hash_size: 5000, min_freq: 1, max_len: 30, pad_value: 0, truncate: post, separator: ','}
    # 嵌入配置
    embedding_config: {name: hist_item_seq, vocab_size: 5000, max_len: 30, combiner: mean, embedding_dim: 8, padding_idx: 0, shared_with: item}
```

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
  # 推理集的id列
  id_column: phone
  # 推理结果文件名
  # 结果文件将保存至{checkpoint_path}/predictions/{name}.{save_data_format}
  name: pred 
  # 推理集保存格式，支持csv与parquet
  save_data_format: csv 
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
  id_column: user_id                    
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

## 下一步

- [NextRec Studio](../cli/nextrec-studio.md) - 了解使用NextRec Studio，进一步降低NextRec CLI的上手成本
