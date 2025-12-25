# NextRec CLI 使用指南

NextRec 提供了强大的命令行界面（CLI），支持通过 YAML 配置文件进行模型训练和预测。本文档详细介绍了 CLI 的使用方法和配置规范。

## 目录

- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
  - [训练配置文件](#训练配置文件)
  - [预测配置文件](#预测配置文件)
  - [特征配置文件](#特征配置文件)
  - [模型配置文件](#模型配置文件)
- [命令行参数](#命令行参数)
- [完整示例](#完整示例)
- [高级功能](#高级功能)

---

## 快速开始

### 安装

```bash
pip install nextrec
```

### 基本用法

```bash
# 训练模型
nextrec --mode=train --train_config=path/to/train_config.yaml

# 运行预测
nextrec --mode=predict --predict_config=path/to/predict_config.yaml
```

---

## 配置文件说明

NextRec CLI 使用 YAML 配置文件来定义训练和预测流程。我们在`nextrec_cli_preset`下提供了配置文件的模板，其中包括四类配置文件：

1. **训练配置文件** (`train_config.yaml`) - 定义训练流程
2. **预测配置文件** (`predict_config.yaml`) - 定义预测流程
3. **特征配置文件** (`feature_config.yaml`) - 定义特征处理方式
4. **模型配置文件** (`model_config.yaml`) - 定义模型架构和参数

### 训练配置文件

训练配置文件定义了完整的训练流程，包括数据路径、训练参数、优化器等。

#### 配置结构

```yaml
session:
  id: my_experiment_session            # 实验唯一标识
                                       # 用于日志与 checkpoint 目录命名
  artifact_root: nextrec_logs          # 产物根目录
                                       # 最终路径: {artifact_root}/{id}/

data:
  path: /path/to/training/data         # 训练数据路径（文件或目录）
  format: parquet                      # 数据格式：csv, parquet, feather
                                       # 使用 auto 自动识别
  target: label                        # 目标列
                                       # 单任务: 'label' (字符串)
                                       # 多任务: ['label1', 'label2'] (列表)
  # id_column: user_id                 # 用户/样本ID列（可选，用于 GAUC）
  # user_id_column: user_id            # id_column 的别名
  valid_ratio: 0.2                     # 自动切分验证集比例 (0.0-1.0)
  # val_path: /path/to/validation/data # 手动指定验证集路径
  # valid_path: /path/to/validation/data
  random_state: 2024                   # 随机种子
  streaming: false                     # 流式模式（大数据集）
                                       # false: 一次性加载到内存
                                       # true: 按块读取

feature_config: path/to/feature_config.yaml  # 特征配置文件路径
model_config: path/to/model_config.yaml      # 模型配置文件路径

dataloader:
  train_batch_size: 512                 # 训练批次大小
  train_shuffle: true                   # 每轮打乱训练数据
  valid_batch_size: 512                 # 验证批次大小
  valid_shuffle: false                  # 是否打乱验证数据
  num_workers: 4                        # 数据加载并行进程数
                                        # 0=单进程, >0=多进程
  # chunk_size: 20000                   # streaming=true 时的分块大小

train:
  optimizer: adam                       # 优化器：adam, sgd, adamw, rmsprop, adagrad
  optimizer_params:
    lr: 0.001                          # 学习率
    weight_decay: 0.0                  # L2 正则系数
  loss: bce                            # 损失函数
                                       # 单任务: bce, weighted_bce, focal_loss, mse
                                       # 多任务: [bce, weighted_bce, focal_loss]
  # loss_params:                       # 可选，多任务逐项配置
  # loss_weights:                      # 可选，损失权重或 GradNorm
  #   - pos_weight: 1.0
  #     logits: false
  metrics:                             # 评估指标
    - auc
    # - gauc
    # - recall
    # - precision
    # - accuracy
    # - f1
    # - logloss
    # - mse
    # - mae
    # - rmse
  epochs: 10                           # 训练轮数
  batch_size: 512                      # 可覆盖 dataloader.train_batch_size
  shuffle: true                        # 是否打乱数据
  device: cpu                          # 设备：cpu, cuda, cuda:0, mps
```

#### 参数说明

##### session 部分
- `id`: 会话标识符，训练产物将保存在 `{artifact_root}/{id}/` 目录下
- `artifact_root`: 产物根目录，默认为 `nextrec_logs`

##### data 部分
- `path`: 训练数据路径，支持：
  - 单个文件：`/path/to/data.csv` 或 `/path/to/data.parquet`
  - 目录：`/path/to/data_dir/`（自动读取目录下所有相同格式的文件）
- `format`: 数据格式，支持 `csv`, `parquet`, `feather`, 或 `auto`
- `target`: 目标列名，可以是字符串或列表
  - 单目标：`target: label`
  - 多目标：`target: [label_apply, label_credit]`
- `id_column`: 用户/样本ID列（可选，GAUC 需要）
- `user_id_column`: `id_column` 的别名
- `valid_ratio`: 验证集比例（0-1之间），仅当 `val_path` 未指定时生效
- `val_path`: 独立验证集路径（可选）
- `valid_path`: `val_path` 的别名
- `random_state`: 随机种子，确保数据划分可复现
- `streaming`: 是否使用流式处理
  - `true`: 适用于大数据集，按块加载数据
  - `false`: 一次性加载所有数据到内存

##### dataloader 部分
- `train_batch_size`: 训练时的批次大小
- `train_shuffle`: 是否打乱训练数据
- `valid_batch_size`: 验证时的批次大小
- `valid_shuffle`: 是否打乱验证数据
- `num_workers`: 数据加载进程数
- `chunk_size`: 流式处理时每次读取的数据量（`streaming=true`）

##### train 部分
- `optimizer`: 优化器类型
  - `adam`: Adam 优化器（推荐）
  - `sgd`: 随机梯度下降
  - `adamw`: AdamW 优化器
- `optimizer_params`: 优化器参数
  - `lr`: 学习率
  - `weight_decay`: 权重衰减（L2正则化）
- `loss`: 损失函数
  - `bce`: Binary Cross Entropy
  - `weighted_bce`: 带类别权重的 BCE
  - `focal_loss`: Focal Loss（不平衡数据）
  - `mse`: Mean Squared Error
- `loss_params`: 损失函数参数（可选，多任务逐项配置）
- `loss_weights`: 损失权重（列表/数值）或 GradNorm 配置
- `metrics`: 评估指标列表，支持：
  - `auc`: Area Under ROC Curve
  - `recall`: 召回率
  - `precision`: 精确率
  - `f1`: F1 分数
  - `gauc`: Group AUC
- `epochs`: 训练轮数
- `batch_size`: 可覆盖 dataloader 的批次大小
- `shuffle`: 是否打乱训练数据
- `device`: 运行设备
  - `cpu`: CPU
  - `cuda`: NVIDIA GPU
  - `cuda:0`: 指定 GPU 索引
  - `mps`: Apple Silicon GPU

---

### 预测配置文件

预测配置文件定义了模型推理流程。

#### 配置结构

```yaml
checkpoint_path: /path/to/checkpoint/directory  # 必填 checkpoint 目录
# model_config: /path/to/model_config.yaml       # 可选模型配置覆盖
# session:
#   id: my_experiment_session                    # 可选会话ID

predict:
  data_path: /path/to/prediction/data   # 预测数据路径
  source_data_format: parquet           # 输入数据格式：csv, parquet, feather
                                        # 使用 auto 自动识别
  id_column: user_id                    # ID列名（可选，用于关联预测结果）
  name: pred                            # 输出文件名（不含扩展名）
                                        # 最终路径: {checkpoint_path}/predictions/{name}.{save_data_format}
  save_data_format: csv                 # 输出格式：csv, parquet, feather
  preview_rows: 5                       # 预览行数（输出到日志）
  batch_size: 512                       # 预测批次大小
  num_workers: 4                        # 数据加载线程数
  device: cpu                           # 运行设备
  streaming: true                       # 是否启用流式加载
  chunk_size: 20000                     # 流式处理时的分块大小
```

#### 参数说明

- `checkpoint_path`: 训练好的模型 checkpoint 目录
  - 可以是目录（自动选择最新的 `.model` 文件）
  - 可以是具体的模型文件：`path/to/model.model`
- `model_config`: 模型配置覆盖（可选，未指定会在 checkpoint 目录自动查找）
- `predict.data_path`: 待预测的数据路径
- `predict.source_data_format`: 输入数据格式或 `auto`
- `predict.id_column`: ID列名（可选）
  - 如果指定，预测结果将包含此列
- `predict.name`: 输出文件名（不含扩展名）
- `predict.save_data_format`: 输出格式
  - `csv`: CSV 文件
  - `parquet`: Parquet 文件
  - `feather`: Feather 文件
- `predict.batch_size`: 预测时的批次大小
- `predict.num_workers`: 数据加载进程数
- `predict.streaming`: 是否启用流式加载
  - `true`: 流式处理，适用于大数据集
  - `false`: 一次性加载到内存（小数据集）
- `predict.chunk_size`: 流式处理时的分块大小
- `predict.preview_rows`: 预览行数
  - 预测完成后在日志中显示前 N 行结果

---

### 特征配置文件

特征配置文件定义了如何处理和转换输入特征。

#### 配置结构

```yaml
dense:
  age:
    processor_config:
      type: numeric                     # 数值特征
      scaler: standard                  # 标准化方法：standard, minmax, robust
    embedding_config:
      name: age                         # 特征名称
      input_dim: 1                      # 输入维度
      embedding_dim: 8                  # Embedding 维度
      use_embedding: true               # 是否使用 embedding
  
  income:
    processor_config:
      type: numeric
      scaler: minmax
    embedding_config:
      name: income
      input_dim: 1
      embedding_dim: 8
      use_embedding: false              # 不使用 embedding，直接使用原始值

sparse:
  city:
    processor_config:
      type: sparse
      encode_method: label              # 编码方法：label, hash, onehot
      # vocab_size: 1000                # 词表大小（label 编码时可选）
    embedding_config:
      name: city
      vocab_size: 1000                  # Embedding 词表大小
      embedding_dim: 16                 # Embedding 维度
  
  gender:
    processor_config:
      type: sparse
      encode_method: hash               # Hash 编码
      hash_size: 100                    # Hash 表大小
    embedding_config:
      name: gender
      vocab_size: 100
      embedding_dim: 8

sequence:
  click_history:
    processor_config:
      type: sequence
      encode_method: hash               # 序列编码方法
      hash_size: 10000                  # Hash 表大小
      max_len: 50                       # 最大序列长度
      pad_value: 0                      # 填充值
      truncate: post                    # 截断方式：post, pre
      separator: ","                    # 分隔符
    embedding_config:
      name: click_history
      vocab_size: 10000                 # Embedding 词表大小
      embedding_dim: 32                 # Embedding 维度
      padding_idx: 0                    # 填充索引
      combiner: mean                    # 聚合方式：mean, sum, attention

# 特征分组（可选，用于某些特定模型）
feature_groups:
  user_features:
    - age
    - gender
    - city
  item_features:
    - item_id
    - category
  context_features:
    - time
    - device
```

#### 参数说明

##### dense（数值特征）
- `processor_config.type`: 必须为 `numeric`
- `processor_config.scaler`: 标准化方法
  - `standard`: 标准化（均值为0，标准差为1）
  - `minmax`: 最小-最大归一化（缩放到0-1）
  - `robust`: 鲁棒标准化（使用中位数和四分位数）
- `embedding_config.use_embedding`: 是否对数值特征使用 embedding
  - `true`: 使用 embedding 层
  - `false`: 直接使用标准化后的值

##### sparse（离散特征）
- `processor_config.type`: 必须为 `sparse`
- `processor_config.encode_method`: 编码方法
  - `label`: Label 编码（推荐）
  - `hash`: Hash 编码（适用于高基数特征）
  - `onehot`: One-Hot 编码
- `processor_config.hash_size`: Hash 表大小（仅用于 hash 编码）
- `embedding_config.vocab_size`: Embedding 词表大小
- `embedding_config.embedding_dim`: Embedding 维度

##### sequence（序列特征）
- `processor_config.type`: 必须为 `sequence`
- `processor_config.encode_method`: 编码方法（通常为 `hash` 或 `label`）
- `processor_config.max_len`: 最大序列长度
- `processor_config.pad_value`: 填充值（通常为 0）
- `processor_config.truncate`: 截断方式
  - `post`: 从后面截断
  - `pre`: 从前面截断
- `processor_config.separator`: 序列分隔符
- `embedding_config.padding_idx`: 填充索引
- `embedding_config.combiner`: 序列聚合方式
  - `mean`: 平均池化
  - `sum`: 求和池化
  - `attention`: 注意力机制

##### feature_groups（特征分组）
用于某些需要特征分组的模型（如 MaskNet、PLE 等）。

---

### 模型配置文件

模型配置文件定义了模型架构和超参数。

#### 通用格式

```yaml
model: model_name                       # 模型名称
params:
  # 模型特定参数
  param1: value1
  param2: value2
```

#### 支持的模型

##### 排序模型（Ranking）

**DeepFM**
```yaml
model: deepfm
params:
  mlp_params:
    dims: [256, 128, 64]               # MLP 层维度
    activation: relu                   # 激活函数
    dropout: 0.3                       # Dropout 比率
  embedding_l2_reg: 1.0e-5             # Embedding L2 正则化
  dense_l2_reg: 1.0e-4                 # Dense 层 L2 正则化
```

**DIN (Deep Interest Network)**
```yaml
model: din
params:
  attention_mlp_dims: [80, 40]         # 注意力网络维度
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**DCN (Deep & Cross Network)**
```yaml
model: dcn
params:
  cross_num: 3                         # Cross 层数量
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**DCN-V2**
```yaml
model: dcn_v2
params:
  cross_num: 3                         # Cross 层数量
  cross_type: matrix                   # Cross 类型：vector, matrix, mix
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**MaskNet**
```yaml
model: masknet
params:
  architecture: parallel               # 模型类型：parallel, serial
  num_blocks: 3                        # Block 数量
  mask_hidden_dim: 64                  # Mask 隐藏层维度
  block_hidden_dim: 256                # Block 隐藏层维度
  block_dropout: 0.2                   # Dropout 比率
  embedding_l1_reg: 1.0e-6            # Embedding L1 正则化
  dense_l1_reg: 1.0e-5                # Dense L1 正则化
  embedding_l2_reg: 1.0e-5            # Embedding L2 正则化
  dense_l2_reg: 1.0e-4                # Dense L2 正则化
```

**AutoInt**
```yaml
model: autoint
params:
  attention_size: 16                   # 注意力维度
  num_heads: 2                         # 多头注意力头数
  num_layers: 3                        # Transformer 层数
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

##### 多任务模型（Multi-Task）

**MMOE (Multi-gate Mixture-of-Experts)**
```yaml
model: mmoe
params:
  num_experts: 8                       # 专家网络数量
  expert_dims: [256, 128]              # 专家网络维度
  gate_dims: [64]                      # 门控网络维度
  tower_dims: [64, 32]                 # Tower 网络维度
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**PLE (Progressive Layered Extraction)**
```yaml
model: ple
params:
  num_levels: 2                        # PLE 层数
  num_experts_specific: 4              # 任务特定专家数量
  num_experts_shared: 4                # 共享专家数量
  expert_dims: [256, 128]
  gate_dims: [64]
  tower_dims: [64, 32]
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

**ESMM (Entire Space Multi-Task Model)**
```yaml
model: esmm
params:
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

---

## 命令行参数

### 基本命令

```bash
nextrec --mode=MODE [OPTIONS]
```

### 参数说明

- `--mode`: 运行模式（必需）
  - `train`: 训练模式
  - `predict`: 预测模式

- `--train_config`: 训练配置文件路径（训练模式必需）
- `--predict_config`: 预测配置文件路径（预测模式必需）
- `--config`: 通用配置文件路径（已废弃，建议使用上述两个参数）

### 示例

```bash
# 训练模型
nextrec --mode=train --train_config=configs/deepfm_train.yaml

# 运行预测
nextrec --mode=predict --predict_config=configs/deepfm_predict.yaml
```

---

## 完整示例

### 示例 1：训练 DeepFM 模型

#### 1. 准备数据

假设你有一个电商数据集 `ecommerce_data.csv`：

```csv
user_id,item_id,age,gender,city,category,price,click_history,label
1,101,25,M,BJ,Electronics,999.0,"[98,99,100]",1
2,102,30,F,SH,Fashion,299.0,"[101,102,103]",0
...
```

#### 2. 创建特征配置 `feature_config.yaml`

```yaml
dense:
  age:
    processor_config:
      type: numeric
      scaler: standard
    embedding_config:
      name: age
      input_dim: 1
      embedding_dim: 8
      use_embedding: true
  
  price:
    processor_config:
      type: numeric
      scaler: minmax
    embedding_config:
      name: price
      input_dim: 1
      embedding_dim: 8
      use_embedding: true

sparse:
  user_id:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: user_id
      vocab_size: 100000
      embedding_dim: 32
  
  item_id:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: item_id
      vocab_size: 50000
      embedding_dim: 32
  
  gender:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: gender
      vocab_size: 10
      embedding_dim: 8
  
  city:
    processor_config:
      type: sparse
      encode_method: hash
      hash_size: 1000
    embedding_config:
      name: city
      vocab_size: 1000
      embedding_dim: 16
  
  category:
    processor_config:
      type: sparse
      encode_method: label
    embedding_config:
      name: category
      vocab_size: 100
      embedding_dim: 16

sequence:
  click_history:
    processor_config:
      type: sequence
      encode_method: label
      max_len: 50
      pad_value: 0
      truncate: post
      separator: ","
    embedding_config:
      name: click_history
      vocab_size: 50000
      embedding_dim: 32
      padding_idx: 0
      combiner: mean
```

#### 3. 创建模型配置 `model_config.yaml`

```yaml
model: deepfm
params:
  mlp_params:
    dims: [256, 128, 64]
    activation: relu
    dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

#### 4. 创建训练配置 `train_config.yaml`

```yaml
session:
  id: deepfm_ecommerce
  artifact_root: nextrec_logs

data:
  path: data/ecommerce_data.csv
  format: csv
  target: label
  valid_ratio: 0.2
  random_state: 2024
  streaming: false

feature_config: feature_config.yaml
model_config: model_config.yaml

dataloader:
  train_batch_size: 512
  train_shuffle: true
  valid_batch_size: 512
  valid_shuffle: false
  num_workers: 4

train:
  optimizer: adam
  optimizer_params:
    lr: 0.001
    weight_decay: 0.00001
  loss: focal
  loss_params:
    alpha: 0.25
    gamma: 2.0
  loss_weights:
    method: grad_norm
    alpha: 1.5
    lr: 0.025
    init_ema_steps: 50
    init_ema_decay: 0.9
  metrics:
    - auc
    - recall
    - precision
  epochs: 10
  batch_size: 512
  shuffle: true
  device: cuda
```

#### 5. 运行训练

```bash
nextrec --mode=train --train_config=train_config.yaml
```

#### 6. 创建预测配置 `predict_config.yaml`

```yaml
session:
  id: deepfm_ecommerce_predict
  artifact_root: nextrec_logs

checkpoint_path: nextrec_logs/deepfm_ecommerce

predict:
  data_path: data/test_data.csv
  source_data_format: csv
  id_column: user_id
  name: deepfm_predictions
  save_data_format: csv
  batch_size: 1024
  num_workers: 4
  device: cuda
  streaming: true
  chunk_size: 20000
  preview_rows: 10
```

#### 7. 运行预测

```bash
nextrec --mode=predict --predict_config=predict_config.yaml
```

---

### 示例 2：训练多任务模型 MMOE

#### 1. 数据格式

多任务学习需要多个目标列：

```csv
user_id,item_id,features...,label_click,label_purchase,label_favorite
1,101,...,1,0,1
2,102,...,1,1,0
...
```

#### 2. 训练配置

```yaml
session:
  id: mmoe_multitask
  artifact_root: nextrec_logs

data:
  path: data/multitask_data.csv
  format: csv
  target: [label_click, label_purchase, label_favorite]  # 多个目标
  valid_ratio: 0.2
  random_state: 2024
  streaming: false

feature_config: feature_config.yaml
model_config: mmoe_config.yaml

dataloader:
  train_batch_size: 512
  train_shuffle: true
  valid_batch_size: 512
  valid_shuffle: false
  num_workers: 4

train:
  optimizer: adam
  optimizer_params:
    lr: 0.001
    weight_decay: 0.00001
  loss: bce
  loss_weights: [0.3, 0.7]
  metrics:
    - auc
  epochs: 10
  batch_size: 512
  shuffle: true
  device: cuda
```

#### 3. MMOE 模型配置

```yaml
model: mmoe
params:
  num_experts: 8
  expert_dims: [256, 128]
  gate_dims: [64]
  tower_dims: [64, 32]
  dropout: 0.3
  embedding_l2_reg: 1.0e-5
  dense_l2_reg: 1.0e-4
```

---

## 高级功能

### 流式处理大数据集

当数据集无法一次性加载到内存时，可以使用流式处理：

```yaml
data:
  path: /path/to/large_dataset_dir/  # 目录，包含多个数据文件
  format: parquet
  target: label
  streaming: true                     # 启用流式处理
  valid_ratio: 0.2

dataloader:
  chunk_size: 50000                   # 每次读取 50000 行
  train_batch_size: 512
```

### 使用独立验证集

```yaml
data:
  path: /path/to/train_data.parquet
  val_path: /path/to/valid_data.parquet  # 指定独立验证集
  format: parquet
  target: label
```

### 自定义损失函数参数

```yaml
train:
  loss: focal
  loss_params:
    alpha: 0.25                        # 类别权重
    gamma: 2.0                         # 聚焦参数
    reduction: mean                    # 聚合方式
```

### 使用 Apple Silicon GPU (MPS)

在 macOS 上使用 Apple Silicon GPU 加速：

```yaml
train:
  device: mps
```

### 特征分组（用于特定模型）

某些模型（如 MaskNet）支持特征分组：

```yaml
# feature_config.yaml
feature_groups:
  user_group:
    - user_id
    - age
    - gender
  item_group:
    - item_id
    - category
    - price
  context_group:
    - time
    - device
```

### 多目标预测

```yaml
# predict_config.yaml
targets: [label_click, label_purchase]  # 覆盖训练配置的目标
```

---

## 常见问题

### Q1: 如何处理不平衡数据？

使用 Focal Loss：

```yaml
train:
  loss: focal
  loss_params:
    alpha: 0.25    # 增大正样本权重
    gamma: 2.0     # 增大难样本权重
```

### Q2: 训练产物保存在哪里？

训练产物保存在 `{artifact_root}/{session_id}/` 目录下：

```
nextrec_logs/
└── my_session/
    ├── processor.pkl              # 数据处理器
    ├── features_config.pkl        # 特征配置
    ├── model_epoch_1.pt        # 模型检查点
    ├── model_epoch_2.pt
    └── runs.log               # 训练日志
```

### Q3: 如何恢复训练？

目前 CLI 不支持断点续训，建议使用 Python API。

### Q4: 如何查看训练日志？

日志保存在 `{artifact_root}/{session_id}/runs.log`，也会输出到终端。

### Q5: 支持分布式训练吗？

当前 CLI 版本暂不支持分布式训练，请使用 Python API。

---

## 路径解析规则

配置文件中的路径支持以下格式：

1. **绝对路径**: `/absolute/path/to/file`
2. **相对路径**: `relative/path/to/file`（相对于配置文件所在目录）
3. **目录**: `/path/to/directory/`（自动扫描目录下的文件）

---

## 下一步

- 查看 [Python API 文档](https://nextrec.readthedocs.io/) 了解更多高级功能
- 浏览 [tutorials/](../tutorials/) 目录获取更多示例
- 访问 [GitHub Issues](https://github.com/zerolovesea/NextRec/issues) 反馈问题
