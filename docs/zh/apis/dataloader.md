---
title: 数据加载
description: 统一数据加载接口，支持 in-memory 与 streaming 大文件按块读取
---

# 数据加载

数据预处理完成后，需要构建一个数据加载器进行分批的数据加载和读取。在NextRec中对Pytorch的DataLoader进行了封装，提供了`RecDataLoader`作为统一的推荐系统数据加载接口，支持内存数据和大规模流式数据。

通过`from nextrec.data.dataloader import RecDataLoader`来进行调用。

## 核心能力

`RecDataLoader`提供了对多种数据格式的支持，包括DataFrame、Dict、文件路径（csv/parquet），并针对不同的场景，提供了内存加载与流式加载两种方式，以及基于分布式/多进程训练的自动数据分片。

## 示例代码

提供了两种加载模式的示例代码：

### 内存加载模式

适合中小规模数据，可直接加载到内存：

```python
from nextrec.data.dataloader import RecDataLoader

rec_loader = RecDataLoader(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    target=["label"],
    id_columns=['user_id'],
    processor=fitted_processor    # 支持加载已经拟合过的DataProcessor，在训练同时对数据做预处理变换
)

# 从 DataFrame 创建
loader = rec_loader.create_dataloader(
    data=df,
    batch_size=256,
    shuffle=True,
    streaming=False,
    num_workers=4

)

```

### 流式加载模式

适合大规模数据，按块流式读取：

```python
from nextrec.data.dataloader import RecDataLoader

rec_loader = RecDataLoader(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    target=["label"],
    id_columns=['user_id'],
    processor=fitted_processor    # 支持加载已经拟合过的DataProcessor，在训练同时对数据做预处理变换
)

loader = rec_loader.create_dataloader(
    data="/path/to/data_dir",       # 文件或目录
    streaming=True,                 # 开启流式
    chunk_size=20000,               # 加载数据时每个 chunk 的大小，也就是流式训练下的一个batch size
    num_workers=4
)
```

---

## 参数说明

`RecDataLoader`对外提供了两个方法：初始化方法`init`与构造方法`create_dataloader`。

### 初始化

通过实例化`RecDataLoader`传入参数，为需要的特征进行注册。

| 参数 | 说明 |
|-----|------|
| `dense_features` | 稠密连续特征列表 List[DenseFeature] |
| `sparse_features` | 稀疏特征列表 List[SparseFeature] |
| `sequence_features` | 序列特征列表 List[SequenceFeature] |
| `target` | 目标列名，如 `"click"` 或 `["click", "buy"]`（多任务） |
| `id_columns` | 数据中的id列 |
| `processor` | **[可选]** fit后的DataProcessor，有值时会在加载数据同时进行数据预处理 |

### 构造方法

通过使用`RecDataLoader.create_dataloader`方法，来返回标准的Pytorch DataLoader实例。

| 参数 | 说明 |
|-----|------|
| `batch_size` | 批次大小 |
| `streaming` | 是否开启流式模式 |
| `chunk_size` | 流式模式时每个 chunk 的行数 |
| `num_workers` | 数据加载线程数 |
| `prefetch_factor` | 预取因子 |
| `shuffle` | 是否 shuffle |
| `shard_rank` | **[通常无需主动设置]** 分布式训练分片 rank |
| `shard_count` | **[通常无需主动设置]** 分布式训练分片数 |
| `sampler` | **[通常无需主动设置]** 分布式sampler |

## NextRec CLI 集成

在命令行工具NextRec CLI中，通过修改对应的配置文件，能够调整数据加载器的参数。例如在`train.yaml`中配置`data`以及`dataloader`参数

```yaml
data:
  train_path: ./data/train
  streaming: true        # 对应 streaming=True

dataloader:
  batch_size: 1024
  shuffle: true
  chunk_size: 20000       
  num_workers: 6
  prefetch_factor: 2
```


## 完整示例

以下是一个`RecDataLoader`的简单完整使用示例代码：

```python
from nextrec.data.dataloader import RecDataLoader
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 1. 定义特征
dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=100000, embedding_dim=32),
    SparseFeature("item_id", vocab_size=200000, embedding_dim=32),
]
sequence_features = [
    SequenceFeature("hist_item_ids", vocab_size=200000, embedding_dim=32, max_len=50)
]

# 2. 创建 DataLoader
rec_loader = RecDataLoader(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    target=["label"]
)

# 3. 创建 loader（支持 DataFrame 或文件路径）
train_loader = rec_loader.create_dataloader(
    data=train_df,  # 或文件路径，例如 "/mnt/home/train_df"
    batch_size=256,
    shuffle=True,
    num_workers=4
)


```

---

## 下一步

- [基类模型的生命周期](./base-model.md) - 了解如何在基类模型中使用数据加载器训练和推理
- [CLI 工具](../cli/nextrec-cli.md) - 命令行工具 NextRec CLI
