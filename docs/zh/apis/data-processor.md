---
title: 数据预处理
description: 统一数据预处理与特征编码的完整参数说明
---

# 数据预处理

真实世界的数据脏乱无序，因此我们需要对这些数据做一系列处理，才能让数据真正能被模型学习。NextRec提供了统一高效的数据预处理工具进行操作：`DataProcessor` 定义在 `nextrec.data.preprocessor`，用于把原始表数据转换成模型可直接使用的数值化特征。

简单的示例代码如下：

```python
from nextrec.data.preprocessor import DataProcessor

# 创建处理器
processor = DataProcessor()

# 添加特征，将会生成{特征名}_{处理方式}的特征列
processor.add_numeric_feature("age", scaler="standard") # 将构造 age_standard 特征
processor.add_numeric_feature("age", scaler="log")      # 将构造 age_log 特征
processor.add_numeric_feature("income", scaler="minmax")
processor.add_sparse_feature("item_id", encode_method="hash", hash_size=1000000)
processor.add_sparse_feature("category", encode_method="label")
processor.add_sequence_feature("click_hist", encode_method="hash", hash_size=1000000, max_len=50)
processor.add_target("click")

# 拟合并转换
processor.fit(df)
train_processed = processor.transform(df)

# 保存处理器
processor.save("./processor.pkl")

# 推理时加载
processor = DataProcessor.load("./processor.pkl")
test_processed = processor.transform(test_df)
```

现在我们来看一下 `DataProcessor` 提供的方法和参数：

在初始化`DataProcessor`后，通过注册每个特征的处理参数，来进行记录和持久化。

## add_numeric_feature

添加稠密连续特征。我们已经看过了示例代码：

`processor.add_numeric_feature("age", scaler="standard")`

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称 |
| `scaler` | 缩放器类型：`"standard"`,`"minmax"`,`"robust"`,`"maxabs"`,`"log"`,`"none"`，将根据`scaler`类型构造生成特征名|
| `fill_na` | 缺失值的填充值 |

### scaler

目前支持的缩放器类型：

| 方法 | 说明 | 适用场景 |
|-----|------|---------|
| `standard` | Z-score 标准化 | 数据近似正态分布 |
| `minmax` | 归一化到 [0,1] | 有明确边界值 |
| `robust` | 中位数/四分位鲁棒缩放 | 存在异常值 |
| `maxabs` | 归一化到 [-1,1] | 稀疏数据 |
| `log` | 对数变换 | 偏斜分布 |
| `none` | 不做缩放 | 已有归一化数据 |

## add_sparse_feature

添加稀疏特征（离散特征）。示例代码：

`processor.add_sparse_feature("item_id", encode_method="hash", hash_size=1000000)`

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称 |
| `encode_method` | 编码方法：哈希编码：`"hash"` 或 标签编码`"label"`，将根据`encode_method`类型构造生成特征名|
| `hash_size` | 哈希编码的哈希桶大小 |
| `min_freq` | 最小频率，数据里低于此频率的值将被视为 `<UNK>` |
| `fill_na` | 缺失值的填充值 |
| `filter_value` | 可选；过滤值列表，命中这些值的样本会被过滤掉 |
| `keep_value` | 可选；保留值列表，只保留命中这些值的样本 |

## add_sequence_feature

添加序列特征。示例代码：

`processor.add_sequence_feature("click_hist", encode_method="hash", hash_size=1000000, max_len=50)`

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 特征名称 |
| `encode_method` | 编码方法：哈希编码：`"hash"` 或 标签编码`"label"`，将根据`encode_method`类型构造生成特征名 |
| `hash_size` | 哈希编码的哈希桶大小 |
| `min_freq` | 最小频率，数据里低于此频率的值将被视为 `<UNK>` |
| `max_len` | 最大序列长度 |
| `pad_value` | 序列的padding填充值 |
| `truncate` | 截断策略：`"pre"`：保留最后`max_len`个item；`"post"`：保留前面`max_len`个item |
| `separator` | 字符串序列的分隔符 |
| `filter_value` | 可选；过滤token列表，序列中包含任一token时该样本会被过滤掉 |
| `keep_value` | 可选；保留token列表，序列中至少包含一个token时该样本才会保留 |
| `match_mode` | 匹配模式：`"exact"`（精确匹配，默认）/ `"contains"`（子串匹配）/ `"regex"`（正则匹配） |

## sparse/sequence 样本过滤

在部分业务里，需要在预处理阶段做规则过滤，例如：

- 过滤掉异常用户ID（黑名单）
- 只保留包含指定行为token的序列样本

`DataProcessor` 支持在 `sparse` 和 `sequence` 特征上配置 `filter_value` 与 `keep_value`。

### 规则语义

- sparse 特征：按“列值是否命中列表”判断。
- sequence 特征：按“序列token是否包含列表元素”判断。
- 默认 `match_mode="exact"`；也支持 `contains` 与 `regex`。
- `keep_value` 与 `filter_value` 同时配置时，按“与（AND）”组合：
  样本必须先满足 `keep_value`，再不命中 `filter_value` 才会保留。
- 过滤规则在 `fit` 和 `transform` 阶段都会生效。

### Python API 示例

```python
from nextrec.data.preprocessor import DataProcessor

processor = DataProcessor()
processor.add_sparse_feature(
    "user_id",
    encode_method="label",
    keep_value=["u1", "u2"],         # 只保留 u1/u2
    filter_value=["u_bad"],          # 额外过滤黑名单
    match_mode="exact",              # exact / contains / regex
)
processor.add_sequence_feature(
    "hist_item_seq",
    encode_method="hash",
    hash_size=5000,
    separator=",",
    keep_value=["item_a", "item_b"], # 序列至少包含其一
    filter_value=["item_x"],         # 序列包含 item_x 则过滤
    match_mode="contains",
)
processor.add_target("label", target_type="binary")

processor.fit(df_train)
train_processed = processor.transform(df_train)
```

### NextRec CLI 集成

在命令行工具中，同样支持配置样本过滤。

```yaml
sparse:
  user_id:
    processor_config:
      - {encode_method: label, keep_value: [u1, u2], filter_value: [u_bad], match_mode: exact}
    embedding_config:
      - {embedding_dim: 8, padding_idx: 0}

sequence:
  hist_item_seq:
    processor_config:
      - {encode_method: hash, hash_size: 5000, separator: ",", keep_value: [item_a, item_b], filter_value: [item_x], match_mode: contains}
    embedding_config:
      - {vocab_size: 5000, max_len: 50, combiner: mean, embedding_dim: 8, padding_idx: 0}
```

## add_target

添加目标列，用于目标列特征也需要转换的场景。

| 参数 | 说明 |
|-----|------|
| `name` | **[必需]** 目标列名称 |
| `target_type` | 目标类型：`"binary"` 或 `"regression"` |
| `label_map` | 二进制目标的标签映射，例如 `{'click': 1, 'no_click': 0}` |


## fit

对输入的数据进行拟合。fit方法支持多种输入，包括pandas和polars类型的DataFrame，字典，或包含多个parquet/csv文件的路径。fit会将数据的scaler，max_len等关键参数注册在`DataProcessor`中。示例代码：

`processor.fit(df)`

| 参数 | 说明 |
|-----|------|
| `data` | 输入数据：pd.DataFrame/ pl.DataFrame / Dict / 文件路径 |

## transform

将已经拟合好的`DataProcessor`在原始数据上进行变换。示例代码：

`train_processed = processor.transform(df)`

| 参数 | 说明 |
|-----|------|
| `data` | 要转换的输入数据：pd.DataFrame/ pl.DataFrame / Dict / 文件路径 |
| `return_dict` | **[Bool]** 是否返回字典，选择False时返回numpy array，默认为True |
| `save_format` | **[可选]** 保存数据格式 `"csv"` 或 `"parquet"`，不输入时不会将保存数据进行落盘 |
| `output_path` | **[可选]** 保存转换后数据的输出路径 |

## save

将已经拟合好的`DataProcessor`在进行本地持久化，默认将保存为pkl文件。示例代码：

`processor.save("./processor.pkl")`

| 参数 | 说明 |
|-----|------|
| `save_path` | 保存路径，支持直接填写文件名，将会自动补全为`.pkl`文件 |

## load

加载本地持久化的`DataProcessor`。示例代码：

`processor = DataProcessor.load("./processor.pkl")`

| 参数 | 说明 |
|-----|------|
| `load_path` | 加载路径 |

## get_vocab_sizes

获取`DataProcessor`内记录的词表大小，将会返回每个特征与其词表大小的字典。示例代码：

`vocab_sizes = processor.get_vocab_sizes()`

## summary

获取`DataProcessor`中注册数据的整体汇总。示例代码：

`processor.summary()`


## 更高效的使用DataProcessor

面对大数据量级时，直接使用`transform`也许不是一个很好的选择，你将会面对：

- 数据量过大，一次性对变换会对内存造成极大压力
- 如果选择落盘到本地，对硬盘空间也有压力

这时你可以在数据加载器`RecDataLoader`里直接加载已经拟合过的`DataProcessor`，它会在模型前向训练/推理的准备数据阶段，流式预处理文件，这将大大减轻内存和硬盘压力。在下一章：[数据加载](./dataloader.md)可以看到相关的使用说明。

---

## 下一步

- [数据加载](./dataloader.md) - 使用数据加载器
- [快速开始](../getting-started.md) - 完整的数据处理流程
