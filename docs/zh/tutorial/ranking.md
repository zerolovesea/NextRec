---
title: 训练精排模型
description: 完整的 DeepFM 训练、评估、推理教程，包含数据预处理、模型训练、ONNX 导出
---

# 训练精排模型

本教程将带你完成一个完整的推荐系统精排模型开发流程：

1. 数据预处理与特征工程
2. DeepFM 模型训练
3. 模型评估
4. 在线推理
5. ONNX 导出部署

## 什么是 DeepFM？

DeepFM 是哈工大和华为合作共同提出的排序模型，结合了 FM（因子分解机）和 Deep（深度神经网络）的优点。

```
┌─────────────────────────────────────────┐
│              DeepFM 结构                  │
├─────────────────────────────────────────┤
│                                         │
│   Sparse Features                       │
│         │                               │
│         ▼                               │
│   ┌─────────────────┐                   │
│   │  Embedding Layer │ (共享)            │
│   └────────┬────────┘                   │
│            │                             │
│      ┌─────┴─────┐                      │
│      ▼           ▼                       │
│   ┌─────┐   ┌──────────┐                │
│   │ FM  │   │   DNN    │                │
│   │ 组件 │   │   组件   │                │
│   └──┬──┘   └────┬─────┘                │
│      │            │                      │
│      └─────┬──────┘                      │
│            ▼                              │
│      ┌─────────┐                          │
│      │ Sigmoid │ ───▶ CTR 预测            │
│      └─────────┘                          │
│                                         │
└─────────────────────────────────────────┘
```

## 环境准备

```python
import pandas as pd
import numpy as np
import torch

# 导入 NextRec 组件
from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.data.preprocessor import DataProcessor

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

## 1. 数据准备与预处理

### 1.1 加载数据

这里使用 MovieLens 100K 数据集：

```python
# 加载数据
data_path = "dataset/movielens_100k.csv"
df = pd.read_csv(data_path)

print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(df.head())
```

数据格式示例：

| user_id | movie_id | gender | age | occupation | genre | rating |
|---------|----------|--------|-----|------------|-------|--------|
| 1       | 119      | M      | 18  | 10         | Drama | 5      |

### 1.2 数据预处理

使用 `DataProcessor` 进行特征处理：

```python
# 创建数据处理器
processor = DataProcessor()

# 添加数值特征（年龄需要标准化）
processor.add_numeric_feature("age", scaler="standard")

# 添加稀疏特征（使用 Hash 编码处理高基数特征）
# Hash 编码适合高基数特征，词表大小固定
processor.add_sparse_feature("gender", encode_method="label")
processor.add_sparse_feature("occupation", encode_method="label")
processor.add_sparse_feature("movie_id", encode_method="hash", hash_size=100000)
processor.add_sparse_feature("genre", encode_method="hash", hash_size=10000)

# 添加目标列
processor.add_target("rating", target_type="regression")

# 拟合并转换
processor.fit(df)
df_processed = processor.transform(df)

print("处理后的数据形状:", df_processed.keys())
```

### 1.3 数据集划分

```python
from sklearn.model_selection import train_test_split

# 划分训练集和验证集
train_df, valid_df = train_test_split(df_processed, test_size=0.2, random_state=42)

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(valid_df)}")
```

## 2. 特征定义

### 2.1 定义 Dense 特征

```python
# Dense 特征：连续数值
dense_features = [
    DenseFeature(name="age"),
]
```

### 2.2 定义 Sparse 特征

```python
# Sparse 特征：离散 ID
sparse_features = [
    SparseFeature(
        name="gender",
        vocab_size=2,          # 性别只有 M/F 两种
        embedding_dim=4
    ),
    SparseFeature(
        name="occupation",
        vocab_size=21,         # 职业 ID 范围 0-20
        embedding_dim=8
    ),
    SparseFeature(
        name="movie_id",
        vocab_size=100000,    # 电影 ID 基数较大
        embedding_dim=32
    ),
    SparseFeature(
        name="genre",
        vocab_size=10000,     # 电影类型
        embedding_dim=16
    ),
]
```

## 3. 模型创建与配置

### 3.1 创建 DeepFM 模型

```python
# 创建 DeepFM 模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="rating",
    task="regression",         # 评分预测是回归任务
    device=device,
    session_id="deepfm_movielens"
)

# 打印模型结构
print(model)
```

### 3.2 配置训练参数

```python
# 配置优化器、损失函数
model.compile(
    optimizer="adam",
    optimizer_params={
        "lr": 0.001,
        "weight_decay": 1e-5
    },
    loss="mse"  # 回归任务使用 MSE 损失
)
```

## 4. 模型训练

### 4.1 训练模型

```python
# 训练模型
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    epochs=10,
    batch_size=256,
    metrics=["mse", "mae", "rmse"],
    early_stop_patience=5
)

print("训练完成!")
```

### 4.2 训练参数说明

| 参数 | 说明 |
|-----|------|
| `train_data` | 训练数据 |
| `valid_data` | 验证数据 |
| `epochs` | 训练轮数 |
| `batch_size` | 批次大小 |
| `metrics` | 评估指标 |
| `early_stop_patience` | 早停耐心值 |

## 5. 模型评估

### 5.1 在验证集上评估

```python
# 评估模型
metrics = model.evaluate(
    valid_df,
    metrics=["mse", "mae", "rmse"]
)

print("\n=== 验证集评估结果 ===")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

### 5.2 常见评估指标

| 指标 | 说明 | 理想值 |
|-----|------|-------|
| MSE | 均方误差 | 越低越好 |
| MAE | 平均绝对误差 | 越低越好 |
| RMSE | 均方根误差 | 越低越好 |
| R2 | 决定系数 | 越接近 1 越好 |

## 6. 模型推理

### 6.1 预测单个样本

```python
# 取一个样本进行预测
sample = valid_df.iloc[0:1]

# 预测
prediction = model.predict(sample)

print(f"预测评分: {prediction['rating'][0]:.2f}")
print(f"实际评分: {sample['rating'].values[0]}")
```

### 6.2 批量预测

```python
# 批量预测
predictions = model.predict(
    valid_df,
    batch_size=512,
    return_dataframe=True
)

print(f"预测结果形状: {predictions.shape}")
print(predictions.head(10))
```

### 6.3 预测并保存

```python
# 预测并保存到文件
predictions = model.predict(
    valid_df,
    save_path="./predictions.csv",
    save_format="csv"
)
```

## 7. ONNX 导出与部署

### 7.1 导出 ONNX 模型

```python
# 导出为 ONNX 格式
onnx_path = model.export_onnx(
    save_path="./deepfm_movielens.onnx",
    batch_size=1
)

print(f"ONNX 模型已保存到: {onnx_path}")
```

### 7.2 使用 ONNX 推理

```python
import nextrec.utils.onnx as onnx

# 使用 ONNX 模型推理
onnx_predictions = onnx.predict_onnx(
    onnx_path="./deepfm_movielens.onnx",
    data=valid_df,
    batch_size=512,
    return_dataframe=True
)

print(f"ONNX 预测结果:\n{onnx_predictions.head()}")
```

### 7.3 ONNX vs PyTorch 对比

```python
# 对比两种方式的预测结果
torch_pred = model.predict(valid_df.iloc[0:100])
onnx_pred = onnx.predict_onnx(onnx_path, valid_df.iloc[0:100])

# 计算差异
diff = np.abs(torch_pred['rating'] - onnx_pred['rating'].values)
print(f"最大差异: {diff.max():.6f}")
print(f"平均差异: {diff.mean():.6f}")
```

## 8. 完整代码

```python
"""
DeepFM 排序模型完整示例
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.data.preprocessor import DataProcessor

# ========== 1. 数据准备 ==========
df = pd.read_csv("dataset/movielens_100k.csv")

# ========== 2. 数据预处理 ==========
processor = DataProcessor()
processor.add_numeric_feature("age", scaler="standard")
processor.add_sparse_feature("gender", encode_method="label")
processor.add_sparse_feature("occupation", encode_method="label")
processor.add_target("rating", target_type="regression")

processor.fit(df)
df_processed = processor.transform(df, return_dict=False)

# 划分数据集
train_df, valid_df = train_test_split(df_processed, test_size=0.2, random_state=42)

# ========== 3. 定义特征 ==========
dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("gender", vocab_size=2, embedding_dim=16),
    SparseFeature("occupation", vocab_size=21, embedding_dim=16),
]

# ========== 4. 创建模型 ==========
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    target="rating",
    task="regression",
    device="cpu"
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 0.001},
    loss="mse"
)

# ========== 5. 训练 ==========
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    epochs=10,
    batch_size=256,
    metrics=["mse", "mae"],
    early_stop_patience=5
)

# ========== 6. 评估 ==========
metrics = model.evaluate(valid_df, metrics=["mse", "mae"])
print(f"验证集 MSE: {metrics['mse']:.4f}")
print(f"验证集 MAE: {metrics['mae']:.4f}")

# ========== 7. 推理 ==========
predictions = model.predict(valid_df, return_dataframe=True)
print(predictions.head())

# ========== 8. ONNX 导出 ==========
onnx_path = model.export_onnx(save_path="./deepfm.onnx")
print(f"ONNX 模型已保存: {onnx_path}")
```

## 下一步

- [训练多任务模型](./multitask.md)
- [训练召回模型](./retrieval.md)
