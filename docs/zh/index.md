---
layout: home

hero:
  name: NextRec
  text: 现代推荐系统框架
  tagline: 基于 PyTorch 的统一、高效、可扩展的推荐系统框架
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/getting-started
    - theme: alt
      text: 查看教程
      link: /zh/tutorial/

features:
  - title: 统一特征抽象
    description: 支持 Dense、Sparse、Sequence 三种特征类型，统一的数据处理与转换流程
    icon: 📊
  - title: 多任务学习
    description: 支持多任务学习，包括 ESMM、MMoE、PLE 等经典多任务架构
    icon: 🎯
  - title: ONNX 导出
    description: 支持模型导出为 ONNX 格式，跨框架推理部署更便捷
    icon: 🚀
  - title: CLI & Studio
    description: 提供命令行工具和可视化配置工具，降低使用门槛
    icon: ⚡
  - title: 丰富模型库
    description: 内置 DeepFM、DIN、DIEN、DCN、MVMoE 等主流推荐模型
    icon: 🧠
  - title: 高性能数据加载
    description: 支持大规模数据的流式加载与高效预处理
    icon: ⚡
---

## 安装

```bash
pip install nextrec
```

## 快速开始

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from nextrec.basic.features import DenseFeature, SparseFeature
from nextrec.models.ranking.deepfm import DeepFM

df = pd.read_csv("https://raw.githubusercontent.com/zerolovesea/NextRec/main/dataset/movielens_100k.csv")

dense_features = [DenseFeature("age")]
sparse_features = [
    SparseFeature("user_id", vocab_size=df["user_id"].max() + 1, embedding_dim=16),
    SparseFeature("item_id", vocab_size=df["item_id"].max() + 1, embedding_dim=16),
    SparseFeature("gender", vocab_size=df["gender"].max() + 1, embedding_dim=16),
    SparseFeature("occupation", vocab_size=df["occupation"].max() + 1, embedding_dim=16),
]

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2},
    target="label",
    device="cpu",
    session_id="movielens_deepfm",   # 管理实验日志与检查点
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="binary_crossentropy",
)

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=["auc", "recall", "precision"],
    epochs=2,
    batch_size=512,
    shuffle=True,
)
```

## 相关链接

- [GitHub 仓库](https://github.com/zerolovesea/NextRec)
- [安装指南](/zh/installatiton)
- [API 文档](/zh/apis/)
- [CLI 工具](/zh/cli/)
