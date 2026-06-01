---
layout: home

hero:
  name: NextRec
  text: 现代推荐系统框架
  tagline: 基于 PyTorch 的统一、高效、可扩展的推荐系统框架
  image:
    src: /logo.svg
    alt: NextRec Logo
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/getting-started
    - theme: alt
      text: 查看教程
      link: /zh/tutorial/

features:
  - title: 统一特征抽象
    details: 支持 Dense、Sparse、Sequence 三类特征，统一编码、转换与输入接口
    icon: 📊
  - title: 模块化模型组件设计
    details: 将特征交互、骨干网络、任务塔等模块解耦，支持按需组合与快速迭代
    icon: 🧩
  - title: 统一训推验流程
    details: 统一训练、推理、评估接口与配置，减少环境切换和流程割裂
    icon: 🔄
  - title: 支持流式预处理
    details: 支持大规模样本的流式读取、在线转换与高效预处理，降低内存压力
    icon: 🌊
  - title: 流式与分布式训推
    details: 支持流式训练推理与分布式训练推理，兼顾吞吐、时效与扩展性
    icon: 🚀
  - title: 丰富任务与模型支持
    details: 覆盖排序、召回、多任务与生成式建模，满足多场景推荐需求
    icon: 🧠
  - title: 全面日志管理
    details: 支持 Weights & Biases、SwanLab、TensorBoard，统一记录指标、曲线与实验产物
    icon: 📈
  - title: 命令行配置驱动
    details: 通过配置文件一键完成训练与推理，降低工程接入与复现实验成本
    icon: ⚡
---

## 安装

```bash
pip install nextrec
```

如果需要 WandB 或 SwanLab 实验跟踪，请额外安装 `pip install "nextrec[tracking]"`。默认安装不会包含这两个可选依赖，以避免部分 Linux 环境安装 `wandb` 时因缺少 `go` 编译环境而失败。

如果需要 ONNX 导出或 ONNX Runtime 推理，请额外安装 `pip install "nextrec[onnx]"`。默认安装不会包含 ONNX 相关依赖，以避免部分 Linux 环境因 `onnxruntime` 不可用而安装失败。

如果需要在召回模型中使用向量检索索引，请按需额外安装 `pip install faiss-cpu`。

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
- [CLI 工具](/zh/cli/nextrec-cli)
