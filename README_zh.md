# NextRec

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.2.7-orange.svg)

[English Version](README.md) | 中文版

**统一、高效、可扩展的推荐系统框架**

</div>


## 简介

NextRec 是一个基于 PyTorch 构建的现代推荐系统框架，为研究人员与工程团队提供统一的建模、训练与评估体验。框架采用模块化设计，内置丰富的模型实现、数据处理工具和工程化训练组件，可快速覆盖多种推荐场景。

### 核心特性

- **多场景推荐能力**：同时覆盖排序（CTR/CVR）、召回、多任务学习以及 TIGER、HSTU 等生成式推荐模型，持续扩充模型库。
- **统一的特征工程与数据流水线**：提供 Dense/Sparse/Sequence 特征定义、可持久化的 DataProcessor、批处理优化的 RecDataLoader，贯穿“定义-处理-加载”全链路。
- **高效训练与评估**：标准化训练引擎内置多种优化器、学习率调度、早停、模型检查点与日志，开箱即用。
- **友好的工程体验**：模块化设计便于扩展，配套教程齐全，并支持 GPU/MPS 加速与可视化监控。

> 项目借鉴了一些优秀的开源推荐算法库，其中通用layer层借鉴了目前更成熟的[torch-rechub](https://github.com/datawhalechina/torch-rechub)的已有实现，欢迎有需要的开发者前往该仓库使用。目前该项目初期代码尚不成熟，正逐步更新为自有实现，如有Bug请移步issue区，也欢迎开发者共同贡献。

---

## 安装

```bash
# 正式版
pip install nextrec

# 测试版
pip install -i https://test.pypi.org/simple/ nextrec
```

## 5分钟快速上手

以下示例展示了使用 DeepFM 进行 movielens数据的训练推理的完整流程：

```python
import pandas as pd

from nextrec.models.ranking.deepfm import DeepFM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

df = pd.read_csv("dataset/movielens_100k.csv")

target = 'label'
dense_features = [DenseFeature('age')]
sparse_features = [
    SparseFeature('user_id', vocab_size=df['user_id'].max()+1, embedding_dim=4),
    SparseFeature('item_id', vocab_size=df['item_id'].max()+1, embedding_dim=4),
]

sparse_features.append(SparseFeature('gender', vocab_size=df['gender'].max()+1, embedding_dim=4))
sparse_features.append(SparseFeature('occupation', vocab_size=df['occupation'].max()+1, embedding_dim=4))

model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.5},
    target=target,
    device='cpu',
    session_id="deepfm_with_processor",
    embedding_l1_reg=1e-6,
    dense_l1_reg=1e-5,
    embedding_l2_reg=1e-5,
    dense_l2_reg=1e-4,
)

model.compile(optimizer="adam", optimizer_params={"lr": 1e-3, "weight_decay": 1e-5}, loss="bce")
model.fit(train_data=df, metrics=['auc', 'recall', 'precision'], epochs=10, batch_size=512, shuffle=True, verbose=1)
preds = model.predict(df)
print(f'preds: {preds}')
```

### 更多示例

`tutorials/` 目录提供了覆盖排序、召回、多任务、数据处理等场景的示例：

- `movielen_match_dssm.py` - movielen 100k数据集上的 DSSM 召回模型示例
- `movielen_ranking_deepfm.py` - movielen 100k数据集上的 DeepFM 模型示例
- `example_ranking_din.py` - 模拟数据上的DIN 深度兴趣网络示例
- `example_match_dssm.py` - 模拟数据上的DSSM 召回模型示例
- `example_multitask.py` - 多任务学习示例

### 数据预处理示例

NextRec 提供统一的数据预处理接口，支持对稀疏特征，序列特征进行数据预处理：

```python
import pandas as pd
from nextrec.data.preprocessor import DataProcessor

df = pd.read_csv("dataset/movielens_100k.csv")

processor = DataProcessor()
processor.add_sparse_feature('movie_title', encode_method='hash', hash_size=1000)
processor.fit(df)

df = processor.transform(df, return_dict=False) # return_dict为false时，默认返回dataframe

print(f"\nSample training data:")
print(df.head())
```
---

## 支持的模型

### 排序模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **FM** | Factorization Machines | ICDM 2010 | 已支持 |
| **AFM** | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | IJCAI 2017 | 已支持 |
| **DeepFM** | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | IJCAI 2017 | 已支持 |
| **Wide&Deep** | Wide & Deep Learning for Recommender Systems | DLRS 2016 | 已支持 |
| **xDeepFM** | xDeepFM: Combining Explicit and Implicit Feature Interactions | KDD 2018 | 已支持 |
| **FiBiNET** | FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction | RecSys 2019 | 已支持 |
| **PNN** | Product-based Neural Networks for User Response Prediction | ICDM 2016 | 已支持 |
| **AutoInt** | AutoInt: Automatic Feature Interaction Learning | CIKM 2019 | 已支持 |
| **DCN** | Deep & Cross Network for Ad Click Predictions | ADKDD 2017 | 已支持 |
| **DIN** | Deep Interest Network for Click-Through Rate Prediction | KDD 2018 | 已支持 |
| **DIEN** | Deep Interest Evolution Network for Click-Through Rate Prediction | AAAI 2019 | 已支持 |
| **MaskNet** | MaskNet: Introducing Feature-wise Gating Blocks for High-dimensional Sparse Recommendation Data | 2020 | 已支持 |

### 召回模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **DSSM** | Learning Deep Structured Semantic Models | CIKM 2013 | 已支持 |
| **DSSM v2** | DSSM with pairwise BPR-style optimization | - | 已支持 |
| **YouTube DNN** | Deep Neural Networks for YouTube Recommendations | RecSys 2016 | 已支持 |
| **MIND** | Multi-Interest Network with Dynamic Routing | CIKM 2019 | 已支持 |
| **SDM** | Sequential Deep Matching Model | - | 已支持 |

### 多任务模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **MMOE** | Modeling Task Relationships in Multi-task Learning | KDD 2018 | 已支持 |
| **PLE** | Progressive Layered Extraction | RecSys 2020 | 已支持 |
| **ESMM** | Entire Space Multi-Task Model | SIGIR 2018 | 已支持 |
| **ShareBottom** | Multitask Learning | - | 已支持 |

### 生成式模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **TIGER** | Recommender Systems with Generative Retrieval | NeurIPS 2023 | 开发中 |
| **HSTU** | Hierarchical Sequential Transduction Units | - | 开发中 |

---

## 贡献指南

我们欢迎任何形式的贡献！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

> 在提交 PR 之前，请运行 `pytest test/ -v` 或 `python -m pytest` 确保所有测试通过。

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 为新增功能补充单元测试
- 同步更新相关文档

### 报告错误

在 [Issues](https://github.com/zerolovesea/NextRec/issues) 页面提交问题时，请包含：

- 错误描述
- 重现步骤
- 期望行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本等）

---

## 许可证

本项目采用 [Apache 2.0 许可证](./LICENSE)。

---

## 联系方式

- **GitHub Issues**: [提交问题](https://github.com/zerolovesea/NextRec/issues)
- **邮箱**: zyaztec@gmail.com

---

## 致谢

NextRec 的开发受到以下优秀项目的启发：

- [torch-rechub](https://github.com/datawhalechina/torch-rechub) - 灵活且易于扩展的推荐系统框架
- [FuxiCTR](https://github.com/reczoo/FuxiCTR) - 可配置、可调优、可复现的 CTR 预测库
- [RecBole](https://github.com/RUCAIBox/RecBole) - 统一、全面、高效的推荐库

感谢开源社区的所有贡献者！

---

<div align="center">

**[返回顶部](#nextrec)**

</div>
