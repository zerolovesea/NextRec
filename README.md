# NextRec

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.2.7-orange.svg)

English | [中文版](README_zh.md)

**A Unified, Efficient, and Scalable Recommendation System Framework**

</div>

## Introduction

NextRec is a modern recommendation system framework built on PyTorch, providing a unified modeling, training, and evaluation experience for researchers and engineering teams. The framework adopts a modular design with rich built-in model implementations, data-processing tools, and production-ready training components, enabling quick coverage of multiple recommendation scenarios.

> This project draws on several open-source recommendation libraries, with the general layers referencing the mature implementations in [torch-rechub](https://github.com/datawhalechina/torch-rechub)￼. These part of codes is still in its early stage and is being gradually replaced with our own implementations. If you find any bugs, please submit them in the issue section. Contributions are welcome.

### Key Features

- **Multi-scenario Recommendation**: Supports ranking (CTR/CVR), retrieval, multi-task learning, and generative recommendation models such as TIGER and HSTU — with more models continuously added.
- **Unified Feature Engineering & Data Pipeline**: Provides Dense/Sparse/Sequence feature definitions, persistent DataProcessor, and optimized RecDataLoader, forming a complete “Define → Process → Load” workflow.
- **Efficient Training & Evaluation**: A standardized training engine with optimizers, LR schedulers, early stopping, checkpoints, and logging — ready out-of-the-box.
- **Developer-friendly Engineering Experience**: Modular and extensible design, full tutorial support, GPU/MPS acceleration, and visualization tools.

---

## Installation

```bash
# release version
pip install nextrec

# pre-release version
pip install -i https://test.pypi.org/simple/ nextrec
```
---

## 5-Minute Quick Start

The following example demonstrates a full DeepFM training & inference pipeline using the MovieLens dataset:

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

### More Tutorials

The `tutorials/` directory provides examples for ranking, retrieval, multi-task learning, and data processing:

- `movielen_match_dssm.py` — DSSM retrieval on MovieLens 100k  
- `movielen_ranking_deepfm.py` — DeepFM ranking on MovieLens 100k  
- `example_ranking_din.py` — DIN (Deep Interest Network) example  
- `example_match_dssm.py` — DSSM retrieval example  
- `example_multitask.py` — Multi-task learning example  

---

## Data Processing Example

NextRec offers a unified interface for preprocessing sparse and sequence features:

```python
import pandas as pd
from nextrec.data.preprocessor import DataProcessor

df = pd.read_csv("dataset/movielens_100k.csv")

processor = DataProcessor()
processor.add_sparse_feature('movie_title', encode_method='hash', hash_size=1000)
processor.fit(df)

df = processor.transform(df, return_dict=False)

print("\nSample training data:")
print(df.head())
```

---

## Supported Models

### Ranking Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| **FM** | Factorization Machines | ICDM 2010 | Supported |
| **AFM** | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | IJCAI 2017 | Supported |
| **DeepFM** | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | IJCAI 2017 | Supported |
| **Wide&Deep** | Wide & Deep Learning for Recommender Systems | DLRS 2016 | Supported |
| **xDeepFM** | xDeepFM: Combining Explicit and Implicit Feature Interactions | KDD 2018 | Supported |
| **FiBiNET** | FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction | RecSys 2019 | Supported |
| **PNN** | Product-based Neural Networks for User Response Prediction | ICDM 2016 | Supported |
| **AutoInt** | AutoInt: Automatic Feature Interaction Learning | CIKM 2019 | Supported |
| **DCN** | Deep & Cross Network for Ad Click Predictions | ADKDD 2017 | Supported |
| **DIN** | Deep Interest Network for CTR Prediction | KDD 2018 | Supported |
| **DIEN** | Deep Interest Evolution Network | AAAI 2019 | Supported |
| **MaskNet** | MaskNet: Feature-wise Gating Blocks for High-dimensional Sparse Recommendation Data | 2020 | Supported |

### Retrieval Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| **DSSM** | Learning Deep Structured Semantic Models | CIKM 2013 | Supported |
| **DSSM v2** | DSSM with pairwise BPR-style optimization | - | Supported |
| **YouTube DNN** | Deep Neural Networks for YouTube Recommendations | RecSys 2016 | Supported |
| **MIND** | Multi-Interest Network with Dynamic Routing | CIKM 2019 | Supported |
| **SDM** | Sequential Deep Matching Model | - | Supported |

### Multi-task Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| **MMOE** | Modeling Task Relationships in Multi-task Learning | KDD 2018 | Supported |
| **PLE** | Progressive Layered Extraction | RecSys 2020 | Supported |
| **ESMM** | Entire Space Multi-task Model | SIGIR 2018 | Supported |
| **ShareBottom** | Multitask Learning | - | Supported |

### Generative Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| **TIGER** | Recommender Systems with Generative Retrieval | NeurIPS 2023 | In Progress |
| **HSTU** | Hierarchical Sequential Transduction Units | - | In Progress |

---

## Contributing

We welcome contributions of any form!

### How to Contribute

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)  
4. Push your branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

> Before submitting a PR, please run tests using `pytest test/ -v` or `python -m pytest` to ensure everything passes.

### Code Style

- Follow PEP8  
- Provide unit tests for new functionality  
- Update documentation accordingly  

### Reporting Issues

When submitting issues on GitHub, please include:

- Description of the problem  
- Reproduction steps  
- Expected behavior  
- Actual behavior  
- Environment info (Python version, PyTorch version, etc.)  

---

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

## Contact

- **GitHub Issues**: Submit issues on GitHub  
- **Email**: zyaztec@gmail.com  

---

## Acknowledgements

NextRec is inspired by the following great open-source projects:

- **torch-rechub** - A Lighting Pytorch Framework for Recommendation Models, Easy-to-use and Easy-to-extend.
- **FuxiCTR** — Configurable and reproducible CTR prediction library  
- **RecBole** — Unified and efficient recommendation library  

Special thanks to all open-source contributors!

---

<div align="center">

**[Back to Top](#nextrec)**

</div>
