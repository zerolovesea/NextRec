<p align="center">
<img align="center" src="assets/logo.png" width="40%">
<p>

<div align="center">

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/nextrec?period=total&units=NONE&left_color=BLACK&right_color=GREEN&left_text=PyPI-downloads)](https://pypistats.org/packages/nextrec)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.4.20-orange.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zerolovesea/NextRec)

English | [中文文档](README.md)

**A Unified, Efficient, and Scalable Recommendation System Framework**

</div>

## Table of Contents

- [Introduction](#introduction)
- [NextRec Progress](#nextrec-progress)
- [Installation](#installation)
- [Architecture](#architecture)
- [Tutorials](#tutorials)
- [5-Minute Quick Start](#5-minute-quick-start)
- [CLI Usage](#cli-usage)
- [Platform Compatibility](#platform-compatibility)
- [Supported Models](#supported-models)
- [Contributing](#contributing)

## Introduction

NextRec is a modern recommendation framework built on PyTorch, delivering a unified experience for modeling, training, and evaluation. Designed with rich model implementations, data-processing utilities, engineering-ready training components, and easy-to-use CLI/tools/tutorials, NextRec helps research/engineering teams quickly build, train, reproduce, and deploy recommendation models.

## Why NextRec

- **Multi-scenario coverage**: Ranking (CTR/CVR), retrieval, multi-task learning, and more marketing/rec models, with a continuously expanding model zoo.
- **Unified feature engineering & data pipeline**: NextRec provide unified Dense/Sparse/Sequence feature definitions, DataProcessor, and batch-optimized RecDataLoader, matching offline feature training/inference in industrial big-data settings.
- **Developer-friendly experience**: `Stream processing/distributed training/inference` for `csv/parquet/pathlike` data, plus GPU/MPS acceleration and visualization support.
- **Flexible CLI tooling**: Start training and inference via command line and config files for rapid experiment iteration and agile deployment.
- **Efficient training & evaluation**: Standardized engine with optimizers, LR schedulers, early stopping, checkpoints, and detailed logging out of the box.

## NextRec Progress

- **21/12/2025** Added support for [GradNorm](/nextrec/loss/grad_norm.py) in v0.4.16, configurable via `loss_weight='grad_norm'` in the compile method
- **12/12/2025** Added [RQ-VAE](/nextrec/models/representation/rqvae.py), a common module for generative retrieval in v0.4.9. Paired [dataset](/dataset/ecommerce_task.csv) and [notebook code](tutorials/notebooks/en/Build%20semantic%20ID%20with%20RQ-VAE.ipynb) are available.
- **07/12/2025** Released the NextRec CLI tool to run training/inference from configs. See the [guide](/nextrec_cli_preset/NextRec-CLI.md) and [reference code](/nextrec_cli_preset).
- **03/12/2025** NextRec reached 100 ⭐—thanks for the support!
- **06/12/2025** Added single-machine multi-GPU DDP training in v0.4.1 with supporting [code](tutorials/distributed).
- **11/11/2025** NextRec v0.1.0 released with 10+ ranking models, 4 multi-task models, 4 retrieval models, and a unified training/logging/metrics system.

## Architecture

NextRec adopts a modular engineering design. The core components include: a unified feature-driven BaseModel architecture; independent Layer modules; a unified DataLoader supporting both training and inference; the NextCLI command-line tool, and more.

![NextRec Architecture](assets/nextrec_diagram.png)


## Installation

You can quickly install the latest NextRec via `pip install nextrec`; Python 3.10+ is required. If you want to run some tutorial codes, pull this project first: 

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec/
pip install nextrec # or pip install -e .
```

## Tutorials

See `tutorials/` for examples covering ranking, retrieval, multi-task learning, and data processing:

- [movielen_ranking_deepfm.py](/tutorials/movielen_ranking_deepfm.py) — DeepFM training on MovieLens 100k dataset
- [example_ranking_din.py](/tutorials/example_ranking_din.py) — DIN Deep Interest Network training on e-commerce dataset
- [example_multitask.py](/tutorials/example_multitask.py) — ESMM multi-task learning training on e-commerce dataset
- [movielen_match_dssm.py](/tutorials/movielen_match_dssm.py) — DSSM retrieval model training on MovieLens 100k dataset

- [example_distributed_training.py](/tutorials/distributed/example_distributed_training.py) — Single-machine multi-GPU training with NextRec

- [run_all_ranking_models.py](/tutorials/run_all_ranking_models.py) — Quickly validate availability of all ranking models
- [run_all_multitask_models.py](/tutorials/run_all_multitask_models.py) — Quickly validate availability of all multi-task models
- [run_all_match_models.py](/tutorials/run_all_match_models.py) — Quickly validate availability of all retrieval models

To dive deeper into NextRec framework details, Jupyter notebooks are available:

- [Hands on the NextRec framework](/tutorials/notebooks/en/Hands%20on%20nextrec.ipynb)
- [Using the data processor for preprocessing](/tutorials/notebooks/en/Hands%20on%20dataprocessor.ipynb)
- [Build semantic ID with RQ-VAE](/tutorials/notebooks/en/Build%20semantic%20ID%20with%20RQ-VAE.ipynb)

## 5-Minute Quick Start

We provide a detailed quick-start guide and paired datasets to help you get familiar with different features of NextRec framework. In `datasets/` you'll find an e-commerce scenario test dataset like this:

| user_id | item_id | dense_0     | dense_1     | dense_2     | dense_3    | dense_4     | dense_5     | dense_6     | dense_7     | sparse_0 | sparse_1 | sparse_2 | sparse_3 | sparse_4 | sparse_5 | sparse_6 | sparse_7 | sparse_8 | sparse_9 | sequence_0                                               | sequence_1                                                | label |
|--------|---------|-------------|-------------|-------------|------------|-------------|-------------|-------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------------------------------------------------------|-----------------------------------------------------------|-------|
| 1      | 7817    | 0.14704075  | 0.31020382  | 0.77780896  | 0.944897   | 0.62315375  | 0.57124174  | 0.77009535  | 0.3211029   | 315      | 260      | 379      | 146      | 168      | 161      | 138      | 88       | 5        | 312      | [170,175,97,338,105,353,272,546,175,545,463,128,0,0,0]   | [368,414,820,405,548,63,327,0,0,0,0,0,0,0,0]              | 0     |
| 1      | 3579    | 0.77811223  | 0.80359334  | 0.5185201   | 0.91091245 | 0.043562356 | 0.82142705  | 0.8803686   | 0.33748195 | 149      | 229      | 442      | 6        | 167      | 252      | 25       | 402      | 7        | 168      | [179,48,61,551,284,165,344,151,0,0,0,0,0,0,0]            | [814,0,0,0,0,0,0,0,0,0,0,0,0,0,0]                          | 1     |

Below is a short example showing how to train a DIN (Deep Interest Network) model. You can also run `python tutorials/example_ranking_din.py` directly to execute the training and inference code.

After training starts, you can find detailed training logs at `nextrec_logs/din_tutorial`.

```python
import pandas as pd

from nextrec.models.ranking.din import DIN
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

df = pd.read_csv('dataset/ranking_task.csv')

for col in df.columns and 'sequence' in col: # csv loads lists as text; convert them back to objects
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Define feature columns
dense_features = [DenseFeature(name=f'dense_{i}', input_dim=1) for i in range(8)]

sparse_features = [SparseFeature(name='user_id', embedding_name='user_emb', vocab_size=int(df['user_id'].max() + 1), embedding_dim=32), SparseFeature(name='item_id', embedding_name='item_emb', vocab_size=int(df['item_id'].max() + 1), embedding_dim=32),]

sparse_features.extend([SparseFeature(name=f'sparse_{i}', embedding_name=f'sparse_{i}_emb', vocab_size=int(df[f'sparse_{i}'].max() + 1), embedding_dim=32) for i in range(10)])

sequence_features = [
    SequenceFeature(name='sequence_0', vocab_size=int(df['sequence_0'].apply(lambda x: max(x)).max() + 1), embedding_dim=32, padding_idx=0, embedding_name='item_emb'),
    SequenceFeature(name='sequence_1', vocab_size=int(df['sequence_1'].apply(lambda x: max(x)).max() + 1), embedding_dim=16, padding_idx=0, embedding_name='sparse_0_emb'),]

mlp_params = {
    "dims": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.3,
}

model = DIN(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params=mlp_params,
    attention_hidden_units=[80, 40],
    attention_activation='sigmoid',
    attention_use_softmax=True,
    target=['label'],                                     # target variable
    device='mps',                                         
    embedding_l1_reg=1e-6,
    embedding_l2_reg=1e-5,
    dense_l1_reg=1e-5,
    dense_l2_reg=1e-4,
    session_id="din_tutorial",                            # experiment id for logs
)

# Compile model with optimizer and loss
model.compile(
            optimizer = "adam",
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5},
            loss = "focal",
            loss_params={"gamma": 2.0, "alpha": 0.25},
        )

model.fit(
    train_data=df,
    metrics=['auc', 'gauc', 'logloss'],  # metrics to track
    epochs=3,
    batch_size=512,
    shuffle=True,
    user_id_column='user_id'             # used for GAUC
)

# Evaluate after training
metrics = model.evaluate(
    df,
    metrics=['auc', 'gauc', 'logloss'],
    batch_size=512,
    user_id_column='user_id'
)
```

## CLI Usage

NextRec provides a powerful command-line interface for model training and prediction using YAML configuration files. For detailed CLI documentation, see:

- [NextRec CLI User Guide](/nextrec_cli_preset/NextRec-CLI.md) - Complete guide for using the CLI
- [NextRec CLI Configuration Examples](/nextrec_cli_preset/) - CLI configuration file examples

```bash
# Train a model
nextrec --mode=train --train_config=path/to/train_config.yaml

# Run prediction
nextrec --mode=predict --predict_config=path/to/predict_config.yaml
```

Prediction outputs are saved under `{checkpoint_path}/predictions/{name}.{save_data_format}`.

> As of version 0.4.20, NextRec CLI supports single-machine training; distributed training features are currently under development.

## Platform Compatibility

The current version is 0.4.20. All models and test code have been validated on the following platforms. If you encounter compatibility issues, please report them in the issue tracker with your system version:

| Platform | Configuration | 
|----------|---------------|
| MacOS latest | MacBook Pro M4 Pro 24GB RAM |
| Ubuntu latest | AutoDL 4070D Dual GPU |
| Ubuntu 24.04 | NVIDIA TITAN V x5 |
| CentOS 7 | Intel Xeon 5138Y 96 cores 377GB RAM |

---

## Supported Models

### Ranking Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [FM](nextrec/models/ranking/fm.py) | Factorization Machines | ICDM 2010 | Supported |
| [LR](nextrec/models/ranking/lr.py) | Logistic Regression | - | Supported |
| [AFM](nextrec/models/ranking/afm.py) | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | IJCAI 2017 | Supported |
| [FFM](nextrec/models/ranking/ffm.py) | Field-aware Factorization Machines | RecSys 2016 | Supported |
| [DeepFM](nextrec/models/ranking/deepfm.py) | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | IJCAI 2017 | Supported |
| [Wide&Deep](nextrec/models/ranking/widedeep.py) | Wide & Deep Learning for Recommender Systems | DLRS 2016 | Supported |
| [xDeepFM](nextrec/models/ranking/xdeepfm.py) | xDeepFM: Combining Explicit and Implicit Feature Interactions | KDD 2018 | Supported |
| [FiBiNET](nextrec/models/ranking/fibinet.py) | FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction | RecSys 2019 | Supported |
| [PNN](nextrec/models/ranking/pnn.py) | Product-based Neural Networks for User Response Prediction | ICDM 2016 | Supported |
| [AutoInt](nextrec/models/ranking/autoint.py) | AutoInt: Automatic Feature Interaction Learning | CIKM 2019 | Supported |
| [DCN](nextrec/models/ranking/dcn.py) | Deep & Cross Network for Ad Click Predictions | ADKDD 2017 | Supported |
| [DCN v2](nextrec/models/ranking/dcn_v2.py) | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems | KDD 2021 | In Progress |
| [DIN](nextrec/models/ranking/din.py) | Deep Interest Network for CTR Prediction | KDD 2018 | Supported |
| [DIEN](nextrec/models/ranking/dien.py) | Deep Interest Evolution Network | AAAI 2019 | Supported |
| [MaskNet](nextrec/models/ranking/masknet.py) | MaskNet: Feature-wise Gating Blocks for High-dimensional Sparse Recommendation Data | 2020 | Supported |
| [EulerNet](nextrec/models/ranking/eulernet.py) | EulerNet: Efficient and Effective Feature Interaction Modeling with Euler's Formula | SIGIR 2021 | Supported |

### Retrieval Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [DSSM](nextrec/models/retrieval/dssm.py) | Learning Deep Structured Semantic Models | CIKM 2013 | Supported |
| [DSSM v2](nextrec/models/retrieval/dssm_v2.py) | DSSM with pairwise BPR-style optimization | - | Supported |
| [YouTube DNN](nextrec/models/retrieval/youtube_dnn.py) | Deep Neural Networks for YouTube Recommendations | RecSys 2016 | Supported |
| [MIND](nextrec/models/retrieval/mind.py) | Multi-Interest Network with Dynamic Routing | CIKM 2019 | Supported |
| [SDM](nextrec/models/retrieval/sdm.py) | Sequential Deep Matching Model | - | Supported |

### Sequential Recommendation Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [SASRec](nextrec/models/sequential/sasrec.py) | Self-Attentive Sequential Recommendation | KDD 2018 | In Progress |
| [HSTU](nextrec/models/sequential/hstu.py) | Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations | arXiv 2024 | Supported |

### Multi-task Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [MMOE](nextrec/models/multi_task/mmoe.py) | Modeling Task Relationships in Multi-task Learning | KDD 2018 | Supported |
| [PLE](nextrec/models/multi_task/ple.py) | Progressive Layered Extraction | RecSys 2020 | Supported |
| [ESMM](nextrec/models/multi_task/esmm.py) | Entire Space Multi-task Model | SIGIR 2018 | Supported |
| [ShareBottom](nextrec/models/multi_task/share_bottom.py) | Multitask Learning | - | Supported |
| [POSO](nextrec/models/multi_task/poso.py) | POSO: Personalized Cold-start Modules for Large-scale Recommender Systems | 2021 | Supported |

### Generative Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [TIGER](nextrec/models/generative/tiger.py) | Recommender Systems with Generative Retrieval | NeurIPS 2023 | In Progress |

### Representation Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| [RQ-VAE](nextrec/models/representation/rqvae.py) | RQ-VAE: RQVAE for Generative Retrieval | - | Supported |
| [BPR](nextrec/models/representation/bpr.py) | Bayesian Personalized Ranking | UAI 2009 | In Progress |
| [MF](nextrec/models/representation/mf.py) | Matrix Factorization Techniques for Recommender Systems | - | In Progress |
| [AutoRec](nextrec/models/representation/autorec.py) | AutoRec: Autoencoders Meet Collaborative Filtering | WWW 2015 | In Progress |
| [LightGCN](nextrec/models/representation/lightgcn.py) | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | SIGIR 2020 | In Progress |
| [S3Rec](nextrec/models/representation/s3rec.py) | S3-Rec: Self-Supervised Learning for Sequential Recommendation | CIKM 2020 | In Progress |
| [CL4SRec](nextrec/models/representation/cl4srec.py) | CL4SRec: Contrastive Learning for Sequential Recommendation | 2021 | In Progress |

---

## Contributing

We welcome contributions of any form!

### How to Contribute

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)  
4. Push your branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

> Before submitting a PR, please run `python test/run_tests.py` and `python scripts/format_code.py` to ensure all tests pass and code style is consistent.

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

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

## Contact

- **GitHub Issues**: [Submit an issue](https://github.com/zerolovesea/NextRec/issues)  
- **Email**: zyaztec@gmail.com  

## Acknowledgements

NextRec is inspired by the following great open-source projects:

- [torch-rechub](https://github.com/datawhalechina/torch-rechub) — Flexible, easy-to-extend recommendation framework  
- [FuxiCTR](https://github.com/reczoo/FuxiCTR) — Configurable, tunable, and reproducible CTR library  
- [RecBole](https://github.com/RUCAIBox/RecBole) — Unified, comprehensive, and efficient recommendation library  

Special thanks to all open-source contributors!

## Citation

If you use this framework in your research or work, please consider citing:

```bibtex
@misc{nextrec,
    title = {NextRec},
    author = {Yang Zhou},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/zerolovesea/NextRec}},
    note = {A unified, efficient, and extensible PyTorch-based recommendation library}
}

```

---

<div align="center">

**[Back to Top](#nextrec)**

</div>
