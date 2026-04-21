<p align="center">
<img align="center" src="assets/logo.png" width="40%">
<p>

<div align="center">

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/nextrec?period=total&units=NONE&left_color=grey&right_color=GREEN&left_text=PyPI-downloads)](https://pypistats.org/packages/nextrec)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.6.13-orange.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Hands%20on%20nextrec.ipynb)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zerolovesea/NextRec)

English | [中文文档](README.md)

**A Unified, Efficient, and Scalable Recommendation System Framework**

</div>

## Table of Contents

- [Introduction](#introduction)
- [NextRec Progress](#nextrec-progress)
- [Installation](#installation)
- [Documentation and Visualization](#documentation-and-visualization)
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

- **Multi-scenario coverage**: Ranking (CTR/CVR), match, multi-task learning, and more marketing/rec models, with a continuously expanding model zoo.
- **Unified feature engineering & data pipeline**: NextRec provides unified Dense/Sparse/Sequence feature definitions, DataProcessor, and batch-optimized RecDataLoader, matching offline feature training/inference in industrial big-data settings.
- **Developer-friendly experience**: `Stream processing/distributed training/inference` for `csv/parquet/pathlike` data, plus GPU/MPS acceleration and visualization support.
- **Flexible CLI tooling**: Start training and inference via command line and config files for rapid experiment iteration and agile deployment.
- **Rich model annotations**: Detailed model background, implementation workflow, and tensor shape transformations to help learners quickly understand the architecture at first glance.
- **Efficient training & evaluation**: Standardized engine with optimizers, LR schedulers, early stopping, checkpoints, and detailed logging (WandB/SwanLab/TensorBoard) out of the box.

## NextRec Progress

- **28/01/2026** Added support for ONNX export and loading in v0.4.39, and significantly accelerated data preprocessing speed (up to 9x speedup)
- **01/01/2026** Happy New Year! In v0.4.27, added support for multiple multi-task models: [APG](/nextrec/models/multitask/apg.py), [ESCM](/nextrec/models/multitask/escm.py), [HMoE](/nextrec/models/multitask/hmoe.py), [Cross Stitch](/nextrec/models/multitask/cross_stitch.py)
- **21/12/2025** Added support for [GradNorm](/nextrec/loss/grad_norm.py) in v0.4.16, configurable via `loss_weight='grad_norm'` in the compile method
- **12/12/2025** Added [RQ-VAE](/nextrec/models/pretrain/rqvae.py), a common module for generative recommendation in v0.4.9. Paired [dataset](/dataset/ecommerce_task.csv) and [notebook code](tutorials/notebooks/en/Build%20semantic%20ID%20with%20RQ-VAE.ipynb) are available.
- **07/12/2025** Released the NextRec CLI tool to run training/inference from configs. See the [guide](https://zerolovesea.github.io/NextRec/zh/cli/nextrec-cli.html) and [config templates](/configs).
- **11/11/2025** NextRec v0.1.0 released with 10+ ranking models, 11 multi-task models, 4 match models, and a unified training/logging/metrics system.

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

> Note: Experiment tracking backends `wandb` and `swanlab` are not installed by default. Some Linux environments may fall back to building `wandb` from source, which can fail if the `go` toolchain is missing.
>
> If you only need training, evaluation, and inference, `pip install nextrec` is enough.
>
> If you need ONNX export or ONNX Runtime inference, install the optional extra:
>
> ```bash
> pip install "nextrec[onnx]"
> ```
>
> If you want WandB or SwanLab logging, install the optional extra:
>
> ```bash
> pip install "nextrec[tracking]"
> ```

## Documentation and Visualization

NextRec provides a locally deployable documentation service and visualization platform. After installation, you can start the local docs service with `nextrec docs` and launch the local visualization app with `nextrec studio`.

## Tutorials

See `tutorials/` for examples covering ranking, match, multi-task learning, and data processing:

- [movielen_ranking_deepfm.py](/tutorials/movielen_ranking_deepfm.py) — DeepFM training on MovieLens 100k dataset
- [example_ranking_din.py](/tutorials/example_ranking_din.py) — DIN Deep Interest Network training on e-commerce dataset
- [example_multitask.py](/tutorials/example_multitask.py) — ESMM multi-task learning training on e-commerce dataset
- [movielen_matching_dssm.py](/tutorials/movielen_matching_dssm.py) — DSSM retrieval model training on MovieLens 100k dataset
- [example_onnx.py](/tutorials/example_onnx.py) — Train and export models to ONNX format with NextRec

- [run_all_ranking_models.py](/tutorials/run_all_ranking_models.py) — Quickly validate availability of all ranking models
- [run_all_multitask_models.py](/tutorials/run_all_multitask_models.py) — Quickly validate availability of all multi-task models
- [run_all_matching_models.py](/tutorials/run_all_matching_models.py) — Quickly validate availability of all retrieval models
- [run_all_sequential_models.py](/tutorials/run_all_sequential_models.py) — Quickly validate availability of all sequential recommendation models

To dive deeper into NextRec framework details, Jupyter notebooks are available:

- [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Hands%20on%20nextrec.ipynb) [Hands on the NextRec framework](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Hands%20on%20nextrec.ipynb)
- [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Hands%20on%20dataprocessor.ipynb) [Using the data processor for preprocessing](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Hands%20on%20dataprocessor.ipynb)
- [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Build%20semantic%20ID%20with%20RQ-VAE.ipynb) [Build semantic ID with RQ-VAE](https://colab.research.google.com/github/zerolovesea/NextRec/blob/main/tutorials/notebooks/en/Build%20semantic%20ID%20with%20RQ-VAE.ipynb)

## 5-Minute Quick Start

We provide a detailed quick-start guide and paired datasets to help you get familiar with different features of NextRec framework. In `dataset/` you'll find an e-commerce scenario test dataset like this:

| user_id | item_id | dense_0     | dense_1     | dense_2     | dense_3    | dense_4     | dense_5     | dense_6     | dense_7     | sparse_0 | sparse_1 | sparse_2 | sparse_3 | sparse_4 | sparse_5 | sparse_6 | sparse_7 | sparse_8 | sparse_9 | sequence_0                                               | sequence_1                                                | label |
|--------|---------|-------------|-------------|-------------|------------|-------------|-------------|-------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------------------------------------------------------|-----------------------------------------------------------|-------|
| 1      | 7817    | 0.14704075  | 0.31020382  | 0.77780896  | 0.944897   | 0.62315375  | 0.57124174  | 0.77009535  | 0.3211029   | 315      | 260      | 379      | 146      | 168      | 161      | 138      | 88       | 5        | 312      | [170,175,97,338,105,353,272,546,175,545,463,128,0,0,0]   | [368,414,820,405,548,63,327,0,0,0,0,0,0,0,0]              | 0     |
| 1      | 3579    | 0.77811223  | 0.80359334  | 0.5185201   | 0.91091245 | 0.043562356 | 0.82142705  | 0.8803686   | 0.33748195 | 149      | 229      | 442      | 6        | 167      | 252      | 25       | 402      | 7        | 168      | [179,48,61,551,284,165,344,151,0,0,0,0,0,0,0]            | [814,0,0,0,0,0,0,0,0,0,0,0,0,0,0]                          | 1     |

Below is a short example showing how to train a DeepFM model. After training starts, you can find detailed training logs at `nextrec_logs/din_tutorial`.

```python
import pandas as pd
from nextrec.models.ranking.deepfm import DeepFM
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

df = pd.read_csv('https://raw.githubusercontent.com/zerolovesea/NextRec/main/dataset/ranking_task.csv')

dense_features = [DenseFeature(name=f'dense_{i}', input_dim=1) for i in range(8)]

sparse_features = [SparseFeature(name='user_id', embedding_name='user_emb', vocab_size=50000, embedding_dim=16), SparseFeature(name='item_id', embedding_name='item_emb', vocab_size=50000, embedding_dim=16),]

sparse_features.extend([SparseFeature(name=f'sparse_{i}', embedding_name=f'sparse_{i}_emb', vocab_size=50000, embedding_dim=16) for i in range(10)])

sequence_features = [
    SequenceFeature(name='sequence_0', vocab_size=50000, embedding_dim=16, padding_idx=0, embedding_name='item_emb'),
    SequenceFeature(name='sequence_1', vocab_size=50000, embedding_dim=16, padding_idx=0, embedding_name='sequence_1_emb'),]

model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params={"hidden_dims": [256, 128, 64], "activation": "relu", "dropout": 0.3},
    target=['label'], 
    session_id='deepfm_tutorial',
)

model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="bce"
)

model.fit(
    train_data=df,
    valid_split=0.2,
    metrics=['auc', 'gauc', 'logloss'],
    group_id='user_id',
    epochs=1,
    batch_size=512,
)
```

## CLI Usage

NextRec provides a powerful command-line interface for model training and prediction using YAML configuration files. For detailed CLI documentation, see:

- [NextRec CLI User Guide](https://zerolovesea.github.io/NextRec/zh/cli/nextrec-cli.html) - Complete guide for using the CLI
- [NextRec CLI Configuration Examples](/nextrec_cli_preset/) - CLI configuration file examples

```bash
# Start local docs site
nextrec docs

# Start local visualization studio
nextrec studio

# Train a model
nextrec --mode=train --train_config=path/to/train_config.yaml

# Run prediction
nextrec --mode=predict --predict_config=path/to/predict_config.yaml
```

Prediction outputs are saved under `{checkpoint_path}/predictions/{name}.{save_data_format}`.

## Platform Compatibility

The current version is 0.6.13. All models and test code have been validated on the following platforms. If you encounter compatibility issues, please report them in the issue tracker with your system version:

| Platform | Configuration | 
|----------|---------------|
| MacOS latest | MacBook Pro M4 Pro 24GB RAM |
| Ubuntu latest | AutoDL 4070D Dual GPU |
| Ubuntu 24.04 | NVIDIA TITAN V x5 |
| CentOS 7 | Intel Xeon 5138Y 96 cores 377GB RAM |

---

## Supported Models

### Ranking Models

| Model | Paper | Status |
| ------- | ------- | -------- |
| [FM](nextrec/models/ranking/fm.py) | Factorization machines | Supported |
| [LR](nextrec/models/ranking/lr.py) | Applied Logistic Regression | Supported |
| [AFM](nextrec/models/ranking/afm.py) | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | Supported |
| [FFM](nextrec/models/ranking/ffm.py) | Field-aware Factorization Machines for CTR Prediction | Supported |
| [DeepFM](nextrec/models/ranking/deepfm.py) | DeepFM: A factorization-machine based neural network for CTR prediction | Supported |
| [NFM](nextrec/models/ranking/nfm.py) | Neural factorization machines for sparse predictive analytics | Supported |
| [Wide&Deep](nextrec/models/ranking/widedeep.py) | Wide & Deep learning for recommender systems | Supported |
| [xDeepFM](nextrec/models/ranking/xdeepfm.py) | xdeepfm: Combining explicit and implicit feature interactions for recommender systems | Supported |
| [FiBiNET](nextrec/models/ranking/fibinet.py) | FiBiNET: Combining feature importance and bilinear feature interaction for click-through rate prediction | Supported |
| [PNN](nextrec/models/ranking/pnn.py) | Product-based neural networks for user response prediction | Supported |
| [AutoInt](nextrec/models/ranking/autoint.py) | AutoInt: Automatic feature interaction learning via self-attentive neural networks | Supported |
| [DCN](nextrec/models/ranking/dcn.py) | Deep & cross network for ad click predictions | Supported |
| [DCN v2](nextrec/models/ranking/dcn_v2.py) | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems | Supported |
| [DIN](nextrec/models/ranking/din.py) | Deep interest network for click-through rate prediction | Supported |
| [DIEN](nextrec/models/ranking/dien.py) | Deep interest evolution network for click-through rate prediction | Supported |
| [BST](nextrec/models/ranking/bst.py) | Behavior Sequence Transformer for E-commerce Recommendation in Alibaba | Supported |
| [DLRM](nextrec/models/ranking/dlrm.py) | Deep Learning Recommendation Model for Personalization and Recommendation Systems | Supported |
| [MaskNet](nextrec/models/ranking/masknet.py) | MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask | Supported |
| [EulerNet](nextrec/models/ranking/eulernet.py) | EulerNet: Efficient and Effective Feature Interaction Modeling with Euler's Formula | Supported |

### Match Models

| Model | Paper | Status |
| ------- | ------- | -------- |
| [DSSM](nextrec/models/matching/dssm.py) | Learning deep structured semantic models for web search using clickthrough data | Supported |
| [DSSM v2](nextrec/models/matching/dssm_v2.py) | DSSM v2 - DSSM with pairwise training using BPR loss | Supported |
| [YouTube DNN](nextrec/models/matching/youtube_dnn.py) | Deep neural networks for youtube recommendations | Supported |
| [MIND](nextrec/models/matching/mind.py) | Multi-interest network with dynamic routing for recommendation at Tmall | Supported |
| [SDM](nextrec/models/matching/sdm.py) | Sequential recommender system based on hierarchical attention networks | Supported |

### Sequential Recommendation Models

| Model | Paper | Status |
| ------- | ------- | -------- |
| [SASRec](nextrec/models/sequential/sasrec.py) | Self-Attentive Sequential Recommendation | Supported |
| [Tiger](nextrec/models/sequential/tiger.py) | Recommender Systems with Generative Recommendation | Supported |
| [HSTU](nextrec/models/sequential/hstu.py) | Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations | Supported |
| [GRU4Rec](nextrec/models/sequential/gru4rec.py) | Session-based Recommendations with Recurrent Neural Networks | Supported |

### Multi-task Models

| Model | Paper | Status |
| ------- | ------- | -------- |
| [MMOE](nextrec/models/multitask/mmoe.py) | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Supported |
| [PLE](nextrec/models/multitask/ple.py) | Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations | Supported |
| [ESMM](nextrec/models/multitask/esmm.py) | Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate | Supported |
| [ShareBottom](nextrec/models/multitask/share_bottom.py) | Multitask Learning | Supported |
| [POSO](nextrec/models/multitask/poso.py) | POSO: Personalized Cold Start Modules for Large-scale Recommender Systems | Supported |
| [PEPNet](nextrec/models/multitask/pepnet.py) | PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information | Supported |
| [APG](nextrec/models/multitask/apg.py) | APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction | Supported |
| [CrossStitch](nextrec/models/multitask/cross_stitch.py) | Cross-Stitch Networks for Multi-Task Learning | Supported |
| [HMOE](nextrec/models/multitask/hmoe.py) | Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space | Supported |
| [SNRTrans](nextrec/models/multitask/snr_trans.py) | SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning in E-Commerce by Exploiting Task Relationships in the Label Space | Supported |
| [AITM](nextrec/models/multitask/aitm.py) | Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising | Supported |

### Pretrain Models

| Model | Paper | Status |
| ------- | ------- | -------- |
| [RQ-VAE](nextrec/models/pretrain/rqvae.py) | Autoregressive Image Generation using Residual Quantization | Supported |
| [BPR](nextrec/models/pretrain/bpr.py) | Bayesian Personalized Ranking | In Progress |
| [MF](nextrec/models/pretrain/mf.py) | Matrix Factorization Techniques for Recommender Systems | In Progress |
| [AutoRec](nextrec/models/pretrain/autorec.py) | AutoRec: Autoencoders Meet Collaborative Filtering | In Progress |
| [LightGCN](nextrec/models/pretrain/lightgcn.py) | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | In Progress |
| [S3Rec](nextrec/models/pretrain/s3rec.py) | S3-Rec: Self-Supervised Learning for Sequential Recommendation | In Progress |
| [CL4SRec](nextrec/models/pretrain/cl4srec.py) | CL4SRec: Contrastive Learning for Sequential Recommendation | In Progress |

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
    year = {2026},
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
