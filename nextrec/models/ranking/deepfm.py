"""
Date: create on 27/10/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
[1] Guo H, Tang R, Ye Y, et al. DeepFM: A factorization-machine based neural network
for CTR prediction[J]. arXiv preprint arXiv:1703.04247, 2017.
(https://arxiv.org/abs/1703.04247)

DeepFM combines a Factorization Machine (FM) for explicit second-order feature
interactions with a deep MLP for high-order nonlinear patterns. Both parts share
the same embeddings, avoiding manual feature engineering and delivering strong CTR
performance with end-to-end training.

Workflow:
  (1) Shared embeddings encode sparse/sequence fields; dense features are concatenated
  (2) Wide (LR) term models first-order signals
  (3) FM term captures pairwise interactions via inner products
  (4) Deep MLP learns higher-order interactions over concatenated embeddings
  (5) Outputs from wide, FM, and deep parts are summed before the final prediction

Key Advantages:
- Joint explicit (FM) and implicit (MLP) interaction modeling
- Shared embeddings remove the need for manual cross features
- Simple to train end-to-end with minimal feature engineering
- Strong baseline for CTR/CVR style ranking tasks

DeepFM 将 FM 的显式二阶特征交互与 MLP 的高阶非线性交互结合，三部分共享
embedding，无需手工构造交叉特征即可端到端训练，常用于 CTR/CVR 预估。

流程：
  (1) 共享 embedding 处理稀疏/序列特征，稠密特征拼接
  (2) Wide（LR）建模一阶信号
  (3) FM 建模二阶交互
  (4) MLP 学习高阶非线性交互
  (5) Wide + FM + Deep 求和后进入预测

主要优点：
- 显式与隐式交互联合建模
- 共享 embedding，减少人工交叉特征
- 端到端训练简单，易于落地
- CTR/CVR 任务的常用强基线
"""

import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import FM, LR, MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class DeepFM(BaseModel):
    @property
    def model_name(self):
        return "DeepFM"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        mlp_params: dict | None = None,
        target: list[str] | str | None = None,
        task: str | list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(DeepFM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.loss = loss
        self.fm_features = sparse_features + sequence_features
        self.deep_features = dense_features + sparse_features + sequence_features
        self.embedding = EmbeddingLayer(features=self.deep_features)
        fm_emb_dim_total = sum([f.embedding_dim for f in self.fm_features])
        # deep_emb_dim_total = sum([f.embedding_dim for f in self.deep_features if not isinstance(f, DenseFeature)])
        # dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        mlp_input_dim = self.embedding.input_dim
        self.linear = LR(fm_emb_dim_total)
        self.fm = FM(reduce_sum=True)
        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.default_task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["linear", "mlp"]
        )
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)  # [B, 1]

        y = y_linear + y_fm + y_deep
        return self.prediction_layer(y)
