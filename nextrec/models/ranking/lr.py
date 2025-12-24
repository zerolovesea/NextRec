"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Hosmer D W, Lemeshow S, Sturdivant R X. Applied Logistic Regression.

Logistic Regression (LR) is a classic linear baseline for CTR/ranking tasks.
It maps each feature (dense, sparse, or sequence) into a numeric vector and
learns a single linear logit. Despite its simplicity, LR is strong for
high-dimensional sparse data and is commonly used as a baseline or a "wide"
component in hybrid models.

Pipeline:
  (1) Embed sparse/sequence fields; project dense fields if configured
  (2) Concatenate all feature vectors into a single linear input
  (3) Apply a linear layer to produce logits
  (4) Use the prediction layer to output task-specific probabilities

Key Advantages:
- Fast and easy to train
- Strong baseline for sparse, high-dimensional features
- Interpretable linear weights

LR 是 CTR/排序任务中最经典的线性基线模型。它将稠密、稀疏以及序列特征
映射为数值向量后做线性组合，输出 logit。虽然结构简单，但在稀疏高维场景
依然具有很强的基线效果，并常作为 Wide 端与深模型组合。

处理流程：
  (1) 稀疏/序列特征做 embedding，稠密特征按需投影
  (2) 拼接所有特征向量形成线性输入
  (3) 线性层输出 logit
  (4) 通过预测层输出任务概率

主要优点：
- 训练与推理速度快
- 稀疏高维特征下表现稳定
- 权重可解释性强
"""

import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, LR as LinearLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class LR(BaseModel):
    @property
    def model_name(self):
        return "LR"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
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
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(LR, self).__init__(
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

        self.embedding = EmbeddingLayer(features=self.all_features)
        linear_input_dim = self.embedding.input_dim
        self.linear = LinearLayer(linear_input_dim)
        self.prediction_layer = TaskHead(task_type=self.task)

        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["linear"]
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_linear = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        y = self.linear(input_linear)
        return self.prediction_layer(y)
