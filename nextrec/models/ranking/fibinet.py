"""
Date: create on 09/11/2025
Checkpoint: edit on 09/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Huang T, Zhang Z, Zhang B, et al. FiBiNET: Combining feature importance and bilinear
feature interaction for click-through rate prediction[C]//RecSys. 2019: 169-177.
(https://arxiv.org/abs/1905.09433)

FiBiNET (Feature Importance and Bilinear Interaction Network) is a CTR model that
jointly learns which fields matter most and how they interact. It first uses SENET
to produce field-wise importance weights and recalibrate embeddings, then applies a
bilinear interaction layer to capture pairwise feature relationships with enhanced
expressiveness.

Pipeline:
  (1) Embed sparse and sequence features that share a common embedding dimension
  (2) SENET squeezes and excites across fields to generate importance scores
  (3) Reweight embeddings with SENET scores to highlight informative fields
  (4) Compute bilinear interactions on both the original and SENET-reweighted
      embeddings to model pairwise relations
  (5) Concatenate interaction outputs and feed them into an MLP alongside a linear
      term for final prediction

Key Advantages:
- SENET recalibration emphasizes the most informative feature fields
- Bilinear interactions explicitly model pairwise relationships beyond simple dot
  products
- Dual-path (standard + SENET-reweighted) interactions enrich representation power
- Combines linear and deep components for both memorization and generalization

FiBiNET 是一个 CTR 预估模型，通过 SENET 重新分配特征字段的重要性，再用双线性
交互层捕捉成对特征关系。模型先对稀疏/序列特征做 embedding，SENET 生成字段权重并
重标定 embedding，随后在原始和重标定的 embedding 上分别计算双线性交互，最后将
交互结果与线性部分一起输入 MLP 得到预测。
主要优点：
- SENET 让模型聚焦最重要的特征字段
- 双线性交互显式建模特征对关系，表达力强于简单点积
- 标准与重标定两路交互结合，丰富特征表示
- 线性与深层结构并行，兼顾记忆与泛化
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    LR,
    MLP,
    BiLinearInteractionLayer,
    EmbeddingLayer,
    HadamardInteractionLayer,
    SENETLayer,
)
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class FiBiNET(BaseModel):
    @property
    def model_name(self):
        return "FiBiNET"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        mlp_params: dict | None = None,
        interaction_combo: str = "11",  # "0": Hadamard, "1": Bilinear
        bilinear_type: str = "field_interaction",
        senet_reduction: int = 3,
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

        super(FiBiNET, self).__init__(
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
        self.linear_features = sparse_features + sequence_features
        self.interaction_features = sparse_features + sequence_features

        if len(self.interaction_features) < 2:
            raise ValueError(
                "FiBiNET requires at least two sparse/sequence features for interactions."
            )

        self.embedding = EmbeddingLayer(features=self.all_features)

        self.num_fields = len(self.interaction_features)
        self.embedding_dim = self.interaction_features[0].embedding_dim
        if any(
            f.embedding_dim != self.embedding_dim for f in self.interaction_features
        ):
            raise ValueError(
                "All interaction features must share the same embedding_dim in FiBiNET."
            )

        self.senet = SENETLayer(
            num_fields=self.num_fields, reduction_ratio=senet_reduction
        )

        self.interaction_combo = interaction_combo

        # E interaction layers: original embeddings
        if interaction_combo[0] == "0":  # Hadamard
            self.interaction_E = HadamardInteractionLayer(
                num_fields=self.num_fields
            )  # [B, num_pairs, D]
        elif interaction_combo[0] == "1":  # Bilinear
            self.interaction_E = BiLinearInteractionLayer(
                input_dim=self.embedding_dim,
                num_fields=self.num_fields,
                bilinear_type=bilinear_type,
            )  # [B, num_pairs, D]
        else:
            raise ValueError("interaction_combo must be '01' or '11'")

        # V interaction layers: SENET reweighted embeddings
        if interaction_combo[1] == "0":
            self.interaction_V = HadamardInteractionLayer(num_fields=self.num_fields)
        elif interaction_combo[1] == "1":
            self.interaction_V = BiLinearInteractionLayer(
                input_dim=self.embedding_dim,
                num_fields=self.num_fields,
                bilinear_type=bilinear_type,
            )
        else:
            raise ValueError("Deep-FiBiNET SENET side must be '01' or '11'")

        linear_dim = sum([f.embedding_dim for f in self.linear_features])
        self.linear = LR(linear_dim)

        num_pairs = self.num_fields * (self.num_fields - 1) // 2
        interaction_dim = num_pairs * self.embedding_dim * 2
        self.mlp = MLP(input_dim=interaction_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.default_task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "linear",
                "senet",
                "mlp",
                "interaction_E",
                "interaction_V",
            ],
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        input_linear = self.embedding(
            x=x, features=self.linear_features, squeeze_dim=True
        )
        y_linear = self.linear(input_linear)

        field_emb = self.embedding(
            x=x, features=self.interaction_features, squeeze_dim=False
        )
        senet_emb = self.senet(field_emb)

        out_E = self.interaction_E(field_emb)  # [B, num_pairs, D]

        out_V = self.interaction_V(senet_emb)  # [B, num_pairs, D]

        deep_input = torch.cat(
            [out_E.flatten(start_dim=1), out_V.flatten(start_dim=1)], dim=1
        )

        y_deep = self.mlp(deep_input)

        y = y_linear + y_deep
        return self.prediction_layer(y)
