"""
Date: create on 04/21/2026
Checkpoint: edit on 04/21/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Naumov M, Mudigere D, Shi H J M, et al. Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091, 2019.
URL: https://arxiv.org/abs/1906.00091

DLRM combines dense numerical features and sparse categorical features through
a bottom MLP plus explicit pairwise dot-product interactions. In this
implementation, the final logit is produced by:
    (1) Dense branch: all dense features are concatenated and projected by a shared
            bottom MLP into one dense embedding
    (2) Sparse branch: sparse and sequence features are embedded into field vectors
    (3) Interaction branch: dense embedding and sparse field embeddings are combined
            by pairwise dot-product interactions
    (4) Top branch: interaction features and optional dense embedding are fused by a top MLP

This implementation adds support for sequence features, allowing them to be pooled and treated as additional fields in the interaction branch.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Dense branch: concat(all dense features) -> dense_input: [Batch, Dense_input_dim] -> bottom_mlp -> dense_embedding: [Batch, Dim_embedding]
- Sparse branch: sparse/sequence features -> embedding layer -> sparse_embeddings: [Batch, Field_num, Dim_embedding]
- Interaction input: concat(dense_embedding.unsqueeze(1), sparse_embeddings) -> [Batch, Field_num + 1, Dim_embedding]
- Interaction matrix: bmm(interaction_inputs, interaction_inputs^T) -> [Batch, Field_num + 1, Field_num + 1]
- Pairwise interactions: lower-triangular extraction -> [Batch, Interaction_dim]
- Top input: pairwise interactions (+ dense_embedding) -> [Batch, Interaction_dim (+ Dim_embedding)]
- Output: top_mlp -> [Batch, 1] -> prediction layer

DLRM 通过底部 MLP 和显式点积交互联合建模 dense 数值特征与 sparse 类别特征。
当前实现的最终 logit 由四部分逻辑组成：
    (1) Dense 分支：所有 dense 特征先拼接，再通过共享 bottom MLP 映射为一个 dense embedding
    (2) Sparse 分支：sparse 和 sequence 特征编码为 field 向量
    (3) 交互分支：dense embedding 与 sparse field embedding 做两两点积交互
    (4) Top 分支：将交互特征和可选 dense embedding 拼接后送入 top MLP

本实现加入了对 sequence 特征的支持。sequence 特征会先经过聚合操作（如平均或池化），然后作为额外的 field 与 sparse 特征一起参与交互计算。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Dense 分支：concat(所有 dense 特征) -> dense_input: [Batch, Dense_input_dim] -> bottom_mlp -> dense_embedding: [Batch, Dim_embedding]
- Sparse 分支：sparse/sequence 特征 -> embedding layer -> sparse_embeddings: [Batch, Field_num, Dim_embedding]
- 交互输入：concat(dense_embedding.unsqueeze(1), sparse_embeddings) -> [Batch, Field_num + 1, Dim_embedding]
- 交互矩阵：bmm(interaction_inputs, interaction_inputs^T) -> [Batch, Field_num + 1, Field_num + 1]
- 两两交互：提取严格下三角 -> [Batch, Interaction_dim]
- Top 输入：pairwise interactions (+ dense_embedding) -> [Batch, Interaction_dim (+ Dim_embedding)]
- 输出：top_mlp -> [Batch, 1] -> 预测层
"""

from __future__ import annotations

import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class DLRM(BaseRankingModel):
    @property
    def model_name(self) -> str:
        return "DLRM"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        bottom_mlp_params: dict | None = None,
        top_mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize DLRM model.
        初始化 DLRM 模型。

        Args:
            bottom_mlp_params: Parameters for the dense bottom MLP. e.g. {"hidden_dims": [128, 64], "dropout": 0.0}.
                The bottom MLP consumes the concatenated dense input and projects it to the sparse embedding dimension.
                Dense 底部 MLP 的参数，例如 {"hidden_dims": [128, 64], "dropout": 0.0}。
                该 MLP 接收拼接后的 dense 输入，并将其映射到与 sparse embedding 相同的维度。
            top_mlp_params: Parameters for the top MLP. e.g. {"hidden_dims": [256, 128], "dropout": 0.2}.
                The top MLP consumes pairwise interaction features and the optional dense embedding.
                Top MLP 的参数，例如 {"hidden_dims": [256, 128], "dropout": 0.2}。
                该 MLP 接收两两交互特征以及可选的 dense embedding。

        Note:
            This implementation requires all sparse/sequence features to share the same embedding dimension.
            Sequence features are first pooled by EmbeddingLayer and then treated as additional interaction fields.
            当前实现要求所有 sparse/sequence 特征共享相同的 embedding 维度。
            Sequence 特征会先由 EmbeddingLayer 聚合，再作为额外 field 参与交互。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        bottom_mlp_params = dict(bottom_mlp_params or {})
        top_mlp_params = dict(top_mlp_params or {})

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.sparse_features_all = sparse_features + sequence_features
        if not self.sparse_features_all:
            raise ValueError("[DLRM Error] DLRM requires at least one sparse or sequence feature.")

        sparse_dims = {feature.embedding_dim for feature in self.sparse_features_all}
        if len(sparse_dims) != 1:
            details = ", ".join(f"{feature.name}={feature.embedding_dim}" for feature in self.sparse_features_all)
            raise ValueError(
                "[DLRM Error] All sparse/sequence features must share the same embedding_dim. " f"Got: {details}."
            )

        self.embedding_dim = int(next(iter(sparse_dims)))
        self.embedding = EmbeddingLayer(features=self.sparse_features_all)

        self.has_dense = len(dense_features) > 0
        self.dense_input_dim = sum(feature.input_dim for feature in dense_features)
        if self.has_dense:
            bottom_mlp_params.setdefault("hidden_dims", [128, 64])
            self.bottom_mlp = MLP(
                input_dim=self.dense_input_dim,
                output_dim=self.embedding_dim,
                **bottom_mlp_params,
            )
        else:
            self.bottom_mlp = None

        num_fields = len(self.sparse_features_all) + (1 if self.has_dense else 0)
        interaction_dim = num_fields * (num_fields - 1) // 2
        # Pairwise interactions [B, Interaction_dim] + optional dense embedding [B, D]
        top_input_dim = interaction_dim + (self.embedding_dim if self.has_dense else 0)
        top_mlp_params.setdefault("hidden_dims", [256, 128])
        self.top_mlp = MLP(input_dim=top_input_dim, output_dim=1, **top_mlp_params)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["bottom_mlp", "top_mlp"],
        )

    def build_dense_embedding(self, x) -> torch.Tensor:
        dense_values = []
        for feature in self.dense_features:
            value = x[feature.name].float()
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            else:
                value = value.view(value.size(0), -1)
            dense_values.append(value)

        # concat all dense fields -> [B, Dense_input_dim]
        dense_input = torch.cat(dense_values, dim=1)
        # bottom MLP projects dense inputs into one field embedding -> [B, D]
        return self.bottom_mlp(dense_input)

    def forward(self, x) -> torch.Tensor:
        # Sparse/sequence features -> [B, Field_num, D]
        sparse_embeddings = self.embedding(x=x, features=self.sparse_features_all, squeeze_dim=False)
        interaction_inputs = sparse_embeddings
        dense_embedding = None
        if self.has_dense:
            dense_embedding = self.build_dense_embedding(x)
            # Add dense field as one extra interaction field -> [B, Field_num + 1, D]
            interaction_inputs = torch.cat([dense_embedding.unsqueeze(1), sparse_embeddings], dim=1)

        # Pairwise dot-product interaction matrix -> [B, F, F]
        interaction_matrix = torch.bmm(interaction_inputs, interaction_inputs.transpose(1, 2))
        tril_i, tril_j = torch.tril_indices(
            row=interaction_inputs.size(1),
            col=interaction_inputs.size(1),
            offset=-1,
            device=interaction_inputs.device,
        )
        # [B, Interaction_dim]
        pairwise_interactions = interaction_matrix[:, tril_i, tril_j]

        top_inputs = [pairwise_interactions]
        if dense_embedding is not None:
            # standard DLRM keeps the processed dense embedding alongside interactions
            top_inputs.insert(0, dense_embedding)
        logits = self.top_mlp(torch.cat(top_inputs, dim=1))

        return logits
