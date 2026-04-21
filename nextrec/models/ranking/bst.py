"""
Date: create on 04/21/2026
Checkpoint: edit on 04/21/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Chen Q, Zhao H, Li W, et al. Behavior sequence transformer for e-commerce recommendation in Alibaba. In: Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data, 2019, pp. 1-4.
URL: https://arxiv.org/pdf/1905.06874

BST models user interest with Transformer encoders over behavior sequences and
target item embeddings. In this implementation, the final logit is produced by:
    (1) Behavior branch: each behavior sequence is embedded, concatenated with target tokens,
            encoded by Transformer, and mean-pooled into one interaction vector
    (2) Target branch: candidate sparse features are embedded and flattened directly
    (3) Context branch: remaining dense/sparse/sequence features are embedded by the
            shared EmbeddingLayer and concatenated as auxiliary input

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Target branch: candidate sparse features -> embedding layer -> embed_x_target: [Batch, Target_num, Dim_embedding]
- Behavior branch: one behavior sequence -> raw sequence embedding -> embed_x_history: [Batch, Seq_len, Dim_embedding]
- Transformer input: concat(embed_x_history, embed_x_target) -> transformer_seq: [Batch, Seq_len + Target_num, Dim_embedding]
- Transformer output: transformer_layers(transformer_seq) -> [Batch, Seq_len + Target_num, Dim_embedding]
- Sequence pooling: mean over sequence dimension -> [Batch, Dim_embedding]
- Behavior fusion: concat(all pooled behavior vectors) -> [Batch, Behavior_num * Dim_embedding]
- Final fusion: behavior fusion + flatten(embed_x_target) + optional context embedding -> [Batch, Dim_total]
- Output: MLP -> [Batch, 1] -> prediction layer

BST 通过 Transformer 对用户行为序列和目标物品 embedding 的拼接序列进行建模。
当前实现的最终 logit 由三部分组成：
    (1) 行为分支：每条行为序列与目标 token 拼接后送入 Transformer，再做均值池化得到交互向量
    (2) 目标分支：候选 sparse 特征直接 embedding 并展平
    (3) 上下文分支：其余 dense/sparse/sequence 特征通过共享 EmbeddingLayer 编码后拼接为辅助输入

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- 目标分支：candidate sparse 特征 -> embedding layer -> embed_x_target: [Batch, Target_num, Dim_embedding]
- 行为分支：单条行为序列 -> 原始序列 embedding -> embed_x_history: [Batch, Seq_len, Dim_embedding]
- Transformer 输入：concat(embed_x_history, embed_x_target) -> transformer_seq: [Batch, Seq_len + Target_num, Dim_embedding]
- Transformer 输出：transformer_layers(transformer_seq) -> [Batch, Seq_len + Target_num, Dim_embedding]
- 序列池化：沿序列维做均值池化 -> [Batch, Dim_embedding]
- 行为融合：concat(所有行为向量) -> [Batch, Behavior_num * Dim_embedding]
- 最终融合：行为向量 + flatten(embed_x_target) + 可选上下文 embedding -> [Batch, Dim_total]
- 输出：MLP -> [Batch, 1] -> 预测层
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import TaskTypeInput


class BST(BaseRankingModel):
    @property
    def model_name(self):
        return "BST"

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
        behavior_feature_name: str | list[str] | None = None,
        candidate_feature_name: str | list[str] | None = None,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.2,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize BST model.
        初始化 BST 模型。

        Args:
            behavior_feature_name: Behavior sequence feature name or names used by the Transformer branch.
                Each sequence feature is paired with the shared target tokens and encoded independently.
                Transformer 行为分支使用的行为序列特征名，可以是单个字符串或字符串列表。
                每条行为序列都会与共享的目标 token 拼接后独立编码。
            candidate_feature_name: Candidate sparse feature name or names used as target tokens.
                候选 sparse 特征名，可以是单个字符串或字符串列表，作为目标 token 输入。
            num_heads: Number of attention heads in the Transformer encoder.
                Transformer 编码器中的注意力头数。
            num_layers: Number of stacked Transformer encoder layers.
                Transformer 编码器的堆叠层数。
            dropout: Dropout rate inside the Transformer encoder.
                Transformer 编码器内部的 dropout 比例。
            mlp_params: Parameters for the final MLP. e.g. {"hidden_dims": [256, 128], "dropout": 0.2}.
                The input dimension is inferred automatically from behavior, target, and context branches.
                最终 MLP 的参数，例如 {"hidden_dims": [256, 128], "dropout": 0.2}。
                输入维度会根据行为分支、目标分支和上下文分支自动推断。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = dict(mlp_params or {})

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        if behavior_feature_name is None:
            raise ValueError("[BST Error] behavior_feature_name must be explicitly provided.")
        if candidate_feature_name is None:
            raise ValueError("[BST Error] candidate_feature_name must be explicitly provided.")

        behavior_feature_names = (
            [behavior_feature_name] if isinstance(behavior_feature_name, str) else list(behavior_feature_name)
        )
        candidate_feature_names = (
            [candidate_feature_name] if isinstance(candidate_feature_name, str) else list(candidate_feature_name)
        )
        if not behavior_feature_names:
            raise ValueError("[BST Error] behavior_feature_name cannot be empty.")
        if not candidate_feature_names:
            raise ValueError("[BST Error] candidate_feature_name cannot be empty.")

        self.behavior_features = select_feature_objects(
            sequence_features,
            behavior_feature_names,
            "behavior_feature_name",
        )
        self.candidate_features = select_feature_objects(
            sparse_features,
            candidate_feature_names,
            "candidate_feature_name",
        )

        self.embed_dim = int(self.behavior_features[0].embedding_dim)
        # behavior and candidate features must have same embedding dimension
        if any(
            int(feature.embedding_dim) != self.embed_dim
            for feature in (*self.behavior_features, *self.candidate_features)
        ):
            raise ValueError("[BST Error] behavior and candidate features must share the same embedding_dim.")
        if self.embed_dim % num_heads != 0:
            raise ValueError(
                f"[BST Error] embedding_dim({self.embed_dim}) must be divisible by num_heads({num_heads})."
            )

        exclude_feature_names = {feature.name for feature in (*self.behavior_features, *self.candidate_features)}
        self.other_features = [feature for feature in self.all_features if feature.name not in exclude_feature_names]

        self.embedding = EmbeddingLayer(features=self.all_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        other_dim = self.embedding.compute_output_dim(self.other_features)
        target_dim = self.embedding.compute_output_dim(self.candidate_features)
        transformer_dim = len(self.behavior_features) * self.embed_dim

        # behavior fusion [B, Behavior_num * D] + target flatten [B, Target_num * D] + optional context
        mlp_input_dim = transformer_dim + target_dim + other_dim
        mlp_params.setdefault("hidden_dims", [256, 128])
        self.mlp = MLP(input_dim=mlp_input_dim, output_dim=1, **mlp_params)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["transformer_layers", "mlp"],
        )

    def forward(self, x) -> torch.Tensor:
        # candidate sparse features -> [B, Target_num, D]
        embed_x_target = self.embedding(x=x, features=self.candidate_features, squeeze_dim=False)

        # concat behavior sequence and target embeddings
        transformer_pooling = []
        for behavior_feature in self.behavior_features:
            # raw behavior sequence embedding -> [B, Seq_len, D]
            embed_x_history = self.embedding(
                x=x,
                features=[behavior_feature],
                squeeze_dim=False,
                sequence_output="raw",
            ).squeeze(1)
            # concat history tokens and target tokens -> [B, Seq_len + Target_num, D]
            transformer_seq = self.transformer_layers(torch.cat([embed_x_history, embed_x_target], dim=1))
            # mean-pool one behavior branch -> [B, 1, D]
            transformer_pooling.append(torch.mean(transformer_seq, dim=1, keepdim=True))

        # all behavior branches -> [B, Behavior_num, D]
        transformer_pooling = torch.cat(transformer_pooling, dim=1)

        features = [
            # flatten behavior outputs -> [B, Behavior_num * D]
            transformer_pooling.flatten(start_dim=1),
            # flatten target tokens -> [B, Target_num * D]
            embed_x_target.flatten(start_dim=1),
        ]
        if self.other_features:
            # auxiliary context features -> [B, Context_dim]
            features.append(self.embedding(x=x, features=self.other_features, squeeze_dim=True))
        logits = self.mlp(torch.cat(features, dim=1))

        return logits
