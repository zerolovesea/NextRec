"""
Date: create on 09/11/2025
Checkpoint: edit on 09/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate
prediction[C] //Proceedings of the 24th ACM SIGKDD international conference on
knowledge discovery & data mining. 2018: 1059-1068.
(https://arxiv.org/abs/1706.06978)

Deep Interest Network (DIN) is a CTR model that builds a target-aware user
representation by attending over the historical behavior sequence. Instead of
compressing all behaviors into one static vector, DIN highlights the behaviors
most relevant to the current candidate item, enabling adaptive interest
modeling for each request.

Pipeline:
  (1) Embed candidate item, user behavior sequence, and other sparse/dense fields
  (2) Use a small attention MLP to score each historical behavior against the
      candidate embedding
  (3) Apply masked weighted pooling to obtain a target-specific interest vector
  (4) Concatenate candidate, interest vector, other sparse embeddings, and dense
      features
  (5) Feed the combined representation into an MLP for final prediction

Key Advantages:
- Target-aware attention captures fine-grained interests per candidate item
- Adaptive pooling handles diverse behavior patterns without heavy feature crafting
- Masked weighting reduces noise from padded sequence positions
- Easily incorporates additional sparse/dense context features alongside behavior

DIN 是一个 CTR 预估模型，通过对用户历史行为序列进行目标感知的注意力加权，
构建针对当前候选物品的兴趣表示。它不是将全部行为压缩为固定向量，而是突出
与候选物品最相关的行为，实现请求级的自适应兴趣建模。

处理流程：
  (1) 对候选物品、用户行为序列及其他稀疏/稠密特征做 embedding
  (2) 使用小型注意力 MLP 计算每个历史行为与候选 embedding 的相关性
  (3) 通过掩码加权池化得到目标特定的兴趣向量
  (4) 拼接候选、兴趣向量、其他稀疏 embedding 与稠密特征
  (5) 输入 MLP 完成最终点击率预测

主要优点：
- 目标感知注意力捕捉候选级的细粒度兴趣
- 自适应池化应对多样化行为模式，减少手工特征工程
- 掩码加权降低序列填充位置的噪声
- 便捷融合行为与额外稀疏/稠密上下文信息
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    MLP,
    AttentionPoolingLayer,
    EmbeddingLayer,
)
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class DIN(BaseModel):
    @property
    def model_name(self):
        return "DIN"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        behavior_feature_name: str | None = None,
        candidate_feature_name: str | None = None,
        mlp_params: dict | None = None,
        attention_hidden_units: list[int] | None = None,
        attention_activation: str = "dice",
        attention_use_softmax: bool = True,
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
        attention_hidden_units = attention_hidden_units or [80, 40]
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(DIN, self).__init__(
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

        # DIN requires: user behavior sequence + candidate item + other features
        if len(sequence_features) == 0:
            raise ValueError(
                "DIN requires at least one sequence feature for user behavior history"
            )
        if behavior_feature_name is None:
            raise ValueError("DIN requires an explicit behavior_feature_name")

        if candidate_feature_name is None:
            raise ValueError("DIN requires an explicit candidate_feature_name")

        self.behavior_feature = [
            f for f in sequence_features if f.name == behavior_feature_name
        ][0]
        self.candidate_feature = [
            f for f in sparse_features if f.name == candidate_feature_name
        ][0]

        # Other sparse features
        self.other_sparse_features = [
            f for f in sparse_features if f.name != self.candidate_feature.name
        ]

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Attention layer for behavior sequence
        behavior_emb_dim = self.behavior_feature.embedding_dim
        self.candidate_attention_proj = None
        if (
            self.candidate_feature is not None
            and self.candidate_feature.embedding_dim != behavior_emb_dim
        ):
            self.candidate_attention_proj = nn.Linear(
                self.candidate_feature.embedding_dim, behavior_emb_dim
            )
        self.attention = AttentionPoolingLayer(
            embedding_dim=behavior_emb_dim,
            hidden_units=attention_hidden_units,
            activation=attention_activation,
            use_softmax=attention_use_softmax,
        )

        # Calculate MLP input dimension
        # candidate + attention_pooled_behavior + other_sparse + dense
        mlp_input_dim = 0
        if self.candidate_feature:
            mlp_input_dim += self.candidate_feature.embedding_dim
        mlp_input_dim += behavior_emb_dim  # attention pooled
        mlp_input_dim += sum([f.embedding_dim for f in self.other_sparse_features])
        mlp_input_dim += sum(
            [getattr(f, "embedding_dim", 1) or 1 for f in dense_features]
        )

        # MLP for final prediction
        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["attention", "mlp", "candidate_attention_proj"],
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get candidate item embedding
        if self.candidate_feature is None:
            raise ValueError("DIN requires a candidate item feature")
        candidate_emb = self.embedding.embed_dict[
            self.candidate_feature.embedding_name
        ](
            x[self.candidate_feature.name].long()
        )  # [B, emb_dim]

        # Get behavior sequence embedding
        behavior_seq = x[self.behavior_feature.name].long()  # [B, seq_len]
        behavior_emb = self.embedding.embed_dict[self.behavior_feature.embedding_name](
            behavior_seq
        )  # [B, seq_len, emb_dim]

        # Create mask for padding
        if self.behavior_feature.padding_idx is not None:
            mask = (
                (behavior_seq != self.behavior_feature.padding_idx)
                .unsqueeze(-1)
                .float()
            )
        else:
            mask = (behavior_seq != 0).unsqueeze(-1).float()

        # Apply attention pooling
        candidate_query = candidate_emb
        if self.candidate_attention_proj is not None:
            candidate_query = self.candidate_attention_proj(candidate_query)
        pooled_behavior = self.attention(
            query=candidate_query, keys=behavior_emb, mask=mask
        )  # [B, emb_dim]

        # Get other features
        other_embeddings = []

        other_embeddings.append(candidate_emb)

        other_embeddings.append(pooled_behavior)

        # Other sparse features
        for feat in self.other_sparse_features:
            feat_emb = self.embedding.embed_dict[feat.embedding_name](
                x[feat.name].long()
            )
            other_embeddings.append(feat_emb)

        # Dense features
        for feat in self.dense_features:
            dense_val = self.embedding.project_dense(feat, x)
            other_embeddings.append(dense_val)

        # Concatenate all features
        concat_input = torch.cat(other_embeddings, dim=-1)  # [B, total_dim]

        # MLP prediction
        y = self.mlp(concat_input)  # [B, 1]
        return self.prediction_layer(y)
