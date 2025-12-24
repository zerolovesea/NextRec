"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Song W, Shi C, Xiao Z, et al. Autoint: Automatic feature interaction learning via
self-attentive neural networks[C]//Proceedings of the 28th ACM international conference
on information and knowledge management. 2019: 1161-1170.
(https://arxiv.org/abs/1810.11921)

AutoInt is a CTR prediction model that leverages multi-head self-attention
to automatically learn high-order feature interactions in an explicit and
interpretable way. Instead of relying on manual feature engineering or
implicit MLP-based transformations, AutoInt models feature dependencies
by attending over all embedded fields and capturing their contextual
relationships.

In each Interacting Layer:
  (1) Each field embedding is projected into multiple attention heads
  (2) Scaled dot-product attention computes feature-to-feature interactions
  (3) Outputs are aggregated and passed through residual connections
  (4) Layer Normalization ensures stable optimization

By stacking multiple Interacting Layers, AutoInt progressively discovers
higher-order feature interactions, while maintaining transparency since
attention weights explicitly show which features interact.

Key Advantages:
- Explicit modeling of high-order feature interactions
- Multi-head attention enhances representation diversity
- Residual structure facilitates deep interaction learning
- Attention weights provide interpretability of feature relations
- Eliminates heavy manual feature engineering

AutoInt 是一个 CTR 预估模型，通过多头自注意力机制显式学习高阶特征交互，
并具有良好的可解释性。不同于依赖人工特征工程或 MLP 隐式建模的方法，
AutoInt 通过对所有特征 embedding 进行注意力计算，捕捉特征之间的上下文依赖关系。

在每个 Interacting Layer（交互层）中：
  (1) 每个特征 embedding 通过投影分成多个注意力头
  (2) 使用缩放点积注意力计算特征间交互权重
  (3) 将多头输出进行聚合，并使用残差连接
  (4) Layer Normalization 确保训练稳定性

通过堆叠多个交互层，AutoInt 能逐步学习更高阶的特征交互；
同时由于注意力权重可视化，模型具有明确的可解释能力，
能展示哪些特征之间的关系最重要。

主要优点：
- 显式建模高阶特征交互
- 多头机制增强表示能力
- 残差结构支持深层交互学习
- 注意力权重天然具备可解释性
- 减少繁重的人工特征工程工作
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MultiHeadSelfAttention
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class AutoInt(BaseModel):
    @property
    def model_name(self):
        return "AutoInt"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        att_layer_num: int = 3,
        att_embedding_dim: int = 8,
        att_head_num: int = 2,
        att_dropout: float = 0.0,
        att_use_residual: bool = True,
        target: list[str] | None = None,
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

        super(AutoInt, self).__init__(
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

        if target is None:
            target = []
        if optimizer_params is None:
            optimizer_params = {}
        if loss is None:
            loss = "bce"

        self.att_layer_num = att_layer_num
        self.att_embedding_dim = att_embedding_dim

        # Use sparse and sequence features for interaction
        # **INFO**: this is different from the original paper, we also include dense features
        # if you want to follow the paper strictly, set dense_features=[]
        # or modify the code accordingly
        self.interaction_features = dense_features + sparse_features + sequence_features

        # All features for embedding
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Project embeddings to attention embedding dimension
        num_fields = len(self.interaction_features)

        # If embeddings have different dimensions, project them to att_embedding_dim
        self.need_projection = not all(
            f.embedding_dim == att_embedding_dim for f in self.interaction_features
        )
        self.projection_layers = None
        if self.need_projection:
            self.projection_layers = nn.ModuleList(
                [
                    nn.Linear(f.embedding_dim, att_embedding_dim, bias=False)
                    for f in self.interaction_features
                ]
            )

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList(
            [
                MultiHeadSelfAttention(
                    embedding_dim=att_embedding_dim,
                    num_heads=att_head_num,
                    dropout=att_dropout,
                    use_residual=att_use_residual,
                )
                for _ in range(att_layer_num)
            ]
        )

        # Final prediction layer
        self.fc = nn.Linear(num_fields * att_embedding_dim, 1)
        self.prediction_layer = TaskHead(task_type=self.default_task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["projection_layers", "attention_layers", "fc"],
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get embeddings field-by-field so mixed dimensions can be projected safely
        field_embeddings = []
        if len(self.interaction_features) == 0:
            raise ValueError(
                "AutoInt requires at least one sparse or sequence feature for interactions."
            )
        for idx, feature in enumerate(self.interaction_features):
            feature_emb = self.embedding(x=x, features=[feature], squeeze_dim=False)
            feature_emb = feature_emb.squeeze(1)  # [B, embedding_dim]
            if self.need_projection and self.projection_layers is not None:
                feature_emb = self.projection_layers[idx](feature_emb)
            field_embeddings.append(
                feature_emb.unsqueeze(1)
            )  # [B, 1, att_embedding_dim or original_dim]
        embeddings = torch.cat(field_embeddings, dim=1)

        # Apply multi-head self-attention layers
        attention_output = embeddings
        for att_layer in self.attention_layers:
            attention_output = att_layer(
                attention_output
            )  # [B, num_fields, att_embedding_dim]

        # Flatten and predict
        attention_output_flat = attention_output.flatten(
            start_dim=1
        )  # [B, num_fields * att_embedding_dim]
        y = self.fc(attention_output_flat)  # [B, 1]
        return self.prediction_layer(y)
