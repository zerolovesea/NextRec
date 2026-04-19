"""
Date: create on 09/11/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic feature interaction learning via self-attentive neural networks. In: Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM ’19), 2019, pp. 1161–1170.
URL: https://arxiv.org/abs/1810.11921

AutoInt models explicit high-order feature interactions with stacked multi-head
self-attention over field embeddings. In this implementation, the final logit is
the sum of:
  (1) Attention branch: self-attentive interaction representation -> linear projection
  (2) Wide branch: linear term over flattened feature embeddings (always enabled)
  (3) Optional DNN branch: MLP over flattened feature embeddings (enabled when mlp_params is provided)

This implementation also includes a projection step to unify embedding dimensions
before attention when features have mixed dimensions
(e.g. sequence features with combiner="concat" or inclusion of dense features).

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding layer -> [Batch, Dim_embedding]
- Projection (optional): [Batch, Dim_embedding] -> linear projection -> field_i_attention: [Batch, Dim_attention]
- Attention branch: stack(field_i_attention) -> attention_input: [Batch, Field_num, Dim_attention] -> multi-head self-attention layers -> attention_out: [Batch, Field_num, Dim_attention]
- Attention branch output: flatten(attention_out) -> [Batch, Field_num * Dim_attention] -> linear layer -> y_attention: [Batch, 1]
- Wide branch: embedding flatten -> flat_input: [Batch, Dim_embedding] -> LR -> y_wide: [Batch, 1]
- Deep branch (optional): flat_input: [Batch, Dim_embedding] -> MLP -> y_dnn: [Batch, 1]
- Fusion: y_attention: [Batch, 1] + y_wide: [Batch, 1] (+ y_dnn: [Batch, 1]) -> [Batch, 1]
- Output: [Batch, 1] -> prediction layer

AutoInt 通过堆叠多头自注意力在field级 embedding上显式学习高阶特征交互。
当前实现的最终 logit 由三部分组成：
  (1) Attention 分支：自注意力交互表示后接线性映射
  (2) Wide 分支：对展平后的特征 embedding 做线性建模（默认始终启用）
  (3) 可选 DNN 分支：对展平后的特征 embedding 送入 MLP（提供 mlp_params 时启用）

本实现中当特征维度不一致（例如 sequence 的 combiner="concat"或传入dense特征）时，会先把每个字段投影到统一
的注意力维度，再进行自注意力计算。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length] -> embedding layer-> [Batch, Dim_embedding]
- 投影（可选）：[Batch, Dim_embedding] -> 线性投影 -> field_i_attention: [Batch, Dim_attention]
- Attention 分支：stack(field_i_attention) -> attention_input: [Batch, Field_num, Dim_attention] -> 多层多头自注意力 -> attention_out: [Batch, Field_num, Dim_attention]
- Attention 分支输出：flatten(attention_out) -> [Batch, Field_num * Dim_attention] -> 线性层 -> y_attention: [Batch, 1]
- Wide 分支：embedding 展平 -> flat_input: [Batch, Dim_embedding] -> LR -> y_wide: [Batch, 1]
- Deep 分支（可选）：flat_input: [Batch, Dim_embedding] -> MLP -> y_dnn: [Batch, 1]
- 融合：y_attention: [Batch, 1] + y_wide: [Batch, 1] (+ y_dnn: [Batch, 1]) -> [Batch, 1]
- 输出：[Batch, 1] -> 预测层

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, LR, MLP, MultiHeadSelfAttention
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class AutoInt(BaseRankingModel):
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
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        att_layer_num: int = 3,
        att_embedding_dim: int = 8,
        att_head_num: int = 2,
        att_dropout: float = 0.0,
        att_use_residual: bool = True,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize AutoInt model.
        初始化 AutoInt 模型。

        Args:
            att_layer_num: Number of stacked multi-head self-attention layers.
                多头自注意力层的堆叠层数。
            att_embedding_dim: Unified embedding dimension used by attention layers.
                注意力层使用的统一特征维度。
            att_head_num: Number of attention heads in each self-attention layer.
                每层自注意力中的头数。
            att_dropout: Dropout rate inside attention layers.
                注意力层内部的 dropout 比例。
            att_use_residual: Whether to use residual connections in attention layers.
                是否在注意力层中启用残差连接。
            mlp_params: Parameters for optional deep branch MLP. e.g. {"hidden_dims": [64, 32], "dropout": 0.2}.
                If None, deep branch is disabled.
                可选深度分支 MLP 的参数，例如 {"hidden_dims": [64, 32], "dropout": 0.2}；为 None 时不启用深度分支。
        """

        super(AutoInt, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []

        self.att_layer_num = att_layer_num
        self.att_embedding_dim = att_embedding_dim

        # Use sparse/sequence/dense features for interaction
        # **INFO**: this is different from the original paper, we also include dense features
        # if you want to follow the paper strictly, set dense_features=[]
        # or modify the code accordingly
        self.interaction_features = dense_features + sparse_features + sequence_features
        self.embedding = EmbeddingLayer(features=self.interaction_features)

        # Project embeddings to attention embedding dimension
        num_fields = len(self.interaction_features)

        # Field output dims come from EmbeddingLayer output semantics (e.g. sequence concat pooling).
        self.field_output_dims = [self.embedding.get_input_dim([feature]) for feature in self.interaction_features]

        # If embeddings have different dimensions, project them to att_embedding_dim
        self.need_projection = not all(dim == att_embedding_dim for dim in self.field_output_dims)
        self.projection_layers = None
        if self.need_projection:
            self.projection_layers = nn.ModuleList(
                [nn.Linear(in_dim, att_embedding_dim, bias=False) for in_dim in self.field_output_dims]
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
        self.attn_fc = nn.Linear(num_fields * att_embedding_dim, 1)

        self.use_dnn = mlp_params is not None
        if self.use_dnn:
            dnn_params = dict(mlp_params or {})
            dnn_input_dim = self.embedding.get_input_dim(self.interaction_features)
            user_input_dim = dnn_params.pop("input_dim", None)
            if user_input_dim is not None and int(user_input_dim) != dnn_input_dim:
                raise ValueError(
                    f"AutoInt sets MLP input_dim automatically to {dnn_input_dim}, but mlp_params['input_dim']={user_input_dim}."
                )
            dnn_params.setdefault("output_dim", 1)
            self.mlp = MLP(input_dim=dnn_input_dim, **dnn_params)
        else:
            self.mlp = None

        self.linear = LR(self.embedding.get_input_dim(self.interaction_features))

        include_modules = ["projection_layers", "attention_layers", "attn_fc"]
        if self.use_dnn:
            include_modules.append("mlp")
        include_modules.append("linear")
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=include_modules,
        )

    def forward(self, x):
        # Get embeddings field-by-field so mixed dimensions can be projected safely
        field_embeddings = []
        for idx, feature in enumerate(self.interaction_features):
            feature_emb = self.embedding(x=x, features=[feature], squeeze_dim=False)
            feature_emb = feature_emb.squeeze(1)  # [Batch, Dim_embedding]
            if self.need_projection and self.projection_layers is not None:
                feature_emb = self.projection_layers[idx](feature_emb)
            field_embeddings.append(feature_emb.unsqueeze(1))  # [Batch, 1, Dim_attention]
        embeddings = torch.cat(field_embeddings, dim=1)

        # Apply multi-head self-attention layers
        attention_output = embeddings
        for att_layer in self.attention_layers:
            attention_output = att_layer(attention_output)  # [Batch, num_fields, Dim_attention]

        # Attention branch
        attention_output_flat = attention_output.flatten(start_dim=1)  # [Batch, num_fields * Dim_attention]
        logits = self.attn_fc(attention_output_flat)  # [Batch, 1]

        flat_input = self.embedding(x=x, features=self.interaction_features, squeeze_dim=True)

        # Optional DNN branch
        if self.use_dnn and self.mlp is not None:
            logits = logits + self.mlp(flat_input)

        # Wide branch
        logits = logits + self.linear(flat_input)
        return logits
