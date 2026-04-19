"""
Date: create on 17/02/2026
Checkpoint: edit on 17/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. 2017: 355-364.
  URL: https://arxiv.org/abs/1708.05027

Neural Factorization Machine (NFM) combines a linear first-order branch with a
Bi-Interaction pooling branch followed by an MLP. Compared with DeepFM, NFM
explicitly uses the second-order Bi-Interaction vector as the deep input base,
then fuses optional dense signals before nonlinear transformation.

Workflow:
- Embed sparse/sequence fields for linear and Bi-Interaction branches
- Linear branch models first-order effects on flattened field embeddings
- Bi-Interaction pooling computes 0.5 * ((sum v)^2 - sum(v^2))
- Concatenate Bi-Interaction output with optional dense features
- Feed fused input into MLP and sum with linear logits for prediction

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- FM embedding: sparse + sequence -> input_fm: [Batch, Field_num, Dim_embedding]
- Linear branch: flatten(input_fm) -> linear_input: [Batch, Field_num * Dim_embedding] -> y_linear: [Batch, 1]
- Bi-Interaction branch: input_fm -> bi_out: [Batch, Dim_embedding]
- Optional dense branch: dense -> dense_input: [Batch, Dim_dense]
- Deep branch input:
  with dense -> deep_input: [Batch, Dim_embedding + Dim_dense]
  without dense -> deep_input: [Batch, Dim_embedding]
- Deep branch: deep_input -> MLP -> y_deep: [Batch, 1]
- Fusion: y_linear + y_deep -> y: [Batch, 1]
- Output: y -> prediction layer -> [Batch, 1]

NFM（Neural Factorization Machine）将线性一阶分支与 Bi-Interaction 池化分支结合，
并在交互向量后接 MLP 进行高阶建模。相较于 DeepFM，NFM 将显式二阶交互向量作为
深层输入主干，再按需融合稠密特征。

流程：
- 对 sparse/sequence 字段做 embedding，供线性与交互分支共享
- 线性分支对展平后的字段向量建模一阶信号
- Bi-Interaction 分支计算 0.5 * ((sum v)^2 - sum(v^2))
- 将交互向量与可选 dense 特征拼接
- 输入 MLP 后与线性分支求和，进入预测层

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- FM embedding：sparse + sequence -> input_fm: [Batch, Field_num, Dim_embedding]
- 线性分支：flatten(input_fm) -> linear_input: [Batch, Field_num * Dim_embedding] -> y_linear: [Batch, 1]
- Bi-Interaction 分支：input_fm -> bi_out: [Batch, Dim_embedding]
- 可选 dense 分支：dense -> dense_input: [Batch, Dim_dense]
- 深层输入：
  含 dense -> deep_input: [Batch, Dim_embedding + Dim_dense]
  不含 dense -> deep_input: [Batch, Dim_embedding]
- 深层分支：deep_input -> MLP -> y_deep: [Batch, 1]
- 融合：y_linear + y_deep -> y: [Batch, 1]
- 输出：y -> 预测层 -> [Batch, 1]
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, LR, MLP
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class NFM(BaseRankingModel):
    @property
    def model_name(self):
        return "NFM"

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
        mlp_params: dict | None = None,
        bi_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initialize NFM model.
        初始化 NFM 模型。

        Args:
            mlp_params: Parameters for NFM deep MLP branch, e.g.
                {"hidden_dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}.
                NFM 深层 MLP 分支参数。
            bi_dropout: Dropout rate applied on Bi-Interaction output before MLP.
                作用在 Bi-Interaction 输出上的 dropout 比例（进入 MLP 之前）。

        Notes:
            - NFM requires at least two sparse/sequence features to build feature interactions.
            - All sparse/sequence features in NFM interaction branch must share the same effective output dimension.
            - NFM 至少需要两个 sparse/sequence 特征用于构造交互。
            - NFM 交互分支中的 sparse/sequence 特征必须具有相同的有效输出维度。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}

        super(NFM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.fm_features = sparse_features + sequence_features
        self.dense_features = dense_features
        if len(self.fm_features) < 2:
            raise ValueError("[NFM Error]: requires at least two sparse or sequence features.")

        self.embedding = EmbeddingLayer(features=self.dense_features + self.fm_features)

        fm_field_dims = [self.embedding.compute_output_dim([feature]) for feature in self.fm_features]
        if len(set(fm_field_dims)) != 1:
            feature_dims = ", ".join(f"{feature.name}:{dim}" for feature, dim in zip(self.fm_features, fm_field_dims))
            raise ValueError(
                f"[NFM Error]: Bi-Interaction branch requires sparse/sequence fields to share the same effective dimension for squeeze_dim=False, but got [{feature_dims}]."
            )

        self.bi_output_dim = fm_field_dims[0]
        dense_input_dim = self.embedding.compute_output_dim(self.dense_features)
        self.deep_input_dim = self.bi_output_dim + dense_input_dim

        linear_input_dim = self.embedding.compute_output_dim(self.fm_features)
        self.linear = LR(linear_input_dim)
        self.bi_dropout = nn.Dropout(bi_dropout) if bi_dropout > 0 else nn.Identity()
        self.mlp = MLP(input_dim=self.deep_input_dim, **mlp_params)

        self.register_regularization_weights(embedding_attr="embedding", include_modules=["linear", "mlp"])

    def forward(self, x):
        # FM embedding: sparse/sequence -> input_fm: [Batch, Field_num, Dim_embedding]
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)

        # Linear branch: flatten(input_fm): [Batch, Field_num * Dim_embedding] -> y_linear: [Batch, 1]
        y_linear = self.linear(input_fm.flatten(start_dim=1))

        # Bi-Interaction pooling: input_fm -> bi_out: [Batch, Dim_embedding]
        sum_of_emb = torch.sum(input_fm, dim=1)
        bi_out = 0.5 * (sum_of_emb * sum_of_emb - torch.sum(input_fm * input_fm, dim=1))
        # Dropout keeps shape unchanged: [Batch, Dim_embedding]
        bi_out = self.bi_dropout(bi_out)

        if self.dense_features:
            # Dense branch: dense -> dense_input: [Batch, Dim_dense]
            dense_input = self.embedding(x=x, features=self.dense_features, squeeze_dim=True)
            # Deep input (with dense): [Batch, Dim_embedding + Dim_dense]
            deep_input = torch.cat([bi_out, dense_input], dim=1)
        else:
            # Deep input (without dense): [Batch, Dim_embedding]
            deep_input = bi_out

        # Deep branch: deep_input -> y_deep: [Batch, 1]
        y_deep = self.mlp(deep_input)

        logits = y_linear + y_deep
        return logits
