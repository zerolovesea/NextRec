"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Cheng H T, Koc L, Harmsen J, et al. Wide & Deep learning for recommender systems[C] //Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 2016: 7-10.
  URL: https://arxiv.org/abs/1606.07792

Wide & Deep blends a linear wide component (memorization of cross features) with a
deep neural network (generalization) sharing the same feature space. The wide part
captures co-occurrence patterns and manual crosses, while the deep part learns dense
representations and nonlinear interactions, improving both accuracy and coverage.

Workflow:
  (1) Wide: linear/logistic model over raw or embedded features
  (2) Deep: embeddings plus dense features feed into an MLP
  (3) Sum wide and deep logits, then apply the final prediction layer

Key Advantages:
- Balances memorization (wide) and generalization (deep)
- Compatible with manual crosses and automatically learned embeddings
- Simple architecture with strong baselines for CTR/ranking
- Shared feature space reduces duplication and engineering overhead

Wide & Deep 同时使用宽线性部分（记忆共现/手工交叉）与深网络部分（泛化非线性交互），
共享特征表示，既保留记忆能力又具备泛化能力，常用于 CTR/排序任务。

流程：
  (1) Wide：线性/逻辑回归建模原始或 embedding 后的特征
  (2) Deep：embedding 与稠密特征输入 MLP
  (3) 宽深输出求和后进入最终预测

主要优点：
- 兼顾记忆与泛化
- 支持手工交叉与自动 embedding 联合
- 结构简单，基线性能稳定
- 共享特征空间，减少工程开销
"""

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import LR, MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class WideDeep(BaseModel):
    @property
    def model_name(self):
        return "WideDeep"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        mlp_params: dict,
        **kwargs,
    ):

        super(WideDeep, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            **kwargs,
        )

        # Wide part: use all features for linear model
        self.wide_features = sparse_features + sequence_features
        # Deep part: use all features
        self.deep_features = dense_features + sparse_features + sequence_features

        # Embedding layer for deep part
        self.embedding = EmbeddingLayer(features=self.deep_features)

        # Wide part: Linear layer
        wide_dim = sum([f.embedding_dim for f in self.wide_features])
        self.linear = LR(wide_dim)

        # Deep part: MLP
        input_dim = self.embedding.input_dim
        # deep_emb_dim_total = sum([f.embedding_dim for f in self.deep_features if not isinstance(f, DenseFeature)])
        # dense_input_dim = sum([(f.embedding_dim or 1) for f in dense_features])
        self.mlp = MLP(input_dim=input_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["linear", "mlp"]
        )

    def forward(self, x):
        # Deep part
        input_deep = self.embedding(x=x, features=self.deep_features, squeeze_dim=True)
        y_deep = self.mlp(input_deep)  # [B, 1]

        # Wide part
        input_wide = self.embedding(x=x, features=self.wide_features, squeeze_dim=True)
        y_wide = self.linear(input_wide)

        # Combine wide and deep
        y = y_wide + y_deep
        return self.prediction_layer(y)
