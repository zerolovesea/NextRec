"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Rendle S. Factorization machines[C]//ICDM. 2010: 995-1000.

Factorization Machines (FM) capture second-order feature interactions with
linear complexity by factorizing the pairwise interaction matrix. Each field
is embedded into a latent vector; FM models the dot product of every pair of
embeddings and sums them along with a linear term, enabling strong performance
with sparse high-dimensional data and minimal feature engineering.

Pipeline:
  (1) Embed sparse and sequence fields into low-dimensional vectors
  (2) Compute linear logit over concatenated embeddings
  (3) Compute pairwise interaction logit via factorized dot products
  (4) Sum linear + interaction terms and apply prediction layer

Key Advantages:
- Models pairwise interactions efficiently (O(nk) vs. O(n^2))
- Works well on sparse inputs without handcrafted crosses
- Simple architecture with strong baseline performance

FM 是一种通过分解二阶特征交互矩阵、以线性复杂度建模特征对的 CTR 模型。
每个特征映射为低维向量，FM 对任意特征对进行内积求和并叠加线性项，
无需复杂特征工程即可在稀疏高维场景取得稳健效果。

处理流程：
  (1) 对稀疏/序列特征做 embedding
  (2) 计算线性部分的 logit
  (3) 计算嵌入对之间的二阶交互 logit
  (4) 线性项与交互项求和，再通过预测层输出

主要优点：
- 线性复杂度建模二阶交互，效率高
- 对稀疏特征友好，减少人工特征交叉
- 结构简单、表现强健，常作 CTR 基线
"""

import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import FM as FMInteraction
from nextrec.basic.heads import TaskHead
from nextrec.basic.layers import LR, EmbeddingLayer
from nextrec.basic.model import BaseModel


class FM(BaseModel):
    @property
    def model_name(self):
        return "FM"

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

        super(FM, self).__init__(
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
        if self.loss is None:
            self.loss = "bce"

        self.fm_features = sparse_features + sequence_features
        if len(self.fm_features) == 0:
            raise ValueError("FM requires at least one sparse or sequence feature.")

        self.embedding = EmbeddingLayer(features=self.fm_features)

        fm_input_dim = sum([f.embedding_dim for f in self.fm_features])
        self.linear = LR(fm_input_dim)
        self.fm = FMInteraction(reduce_sum=True)
        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
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
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)
        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y = y_linear + y_fm
        return self.prediction_layer(y)
