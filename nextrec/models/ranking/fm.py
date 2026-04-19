"""
Date: create on 09/11/2025
Checkpoint: edit on 15/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Rendle S. Factorization machines[C]//ICDM. 2010: 995-1000.

Factorization Machines (FM) capture second-order feature interactions with
linear complexity by factorizing the pairwise interaction matrix. Each field
is embedded into a latent vector; FM models the dot product of every pair of
embeddings and sums them along with a linear term, enabling strong performance
with sparse high-dimensional data and minimal feature engineering.

Workflow:
- Embed sparse and sequence fields into low-dimensional vectors
- Compute linear logit over concatenated embeddings
- Compute pairwise interaction logit via factorized dot products
- Sum linear + interaction terms and apply prediction layer

FM 是一种通过分解二阶特征交互矩阵、以线性复杂度建模特征对的 CTR 模型。
每个特征映射为低维向量，FM 对任意特征对进行内积求和并叠加线性项，
无需复杂特征工程即可在稀疏高维场景取得稳健效果。

流程：
- 对稀疏/序列特征做 embedding
- 计算线性部分的 logit
- 计算嵌入对之间的二阶交互 logit
- 线性项与交互项求和，再通过预测层输出
"""

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import FM as FMInteraction
from nextrec.basic.layers import LR, EmbeddingLayer
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class FM(BaseRankingModel):
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
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
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
            task=task,
            **kwargs,
        )

        self.fm_features = sparse_features + sequence_features
        if len(self.fm_features) == 0:
            raise ValueError("FM requires at least one sparse or sequence feature.")

        self.embedding = EmbeddingLayer(features=self.fm_features)

        fm_input_dim = sum([f.embedding_dim for f in self.fm_features])
        self.linear = LR(fm_input_dim)
        self.fm = FMInteraction(reduce_sum=True)

        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["linear"])

    def forward(self, x):
        input_fm = self.embedding(x=x, features=self.fm_features, squeeze_dim=False)
        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        logits = y_linear + y_fm
        return logits
