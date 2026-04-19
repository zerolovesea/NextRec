"""
Date: create on 19/12/2025
Checkpoint: edit on 14/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Juan Y, Zhuang Y, Chin W-S, et al. Field-aware Factorization Machines for CTR Prediction[C]//RecSys. 2016: 43-50.

Field-aware Factorization Machines (FFM) extend FM by learning a distinct
embedding of each feature for every target field. For a pair of fields (i, j),
FFM uses v_{i,f_j} · v_{j,f_i} instead of a shared embedding, enabling richer
context-aware interactions and stronger CTR performance on sparse categorical
data.

Workflow:
- Build field-aware embeddings v_{i,f} for each feature i toward every field f
- Compute first-order linear terms for sparse/sequence (and optional dense) fields
- For each field pair (i, j), compute v_{i,f_j} · v_{j,f_i}
- Sum linear + field-aware interaction logits and output prediction

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- First-order dense branch: concat(dense_i) -> dense_stack: [Batch, Dim_dense] -> linear_dense -> y_dense: [Batch, 1]
- First-order sparse/sequence branch:
  - sparse_i id -> embedding_dim=1 -> term_i: [Batch, 1]
  - sequence_j id list -> embedding_dim=1 + mask pooling -> term_j: [Batch, 1]
  - sum all terms -> y_sparse_seq: [Batch, 1]
- First-order fusion: y_linear = y_dense + y_sparse_seq -> [Batch, 1]
- Field-aware branch: field_aware_outputs: [Batch, Field_num, Field_num, Dim_embedding]
- Pair interaction: for each (i, j), v_{i,f_j}:[Batch, Dim_embedding] · v_{j,f_i}:[Batch, Dim_embedding] -> dot_ij:[Batch, 1]
- Interaction sum: y_interaction: [Batch, 1]
- Fusion: y = y_linear + y_interaction -> [Batch, 1]
- Output: [Batch, 1] -> prediction layer

FFM 在 FM 基础上引入域感知机制：每个特征在不同域下拥有不同的 embedding。
对于域对 (i, j)，模型使用 v_{i,f_j} 与 v_{j,f_i} 的内积，从而更细粒度地建模跨域交互，在稀疏高维 CTR 场景中表现更优。

流程：
- 为每个特征 i 构造面向每个域 f 的嵌入 v_{i,f}
- 计算一阶线性项（稀疏/序列特征，及可选的稠密特征）
- 对每一对域 (i, j) 计算 v_{i,f_j} · v_{j,f_i}
- 将线性项与交互项相加得到最终预测

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- 一阶稠密分支：拼接 dense_i -> dense_stack: [Batch, Dim_dense] -> linear_dense -> y_dense: [Batch, 1]
- 一阶稀疏/序列分支：
  - sparse_i 的 id -> embedding_dim=1 -> term_i: [Batch, 1]
  - sequence_j 的 id 序列 -> embedding_dim=1 + mask pooling -> term_j: [Batch, 1]
  - 各项求和 -> y_sparse_seq: [Batch, 1]
- 一阶融合：y_linear = y_dense + y_sparse_seq -> [Batch, 1]
- 域感知分支：field_aware_outputs: [Batch, Field_num, Field_num, Dim_embedding]
- 配对交互：对每个 (i, j)，v_{i,f_j}:[Batch, Dim_embedding] 与 v_{j,f_i}:[Batch, Dim_embedding] 内积 -> dot_ij:[Batch, 1]
- 交互求和：y_interaction: [Batch, 1]
- 融合：y = y_linear + y_interaction -> [Batch, 1]
- 输出：[Batch, 1] -> 预测层

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import FieldAwareEmbeddingLayer, InputMask
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class FFM(BaseRankingModel):
    @property
    def model_name(self):
        return "FFM"

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

        super(FFM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.fm_features = sparse_features + sequence_features
        if len(self.fm_features) < 2:
            raise ValueError("FFM requires at least two sparse/sequence features to build field-aware interactions.")

        self.field_aware_embedding = FieldAwareEmbeddingLayer(features=self.fm_features)

        # First-order terms for sparse/sequence features: one hot
        self.first_order_embeddings = nn.ModuleDict()
        for feature in self.fm_features:
            if feature.embedding_name in self.first_order_embeddings:
                continue
            emb = nn.Embedding(
                num_embeddings=feature.vocab_size,
                embedding_dim=1,
                padding_idx=feature.padding_idx,
            )
            self.first_order_embeddings[feature.embedding_name] = emb

        # Optional dense linear term
        self.dense_features = list(dense_features)
        dense_input_dim = sum([f.input_dim for f in self.dense_features])
        self.linear_dense = nn.Linear(dense_input_dim, 1, bias=True) if dense_input_dim > 0 else None

        self.input_mask = InputMask()

        self.embedding_params.extend(emb.weight for emb in self.field_aware_embedding.embed_dict.values())
        self.embedding_params.extend(emb.weight for emb in self.first_order_embeddings.values())
        self.register_regularization_weights(embedding_attr="field_aware_embedding", include_modules=["linear_dense"])

    def forward(self, x):
        batch_size = x[self.fm_features[0].name].size(0)
        device = x[self.fm_features[0].name].device
        y_linear = torch.zeros(batch_size, 1, device=device)  # [Batch, 1]

        # First-order dense part
        if self.linear_dense is not None:
            dense_inputs = [x[f.name].float().view(batch_size, -1) for f in self.dense_features]
            dense_stack = torch.cat(dense_inputs, dim=1) if dense_inputs else None  # [Batch, Dim_dense]
            if dense_stack is not None:
                y_linear = y_linear + self.linear_dense(dense_stack)  # [Batch, 1]

        # First-order sparse/sequence part
        first_order_terms = []
        for feature in self.fm_features:
            emb = self.first_order_embeddings[feature.embedding_name]
            if isinstance(feature, SparseFeature):
                term = emb(x[feature.name].long())  # [Batch, 1] or [Batch, 1, 1]
            else:
                seq_input = x[feature.name].long()
                if feature.max_len is not None:
                    seq_input = seq_input[:, -feature.max_len :]
                mask = self.input_mask(x, feature, seq_input).squeeze(1)  # [Batch, Seq_len]
                seq_weight = emb(seq_input).squeeze(-1)  # [Batch, Seq_len]
                term = (seq_weight * mask).sum(dim=1, keepdim=True)  # [Batch, 1]
            first_order_terms.append(term)
        if first_order_terms:
            y_linear = y_linear + torch.sum(torch.stack(first_order_terms, dim=1), dim=1)  # [Batch, 1]

        # Field-aware interactions
        field_aware_outputs = self.field_aware_embedding(
            x=x,
            source_features=self.fm_features,
            target_features=self.fm_features,
        )  # [Batch, Field_num, Field_num, Dim_embedding]
        y_interaction = torch.zeros(batch_size, 1, device=device)  # [Batch, 1]
        num_fields = len(self.fm_features)
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                v_i_fj = field_aware_outputs[:, i, j, :]  # [Batch, Dim_embedding]
                v_j_fi = field_aware_outputs[:, j, i, :]  # [Batch, Dim_embedding]
                y_interaction = y_interaction + torch.sum(v_i_fj * v_j_fi, dim=-1, keepdim=True)  # [Batch, 1]

        logits = y_linear + y_interaction  # [Batch, 1]
        return logits
