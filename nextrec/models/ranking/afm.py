"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of
feature interactions via attention networks[C]//IJCAI. 2017: 3119-3125.

Attentional Factorization Machine (AFM) builds on FM by learning an importance
weight for every second-order interaction instead of treating all pairs equally.
It retains FM’s linear (first-order) component for sparsity-friendly modeling,
while using an attention network to reweight the element-wise product of field
embeddings before aggregation.

In each forward pass:
  (1) Embed each field and compute pairwise element-wise products v_i ⊙ v_j
  (2) Pass interactions through an attention MLP (ReLU + projection) to score them
  (3) Softmax-normalize scores to obtain interaction weights
  (4) Weighted sum of interactions -> linear projection -> add FM first-order term

Key Advantages:
- Learns which feature pairs contribute most via attention weights
- Keeps FM efficiency and interpretability by preserving first-order terms
- Softmax-normalized reweighting reduces noise from uninformative interactions

AFM 在 FM 的二阶交互上引入注意力，为每个特征对学习重要性权重；同时保留 FM 的一阶项，
保持对稀疏特征的友好与可解释性。具体流程：
  (1) 对各字段做 embedding，并计算所有特征对的元素积 v_i ⊙ v_j
  (2) 经由注意力 MLP（ReLU + 线性映射）得到交互得分
  (3) 通过 softmax 归一化交互得分，得到权重
  (4) 将加权交互求和、线性映射，再与一阶项相加得到最终预测

主要优点：
- 注意力显式告诉哪些特征对更重要
- 保留 FM 的效率和可解释性
- softmax 归一化减弱噪声交互的影响
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, InputMask
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class AFM(BaseModel):
    @property
    def model_name(self):
        return "AFM"

    @property
    def default_task(self):
        return "binary"

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        attention_dim: int = 32,
        attention_dropout: float = 0.0,
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
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(AFM, self).__init__(
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

        self.fm_features = sparse_features + sequence_features
        if len(self.fm_features) < 2:
            raise ValueError(
                "AFM requires at least two sparse/sequence features to build pairwise interactions."
            )

        # make sure all embedding dimension are the same for FM features
        self.embedding_dim = self.fm_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.fm_features):
            raise ValueError(
                "All FM features must share the same embedding_dim for AFM."
            )

        self.embedding = EmbeddingLayer(
            features=self.fm_features
        )  # [Batch, Field, Dim ]

        # First-order terms: dense linear + one hot embeddings
        self.dense_features = list(dense_features)
        dense_input_dim = sum([f.input_dim for f in self.dense_features])
        self.linear_dense = (
            nn.Linear(dense_input_dim, 1, bias=True) if dense_input_dim > 0 else None
        )

        # First-order term: sparse/sequence features one-hot
        # **INFO**: source paper does not contain sequence features in experiments,
        # but we implement it here for completeness. if you want follow the paper strictly,
        # remove sequence features from fm_features.
        self.first_order_embeddings = nn.ModuleDict()
        for feature in self.fm_features:
            if (
                feature.embedding_name in self.first_order_embeddings
            ):  # shared embedding
                continue
            emb = nn.Embedding(
                num_embeddings=feature.vocab_size,
                embedding_dim=1,
                padding_idx=feature.padding_idx,
            )  # equal to one-hot encoding weight
            # nn.init.zeros_(emb.weight)
            self.first_order_embeddings[feature.embedding_name] = emb

        self.attention_linear = nn.Linear(self.embedding_dim, attention_dim)
        self.attention_p = nn.Linear(attention_dim, 1, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(self.embedding_dim, 1, bias=False)
        self.prediction_layer = TaskHead(task_type=self.default_task)
        self.input_mask = InputMask()

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "linear_dense",
                "attention_linear",
                "attention_p",
                "output_projection",
            ],
        )
        # add first-order embeddings to embedding regularization list
        self.embedding_params.extend(
            emb.weight for emb in self.first_order_embeddings.values()
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        field_emb = self.embedding(
            x=x, features=self.fm_features, squeeze_dim=False
        )  # [B, F, D]
        batch_size = field_emb.size(0)
        y_linear = torch.zeros(batch_size, 1, device=field_emb.device)

        # First-order dense part
        if self.linear_dense is not None:
            dense_inputs = [
                x[f.name].float().view(batch_size, -1) for f in self.dense_features
            ]
            dense_stack = torch.cat(dense_inputs, dim=1) if dense_inputs else None
            if dense_stack is not None:
                y_linear = y_linear + self.linear_dense(dense_stack)

        # First-order sparse/sequence part
        first_order_terms = []
        for feature in self.fm_features:
            emb = self.first_order_embeddings[feature.embedding_name]
            if isinstance(feature, SparseFeature):
                term = emb(x[feature.name].long())  # [B, 1]
            else:  # SequenceFeature
                seq_input = x[feature.name].long()  # [B, 1]
                if feature.max_len is not None and seq_input.size(1) > feature.max_len:
                    seq_input = seq_input[:, -feature.max_len :]
                mask = self.input_mask(x, feature, seq_input).squeeze(1)  # [B, 1]
                seq_weight = emb(seq_input).squeeze(-1)  # [B, L]
                term = (seq_weight * mask).sum(dim=1, keepdim=True)  # [B, 1]
            first_order_terms.append(term)
        if first_order_terms:
            y_linear = y_linear + torch.sum(
                torch.cat(first_order_terms, dim=1), dim=1, keepdim=True
            )

        interactions = []
        feature_values = []
        for feature in self.fm_features:
            value = x.get(f"{feature.name}_value")
            if value is not None:
                value = value.float()
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
            else:
                if isinstance(feature, SequenceFeature):
                    seq_input = x[feature.name].long()
                    if (
                        feature.max_len is not None
                        and seq_input.size(1) > feature.max_len
                    ):
                        seq_input = seq_input[:, -feature.max_len :]
                    value = self.input_mask(x, feature, seq_input).sum(dim=2)  # [B, 1]
                else:
                    value = torch.ones(batch_size, 1, device=field_emb.device)
            feature_values.append(value)
        feature_values_tensor = torch.cat(feature_values, dim=1).unsqueeze(
            -1
        )  # [B, F, 1]
        field_emb = field_emb * feature_values_tensor

        num_fields = field_emb.shape[1]
        for i in range(num_fields - 1):
            vi = field_emb[:, i, :]
            for j in range(i + 1, num_fields):
                vj = field_emb[:, j, :]
                interactions.append(vi * vj)

        pair_tensor = torch.stack(interactions, dim=1)  # [B, num_pairs, D]
        attention_scores = torch.relu(self.attention_linear(pair_tensor))
        attention_scores = self.attention_p(attention_scores)  # [B, num_pairs, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_sum = torch.sum(attention_weights * pair_tensor, dim=1)
        weighted_sum = self.attention_dropout(weighted_sum)
        y_afm = self.output_projection(weighted_sum)

        y = y_linear + y_afm
        return self.prediction_layer(y)
