"""
Date: create on 19/12/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Reference:
[1] Juan Y, Zhuang Y, Chin W-S, et al. Field-aware Factorization Machines for CTR
Prediction[C]//RecSys. 2016: 43-50.

Field-aware Factorization Machines (FFM) extend FM by learning a distinct
embedding of each feature for every target field. For a pair of fields (i, j),
FFM uses v_{i,f_j} · v_{j,f_i} instead of a shared embedding, enabling richer
context-aware interactions and stronger CTR performance on sparse categorical
data.

Pipeline:
  (1) Build field-aware embeddings v_{i,f} for each feature i toward every field f
  (2) Compute first-order linear terms for sparse/sequence (and optional dense) fields
  (3) For each field pair (i, j), compute v_{i,f_j} · v_{j,f_i}
  (4) Sum linear + field-aware interaction logits and output prediction

Key Advantages:
- Field-aware embeddings capture asymmetric interactions between fields
- Improves CTR accuracy on sparse categorical features
- Retains interpretable second-order structure

FFM 在 FM 基础上引入字段感知机制：每个特征在不同目标字段下拥有不同的 embedding。
对于字段对 (i, j)，模型使用 v_{i,f_j} 与 v_{j,f_i} 的内积，从而更细粒度地建模
跨字段交互，在稀疏高维 CTR 场景中表现更优。

处理流程：
  (1) 为每个特征 i 构造面向每个字段 f 的嵌入 v_{i,f}
  (2) 计算一阶线性项（稀疏/序列特征，及可选的稠密特征）
  (3) 对每一对字段 (i, j) 计算 v_{i,f_j} · v_{j,f_i}
  (4) 将线性项与交互项相加得到最终预测

主要优点：
- 字段感知嵌入可捕捉非对称交互
- 稀疏类别特征下预测更准确
- 保持二阶结构的可解释性
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import AveragePooling, InputMask, SumPooling
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.torch_utils import get_initializer


class FFM(BaseModel):
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

        super(FFM, self).__init__(
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
                "FFM requires at least two sparse/sequence features to build field-aware interactions."
            )

        self.embedding_dim = self.fm_features[0].embedding_dim
        if any(f.embedding_dim != self.embedding_dim for f in self.fm_features):
            raise ValueError(
                "All FFM features must share the same embedding_dim for field-aware interactions."
            )
        for feature in self.fm_features:
            if isinstance(feature, SequenceFeature) and feature.combiner == "concat":
                raise ValueError(
                    "FFM does not support SequenceFeature with combiner='concat' because it breaks shared embedding_dim."
                )

        self.field_aware_embeddings = nn.ModuleDict()
        for src_feature in self.fm_features:
            for target_field in self.fm_features:
                key = self.field_aware_key(src_feature, target_field)
                if key in self.field_aware_embeddings:
                    continue
                self.field_aware_embeddings[key] = self.build_embedding(src_feature)

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
        self.linear_dense = (
            nn.Linear(dense_input_dim, 1, bias=True) if dense_input_dim > 0 else None
        )

        self.prediction_layer = TaskHead(task_type=self.task)
        self.input_mask = InputMask()
        self.mean_pool = AveragePooling()
        self.sum_pool = SumPooling()

        self.embedding_params.extend(
            emb.weight for emb in self.field_aware_embeddings.values()
        )
        self.embedding_params.extend(
            emb.weight for emb in self.first_order_embeddings.values()
        )
        self.register_regularization_weights(
            embedding_attr="field_aware_embeddings", include_modules=["linear_dense"]
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def field_aware_key(
        self, src_feature: SparseFeature | SequenceFeature, target_field
    ) -> str:
        return f"{src_feature.embedding_name}__to__{target_field.name}"

    def build_embedding(self, feature: SparseFeature | SequenceFeature) -> nn.Embedding:
        if getattr(feature, "pretrained_weight", None) is not None:
            weight = feature.pretrained_weight
            if weight is None:
                raise ValueError(
                    f"[FFM Error]: Pretrained weight for '{feature.embedding_name}' is None."
                )
            if weight.shape != (feature.vocab_size, feature.embedding_dim):
                raise ValueError(
                    f"[FFM Error]: Pretrained weight for '{feature.embedding_name}' has shape {weight.shape}, expected ({feature.vocab_size}, {feature.embedding_dim})."
                )
            embedding = nn.Embedding.from_pretrained(
                embeddings=weight,
                freeze=feature.freeze_pretrained,
                padding_idx=feature.padding_idx,
            )
            embedding.weight.requires_grad = (
                feature.trainable and not feature.freeze_pretrained
            )
        else:
            embedding = nn.Embedding(
                num_embeddings=feature.vocab_size,
                embedding_dim=feature.embedding_dim,
                padding_idx=feature.padding_idx,
            )
            embedding.weight.requires_grad = feature.trainable
            initialization = get_initializer(
                init_type=feature.init_type,
                activation="linear",
                param=feature.init_params,
            )
            initialization(embedding.weight)
        return embedding

    def embed_for_field(
        self,
        feature: SparseFeature | SequenceFeature,
        target_field,
        x: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        key = self.field_aware_key(feature, target_field)
        emb = self.field_aware_embeddings[key]
        if isinstance(feature, SparseFeature):
            return emb(x[feature.name].long())

        seq_input = x[feature.name].long()
        if feature.max_len is not None and seq_input.size(1) > feature.max_len:
            seq_input = seq_input[:, -feature.max_len :]
        seq_emb = emb(seq_input)  # [B, L, D]
        mask = self.input_mask(x, feature, seq_input)
        if feature.combiner == "mean":
            return self.mean_pool(seq_emb, mask)
        if feature.combiner == "sum":
            return self.sum_pool(seq_emb, mask)
        raise ValueError(
            f"[FFM Error]: Unsupported combiner '{feature.combiner}' for sequence feature '{feature.name}'."
        )

    def forward(self, x):
        batch_size = x[self.fm_features[0].name].size(0)
        device = x[self.fm_features[0].name].device
        y_linear = torch.zeros(batch_size, 1, device=device)

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
            else:
                seq_input = x[feature.name].long()
                if feature.max_len is not None and seq_input.size(1) > feature.max_len:
                    seq_input = seq_input[:, -feature.max_len :]
                mask = self.input_mask(x, feature, seq_input).squeeze(1)  # [B, L]
                seq_weight = emb(seq_input).squeeze(-1)  # [B, L]
                term = (seq_weight * mask).sum(dim=1, keepdim=True)  # [B, 1]
            first_order_terms.append(term)
        if first_order_terms:
            y_linear = y_linear + torch.sum(
                torch.stack(first_order_terms, dim=1), dim=1
            )

        # Field-aware interactions
        y_interaction = torch.zeros(batch_size, 1, device=device)
        num_fields = len(self.fm_features)
        for i in range(num_fields - 1):
            feature_i = self.fm_features[i]
            for j in range(i + 1, num_fields):
                feature_j = self.fm_features[j]
                v_i_fj = self.embed_for_field(feature_i, feature_j, x)
                v_j_fi = self.embed_for_field(feature_j, feature_i, x)
                y_interaction = y_interaction + torch.sum(
                    v_i_fj * v_j_fi, dim=1, keepdim=True
                )

        y = y_linear + y_interaction
        return self.prediction_layer(y)
