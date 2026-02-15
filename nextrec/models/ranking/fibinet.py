"""
Date: create on 09/11/2025
Checkpoint: edit on 14/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Huang T, Zhang Z, Zhang B, et al. FiBiNET: Combining feature importance and bilinear feature interaction for click-through rate prediction[C]//RecSys. 2019: 169-177.
  URL: https://arxiv.org/abs/1905.09433

Workflow:
- Embed dense+sparse+sequence features for wide branch, and compute linear logit.
- Build field-wise embeddings from sparse+sequence interaction features.
- Project fields to a shared interaction dimension when needed.
- Apply SENET to produce reweighted field embeddings.
- Compute pairwise interactions on original fields (E) and SENET fields (V).
- Flatten and concatenate E/V outputs (plus optional dense extras), then feed MLP.
- Add wide and deep logits, then apply prediction head.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Linear input: embedding(flatten all linear features) -> linear_input: [Batch, Dim_linear]
- Field embeddings: each interaction field -> [Batch, Dim_field_i] -> (optional adapter)
  -> [Batch, Dim_embedding] -> stack -> field_emb: [Batch, Field_num, Dim_embedding]
- SENET: field_emb -> senet_emb: [Batch, Field_num, Dim_embedding]
- E branch: field_emb -> out_E: [Batch, Pair_num, Dim_embedding]
- V branch: senet_emb -> out_V: [Batch, Pair_num, Dim_embedding]
- Deep input: concat(flatten(out_E), flatten(out_V), dense features)
  -> deep_input: [Batch, 2 * Pair_num * Dim_embedding + Dim_deep_extra]
- Fusion: y_linear: [Batch, 1] + y_deep: [Batch, 1] -> [Batch, 1]
- Output: [Batch, 1] -> prediction layer

where `Pair_num = Field_num * (Field_num - 1) / 2`.

FiBiNET由新浪微博机器学习团队发表于RecSys19，主要的改进方向是特征建模。
通过引入SENET机制对特征交互进行重加权，并使用双线性交互捕捉更丰富的特征关系。

流程：
- 先对 dense+sparse+sequence 特征 embedding 后走线性分支，得到线性 logit。
- 从 sparse+sequence 交互域构建域级 embedding。
- 若域维度不一致，先投影到统一交互维度。
- 用 SENET 对域 embedding 重标定，得到加权域表示。
- 在原始域（E）与 SENET 域（V）两路分别做两两交互。
- 将 E/V 交互展平并拼接（可选再拼接 dense 额外输入），送入 MLP。
- 最后将线性分支和深度分支相加，送入 prediction head 输出。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- 线性分支输入：embedding(dense特征展平) -> linear_input: [Batch, Dim_linear]
- 域 embedding：每个交互域 -> [Batch, Dim_field_i] ->（可选投影）
  -> [Batch, Dim_embedding] -> 拼接为 field_emb: [Batch, Field_num, Dim_embedding]
- SENET：field_emb -> senet_emb: [Batch, Field_num, Dim_embedding]
- E 分支：field_emb -> out_E: [Batch, Pair_num, Dim_embedding]
- V 分支：senet_emb -> out_V: [Batch, Pair_num, Dim_embedding]
- Deep 输入：concat(flatten(out_E), flatten(out_V), dense features)
  -> deep_input: [Batch, 2 * Pair_num * Dim_embedding + Dim_deep_extra]
- 融合：y_linear: [Batch, 1] + y_deep: [Batch, 1] -> [Batch, 1]
- 输出：[Batch, 1] -> 预测层

其中 `Pair_num = Field_num * (Field_num - 1) / 2`。

"""

import torch
import torch.nn as nn
from typing import Literal
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    LR,
    MLP,
    BiLinearInteractionLayer,
    EmbeddingLayer,
    HadamardInteractionLayer,
    SENETLayer,
)
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class FiBiNET(BaseModel):
    @property
    def model_name(self):
        return "FiBiNET"

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
        interaction_combo: Literal["01", "11", "10", "00"] = "11",  # "0": Hadamard, "1": Bilinear
        bilinear_type: Literal["field_all", "field_each", "field_interaction"] = "field_interaction",
        senet_reduction: int = 3,
        **kwargs,
    ):
        """
        Initialize FiBiNET model.
        初始化 FiBiNET 模型。

        Args:
            mlp_params: Parameters for interaction MLP; e.g. {"hidden_dims": [256, 128], "dropout": 0.5}.
                交互分支 MLP 参数；例如 {"hidden_dims": [256, 128], "dropout": 0.5}。
            interaction_combo: Two-char code controlling interaction operators for E/V branches.
                "0" means Hadamard, "1" means Bilinear. Default is "11" (Bilinear for both).
                两位字符串控制 E/V 两路交互算子，"0" 表示 Hadamard，"1" 表示 Bilinear，默认为 "11"（两路均 Bilinear）。
            bilinear_type: Bilinear parameter sharing mode in BilinearInteraction layer.
                双线性交互层参数共享方式。
            senet_reduction: Reduction ratio in SENET squeeze-excitation MLP.
                SENET 中 squeeze-excitation 的降维比例。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}

        super(FiBiNET, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.linear_features = dense_features + sparse_features + sequence_features
        self.deep_extra_features = dense_features
        self.interaction_features = sparse_features + sequence_features

        if len(self.interaction_features) < 2:
            raise ValueError("FiBiNET requires at least two sparse/sequence features for interactions.")

        self.embedding = EmbeddingLayer(features=self.all_features)

        self.num_fields = len(self.interaction_features)
        interaction_field_dims = []
        non_concat_dims = []
        for feature in self.interaction_features:
            if isinstance(feature, SequenceFeature) and feature.combiner == "concat":
                if feature.max_len is None:
                    raise ValueError(
                        f"FiBiNET requires SequenceFeature('{feature.name}') to set max_len when combiner='concat'."
                    )
                field_dim = feature.embedding_dim * feature.max_len
            else:
                field_dim = feature.embedding_dim
                non_concat_dims.append(feature.embedding_dim)
            interaction_field_dims.append((feature.name, field_dim))

        self.embedding_dim = non_concat_dims[0] if non_concat_dims else interaction_field_dims[0][1]
        self.interaction_adapters = nn.ModuleDict()
        for feature_name, field_dim in interaction_field_dims:
            if field_dim != self.embedding_dim:
                self.interaction_adapters[feature_name] = nn.Linear(field_dim, self.embedding_dim, bias=False)

        self.senet = SENETLayer(num_fields=self.num_fields, reduction_ratio=senet_reduction)

        self.interaction_combo = interaction_combo
        if self.interaction_combo not in {"00", "01", "10", "11"}:
            raise ValueError("interaction_combo must be one of: '00', '01', '10', '11'")

        # E interaction layers: original embeddings
        if interaction_combo[0] == "0":  # Hadamard
            self.interaction_E = HadamardInteractionLayer(num_fields=self.num_fields)  # [B, num_pairs, D]
        elif interaction_combo[0] == "1":  # Bilinear
            self.interaction_E = BiLinearInteractionLayer(
                input_dim=self.embedding_dim,
                num_fields=self.num_fields,
                bilinear_type=bilinear_type,
            )  # [B, num_pairs, D]
        else:
            raise ValueError("interaction_combo must use only '0' or '1', e.g. '00'/'01'/'10'/'11'")

        # V interaction layers: SENET reweighted embeddings
        if interaction_combo[1] == "0":
            self.interaction_V = HadamardInteractionLayer(num_fields=self.num_fields)
        elif interaction_combo[1] == "1":
            self.interaction_V = BiLinearInteractionLayer(
                input_dim=self.embedding_dim,
                num_fields=self.num_fields,
                bilinear_type=bilinear_type,
            )
        else:
            raise ValueError("interaction_combo must use only '0' or '1', e.g. '00'/'01'/'10'/'11'")

        linear_dim = self.embedding.compute_output_dim(self.linear_features)
        self.linear = LR(linear_dim)

        num_pairs = self.num_fields * (self.num_fields - 1) // 2
        interaction_dim = num_pairs * self.embedding_dim * 2
        deep_extra_dim = self.embedding.compute_output_dim(self.deep_extra_features)
        self.mlp = MLP(input_dim=interaction_dim + deep_extra_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "linear",
                "senet",
                "mlp",
                "interaction_adapters",
                "interaction_E",
                "interaction_V",
            ],
        )

    def forward(self, x):
        linear_input = self.embedding(x=x, features=self.linear_features, squeeze_dim=True)  # [Batch, Dim_linear]
        y_linear = self.linear(linear_input)  # [Batch, 1]

        field_emb_list = []
        for feature in self.interaction_features:
            feat_emb = self.embedding(x=x, features=[feature], squeeze_dim=True)  # [Batch, Dim_field_i]
            if feature.name in self.interaction_adapters:
                feat_emb = self.interaction_adapters[feature.name](feat_emb)  # [Batch, Dim_embedding]
            field_emb_list.append(feat_emb.unsqueeze(1))  # [Batch, 1, Dim_embedding]
        field_emb = torch.cat(field_emb_list, dim=1)  # [Batch, Field_num, Dim_embedding]
        senet_emb = self.senet(field_emb)  # [Batch, Field_num, Dim_embedding]

        out_E = self.interaction_E(field_emb)  # [Batch, Pair_num, Dim_embedding]

        out_V = self.interaction_V(senet_emb)  # [Batch, Pair_num, Dim_embedding]

        deep_parts = [out_E.flatten(start_dim=1), out_V.flatten(start_dim=1)]  # each: [Batch, Pair_num * Dim_embedding]
        if self.deep_extra_features:
            deep_parts.append(
                self.embedding(x=x, features=self.deep_extra_features, squeeze_dim=True)
            )  # [Batch, Dim_deep_extra]
        deep_input = torch.cat(deep_parts, dim=1)  # [Batch, 2 * Pair_num * Dim_embedding + Dim_deep_extra]

        y_deep = self.mlp(deep_input)  # [Batch, 1]

        y = y_linear + y_deep  # [Batch, 1]
        return self.prediction_layer(y)
