"""
Date: create on 09/11/2025
Checkpoint: edit on 14/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C] //Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018: 1059-1068.
  URL: https://arxiv.org/abs/1706.06978

Deep Interest Network (DIN) was proposed by Alibaba's e-commerce team in KDD18.
The core idea is to introduce the attention mechanism into user interest modeling.
In traditional CTR models, a user's historical behaviors are often pooled into a static interest vector,
which may lose important information relevant to the current candidate item.
DIN computes attention scores between each historical click's item ID and the current candidate item ID,
dynamically constructing an interest representation specific to the current candidate item.

Workflow:
- Embed candidate item, user behavior sequence, and other sparse/dense fields
- Use a small attention MLP to score each historical behavior against the
      candidate embedding
- Apply masked weighted pooling to obtain a target-specific interest vector
- Concatenate candidate, interest vector, other sparse embeddings, and dense
      features
- Feed the combined representation into an MLP for final prediction

Dimension Flow:
- Input: candidate id[Batch] + behavior sequence[Batch, Length] + optional context features[Batch, Dim_context]
- Embedding: candidate -> [Batch, Dim_candidate], behavior -> [Batch, Length, Dim_behavior]
- Target attention: (candidate, behavior_t) -> per-step score -> att_weights: [Batch, Length, 1]
- Interest pooling: weighted sum over behavior sequence -> interest: [Batch, Dim_behavior]
- Fusion: concat(candidate, interest, other sparse/sequence/dense features) -> [Batch, Dim_total]
- Output: MLP -> [Batch, 1] -> prediction head

DIN 由阿里巴巴电商团队发表在KDD18，核心思想是把attention机制引入了到用户兴趣建模上。
在传统的CTR模型中，用户的历史行为通常被池化成一个静态的兴趣向量，这可能会丢失与当前候选物品相关的重要信息。
DIN 通过对用户的历史行为序列进行注意力计算，将每个历史点击的物品id与当前候选物品id计算相关性评分，
从而动态地构建一个针对当前候选物品的兴趣表示。

流程：
- 对候选物品、用户行为序列及其他稀疏/稠密特征做 embedding
- 使用小型注意力 MLP 计算每个历史行为与候选 embedding 的相关性
- 通过掩码加权池化得到目标特定的兴趣向量
- 拼接候选、兴趣向量、其他稀疏 embedding 与稠密特征
- 输入 MLP 完成最终点击率预测

维度变化：
- 输入：candidate id[Batch] + 行为序列[Batch, Length] + 可选上下文特征[Batch, Dim_context]
- Embedding：candidate -> [Batch, Dim_candidate]，behavior -> [Batch, Length, Dim_behavior]
- 目标注意力：(candidate, behavior_t) -> 每步注意力分数 -> att_weights: [Batch, Length, 1]
- 兴趣池化：对行为序列做加权和 -> interest: [Batch, Dim_behavior]
- 融合：concat(candidate, interest, 其他稀疏/序列/稠密特征) -> [Batch, Dim_total]
- 输出：MLP -> [Batch, 1] -> 预测层

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    MLP,
    AttentionPoolingLayer,
    EmbeddingLayer,
)
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.types import TaskTypeInput


class DIN(BaseRankingModel):
    @property
    def model_name(self):
        return "DIN"

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
        behavior_feature_name: str | list[str] | None = None,
        candidate_feature_name: str | list[str] | None = None,
        mlp_params: dict | None = None,
        attention_mlp_params: dict | None = None,
        attention_use_softmax: bool = True,
        **kwargs,
    ):
        """
        Initialize DIN model.
        初始化 DIN 模型。

        Args:
            behavior_feature_name: Behavior sequence feature name(s) in
                `sequence_features`, e.g. click history.
                `sequence_features` 中的行为序列特征名（如点击历史）。
            candidate_feature_name: Candidate item feature name(s) in
                `sparse_features`, used as attention query.
                `sparse_features` 中候选物品特征名，作为 attention query。
            mlp_params: Parameters for final prediction MLP, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2}.
                最终预测 MLP 参数，如
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.2}。
            attention_mlp_params: Config for attention scoring MLP. Default
                hidden dims are [80, 40], default activation is "dice",
                default dropout is 0.0.
                注意力打分 MLP 参数，默认 hidden_dims=[80,40]、activation="dice"、
                dropout=0.0。
            attention_use_softmax: Whether to normalize attention scores across
                sequence length using softmax. If False, uses sigmoid gating.
                是否在序列维度使用 softmax 归一化注意力分数；否则使用 sigmoid 门控。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        attention_mlp_params = attention_mlp_params or {}
        attention_mlp_params.setdefault("hidden_dims", [80, 40])
        attention_mlp_params.setdefault("activation", "dice")
        attention_mlp_params.setdefault("dropout", 0.0)

        super(DIN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # DIN requires: user behavior sequence + candidate item + other features
        if len(sequence_features) == 0:
            raise ValueError("DIN requires at least one sequence feature for user behavior history")
        if behavior_feature_name is None:
            raise ValueError("DIN requires an explicit behavior_feature_name")

        if candidate_feature_name is None:
            raise ValueError("DIN requires an explicit candidate_feature_name")

        behavior_feature_names = (
            [behavior_feature_name] if isinstance(behavior_feature_name, str) else list(behavior_feature_name)
        )
        candidate_feature_names = (
            [candidate_feature_name] if isinstance(candidate_feature_name, str) else list(candidate_feature_name)
        )
        if len(behavior_feature_names) == 0:
            raise ValueError("behavior_feature_name must include at least one sequence feature name")
        if len(candidate_feature_names) == 0:
            raise ValueError("candidate_feature_name must include at least one sparse feature name")
        if len(behavior_feature_names) != len(candidate_feature_names):
            raise ValueError(
                "behavior_feature_name and candidate_feature_name must have the same number of entries, "
                f"got {len(behavior_feature_names)} and {len(candidate_feature_names)}"
            )

        sequence_by_name = {f.name: f for f in sequence_features}
        sparse_by_name = {f.name: f for f in sparse_features}
        self.behavior_features: list[SequenceFeature] = []
        self.candidate_features: list[SparseFeature] = []
        for idx, (b_name, c_name) in enumerate(zip(behavior_feature_names, candidate_feature_names)):
            behavior_feature = sequence_by_name.get(b_name)
            if behavior_feature is None:
                raise ValueError(
                    f"behavior_feature_name[{idx}]='{b_name}' was not found in sequence_features. "
                    f"Available sequence features: {list(sequence_by_name.keys())}"
                )
            candidate_feature = sparse_by_name.get(c_name)
            if candidate_feature is None:
                raise ValueError(
                    f"candidate_feature_name[{idx}]='{c_name}' was not found in sparse_features. "
                    f"Available sparse features: {list(sparse_by_name.keys())}"
                )
            self.behavior_features.append(behavior_feature)
            self.candidate_features.append(candidate_feature)

        self.behavior_feature = self.behavior_features[0]
        self.candidate_feature = self.candidate_features[0]
        self.num_interest_pairs = len(self.behavior_features)

        candidate_name_set = {f.name for f in self.candidate_features}
        behavior_name_set = {f.name for f in self.behavior_features}
        # Other sparse features
        self.other_sparse_features = [f for f in sparse_features if f.name not in candidate_name_set]
        # Other sequence features (excluding behavior sequences used by attention)
        self.other_sequence_features = [f for f in sequence_features if f.name not in behavior_name_set]

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Attention layers for each (behavior, candidate) pair
        self.candidate_attention_proj_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        for behavior_feature, candidate_feature in zip(self.behavior_features, self.candidate_features):
            behavior_emb_dim = behavior_feature.embedding_dim
            if candidate_feature.embedding_dim != behavior_emb_dim:
                self.candidate_attention_proj_layers.append(
                    nn.Linear(candidate_feature.embedding_dim, behavior_emb_dim)
                )
            else:
                self.candidate_attention_proj_layers.append(nn.Identity())
            self.attention_layers.append(
                AttentionPoolingLayer(
                    embedding_dim=behavior_emb_dim,
                    hidden_units=attention_mlp_params["hidden_dims"],
                    activation=attention_mlp_params["activation"],
                    use_softmax=attention_use_softmax,
                    dropout=attention_mlp_params["dropout"],
                )
            )

        self.candidate_attention_proj = self.candidate_attention_proj_layers[0]
        self.attention = self.attention_layers[0]

        # Calculate MLP input dimension
        # candidates + attention_pooled_behaviors + other_sparse + other_sequence + dense
        mlp_input_dim = 0
        mlp_input_dim += sum(f.embedding_dim for f in self.candidate_features)
        mlp_input_dim += sum(f.embedding_dim for f in self.behavior_features)
        mlp_input_dim += self.embedding.compute_output_dim(self.other_sparse_features)
        mlp_input_dim += self.embedding.compute_output_dim(self.other_sequence_features)
        mlp_input_dim += self.embedding.compute_output_dim(self.dense_features)

        # MLP for final prediction
        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)

        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["attention_layers", "mlp", "candidate_attention_proj_layers"],
        )

    def forward(self, x):
        other_embeddings = []
        for idx, (behavior_feature, candidate_feature) in enumerate(
            zip(self.behavior_features, self.candidate_features)
        ):
            # candidate_emb: [Batch, Dim_candidate]
            candidate_emb = self.embedding.embed_dict[candidate_feature.embedding_name](
                x[candidate_feature.name].long()
            )
            # behavior_seq: [Batch, Length], behavior_emb: [Batch, Length, Dim_behavior]
            behavior_seq = x[behavior_feature.name].long()
            behavior_emb = self.embedding.embed_dict[behavior_feature.embedding_name](behavior_seq)
            if behavior_feature.padding_idx is not None:
                # mask: [Batch, Length, 1], 1 for valid token, 0 for padding
                mask = (behavior_seq != behavior_feature.padding_idx).unsqueeze(-1).float()
            else:
                mask = (behavior_seq != 0).unsqueeze(-1).float()
            # candidate_query: [Batch, Dim_behavior] after optional projection
            candidate_query = self.candidate_attention_proj_layers[idx](candidate_emb)
            # pooled_behavior: [Batch, Dim_behavior]
            pooled_behavior = self.attention_layers[idx](query=candidate_query, keys=behavior_emb, mask=mask)
            other_embeddings.append(candidate_emb)
            other_embeddings.append(pooled_behavior)

        # Other sparse features
        for feat in self.other_sparse_features:
            feat_emb = self.embedding.embed_dict[feat.embedding_name](x[feat.name].long())
            other_embeddings.append(feat_emb)

        # Other sequence features
        for feat in self.other_sequence_features:
            seq_emb = self.embedding(x, [feat], squeeze_dim=True)
            other_embeddings.append(seq_emb)

        # Dense features
        for feat in self.dense_features:
            dense_val = self.embedding.project_dense(feat, x)
            other_embeddings.append(dense_val)

        # Concatenate all features -> concat_input: [Batch, Dim_total]
        concat_input = torch.cat(other_embeddings, dim=-1)

        logits = self.mlp(concat_input)  # [Batch, 1]
        return logits
