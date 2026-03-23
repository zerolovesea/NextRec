"""
Date: create on 09/11/2025
Author: Yang Zhou, zyaztec@gmail.com
Checkpoint: edit on 14/02/2026
Reference:
- [1] Zhou G, Mou N, Fan Y, et al. Deep interest evolution network for click-through rate prediction[C] // Proceedings of the AAAI conference on artificial intelligence. 2019, 33(01): 5941-5948. (https://arxiv.org/abs/1809.03672)

Dien (Deep Interest Evolution Network) is an improved version of DIN (Deep Interest Network).
While DIN captures the relative importance of user behaviors with respect to the target item using attention,
it treats the extracted user interests as independent and does not model their dynamic evolution.
DIEN focuses on modeling the evolution of user interests over time and introduces new network structures to capture this process.
It includes an interest extraction layer with an auxiliary loss to enhance the accuracy of interest
representation, and an interest evolution layer to better capture the dynamic changes in user interests.

The overall architecture of DIEN first embeds the categorical features and processes the sequential features.
Then it concatenates the candidate feature embedding, the result from processing the sequential features,
and the continuous features, and feeds them into an MLP for prediction.

Workflow:
- Behavior embeddings -> DynamicGRU -> interest trajectory
- Target-aware attention scores highlight behaviors relevant to the candidate
- AUGRU modulates GRU updates with attention to emphasize impactful behaviors
- Final evolved interest, candidate embedding, and context features -> MLP -> CTR

Dimension Flow:
- Input: candidate id[Batch] + behavior sequence[Batch, Length] + optional context features[Batch, Dim_context]
- Embedding: candidate -> [Batch, Dim_candidate], behavior -> [Batch, Length, Dim_behavior]
- Interest extraction: behavior embedding: [Batch, Length, Dim_behavior] -> DynamicGRU -> interest_states: [Batch, Length, Dim_hidden]
- Target attention: (candidate, interest_t) -> attention score per step -> att_weights: [Batch, Length, 1]
- Interest evolution: (interest_states, att_weights) -> AUGRU -> final_interest: [Batch, Dim_hidden]
- Fusion: concat(candidate[Batch, Dim_candidate], final_interest[Batch, Dim_hidden], other features) -> [Batch, Dim_total]
- Output: MLP -> [Batch, 1] -> prediction head

DIEN是DIN模型的改进版本。DIN模型使用基于注意力模型来捕获目标项目的相对兴趣。
但是用户的兴趣是不断进化的，而DIN抽取的用户兴趣之间是独立无关联的，没有捕获到兴趣的动态进化性。
因此DIEN更关注兴趣演化的过程，并提出了新的网络结构来建模兴趣进化的过程。
设计了兴趣抽取层，并通过计算一个辅助loss，来提升兴趣表达的准确性。
设计了兴趣进化层，来更加准确的表达用户兴趣的动态变化性。

整体结构上，先对类别特征做embedding，对序列特征处理。然后将类别特征embedding向量，
序列特征处理之后的结果，以及连续特征进行拼接，输入到mlp网路中进行预测

流程：
- 行为 embedding 输入 DynamicGRU，得到兴趣轨迹
- 目标相关的注意力得分突出关键行为
- AUGRU 用注意力调制更新，抑制无关历史噪声
- 最终演化兴趣 + 候选 embedding + 其他上下文特征，经 MLP 输出 CTR

维度变化：
- 输入：candidate id[Batch] + 行为序列[Batch, Length] + 可选上下文特征[Batch, Dim_context]
- Embedding：candidate -> [Batch, Dim_candidate]，behavior -> [Batch, Length, Dim_behavior]
- 兴趣抽取：behavior embedding: [Batch, Length, Dim_behavior] -> DynamicGRU -> interest_states: [Batch, Length, Dim_hidden]
- 目标注意力：(candidate, interest_t) -> 每步注意力分数 -> att_weights: [Batch, Length, 1]
- 兴趣演化：(interest_states, att_weights) -> AUGRU -> final_interest: [Batch, Dim_hidden]
- 融合：concat(candidate[Batch, Dim_candidate], final_interest[Batch, Dim_hidden], 其他特征) -> [Batch, Dim_total]
- 输出：MLP -> [Batch, 1] -> 预测层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import (
    MLP,
    AttentionPoolingLayer,
    EmbeddingLayer,
)
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeInput


class AUGRU(nn.Module):
    """
    Attention-aware GRU update gate used in DIEN (Zhou et al., 2019).
    Uses attention scores to weight the update of hidden states
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x, att_scores):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            att_scores: [batch_size, seq_len, 1] - attention scores
        Returns:
            output: [batch_size, seq_len, hidden_size]
            hidden: [batch_size, hidden_size] - final hidden state
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, input_size]
            att_t = att_scores[:, t, :]  # [B, 1]

            gi = F.linear(x_t, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r)  # r_t
            updategate = torch.sigmoid(i_i + h_i)  # z_t
            newgate = torch.tanh(i_n + resetgate * h_n)  # n_t

            # att_t: [B,1]，broadcast to [B,H]
            z_att = updategate * att_t

            # h_t = (1 - z'_t) * h_{t-1} + z'_t * n_t
            h = (1.0 - z_att) * h + z_att * newgate
            outputs.append(h.unsqueeze(1))

        output = torch.cat(outputs, dim=1)  # [B, L, H]
        return output, h


class DynamicGRU(nn.Module):
    """
    Dynamic GRU unit with auxiliary loss path from DIEN (Zhou et al., 2019).
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=bias,
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            lengths: [batch_size] actual sequence lengths (optional)
        Returns:
            output: [batch_size, seq_len, hidden_size]
            hidden: [batch_size, hidden_size] - final hidden state
        """
        batch_size, seq_len, _ = x.shape

        output, _ = self.gru(x)  # [B, L, H]
        if lengths is None:
            hidden = output[:, -1, :]
            return output, hidden

        lengths = lengths.long().view(-1)
        if lengths.numel() != batch_size:
            raise ValueError(
                f"[DynamicGRU Error]: lengths size ({lengths.numel()}) does not match batch_size ({batch_size})."
            )
        lengths = lengths.clamp(min=0, max=seq_len)

        seq_range = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, L]
        valid_mask = seq_range < lengths.unsqueeze(1)  # [B, L]
        output = output * valid_mask.unsqueeze(-1).to(output.dtype)

        last_index = (lengths - 1).clamp(min=0)  # [B]
        gather_index = last_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        hidden = torch.gather(output, dim=1, index=gather_index).squeeze(1)  # [B, H]
        return output, hidden


class DIEN(BaseModel):
    @property
    def model_name(self):
        return "DIEN"

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
        neg_behavior_feature_name: str | list[str] | None = None,
        mlp_params: dict | None = None,
        gru_hidden_size: int = 64,
        attention_mlp_params: dict | None = None,
        use_negsampling: bool = False,
        aux_loss_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize DIEN model.
        初始化 DIEN 模型。

        Args:
            behavior_feature_name: Name of behavior sequence feature
                (e.g. clicked item history) in `sequence_features`.
                `sequence_features` 中行为序列特征名（如点击历史）。
            candidate_feature_name: Name of candidate item feature in
                `sparse_features` (target item for scoring).
                `sparse_features` 中候选物品特征名（待打分目标物品）。
            neg_behavior_feature_name: Name of negative behavior sequence
                feature used by DIEN auxiliary loss. Required when
                `use_negsampling=True`.
                DIEN 辅助损失所用负样本行为序列特征名；
                当 `use_negsampling=True` 时必填。
            mlp_params: Parameters passed to final `MLP` tower
                (e.g. {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2, "output_dim": 1}).
                传给最终 `MLP` 预测塔的参数，如 {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.2, "output_dim": 1}。
            gru_hidden_size: Hidden dimension for both interest extraction
                `DynamicGRU` and interest evolution `AUGRU`.
                兴趣抽取 `DynamicGRU` 与兴趣演化 `AUGRU` 的隐藏维度。
            attention_mlp_params: Config for attention MLP that computes
                target-aware per-step scores over interest states. e.g.
                {"hidden_dims": [80, 40], "activation": "sigmoid", "dropout": 0.0}.
                目标感知注意力 MLP 的配置，用于计算每个时间步的注意力分数；
                例如 {"hidden_dims": [80, 40], "activation": "sigmoid", "dropout": 0.0}。
            use_negsampling: Whether to enable auxiliary next-click loss branch.
                When True, forward caches intermediate tensors and
                `compute_loss` adds auxiliary loss.
                是否启用负采样辅助分支。为 True 时，forward 会缓存中间张量，
                并在 `compute_loss` 中叠加辅助损失。
            aux_loss_weight: Coefficient for auxiliary loss in total objective:
                `total_loss = main_loss + aux_loss_weight * aux_loss`.
                辅助损失权重系数：
                `total_loss = main_loss + aux_loss_weight * aux_loss`。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        attention_mlp_params = attention_mlp_params or {}
        attention_mlp_params.setdefault("hidden_dims", [80, 40])
        attention_mlp_params.setdefault("activation", "sigmoid")
        attention_mlp_params.setdefault("dropout", 0.0)

        super(DIEN, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        self.use_negsampling = use_negsampling
        self.aux_loss_weight = float(aux_loss_weight)
        self.auxiliary_cache = None

        if len(sequence_features) == 0:
            raise ValueError("DIEN requires at least one sequence feature for user behavior history")

        if behavior_feature_name is None:
            raise ValueError("DIEN requires at least one sequence feature as behavior item feature")

        if candidate_feature_name is None:
            raise ValueError("DIEN requires at least one sparse_feature as candidate item feature")

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

        neg_behavior_feature_names: list[str] = []
        if self.use_negsampling:
            if neg_behavior_feature_name is None:
                raise ValueError("use_negsampling=True requires neg_behavior_feature_name")
            neg_behavior_feature_names = (
                [neg_behavior_feature_name]
                if isinstance(neg_behavior_feature_name, str)
                else list(neg_behavior_feature_name)
            )
            if len(neg_behavior_feature_names) == 0:
                raise ValueError("neg_behavior_feature_name must include at least one sequence feature name")
            if len(neg_behavior_feature_names) == 1 and self.num_interest_pairs > 1:
                neg_behavior_feature_names = neg_behavior_feature_names * self.num_interest_pairs
            if len(neg_behavior_feature_names) != self.num_interest_pairs:
                raise ValueError(
                    "neg_behavior_feature_name must have one entry or match behavior_feature_name length when "
                    f"use_negsampling=True, got {len(neg_behavior_feature_names)} and {self.num_interest_pairs}"
                )

        self.neg_behavior_features: list[SequenceFeature] = []
        for idx, n_name in enumerate(neg_behavior_feature_names):
            neg_behavior_feature = sequence_by_name.get(n_name)
            if neg_behavior_feature is None:
                raise ValueError(
                    f"neg_behavior_feature_name[{idx}]='{n_name}' was not found in sequence_features. "
                    f"Available sequence features: {list(sequence_by_name.keys())}"
                )
            self.neg_behavior_features.append(neg_behavior_feature)
        self.neg_behavior_feature = self.neg_behavior_features[0] if self.neg_behavior_features else None

        candidate_name_set = {f.name for f in self.candidate_features}
        self.other_sparse_features = [f for f in sparse_features if f.name not in candidate_name_set]

        excluded_sequence_names = {f.name for f in self.behavior_features}
        excluded_sequence_names.update(f.name for f in self.neg_behavior_features)
        self.other_sequence_features = [f for f in sequence_features if f.name not in excluded_sequence_names]

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        self.candidate_proj_layers = nn.ModuleList()
        self.interest_extractors = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.interest_evolutions = nn.ModuleList()
        self.auxiliary_nets = nn.ModuleList() if self.use_negsampling else None

        for idx, (behavior_feature, candidate_feature) in enumerate(
            zip(self.behavior_features, self.candidate_features)
        ):
            behavior_emb_dim = behavior_feature.embedding_dim
            if candidate_feature.embedding_dim != gru_hidden_size:
                self.candidate_proj_layers.append(nn.Linear(candidate_feature.embedding_dim, gru_hidden_size))
            else:
                self.candidate_proj_layers.append(nn.Identity())
            self.interest_extractors.append(DynamicGRU(input_size=behavior_emb_dim, hidden_size=gru_hidden_size))
            self.attention_layers.append(
                AttentionPoolingLayer(
                    embedding_dim=gru_hidden_size,
                    hidden_units=attention_mlp_params["hidden_dims"],
                    activation=attention_mlp_params["activation"],
                    use_softmax=False,
                    dropout=attention_mlp_params["dropout"],
                )
            )
            self.interest_evolutions.append(AUGRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size))
            if self.use_negsampling and self.auxiliary_nets is not None:
                self.auxiliary_nets.append(
                    nn.Sequential(
                        nn.Linear(gru_hidden_size + behavior_emb_dim, gru_hidden_size),
                        nn.PReLU(),
                        nn.Linear(gru_hidden_size, 1),
                    )
                )

        self.candidate_proj = self.candidate_proj_layers[0]
        self.interest_extractor = self.interest_extractors[0]
        self.attention_layer = self.attention_layers[0]
        self.interest_evolution = self.interest_evolutions[0]
        self.auxiliary_net = (
            self.auxiliary_nets[0] if self.auxiliary_nets is not None and len(self.auxiliary_nets) > 0 else None
        )

        mlp_input_dim = 0
        mlp_input_dim += sum(f.embedding_dim for f in self.candidate_features)
        mlp_input_dim += gru_hidden_size * self.num_interest_pairs
        mlp_input_dim += self.embedding.compute_output_dim(self.other_sparse_features)
        mlp_input_dim += self.embedding.compute_output_dim(self.other_sequence_features)
        mlp_input_dim += self.embedding.compute_output_dim(self.dense_features)

        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "interest_extractors",
                "interest_evolutions",
                "attention_layers",
                "mlp",
                "candidate_proj_layers",
                "auxiliary_nets",
            ],
        )

    def forward(self, x):
        self.auxiliary_cache = None
        other_embeddings = []
        aux_caches = []

        for idx, (behavior_feature, candidate_feature) in enumerate(
            zip(self.behavior_features, self.candidate_features)
        ):
            candidate_emb = self.embedding.embed_dict[candidate_feature.embedding_name](
                x[candidate_feature.name].long()
            )  # [Batch, Dim_candidate]
            behavior_seq = x[behavior_feature.name].long()  # [Batch, seq_len]
            behavior_emb = self.embedding.embed_dict[behavior_feature.embedding_name](
                behavior_seq
            )  # [Batch, seq_len, Dim_behavior]

            if behavior_feature.padding_idx is not None:
                mask = (behavior_seq != behavior_feature.padding_idx).unsqueeze(-1)
            else:
                mask = (behavior_seq != 0).unsqueeze(-1)
            mask = mask.float()

            seq_lengths = mask[..., 0].sum(dim=1).long()
            interest_states, _ = self.interest_extractors[idx](behavior_emb, seq_lengths)

            _, seq_len, hidden_size = interest_states.shape
            candidate_for_attention = self.candidate_proj_layers[idx](candidate_emb)

            att_scores_list = []
            for t in range(seq_len):
                concat_feat = torch.cat(
                    [
                        candidate_for_attention,
                        interest_states[:, t, :],
                        candidate_for_attention - interest_states[:, t, :],
                        candidate_for_attention * interest_states[:, t, :],
                    ],
                    dim=-1,
                )
                score_t = self.attention_layers[idx].attention_net(concat_feat)
                att_scores_list.append(score_t.unsqueeze(1))
            att_scores = torch.cat(att_scores_list, dim=1)

            scores_flat = att_scores[..., 0]
            mask_flat = mask[..., 0]
            scores_flat = scores_flat.masked_fill(mask_flat == 0, -1e9)
            att_weights = torch.softmax(scores_flat, dim=1).unsqueeze(-1)
            att_weights = att_weights * mask

            _, final_interest = self.interest_evolutions[idx](interest_states, att_weights)

            other_embeddings.append(candidate_emb)
            other_embeddings.append(final_interest)

            if self.use_negsampling and self.training:
                if idx >= len(self.neg_behavior_features):
                    raise ValueError(
                        "Negative behavior features are not configured correctly while use_negsampling=True"
                    )
                neg_behavior_feature = self.neg_behavior_features[idx]
                neg_seq = x[neg_behavior_feature.name].long()
                neg_behavior_emb = self.embedding.embed_dict[neg_behavior_feature.embedding_name](neg_seq)
                aux_caches.append(
                    {
                        "interest_states": interest_states,
                        "behavior_emb": behavior_emb,
                        "neg_behavior_emb": neg_behavior_emb,
                        "mask": mask,
                        "auxiliary_net": self.auxiliary_nets[idx] if self.auxiliary_nets is not None else None,
                    }
                )

        if self.use_negsampling and self.training:
            self.auxiliary_cache = aux_caches

        for feat in self.other_sparse_features:
            feat_emb = self.embedding.embed_dict[feat.embedding_name](x[feat.name].long())
            other_embeddings.append(feat_emb)

        for feat in self.other_sequence_features:
            seq_emb = self.embedding(x, [feat], squeeze_dim=True)
            other_embeddings.append(seq_emb)

        for feat in self.dense_features:
            val = self.embedding.project_dense(feat, x)
            other_embeddings.append(val)

        concat_input = torch.cat(other_embeddings, dim=-1)  # [Batch, total_dim]

        logits = self.mlp(concat_input)  # [Batch, 1]
        return logits

    def compute_auxiliary_loss(self):
        if not (self.training and self.use_negsampling and self.auxiliary_nets is not None):
            return torch.tensor(0.0, device=self.device)
        if not self.auxiliary_cache:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_weight = torch.tensor(0.0, device=self.device)

        for cache in self.auxiliary_cache:
            interest_states = cache["interest_states"]
            behavior_emb = cache["behavior_emb"]
            neg_behavior_emb = cache["neg_behavior_emb"]
            mask = cache["mask"]
            auxiliary_net = cache["auxiliary_net"]
            if auxiliary_net is None:
                continue

            interest_states = interest_states[:, :-1, :]
            pos_seq = behavior_emb[:, 1:, :]
            neg_seq = neg_behavior_emb[:, 1:, :]
            aux_mask = mask[:, 1:, 0]
            if aux_mask.sum() == 0:
                continue

            pos_input = torch.cat([interest_states, pos_seq], dim=-1)
            neg_input = torch.cat([interest_states, neg_seq], dim=-1)
            pos_logits = auxiliary_net(pos_input)[..., 0]
            neg_logits = auxiliary_net(neg_input)[..., 0]

            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
            aux_loss = (pos_loss + neg_loss) * aux_mask
            total_loss = total_loss + aux_loss.sum()
            total_weight = total_weight + aux_mask.sum()

        if total_weight.item() == 0.0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / total_weight.clamp(min=1.0)

    def compute_loss(self, y_pred, y_true):
        main_loss = super().compute_loss(y_pred, y_true)
        aux_loss = self.compute_auxiliary_loss()
        return main_loss + self.aux_loss_weight * aux_loss
