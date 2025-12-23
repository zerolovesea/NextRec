"""
Date: create on 09/11/2025
Author: Yang Zhou, zyaztec@gmail.com
Checkpoint: edit on 09/12/2025
Reference:
[1] Zhou G, Mou N, Fan Y, et al. Deep interest evolution network for click-through
rate prediction[C] // Proceedings of the AAAI conference on artificial intelligence.
2019, 33(01): 5941-5948. (https://arxiv.org/abs/1809.03672)

DIEN is a CTR prediction model that explicitly models how user interests evolve
over time. It introduces a two-stage pipeline:
  (1) Interest Extraction: a GRU encodes raw behavior sequences into interest states
  (2) Interest Evolution: an attention-aware GRU (AUGRU) updates interests by
      focusing on behaviors most related to the target item
An auxiliary loss on next-click prediction guides the GRU to learn finer-grained
interest transitions and alleviates vanishing signals in long sequences.

Processing flow:
  - Behavior embeddings -> DynamicGRU -> interest trajectory
  - Target-aware attention scores highlight behaviors relevant to the candidate
  - AUGRU modulates GRU updates with attention to emphasize impactful behaviors
  - Final evolved interest, candidate embedding, and context features -> MLP -> CTR

Key advantages:
- Captures temporal evolution of user interests instead of a static summary
- Target-aware attention steers the evolution toward the candidate item
- AUGRU gates mitigate noise from irrelevant historical behaviors
- Auxiliary loss provides additional supervision for sequential dynamics

DIEN 是一个 CTR 预估模型，用于显式建模用户兴趣的时间演化。核心包含两阶段：
  (1) 兴趣抽取：通过 GRU 将原始行为序列编码为兴趣状态轨迹
  (2) 兴趣演化：利用目标感知的注意力门控 GRU（AUGRU），强调与候选目标相关的行为，
      引导兴趣随时间更新
同时引入针对下一个行为点击的辅助损失，缓解长序列信号衰减并强化兴趣转移学习。

流程概览：
  - 行为 embedding 输入 DynamicGRU，得到兴趣轨迹
  - 目标相关的注意力得分突出关键行为
  - AUGRU 用注意力调制更新，抑制无关历史噪声
  - 最终演化兴趣 + 候选 embedding + 其他上下文特征，经 MLP 输出 CTR

主要优点：
- 建模兴趣随时间的演化，而非静态聚合
- 目标感知注意力将兴趣演化对齐到候选物品
- AUGRU 门控削弱无关行为的干扰
- 辅助损失为序列动态提供额外监督信号
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
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class AUGRU(nn.Module):
    """Attention-aware GRU update gate used in DIEN (Zhou et al., 2019)."""

    """
    Attention-based GRU for DIEN
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
    """Dynamic GRU unit with auxiliary loss path from DIEN (Zhou et al., 2019)."""

    """
    GRU with dynamic routing for DIEN
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU parameters
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

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
            hidden: [batch_size, hidden_size] - final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]

            # GRU computation
            gi = F.linear(x_t, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r)
            updategate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)

            # h_t = (1 - z_t) * h_{t-1} + z_t * n_t
            h = (1.0 - updategate) * h + updategate * newgate

            outputs.append(h.unsqueeze(1))

        output = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        return output, h


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
        behavior_feature_name: str | None = None,
        candidate_feature_name: str | None = None,
        neg_behavior_feature_name: str | None = None,
        mlp_params: dict | None = None,
        gru_hidden_size: int = 64,
        attention_hidden_units: list[int] | None = None,
        attention_activation: str = "sigmoid",
        use_negsampling: bool = False,
        aux_loss_weight: float = 1.0,
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
        mlp_params = mlp_params or {}
        attention_hidden_units = attention_hidden_units or [80, 40]
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"

        super(DIEN, self).__init__(
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
        self.use_negsampling = use_negsampling
        self.aux_loss_weight = float(aux_loss_weight)
        self.auxiliary_cache = None

        if len(sequence_features) == 0:
            raise ValueError(
                "DIEN requires at least one sequence feature for user behavior history"
            )

        if behavior_feature_name is None:
            raise ValueError(
                "DIEN requires at least one sequence feature as behavior item feature"
            )

        if candidate_feature_name is None:
            raise ValueError(
                "DIEN requires at least one sparse_feature as candidate item feature"
            )

        self.behavior_feature = [
            f for f in sequence_features if f.name == behavior_feature_name
        ][0]
        self.candidate_feature = [
            f for f in sparse_features if f.name == candidate_feature_name
        ][0]

        self.other_sparse_features = (
            sparse_features[:-1] if self.candidate_feature else sparse_features
        )

        self.neg_behavior_feature = None

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        behavior_emb_dim = self.behavior_feature.embedding_dim

        # projection candidate feature to match GRU hidden size if needed
        self.candidate_proj = None
        if (
            self.candidate_feature is not None
            and self.candidate_feature.embedding_dim != gru_hidden_size
        ):
            self.candidate_proj = nn.Linear(
                self.candidate_feature.embedding_dim, gru_hidden_size
            )

        # gru for interest extraction
        self.interest_extractor = DynamicGRU(
            input_size=behavior_emb_dim, hidden_size=gru_hidden_size
        )

        self.attention_layer = AttentionPoolingLayer(
            embedding_dim=gru_hidden_size,
            hidden_units=attention_hidden_units,
            activation=attention_activation,
            use_softmax=False,
        )

        # Interest Evolution Layer (AUGRU)
        self.interest_evolution = AUGRU(
            input_size=gru_hidden_size, hidden_size=gru_hidden_size
        )

        # build auxiliary loss net if provided neg sampling and neg_behavior_feature_name
        # auxiliary loss uses the interest states to predict the next behavior in the sequence
        # that's the second task of DIEN
        if self.use_negsampling:
            neg_candidates = [
                f for f in sequence_features if f.name == neg_behavior_feature_name
            ]
            if len(neg_candidates) == 0:
                raise ValueError(
                    f"use_negsampling=True requires a negative sequence feature named '{neg_behavior_feature_name}'"
                )
            self.neg_behavior_feature = neg_candidates[0]
            self.auxiliary_net = nn.Sequential(
                nn.Linear(gru_hidden_size + behavior_emb_dim, gru_hidden_size),
                nn.PReLU(),
                nn.Linear(gru_hidden_size, 1),
            )
        else:
            self.auxiliary_net = None

        mlp_input_dim = 0
        if self.candidate_feature:
            mlp_input_dim += self.candidate_feature.embedding_dim
        mlp_input_dim += gru_hidden_size  # final interest state
        mlp_input_dim += sum([f.embedding_dim for f in self.other_sparse_features])
        mlp_input_dim += sum(
            [getattr(f, "embedding_dim", 1) or 1 for f in dense_features]
        )

        self.mlp = MLP(input_dim=mlp_input_dim, **mlp_params)
        self.prediction_layer = TaskHead(task_type=self.task)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "interest_extractor",
                "interest_evolution",
                "attention_layer",
                "mlp",
                "candidate_proj",
                "auxiliary_net",
            ],
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        self.auxiliary_cache = None
        if self.candidate_feature:
            candidate_emb = self.embedding.embed_dict[
                self.candidate_feature.embedding_name
            ](
                x[self.candidate_feature.name].long()
            )  # [B, emb_dim]
        else:
            raise ValueError("DIEN requires a candidate item feature")

        behavior_seq = x[self.behavior_feature.name].long()  # [B, seq_len]
        behavior_emb = self.embedding.embed_dict[self.behavior_feature.embedding_name](
            behavior_seq
        )  # [B, seq_len, emb_dim]

        if self.behavior_feature.padding_idx is not None:
            mask = (behavior_seq != self.behavior_feature.padding_idx).unsqueeze(-1)
        else:
            mask = (behavior_seq != 0).unsqueeze(-1)
        mask = mask.float()  # [B, seq_len, 1]

        interest_states, _ = self.interest_extractor(
            behavior_emb
        )  # [B, seq_len, hidden_size]

        batch_size, seq_len, hidden_size = interest_states.shape

        if self.candidate_proj is not None:
            candidate_for_attention = self.candidate_proj(candidate_emb)
        else:
            candidate_for_attention = candidate_emb  # [B, hidden_size]
        att_scores_list = []
        for t in range(seq_len):
            # [B, 4H]
            concat_feat = torch.cat(
                [
                    candidate_for_attention,
                    interest_states[:, t, :],
                    candidate_for_attention - interest_states[:, t, :],
                    candidate_for_attention * interest_states[:, t, :],
                ],
                dim=-1,
            )
            score_t = self.attention_layer.attention_net(concat_feat)  # [B, 1]
            att_scores_list.append(score_t)

        # [B, seq_len, 1]
        att_scores = torch.cat(att_scores_list, dim=1)

        scores_flat = att_scores.squeeze(-1)  # [B, seq_len]
        mask_flat = mask.squeeze(-1)  # [B, seq_len]

        scores_flat = scores_flat.masked_fill(mask_flat == 0, -1e9)
        att_weights = torch.softmax(scores_flat, dim=1)  # [B, seq_len]
        att_weights = att_weights.unsqueeze(-1)  # [B, seq_len, 1]

        att_weights = att_weights * mask

        # 6. Interest Evolution（AUGRU）
        final_states, final_interest = self.interest_evolution(
            interest_states, att_weights
        )  # final_interest: [B, hidden_size]

        if self.use_negsampling and self.training:
            if self.neg_behavior_feature is None:
                raise ValueError(
                    "Negative behavior feature is not configured while use_negsampling=True"
                )
            neg_seq = x[self.neg_behavior_feature.name].long()
            neg_behavior_emb = self.embedding.embed_dict[
                self.neg_behavior_feature.embedding_name
            ](neg_seq)
            self.auxiliary_cache = {
                "interest_states": interest_states,
                "behavior_emb": behavior_emb,
                "neg_behavior_emb": neg_behavior_emb,
                "mask": mask,
            }

        other_embeddings = []
        other_embeddings.append(candidate_emb)
        other_embeddings.append(final_interest)

        for feat in self.other_sparse_features:
            feat_emb = self.embedding.embed_dict[feat.embedding_name](
                x[feat.name].long()
            )
            other_embeddings.append(feat_emb)

        for feat in self.dense_features:
            val = x[feat.name].float()
            if val.dim() == 1:
                val = val.unsqueeze(1)
            other_embeddings.append(val)

        concat_input = torch.cat(other_embeddings, dim=-1)  # [B, total_dim]

        y = self.mlp(concat_input)  # [B, 1]
        return self.prediction_layer(y)

    def compute_auxiliary_loss(self):
        if not (self.training and self.use_negsampling and self.auxiliary_net):
            return torch.tensor(0.0, device=self.device)
        if self.auxiliary_cache is None:
            return torch.tensor(0.0, device=self.device)

        interest_states = self.auxiliary_cache["interest_states"]
        behavior_emb = self.auxiliary_cache["behavior_emb"]
        neg_behavior_emb = self.auxiliary_cache["neg_behavior_emb"]
        mask = self.auxiliary_cache["mask"]

        interest_states = interest_states[:, :-1, :]
        pos_seq = behavior_emb[:, 1:, :]
        neg_seq = neg_behavior_emb[:, 1:, :]
        aux_mask = mask[:, 1:, :].squeeze(-1)

        if aux_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        pos_input = torch.cat([interest_states, pos_seq], dim=-1)
        neg_input = torch.cat([interest_states, neg_seq], dim=-1)
        pos_logits = self.auxiliary_net(pos_input).squeeze(-1)
        neg_logits = self.auxiliary_net(neg_input).squeeze(-1)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction="none"
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction="none"
        )
        aux_loss = (pos_loss + neg_loss) * aux_mask
        aux_loss = aux_loss.sum() / torch.clamp(aux_mask.sum(), min=1.0)
        return aux_loss

    def compute_loss(self, y_pred, y_true):
        main_loss = super().compute_loss(y_pred, y_true)
        aux_loss = self.compute_auxiliary_loss()
        return main_loss + self.aux_loss_weight * aux_loss
