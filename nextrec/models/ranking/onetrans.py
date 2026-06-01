"""
Date: create on 17/05/2026
Checkpoint: edit on 17/05/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender.

OneTrans uses one Transformer-style backbone to jointly model behavior sequences
and non-sequential ranking features. In this implementation, sequence features
are encoded as S-tokens, while dense and sparse features are compressed into
NS-tokens with either auto-split or group-wise tokenization. Stacked mixed causal
blocks then perform sequence modeling and feature interaction in the same token
stream, followed by a DNN prediction head.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Sequence tokenizer: each sequence feature -> raw embeddings -> projection -> S-tokens [Batch, Seq_tokens, Dim_model]
- NS tokenizer: dense/sparse embeddings -> token compression -> NS-tokens [Batch, NS_tokens, Dim_model]
- OneTrans stream: concat(S-tokens, NS-tokens) -> mixed causal blocks -> pyramid shrink on S-token prefix
- Pooling: masked mean over final S-tokens and NS-tokens -> [Batch, 2 * Dim_model]
- Output: DNN -> [Batch, 1] -> prediction layer

OneTrans 将行为序列和非序列排序特征放入同一个 Transformer 主干中建模。
当前实现中，sequence features 作为 S-token 流，dense/sparse features 经过
auto-split 或 group-wise 压缩后作为 NS-token。Mixed causal block 同时完成序列建模
和特征交互，最后通过 DNN 输出排序 logit。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- 序列 tokenizer：每个 sequence feature -> raw embedding -> 投影 -> S-tokens [Batch, Seq_tokens, Dim_model]
- NS tokenizer：dense/sparse embedding -> token 压缩 -> NS-tokens [Batch, NS_tokens, Dim_model]
- OneTrans 流：拼接 S-token 与 NS-token -> mixed causal blocks -> 逐层收缩 S-token 前缀
- Pooling：对最终 S-token 与 NS-token 分别做 masked mean -> [Batch, 2 * Dim_model]
- 输出：DNN -> [Batch, 1] -> 预测层
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP, RMSNorm
from nextrec.models.ranking.base import BaseRankingModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import TaskTypeInput


class OneTransFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixedCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"[OneTrans Error] d_model({d_model}) must be divisible by num_heads({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.shared_qkv = nn.Linear(d_model, d_model * 3)
        self.ns_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_ns_tokens)])
        self.ns_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_ns_tokens)])
        self.ns_v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_ns_tokens)])
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        q, k, v = self.shared_qkv(tokens).chunk(3, dim=-1)
        q = q.clone()
        k = k.clone()
        v = v.clone()

        # NS tokens use position-specific projections; S tokens share one projection.
        for ns_index, (q_proj, k_proj, v_proj) in enumerate(zip(self.ns_q, self.ns_k, self.ns_v, strict=True)):
            position = seq_token_count + ns_index
            if position >= token_count:
                continue
            token = tokens[:, position, :]
            q[:, position, :] = q_proj(token)
            k[:, position, :] = k_proj(token)
            v[:, position, :] = v_proj(token)

        q = q.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(token_count, token_count, device=tokens.device, dtype=torch.bool))
        key_mask = ~padding_mask.bool()
        valid_mask = causal_mask.view(1, 1, token_count, token_count) & key_mask.view(batch_size, 1, 1, token_count)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, token_count, self.d_model)
        output = self.out(output)
        return output * key_mask.unsqueeze(-1)


class MixedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.shared = OneTransFeedForward(d_model, hidden_mult, dropout)
        self.ns_specific = nn.ModuleList(
            [OneTransFeedForward(d_model, hidden_mult, dropout) for _ in range(num_ns_tokens)]
        )

    def forward(self, tokens: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        output = self.shared(tokens)
        for ns_index, ffn in enumerate(self.ns_specific):
            position = seq_token_count + ns_index
            if position >= tokens.size(1):
                continue
            output[:, position, :] = ffn(tokens[:, position, :])
        return output


class OneTransBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = MixedCausalAttention(d_model, num_heads, num_ns_tokens, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = MixedFeedForward(d_model, hidden_mult, num_ns_tokens, dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        tokens = tokens + self.attention(self.attn_norm(tokens), padding_mask, seq_token_count)
        return tokens + self.ffn(self.ffn_norm(tokens), seq_token_count)


class AutoSplitTokenizer(nn.Module):
    def __init__(self, input_dim: int, num_tokens: int, d_model: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.projector = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_tokens * d_model,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x).view(x.size(0), self.num_tokens, self.d_model)


class GroupWiseTokenizer(nn.Module):
    def __init__(
        self,
        group_input_dims: list[int],
        d_model: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=d_model,
                    activation="relu",
                    dropout=dropout,
                )
                for input_dim in group_input_dims
            ]
        )

    def forward(self, group_inputs: list[torch.Tensor]) -> torch.Tensor:
        tokens = [projector(group_input).unsqueeze(1) for projector, group_input in zip(self.projectors, group_inputs)]
        return torch.cat(tokens, dim=1)


class OneTrans(BaseRankingModel):
    @property
    def model_name(self):
        return "OneTrans"

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
        d_model: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        ns_tokenizer_type: Literal["auto_split", "group_wise"] = "auto_split",
        num_ns_tokens: int = 4,
        ns_feature_groups: list[list[str]] | None = None,
        tokenizer_hidden_dims: list[int] | None = None,
        mlp_params: dict | None = None,
        **kwargs,
    ):
        """
        Initialize OneTrans model.
        初始化 OneTrans 模型。

        Args:
            d_model: Hidden dimension used by the unified token stream.
                统一 token 流的隐层维度。
            num_heads: Number of attention heads in mixed causal attention.
                mixed causal attention 中的注意力头数。
            num_blocks: Number of stacked OneTrans blocks.
                OneTrans block 的堆叠层数。
            hidden_mult: FFN hidden size multiplier.
                FFN 隐层相对 d_model 的放大倍数。
            dropout: Dropout rate in attention, FFN and token projectors.
                attention、FFN 和 token projector 中的 dropout 比例。
            ns_tokenizer_type: Non-sequential tokenizer type: "auto_split" or "group_wise".
                非序列特征压缩方式："auto_split" 或 "group_wise"。
            num_ns_tokens: Number of NS tokens for auto-split tokenizer.
                auto-split tokenizer 输出的 NS-token 数量。
            ns_feature_groups: Feature name groups used by group-wise tokenizer.
                group-wise tokenizer 使用的特征名分组。
            tokenizer_hidden_dims: Hidden dimensions for token compression MLPs.
                token 压缩 MLP 的隐层维度。
            mlp_params: Parameters for final DNN. e.g. {"hidden_dims": [256, 128], "dropout": 0.2}.
                最终 DNN 的参数，例如 {"hidden_dims": [256, 128], "dropout": 0.2}。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        tokenizer_hidden_dims = tokenizer_hidden_dims or []
        mlp_params = dict(mlp_params or {})

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        if not self.all_features:
            raise ValueError("[OneTrans Error] OneTrans requires at least one input feature.")
        if d_model <= 0:
            raise ValueError("[OneTrans Error] d_model must be positive.")
        if num_blocks <= 0:
            raise ValueError("[OneTrans Error] num_blocks must be positive.")
        if hidden_mult <= 0:
            raise ValueError("[OneTrans Error] hidden_mult must be positive.")

        self.d_model = int(d_model)
        self.sequence_features_all = sequence_features
        self.ns_features = dense_features + sparse_features
        self.embedding = EmbeddingLayer(features=self.all_features)

        self.sequence_projectors = nn.ModuleDict()
        for feature in self.sequence_features_all:
            self.sequence_projectors[feature.name] = nn.Linear(feature.embedding_dim, self.d_model)

        if self.ns_features:
            if ns_tokenizer_type == "auto_split":
                if num_ns_tokens <= 0:
                    raise ValueError("[OneTrans Error] num_ns_tokens must be positive.")
                ns_input_dim = self.embedding.compute_output_dim(self.ns_features)
                self.num_ns_tokens = int(num_ns_tokens)
                self.ns_tokenizer = AutoSplitTokenizer(
                    input_dim=ns_input_dim,
                    num_tokens=self.num_ns_tokens,
                    d_model=self.d_model,
                    hidden_dims=tokenizer_hidden_dims,
                    dropout=dropout,
                )
                self.ns_group_features = None
            elif ns_tokenizer_type == "group_wise":
                if not ns_feature_groups:
                    raise ValueError("[OneTrans Error] ns_feature_groups must be provided for group_wise tokenizer.")
                self.ns_group_features = [
                    select_feature_objects(self.ns_features, group, "ns_feature_groups") for group in ns_feature_groups
                ]
                used_names = [feature.name for group in self.ns_group_features for feature in group]
                if len(used_names) != len(set(used_names)):
                    raise ValueError("[OneTrans Error] ns_feature_groups contains duplicate feature names.")
                missing_names = [feature.name for feature in self.ns_features if feature.name not in set(used_names)]
                if missing_names:
                    raise ValueError(f"[OneTrans Error] ns_feature_groups does not cover features: {missing_names}.")
                group_input_dims = [self.embedding.compute_output_dim(group) for group in self.ns_group_features]
                self.num_ns_tokens = len(self.ns_group_features)
                self.ns_tokenizer = GroupWiseTokenizer(
                    group_input_dims=group_input_dims,
                    d_model=self.d_model,
                    hidden_dims=tokenizer_hidden_dims,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"[OneTrans Error] Unknown ns_tokenizer_type: {ns_tokenizer_type}")
        else:
            self.num_ns_tokens = 0
            self.ns_tokenizer = None
            self.ns_group_features = None

        self.separator_tokens = nn.Parameter(
            torch.randn(max(1, len(self.sequence_features_all) - 1), self.d_model) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    hidden_mult=hidden_mult,
                    num_ns_tokens=self.num_ns_tokens,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = RMSNorm(self.d_model)

        mlp_params.setdefault("hidden_dims", [256, 128])
        self.mlp = MLP(input_dim=self.d_model * 2, output_dim=1, **mlp_params)

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["sequence_projectors", "ns_tokenizer", "blocks", "mlp"],
        )

    def _sequence_mask(self, x, feature: SequenceFeature) -> torch.Tensor:
        seq_input = x[feature.name].long()
        if feature.max_len is not None:
            seq_input = seq_input[:, -feature.max_len :]
        return seq_input.ne(feature.padding_idx)

    def _encode_sequence_stream(self, x) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pieces = []
        masks = []
        sep_index = 0
        for feature_index, feature in enumerate(self.sequence_features_all):
            seq_tokens = self.embedding(
                x=x,
                features=[feature],
                squeeze_dim=False,
                sequence_output="raw",
            ).squeeze(1)
            seq_tokens = self.sequence_projectors[feature.name](seq_tokens)
            pieces.append(seq_tokens)
            masks.append(self._sequence_mask(x, feature))

            if feature_index < len(self.sequence_features_all) - 1:
                sep = self.separator_tokens[sep_index].view(1, 1, -1).expand(seq_tokens.size(0), -1, -1)
                pieces.append(sep)
                masks.append(torch.ones(seq_tokens.size(0), 1, device=seq_tokens.device, dtype=torch.bool))
                sep_index += 1

        if not pieces:
            return None, None
        return torch.cat(pieces, dim=1), torch.cat(masks, dim=1)

    def _encode_ns_tokens(self, x) -> torch.Tensor | None:
        if self.ns_tokenizer is None:
            return None
        if self.ns_group_features is None:
            ns_input = self.embedding(x=x, features=self.ns_features, squeeze_dim=True)
            return self.ns_tokenizer(ns_input)

        group_inputs = [
            self.embedding(x=x, features=group_features, squeeze_dim=True) for group_features in self.ns_group_features
        ]
        return self.ns_tokenizer(group_inputs)

    def _pyramid_keep_count(self, seq_token_count: int, layer_index: int) -> int:
        if seq_token_count <= max(1, self.num_ns_tokens):
            return seq_token_count
        remaining_layers = max(1, len(self.blocks) - layer_index)
        target = max(1, self.num_ns_tokens)
        decay = (target / seq_token_count) ** (1.0 / remaining_layers)
        return max(target, min(seq_token_count, math.ceil(seq_token_count * decay)))

    def _masked_mean(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~padding_mask.bool()).unsqueeze(-1).to(tokens.dtype)
        return (tokens * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

    def forward(self, x) -> torch.Tensor:
        sequence_tokens, sequence_valid_mask = self._encode_sequence_stream(x)
        ns_tokens = self._encode_ns_tokens(x)

        if sequence_tokens is None and ns_tokens is None:
            raise ValueError("[OneTrans Error] No tokens were built from input features.")

        tokens = []
        masks = []
        seq_token_count = 0
        if sequence_tokens is not None and sequence_valid_mask is not None:
            tokens.append(sequence_tokens)
            masks.append(~sequence_valid_mask)
            seq_token_count = sequence_tokens.size(1)
        if ns_tokens is not None:
            tokens.append(ns_tokens)
            masks.append(
                torch.zeros(ns_tokens.size(0), ns_tokens.size(1), dtype=torch.bool, device=ns_tokens.device)
            )

        token_stream = torch.cat(tokens, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        for layer_index, block in enumerate(self.blocks):
            token_stream = block(token_stream, padding_mask, seq_token_count)
            keep_count = self._pyramid_keep_count(seq_token_count, layer_index)
            if keep_count < seq_token_count:
                sequence_part = token_stream[:, :seq_token_count, :]
                ns_part = token_stream[:, seq_token_count:, :]
                sequence_mask = padding_mask[:, :seq_token_count]
                ns_mask = padding_mask[:, seq_token_count:]
                token_stream = torch.cat([sequence_part[:, -keep_count:, :], ns_part], dim=1)
                padding_mask = torch.cat([sequence_mask[:, -keep_count:], ns_mask], dim=1)
                seq_token_count = keep_count

        token_stream = self.final_norm(token_stream)
        if seq_token_count > 0:
            seq_summary = self._masked_mean(token_stream[:, :seq_token_count, :], padding_mask[:, :seq_token_count])
        else:
            seq_summary = token_stream.new_zeros(token_stream.size(0), self.d_model)

        if self.num_ns_tokens > 0:
            ns_summary = self._masked_mean(token_stream[:, seq_token_count:, :], padding_mask[:, seq_token_count:])
        else:
            ns_summary = token_stream.new_zeros(token_stream.size(0), self.d_model)

        logits = self.mlp(torch.cat([seq_summary, ns_summary], dim=1))
        return logits
