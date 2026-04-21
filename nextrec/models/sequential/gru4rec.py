"""
Date: create on 21/04/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Tan Y K, Xu X, Liu Y. Improved recurrent neural networks for session-based recommendations. DLRS 2016.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import SequenceModeName, TaskTypeInput


class GRU4Rec(BaseSequentialModel):
    @property
    def model_name(self) -> str:
        return "GRU4Rec"

    @property
    def default_task(self) -> str:
        return "sequential"

    def __init__(
        self,
        sequence_features: list[SequenceFeature],
        dense_features: Optional[list[DenseFeature]] = None,
        sparse_features: Optional[list[SparseFeature]] = None,
        item_history_name: str = "item_history",
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        max_seq_len: Optional[int] = None,
        dropout_rate: float | None = 0.0,
        dropout: float | None = None,
        sequence_mode: SequenceModeName = "autoregressive",
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        if not sequence_features:
            raise ValueError("[GRU4Rec Error] GRU4Rec requires at least one SequenceFeature.")
        if sequence_mode != "autoregressive":
            raise ValueError("[GRU4Rec Error] GRU4Rec currently only supports sequence_mode='autoregressive'.")
        if num_layers < 1:
            raise ValueError(f"[GRU4Rec Error] num_layers must be >= 1, got {num_layers}.")

        if dropout is not None:
            dropout_rate = dropout
        dropout_rate = float(dropout_rate if dropout_rate is not None else 0.0)

        self.item_history_feature = select_feature_objects(
            sequence_features,
            [item_history_name],
            "item_history_name",
        )[0]
        self.vocab_size = int(self.item_history_feature.vocab_size)
        self.max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        self.padding_idx = (
            self.item_history_feature.padding_idx if self.item_history_feature.padding_idx is not None else 0
        )

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            sequence_mode=sequence_mode,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.context_features = [feat for feat in self.all_features if feat.name != self.item_history_feature.name]
        self.feature_embedding = EmbeddingLayer(features=self.all_features)
        self.item_embedding = self.feature_embedding.embed_dict[self.item_history_feature.embedding_name]
        self.context_embedding = self.feature_embedding if self.context_features else None

        item_dim = int(self.item_history_feature.embedding_dim)
        context_dim = int(self.feature_embedding.output_dim - item_dim) if self.context_features else 0
        input_dim = item_dim + context_dim
        self.hidden_dim = int(hidden_dim or item_dim)

        if input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        gru_dropout = dropout_rate if num_layers > 1 else 0.0
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bias=True,
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        if self.item_embedding.embedding_dim == self.hidden_dim:
            self.output_proj.weight = self.item_embedding.weight

        self.register_regularization_weights(
            embedding_attr="feature_embedding",
            include_modules=["gru", "output_proj", "input_proj"],
        )

    def encode_sequence(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        seq, padding_mask, valid_mask, _ = self.prepare_sequence_batch(
            x=x,
            sequence_name=self.item_history_feature.name,
            max_seq_len=self.max_seq_len,
            padding_idx=self.padding_idx,
        )
        batch_size, seq_len = seq.shape

        item_emb = self.item_embedding(seq)  # [B, L, E]
        if self.context_features:
            context_repr = self.context_embedding(x, self.context_features, squeeze_dim=True)  # [B, C]
            context_repr = context_repr.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, C]
            seq_emb = torch.cat([item_emb, context_repr], dim=-1)  # [B, L, E + C]
        else:
            seq_emb = item_emb

        seq_emb = self.input_proj(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(
            seq_emb,
            lengths=valid_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        hidden_states, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len,
        )
        hidden_states = hidden_states * valid_mask
        return hidden_states, padding_mask

    def forward(self, x) -> torch.Tensor:
        hidden_states, _ = self.encode_sequence(x)
        logits = self.output_proj(hidden_states)  # [B, L, V]
        return logits
