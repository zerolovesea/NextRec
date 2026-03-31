"""
Date: create on 30/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Hidasi B, Karatzoglou A, Baltrunas L, et al. Session-based Recommendations with Recurrent Neural Networks. ICLR 2016.
- [2] hidasib/GRU4Rec. https://github.com/hidasib/GRU4Rec
- [3] RUCAIBox/RecBole GRU4Rec implementation. https://github.com/RUCAIBox/RecBole
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import TaskTypeInput


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
        item_history: str | None = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        max_seq_len: Optional[int] = None,
        dropout_rate: float = 0.2,
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

        if item_history is not None:
            item_history_name = item_history

        self.item_history_feature = select_feature_objects(
            sequence_features,
            [item_history_name],
            "item_history_name",
        )[0]
        self.vocab_size = int(self.item_history_feature.vocab_size)
        self.hidden_dim = int(hidden_dim or self.item_history_feature.embedding_dim)
        self.max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        self.padding_idx = int(self.item_history_feature.padding_idx or 0)

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            sequence_mode="autoregressive",
            target_source=self.item_history_feature.name,
            target_shift_steps=1,
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
        self.input_proj = nn.Linear(input_dim, self.hidden_dim) if input_dim != self.hidden_dim else nn.Identity()

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.input_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        if self.item_embedding.embedding_dim == self.hidden_dim:
            self.output_proj.weight = self.item_embedding.weight

        self.register_regularization_weights(
            embedding_attr="feature_embedding",
            include_modules=["gru", "input_proj", "output_proj"],
        )

    def encode_sequence(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        seq, padding_mask, valid_mask, _ = self.prepare_sequence_batch(
            x=x,
            sequence_name=self.item_history_feature.name,
            max_seq_len=self.max_seq_len,
            padding_idx=self.padding_idx,
        )

        item_emb = self.item_embedding(seq)
        if self.context_features:
            context_repr = self.context_embedding(x, self.context_features, squeeze_dim=True)
            context_repr = context_repr.unsqueeze(1).expand(-1, seq.size(1), -1)
            seq_emb = torch.cat([item_emb, context_repr], dim=-1)
        else:
            seq_emb = item_emb

        seq_emb = self.input_dropout(self.input_proj(seq_emb))
        hidden_states, _ = self.gru(seq_emb)
        hidden_states = self.output_dropout(hidden_states)
        hidden_states = hidden_states * valid_mask
        return hidden_states, padding_mask

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_states, _ = self.encode_sequence(x)
        return self.output_proj(hidden_states)
