"""
Date: create on 26/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Kang W C, McAuley J. Self-attentive sequential recommendation. ICDM 2018.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import SequenceModeName, TaskTypeInput


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ff_hidden_dim: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block
        residual = x
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_output

        # Point-wise FFN block
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SASRec(BaseSequentialModel):
    @property
    def model_name(self) -> str:
        return "SASRec"

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
        num_heads: int = 1,
        num_layers: int | None = None,
        num_blocks: int | None = 2,
        ff_hidden_dim: Optional[int] = None,
        sequence_mode: SequenceModeName = "autoregressive",
        max_seq_len: Optional[int] = None,
        dropout_rate: float | None = 0.2,
        dropout: float | None = None,
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        if not sequence_features:
            raise ValueError("[SASRec Error] SASRec requires at least one SequenceFeature.")

        if item_history is not None:
            item_history_name = item_history
        if num_layers is None:
            num_layers = num_blocks if num_blocks is not None else 2
        if dropout is not None:
            dropout_rate = dropout
        dropout_rate = float(dropout_rate if dropout_rate is not None else 0.2)

        self.item_history_feature = select_feature_objects(
            sequence_features,
            [item_history_name],
            "item_history_name",
        )[0]

        self.vocab_size = int(self.item_history_feature.vocab_size)
        self.hidden_dim = int(hidden_dim or self.item_history_feature.embedding_dim)
        self.ff_hidden_dim = int(ff_hidden_dim or (self.hidden_dim * 4))
        self.max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        self.padding_idx = (
            self.item_history_feature.padding_idx if self.item_history_feature.padding_idx is not None else 0
        )

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"[SASRec Error] hidden_dim({self.hidden_dim}) must be divisible by num_heads({num_heads})."
            )

        # SASRec can consume optional context features in addition to the item history.
        target_source = self.item_history_feature.name if sequence_mode == "autoregressive" else None
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            sequence_mode=sequence_mode,
            target_source=target_source,
            target_shift_steps=1 if target_source is not None else 0,
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
        if input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SASRecBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    ff_hidden_dim=self.ff_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-8)

        # Full-vocab output head
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # Weight tying if dimensions match
        if self.item_embedding.embedding_dim == self.hidden_dim:
            self.output_proj.weight = self.item_embedding.weight

        self.register_regularization_weights(
            embedding_attr="feature_embedding",
            include_modules=["blocks", "output_proj", "input_proj"],
        )

    def log2feats(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert item sequence to hidden states.
        Args:
            x: Batch input dict containing item history and optional context features.
        Returns:
            hidden_states: [B, L, H]
            padding_mask: [B, L], True means padding
        """
        seq, padding_mask, valid_mask, attention_mask = self.prepare_sequence_batch(
            x=x,
            sequence_name=self.item_history_feature.name,
            max_seq_len=self.max_seq_len,
            padding_idx=self.padding_idx,
        )
        batch_size, seq_len = seq.shape
        device = seq.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        item_emb = self.item_embedding(seq)  # [B, L, E]
        if self.context_features:
            context_repr = self.context_embedding(x, self.context_features, squeeze_dim=True)  # [B, C]
            context_repr = context_repr.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, C]
            seq_emb = torch.cat([item_emb, context_repr], dim=-1)  # [B, L, E + C]
        else:
            seq_emb = item_emb
        seq_emb = self.input_proj(seq_emb)  # [B, L, H]

        hidden_states = seq_emb * (self.hidden_dim**0.5)
        hidden_states = hidden_states + self.position_embedding(pos_ids)
        hidden_states = self.dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask,
            )
            hidden_states = hidden_states * valid_mask

        hidden_states = self.last_layer_norm(hidden_states)
        hidden_states = hidden_states * valid_mask
        return hidden_states, padding_mask

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard training forward:
            input sequence -> hidden states at every position -> logits over all items
        Returns:
            logits: [B, L, V]
        """
        hidden_states, _ = self.log2feats(x)
        logits = self.output_proj(hidden_states)  # [B, L, V]
        return logits
