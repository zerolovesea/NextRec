"""
Date: create on 30/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Sun F, Liu J, Wu J, et al. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM 2019.
- [2] FeiSun/BERT4Rec. https://github.com/FeiSun/BERT4Rec
- [3] RUCAIBox/RecBole BERT4Rec implementation. https://github.com/RUCAIBox/RecBole
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nextrec.basic.adapters import MaskedSequentialAdapter
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer
from nextrec.models.sequential.base import BaseSequentialModel
from nextrec.utils.model import select_feature_objects
from nextrec.utils.types import TaskTypeInput


class BERT4RecBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BERT4Rec(BaseSequentialModel):
    @property
    def model_name(self) -> str:
        return "BERT4Rec"

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
        num_heads: int = 2,
        num_layers: int = 2,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout_rate: float = 0.2,
        mask_ratio: float = 0.2,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        target: str | list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        **kwargs,
    ):
        if not sequence_features:
            raise ValueError("[BERT4Rec Error] BERT4Rec requires at least one SequenceFeature.")

        if item_history is not None:
            item_history_name = item_history

        self.item_history_feature = select_feature_objects(
            sequence_features,
            [item_history_name],
            "item_history_name",
        )[0]

        self.vocab_size = int(self.item_history_feature.vocab_size)
        self.hidden_dim = int(hidden_dim or self.item_history_feature.embedding_dim)
        self.ff_hidden_dim = int(ff_hidden_dim or (self.hidden_dim * 4))
        self.max_seq_len = int(max_seq_len or self.item_history_feature.max_len)
        self.padding_idx = int(self.item_history_feature.padding_idx or 0)
        self.mask_ratio = float(mask_ratio)
        self.mask_token_ratio = float(mask_token_ratio)
        self.random_token_ratio = float(random_token_ratio)
        self.mask_token_id = self.vocab_size
        self.input_vocab_size = self.vocab_size + 1

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"[BERT4Rec Error] hidden_dim({self.hidden_dim}) must be divisible by num_heads({num_heads})."
            )
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("[BERT4Rec Error] mask_ratio must be in (0, 1].")

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task or self.default_task,
            sequence_mode="masked",
            target_source=None,
            target_shift_steps=0,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.context_features = [feat for feat in self.all_features if feat.name != self.item_history_feature.name]
        self.item_embedding = nn.Embedding(
            num_embeddings=self.input_vocab_size,
            embedding_dim=int(self.item_history_feature.embedding_dim),
            padding_idx=self.padding_idx,
        )
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx < self.item_embedding.num_embeddings:
            nn.init.zeros_(self.item_embedding.weight[self.padding_idx])

        self.context_embedding = EmbeddingLayer(self.context_features) if self.context_features else None

        item_dim = int(self.item_history_feature.embedding_dim)
        context_dim = int(self.context_embedding.output_dim) if self.context_embedding is not None else 0
        input_dim = item_dim + context_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_dim) if input_dim != self.hidden_dim else nn.Identity()

        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                BERT4RecBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    ff_hidden_dim=self.ff_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-8)

        self.tie_output = item_dim == self.hidden_dim
        if self.tie_output:
            self.output_proj = None
            self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
            self.output_bias = None

        self._last_mask_labels: torch.Tensor | None = None

        modules = ["blocks", "input_proj", "position_embedding"]
        if self.context_embedding is not None:
            modules.append("context_embedding")
        if self.output_proj is not None:
            modules.append("output_proj")
        self.register_regularization_weights(
            embedding_attr="item_embedding",
            include_modules=modules,
        )

    def set_adapter(self):
        self.training_adapter = MaskedSequentialAdapter()
        self.prediction_layer = None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tie_output:
            logits = torch.matmul(hidden_states, self.item_embedding.weight[: self.vocab_size].t())
            return logits + self.output_bias
        assert self.output_proj is not None
        return self.output_proj(hidden_states)

    def build_masked_sequence(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_seq = seq.clone()
        labels = torch.full_like(seq, fill_value=int(self.ignore_label if self.ignore_label is not None else -1))
        valid_mask = seq.ne(self.padding_idx)

        random_tensor = torch.rand_like(seq.float())
        mask_positions = valid_mask & (random_tensor < self.mask_ratio)

        # Ensure every non-empty sequence contributes at least one masked position.
        valid_counts = valid_mask.sum(dim=1)
        no_mask_rows = (mask_positions.sum(dim=1) == 0) & (valid_counts > 0)
        if torch.any(no_mask_rows):
            valid_indices = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
            last_valid = (valid_counts - 1).clamp(min=0).unsqueeze(1)
            fallback_mask = no_mask_rows.unsqueeze(1) & valid_indices.eq(last_valid)
            mask_positions = mask_positions | fallback_mask

        labels[mask_positions] = seq[mask_positions]

        replace_prob = torch.rand_like(seq.float())
        mask_token_mask = mask_positions & (replace_prob < self.mask_token_ratio)
        random_token_mask = mask_positions & (
            (replace_prob >= self.mask_token_ratio)
            & (replace_prob < self.mask_token_ratio + self.random_token_ratio)
        )

        masked_seq[mask_token_mask] = self.mask_token_id
        if torch.any(random_token_mask):
            random_ids = torch.randint(
                low=1,
                high=max(self.vocab_size, 2),
                size=(int(random_token_mask.sum().item()),),
                device=seq.device,
            )
            masked_seq[random_token_mask] = random_ids
        return masked_seq, labels

    def prepare_prediction_sequence(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = seq.clone()
        valid_counts = seq.ne(self.padding_idx).sum(dim=1)
        target_positions = torch.clamp(valid_counts, max=seq.size(1) - 1)
        full_rows = valid_counts >= seq.size(1)
        if torch.any(full_rows):
            seq[full_rows, :-1] = seq[full_rows, 1:].clone()
            seq[full_rows, -1] = self.mask_token_id
            target_positions[full_rows] = seq.size(1) - 1
        not_full_rows = ~full_rows
        if torch.any(not_full_rows):
            batch_index = torch.arange(seq.size(0), device=seq.device)
            seq[batch_index[not_full_rows], target_positions[not_full_rows]] = self.mask_token_id
        return seq, target_positions

    def encode_sequence(
        self,
        seq: torch.Tensor,
        x: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self.build_sequence_padding_mask(seq, padding_idx=self.padding_idx)
        valid_mask = self.build_sequence_valid_mask(padding_mask)
        batch_size, seq_len = seq.shape
        pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        item_emb = self.item_embedding(seq)
        if self.context_embedding is not None:
            context_repr = self.context_embedding(x, self.context_features, squeeze_dim=True)
            context_repr = context_repr.unsqueeze(1).expand(-1, seq_len, -1)
            seq_emb = torch.cat([item_emb, context_repr], dim=-1)
        else:
            seq_emb = item_emb

        hidden_states = self.input_proj(seq_emb)
        hidden_states = hidden_states + self.position_embedding(pos_ids)
        hidden_states = self.dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, key_padding_mask=padding_mask)
            hidden_states = hidden_states * valid_mask

        hidden_states = self.last_layer_norm(hidden_states)
        hidden_states = hidden_states * valid_mask
        return hidden_states, padding_mask

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = x[self.item_history_feature.name].long()[:, -self.max_seq_len :]
        if self.training:
            masked_seq, labels = self.build_masked_sequence(seq)
            self._last_mask_labels = labels
        else:
            masked_seq = seq
            self._last_mask_labels = torch.full_like(
                seq,
                fill_value=int(self.ignore_label if self.ignore_label is not None else -1),
            )

        hidden_states, _ = self.encode_sequence(masked_seq, x)
        return self.compute_logits(hidden_states)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor | None) -> torch.Tensor:
        if self._last_mask_labels is None:
            raise RuntimeError("[BERT4Rec-compute_loss Error] Mask labels are not initialized. Call forward() first.")
        return super().compute_loss(y_pred, self._last_mask_labels)

    def predict_last(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = x[self.item_history_feature.name].long()[:, -self.max_seq_len :]
        predict_seq, target_positions = self.prepare_prediction_sequence(seq)
        hidden_states, _ = self.encode_sequence(predict_seq, x)
        logits = self.compute_logits(hidden_states)
        batch_index = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_index, target_positions]
