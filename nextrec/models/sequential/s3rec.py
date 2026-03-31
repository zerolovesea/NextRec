"""
Date: create on 30/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Zhou K, Wang H, Zhao W X, et al. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020.
- [2] RUCAIBox/CIKM2020-S3Rec. https://github.com/RUCAIBox/CIKM2020-S3Rec
- [3] RUCAIBox/RecBole S3Rec implementation. https://github.com/RUCAIBox/RecBole
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.models.sequential.bert4rec import BERT4Rec
from nextrec.models.sequential.base import BaseSequentialModel


class S3Rec(BERT4Rec):
    @property
    def model_name(self) -> str:
        return "S3Rec"

    def __init__(
        self,
        *args,
        mip_weight: float = 1.0,
        sp_weight: float = 0.5,
        segment_ratio: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mip_weight = float(mip_weight)
        self.sp_weight = float(sp_weight)
        self.segment_ratio = float(segment_ratio)
        item_dim = int(self.item_history_feature.embedding_dim)
        self.segment_proj = nn.Identity() if item_dim == self.hidden_dim else nn.Linear(item_dim, self.hidden_dim)
        self._last_mip_labels: torch.Tensor | None = None
        self._last_sp_loss = torch.tensor(0.0, device=self.device)

    def build_segment_prediction_task(
        self,
        seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = seq.shape
        context_seq = seq.clone()
        positive_segments = torch.full_like(seq, fill_value=self.padding_idx)
        negative_segments = torch.full_like(seq, fill_value=self.padding_idx)
        valid_mask = seq.ne(self.padding_idx)
        valid_lens = valid_mask.sum(dim=1)

        for idx in range(batch_size):
            valid_len = int(valid_lens[idx].item())
            if valid_len < 2:
                continue
            seg_len = max(1, int(math.ceil(valid_len * self.segment_ratio)))
            seg_len = min(seg_len, valid_len)
            start = int(torch.randint(0, valid_len - seg_len + 1, (1,), device=seq.device).item())
            end = start + seg_len

            segment = seq[idx, start:end]
            positive_segments[idx, :seg_len] = segment
            context_seq[idx, start:end] = self.mask_token_id

            neg_idx = int(torch.randint(0, batch_size, (1,), device=seq.device).item())
            if batch_size > 1:
                while neg_idx == idx:
                    neg_idx = int(torch.randint(0, batch_size, (1,), device=seq.device).item())
            neg_valid_len = int(valid_lens[neg_idx].item())
            if neg_valid_len == 0:
                continue
            neg_seg_len = min(seg_len, neg_valid_len)
            neg_start = int(torch.randint(0, neg_valid_len - neg_seg_len + 1, (1,), device=seq.device).item())
            negative_segments[idx, :neg_seg_len] = seq[neg_idx, neg_start : neg_start + neg_seg_len]

        return context_seq, positive_segments, negative_segments, valid_lens

    def segment_pool(self, segment_seq: torch.Tensor) -> torch.Tensor:
        segment_emb = self.item_embedding(segment_seq.clamp(max=self.vocab_size))
        segment_mask = segment_seq.ne(self.padding_idx).float().unsqueeze(-1)
        denom = segment_mask.sum(dim=1).clamp(min=1.0)
        pooled = (segment_emb * segment_mask).sum(dim=1) / denom
        return self.segment_proj(pooled)

    def compute_segment_prediction_loss(self, seq: torch.Tensor, x: dict[str, torch.Tensor]) -> torch.Tensor:
        if seq.size(0) == 0:
            return torch.tensor(0.0, device=seq.device)

        context_seq, positive_segments, negative_segments, valid_lens = self.build_segment_prediction_task(seq)
        if not torch.any(valid_lens >= 2):
            return torch.tensor(0.0, device=seq.device)

        hidden_states, padding_mask = self.encode_sequence(context_seq, x)
        context_repr, _ = self.select_last_valid_state(hidden_states, padding_mask)
        pos_repr = self.segment_pool(positive_segments)
        neg_repr = self.segment_pool(negative_segments)

        pos_score = (context_repr * pos_repr).sum(dim=-1)
        neg_score = (context_repr * neg_repr).sum(dim=-1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = x[self.item_history_feature.name].long()[:, -self.max_seq_len :]
        if self.training:
            masked_seq, labels = self.build_masked_sequence(seq)
            self._last_mip_labels = labels
            self._last_sp_loss = self.compute_segment_prediction_loss(seq, x)
        else:
            masked_seq = seq
            self._last_mip_labels = torch.full_like(
                seq,
                fill_value=int(self.ignore_label if self.ignore_label is not None else -1),
            )
            self._last_sp_loss = torch.tensor(0.0, device=seq.device)

        hidden_states, _ = self.encode_sequence(masked_seq, x)
        return self.compute_logits(hidden_states)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor | None) -> torch.Tensor:
        if self._last_mip_labels is None:
            raise RuntimeError("[S3Rec-compute_loss Error] Pretrain labels are not initialized. Call forward() first.")
        mip_loss = BaseSequentialModel.compute_loss(self, y_pred, self._last_mip_labels)
        return self.mip_weight * mip_loss + self.sp_weight * self._last_sp_loss
