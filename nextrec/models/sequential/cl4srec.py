"""
Date: create on 30/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Xie X, Sun F, Liu Z, et al. Contrastive Learning for Sequential Recommendation. ICDE 2022.
- [2] RUCAIBox/RecBole CL4SRec implementation. https://github.com/RUCAIBox/RecBole
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nextrec.models.sequential.sasrec import SASRec


class CL4SRec(SASRec):
    @property
    def model_name(self) -> str:
        return "CL4SRec"

    def __init__(
        self,
        *args,
        cl_weight: float = 0.1,
        temperature: float = 0.2,
        crop_ratio: float = 0.2,
        mask_ratio: float = 0.2,
        reorder_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cl_weight = float(cl_weight)
        self.temperature = float(temperature)
        self.crop_ratio = float(crop_ratio)
        self.mask_ratio = float(mask_ratio)
        self.reorder_ratio = float(reorder_ratio)
        self._last_contrastive_loss = torch.tensor(0.0, device=self.device)

    def _augment_crop(self, seq: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(seq, fill_value=self.padding_idx)
        valid_len = int(seq.ne(self.padding_idx).sum().item())
        if valid_len <= 1:
            return seq.clone()
        keep_len = max(1, int(math.ceil(valid_len * (1.0 - self.crop_ratio))))
        start = int(torch.randint(0, valid_len - keep_len + 1, (1,), device=seq.device).item())
        cropped = seq[start : start + keep_len]
        out[:keep_len] = cropped
        return out

    def _augment_mask(self, seq: torch.Tensor) -> torch.Tensor:
        out = seq.clone()
        valid_positions = torch.nonzero(seq.ne(self.padding_idx), as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return out
        num_mask = max(1, int(math.ceil(valid_positions.numel() * self.mask_ratio)))
        perm = torch.randperm(valid_positions.numel(), device=seq.device)[:num_mask]
        out[valid_positions[perm]] = self.padding_idx
        return out

    def _augment_reorder(self, seq: torch.Tensor) -> torch.Tensor:
        out = seq.clone()
        valid_len = int(seq.ne(self.padding_idx).sum().item())
        if valid_len <= 2:
            return out
        reorder_len = max(2, int(math.ceil(valid_len * self.reorder_ratio)))
        reorder_len = min(reorder_len, valid_len)
        start = int(torch.randint(0, valid_len - reorder_len + 1, (1,), device=seq.device).item())
        perm = torch.randperm(reorder_len, device=seq.device)
        out[start : start + reorder_len] = out[start : start + reorder_len][perm]
        return out

    def augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        ops = (self._augment_crop, self._augment_mask, self._augment_reorder)
        aug_seq = []
        for row in seq:
            op_idx = int(torch.randint(0, len(ops), (1,), device=row.device).item())
            aug_seq.append(ops[op_idx](row))
        return torch.stack(aug_seq, dim=0)

    def encode_sequence_representation(self, x: dict[str, torch.Tensor], seq: torch.Tensor) -> torch.Tensor:
        x_aug = dict(x)
        x_aug[self.item_history_feature.name] = seq
        hidden_states, padding_mask = self.log2feats(x_aug)
        last_hidden, _ = self.select_last_valid_state(hidden_states, padding_mask)
        return F.normalize(last_hidden, dim=-1)

    def compute_contrastive_loss(self, seq_view_1: torch.Tensor, seq_view_2: torch.Tensor) -> torch.Tensor:
        if seq_view_1.size(0) <= 1:
            return torch.tensor(0.0, device=seq_view_1.device)

        representations = torch.cat([seq_view_1, seq_view_2], dim=0)
        similarity = torch.matmul(representations, representations.t()) / self.temperature

        batch_size = seq_view_1.size(0)
        diag_mask = torch.eye(batch_size * 2, device=similarity.device, dtype=torch.bool)
        similarity = similarity.masked_fill(diag_mask, float("-inf"))

        positive_index = torch.arange(batch_size * 2, device=similarity.device)
        positive_index = (positive_index + batch_size) % (batch_size * 2)
        return F.cross_entropy(similarity, positive_index)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = super().forward(x)
        if self.training and self.cl_weight > 0.0:
            seq = x[self.item_history_feature.name].long()[:, -self.max_seq_len :]
            view_1 = self.augment_sequence(seq)
            view_2 = self.augment_sequence(seq)
            rep_1 = self.encode_sequence_representation(x, view_1)
            rep_2 = self.encode_sequence_representation(x, view_2)
            self._last_contrastive_loss = self.compute_contrastive_loss(rep_1, rep_2)
        else:
            self._last_contrastive_loss = torch.tensor(0.0, device=logits.device)
        return logits

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor | None) -> torch.Tensor:
        base_loss = super().compute_loss(y_pred, y_true)
        return base_loss + self.cl_weight * self._last_contrastive_loss
