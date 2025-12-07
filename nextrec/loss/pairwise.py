"""
Pairwise loss functions for learning-to-rank and matching tasks.

Date: create on 27/10/2025
Checkpoint: edit on 29/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss with support for multiple negatives.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        if neg_score.dim() == 2:
            pos_score = pos_score.unsqueeze(1)
            diff = pos_score - neg_score
        else:
            diff = pos_score - neg_score

        loss = -torch.log(torch.sigmoid(diff) + 1e-8)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HingeLoss(nn.Module):
    """
    Hinge loss for pairwise ranking.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        if neg_score.dim() == 2:
            pos_score = pos_score.unsqueeze(1)

        diff = pos_score - neg_score
        loss = torch.clamp(self.margin - diff, min=0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet margin loss with cosine or euclidean distance.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "mean",
        distance: Literal["euclidean", "cosine"] = "euclidean",
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = distance

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "euclidean":
            pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
            if negative.dim() == 3:
                anchor_expanded = anchor.unsqueeze(1)
                neg_dist = torch.sum((anchor_expanded - negative) ** 2, dim=-1)
            else:
                neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)
            if neg_dist.dim() == 2:
                pos_dist = pos_dist.unsqueeze(1)
        elif self.distance == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            if negative.dim() == 3:
                anchor_expanded = anchor.unsqueeze(1)
                neg_dist = 1 - F.cosine_similarity(anchor_expanded, negative, dim=-1)
            else:
                neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
            if neg_dist.dim() == 2:
                pos_dist = pos_dist.unsqueeze(1)
        else:
            raise ValueError(f"[Loss Error] Unsupported distance: {self.distance}")

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
