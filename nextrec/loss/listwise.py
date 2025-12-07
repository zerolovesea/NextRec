"""
Listwise loss functions for ranking and contrastive training.

Date: create on 27/10/2025
Checkpoint: edit on 29/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledSoftmaxLoss(nn.Module):
    """
    Softmax over one positive and multiple sampled negatives.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pos_logits: torch.Tensor, neg_logits: torch.Tensor
    ) -> torch.Tensor:
        pos_logits = pos_logits.unsqueeze(1)
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        targets = torch.zeros(
            all_logits.size(0), dtype=torch.long, device=all_logits.device
        )
        loss = F.cross_entropy(all_logits, targets, reduction=self.reduction)
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning with one positive and many negatives.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self, query: torch.Tensor, pos_key: torch.Tensor, neg_keys: torch.Tensor
    ) -> torch.Tensor:
        pos_sim = torch.sum(query * pos_key, dim=-1) / self.temperature
        pos_sim = pos_sim.unsqueeze(1)
        query_expanded = query.unsqueeze(1)
        neg_sim = torch.sum(query_expanded * neg_keys, dim=-1) / self.temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return loss


class ListNetLoss(nn.Module):
    """
    ListNet loss using top-1 probability distribution.
    Reference: Cao et al. (ICML 2007)
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(scores / self.temperature, dim=1)
        true_probs = F.softmax(labels / self.temperature, dim=1)
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ListMLELoss(nn.Module):
    """
    ListMLE (Maximum Likelihood Estimation) loss.
    Reference: Xia et al. (ICML 2008)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sorted_labels, sorted_indices = torch.sort(labels, descending=True, dim=1)
        batch_size, list_size = scores.shape
        batch_indices = (
            torch.arange(batch_size, device=scores.device)
            .unsqueeze(1)
            .expand(-1, list_size)
        )
        sorted_scores = scores[batch_indices, sorted_indices]

        loss = torch.tensor(0.0, device=scores.device)
        for i in range(list_size):
            remaining_scores = sorted_scores[:, i:]
            log_sum_exp = torch.logsumexp(remaining_scores, dim=1)
            loss = loss + (log_sum_exp - sorted_scores[:, i]).sum()

        if self.reduction == "mean":
            return loss / batch_size
        if self.reduction == "sum":
            return loss
        return loss / batch_size


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss for learning to rank.
    Reference: Qin et al. (2010)
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def _ideal_dcg(self, labels: torch.Tensor, k: int | None) -> torch.Tensor:
        # labels: [B, L]
        sorted_labels, _ = torch.sort(labels, dim=1, descending=True)
        if k is not None:
            sorted_labels = sorted_labels[:, :k]

        gains = torch.pow(2.0, sorted_labels) - 1.0  # [B, K]
        positions = torch.arange(
            1, gains.size(1) + 1, device=gains.device, dtype=torch.float32
        )  # [K]
        discounts = 1.0 / torch.log2(positions + 1.0)  # [K]
        ideal_dcg = torch.sum(gains * discounts, dim=1)  # [B]
        return ideal_dcg

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, k: int | None = None
    ) -> torch.Tensor:
        """
        scores: [B, L]
        labels: [B, L]
        """
        batch_size, list_size = scores.shape
        device = scores.device

        # diff[b, i, j] = (s_j - s_i) / T
        scores_i = scores.unsqueeze(2)  # [B, L, 1]
        scores_j = scores.unsqueeze(1)  # [B, 1, L]
        diff = (scores_j - scores_i) / self.temperature  # [B, L, L]

        P_ji = torch.sigmoid(diff)  # [B, L, L]
        eye = torch.eye(list_size, device=device).unsqueeze(0)  # [1, L, L]
        P_ji = P_ji * (1.0 - eye)

        exp_rank = 1.0 + P_ji.sum(dim=-1)  # [B, L]

        discounts = 1.0 / torch.log2(exp_rank + 1.0)  # [B, L]

        gains = torch.pow(2.0, labels) - 1.0  # [B, L]
        approx_dcg = torch.sum(gains * discounts, dim=1)  # [B]

        ideal_dcg = self._ideal_dcg(labels, k)  # [B]

        ndcg = approx_dcg / (ideal_dcg + 1e-10)  # [B]
        loss = 1.0 - ndcg

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
