"""
Pointwise loss functions, including imbalance-aware variants.

Date: create on 27/10/2025
Checkpoint: edit on 29/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss using cosine similarity for positive/negative pairs.
    """

    def __init__(self, margin: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        labels = labels.float()
        similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        pos_loss = torch.clamp(self.margin - similarity, min=0) * labels
        neg_loss = torch.clamp(similarity - self.margin, min=0) * (1 - labels)
        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Binary cross entropy with controllable positive class weight.
    Supports probability or logit inputs via `logits` flag.
    If `auto_balance=True` and `pos_weight` is None, the positive weight is
    computed from the batch as (#neg / #pos) for stable imbalance handling.
    """

    def __init__(
        self,
        pos_weight: float | torch.Tensor | None = None,
        reduction: str = "mean",
        logits: bool = False,
        auto_balance: bool = False,
    ):
        super().__init__()
        self.reduction = reduction
        self.logits = logits
        self.auto_balance = auto_balance

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.as_tensor(pos_weight, dtype=torch.float32),
            )
        else:
            self.pos_weight = None

    def resolve_pos_weight(self, labels: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            return self.pos_weight.to(device=labels.device)

        if not self.auto_balance:
            return torch.tensor(1.0, device=labels.device, dtype=labels.dtype)

        labels_float = labels.float()
        pos = torch.clamp(labels_float.sum(), min=1.0)
        neg = torch.clamp(labels_float.numel() - labels_float.sum(), min=1.0)
        return (neg / pos).to(device=labels.device, dtype=labels.dtype)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        current_pos_weight = self.resolve_pos_weight(labels)
        current_pos_weight = current_pos_weight.to(inputs.dtype)

        if self.logits:
            loss = F.binary_cross_entropy_with_logits(
                inputs, labels, pos_weight=current_pos_weight, reduction="none"
            )
        else:
            probs = torch.clamp(inputs, min=1e-6, max=1 - 1e-6)
            base_loss = F.binary_cross_entropy(probs, labels, reduction="none")
            loss = torch.where(labels == 1, base_loss * current_pos_weight, base_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Standard focal loss for binary or multi-class classification.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float | Sequence[float] | torch.Tensor] = None,
        reduction: str = "mean",
        logits: bool = False,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Multi-class case
        if inputs.dim() > 1 and inputs.size(1) > 1:
            log_probs = F.log_softmax(inputs, dim=1)
            probs = log_probs.exp()
            targets_one_hot = F.one_hot(
                targets.long(), num_classes=inputs.size(1)
            ).float()

            alpha = self.get_alpha(inputs)
            alpha_factor = targets_one_hot * alpha
            focal_weight = (1.0 - probs) ** self.gamma
            loss = torch.sum(alpha_factor * focal_weight * (-log_probs), dim=1)
        else:
            targets = targets.float()
            if self.logits:
                ce_loss = F.binary_cross_entropy_with_logits(
                    inputs, targets, reduction="none"
                )
                probs = torch.sigmoid(inputs)
            else:
                ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
                probs = torch.clamp(inputs, min=1e-6, max=1 - 1e-6)

            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_factor = self.get_binary_alpha(targets, inputs.device)
            focal_weight = (1.0 - p_t) ** self.gamma
            loss = alpha_factor * focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def get_alpha(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return torch.ones_like(inputs)
        if isinstance(self.alpha, torch.Tensor):
            return self.alpha.to(inputs.device)
        alpha_tensor = torch.tensor(
            self.alpha, device=inputs.device, dtype=inputs.dtype
        )
        return alpha_tensor

    def get_binary_alpha(
        self, targets: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if self.alpha is None:
            return torch.ones_like(targets)
        if isinstance(self.alpha, (float, int)):
            return torch.where(targets == 1, self.alpha, 1 - float(self.alpha)).to(
                device
            )
        alpha_tensor = torch.tensor(self.alpha, device=device, dtype=targets.dtype)
        return torch.where(targets == 1, alpha_tensor, 1 - alpha_tensor)


class ClassBalancedFocalLoss(nn.Module):
    """
    Focal loss weighted by effective number of samples per class.
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    """

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-12)
        weights = weights / weights.sum() * len(weights)
        self.register_buffer("class_weights", weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = FocalLoss(
            gamma=self.gamma, alpha=self.class_weights, reduction="none", logits=True
        )
        loss = focal(inputs, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
