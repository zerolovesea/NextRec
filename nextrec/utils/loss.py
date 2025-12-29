"""
Loss utilities for NextRec.

Date: create on 28/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.loss.listwise import (
    ApproxNDCGLoss,
    InfoNCELoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
)
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.loss.pointwise import ClassBalancedFocalLoss, FocalLoss, WeightedBCELoss
from nextrec.utils.types import LossName


def normalize_task_loss(
    task_loss,
    valid_count,
    total_count,
    eps=1e-8,
) -> torch.Tensor:
    if not torch.is_tensor(valid_count):
        valid_count = torch.tensor(float(valid_count), device=task_loss.device)
    if not torch.is_tensor(total_count):
        total_count = torch.tensor(float(total_count), device=task_loss.device)
    scale = valid_count.to(task_loss.dtype) / (total_count.to(task_loss.dtype) + eps)
    return task_loss * scale


def build_cb_focal(kw):
    if "class_counts" not in kw:
        raise ValueError("class_balanced_focal requires class_counts")
    return ClassBalancedFocalLoss(**kw)


def get_loss_fn(
    loss: LossName | None | nn.Module = None,
    **kw,
) -> nn.Module:
    """
    Get loss function by name or return the provided loss module.

    Args:
        loss: Loss function name or nn.Module instance. Supported options:

            **Pointwise Losses:**
            - "bce", "binary_crossentropy": Binary Cross-Entropy Loss
            - "weighted_bce": Weighted Binary Cross-Entropy Loss
            - "focal", "focal_loss": Focal Loss (for class imbalance)
            - "cb_focal", "class_balanced_focal": Class-Balanced Focal Loss (requires class_counts parameter)
            - "crossentropy", "ce": Cross-Entropy Loss for multi-class classification
            - "mse": Mean Squared Error Loss
            - "mae": Mean Absolute Error Loss

            **Pairwise Ranking Losses:**
            - "bpr": Bayesian Personalized Ranking Loss
            - "hinge": Hinge Loss
            - "triplet": Triplet Loss

            **Listwise Ranking Losses:**
            - "sampled_softmax", "softmax": Sampled Softmax Loss
            - "infonce": InfoNCE Loss
            - "listnet": ListNet Loss
            - "listmle": ListMLE Loss
            - "approx_ndcg": Approximate NDCG Loss

        **kw: Additional keyword arguments passed to the loss function

    Returns:
        nn.Module: Loss function instance

    Raises:
        ValueError: If loss is None or unsupported type

    Examples:
        >>> loss_fn = get_loss_fn("bce")
        >>> loss_fn = get_loss_fn("focal", alpha=0.25, gamma=2.0)
        >>> loss_fn = get_loss_fn("cb_focal", class_counts=[100, 50, 200])
    """
    if isinstance(loss, nn.Module):
        return loss
    if loss is None:
        raise ValueError("[Loss Error] loss must be provided explicitly")
    if loss in ["bce", "binary_crossentropy"]:
        return nn.BCELoss(**kw)
    if loss == "weighted_bce":
        return WeightedBCELoss(**kw)
    if loss in ["focal", "focal_loss"]:
        return FocalLoss(**kw)
    if loss in ["cb_focal", "class_balanced_focal"]:
        return build_cb_focal(kw)
    if loss in ["crossentropy", "ce"]:
        return nn.CrossEntropyLoss(**kw)
    if loss == "mse":
        return nn.MSELoss(**kw)
    if loss == "mae":
        return nn.L1Loss(**kw)

    # Pairwise ranking Loss
    if loss == "bpr":
        return BPRLoss(**kw)
    if loss == "hinge":
        return HingeLoss(**kw)
    if loss == "triplet":
        return TripletLoss(**kw)

    # Listwise ranking Loss
    if loss in ["sampled_softmax", "softmax"]:
        return SampledSoftmaxLoss(**kw)
    if loss == "infonce":
        return InfoNCELoss(**kw)
    if loss == "listnet":
        return ListNetLoss(**kw)
    if loss == "listmle":
        return ListMLELoss(**kw)
    if loss == "approx_ndcg":
        return ApproxNDCGLoss(**kw)

    raise ValueError(f"[Loss Error] Unsupported loss: {loss}")


def get_loss_kwargs(loss_params: dict | list[dict] | None, index: int = 0) -> dict:
    """
    Parse loss_kwargs for each head.

    - loss_params is None       -> {}
    - loss_params is dict       -> shared by all heads
    - loss_params is list[dict] -> use loss_params[index] (if exists and not None), else {}
    """
    if loss_params is None:
        return {}
    if isinstance(loss_params, list):
        if index < len(loss_params) and loss_params[index] is not None:
            return loss_params[index]
        return {}
    return loss_params
