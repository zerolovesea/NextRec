"""
Loss utilities for NextRec.
"""

from typing import Literal

import torch.nn as nn

from nextrec.loss.listwise import (
    ApproxNDCGLoss,
    InfoNCELoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
)
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.loss.pointwise import (
    ClassBalancedFocalLoss,
    CosineContrastiveLoss,
    FocalLoss,
    WeightedBCELoss,
)

# Valid task types for validation
VALID_TASK_TYPES = [
    "binary",
    "multiclass",
    "regression",
    "multivariate_regression",
    "match",
    "ranking",
    "multitask",
    "multilabel",
]


def get_loss_fn(
    task_type: str = "binary",
    training_mode: str | None = None,
    loss: str | nn.Module | None = None,
    **loss_kwargs,
) -> nn.Module:
    """
    Get loss function based on task type and training mode.
    """

    if isinstance(loss, nn.Module):
        return loss

    # Common mappings
    if task_type == "match":
        return _get_match_loss(training_mode, loss, **loss_kwargs)

    if task_type in ["ranking", "multitask", "binary", "multilabel"]:
        return _get_classification_loss(loss, **loss_kwargs)

    if task_type == "multiclass":
        return _get_multiclass_loss(loss, **loss_kwargs)

    if task_type == "regression":
        if loss is None or loss == "mse":
            return nn.MSELoss(**loss_kwargs)
        if loss == "mae":
            return nn.L1Loss(**loss_kwargs)
        if isinstance(loss, str):
            raise ValueError(f"Unsupported regression loss: {loss}")

    raise ValueError(f"Unsupported task_type: {task_type}")


def _get_match_loss(training_mode: str | None, loss: str | None, **loss_kwargs) -> nn.Module:
    if training_mode == "pointwise":
        if loss is None or loss in {"bce", "binary_crossentropy"}:
            return nn.BCELoss(**loss_kwargs)
        if loss == "weighted_bce":
            return WeightedBCELoss(**loss_kwargs)
        if loss == "focal":
            return FocalLoss(**loss_kwargs)
        if loss == "class_balanced_focal":
            return _build_cb_focal(loss_kwargs)
        if loss == "cosine_contrastive":
            return CosineContrastiveLoss(**loss_kwargs)
        if isinstance(loss, str):
            raise ValueError(f"Unsupported pointwise loss: {loss}")

    if training_mode == "pairwise":
        if loss is None or loss == "bpr":
            return BPRLoss(**loss_kwargs)
        if loss == "hinge":
            return HingeLoss(**loss_kwargs)
        if loss == "triplet":
            return TripletLoss(**loss_kwargs)
        if isinstance(loss, str):
            raise ValueError(f"Unsupported pairwise loss: {loss}")

    if training_mode == "listwise":
        if loss is None or loss in {"sampled_softmax", "softmax"}:
            return SampledSoftmaxLoss(**loss_kwargs)
        if loss == "infonce":
            return InfoNCELoss(**loss_kwargs)
        if loss == "listnet":
            return ListNetLoss(**loss_kwargs)
        if loss == "listmle":
            return ListMLELoss(**loss_kwargs)
        if loss == "approx_ndcg":
            return ApproxNDCGLoss(**loss_kwargs)
        if loss in {"crossentropy", "ce"}:
            return nn.CrossEntropyLoss(**loss_kwargs)
        if isinstance(loss, str):
            raise ValueError(f"Unsupported listwise loss: {loss}")

    raise ValueError(f"Unknown training_mode: {training_mode}")


def _get_classification_loss(loss: str | None, **loss_kwargs) -> nn.Module:
    if loss is None or loss in {"bce", "binary_crossentropy"}:
        return nn.BCELoss(**loss_kwargs)
    if loss == "weighted_bce":
        return WeightedBCELoss(**loss_kwargs)
    if loss == "focal":
        return FocalLoss(**loss_kwargs)
    if loss == "class_balanced_focal":
        return _build_cb_focal(loss_kwargs)
    if loss == "mse":
        return nn.MSELoss(**loss_kwargs)
    if loss == "mae":
        return nn.L1Loss(**loss_kwargs)
    if loss in {"crossentropy", "ce"}:
        return nn.CrossEntropyLoss(**loss_kwargs)
    if isinstance(loss, str):
        raise ValueError(f"Unsupported loss function: {loss}")
    raise ValueError("Loss must be specified for classification task.")


def _get_multiclass_loss(loss: str | None, **loss_kwargs) -> nn.Module:
    if loss is None or loss in {"crossentropy", "ce"}:
        return nn.CrossEntropyLoss(**loss_kwargs)
    if loss == "focal":
        return FocalLoss(**loss_kwargs)
    if loss == "class_balanced_focal":
        return _build_cb_focal(loss_kwargs)
    if isinstance(loss, str):
        raise ValueError(f"Unsupported multiclass loss: {loss}")
    raise ValueError("Loss must be specified for multiclass task.")


def _build_cb_focal(loss_kwargs: dict) -> ClassBalancedFocalLoss:
    if "class_counts" not in loss_kwargs:
        raise ValueError("class_balanced_focal requires `class_counts` argument.")
    return ClassBalancedFocalLoss(**loss_kwargs)


def get_loss_kwargs(loss_params: dict | list[dict] | None, index: int = 0) -> dict:
    """
    Resolve per-task loss kwargs from a dict or list of dicts.
    """
    if loss_params is None:
        return {}
    if isinstance(loss_params, list):
        if index < len(loss_params) and loss_params[index] is not None:
            return loss_params[index]
        return {}
    return loss_params
