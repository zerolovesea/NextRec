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


VALID_TASK_TYPES = [
    "binary",      
    "multiclass", 
    "multilabel", 
    "regression",  
]

def _build_cb_focal(kw):
    if "class_counts" not in kw:
        raise ValueError("class_balanced_focal requires class_counts")
    return ClassBalancedFocalLoss(**kw)


def get_loss_fn(loss=None, **kw):
    if isinstance(loss, nn.Module):
        return loss
    if loss is None:
        raise ValueError("loss must be provided explicitly")
    if loss in ["bce", "binary_crossentropy"]:
        return nn.BCELoss(**kw)
    if loss == "weighted_bce":
        return WeightedBCELoss(**kw)
    if loss in ["focal", "focal_loss"]:
        return FocalLoss(**kw)
    if loss in ["cb_focal", "class_balanced_focal"]:
        return _build_cb_focal(kw)
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

    raise ValueError(f"Unsupported loss: {loss}")

def get_loss_kwargs(loss_params: dict | list[dict] | None, index: int = 0) -> dict:
    """
    解析每个 head 对应的 loss_kwargs。

    - loss_params 为 None       -> {}
    - loss_params 为 dict       -> 所有 head 共用
    - loss_params 为 list[dict] -> 用 loss_params[index]（若存在且非 None），否则 {}
    """
    if loss_params is None:
        return {}
    if isinstance(loss_params, list):
        if index < len(loss_params) and loss_params[index] is not None:
            return loss_params[index]
        return {}
    return loss_params