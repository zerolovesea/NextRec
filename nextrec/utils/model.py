"""
Model-related utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 29/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from collections import OrderedDict

import torch
from torch import nn

from nextrec.loss import (
    ApproxNDCGLoss,
    BPRLoss,
    HingeLoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
    TripletLoss,
)
from nextrec.utils.types import (
    LossName,
    OptimizerName,
    SchedulerName,
    TrainingModeName,
    TaskTypeName,
    MetricsName,
)


def merge_features(primary, secondary) -> list:
    merged: OrderedDict[str, object] = OrderedDict()
    for feat in list(primary or []) + list(secondary or []):
        merged.setdefault(feat.name, feat)
    return list(merged.values())


def get_mlp_output_dim(params: dict, fallback: int) -> int:
    hidden_dims = params.get("hidden_dims")
    if hidden_dims:
        return hidden_dims[-1]
    return fallback


def select_features(
    available_features: list,
    names: list[str],
    param_name: str,
) -> list:
    if not names:
        return []

    if len(names) != len(set(names)):
        raise ValueError(f"{param_name} contains duplicate feature names: {names}")

    feature_map = {feat.name: feat for feat in available_features}
    missing = [name for name in names if name not in feature_map]
    if missing:
        raise ValueError(
            f"{param_name} contains unknown feature names {missing}. "
            f"Available features: {list(feature_map)}"
        )

    return [feature_map[name] for name in names]


def compute_pair_scores(model, data, batch_size: int = 512):
    user_emb = model.encode_user(data, batch_size=batch_size)
    item_emb = model.encode_item(data, batch_size=batch_size)
    with torch.no_grad():
        user_tensor = torch.as_tensor(user_emb, device=model.device)
        item_tensor = torch.as_tensor(item_emb, device=model.device)
        scores = model.compute_similarity(user_tensor, item_tensor)
        mode = model.training_mode
        if isinstance(mode, list):
            mode = mode[0] if mode else "pointwise"
        if mode == "pointwise":
            scores = torch.sigmoid(scores)
    return scores.detach().cpu().numpy()


def get_training_modes(
    training_mode,
    nums_task: int,
    valid_modes: set[str] | None = None,
) -> list:
    valid_modes = valid_modes or {"pointwise", "pairwise", "listwise"}
    if isinstance(training_mode, list):
        training_modes = list(training_mode)
        if len(training_modes) != nums_task:
            raise ValueError(
                "[BaseModel-init Error] training_mode list length must match number of tasks."
            )
    else:
        training_modes = [training_mode] * nums_task
    if any(mode not in valid_modes for mode in training_modes):
        raise ValueError(
            "[BaseModel-init Error] training_mode must be one of {'pointwise', 'pairwise', 'listwise'}."
        )
    return training_modes


def get_loss_list(
    loss,
    training_modes: list[str],
    nums_task: int,
    default_losses: dict[str, str],
):
    effective_loss = loss
    if effective_loss is None:
        loss_list = [default_losses[mode] for mode in training_modes]
    elif isinstance(effective_loss, list):
        if not effective_loss:
            loss_list = [default_losses[mode] for mode in training_modes]
        else:
            if len(effective_loss) != nums_task:
                raise ValueError(
                    f"[BaseModel-compile Error] Number of loss functions ({len(effective_loss)}) must match number of tasks ({nums_task})."
                )
            loss_list = list(effective_loss)
    else:
        loss_list = [effective_loss] * nums_task

    for idx, mode in enumerate(training_modes):
        if isinstance(loss_list[idx], str) and loss_list[idx] in {
            "bce",
            "binary_crossentropy",
        }:
            if mode in {"pairwise", "listwise"}:
                loss_list[idx] = default_losses[mode]
    return loss_list


def resolve_loss_weights(loss_weights, nums_task: int):
    if loss_weights is None:
        return None
    if nums_task == 1:
        if isinstance(loss_weights, (list, tuple)):
            if len(loss_weights) != 1:
                raise ValueError(
                    "[BaseModel-compile Error] loss_weights list must have exactly one element for single-task setup."
                )
            loss_weights = loss_weights[0]
        return [float(loss_weights)]
    if isinstance(loss_weights, (int, float)):
        weights = [float(loss_weights)] * nums_task
    elif isinstance(loss_weights, (list, tuple)):
        weights = [float(w) for w in loss_weights]
        if len(weights) != nums_task:
            raise ValueError(
                f"[BaseModel-compile Error] Number of loss_weights ({len(weights)}) must match number of tasks ({nums_task})."
            )
    else:
        raise TypeError(
            f"[BaseModel-compile Error] loss_weights must be int, float, list or tuple, got {type(loss_weights)}"
        )
    return weights


def prepare_ranking_targets(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if y_pred.dim() == 1:
        y_pred = y_pred.view(-1, 1)
    if y_true.dim() == 1:
        y_true = y_true.view(-1, 1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
    return y_pred, y_true


def split_pos_neg_scores(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.dim() != 2 or labels.dim() != 2:
        raise ValueError(
            "[Ranking Error] pairwise/listwise training requires 2D scores and labels."
        )
    list_size = scores.size(1)
    if list_size < 2:
        raise ValueError(
            "[Ranking Error] pairwise/listwise training requires list_size >= 2."
        )
    pos_mask = labels > 0
    pos_counts = pos_mask.sum(dim=1)
    neg_counts = list_size - pos_counts
    if not torch.all(pos_counts == 1).item():
        raise ValueError(
            "[Ranking Error] pairwise/listwise with pos/neg split requires exactly one positive per row."
        )
    if not torch.all(neg_counts == list_size - 1).item():
        raise ValueError(
            "[Ranking Error] pairwise/listwise with pos/neg split requires at least one negative per row."
        )
    pos_scores = scores[pos_mask].view(-1)
    neg_scores = scores[~pos_mask].view(scores.size(0), list_size - 1)
    return pos_scores, neg_scores


def compute_ranking_loss(
    training_mode: str,
    loss_fn: torch.nn.Module,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    y_pred, y_true = prepare_ranking_targets(y_pred, y_true)
    if training_mode == "pairwise":
        pos_scores, neg_scores = split_pos_neg_scores(y_pred, y_true)
        if isinstance(loss_fn, (BPRLoss, HingeLoss, SampledSoftmaxLoss)):
            loss = loss_fn(pos_scores, neg_scores)
        elif isinstance(loss_fn, TripletLoss):
            raise ValueError(
                "[Ranking Error] TripletLoss expects embeddings, not scalar scores."
            )
        else:
            loss = loss_fn(pos_scores, neg_scores)
    elif training_mode == "listwise":
        if isinstance(loss_fn, (ListNetLoss, ListMLELoss, ApproxNDCGLoss)):
            loss = loss_fn(y_pred, y_true)
        elif isinstance(loss_fn, SampledSoftmaxLoss):
            pos_scores, neg_scores = split_pos_neg_scores(y_pred, y_true)
            loss = loss_fn(pos_scores, neg_scores)
        else:
            loss = loss_fn(y_pred, y_true)
    else:
        raise ValueError(f"[Ranking Error] Unknown training mode: {training_mode}")

    return loss
