"""
Model-related utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 24/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from collections import OrderedDict

import torch

from nextrec.loss import (
    ApproxNDCGLoss,
    BPRLoss,
    HingeLoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
    TripletLoss,
)


def merge_features(primary, secondary) -> list:
    merged: OrderedDict[str, object] = OrderedDict()
    for feat in list(primary or []) + list(secondary or []):
        merged.setdefault(feat.name, feat)
    return list(merged.values())


def get_mlp_output_dim(params: dict, fallback: int) -> int:
    dims = params.get("dims")
    if dims:
        return dims[-1]
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
