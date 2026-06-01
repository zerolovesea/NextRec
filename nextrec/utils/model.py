"""
Model-related utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
import numpy as np

from nextrec.loss.listwise import (
    ApproxNDCGLoss,
    ListMLELoss,
    ListNetLoss,
    SampledSoftmaxLoss,
)
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss


def get_mlp_output_dim(params: dict, fallback: int) -> int:
    hidden_dims = params.get("hidden_dims")
    if hidden_dims:
        return hidden_dims[-1]
    return fallback


def select_feature_objects(
    available_features: list,
    names: list[str],
    param_name: str,
) -> list:
    """select features by names from available features."""

    if not names:
        return []

    if len(names) != len(set(names)):
        raise ValueError(f"{param_name} contains duplicate feature names: {names}")

    feature_map = {feat.name: feat for feat in available_features}
    missing = [name for name in names if name not in feature_map]
    if missing:
        raise ValueError(
            f"{param_name} contains unknown feature names {missing}. " f"Available features: {list(feature_map)}"
        )

    return [feature_map[name] for name in names]


def compute_pair_scores(model, data, batch_size: int = 512):
    data_loader = model.prepare_feature_data(
        data=data,
        features=model.all_features,
        batch_size=batch_size,
    )
    score_batches = []
    with torch.no_grad():
        for batch_data in data_loader:
            x_input = model.build_feature_tensors(batch_data["features"], model.all_features)
            output = model.call_model(x_input)
            if isinstance(output, dict) and "scores" in output:
                scores = output["scores"]
            else:
                scores = output
            score_batches.append(scores.detach().cpu().numpy())
    if not score_batches:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(score_batches, axis=0)


def prepare_ranking_targets(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if y_pred.dim() == 1:
        y_pred = y_pred.view(-1, 1)
    if y_true.dim() == 1:
        y_true = y_true.view(-1, 1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
    return y_pred, y_true


def split_pos_neg_scores(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.dim() != 2 or labels.dim() != 2:
        raise ValueError("[Ranking Error] pairwise/listwise training requires 2D scores and labels.")
    list_size = scores.size(1)
    if list_size < 2:
        raise ValueError("[Ranking Error] pairwise/listwise training requires list_size >= 2.")
    pos_mask = labels > 0
    pos_counts = pos_mask.sum(dim=1)
    neg_counts = list_size - pos_counts
    if not torch.all(pos_counts == 1).item():
        raise ValueError("[Ranking Error] pairwise/listwise with pos/neg split requires exactly one positive per row.")
    if not torch.all(neg_counts == list_size - 1).item():
        raise ValueError("[Ranking Error] pairwise/listwise with pos/neg split requires at least one negative per row.")
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
            raise ValueError("[Ranking Error] TripletLoss expects embeddings, not scalar scores.")
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
