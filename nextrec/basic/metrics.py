"""
Metrics computation and configuration for model evaluation.

Date: create on 27/10/2025
Checkpoint: edit on 13/03/2026
Author: Yang Zhou,zyaztec@gmail.com
"""

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from nextrec.utils.types import MetricsName, ModelFamilyName, TaskTypeName

TASK_DEFAULT_METRICS = {
    "binary": ["auc", "gauc", "ks", "logloss", "accuracy", "precision", "recall", "f1"],
    "regression": ["mse", "mae", "rmse", "r2", "mape"],
    "generative": ["hitrate@10"]
    + [f"recall@{k}" for k in (5, 10, 20)]
    + [f"ndcg@{k}" for k in (5, 10, 20)]
    + [f"mrr@{k}" for k in (5, 10, 20)],
}

MODEL_FAMILY_DEFAULT_METRICS = {
    "matching": ["auc", "gauc", "precision@10", "hitrate@10", "map@10", "cosine"]
    + [f"recall@{k}" for k in (5, 10, 20)]
    + [f"ndcg@{k}" for k in (5, 10, 20)]
    + [f"mrr@{k}" for k in (5, 10, 20)],
    "sequential": ["hitrate@10"]
    + [f"recall@{k}" for k in (5, 10, 20)]
    + [f"ndcg@{k}" for k in (5, 10, 20)]
    + [f"mrr@{k}" for k in (5, 10, 20)],
}

RANKING_METRIC_PREFIXES = (
    "recall@",
    "precision@",
    "hitrate@",
    "hr@",
    "mrr@",
    "ndcg@",
    "map@",
    "topk_recall@",
    "topk_precision@",
    "lift@",
)

TASK_EXTRA_ALLOW_METRICS = {
    "binary": {"acc", "micro_f1", "macro_f1"},
    "regression": {"msle"},
}

TASK_METRIC_ALLOWLIST = {
    task_name: set(metrics) | TASK_EXTRA_ALLOW_METRICS.get(task_name, set())
    for task_name, metrics in TASK_DEFAULT_METRICS.items()
}


def needs_group_metrics(metric_names: list[MetricsName]) -> bool:
    for metric_name in metric_names:
        if metric_name == "gauc" or is_ranking_metric(metric_name):
            return True
    return False


def needs_group_ids(*metric_sources: Any) -> bool:
    """Return True when GAUC or ranking@K metrics appear in the provided sources."""
    metric_names = set()
    stack = list(metric_sources)
    while stack:
        item = stack.pop()
        if not item:
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if isinstance(item, str):
            metric_names.add(item)
            continue
        try:
            stack.extend(item)
        except TypeError:
            continue
    return needs_group_metrics(list(metric_names))


def flatten_metric_names(metrics: list[MetricsName] | dict[str, list[MetricsName]]) -> list[str]:
    """Flatten metric names while preserving order and removing duplicates."""
    if isinstance(metrics, dict):
        flattened: list[str] = []
        for task_metrics in metrics.values():
            for metric in task_metrics:
                if metric not in flattened:
                    flattened.append(metric)
        return flattened
    return list(metrics)


def is_ranking_metric(metric: str) -> bool:
    return metric.startswith(RANKING_METRIC_PREFIXES)


def is_sequential_ranking_metric(metric: str) -> bool:
    return is_ranking_metric(metric)


def parse_metric_suffix(metric: str) -> int:
    try:
        return int(metric.split("@", 1)[1])
    except Exception as exc:
        raise ValueError(f"[Metrics Error] Invalid metric name '{metric}'.") from exc


def compute_sequential_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute 1-based ranks of the true label against full-vocab logits.

    Args:
        logits: [N, V]
        labels: [N]
    Returns:
        ranks: [N], where 1 means the true item is ranked first.
    """
    if logits.dim() != 2:
        raise ValueError(f"[Metrics Error] Sequential logits must be 2D, got shape {tuple(logits.shape)}.")
    if labels.dim() != 1:
        raise ValueError(f"[Metrics Error] Sequential labels must be 1D, got shape {tuple(labels.shape)}.")
    true_scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (logits > true_scores.unsqueeze(1)).sum(dim=1) + 1


def accumulate_sequential_ranking_metrics(
    metric_sums: dict[str, float],
    metric_names: list[str],
    ranks: torch.Tensor,
    vocab_size: int,
) -> None:
    """
    Accumulate sequential ranking metric numerators for a batch.

    The caller is responsible for dividing by the number of valid samples.
    """
    if ranks.numel() == 0:
        return

    ranks_f = ranks.to(dtype=torch.float32)
    for metric in metric_names:
        if metric.startswith(("hitrate@", "hr@", "recall@")):
            k = parse_metric_suffix(metric)
            metric_sums[metric] += float((ranks <= k).float().sum().item())
        elif metric.startswith("precision@"):
            k = parse_metric_suffix(metric)
            k_eff = min(max(k, 1), vocab_size)
            metric_sums[metric] += float(((ranks <= k).float() / float(k_eff)).sum().item())
        elif metric.startswith("mrr@") or metric.startswith("map@"):
            k = parse_metric_suffix(metric)
            contribution = torch.where(
                ranks <= k,
                1.0 / ranks_f,
                torch.zeros_like(ranks_f),
            )
            metric_sums[metric] += float(contribution.sum().item())
        elif metric.startswith("ndcg@"):
            k = parse_metric_suffix(metric)
            contribution = torch.where(
                ranks <= k,
                1.0 / torch.log2(ranks_f + 1.0),
                torch.zeros_like(ranks_f),
            )
            metric_sums[metric] += float(contribution.sum().item())
        elif metric.startswith("topk_recall@"):
            k_percent = parse_metric_suffix(metric)
            k_count = min(vocab_size, max(int(np.ceil(vocab_size * (k_percent / 100.0))), 0))
            metric_sums[metric] += float((ranks <= k_count).float().sum().item())
        elif metric.startswith("topk_precision@"):
            k_percent = parse_metric_suffix(metric)
            k_count = min(vocab_size, max(int(np.ceil(vocab_size * (k_percent / 100.0))), 0))
            if k_count > 0:
                metric_sums[metric] += float(((ranks <= k_count).float() / float(k_count)).sum().item())
        elif metric.startswith("lift@"):
            k_percent = parse_metric_suffix(metric)
            k_count = min(vocab_size, max(int(np.ceil(vocab_size * (k_percent / 100.0))), 0))
            if k_count > 0:
                contribution = torch.where(
                    ranks <= k_count,
                    float(vocab_size) / float(k_count),
                    torch.zeros_like(ranks_f),
                )
                metric_sums[metric] += float(contribution.sum().item())


def compute_sequential_metric_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metrics: list[str] | dict[str, list[str]],
    padding_idx: int,
    ignore_label: int | float | None = None,
) -> tuple[dict[str, float], int, int | None]:
    """
    Compute batch-level sequential ranking metric numerators and valid sample count.

    Returns:
        metric_sums: Per-metric numerator contributions for the batch.
        valid_count: Number of valid next-item positions.
        vocab_size: Logit vocabulary size for the batch, or None when no valid samples exist.
    """

    metric_names = flatten_metric_names(metrics)
    metric_names = [metric for metric in metric_names if is_sequential_ranking_metric(metric)]
    metric_sums = {metric: 0.0 for metric in metric_names}
    if not metric_names:
        return metric_sums, 0, None

    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    elif logits.dim() != 2:
        raise ValueError(f"[Metrics Error] Sequential logits must have 2 or 3 dims, got shape {tuple(logits.shape)}.")

    labels = labels.reshape(-1)
    valid_mask = labels.ne(padding_idx)
    if ignore_label is not None:
        valid_mask = valid_mask & labels.ne(int(ignore_label))
    if not torch.any(valid_mask):
        return metric_sums, 0, None

    logits = logits[valid_mask]
    labels = labels[valid_mask].long()
    ranks = compute_sequential_ranks(logits, labels)
    vocab_size = int(logits.size(1))
    accumulate_sequential_ranking_metrics(metric_sums, metric_names, ranks, vocab_size=vocab_size)
    return metric_sums, int(ranks.numel()), vocab_size


def compute_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    if n_pos > 0 and n_neg > 0:
        cum_pos_rate = np.cumsum(y_true_sorted == 1) / n_pos
        cum_neg_rate = np.cumsum(y_true_sorted == 0) / n_neg
        ks_value = np.max(np.abs(cum_pos_rate - cum_neg_rate))
        return float(ks_value)
    return 0.0


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return 0.0


def compute_msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Log Error."""
    y_pred_pos = np.maximum(y_pred, 0)
    return float(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))


def compute_gauc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray | None = None,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    if group_ids is None:
        # If no group_ids provided, fall back to regular AUC
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception as e:
            logging.warning(f"[Metrics Warning: GAUC] Failed to compute AUC without group_ids: {e}")
            return 0.0
    # Group by group_id and calculate AUC for each group
    group_aucs = []
    group_weights = []
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, len(y_true))
    for idx in groups:
        if idx.size == 0:
            continue
        group_y_true = y_true[idx]
        group_y_pred = y_pred[idx]
        # Skip groups with only one class (cannot compute AUC)
        if len(np.unique(group_y_true)) < 2:
            continue
        group_auc = roc_auc_score(group_y_true, group_y_pred)
        group_aucs.append(group_auc)
        group_weights.append(len(group_y_true))

    if len(group_aucs) == 0:
        return 0.0
    # Weighted average
    group_aucs = np.array(group_aucs)
    group_weights = np.array(group_weights)
    gauc = float(np.sum(group_aucs * group_weights) / np.sum(group_weights))
    return gauc


def group_indices_by_group(group_ids: np.ndarray, n_samples: int) -> list[np.ndarray]:
    """Group sample indices by group_id. If group_ids is None, treat all as one group."""
    if group_ids is None:
        return [np.arange(n_samples)]
    group_ids = np.asarray(group_ids)
    if group_ids.shape[0] != n_samples:
        logging.warning(
            f"[Metrics Warning: GAUC] group_ids length {group_ids.shape[0]} != number of samples {n_samples}, treating all samples as a single group for ranking metrics."
        )
        return [np.arange(n_samples)]
    if n_samples == 0:
        return []
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    groups = np.split(order, boundaries)
    return groups


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute Precision@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: Precision@K] group_ids must be provided for Precision@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    precisions = []
    for idx in groups:
        if idx.size == 0:
            continue
        k_group = min(k, idx.size)
        scores = y_pred[idx]
        labels = y_true[idx]
        order = np.argsort(scores)[::-1]
        topk = order[:k_group]
        hits = labels[topk].sum()
        precisions.append(hits / float(k_group))
    return float(np.mean(precisions)) if precisions else 0.0


def compute_recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute Recall@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: Recall@K] group_ids must be provided for Recall@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    recalls = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        num_pos = labels.sum()
        if num_pos == 0:
            continue  # dont count users with no positive labels
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_group = min(k, idx.size)
        topk = order[:k_group]
        hits = labels[topk].sum()
        recalls.append(hits / float(num_pos))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_hitrate_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute HitRate@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: HitRate@K] group_ids must be provided for HitRate@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    hits_per_user = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        if labels.sum() == 0:
            continue  # dont count users with no positive labels
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_group = min(k, idx.size)
        topk = order[:k_group]
        hits = labels[topk].sum()
        hits_per_user.append(1.0 if hits > 0 else 0.0)
    return float(np.mean(hits_per_user)) if hits_per_user else 0.0


def compute_mrr_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute MRR@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: MRR@K] group_ids must be provided for MRR@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    mrrs = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        if labels.sum() == 0:
            continue
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_group = min(k, idx.size)
        topk = order[:k_group]
        ranked_labels = labels[topk]
        rr = 0.0
        for rank, lab in enumerate(ranked_labels[:k_group], start=1):
            if lab > 0:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    return float(np.mean(mrrs)) if mrrs else 0.0


def compute_dcg_at_k(labels: np.ndarray, k: int) -> float:
    k_user = min(k, labels.size)
    if k_user == 0:
        return 0.0
    gains = (2 ** labels[:k_user] - 1).astype(float)
    discounts = np.log2(np.arange(2, k_user + 2))
    return float(np.sum(gains / discounts))


def compute_ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute NDCG@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: NDCG@K] group_ids must be provided for NDCG@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    ndcgs = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        if labels.sum() == 0:
            continue
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        ranked_labels = labels[order]
        dcg = compute_dcg_at_k(ranked_labels, k)
        # ideal DCG
        ideal_labels = np.sort(labels)[::-1]
        idcg = compute_dcg_at_k(ideal_labels, k)
        if idcg == 0.0:
            continue
        ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_map_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Mean Average Precision@K."""
    if group_ids is None:
        raise ValueError("[Metrics Error: MAP@K] group_ids must be provided for MAP@K computation.")
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices if group_indices is not None else group_indices_by_group(group_ids, n)
    aps = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        num_pos = labels.sum()
        if num_pos == 0:
            continue
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_group = min(k, idx.size)
        hits = 0
        sum_precisions = 0.0
        for rank, i in enumerate(order[:k_group], start=1):
            if labels[i] > 0:
                hits += 1
                sum_precisions += hits / float(rank)
        if hits == 0:
            aps.append(0.0)
        else:
            aps.append(sum_precisions / float(num_pos))
    return float(np.mean(aps)) if aps else 0.0


def compute_topk_counts(y_true: np.ndarray, y_pred: np.ndarray, k_percent: int) -> tuple[int, int, int]:
    """Compute Top-K% sample size, hits, and positives for binary labels."""
    # Ensure 1D to avoid argsort on 2D returning column-wise indices
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = (y_true > 0).astype(int)
    n = y_true.size
    if n == 0:
        return 0, 0, 0
    if k_percent <= 0:
        return 0, 0, int(y_true.sum())
    if k_percent >= 100:
        k_count = n
    else:
        k_count = int(np.ceil(n * (k_percent / 100.0)))
        k_count = max(k_count, 1)
    order = np.argsort(y_pred)[::-1]
    topk = order[:k_count]
    hits = int(y_true[topk].sum())
    total_pos = int(y_true.sum())
    return k_count, hits, total_pos


def compute_topk_precision(y_true: np.ndarray, y_pred: np.ndarray, k_percent: int) -> float:
    """Compute Top-K% Precision."""
    k_count, hits, _ = compute_topk_counts(y_true, y_pred, k_percent)
    if k_count == 0:
        return 0.0
    return float(hits / k_count)


def compute_topk_recall(y_true: np.ndarray, y_pred: np.ndarray, k_percent: int) -> float:
    """Compute Top-K% Recall."""
    _, hits, total_pos = compute_topk_counts(y_true, y_pred, k_percent)
    if total_pos == 0:
        return 0.0
    return float(hits / total_pos)


def compute_lift_at_k(y_true: np.ndarray, y_pred: np.ndarray, k_percent: int) -> float:
    """Compute Lift@K from Top-K% precision and overall rate."""
    k_count, hits, total_pos = compute_topk_counts(y_true, y_pred, k_percent)
    if k_count == 0:
        return 0.0
    base_rate = total_pos / float(y_true.size)
    if base_rate == 0.0:
        return 0.0
    precision = hits / float(k_count)
    return float(precision / base_rate)


def compute_cosine_separation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cosine Separation."""
    y_true = (y_true > 0).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    if not np.any(pos_mask) or not np.any(neg_mask):
        return 0.0

    pos_mean = float(np.mean(y_pred[pos_mask]))
    neg_mean = float(np.mean(y_pred[neg_mask]))
    return pos_mean - neg_mean


def configure_metrics(
    task: TaskTypeName | list[TaskTypeName],
    model_family: ModelFamilyName,
    metrics: (
        list[MetricsName] | dict[str, list[MetricsName]] | None
    ),  # ['auc', 'logloss'] or {'task1': ['auc'], 'task2': ['mse']}
    target_names: list[str],  # ['target1', 'target2']
) -> tuple[list[str], dict[str, list[str]] | None, str]:
    """Configure metrics based on task and user input."""

    primary_task = task[0] if isinstance(task, list) else task
    nums_task = len(task) if isinstance(task, list) else 1
    metrics_list = []
    task_specific_metrics = None

    if isinstance(metrics, dict):
        metrics_list = []
        task_specific_metrics = {}
        for task_name, task_metrics in metrics.items():
            if task_name not in target_names:
                logging.warning(
                    f"[Metrics Warning] Task {task_name} not found in targets {target_names}, skipping its metrics"
                )
                continue
            task_specific_metrics[task_name] = task_metrics
        metrics_list = flatten_metric_names(task_specific_metrics)
    elif metrics:
        metrics_list = flatten_metric_names(metrics)
    else:
        if model_family in MODEL_FAMILY_DEFAULT_METRICS:
            metrics_list = list(MODEL_FAMILY_DEFAULT_METRICS[model_family])
        elif nums_task > 1 and isinstance(task, list):
            deduped: list[str] = []
            for task_type in task:
                if task_type not in TASK_DEFAULT_METRICS:
                    raise ValueError(f"Unsupported task: {task_type}")
                for metric in TASK_DEFAULT_METRICS[task_type]:
                    if metric not in deduped:
                        deduped.append(metric)
            metrics_list = deduped
        else:
            if primary_task not in TASK_DEFAULT_METRICS:
                raise ValueError(f"Unsupported task: {primary_task}")
            metrics_list = TASK_DEFAULT_METRICS[primary_task]
    if not metrics_list:
        if model_family in MODEL_FAMILY_DEFAULT_METRICS:
            metrics_list = list(MODEL_FAMILY_DEFAULT_METRICS[model_family])
        else:
            if primary_task not in TASK_DEFAULT_METRICS:
                raise ValueError(f"Unsupported task: {primary_task}")
            metrics_list = TASK_DEFAULT_METRICS[primary_task]
    best_metrics_mode = get_best_metric_mode(metrics_list[0], primary_task)
    return metrics_list, task_specific_metrics, best_metrics_mode


def get_best_metric_mode(first_metric: MetricsName, primary_task: TaskTypeName) -> str:
    """Determine if metric should be maximized or minimized."""
    # Metrics that should be maximized
    if first_metric in {
        "auc",
        "gauc",
        "ks",
        "accuracy",
        "acc",
        "precision",
        "recall",
        "f1",
        "r2",
        "micro_f1",
        "macro_f1",
    }:
        return "max"
    # Ranking metrics that should be maximized (with @K suffix)
    if is_ranking_metric(first_metric):
        return "max"
    # Cosine separation should be maximized
    if first_metric == "cosine":
        return "max"
    # Metrics that should be minimized
    if first_metric in {"logloss", "mse", "mae", "rmse", "mape", "msle"}:
        return "min"
    # Default based on task type
    if primary_task == "regression":
        return "min"
    return "max"


def compute_single_metric(
    metric: MetricsName,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    group_ids: np.ndarray | None = None,
    threshold: float = 0.5,
    group_indices: list[np.ndarray] | None = None,
) -> float:
    """Compute a single metric given true and predicted values."""

    if y_true.size == 0:
        return 0.0

    y_p_binary = (y_pred > threshold).astype(int)
    try:
        if metric.startswith("topk_recall@"):
            k_percent = parse_metric_suffix(metric)
            return compute_topk_recall(y_true, y_pred, k_percent)
        if metric.startswith("topk_precision@"):
            k_percent = parse_metric_suffix(metric)
            return compute_topk_precision(y_true, y_pred, k_percent)
        if metric.startswith("lift@"):
            k_percent = parse_metric_suffix(metric)
            return compute_lift_at_k(y_true, y_pred, k_percent)
        if metric.startswith("recall@"):
            k = parse_metric_suffix(metric)
            return compute_recall_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        if metric.startswith("precision@"):
            k = parse_metric_suffix(metric)
            return compute_precision_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        if metric.startswith("hitrate@") or metric.startswith("hr@"):
            k = parse_metric_suffix(metric)
            return compute_hitrate_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        if metric.startswith("mrr@"):
            k = parse_metric_suffix(metric)
            return compute_mrr_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        if metric.startswith("ndcg@"):
            k = parse_metric_suffix(metric)
            return compute_ndcg_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        if metric.startswith("map@"):
            k = parse_metric_suffix(metric)
            return compute_map_at_k(y_true, y_pred, group_ids, k, group_indices=group_indices)  # type: ignore
        # cosine for matching task
        if metric == "cosine":
            return compute_cosine_separation(y_true, y_pred)
        if metric == "auc":
            value = float(
                roc_auc_score(
                    y_true,
                    y_pred,
                    average="macro" if task_type == "multilabel" else None,
                )
            )
        elif metric == "gauc":
            value = float(compute_gauc(y_true, y_pred, group_ids, group_indices=group_indices))
        elif metric == "ks":
            value = float(compute_ks(y_true, y_pred))
        elif metric == "logloss":
            value = float(log_loss(y_true, y_pred))
        elif metric in ("accuracy", "acc"):
            value = float(accuracy_score(y_true, y_p_binary))
        elif metric == "precision":
            value = float(
                precision_score(
                    y_true,
                    y_p_binary,
                    average=("samples" if task_type == "multilabel" else "binary"),
                    zero_division=0,
                )
            )
        elif metric == "recall":
            value = float(
                recall_score(
                    y_true,
                    y_p_binary,
                    average=("samples" if task_type == "multilabel" else "binary"),
                    zero_division=0,
                )
            )
        elif metric == "f1":
            value = float(
                f1_score(
                    y_true,
                    y_p_binary,
                    average=("samples" if task_type == "multilabel" else "binary"),
                    zero_division=0,
                )
            )
        elif metric == "micro_f1":
            value = float(f1_score(y_true, y_p_binary, average="micro", zero_division=0))
        elif metric == "macro_f1":
            value = float(f1_score(y_true, y_p_binary, average="macro", zero_division=0))
        elif metric == "mse":
            value = float(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            value = float(mean_absolute_error(y_true, y_pred))
        elif metric == "rmse":
            value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "r2":
            value = float(r2_score(y_true, y_pred))
        elif metric == "mape":
            value = float(compute_mape(y_true, y_pred))
        elif metric == "msle":
            value = float(compute_msle(y_true, y_pred))
        else:
            logging.warning(f"[Metric Warning] Metric '{metric}' is not supported, returning 0.0")
            value = 0.0
    except Exception as exception:
        logging.warning(f"[Metric Warning] Failed to compute metric {metric}: {exception}")
        value = 0.0
    return value


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: int | float | None = None,
    threshold: float = 0.5,
) -> dict[str, int] | None:
    """
    Compute confusion matrix for binary classification.

    Returns:
        dict with keys tn/fp/fn/tp, or None if unavailable.
    """
    y_true_vec = np.asarray(y_true).reshape(-1)
    y_pred_vec = np.asarray(y_pred).reshape(-1)
    if ignore_label is not None:
        mask = y_true_vec != ignore_label
        if not np.any(mask):
            return None
        y_true_vec = y_true_vec[mask]
        y_pred_vec = y_pred_vec[mask]
    y_pred_bin = (y_pred_vec > threshold).astype(int)
    tn = int(np.sum((y_true_vec == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_vec == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_vec == 1) & (y_pred_bin == 0)))
    tp = int(np.sum((y_true_vec == 1) & (y_pred_bin == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def get_thresholds(
    thresholds: float | dict[str, float] | list[float] | None,
    target_names: list[str],
) -> dict[str, float]:
    """
    Resolve threshold(s) per target name.
    """
    if thresholds is None:
        return {name: 0.5 for name in target_names}
    if isinstance(thresholds, (int, float)):
        return {name: float(thresholds) for name in target_names}
    if isinstance(thresholds, dict):
        thresholds_map = {name: 0.5 for name in target_names}
        for name, value in thresholds.items():
            if name in thresholds_map:
                thresholds_map[name] = float(value)
        return thresholds_map
    if isinstance(thresholds, list):
        thresholds_map = {name: 0.5 for name in target_names}
        for idx, name in enumerate(target_names):
            if idx < len(thresholds):
                thresholds_map[name] = float(thresholds[idx])
        return thresholds_map
    return {name: 0.5 for name in target_names}


def evaluate_metrics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    metrics: list[MetricsName],
    task: TaskTypeName | list[TaskTypeName],
    model_family: ModelFamilyName,
    target_names: list[str],
    task_specific_metrics: dict[str, list[MetricsName]] | None = None,
    group_ids: np.ndarray | None = None,
    ignore_label: int | float | None = None,
    thresholds: float | dict[str, float] | list[float] | None = None,
) -> dict:
    """
    Evaluate specified metrics for given true and predicted values.
    Supports single-task and multi-task evaluation.
    Handles optional ignore_label to exclude certain samples.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted values.
        metrics: List of metric names to compute.
        task: Task type(s) - 'binary', 'regression', etc.
        model_family: Model family used to interpret metric defaults and task style.
        target_names: Names of target variables. e.g., ['target1', 'target2']
        task_specific_metrics: Optional dict mapping target names to specific metrics. e.g., {'target1': ['auc', 'logloss'], 'target2': ['mse']}
        group_ids: Optional grouping IDs for GAUC and ranking metrics.
        ignore_label: Optional label value to ignore during evaluation.
        thresholds: Threshold(s) for binary metrics. Supports a float, a list aligned to
            target order, or a dict keyed by target name. Defaults to 0.5.

    Returns: Dictionary of computed metric values. {'auc': 0.75, 'logloss': 0.45, 'mse_target2': 3.2}

    """

    result = {}

    if y_true is None or y_pred is None:
        return result
    # Main evaluation logic
    primary_task = task[0] if isinstance(task, list) else task
    nums_task = len(task) if isinstance(task, list) else 1
    # Single task evaluation
    thresholds_by_target = get_thresholds(thresholds, target_names)
    if nums_task == 1:
        if ignore_label is not None:
            valid_mask = y_true != ignore_label
            if np.any(valid_mask):
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                if group_ids is not None:
                    group_ids = group_ids[valid_mask]
            else:
                return result
        group_indices = (
            group_indices_by_group(group_ids, len(y_true))
            if group_ids is not None and needs_group_metrics(metrics)
            else None
        )
        for metric in metrics:
            threshold = thresholds_by_target.get(target_names[0], 0.5)
            value = compute_single_metric(
                metric,
                y_true,
                y_pred,
                "multilabel" if model_family == "multitask" and nums_task > 1 else primary_task,
                group_ids,
                threshold=threshold,
                group_indices=group_indices,
            )
            result[metric] = value
    # Multi-task evaluation
    else:
        task_types = []
        for task_idx in range(nums_task):
            if isinstance(task, list) and task_idx < len(task):
                task_types.append(task[task_idx])
            elif isinstance(task, str):
                task_types.append(task)
            else:
                task_types.append("binary")
        for task_idx in range(nums_task):
            task_type = task_types[task_idx]
            target_name = target_names[task_idx]
            if task_specific_metrics is not None and task_idx < len(target_names):
                allowed_metrics = {m for m in task_specific_metrics.get(target_name, [])}
            else:
                allowed_metrics = TASK_METRIC_ALLOWLIST.get(task_type)
            y_true_task = y_true[:, task_idx]
            y_pred_task = y_pred[:, task_idx]
            task_group_ids = group_ids
            if ignore_label is not None:
                valid_mask = y_true_task != ignore_label
                if np.any(valid_mask):
                    y_true_task = y_true_task[valid_mask]
                    y_pred_task = y_pred_task[valid_mask]
                    if task_group_ids is not None:
                        task_group_ids = task_group_ids[valid_mask]
                else:
                    for metric in metrics:
                        if allowed_metrics is not None and metric not in allowed_metrics:
                            if not is_ranking_metric(metric):
                                continue
                        result[f"{metric}_{target_name}"] = 0.0
                    continue
            task_metrics_to_compute = [
                m for m in metrics if allowed_metrics is None or m in allowed_metrics or is_ranking_metric(m)
            ]
            need_group_for_task = needs_group_metrics(task_metrics_to_compute)
            group_indices_task = (
                group_indices_by_group(task_group_ids, len(y_true_task))
                if task_group_ids is not None and need_group_for_task
                else None
            )
            for metric in metrics:
                if allowed_metrics is not None and metric not in allowed_metrics:
                    if is_ranking_metric(metric):
                        pass
                    else:
                        continue
                # Compute metric
                threshold = thresholds_by_target.get(target_name, 0.5)
                value = compute_single_metric(
                    metric,
                    y_true_task,
                    y_pred_task,
                    task_type,
                    task_group_ids,
                    threshold=threshold,
                    group_indices=group_indices_task,
                )
                result[f"{metric}_{target_name}"] = value
    return result
