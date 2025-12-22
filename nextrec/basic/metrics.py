"""
Metrics computation and configuration for model evaluation.

Date: create on 27/10/2025
Checkpoint: edit on 20/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import logging
from typing import Any

import numpy as np
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

CLASSIFICATION_METRICS = {
    "auc",
    "gauc",
    "ks",
    "logloss",
    "accuracy",
    "acc",
    "precision",
    "recall",
    "f1",
    "micro_f1",
    "macro_f1",
}
REGRESSION_METRICS = {"mse", "mae", "rmse", "r2", "mape", "msle"}
TASK_DEFAULT_METRICS = {
    "binary": ["auc", "gauc", "ks", "logloss", "accuracy", "precision", "recall", "f1"],
    "regression": ["mse", "mae", "rmse", "r2", "mape"],
    "multilabel": ["auc", "hamming_loss", "subset_accuracy", "micro_f1", "macro_f1"],
    "matching": ["auc", "gauc", "precision@10", "hitrate@10", "map@10", "cosine"]
    + [f"recall@{k}" for k in (5, 10, 20)]
    + [f"ndcg@{k}" for k in (5, 10, 20)]
    + [f"mrr@{k}" for k in (5, 10, 20)],
}


def check_user_id(*metric_sources: Any) -> bool:
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
            metric_names.add(item.lower())
            continue
        try:
            stack.extend(item)
        except TypeError:
            continue
    for name in metric_names:
        if name == "gauc":
            return True
        if name.startswith(
            ("recall@", "precision@", "hitrate@", "hr@", "mrr@", "ndcg@", "map@")
        ):
            return True
    return False


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
        return float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )
    return 0.0


def compute_msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Log Error."""
    y_pred_pos = np.maximum(y_pred, 0)
    return float(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))


def compute_gauc(
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray | None = None
) -> float:
    if user_ids is None:
        # If no user_ids provided, fall back to regular AUC
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception as e:
            logging.warning(
                f"[Metrics Warning: GAUC] Failed to compute AUC without user_ids: {e}"
            )
            return 0.0
    # Group by user_id and calculate AUC for each user
    user_aucs = []
    user_weights = []
    unique_users = np.unique(user_ids)
    for user_id in unique_users:
        mask = user_ids == user_id
        user_y_true = y_true[mask]
        user_y_pred = y_pred[mask]
        # Skip users with only one class (cannot compute AUC)
        if len(np.unique(user_y_true)) < 2:
            continue
        user_auc = roc_auc_score(user_y_true, user_y_pred)
        user_aucs.append(user_auc)
        user_weights.append(len(user_y_true))

    if len(user_aucs) == 0:
        return 0.0
    # Weighted average
    user_aucs = np.array(user_aucs)
    user_weights = np.array(user_weights)
    gauc = float(np.sum(user_aucs * user_weights) / np.sum(user_weights))
    return gauc


def group_indices_by_user(user_ids: np.ndarray, n_samples: int) -> list[np.ndarray]:
    """Group sample indices by user_id. If user_ids is None, treat all as one group."""
    if user_ids is None:
        return [np.arange(n_samples)]
    user_ids = np.asarray(user_ids)
    if user_ids.shape[0] != n_samples:
        logging.warning(
            f"[Metrics Warning: GAUC] user_ids length {user_ids.shape[0]} != number of samples {n_samples}, treating all samples as a single group for ranking metrics."
        )
        return [np.arange(n_samples)]
    unique_users = np.unique(user_ids)
    groups = [np.where(user_ids == u)[0] for u in unique_users]
    return groups


def compute_precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Compute Precision@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: Precision@K] user_ids must be provided for Precision@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
    precisions = []
    for idx in groups:
        if idx.size == 0:
            continue
        k_user = min(k, idx.size)
        scores = y_pred[idx]
        labels = y_true[idx]
        order = np.argsort(scores)[::-1]
        topk = order[:k_user]
        hits = labels[topk].sum()
        precisions.append(hits / float(k_user))
    return float(np.mean(precisions)) if precisions else 0.0


def compute_recall_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Compute Recall@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: Recall@K] user_ids must be provided for Recall@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
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
        k_user = min(k, idx.size)
        topk = order[:k_user]
        hits = labels[topk].sum()
        recalls.append(hits / float(num_pos))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_hitrate_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Compute HitRate@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: HitRate@K] user_ids must be provided for HitRate@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
    hits_per_user = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        if labels.sum() == 0:
            continue  # dont count users with no positive labels
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_user = min(k, idx.size)
        topk = order[:k_user]
        hits = labels[topk].sum()
        hits_per_user.append(1.0 if hits > 0 else 0.0)
    return float(np.mean(hits_per_user)) if hits_per_user else 0.0


def compute_mrr_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Compute MRR@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: MRR@K] user_ids must be provided for MRR@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
    mrrs = []
    for idx in groups:
        if idx.size == 0:
            continue
        labels = y_true[idx]
        if labels.sum() == 0:
            continue
        scores = y_pred[idx]
        order = np.argsort(scores)[::-1]
        k_user = min(k, idx.size)
        topk = order[:k_user]
        ranked_labels = labels[topk]
        rr = 0.0
        for rank, lab in enumerate(ranked_labels[:k_user], start=1):
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
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Compute NDCG@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: NDCG@K] user_ids must be provided for NDCG@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
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
    y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray, k: int
) -> float:
    """Mean Average Precision@K."""
    if user_ids is None:
        raise ValueError(
            "[Metrics Error: MAP@K] user_ids must be provided for MAP@K computation."
        )
    y_true = (y_true > 0).astype(int)
    n = len(y_true)
    groups = group_indices_by_user(user_ids, n)
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
        k_user = min(k, idx.size)
        hits = 0
        sum_precisions = 0.0
        for rank, i in enumerate(order[:k_user], start=1):
            if labels[i] > 0:
                hits += 1
                sum_precisions += hits / float(rank)
        if hits == 0:
            aps.append(0.0)
        else:
            aps.append(sum_precisions / float(num_pos))
    return float(np.mean(aps)) if aps else 0.0


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
    task: str | list[str],  # 'binary' or ['binary', 'regression']
    metrics: (
        list[str] | dict[str, list[str]] | None
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
            lowered = [m.lower() for m in task_metrics]
            task_specific_metrics[task_name] = lowered
            for metric in lowered:
                if metric not in metrics_list:
                    metrics_list.append(metric)
    elif metrics:
        metrics_list = [m.lower() for m in metrics]
    else:
        # No user provided metrics, derive per task type
        if nums_task > 1 and isinstance(task, list):
            deduped: list[str] = []
            for task_type in task:
                # Inline get_default_metrics_for_task logic
                if task_type not in TASK_DEFAULT_METRICS:
                    raise ValueError(f"Unsupported task type: {task_type}")
                for metric in TASK_DEFAULT_METRICS[task_type]:
                    if metric not in deduped:
                        deduped.append(metric)
            metrics_list = deduped
        else:
            # Inline get_default_metrics_for_task logic
            if primary_task not in TASK_DEFAULT_METRICS:
                raise ValueError(f"Unsupported task type: {primary_task}")
            metrics_list = TASK_DEFAULT_METRICS[primary_task]
    if not metrics_list:
        # Inline get_default_metrics_for_task logic
        if primary_task not in TASK_DEFAULT_METRICS:
            raise ValueError(f"Unsupported task type: {primary_task}")
        metrics_list = TASK_DEFAULT_METRICS[primary_task]
    best_metrics_mode = getbest_metric_mode(metrics_list[0], primary_task)
    return metrics_list, task_specific_metrics, best_metrics_mode


def getbest_metric_mode(first_metric: str, primary_task: str) -> str:
    """Determine if metric should be maximized or minimized."""
    first_metric_lower = first_metric.lower()
    # Metrics that should be maximized
    if first_metric_lower in {
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
    if (
        first_metric_lower.startswith("recall@")
        or first_metric_lower.startswith("precision@")
        or first_metric_lower.startswith("hitrate@")
        or first_metric_lower.startswith("hr@")
        or first_metric_lower.startswith("mrr@")
        or first_metric_lower.startswith("ndcg@")
        or first_metric_lower.startswith("map@")
    ):
        return "max"
    # Cosine separation should be maximized
    if first_metric_lower == "cosine":
        return "max"
    # Metrics that should be minimized
    if first_metric_lower in {"logloss", "mse", "mae", "rmse", "mape", "msle"}:
        return "min"
    # Default based on task type
    if primary_task == "regression":
        return "min"
    return "max"


def compute_single_metric(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    user_ids: np.ndarray | None = None,
) -> float:
    """Compute a single metric given true and predicted values."""

    y_p_binary = (y_pred > 0.5).astype(int)
    metric_lower = metric.lower()
    try:
        if metric_lower.startswith("recall@"):
            k = int(metric_lower.split("@")[1])
            return compute_recall_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        if metric_lower.startswith("precision@"):
            k = int(metric_lower.split("@")[1])
            return compute_precision_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        if metric_lower.startswith("hitrate@") or metric_lower.startswith("hr@"):
            k_str = metric_lower.split("@")[1]
            k = int(k_str)
            return compute_hitrate_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        if metric_lower.startswith("mrr@"):
            k = int(metric_lower.split("@")[1])
            return compute_mrr_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        if metric_lower.startswith("ndcg@"):
            k = int(metric_lower.split("@")[1])
            return compute_ndcg_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        if metric_lower.startswith("map@"):
            k = int(metric_lower.split("@")[1])
            return compute_map_at_k(y_true, y_pred, user_ids, k)  # type: ignore
        # cosine for matching task
        if metric_lower == "cosine":
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
            value = float(compute_gauc(y_true, y_pred, user_ids))
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
                    average="samples" if task_type == "multilabel" else "binary",
                    zero_division=0,
                )
            )
        elif metric == "recall":
            value = float(
                recall_score(
                    y_true,
                    y_p_binary,
                    average="samples" if task_type == "multilabel" else "binary",
                    zero_division=0,
                )
            )
        elif metric == "f1":
            value = float(
                f1_score(
                    y_true,
                    y_p_binary,
                    average="samples" if task_type == "multilabel" else "binary",
                    zero_division=0,
                )
            )
        elif metric == "micro_f1":
            value = float(
                f1_score(y_true, y_p_binary, average="micro", zero_division=0)
            )
        elif metric == "macro_f1":
            value = float(
                f1_score(y_true, y_p_binary, average="macro", zero_division=0)
            )
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
            logging.warning(
                f"[Metric Warning] Metric '{metric}' is not supported, returning 0.0"
            )
            value = 0.0
    except Exception as exception:
        logging.warning(
            f"[Metric Warning] Failed to compute metric {metric}: {exception}"
        )
        value = 0.0
    return value


def evaluate_metrics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    metrics: list[str],  # example: ['auc', 'logloss']
    task: str | list[str],  # example: 'binary' or ['binary', 'regression']
    target_names: list[str],  # example: ['target1', 'target2']
    task_specific_metrics: (
        dict[str, list[str]] | None
    ) = None,  # example: {'target1': ['auc', 'logloss'], 'target2': ['mse']}
    user_ids: np.ndarray | None = None,  # example: User IDs for GAUC computation
) -> dict:  # {'auc': 0.75, 'logloss': 0.45, 'mse_target2': 3.2}
    """Evaluate specified metrics for given true and predicted values."""

    result = {}
    if y_true is None or y_pred is None:
        return result
    # Main evaluation logic
    primary_task = task[0] if isinstance(task, list) else task
    nums_task = len(task) if isinstance(task, list) else 1
    # Single task evaluation
    if nums_task == 1:
        for metric in metrics:
            metric_lower = metric.lower()
            value = compute_single_metric(
                metric_lower, y_true, y_pred, primary_task, user_ids
            )
            result[metric_lower] = value
    # Multi-task evaluation
    else:
        for metric in metrics:
            metric_lower = metric.lower()
            for task_idx in range(nums_task):
                # Check if metric should be computed for given task
                should_compute = True
                if task_specific_metrics is not None and task_idx < len(target_names):
                    task_name = target_names[task_idx]
                    should_compute = metric_lower in task_specific_metrics.get(
                        task_name, []
                    )
                else:
                    # Get task type for specific index
                    if isinstance(task, list) and task_idx < len(task):
                        task_type = task[task_idx]
                    elif isinstance(task, str):
                        task_type = task
                    else:
                        task_type = "binary"
                    if task_type in ["binary", "multilabel"]:
                        should_compute = metric_lower in {
                            "auc",
                            "gauc",
                            "ks",
                            "logloss",
                            "accuracy",
                            "acc",
                            "precision",
                            "recall",
                            "f1",
                            "micro_f1",
                            "macro_f1",
                        }
                    elif task_type == "regression":
                        should_compute = metric_lower in {
                            "mse",
                            "mae",
                            "rmse",
                            "r2",
                            "mape",
                            "msle",
                        }
                if not should_compute:
                    continue
                target_name = target_names[task_idx]
                # Get task type for specific index
                if isinstance(task, list) and task_idx < len(task):
                    task_type = task[task_idx]
                elif isinstance(task, str):
                    task_type = task
                else:
                    task_type = "binary"
                y_true_task = y_true[:, task_idx]
                y_pred_task = y_pred[:, task_idx]
                # Compute metric
                value = compute_single_metric(
                    metric_lower, y_true_task, y_pred_task, task_type, user_ids
                )
                result[f"{metric_lower}_{target_name}"] = value
    return result
