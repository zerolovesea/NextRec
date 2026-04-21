"""
Trainer module for NextRec models, providing training loop with support for
various optimizers, schedulers, loss functions, and metrics.

Date: create on 16/04/2025
Checkpoint: edit on 21/04/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nextrec.basic.loggers import colorize, format_kv
from nextrec.basic.metrics import compute_confusion_matrix, evaluate_metrics, get_thresholds, needs_group_ids
from nextrec.data.dataloader import RecDataLoader
from nextrec.data.data_processing import get_group_ids
from nextrec.utils.console import display_metrics_table, render_confusion_block
from nextrec.utils.torch_utils import smart_inference_mode
from nextrec.utils.types import MetricsName


class BaseValidator:
    @smart_inference_mode()
    def evaluate(
        self,
        data: dict | pd.DataFrame | DataLoader,
        metrics: list[MetricsName] | dict[str, list[MetricsName]] | None = None,
        batch_size: int = 32,
        group_ids: np.ndarray | None = None,
        group_id: str | None = None,
        group_by: str | list[str] | None = None,
        num_workers: int = 0,
        thresholds: float | dict[str, float] | list[float] | None = None,
        show_data_summary: bool = False,
        show_confusion_matrix: bool = False,
    ) -> dict:
        """
        Evaluate the model on the given data.

        Args:
            data: Evaluation data (dict/df/DataLoader).
            metrics: Metrics names or per-target dict. e.g. {'target1': ['auc', 'logloss'], 'target2': ['mse']} or ['auc', 'logloss'].
            batch_size: Batch size (per process when distributed).
            group_ids: Optional grouping IDs for GAUC and ranking metrics. If None and needed, values are extracted from data using group_id.
            group_id: Column name for grouping IDs if group_ids is not provided. e.g. 'user_id'
            group_by: Optional column name(s) for grouped evaluation. When provided,
                returns a dict with `overall` and `grouped` results.
            num_workers: DataLoader worker count.
            thresholds: Threshold(s) for binary metrics/confusion matrix. Supports a single
                float for all targets, a list aligned to target order, or a dict keyed by
                target name. Defaults to 0.5. e.g. 0.5, [0.5, 0.7], {'target1': 0.5, 'target2': 0.7}
            show_data_summary: If True, log data summary statistics.
            show_confusion_matrix: If True, render confusion matrix blocks to logs.
        """

        self.eval()
        eval_metrics = metrics if metrics is not None else self.metrics
        if eval_metrics is None:
            raise ValueError(
                "[BaseModel-evaluate Error] No metrics specified for evaluation. Please provide metrics parameter or call fit() first."
            )
        task_specific_metrics = self.task_specific_metrics
        require_group_ids = needs_group_ids(eval_metrics, task_specific_metrics)
        primary_family = self.model_family
        group_by_columns = [group_by] if isinstance(group_by, str) else [str(name) for name in (group_by or [])]
        if require_group_ids and group_ids is None and not group_id:
            raise ValueError(
                "[BaseModel-evaluate Error] group_id must be specified when grouped metrics such as GAUC or ranking@K are enabled."
            )

        if isinstance(data, DataLoader):
            data_loader = data
        else:
            if group_ids is None and require_group_ids:
                group_ids = get_group_ids(data=data, group_id=group_id)
            eval_key_columns = list(
                dict.fromkeys(
                    [
                        *(self.key_columns or []),
                        *(group_by_columns or []),
                        *([group_id] if require_group_ids and group_ids is None and group_id else []),
                    ]
                )
            )
            if group_by_columns or (require_group_ids and group_ids is None):
                rec_loader = RecDataLoader(
                    dense_features=self.dense_features,
                    sparse_features=self.sparse_features,
                    sequence_features=self.sequence_features,
                    target=self.target_columns,
                    key_columns=eval_key_columns,
                    processor=None,
                )
                data_loader = rec_loader.create_dataloader(
                    data=data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
            else:
                data_loader = self.prepare_data_loader(
                    data,
                    batch_size=batch_size,
                    shuffle=False,
                    key_columns=eval_key_columns,
                    num_workers=num_workers,
                )
        y_true_list = []
        y_pred_list = []
        collected_group_ids = []
        collected_groups = {name: [] for name in group_by_columns}
        with torch.no_grad():
            for batch_data in data_loader:
                X_input, y_true = self.get_input(batch_data, require_labels=True)
                y_pred = self.training_adapter.forward(self, X_input)
                if (
                    primary_family != "sequential"
                    and isinstance(y_pred, torch.Tensor)
                    and y_true is not None
                    and y_pred.dim() <= 2
                    and y_true.dim() <= 2
                ):
                    y_true_list.append(y_true.cpu().numpy())
                    y_pred_list.append(y_pred.cpu().numpy())
                if require_group_ids and group_ids is None:
                    if not group_id:
                        raise ValueError(
                            "[BaseModel-evaluate Error] group_id must be specified when grouped metrics are enabled."
                        )
                    batch_group_id = get_group_ids(data=batch_data, group_id=group_id)
                    if batch_group_id is not None:
                        collected_group_ids.append(batch_group_id)
                if group_by_columns:
                    keys_dict = batch_data.get("keys") or {}
                    for column in group_by_columns:
                        if column not in keys_dict:
                            raise KeyError(
                                f"[BaseModel-evaluate Error] group_by column '{column}' not found in evaluation data."
                            )
                        group_values = keys_dict[column]
                        group_np = (
                            group_values.detach().cpu().numpy()
                            if isinstance(group_values, torch.Tensor)
                            else np.asarray(group_values)
                        )
                        collected_groups[column].append(group_np.reshape(group_np.shape[0]))

        y_true_all_local = np.concatenate(y_true_list, axis=0) if y_true_list else None
        y_pred_all_local = np.concatenate(y_pred_list, axis=0) if y_pred_list else None

        if isinstance(eval_metrics, dict):
            unique_metrics = []
            for task_metrics in eval_metrics.values():
                for metric in task_metrics:
                    if metric not in unique_metrics:
                        unique_metrics.append(metric)
            metrics_to_use = unique_metrics
        else:
            metrics_to_use = eval_metrics
        final_group_ids_local = group_ids
        if final_group_ids_local is None and collected_group_ids:
            final_group_ids_local = np.concatenate(collected_group_ids, axis=0)

        y_true_all = y_true_all_local
        y_pred_all = y_pred_all_local
        final_group_ids = final_group_ids_local if require_group_ids else None
        group_arrays = (
            {name: np.concatenate(values, axis=0) if values else None for name, values in collected_groups.items()}
            if group_by_columns
            else {}
        )
        if y_true_all is None or y_pred_all is None or len(y_true_all) == 0 or len(y_pred_all) == 0:
            logging.info(
                colorize("  Warning: Not enough evaluation data to compute metrics after gathering", color="yellow")
            )
            return {"overall": {}, "grouped": []}

        logging.info("")
        metrics_dict = evaluate_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            metrics=metrics_to_use,
            task=self.task,
            model_family=self.model_family,
            target_names=self.target_columns,
            task_specific_metrics=task_specific_metrics,
            group_ids=final_group_ids,
            ignore_label=self.ignore_label,
            thresholds=thresholds,
        )

        if group_by_columns:
            logging.info("")
            logging.info(colorize("[Metrics]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))
            display_metrics_table(
                epoch=1,
                epochs=1,
                split="Eval",
                loss=None,
                metrics=metrics_dict,
                target_names=self.target_columns,
                base_metrics=(metrics_to_use if isinstance(metrics_to_use, list) else None),
                colorize=lambda s: colorize(s),
            )

            group_frame = pd.DataFrame(group_arrays)
            groupby_key = group_by_columns[0] if len(group_by_columns) == 1 else group_by_columns
            grouped_results = []
            for key, group in group_frame.groupby(groupby_key, sort=False, dropna=False):
                indices = group.index.to_numpy()
                group_group_ids = final_group_ids[indices] if final_group_ids is not None else None
                group_metrics = evaluate_metrics(
                    y_true=y_true_all[indices],
                    y_pred=y_pred_all[indices],
                    metrics=metrics_to_use,
                    task=self.task,
                    model_family=self.model_family,
                    target_names=self.target_columns,
                    task_specific_metrics=task_specific_metrics,
                    group_ids=group_group_ids,
                    ignore_label=self.ignore_label,
                    thresholds=thresholds,
                )
                key_tuple = key if isinstance(key, tuple) else (key,)
                row = {column: value for column, value in zip(group_by_columns, key_tuple)}
                row["samples"] = int(len(indices))
                row.update(group_metrics)
                grouped_results.append(row)

            logging.info("")
            logging.info(colorize("[Grouped Metrics]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))
            for row in grouped_results:
                group_label = ", ".join(f"{name}={row[name]}" for name in group_by_columns)
                row_metrics = {
                    key: value for key, value in row.items() if key not in set(group_by_columns) | {"samples"}
                }
                display_metrics_table(
                    epoch=1,
                    epochs=1,
                    split=f"Eval[{group_label}]",
                    loss=None,
                    metrics=row_metrics,
                    target_names=self.target_columns,
                    base_metrics=(metrics_to_use if isinstance(metrics_to_use, list) else None),
                    colorize=lambda s: colorize(s),
                )

            return {"overall": metrics_dict, "grouped": grouped_results}

        if show_data_summary:
            logging.info("")
            logging.info(colorize("[Data Summary]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))
            logging.info(format_kv("Samples", len(y_true_all)))

            task_types = list(self.task) if isinstance(self.task, list) else [self.task]
            if len(task_types) != len(self.target_columns):
                task_types = [task_types[0]] * len(self.target_columns)

            if y_true_all.ndim == 1 or (y_true_all.ndim == 2 and y_true_all.shape[1] == 1):
                y_true_matrix = y_true_all.reshape(-1, 1) if y_true_all.ndim == 1 else y_true_all
            else:
                y_true_matrix = y_true_all

            for idx, (target_name, task_type) in enumerate(zip(self.target_columns, task_types)):
                if idx >= y_true_matrix.shape[1]:
                    continue
                values = y_true_matrix[:, idx]
                if self.ignore_label is not None:
                    values = values[values != self.ignore_label]
                if values.size == 0:
                    continue
                logging.info("")
                logging.info(colorize(f"{target_name}", color="cyan", bold=True))
                if task_type == "regression":
                    values = values.astype(float)
                    stats = {
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values),
                        "min": np.nanmin(values),
                        "p25": np.nanpercentile(values, 25),
                        "p50": np.nanpercentile(values, 50),
                        "p75": np.nanpercentile(values, 75),
                        "max": np.nanmax(values),
                    }
                    stat_text = ", ".join(f"{key}={value:.6g}" for key, value in stats.items())
                    logging.info(format_kv("stats", stat_text, indent=2))
                else:
                    uniques, counts = np.unique(values, return_counts=True)
                    total = counts.sum()
                    for label_value, count in zip(uniques, counts):
                        label_str = (
                            f"{int(label_value)}"
                            if isinstance(label_value, (int, np.integer))
                            or (
                                isinstance(label_value, (float, np.floating))
                                and np.isclose(label_value, int(label_value))
                            )
                            else f"{label_value}"
                        )
                        ratio = count / total if total else 0.0
                        logging.info(format_kv(label_str, f"{count} ({ratio:.2%})", indent=2))

        if show_data_summary or show_confusion_matrix:
            logging.info("")
            logging.info(colorize("[Metrics]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))
            metrics_for_table = {k: v for k, v in metrics_dict.items() if not k.startswith("confusion_matrix")}
            if metrics_for_table:
                display_metrics_table(
                    epoch=1,
                    epochs=1,
                    split="Eval",
                    loss=None,
                    metrics=metrics_for_table,
                    target_names=self.target_columns,
                    base_metrics=(metrics_to_use if isinstance(metrics_to_use, list) else None),
                    colorize=lambda s: colorize(s),
                )
            else:
                logging.info(colorize("[BaseModel-evaluate Warning] No metrics computed.", color="yellow"))

        if show_confusion_matrix:
            task = self.task
            target_names = self.target_columns
            task_types = task if isinstance(task, list) else [task] * max(1, len(target_names))
            thresholds_by_target = get_thresholds(thresholds, target_names)

            logging.info("")
            logging.info(colorize("[Confusion Matrix]", color="cyan", bold=True))
            logging.info(colorize("-" * 80, color="cyan"))

            if y_true_all.ndim == 1 or (y_true_all.ndim == 2 and y_true_all.shape[1] == 1):
                if task_types[0] == "binary":
                    y_true_vec = y_true_all.reshape(-1) if y_true_all.ndim == 1 else y_true_all[:, 0]
                    y_pred_vec = y_pred_all.reshape(-1) if y_pred_all.ndim == 1 else y_pred_all[:, 0]
                    cm = compute_confusion_matrix(
                        y_true_vec,
                        y_pred_vec,
                        self.ignore_label,
                        threshold=thresholds_by_target.get(target_names[0], 0.5),
                    )
                    if cm is not None:
                        metrics_dict["confusion_matrix"] = cm
                        logging.info(
                            colorize(
                                f"{target_names[0]} (threshold={thresholds_by_target.get(target_names[0], 0.5)})",
                                color="cyan",
                                bold=True,
                            )
                        )
                        rendered = render_confusion_block(tn=cm["tn"], fp=cm["fp"], fn=cm["fn"], tp=cm["tp"])
                        for line in rendered.splitlines():
                            logging.info(line)
            else:
                for idx, target_name in enumerate(target_names):
                    task_type = task_types[idx] if idx < len(task_types) else "binary"
                    if task_type != "binary":
                        continue
                    y_true_vec = y_true_all[:, idx]
                    y_pred_vec = y_pred_all[:, idx]
                    cm = compute_confusion_matrix(
                        y_true_vec,
                        y_pred_vec,
                        self.ignore_label,
                        threshold=thresholds_by_target.get(target_name, 0.5),
                    )
                    if cm is not None:
                        metrics_dict[f"confusion_matrix_{target_name}"] = cm
                        logging.info("")
                        logging.info(
                            colorize(
                                f"{target_name} (threshold={thresholds_by_target.get(target_name, 0.5)})",
                                color="cyan",
                                bold=True,
                            )
                        )
                        rendered = render_confusion_block(tn=cm["tn"], fp=cm["fp"], fn=cm["fn"], tp=cm["tp"])
                        for line in rendered.splitlines():
                            logging.info(line)

        return {"overall": metrics_dict, "grouped": []}
