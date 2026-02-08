"""
Summary utilities for BaseModel.

Date: create on 03/12/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou,zyaztec@gmail.com
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import DataLoader

from nextrec.basic.loggers import colorize, format_kv
from nextrec.data.data_processing import get_column_data, get_data_length
from nextrec.utils.torch_utils import to_numpy
from nextrec.utils.types import TaskTypeName


class SummarySet:
    model_name: str
    dense_features: list[Any]
    sparse_features: list[Any]
    sequence_features: list[Any]
    task: TaskTypeName | list[TaskTypeName]
    target_columns: list[str]
    nums_task: int
    metrics: Any
    device: Any
    optimizer_name: str
    optimizer_params: dict[str, Any]
    scheduler_name: str | None
    scheduler_params: dict[str, Any]
    loss_config: Any
    loss_params: Any
    loss_weights: Any
    grad_norm: Any
    embedding_l1_reg: float
    embedding_l2_reg: float
    dense_l1_reg: float
    dense_l2_reg: float
    early_stop_patience: int
    max_gradient_norm: float | None
    session_id: str | None
    features_config_path: str
    checkpoint_path: str
    train_data_summary: dict[str, Any] | None
    valid_data_summary: dict[str, Any] | None
    note: str | None

    def collect_dataloader_summary(self, data_loader: DataLoader | None):
        if data_loader is None:
            return None

        summary = {
            "batch_size": data_loader.batch_size,
            "num_workers": data_loader.num_workers,
            "pin_memory": data_loader.pin_memory,
            "persistent_workers": data_loader.persistent_workers,
        }
        prefetch_factor = getattr(data_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            summary["prefetch_factor"] = prefetch_factor

        sampler = getattr(data_loader, "sampler", None)
        if sampler is not None:
            summary["sampler"] = sampler.__class__.__name__

        return summary or None

    def build_data_summary(self, data: Any, data_loader: DataLoader | None, sample_key: str):

        dataset = data_loader.dataset if data_loader is not None else None

        train_size = get_data_length(dataset)
        if train_size is None:
            train_size = get_data_length(data)

        labels = None
        if self.target_columns:
            for source in (dataset, data):
                if source is None:
                    continue
                label_source = source.labels if hasattr(source, "labels") else source  # type: ignore
                if not isinstance(label_source, (dict, pd.DataFrame, pl.DataFrame)):
                    continue
                label_map = {}
                for name in self.target_columns:
                    column = get_column_data(label_source, name)  # type: ignore
                    if column is None:
                        continue
                    label_map[name] = to_numpy(column)
                labels = label_map or None
                if labels:
                    break

        summary = {}
        if train_size is not None:
            summary[sample_key] = int(train_size)

        dataloader_summary = self.collect_dataloader_summary(data_loader)
        if dataloader_summary:
            summary["dataloader"] = dataloader_summary

        if labels:
            task_types = list(self.task) if isinstance(self.task, list) else [self.task]
            if len(task_types) != len(self.target_columns):
                task_types = [task_types[0]] * len(self.target_columns)

            label_distributions = {}
            for target_name, task_type in zip(self.target_columns, task_types):
                values = labels.get(target_name)
                if values is None:
                    continue
                values = np.asarray(values).reshape(-1)
                if values.size == 0:
                    continue
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
                    label_distributions[target_name] = {
                        "task": task_type,
                        "lines": [("stats", stat_text)],
                    }
                else:
                    uniques, counts = np.unique(values, return_counts=True)
                    total = counts.sum()
                    if total == 0:
                        continue
                    label_parts = []
                    for label_value, count in zip(uniques, counts):
                        if isinstance(label_value, (int, np.integer)):
                            label_str = f"{int(label_value)}"
                        elif isinstance(label_value, (float, np.floating)) and np.isclose(
                            label_value, int(label_value)
                        ):
                            label_str = f"{int(label_value)}"
                        else:
                            label_str = f"{label_value}"
                        ratio = count / total
                        label_parts.append((label_str, f"{count} ({ratio:.2%})"))
                    label_distributions[target_name] = {
                        "task": task_type,
                        "lines": label_parts,
                    }

            if label_distributions:
                summary["label_distributions"] = label_distributions

        return summary or None

    def build_train_data_summary(self, train_data: Any, train_loader: DataLoader | None):
        return self.build_data_summary(
            data=train_data,
            data_loader=train_loader,
            sample_key="train_samples",
        )

    def build_valid_data_summary(self, valid_data: Any, valid_loader: DataLoader | None):
        return self.build_data_summary(
            data=valid_data,
            data_loader=valid_loader,
            sample_key="valid_samples",
        )

    def summary(
        self,
        sections: list[Literal["feature", "model", "train", "data"]] | None = None,
    ):
        logger = logging.getLogger()
        allowed_sections = {
            "feature": "Feature Configuration",
            "model": "Model Parameters",
            "train": "Training Configuration",
            "data": "Data Summary",
        }
        if sections is None:
            selected_sections = set(allowed_sections.values())
        else:
            selected_sections = set()
            invalid_sections = []
            for section in sections:
                key = str(section).strip().lower()
                if key in allowed_sections:
                    selected_sections.add(allowed_sections[key])
                else:
                    invalid_sections.append(section)
            if invalid_sections:
                raise ValueError(
                    "[BaseModel-summary Error] Unknown summary section(s): "
                    f"{invalid_sections}. Allowed: {list(allowed_sections.keys())}"
                )

        logger.info("")
        logger.info(
            colorize(
                f"Model Summary: {self.model_name.upper()}",
                color="bright_blue",
                bold=True,
            )
        )
        logger.info("")

        if "Feature Configuration" in selected_sections:
            logger.info("")
            logger.info(colorize("Feature Configuration", color="cyan", bold=True))
            logger.info(colorize("-" * 80, color="cyan"))

            if self.dense_features:
                logger.info(f"Dense Features ({len(self.dense_features)}):")
                for i, feat in enumerate(self.dense_features, 1):
                    embed_dim = feat.embedding_dim if hasattr(feat, "embedding_dim") else 1
                    logger.info(f"  {i}. {feat.name:20s}")

            if self.sparse_features:
                logger.info(f"\nSparse Features ({len(self.sparse_features)}):")

                max_name_len = max(len(feat.name) for feat in self.sparse_features)
                max_embed_name_len = max(len(feat.embedding_name) for feat in self.sparse_features)
                name_width = max(max_name_len, 10) + 2
                embed_name_width = max(max_embed_name_len, 15) + 2

                logger.info(
                    f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10}"
                )
                logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10}")
                for i, feat in enumerate(self.sparse_features, 1):
                    vocab_size = feat.vocab_size if hasattr(feat, "vocab_size") else "N/A"
                    embed_dim = feat.embedding_dim if hasattr(feat, "embedding_dim") else "N/A"
                    logger.info(
                        f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10}"
                    )

            if self.sequence_features:
                logger.info(f"\nSequence Features ({len(self.sequence_features)}):")

                max_name_len = max(len(feat.name) for feat in self.sequence_features)
                max_embed_name_len = max(len(feat.embedding_name) for feat in self.sequence_features)
                name_width = max(max_name_len, 10) + 2
                embed_name_width = max(max_embed_name_len, 15) + 2

                logger.info(
                    f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10} {'Max Len':>10}"
                )
                logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10} {'-'*10}")
                for i, feat in enumerate(self.sequence_features, 1):
                    vocab_size = feat.vocab_size if hasattr(feat, "vocab_size") else "N/A"
                    embed_dim = feat.embedding_dim if hasattr(feat, "embedding_dim") else "N/A"
                    max_len = feat.max_len if hasattr(feat, "max_len") else "N/A"
                    logger.info(
                        f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10} {str(max_len):>10}"
                    )

        if "Model Parameters" in selected_sections:
            logger.info("")
            logger.info(colorize("Model Parameters", color="cyan", bold=True))
            logger.info(colorize("-" * 80, color="cyan"))

            # Model Architecture
            logger.info("Model Architecture:")
            logger.info(str(self))
            logger.info("")

            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params

            logger.info(f"Total Parameters:        {total_params:,}")
            logger.info(f"Trainable Parameters:    {trainable_params:,}")
            logger.info(f"Non-trainable Parameters: {non_trainable_params:,}")

            logger.info("Layer-wise Parameters:")
            for name, module in self.named_children():
                layer_params = sum(p.numel() for p in module.parameters())
                if layer_params > 0:
                    logger.info(f"  {name:30s}: {layer_params:,}")

        if "Training Configuration" in selected_sections:
            logger.info("")
            logger.info(colorize("Training Configuration", color="cyan", bold=True))
            logger.info(colorize("-" * 80, color="cyan"))

            logger.info(f"Task Type:               {self.task}")
            logger.info(f"Number of Tasks:         {self.nums_task}")
            logger.info(f"Metrics:                 {self.metrics}")
            logger.info(f"Target Columns:          {self.target_columns}")
            logger.info(f"Device:                  {self.device}")

            if hasattr(self, "optimizer_name"):
                logger.info(f"Optimizer:               {self.optimizer_name}")
                if self.optimizer_params:
                    for key, value in self.optimizer_params.items():
                        logger.info(f"  {key:25s}: {value}")

            if hasattr(self, "scheduler_name") and self.scheduler_name:
                logger.info(f"Scheduler:               {self.scheduler_name}")
                if self.scheduler_params:
                    for key, value in self.scheduler_params.items():
                        logger.info(f"  {key:25s}: {value}")

            if hasattr(self, "loss_config"):
                logger.info(f"Loss Function:           {self.loss_config}")

            loss_params_summary: list[str] = []
            loss_fn = getattr(self, "loss_fn", None)
            if loss_fn is not None:
                loss_modules = list(loss_fn) if isinstance(loss_fn, (list, tuple)) else [loss_fn]
                loss_config = getattr(self, "loss_config", None)
                if isinstance(loss_config, list):
                    loss_names = loss_config
                elif loss_config is not None:
                    loss_names = [loss_config] * len(loss_modules)
                else:
                    loss_names = [None] * len(loss_modules)

                loss_params = getattr(self, "loss_params", None)
                if isinstance(loss_params, list):
                    explicit_params = loss_params
                elif isinstance(loss_params, dict):
                    explicit_params = [loss_params] * len(loss_modules)
                else:
                    explicit_params = [None] * len(loss_modules)

                for idx, loss_module in enumerate(loss_modules):
                    params: dict[str, Any] = {}
                    explicit = explicit_params[idx] if idx < len(explicit_params) else None
                    if explicit:
                        params.update(explicit)
                    try:
                        signature = inspect.signature(loss_module.__class__.__init__)
                    except (TypeError, ValueError):
                        signature = None
                    if signature is not None:
                        for name, param in signature.parameters.items():
                            if name == "self" or name.startswith("_"):
                                continue
                            if hasattr(loss_module, name):
                                value = getattr(loss_module, name)
                                if callable(value):
                                    continue
                                params.setdefault(name, value)
                            elif param.default is not inspect._empty and param.default is not None:
                                params.setdefault(name, param.default)
                    if not params:
                        continue

                    loss_name = loss_names[idx] if idx < len(loss_names) else None
                    if len(loss_modules) > 1:
                        header = f"  [{idx}]"
                        if loss_name is not None:
                            header = f"{header} {loss_name}"
                        loss_params_summary.append(header)
                        indent = "    "
                    else:
                        indent = "  "
                    for key, value in params.items():
                        loss_params_summary.append(f"{indent}{key:25s}: {value}")

            if loss_params_summary:
                logger.info("Loss Params:")
                for line in loss_params_summary:
                    logger.info(line)
            if hasattr(self, "loss_weights"):
                logger.info(f"Loss Weights:            {self.loss_weights}")
            if hasattr(self, "grad_norm"):
                logger.info(f"GradNorm Enabled:        {self.grad_norm is not None}")
                if self.grad_norm is not None:
                    grad_lr = self.grad_norm.optimizer.param_groups[0].get("lr")
                    logger.info(f"  GradNorm alpha:        {self.grad_norm.alpha}")
                    logger.info(f"  GradNorm lr:           {grad_lr}")

            logger.info("Regularization:")
            logger.info(f"  Embedding L1:          {self.embedding_l1_reg}")
            logger.info(f"  Embedding L2:          {self.embedding_l2_reg}")
            logger.info(f"  Dense L1:              {self.dense_l1_reg}")
            logger.info(f"  Dense L2:              {self.dense_l2_reg}")

            logger.info("Other Settings:")
            logger.info(f"  Early Stop Patience:   {self.early_stop_patience}")
            logger.info(f"  Max Gradient Norm:     {self.max_gradient_norm}")
            logger.info(f"  Session ID:            {self.session_id}")
            logger.info(f"  Features Config Path:  {self.features_config_path}")
            logger.info(f"  Latest Checkpoint:     {self.checkpoint_path}")
            logger.info(f"  Note:                  {self.note}")

        if "Data Summary" in selected_sections and (self.train_data_summary or self.valid_data_summary):
            logger.info("")
            logger.info(colorize("Data Summary", color="cyan", bold=True))
            logger.info(colorize("-" * 80, color="cyan"))
            for label, data_summary in (
                ("Train", self.train_data_summary),
                ("Valid", self.valid_data_summary),
            ):
                if not data_summary:
                    continue
                if label == "Valid" and self.train_data_summary:
                    logger.info("")
                sample_key = "train_samples" if label == "Train" else "valid_samples"
                samples = data_summary.get(sample_key)
                if samples is not None:
                    logger.info(format_kv(f"{label} Samples", f"{samples:,}"))

                label_distributions = data_summary.get("label_distributions")
                if isinstance(label_distributions, dict):
                    for target_name, details in label_distributions.items():
                        lines = details.get("lines", [])
                        logger.info(f"{target_name}:")
                        for line_label, value in lines:
                            logger.info(f"  {format_kv(line_label, value)}")

                dataloader_info = data_summary.get("dataloader")
                if isinstance(dataloader_info, dict):
                    logger.info(f"{label} DataLoader:")
                    for key in (
                        "batch_size",
                        "num_workers",
                        "pin_memory",
                        "persistent_workers",
                        "sampler",
                    ):
                        if key in dataloader_info:
                            field_label = key.replace("_", " ").title()
                            logger.info(format_kv(field_label, dataloader_info[key], indent=2))
