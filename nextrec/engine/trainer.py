"""
Trainer module for NextRec models, providing training loop with support for
various optimizers, schedulers, loss functions, and metrics.

Date: create on 16/04/2025
Checkpoint: edit on 16/04/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import swanlab  # type: ignore
except Exception:  # pragma: no cover
    swanlab = None

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from nextrec.basic.asserts import assert_loss_weights
from nextrec.basic.callback import CheckpointSaver, EarlyStopper, LearningRateScheduler
from nextrec.basic.loggers import TrainingLogger, colorize, format_kv, setup_logger
from nextrec.basic.metrics import (
    compute_sequential_metric_batch,
    configure_metrics,
    evaluate_metrics,
    flatten_metric_names,
    get_best_metric_mode,
    is_sequential_ranking_metric,
    needs_group_ids,
)
from nextrec.data.batch_utils import batch_to_dict
from nextrec.data.data_processing import get_group_ids, split_train_valid
from nextrec.data.dataloader import RecDataLoader
from nextrec.loss.grad_norm import GradNormLossWeighting, get_grad_norm_shared_params
from nextrec.utils.config import safe_value
from nextrec.utils.console import display_metrics_table, progress
from nextrec.utils.data import is_path_data, split_path_files
from nextrec.utils.loss import get_loss_fn, get_loss_list
from nextrec.utils.model import compute_ranking_loss
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import (
    get_optimizer,
    get_scheduler,
    get_warmup,
)
from nextrec.utils.types import LossName, MetricsName, OptimizerName, SchedulerName


class BaseTrainer:
    def compile(
        self,
        optimizer: OptimizerName | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            SchedulerName
            | torch.optim.lr_scheduler._LRScheduler
            | torch.optim.lr_scheduler.LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | type[torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        warmup: bool | dict | None = None,
        loss: LossName | nn.Module | list[LossName | nn.Module] | None = None,
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | dict | None = None,
        ignore_label: int | float | None = -1,
    ):
        """
        Configure the model for training.

        Args:
            optimizer: Optimizer name or instance. e.g., 'adam', 'sgd', or torch.optim.Adam().
            optimizer_params: Optimizer parameters. e.g., {'lr': 1e-3, 'weight_decay': 1e-5}.
            scheduler: Learning rate scheduler name or instance. e.g., 'step', 'cosine', or torch.optim.lr_scheduler.StepLR().
            scheduler_params: Scheduler parameters. e.g., {'step_size': 10, 'gamma': 0.1}.
            warmup: Optional warmup config for scheduler.
                - False/None (default): disable warmup.
                - True: enable with defaults {'epochs': 1, 'start_factor': 0.1, 'end_factor': 1.0}.
                - dict: configure {'enabled': bool, 'epochs': int, 'start_factor': float, 'end_factor': float}.
            loss: Loss function name, instance, or list for multi-task. Must be provided explicitly.
                e.g., 'bce', 'mse', or torch.nn.BCELoss().
            loss_params: Loss function parameters, or list for multi-task. e.g., {'weight': tensor([0.25, 0.75])}.
            loss_weights: Weights for each task loss, int/float for single-task or list for multi-task. e.g., 1.0, or [1.0, 0.5].
                Use {"method": "grad_norm", ...} to enable GradNorm for multi-task loss balancing.
            ignore_label: Label value to ignore when computing loss. Use this to skip gradients for unknown labels.
        """

        self.ignore_label = ignore_label
        loss_list = get_loss_list(loss, self.training_mode, self.nums_task)
        self.loss_params = loss_params or {}
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}

        # get warmup config dict: {'enabled': bool, 'epochs': int, 'start_factor': float, 'end_factor': float} or None
        self.warmup = get_warmup(warmup)

        self.optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self.optimizer_fn = get_optimizer(
            optimizer=optimizer,
            params=self.parameters(),
            **self.optimizer_params,
        )

        if scheduler is None:
            self.scheduler_name = None
        elif isinstance(scheduler, str):
            self.scheduler_name = scheduler
        else:
            self.scheduler_name = getattr(scheduler, "__name__", scheduler.__class__.__name__)  # type: ignore
        self.scheduler_fn = get_scheduler(scheduler, self.optimizer_fn, **self.scheduler_params) if scheduler else None
        if self.warmup is not None:
            warmup_epochs = int(self.warmup["epochs"])
            warmup_start = float(self.warmup["start_factor"])
            warmup_end = float(self.warmup["end_factor"])
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_fn,
                start_factor=warmup_start,
                end_factor=warmup_end,
                total_iters=warmup_epochs,
            )
            if self.scheduler_fn is None:
                self.scheduler_fn = warmup_scheduler
                self.scheduler_name = "warmup"
            else:
                if hasattr(self.scheduler_fn, "mode"):
                    raise ValueError(
                        "[BaseModel-compile Error] warmup currently does not support ReduceLROnPlateau-like schedulers."
                    )
                self.scheduler_fn = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer_fn,
                    schedulers=[warmup_scheduler, self.scheduler_fn],
                    milestones=[warmup_epochs],
                )
                self.scheduler_name = f"warmup+{self.scheduler_name}"

        self.loss_config = loss_list if self.nums_task > 1 else loss_list[0]
        if isinstance(self.loss_params, dict):
            loss_params_list = [self.loss_params] * self.nums_task
        else:
            loss_params_list = [self.loss_params[i] if i < len(self.loss_params) else {} for i in range(self.nums_task)]
        self.loss_fn = [get_loss_fn(loss=loss_list[i], **loss_params_list[i]) for i in range(self.nums_task)]

        # grad_norm needs shared parameters across tasks to compute gradient norms
        # grad_norm_config example: {'alpha': 1.5, 'lr': 0.025}
        self.grad_norm = None
        self.grad_norm_shared_params = None
        self.grad_norm_config = None

        if isinstance(loss_weights, dict) and loss_weights.get("method") == "grad_norm":
            if self.nums_task == 1:
                raise ValueError("[BaseModel-compile Error] GradNorm requires multi-task setup.")
            grad_norm_params = dict(loss_weights)
            grad_norm_params.pop("method", None)
            self.grad_norm_config = grad_norm_params
            self.loss_weights = None
        elif loss_weights is None:
            self.loss_weights = None
        else:
            self.loss_weights = assert_loss_weights(
                loss_weights,
                self.nums_task,
                model_name="BaseModel-compile",
            )
        self.compiled = True

    def compute_task_loss(self, y_pred, y_true, task_index: int):
        """Compute loss for a single task."""

        mode = self.training_mode
        loss_fn = self.loss_fn[task_index]
        ignore_label = self.ignore_label
        task_type = self.task[task_index] if isinstance(self.task, list) else self.task

        if mode in {"pairwise", "listwise"}:
            if ignore_label is not None:
                valid_mask = y_true != ignore_label
                if valid_mask.dim() > 1:
                    valid_mask = valid_mask.all(dim=tuple(range(1, valid_mask.dim())))
                if not torch.any(valid_mask):
                    raise ValueError(
                        f"[BaseModel-compute_task_loss Error] Task {task_index} has no valid ranking labels after ignore_label filtering."
                    )
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]
            return compute_ranking_loss(training_mode=mode, loss_fn=loss_fn, y_pred=y_pred, y_true=y_true)

        # Generative tasks use vocab-classification loss in pointwise mode
        elif task_type == "generative":
            labels = y_true.view(-1).long()
            if ignore_label is not None:
                valid_mask = labels != int(ignore_label)
                if not torch.any(valid_mask):
                    raise ValueError(
                        f"[BaseModel-compute_task_loss Error] Task {task_index} has no valid generative labels after ignore_label filtering."
                    )
                y_pred = y_pred[valid_mask]
                labels = labels[valid_mask]
            return loss_fn(y_pred, labels)

        else:
            if ignore_label is not None:
                valid_mask = y_true != ignore_label
                if valid_mask.dim() > 1:
                    valid_mask = valid_mask.all(dim=tuple(range(1, valid_mask.dim())))
                if not torch.any(valid_mask):
                    raise ValueError(
                        f"[BaseModel-compute_task_loss Error] Task {task_index} has no valid labels after ignore_label filtering."
                    )
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]

            return loss_fn(y_pred.view(-1), y_true.view(-1))

    def compute_loss(self, y_pred, y_true):
        """Compute total loss, applying loss weights and GradNorm if configured."""

        adapter_loss = self.training_adapter.compute_loss(self, y_pred, y_true)
        if adapter_loss is not None:
            return adapter_loss
        if y_true is None:
            raise ValueError("[BaseModel-compute_loss Error] Ground truth labels (y_true) are required.")
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(f"[BaseModel-compute_loss Error] Expected y_pred to be a torch.Tensor, got {type(y_pred)}.")

        # single task compute loss
        if self.nums_task == 1:
            task_loss = self.compute_task_loss(y_pred, y_true, task_index=0)
            if isinstance(self.loss_weights, (list, tuple)):
                task_loss = task_loss * self.loss_weights[0]
            return task_loss

        task_slices = list(self.head.task_slices)

        # each task compute loss and combine
        task_losses = []
        for task_index, (start, end) in enumerate(task_slices[: self.nums_task]):
            y_pred_i = y_pred[..., start:end]
            y_true_i = y_true[..., start:end]

            task_losses.append(self.compute_task_loss(y_pred_i, y_true_i, task_index=task_index))

        if self.grad_norm is not None:
            if self.grad_norm_shared_params is None:
                self.grad_norm_shared_params = get_grad_norm_shared_params(self, self.grad_norm_shared_modules)
            return self.grad_norm.compute_weighted_loss(task_losses, self.grad_norm_shared_params)

        if isinstance(self.loss_weights, (list, tuple)):
            task_losses = [task_loss * self.loss_weights[i] for i, task_loss in enumerate(task_losses)]
        return torch.stack(task_losses).sum()

    def prepare_data_loader(
        self,
        data: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        streaming: bool = False,
        chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        processor: Any | None = None,
        key_columns: str | list[str] | None = None,
        expand: dict[str, list[Any]] | None = None,
        profiler: StageTimer | None = None,
    ) -> DataLoader:
        """
        Prepare a DataLoader from input data.

        Args:
            data: Input data, can be a DataFrame, Dataset, path, or existing DataLoader.
            batch_size: Batch size for the DataLoader.
            shuffle: Whether to shuffle the data.
            streaming: Whether to stream file-path input.
            chunk_size: Chunk size for streaming file input.
            num_workers: Number of worker processes for data loading.
            prefetch_factor: Number of batches to prefetch (if num_workers > 0).
            shard_rank: Shard rank for streaming file input.
            shard_count: Shard count for streaming file input.
            processor: Optional fitted DataProcessor used by RecDataLoader.
            key_columns: Column name(s) to carry through when building tensors from data.
            expand: Optional prediction-time row expansion config passed through to RecDataLoader.
            profiler: Optional StageTimer used for streaming data loading profile collection.

        Returns:
            DataLoader instance.
        """

        # if input is DataLoader, return directly
        if isinstance(data, DataLoader):
            return data

        # target_source and target_shift_steps are used by RecDataLoader
        # to determine how to shift labels for next-item prediction tasks
        target_source, target_shift_steps = self.training_adapter.get_target_shift_config(self)
        rec_loader = RecDataLoader(
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            target=self.target_columns,
            key_columns=key_columns if key_columns is not None else self.key_columns,
            target_source=target_source,
            target_shift_steps=target_shift_steps,
            task=self.task,
            model_family=self.model_family,
            training_mode=self.training_mode,
            sampling_mode=self.sampling_mode,
            feature_scopes=self.feature_scopes,
            processor=processor,
            expand=expand,
        )

        loader = rec_loader.create_dataloader(
            data=data,
            batch_size=batch_size,
            shuffle=shuffle,
            streaming=streaming,
            chunk_size=chunk_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shard_rank=shard_rank,
            shard_count=shard_count,
            profiler=profiler,
        )
        return loader

    def fit(
        self,
        train_data: (
            dict | pd.DataFrame | pl.DataFrame | str | os.PathLike | list[str] | list[os.PathLike] | DataLoader | None
        ) = None,
        valid_data: (
            dict | pd.DataFrame | pl.DataFrame | str | os.PathLike | list[str] | list[os.PathLike] | DataLoader | None
        ) = None,
        metrics: list[MetricsName] | dict[str, list[MetricsName]] | None = None,
        epochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 32,
        group_id: str | None = None,
        valid_split: float | None = None,
        split_stratify_by: str | None = None,
        split_group_by: str | None = None,
        early_stop_patience: int = 20,
        early_stop_monitor_task: str | None = None,
        valid_group_by: str | list[str] | None = None,
        streaming: bool = False,
        chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        processor: Any | None = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        use_swanlab: bool = False,
        wandb_api: str | None = None,
        swanlab_api: str | None = None,
        wandb_kwargs: dict | None = None,
        swanlab_kwargs: dict | None = None,
        log_interval: int = 1,
        note: str | None = None,
        summary_sections: list[Literal["feature", "model", "train", "data"]] | None = None,
    ):
        """
        Train the model.

        Args:
            train_data: Training data (dict/df/path/DataLoader).
            valid_data: Optional validation data; if None and valid_split is set, a split is created.
            metrics: Metrics names or per-target dict. e.g. {'target1': ['auc', 'logloss'], 'target2': ['mse']} or ['auc', 'logloss'].
            epochs: Training epochs.
            shuffle: Whether to shuffle training data.
            batch_size: Batch size.
            group_id: Column name used to group GAUC and ranking metrics.
            valid_split: Ratio to split training data when valid_data is None. e.g., 0.1 for 10% validation.
                For path input, files are split by ratio; single-file path split requires explicit valid_data.
            split_stratify_by: Optional column/field used during auto split to preserve label or bucket distribution.
                Only applies when valid_data is None, valid_split is set, and train_data is an in-memory dict or
                DataFrame. Not supported for pre-built DataLoader or path input.
            split_group_by: Optional column/field used during auto split to keep all rows from the same group
                in either train or validation, avoiding group leakage. Only applies when valid_data is None,
                valid_split is set, and train_data is an in-memory dict or DataFrame. Not supported for
                pre-built DataLoader or path input.

            early_stop_patience: Epochs for early stopping. 0 to disable. e.g., 20.
            early_stop_monitor_task: Task name to monitor for early stopping in multi-task scenario. If None, uses first target. e.g., 'click'.
            valid_group_by: Optional validation group-by column(s) for grouped metric reporting during training.
            streaming: Whether to stream path input instead of loading files into memory.
            chunk_size: Chunk size for streaming file input.
            num_workers: DataLoader worker count.
            prefetch_factor: Number of batches to prefetch when num_workers > 0.
            processor: Optional fitted DataProcessor used for path/DataFrame input.

            use_tensorboard: Enable tensorboard logging.
            use_wandb: Enable Weights & Biases logging.
            use_swanlab: Enable SwanLab logging.
            wandb_api: W&B API key for non-tty login.
            swanlab_api: SwanLab API key for non-tty login.
            wandb_kwargs: Optional kwargs for wandb.init(...).
            swanlab_kwargs: Optional kwargs for swanlab.init(...).
            log_interval: Log validation metrics every N epochs (still computes metrics each epoch).
            note: Optional note for the training run.
            summary_sections: Optional summary sections to print. Choose from
                ["feature", "model", "train", "data"]. Defaults to all.
        """
        if self.grad_norm_config is not None:
            self.grad_norm = GradNormLossWeighting(
                nums_task=self.nums_task, device=self.device, **self.grad_norm_config
            )
            self.grad_norm_shared_params = None
        self.to(self.device)

        if self.target_columns and len(self.target_columns) != self.nums_task:
            raise ValueError(
                f"{self.model_name} target count ({len(self.target_columns)}) must match nums_task ({self.nums_task}). "
                "For multi-target training, pass task as a list, e.g. ['binary', 'binary']."
            )

        if not self.compiled:
            raise ValueError(
                "[BaseModel-fit Error] Model must be compiled before fit(). Call compile(loss=...) explicitly."
            )

        if not self.logger_initialized:
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True
        # group id for gauc metrics
        self.metric_group_id = group_id
        metrics_task = self.task
        self.metrics, self.task_specific_metrics, self.best_metrics_mode = configure_metrics(
            task=metrics_task,
            model_family=self.model_family,
            metrics=metrics,
            target_names=self.target_columns,
        )

        self.early_stop_patience = early_stop_patience
        self.early_stop_monitor_task = early_stop_monitor_task
        self.valid_group_by = valid_group_by
        self.note = note

        training_config = {
            "model_name": self.model_name,
            "note": self.note,
            "task": self.task,
            "model_family": self.model_family,
            "target_columns": self.target_columns,
            "batch_size": batch_size,
            "epochs": epochs,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "streaming": streaming,
            "chunk_size": chunk_size,
            "prefetch_factor": prefetch_factor,
            "processor": processor.__class__.__name__ if processor is not None else None,
            "group_id": group_id,
            "valid_split": valid_split,
            "split_stratify_by": split_stratify_by,
            "split_group_by": split_group_by,
            "random_state": 42,
            "optimizer": self.optimizer_name,
            "optimizer_params": self.optimizer_params,
            "scheduler": self.scheduler_name,
            "scheduler_params": self.scheduler_params,
            "loss": self.loss_config,
            "loss_weights": self.loss_weights,
            "early_stop_patience": self.early_stop_patience,
            "max_gradient_norm": self.max_gradient_norm,
            "embedding_l1_reg": self.embedding_l1_reg,
            "embedding_l2_reg": self.embedding_l2_reg,
            "dense_l1_reg": self.dense_l1_reg,
            "device": str(self.device),
            "dense_feature_count": len(self.dense_features),
            "sparse_feature_count": len(self.sparse_features),
            "sequence_feature_count": len(self.sequence_features),
        }
        training_config = safe_value(training_config)

        is_tty = sys.stdin.isatty() and sys.stdout.isatty()
        if not is_tty:
            if use_wandb and wandb_api:
                if wandb is None:
                    logging.warning("[BaseModel-fit] wandb not installed, skip wandb login.")
                else:
                    wandb.login(key=wandb_api)
            if use_swanlab and swanlab_api:
                if swanlab is None:
                    logging.warning("[BaseModel-fit] swanlab not installed, skip swanlab login.")
                else:
                    swanlab.login(api_key=swanlab_api)

        if use_wandb and self.note:
            wandb_kwargs = dict(wandb_kwargs or {})
            wandb_kwargs.setdefault("notes", self.note)

        if use_swanlab and self.note:
            swanlab_kwargs = dict(swanlab_kwargs or {})
            swanlab_kwargs.setdefault("description", self.note)

        self.training_logger = TrainingLogger(
            session=self.session,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            use_swanlab=use_swanlab,
            config=training_config,
            wandb_kwargs=wandb_kwargs,
            swanlab_kwargs=swanlab_kwargs,
        )

        # init metrics for early stopping monitoring
        if self.nums_task == 1:
            # for single task, always monitor the first metric
            monitor_metric = f"val_{self.metrics[0]}"
        else:
            monitor_task = self.early_stop_monitor_task
            # if not specified, default to monitoring the first target's primary metric
            if monitor_task is None:
                monitor_task = self.target_columns[0]
            # Use task-specific first metric if available, otherwise fall back to global metrics[0]
            task_metrics = (self.task_specific_metrics or {}).get(monitor_task)
            primary_metric = task_metrics[0] if task_metrics else self.metrics[0]
            monitor_metric = f"val_{primary_metric}_{monitor_task}"

            # best_metrics_mode based on the actual monitor task type and its primary metric
            monitor_task_index = self.target_columns.index(monitor_task)
            monitor_task_type = self.task[monitor_task_index] if isinstance(self.task, list) else self.task
            self.best_metrics_mode = get_best_metric_mode(primary_metric, monitor_task_type)

        existing_callbacks = self.callbacks.callbacks

        has_validation = valid_data is not None or valid_split is not None
        early_stop_monitor = monitor_metric
        early_stop_mode = self.best_metrics_mode
        checkpoint_monitor = monitor_metric
        checkpoint_mode = self.best_metrics_mode

        # monitor train loss if no provide validation
        if not has_validation:
            early_stop_monitor = "loss"
            early_stop_mode = "min"
            checkpoint_monitor = "loss"
            checkpoint_mode = "min"

        if self.early_stop_patience > 0 and not any(isinstance(cb, EarlyStopper) for cb in existing_callbacks):
            self.callbacks.append(
                EarlyStopper(
                    monitor=early_stop_monitor,
                    patience=self.early_stop_patience,
                    mode=early_stop_mode,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        if not any(isinstance(cb, CheckpointSaver) for cb in existing_callbacks):
            self.callbacks.append(
                CheckpointSaver(
                    best_path=self.best_path,
                    checkpoint_path=self.checkpoint_path,
                    monitor=checkpoint_monitor,
                    mode=checkpoint_mode,
                    save_best_only=has_validation,
                    verbose=1,
                )
            )

        if self.scheduler_fn is not None and not any(
            isinstance(cb, LearningRateScheduler) for cb in existing_callbacks
        ):
            self.callbacks.append(
                LearningRateScheduler(
                    scheduler=self.scheduler_fn,
                    verbose=1,
                )
            )

        self.callbacks.set_model(self)
        self.callbacks.set_params({"epochs": epochs, "batch_size": batch_size, "metrics": self.metrics})
        self.best_metric = float("-inf") if self.best_metrics_mode == "max" else float("inf")

        self.needs_group_ids = (
            needs_group_ids(self.metrics, self.task_specific_metrics) if self.model_family != "sequential" else False
        )
        if self.needs_group_ids and not group_id:
            raise ValueError(
                "[BaseModel-fit Error] group_id must be specified when grouped metrics such as GAUC or ranking@K are enabled."
            )
        self.epoch_index = 0
        self.stop_training = False
        self.best_checkpoint_path = self.best_path

        # auto-split validation set from training data
        if valid_split is not None and valid_data is None:
            if isinstance(train_data, DataLoader):
                raise ValueError(
                    "[BaseModel-validation Error] valid_split is not supported when train_data is already a DataLoader."
                )
            if is_path_data(train_data):
                if split_stratify_by is not None or split_group_by is not None:
                    raise ValueError(
                        "[BaseModel-validation Error] split_stratify_by and split_group_by are not supported "
                        "for path valid_split."
                    )
                train_data, valid_data = split_path_files(train_data, valid_split)
                logging.info(
                    "[BaseModel-validation Info] Split files using valid_split="
                    f"{valid_split:.3f}: train_files={len(train_data)}, valid_files={len(valid_data)}"
                )
            else:
                train_data, valid_data = split_train_valid(
                    train_data=train_data,
                    valid_split=valid_split,
                    split_stratify_by=split_stratify_by,
                    split_group_by=split_group_by,
                    random_state=42,
                )

        # get group id and key columns for dataloaders
        train_key_columns = list(
            dict.fromkeys(
                [
                    *(self.key_columns or []),
                    *([group_id] if self.needs_group_ids and group_id else []),
                ]
            )
        )
        train_loader = self.prepare_data_loader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            streaming=streaming,
            chunk_size=chunk_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            key_columns=train_key_columns,
        )

        # prepare validation dataloader
        if valid_data is None:
            valid_loader = None
        else:
            valid_group_by_columns = [valid_group_by] if isinstance(valid_group_by, str) else list(valid_group_by or [])
            valid_key_columns = list(
                dict.fromkeys(
                    [
                        *(self.key_columns or []),
                        *valid_group_by_columns,
                        *([group_id] if self.needs_group_ids and group_id else []),
                    ]
                )
            )
            valid_loader = self.prepare_data_loader(
                valid_data,
                batch_size=batch_size,
                shuffle=False,
                streaming=streaming,
                chunk_size=chunk_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                processor=processor,
                key_columns=valid_key_columns,
            )
        try:
            self.steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:
            self.steps_per_epoch = None
            is_streaming = True

        if is_streaming:
            logging.info(
                colorize(
                    "[Training Info] Streaming mode detected; training metrics collection is disabled to avoid memory growth.",
                    color="yellow",
                )
            )

        self.train_data_summary = None if is_streaming else self.build_train_data_summary(train_data, train_loader)
        self.valid_data_summary = (
            None if valid_loader is None else self.build_valid_data_summary(valid_data, valid_loader)
        )
        self.summary(summary_sections)
        logging.info("")
        self.training_logger.log_tensorboard_hint()

        logging.info("")
        logging.info(colorize("[Training]", color="bright_blue", bold=True))
        logging.info(colorize("-" * 80, color="bright_blue"))
        logging.info(format_kv("Start training", f"{epochs} epochs"))
        logging.info(format_kv("Model device", self.device))
        logging.info("")

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.epoch_index = epoch
            self.callbacks.on_epoch_begin(epoch)

            if is_streaming:
                logging.info("")
                logging.info(colorize(f"Epoch {epoch + 1}/{epochs}", bold=True))

            train_loss, train_metrics = self.train_epoch(train_loader, is_streaming=is_streaming)

            logging.info("")
            train_log_payload = {"loss": float(train_loss)}
            if train_metrics:
                train_log_payload.update(train_metrics)

            display_metrics_table(
                epoch=epoch + 1,
                epochs=epochs,
                split="Train",
                loss=train_loss,
                metrics=train_metrics,
                target_names=self.target_columns,
                base_metrics=(self.metrics if isinstance(self.metrics, list) else None),
                colorize=lambda s: colorize(s),
            )
            self.training_logger.log_metrics(train_log_payload, step=epoch + 1, split="train")

            if valid_loader is not None:
                run_validation = (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs
                if run_validation:
                    self.callbacks.on_validation_begin()

                    # sequential model overrides evaluate: use next-item ranking to calcualte metrics
                    if self.model_family == "sequential":
                        if self.valid_group_by:
                            logging.info(
                                colorize(
                                    "[Validation Info] valid_group_by is ignored for sequential evaluation.",
                                    color="yellow",
                                )
                            )
                        val_metrics = self.evaluate(valid_loader, num_workers=num_workers)
                    else:
                        val_metrics = self.evaluate(
                            valid_loader,
                            group_id=group_id,
                            group_by=self.valid_group_by,
                            num_workers=num_workers,
                        )
                    valid_metrics = val_metrics["overall"]
                    if not self.valid_group_by:
                        display_metrics_table(
                            epoch=epoch + 1,
                            epochs=epochs,
                            split="Valid",
                            loss=None,
                            metrics=valid_metrics,
                            target_names=self.target_columns,
                            base_metrics=(self.metrics if isinstance(self.metrics, list) else None),
                            colorize=lambda s: colorize("  " + s, color="cyan"),
                        )
                    self.callbacks.on_validation_end()

                    if valid_metrics:
                        self.training_logger.log_metrics(valid_metrics, step=epoch + 1, split="valid")
                else:
                    valid_metrics = None
                if not valid_metrics:
                    if run_validation:
                        logging.info(
                            colorize("Warning: No validation metrics computed. Skipping validation for this epoch.")
                        )
                    epoch_logs = {**train_log_payload}
                else:
                    epoch_logs = {**train_log_payload}
                    for k, v in valid_metrics.items():
                        epoch_logs[f"val_{k}"] = v
            else:
                epoch_logs = {**train_log_payload}

            self.callbacks.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break
        self.callbacks.on_train_end()
        logging.info("")
        logging.info(colorize("Training finished.", color="bright_blue", bold=True))
        logging.info("")

        if has_validation:
            logging.info(format_kv("Load best model from", self.best_checkpoint_path))
            if os.path.exists(self.best_checkpoint_path):
                self.load_model(self.best_checkpoint_path, map_location=self.device, verbose=False)
            else:
                logging.info(
                    colorize(
                        f"Warning: Best checkpoint not found at {self.best_checkpoint_path}, skip loading best model.",
                        color="yellow",
                    )
                )
        elif not has_validation:
            logging.info(
                colorize("[Training Info] No validation data provided. Using final epoch model.", color="yellow")
            )

        self.training_logger.close()
        return self

    def train_epoch(self, train_loader: DataLoader, is_streaming: bool = False):
        """
        Train for one epoch and compute training loss and metrics.
        """

        self.train()

        accumulated_loss = 0.0
        num_batches = 0
        y_true_list = []
        y_pred_list = []
        group_ids_list = [] if self.needs_group_ids else None
        sequential_metric_names = (
            [metric for metric in flatten_metric_names(self.metrics) if is_sequential_ranking_metric(metric)]
            if self.model_family == "sequential"
            else []
        )
        sequential_metric_sums = {metric: 0.0 for metric in sequential_metric_names}
        sequential_valid_count = 0
        if self.steps_per_epoch is not None:
            batch_iter = enumerate(
                progress(
                    train_loader,
                    description=f"Epoch {self.epoch_index + 1}",
                    total=self.steps_per_epoch,
                    disable=False,
                )
            )
        else:
            desc = "Batches" if is_streaming else f"Epoch {self.epoch_index + 1}"
            batch_iter = enumerate(progress(train_loader, description=desc, disable=False))

        for _, batch_data in batch_iter:
            batch_dict = batch_to_dict(batch_data)

            # get explict labels if model needs
            require_labels = self.training_adapter.needs_labels(self)
            X_input, y_true = self.get_input(batch_dict, require_labels=require_labels)
            y_pred = self.training_adapter.forward(self, X_input)

            task_loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()
            total_loss = task_loss + reg_loss
            self.optimizer_fn.zero_grad()
            total_loss.backward()

            if self.max_gradient_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
            self.optimizer_fn.step()
            if self.grad_norm is not None:
                self.grad_norm.step()
            accumulated_loss += task_loss.item()

            if not is_streaming and y_true is not None:
                if self.model_family == "sequential":
                    batch_metric_sums, batch_valid_count, _ = compute_sequential_metric_batch(
                        logits=y_pred.detach(),
                        labels=y_true.detach(),
                        metrics=sequential_metric_names,
                        padding_idx=self.padding_idx,
                        ignore_label=self.ignore_label,
                    )
                    sequential_valid_count += batch_valid_count
                    for metric_name, metric_value in batch_metric_sums.items():
                        sequential_metric_sums[metric_name] += metric_value
                elif y_pred.dim() <= 2 and y_true.dim() <= 2:
                    y_true_list.append(y_true.detach().cpu().numpy())
                    y_pred_list.append(y_pred.detach().cpu().numpy())
                    if self.needs_group_ids:
                        batch_group_id = get_group_ids(data=batch_dict, group_id=self.metric_group_id)
                        if batch_group_id is not None:
                            group_ids_list.append(batch_group_id)
            num_batches += 1

        avg_loss = accumulated_loss / max(num_batches, 1)

        if self.model_family == "sequential":
            if sequential_valid_count > 0:
                metrics_dict = {
                    metric: sequential_metric_sums[metric] / float(sequential_valid_count)
                    for metric in sequential_metric_names
                }
                return avg_loss, metrics_dict
            return avg_loss, None

        y_true_all = np.concatenate(y_true_list, axis=0) if y_true_list else None
        y_pred_all = np.concatenate(y_pred_list, axis=0) if y_pred_list else None
        combined_group_ids = np.concatenate(group_ids_list, axis=0) if self.needs_group_ids and group_ids_list else None

        if y_true_all is not None and y_pred_all is not None and len(y_true_all) > 0 and len(y_pred_all) > 0:
            metrics_dict = evaluate_metrics(
                y_true=y_true_all,
                y_pred=y_pred_all,
                metrics=self.metrics,
                task=self.task,
                model_family=self.model_family,
                target_names=self.target_columns,
                task_specific_metrics=self.task_specific_metrics,
                group_ids=combined_group_ids,
                ignore_label=self.ignore_label,
            )
            return avg_loss, metrics_dict
        return avg_loss, None
