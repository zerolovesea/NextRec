"""
Base Model & Base Match Model Class

Date: create on 27/10/2025
Checkpoint: edit on 18/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import tqdm
import pickle
import logging
import getpass
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from pathlib import Path
from typing import Union, Literal, Any
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from nextrec.basic.callback import (
    EarlyStopper,
    CallbackList,
    Callback,
    CheckpointSaver,
    LearningRateScheduler,
)
from nextrec.basic.features import (
    DenseFeature,
    SparseFeature,
    SequenceFeature,
    FeatureSet,
)
from nextrec.data.dataloader import TensorDictDataset, RecDataLoader

from nextrec.basic.loggers import setup_logger, colorize, TrainingLogger
from nextrec.basic.session import resolve_save_path, create_session
from nextrec.basic.metrics import configure_metrics, evaluate_metrics, check_user_id

from nextrec.data.dataloader import build_tensors_from_data
from nextrec.data.batch_utils import collate_fn, batch_to_dict
from nextrec.data.data_processing import get_column_data, get_user_ids

from nextrec.loss import (
    BPRLoss,
    HingeLoss,
    InfoNCELoss,
    SampledSoftmaxLoss,
    TripletLoss,
    get_loss_fn,
    get_loss_kwargs,
)
from nextrec.utils.tensor import to_tensor
from nextrec.utils.device import configure_device
from nextrec.utils.optimizer import get_optimizer, get_scheduler
from nextrec.utils.distributed import (
    gather_numpy,
    init_process_group,
    add_distributed_sampler,
)
from nextrec import __version__


class BaseModel(FeatureSet, nn.Module):
    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def default_task(self) -> str | list[str]:
        raise NotImplementedError

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | str | None = None,
        id_columns: list[str] | str | None = None,
        task: str | list[str] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        device: str = "cpu",
        early_stop_patience: int = 20,
        session_id: str | None = None,
        callbacks: list[Callback] | None = None,
        distributed: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        ddp_find_unused_parameters: bool = False,
    ):
        """
        Initialize a base model.

        Args:
            dense_features: DenseFeature definitions.
            sparse_features: SparseFeature definitions.
            sequence_features: SequenceFeature definitions.
            target: Target column name. e.g., 'label' or ['label1', 'label2'].
            id_columns: Identifier column name, only need to specify if GAUC is required. e.g., 'user_id'.
            task: Task types, e.g., 'binary', 'regression', or ['binary', 'regression']. If None, falls back to self.default_task.

            embedding_l1_reg: L1 regularization strength for embedding params. e.g., 1e-6.
            dense_l1_reg: L1 regularization strength for dense params. e.g., 1e-5.
            embedding_l2_reg: L2 regularization strength for embedding params. e.g., 1e-5.
            dense_l2_reg: L2 regularization strength for dense params. e.g., 1e-4.

            device: Torch device string or torch.device. e.g., 'cpu', 'cuda:0'.
            early_stop_patience: Epochs for early stopping. 0 to disable. e.g., 20.
            session_id: Session id for logging. If None, a default id with timestamps will be created. e.g., 'session_tutorial'.
            callbacks: List of callback instances. If None, default callbacks will be created. e.g., [EarlyStopper(), CheckpointSaver()].

            distributed: Enable DistributedDataParallel flow, set True to enable distributed training.
            rank: Global rank (defaults to env RANK).
            world_size: Number of processes (defaults to env WORLD_SIZE).
            local_rank: Local rank for selecting CUDA device (defaults to env LOCAL_RANK).
            ddp_find_unused_parameters: Default False, set it True only when exist unused parameters in ddp model, in most cases should be False.
        """
        super(BaseModel, self).__init__()

        # distributed training settings
        env_rank = int(os.environ.get("RANK", "0"))
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = distributed or (env_world_size > 1)
        self.rank = env_rank if rank is None else rank
        self.world_size = env_world_size if world_size is None else world_size
        self.local_rank = env_local_rank if local_rank is None else local_rank
        self.is_main_process = self.rank == 0
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.ddp_model: DDP | None = None
        self.device = configure_device(self.distributed, self.local_rank, device)

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = self.session.root  # pwd/session_id, path for this session
        self.checkpoint_path = os.path.join(
            self.session_path, self.model_name + "_checkpoint.pt"
        )  # example: pwd/session_id/DeepFM_checkpoint.pt
        self.best_path = os.path.join(self.session_path, self.model_name + "_best.pt")
        self.features_config_path = os.path.join(
            self.session_path, "features_config.pkl"
        )
        self.set_all_features(
            dense_features, sparse_features, sequence_features, target, id_columns
        )

        self.task = self.default_task if task is None else task
        self.nums_task = len(self.task) if isinstance(self.task, list) else 1

        self.embedding_l1_reg = embedding_l1_reg
        self.dense_l1_reg = dense_l1_reg
        self.embedding_l2_reg = embedding_l2_reg
        self.dense_l2_reg = dense_l2_reg
        self.regularization_weights = []
        self.embedding_params = []
        self.loss_weight = None

        self.early_stop_patience = early_stop_patience
        self.max_gradient_norm = 1.0
        self.logger_initialized = False
        self.training_logger = None
        self.callbacks = CallbackList(callbacks) if callbacks else CallbackList()

    def register_regularization_weights(
        self,
        embedding_attr: str = "embedding",
        exclude_modules: list[str] | None = None,
        include_modules: list[str] | None = None,
    ) -> None:
        exclude_modules = exclude_modules or []
        include_modules = include_modules or []
        embedding_layer = getattr(self, embedding_attr, None)
        embed_dict = getattr(embedding_layer, "embed_dict", None)
        embedding_params: list[torch.Tensor] = []
        if embed_dict is not None:
            embedding_params.extend(
                embed.weight for embed in embed_dict.values() if hasattr(embed, "weight")
            )
        else:
            weight = getattr(embedding_layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                embedding_params.append(weight)

        existing_embedding_ids = {id(param) for param in self.embedding_params}
        for param in embedding_params:
            if id(param) not in existing_embedding_ids:
                self.embedding_params.append(param)
                existing_embedding_ids.add(id(param))

        skip_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
        )
        existing_reg_ids = {id(param) for param in self.regularization_weights}
        for name, module in self.named_modules():
            if (
                module is self
                or embedding_attr in name
                or isinstance(module, skip_types)
                or (include_modules and not any(inc in name for inc in include_modules))
                or any(exc in name for exc in exclude_modules)
            ):
                continue
            if isinstance(module, nn.Linear):
                if id(module.weight) not in existing_reg_ids:
                    self.regularization_weights.append(module.weight)
                    existing_reg_ids.add(id(module.weight))

    def add_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.embedding_params:
            if self.embedding_l1_reg > 0:
                reg_loss += self.embedding_l1_reg * sum(
                    param.abs().sum() for param in self.embedding_params
                )
            if self.embedding_l2_reg > 0:
                reg_loss += self.embedding_l2_reg * sum(
                    (param**2).sum() for param in self.embedding_params
                )
        if self.regularization_weights:
            if self.dense_l1_reg > 0:
                reg_loss += self.dense_l1_reg * sum(
                    param.abs().sum() for param in self.regularization_weights
                )
            if self.dense_l2_reg > 0:
                reg_loss += self.dense_l2_reg * sum(
                    (param**2).sum() for param in self.regularization_weights
                )
        return reg_loss

    def get_input(self, input_data: dict, require_labels: bool = True):
        feature_source = input_data.get("features", {})
        label_source = input_data.get("labels")
        X_input = {}
        for feature in self.all_features:
            if feature.name not in feature_source:
                raise KeyError(
                    f"[BaseModel-input Error] Feature '{feature.name}' not found in input data."
                )
            feature_data = get_column_data(feature_source, feature.name)
            X_input[feature.name] = to_tensor(
                feature_data,
                dtype=(
                    torch.float32 if isinstance(feature, DenseFeature) else torch.long
                ),
                device=self.device,
            )
        y = None
        if len(self.target_columns) > 0 and (
            require_labels
            or (
                label_source
                and any(name in label_source for name in self.target_columns)
            )
        ):  # need labels: training or eval with labels
            target_tensors = []
            for target_name in self.target_columns:
                if label_source is None or target_name not in label_source:
                    if require_labels:
                        raise KeyError(
                            f"[BaseModel-input Error] Target column '{target_name}' not found in input data."
                        )
                    continue
                target_data = get_column_data(label_source, target_name)
                if target_data is None:
                    if require_labels:
                        raise ValueError(
                            f"[BaseModel-input Error] Target column '{target_name}' contains no data."
                        )
                    continue
                target_tensor = to_tensor(
                    target_data, dtype=torch.float32, device=self.device
                )
                target_tensor = target_tensor.view(
                    target_tensor.size(0), -1
                )  # always reshape to (batch_size, num_targets)
                target_tensors.append(target_tensor)
            if target_tensors:
                y = torch.cat(target_tensors, dim=1)
                if y.shape[1] == 1:  # no need to do that again
                    y = y.view(-1)
            elif require_labels:
                raise ValueError(
                    "[BaseModel-input Error] Labels are required but none were found in the input batch."
                )
        return X_input, y

    def handle_validation_split(
        self,
        train_data: dict | pd.DataFrame,
        validation_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ):
        """
        This function will split training data into training and validation sets when:
        1. valid_data is None;
        2. validation_split is provided.
        """
        if not (0 < validation_split < 1):
            raise ValueError(
                f"[BaseModel-validation Error] validation_split must be between 0 and 1, got {validation_split}"
            )
        if not isinstance(train_data, (pd.DataFrame, dict)):
            raise TypeError(
                f"[BaseModel-validation Error] train_data must be a pandas DataFrame or a dict, got {type(train_data)}"
            )
        if isinstance(train_data, pd.DataFrame):
            total_length = len(train_data)
        else:
            sample_key = next(
                iter(train_data)
            )  # pick the first key to check length, for example: 'user_id': [1,2,3,4,5]
            total_length = len(train_data[sample_key])  # len(train_data['user_id'])
            for k, v in train_data.items():
                if len(v) != total_length:
                    raise ValueError(
                        f"[BaseModel-validation Error] Length of field '{k}' ({len(v)}) != length of field '{sample_key}' ({total_length})"
                    )
        rng = np.random.default_rng(42)
        indices = rng.permutation(total_length)
        split_idx = int(total_length * (1 - validation_split))
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]
        if isinstance(train_data, pd.DataFrame):
            train_split = train_data.iloc[train_indices].reset_index(drop=True)
            valid_split = train_data.iloc[valid_indices].reset_index(drop=True)
        else:
            train_split = {}
            valid_split = {}
            for key, value in train_data.items():
                arr = np.asarray(value)
                train_split[key] = arr[train_indices]
                valid_split[key] = arr[valid_indices]
        train_loader = self.prepare_data_loader(
            train_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        logging.info(
            f"Split data: {len(train_indices)} training samples, {len(valid_indices)} validation samples"
        )
        return train_loader, valid_split

    def compile(
        self,
        optimizer: str | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            str
            | torch.optim.lr_scheduler._LRScheduler
            | torch.optim.lr_scheduler.LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | type[torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Configure the model for training.
        Args:
            optimizer: Optimizer name or instance. e.g., 'adam', 'sgd', or torch.optim.Adam().
            optimizer_params: Optimizer parameters. e.g., {'lr': 1e-3, 'weight_decay': 1e-5}.
            scheduler: Learning rate scheduler name or instance. e.g., 'step_lr', 'cosine_annealing', or torch.optim.lr_scheduler.StepLR().
            scheduler_params: Scheduler parameters. e.g., {'step_size': 10, 'gamma': 0.1}.
            loss: Loss function name, instance, or list for multi-task. e.g., 'bce', 'mse', or torch.nn.BCELoss(), you can also use custom loss functions.
            loss_params: Loss function parameters, or list for multi-task. e.g., {'weight': tensor([0.25, 0.75])}.
            loss_weights: Weights for each task loss, int/float for single-task or list for multi-task. e.g., 1.0, or [1.0, 0.5].
            callbacks: Additional callbacks to add to the existing callback list. e.g., [EarlyStopper(), CheckpointSaver()].
        """
        if loss_params is None:
            self.loss_params = {}
        else:
            self.loss_params = loss_params
        optimizer_params = optimizer_params or {}
        self.optimizer_name = (
            optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        )
        self.optimizer_params = optimizer_params
        self.optimizer_fn = get_optimizer(
            optimizer=optimizer,
            params=self.parameters(),
            **optimizer_params,
        )

        scheduler_params = scheduler_params or {}
        if isinstance(scheduler, str):
            self.scheduler_name = scheduler
        elif scheduler is None:
            self.scheduler_name = None
        else:  # for custom scheduler instance, need to provide class name for logging
            self.scheduler_name = getattr(scheduler, "__name__", scheduler.__class__.__name__)  # type: ignore
        self.scheduler_params = scheduler_params
        self.scheduler_fn = (
            get_scheduler(scheduler, self.optimizer_fn, **scheduler_params)
            if scheduler
            else None
        )

        self.loss_config = loss
        self.loss_params = loss_params or {}
        self.loss_fn = []
        if isinstance(loss, list):  # for example: ['bce', 'mse'] -> ['bce', 'mse']
            if len(loss) != self.nums_task:
                raise ValueError(
                    f"[BaseModel-compile Error] Number of loss functions ({len(loss)}) must match number of tasks ({self.nums_task})."
                )
            loss_list = [loss[i] for i in range(self.nums_task)]
        else:  # for example: 'bce' -> ['bce', 'bce']
            loss_list = [loss] * self.nums_task

        if isinstance(self.loss_params, dict):
            params_list = [self.loss_params] * self.nums_task
        else:  # list[dict]
            params_list = [
                self.loss_params[i] if i < len(self.loss_params) else {}
                for i in range(self.nums_task)
            ]
        self.loss_fn = [
            get_loss_fn(loss=loss_list[i], **params_list[i])
            for i in range(self.nums_task)
        ]

        if loss_weights is None:
            self.loss_weights = None
        elif self.nums_task == 1:
            if isinstance(loss_weights, (list, tuple)):
                if len(loss_weights) != 1:
                    raise ValueError(
                        "[BaseModel-compile Error] loss_weights list must have exactly one element for single-task setup."
                    )
                weight_value = loss_weights[0]
            else:
                weight_value = loss_weights
            self.loss_weights = [float(weight_value)]
        else:
            if isinstance(loss_weights, (int, float)):
                weights = [float(loss_weights)] * self.nums_task
            elif isinstance(loss_weights, (list, tuple)):
                weights = [float(w) for w in loss_weights]
                if len(weights) != self.nums_task:
                    raise ValueError(
                        f"[BaseModel-compile Error] Number of loss_weights ({len(weights)}) must match number of tasks ({self.nums_task})."
                    )
            else:
                raise TypeError(
                    f"[BaseModel-compile Error] loss_weights must be int, float, list or tuple, got {type(loss_weights)}"
                )
            self.loss_weights = weights

        # Add callbacks from compile if provided
        if callbacks:
            for callback in callbacks:
                self.callbacks.append(callback)

    def compute_loss(self, y_pred, y_true):
        if y_true is None:
            raise ValueError(
                "[BaseModel-compute_loss Error] Ground truth labels (y_true) are required."
            )
        if self.nums_task == 1:
            if y_pred.dim() == 1:
                y_pred = y_pred.view(-1, 1)
            if y_true.dim() == 1:
                y_true = y_true.view(-1, 1)
            if y_pred.shape != y_true.shape:
                raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
            task_dim = self.task_dims[0] if hasattr(self, "task_dims") else y_pred.shape[1]  # type: ignore
            if task_dim == 1:
                loss = self.loss_fn[0](y_pred.view(-1), y_true.view(-1))
            else:
                loss = self.loss_fn[0](y_pred, y_true)
            if self.loss_weights is not None:
                loss *= self.loss_weights[0]
            return loss
        # multi-task
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
        if hasattr(
            self, "prediction_layer"
        ):  # we need to use registered task_slices for multi-task and multi-class
            slices = self.prediction_layer.task_slices  # type: ignore
        else:
            slices = [(i, i + 1) for i in range(self.nums_task)]
        task_losses = []
        for i, (start, end) in enumerate(slices):  # type: ignore
            y_pred_i = y_pred[:, start:end]
            y_true_i = y_true[:, start:end]
            task_loss = self.loss_fn[i](y_pred_i, y_true_i)
            if isinstance(self.loss_weights, (list, tuple)):
                task_loss *= self.loss_weights[i]
            task_losses.append(task_loss)
        return torch.stack(task_losses).sum()

    def prepare_data_loader(
        self,
        data: dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sampler=None,
        return_dataset: bool = False,
    ) -> DataLoader | tuple[DataLoader, TensorDictDataset | None]:
        if isinstance(data, DataLoader):
            return (data, None) if return_dataset else data
        tensors = build_tensors_from_data(
            data=data,
            raw_data=data,
            features=self.all_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
        )
        if tensors is None:
            raise ValueError(
                "[BaseModel-prepare_data_loader Error] No data available to create DataLoader."
            )
        dataset = TensorDictDataset(tensors)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return (loader, dataset) if return_dataset else loader

    def fit(
        self,
        train_data: dict | pd.DataFrame | DataLoader,
        valid_data: dict | pd.DataFrame | DataLoader | None = None,
        metrics: (
            list[str] | dict[str, list[str]] | None
        ) = None,  # ['auc', 'logloss'] or {'target1': ['auc', 'logloss'], 'target2': ['mse']}
        epochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 32,
        user_id_column: str | None = None,
        validation_split: float | None = None,
        num_workers: int = 0,
        tensorboard: bool = True,
        auto_distributed_sampler: bool = True,
    ):
        """
        Train the model.

        Args:
            train_data: Training data (dict/df/DataLoader). If distributed, each rank uses its own sampler/batches.
            valid_data: Optional validation data; if None and validation_split is set, a split is created.
            metrics: Metrics names or per-target dict. e.g. {'target1': ['auc', 'logloss'], 'target2': ['mse']} or ['auc', 'logloss'].
            epochs: Training epochs.
            shuffle: Whether to shuffle training data (ignored when a sampler enforces order).
            batch_size: Batch size (per process when distributed).
            user_id_column: Column name for GAUC-style metrics;.
            validation_split: Ratio to split training data when valid_data is None.
            num_workers: DataLoader worker count.
            tensorboard: Enable tensorboard logging.
            auto_distributed_sampler: Attach DistributedSampler automatically when distributed, set False to when data is already sharded per rank.

        Notes:
            - Distributed training uses DDP; init occurs via env vars (RANK/WORLD_SIZE/LOCAL_RANK).
            - All ranks must call evaluate() together because it performs collective ops.
        """
        device_id = self.local_rank if self.device.type == "cuda" else None
        init_process_group(
            self.distributed, self.rank, self.world_size, device_id=device_id
        )
        self.to(self.device)

        if (
            self.distributed
            and dist.is_available()
            and dist.is_initialized()
            and self.ddp_model is None
        ):
            device_ids = (
                [self.local_rank] if self.device.type == "cuda" else None
            )  # device_ids means which device to use in ddp
            output_device = (
                self.local_rank if self.device.type == "cuda" else None
            )  # output_device means which device to place the output in ddp
            object.__setattr__(
                self,
                "ddp_model",
                DDP(
                    self,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=self.ddp_find_unused_parameters,
                ),
            )

        if (
            not self.logger_initialized and self.is_main_process
        ):  # only main process initializes logger
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True
        self.training_logger = (
            TrainingLogger(session=self.session, enable_tensorboard=tensorboard)
            if self.is_main_process
            else None
        )

        self.metrics, self.task_specific_metrics, self.best_metrics_mode = (
            configure_metrics(
                task=self.task, metrics=metrics, target_names=self.target_columns
            )
        )  # ['auc', 'logloss'], {'target1': ['auc', 'logloss'], 'target2': ['mse']}, 'max'

        # Setup default callbacks if none exist
        if len(self.callbacks.callbacks) == 0:
            if self.nums_task == 1:
                monitor_metric = f"val_{self.metrics[0]}"
            else:
                monitor_metric = f"val_{self.metrics[0]}_{self.target_columns[0]}"

            if self.early_stop_patience > 0:
                self.callbacks.append(
                    EarlyStopper(
                        monitor=monitor_metric,
                        patience=self.early_stop_patience,
                        mode=self.best_metrics_mode,
                        restore_best_weights=not self.distributed,
                        verbose=1 if self.is_main_process else 0,
                    )
                )

            if self.is_main_process:
                self.callbacks.append(
                    CheckpointSaver(
                        save_path=self.best_path,
                        monitor=monitor_metric,
                        mode=self.best_metrics_mode,
                        save_best_only=True,
                        verbose=1,
                    )
                )

            if self.scheduler_fn is not None:
                self.callbacks.append(
                    LearningRateScheduler(
                        scheduler=self.scheduler_fn,
                        verbose=1 if self.is_main_process else 0,
                    )
                )

        self.callbacks.set_model(self)
        self.callbacks.set_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "metrics": self.metrics,
            }
        )

        self.early_stopper = EarlyStopper(
            patience=self.early_stop_patience, mode=self.best_metrics_mode
        )
        self.best_metric = (
            float("-inf") if self.best_metrics_mode == "max" else float("inf")
        )

        self.needs_user_ids = check_user_id(
            self.metrics, self.task_specific_metrics
        )  # check user_id needed for GAUC metrics
        self.epoch_index = 0
        self.stop_training = False
        self.best_checkpoint_path = self.best_path

        if not auto_distributed_sampler and self.distributed and self.is_main_process:
            logging.info(
                colorize(
                    "[Distributed Info] auto_distributed_sampler=False; assuming data is already sharded per rank.",
                    color="yellow",
                )
            )

        train_sampler: DistributedSampler | None = None
        if validation_split is not None and valid_data is None:
            train_loader, valid_data = self.handle_validation_split(train_data=train_data, validation_split=validation_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # type: ignore
            if (
                auto_distributed_sampler
                and self.distributed
                and dist.is_available()
                and dist.is_initialized()
            ):
                base_dataset = getattr(train_loader, "dataset", None)
                if base_dataset is not None and not isinstance(
                    getattr(train_loader, "sampler", None), DistributedSampler
                ):
                    train_sampler = DistributedSampler(
                        base_dataset,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=shuffle,
                        drop_last=True,
                    )
                    train_loader = DataLoader(
                        base_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        sampler=train_sampler,
                        collate_fn=collate_fn,
                        num_workers=num_workers,
                        drop_last=True,
                    )
        else:
            if isinstance(train_data, DataLoader):
                if auto_distributed_sampler and self.distributed:
                    train_loader, train_sampler = add_distributed_sampler(
                        train_data,
                        distributed=self.distributed,
                        world_size=self.world_size,
                        rank=self.rank,
                        shuffle=shuffle,
                        drop_last=True,
                        default_batch_size=batch_size,
                        is_main_process=self.is_main_process,
                    )
                    # train_loader, train_sampler = add_distributed_sampler(train_data, distributed=self.distributed, world_size=self.world_size, rank=self.rank, shuffle=shuffle, drop_last=True, default_batch_size=batch_size, is_main_process=self.is_main_process)
                else:
                    train_loader = train_data
            else:
                result = self.prepare_data_loader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, return_dataset=True)
                assert isinstance(result, tuple), "Expected tuple from prepare_data_loader with return_dataset=True"
                loader, dataset = result
                if (
                    auto_distributed_sampler
                    and self.distributed
                    and dataset is not None
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    train_sampler = DistributedSampler(
                        dataset,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=shuffle,
                        drop_last=True,
                    )
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        sampler=train_sampler,
                        collate_fn=collate_fn,
                        num_workers=num_workers,
                        drop_last=True,
                    )
                train_loader = loader

        # If split-based loader was built without sampler, attach here when enabled
        if (
            self.distributed
            and auto_distributed_sampler
            and isinstance(train_loader, DataLoader)
            and train_sampler is None
        ):
            raise NotImplementedError(
                "[BaseModel-fit Error] auto_distributed_sampler with pre-defined DataLoader is not supported yet."
            )
            # train_loader, train_sampler = add_distributed_sampler(train_loader, distributed=self.distributed, world_size=self.world_size, rank=self.rank, shuffle=shuffle, drop_last=True, default_batch_size=batch_size, is_main_process=self.is_main_process)

        valid_loader, valid_user_ids = self.prepare_validation_data(
            valid_data=valid_data,
            batch_size=batch_size,
            needs_user_ids=self.needs_user_ids,
            user_id_column=user_id_column,
            num_workers=num_workers,
            auto_distributed_sampler=auto_distributed_sampler,
        )
        try:
            self.steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:  # streaming data loader does not supported len()
            self.steps_per_epoch = None
            is_streaming = True

        if self.is_main_process:
            self.summary()
            logging.info("")
            if self.training_logger and self.training_logger.enable_tensorboard:
                tb_dir = self.training_logger.tensorboard_logdir
                if tb_dir:
                    user = getpass.getuser()
                    host = socket.gethostname()
                    tb_cmd = f"tensorboard --logdir {tb_dir} --port 6006"
                    ssh_hint = f"ssh -L 6006:localhost:6006 {user}@{host}"
                    logging.info(
                        colorize(f"TensorBoard logs saved to: {tb_dir}", color="cyan")
                    )
                    logging.info(colorize("To view logs, run:", color="cyan"))
                    logging.info(colorize(f"    {tb_cmd}", color="cyan"))
                    logging.info(colorize("Then SSH port forward:", color="cyan"))
                    logging.info(colorize(f"    {ssh_hint}", color="cyan"))

            logging.info("")
            logging.info(colorize("=" * 80, bold=True))
            if is_streaming:
                logging.info(colorize("Start streaming training", bold=True))
            else:
                logging.info(colorize("Start training", bold=True))
            logging.info(colorize("=" * 80, bold=True))
            logging.info("")
            logging.info(colorize(f"Model device: {self.device}", bold=True))

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.epoch_index = epoch

            self.callbacks.on_epoch_begin(epoch)

            if is_streaming and self.is_main_process:
                logging.info("")
                logging.info(
                    colorize(f"Epoch {epoch + 1}/{epochs}", bold=True)
                )  # streaming mode, print epoch header before progress bar

            # handle train result
            if (
                self.distributed
                and isinstance(train_loader, DataLoader)
                and hasattr(train_loader, "sampler")
                and isinstance(train_loader.sampler, DistributedSampler)
            ):
                train_loader.sampler.set_epoch(epoch)
            # Type guard: ensure train_loader is DataLoader for train_epoch
            if not isinstance(train_loader, DataLoader):
                raise TypeError(f"Expected DataLoader for training, got {type(train_loader)}")
            train_result = self.train_epoch(train_loader, is_streaming=is_streaming)
            if isinstance(train_result, tuple):  # [avg_loss, metrics_dict]
                train_loss, train_metrics = train_result
            else:
                train_loss = train_result
                train_metrics = None

            train_log_payload: dict[str, float] = {}
            # handle logging for single-task and multi-task
            if self.nums_task == 1:
                log_str = f"Epoch {epoch + 1}/{epochs} - Train: loss={train_loss:.4f}"
                if train_metrics:
                    metrics_str = ", ".join(
                        [f"{k}={v:.4f}" for k, v in train_metrics.items()]
                    )
                    log_str += f", {metrics_str}"
                if self.is_main_process:
                    logging.info(colorize(log_str))
                train_log_payload["loss"] = float(train_loss)
                if train_metrics:
                    train_log_payload.update(train_metrics)
            else:
                total_loss_val = np.sum(train_loss) if isinstance(train_loss, np.ndarray) else train_loss  # type: ignore
                log_str = (
                    f"Epoch {epoch + 1}/{epochs} - Train: loss={total_loss_val:.4f}"
                )
                if train_metrics:
                    # group metrics by task
                    task_metrics = {}
                    for metric_key, metric_value in train_metrics.items():
                        for target_name in self.target_columns:
                            if metric_key.endswith(f"_{target_name}"):
                                if target_name not in task_metrics:
                                    task_metrics[target_name] = {}
                                metric_name = metric_key.rsplit(f"_{target_name}", 1)[0]
                                task_metrics[target_name][metric_name] = metric_value
                                break
                    if task_metrics:
                        task_metric_strs = []
                        for target_name in self.target_columns:
                            if target_name in task_metrics:
                                metrics_str = ", ".join(
                                    [
                                        f"{k}={v:.4f}"
                                        for k, v in task_metrics[target_name].items()
                                    ]
                                )
                                task_metric_strs.append(f"{target_name}[{metrics_str}]")
                        log_str += ", " + ", ".join(task_metric_strs)
                if self.is_main_process:
                    logging.info(colorize(log_str))
                train_log_payload["loss"] = float(total_loss_val)
                if train_metrics:
                    train_log_payload.update(train_metrics)
            if self.training_logger:
                self.training_logger.log_metrics(
                    train_log_payload, step=epoch + 1, split="train"
                )
            if valid_loader is not None:
                # Call on_validation_begin
                self.callbacks.on_validation_begin()

                # pass user_ids only if needed for GAUC metric
                val_metrics = self.evaluate(
                    valid_loader,
                    user_ids=valid_user_ids if self.needs_user_ids else None,
                    num_workers=num_workers,
                )  # {'auc': 0.75, 'logloss': 0.45} or {'auc_target1': 0.75, 'logloss_target1': 0.45, 'mse_target2': 3.2}
                if self.nums_task == 1:
                    metrics_str = ", ".join(
                        [f"{k}={v:.4f}" for k, v in val_metrics.items()]
                    )
                    if self.is_main_process:
                        logging.info(
                            colorize(
                                f"  Epoch {epoch + 1}/{epochs} - Valid: {metrics_str}",
                                color="cyan",
                            )
                        )
                else:
                    # multi task metrics
                    task_metrics = {}
                    for metric_key, metric_value in val_metrics.items():
                        for target_name in self.target_columns:
                            if metric_key.endswith(f"_{target_name}"):
                                if target_name not in task_metrics:
                                    task_metrics[target_name] = {}
                                metric_name = metric_key.rsplit(f"_{target_name}", 1)[0]
                                task_metrics[target_name][metric_name] = metric_value
                                break
                    task_metric_strs = []
                    for target_name in self.target_columns:
                        if target_name in task_metrics:
                            metrics_str = ", ".join(
                                [
                                    f"{k}={v:.4f}"
                                    for k, v in task_metrics[target_name].items()
                                ]
                            )
                            task_metric_strs.append(f"{target_name}[{metrics_str}]")
                    if self.is_main_process:
                        logging.info(
                            colorize(
                                f"  Epoch {epoch + 1}/{epochs} - Valid: "
                                + ", ".join(task_metric_strs),
                                color="cyan",
                            )
                        )

                # Call on_validation_end
                self.callbacks.on_validation_end()
                if val_metrics and self.training_logger:
                    self.training_logger.log_metrics(
                        val_metrics, step=epoch + 1, split="valid"
                    )

                # Handle empty validation metrics
                if not val_metrics:
                    if self.is_main_process:
                        logging.info(
                            colorize(
                                "Warning: No validation metrics computed. Skipping validation for this epoch.",
                                color="yellow",
                            )
                        )
                    continue

                # Prepare epoch logs for callbacks
                epoch_logs = {**train_log_payload}
                if val_metrics:
                    # Add val_ prefix to validation metrics
                    for k, v in val_metrics.items():
                        epoch_logs[f"val_{k}"] = v
            else:
                # No validation data
                epoch_logs = {**train_log_payload}
                if self.is_main_process:
                    self.save_model(
                        self.checkpoint_path, add_timestamp=False, verbose=False
                    )
                    self.best_checkpoint_path = self.checkpoint_path

            # Call on_epoch_end for all callbacks (handles early stopping, checkpointing, lr scheduling)
            self.callbacks.on_epoch_end(epoch, epoch_logs)

            # Broadcast stop_training flag to all processes
            if self.distributed and dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor(
                    [int(self.stop_training)], device=self.device
                )
                dist.broadcast(stop_tensor, src=0)
                self.stop_training = bool(stop_tensor.item())

            if self.stop_training:
                break
        # Call on_train_end for all callbacks
        self.callbacks.on_train_end()

        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()  # dist.barrier() will wait for all processes, like async all_reduce()
        if self.is_main_process:
            logging.info(" ")
            logging.info(colorize("Training finished.", bold=True))
            logging.info(" ")
        if valid_loader is not None:
            if self.is_main_process:
                logging.info(
                    colorize(f"Load best model from: {self.best_checkpoint_path}")
                )
            if os.path.exists(self.best_checkpoint_path):
                self.load_model(
                    self.best_checkpoint_path, map_location=self.device, verbose=False
                )
            elif self.is_main_process:
                logging.info(
                    colorize(
                        f"Warning: Best checkpoint not found at {self.best_checkpoint_path}, skip loading best model.",
                        color="yellow",
                    )
                )
        if self.training_logger:
            self.training_logger.close()
        return self

    def train_epoch(
        self, train_loader: DataLoader, is_streaming: bool = False
    ) -> Union[float, np.ndarray, tuple[Union[float, np.ndarray], dict]]:
        # use ddp model for distributed training
        model = self.ddp_model if getattr(self, "ddp_model") is not None else self
        accumulated_loss = 0.0
        model.train()  # type: ignore
        num_batches = 0
        y_true_list = []
        y_pred_list = []

        user_ids_list = [] if self.needs_user_ids else None
        tqdm_disable = not self.is_main_process
        if self.steps_per_epoch is not None:
            batch_iter = enumerate(
                tqdm.tqdm(
                    train_loader,
                    desc=f"Epoch {self.epoch_index + 1}",
                    total=self.steps_per_epoch,
                    disable=tqdm_disable,
                )
            )
        else:
            desc = "Batches" if is_streaming else f"Epoch {self.epoch_index + 1}"
            batch_iter = enumerate(
                tqdm.tqdm(train_loader, desc=desc, disable=tqdm_disable)
            )
        for batch_index, batch_data in batch_iter:
            batch_dict = batch_to_dict(batch_data)
            X_input, y_true = self.get_input(batch_dict, require_labels=True)
            # call via __call__ so DDP hooks run (no grad sync if calling .forward directly)
            y_pred = model(X_input)  # type: ignore

            loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()
            total_loss = loss + reg_loss
            self.optimizer_fn.zero_grad()
            total_loss.backward()

            params = model.parameters() if self.ddp_model is not None else self.parameters()  # type: ignore # ddp model parameters or self parameters
            nn.utils.clip_grad_norm_(params, self.max_gradient_norm)
            self.optimizer_fn.step()
            accumulated_loss += loss.item()

            if y_true is not None:
                y_true_list.append(y_true.detach().cpu().numpy())
            if self.needs_user_ids and user_ids_list is not None:
                batch_user_id = get_user_ids(
                    data=batch_dict, id_columns=self.id_columns
                )
                if batch_user_id is not None:
                    user_ids_list.append(batch_user_id)
            if y_pred is not None and isinstance(y_pred, torch.Tensor):
                y_pred_list.append(y_pred.detach().cpu().numpy())
            num_batches += 1
        if self.distributed and dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(
                [accumulated_loss, num_batches], device=self.device, dtype=torch.float32
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            accumulated_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        avg_loss = accumulated_loss / max(num_batches, 1)

        y_true_all_local = np.concatenate(y_true_list, axis=0) if y_true_list else None
        y_pred_all_local = np.concatenate(y_pred_list, axis=0) if y_pred_list else None
        combined_user_ids_local = (
            np.concatenate(user_ids_list, axis=0)
            if self.needs_user_ids and user_ids_list
            else None
        )

        # gather across ranks even when local is empty to avoid DDP hang
        y_true_all = gather_numpy(self, y_true_all_local)
        y_pred_all = gather_numpy(self, y_pred_all_local)
        combined_user_ids = (
            gather_numpy(self, combined_user_ids_local) if self.needs_user_ids else None
        )

        if (
            y_true_all is not None
            and y_pred_all is not None
            and len(y_true_all) > 0
            and len(y_pred_all) > 0
        ):
            metrics_dict = evaluate_metrics(
                y_true=y_true_all,
                y_pred=y_pred_all,
                metrics=self.metrics,
                task=self.task,
                target_names=self.target_columns,
                task_specific_metrics=self.task_specific_metrics,
                user_ids=combined_user_ids,
            )
            return avg_loss, metrics_dict
        return avg_loss

    def prepare_validation_data(
        self,
        valid_data: dict | pd.DataFrame | DataLoader | None,
        batch_size: int,
        needs_user_ids: bool,
        user_id_column: str | None = "user_id",
        num_workers: int = 0,
        auto_distributed_sampler: bool = True,
    ) -> tuple[DataLoader | None, np.ndarray | None]:
        if valid_data is None:
            return None, None
        if isinstance(valid_data, DataLoader):
            if auto_distributed_sampler and self.distributed:
                raise NotImplementedError(
                    "[BaseModel-prepare_validation_data Error] auto_distributed_sampler with pre-defined DataLoader is not supported yet."
                )
                # valid_loader, _ = add_distributed_sampler(valid_data, distributed=self.distributed, world_size=self.world_size, rank=self.rank, shuffle=False, drop_last=False, default_batch_size=batch_size, is_main_process=self.is_main_process)
            else:
                valid_loader = valid_data
            return valid_loader, None
        valid_sampler = None
        valid_loader, valid_dataset = self.prepare_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, return_dataset=True)  # type: ignore
        if (
            auto_distributed_sampler
            and self.distributed
            and valid_dataset is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=valid_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        valid_user_ids = None
        if needs_user_ids:
            if user_id_column is None:
                raise ValueError(
                    "[BaseModel-validation Error] user_id_column must be specified when user IDs are needed for validation metrics."
                )
            # In distributed mode, user_ids will be collected during evaluation from each batch
            # and gathered across all processes, so we don't pre-extract them here
            if not self.distributed:
                valid_user_ids = get_user_ids(
                    data=valid_data, id_columns=user_id_column
                )
        return valid_loader, valid_user_ids

    def evaluate(
        self,
        data: dict | pd.DataFrame | DataLoader,
        metrics: list[str] | dict[str, list[str]] | None = None,
        batch_size: int = 32,
        user_ids: np.ndarray | None = None,
        user_id_column: str = "user_id",
        num_workers: int = 0,
    ) -> dict:
        """
        **IMPORTANT for Distributed Training:**
        in distributed mode, this method uses collective communication operations (all_gather).
        all processes must call this method simultaneously, even if you only want results on rank 0.
        failing to do so will cause the program to hang indefinitely.

        Evaluate the model on the given data.

        Args:
            data: Evaluation data (dict/df/DataLoader).
            metrics: Metrics names or per-target dict. e.g. {'target1': ['auc', 'logloss'], 'target2': ['mse']} or ['auc', 'logloss'].
            batch_size: Batch size (per process when distributed).
            user_ids: Optional array of user IDs for GAUC-style metrics; if None and needed, will be extracted from data using user_id_column. e.g. np.array([...])
            user_id_column: Column name for user IDs if user_ids is not provided. e.g. 'user_id'
            num_workers: DataLoader worker count.
        """
        model = self.ddp_model if getattr(self, "ddp_model", None) is not None else self
        model.eval()
        eval_metrics = metrics if metrics is not None else self.metrics
        if eval_metrics is None:
            raise ValueError(
                "[BaseModel-evaluate Error] No metrics specified for evaluation. Please provide metrics parameter or call fit() first."
            )
        needs_user_ids = check_user_id(eval_metrics, self.task_specific_metrics)

        if isinstance(data, DataLoader):
            data_loader = data
        else:
            if user_ids is None and needs_user_ids:
                user_ids = get_user_ids(data=data, id_columns=user_id_column)
            data_loader = self.prepare_data_loader(
                data, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        y_true_list = []
        y_pred_list = []
        collected_user_ids = []
        batch_count = 0
        with torch.no_grad():
            for batch_data in data_loader:
                batch_count += 1
                batch_dict = batch_to_dict(batch_data)
                X_input, y_true = self.get_input(batch_dict, require_labels=True)
                y_pred = model(X_input)
                if y_true is not None:
                    y_true_list.append(y_true.cpu().numpy())
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.cpu().numpy())
                if needs_user_ids and user_ids is None:
                    batch_user_id = get_user_ids(
                        data=batch_dict, id_columns=self.id_columns
                    )
                    if batch_user_id is not None:
                        collected_user_ids.append(batch_user_id)
        if self.is_main_process:
            logging.info(" ")
            logging.info(
                colorize(f"  Evaluation batches processed: {batch_count}", color="cyan")
            )
        y_true_all_local = np.concatenate(y_true_list, axis=0) if y_true_list else None
        y_pred_all_local = np.concatenate(y_pred_list, axis=0) if y_pred_list else None

        # Convert metrics to list if it's a dict
        if isinstance(eval_metrics, dict):
            # For dict metrics, we need to collect all unique metric names
            unique_metrics = []
            for task_metrics in eval_metrics.values():
                for m in task_metrics:
                    if m not in unique_metrics:
                        unique_metrics.append(m)
            metrics_to_use = unique_metrics
        else:
            metrics_to_use = eval_metrics
        final_user_ids_local = user_ids
        if final_user_ids_local is None and collected_user_ids:
            final_user_ids_local = np.concatenate(collected_user_ids, axis=0)

        # gather across ranks even when local arrays are empty to keep collectives aligned
        y_true_all = gather_numpy(self, y_true_all_local)
        y_pred_all = gather_numpy(self, y_pred_all_local)
        final_user_ids = (
            gather_numpy(self, final_user_ids_local) if needs_user_ids else None
        )
        if (
            y_true_all is None
            or y_pred_all is None
            or len(y_true_all) == 0
            or len(y_pred_all) == 0
        ):
            if self.is_main_process:
                logging.info(
                    colorize(
                        "  Warning: Not enough evaluation data to compute metrics after gathering",
                        color="yellow",
                    )
                )
            return {}
        if self.is_main_process:
            logging.info(
                colorize(f"  Evaluation samples: {y_true_all.shape[0]}", color="cyan")
            )
        metrics_dict = evaluate_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            metrics=metrics_to_use,
            task=self.task,
            target_names=self.target_columns,
            task_specific_metrics=self.task_specific_metrics,
            user_ids=final_user_ids,
        )
        return metrics_dict

    def predict(
        self,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: Literal["csv", "parquet"] = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: bool = True,
        streaming_chunk_size: int = 10000,
        num_workers: int = 0,
    ) -> pd.DataFrame | np.ndarray:
        """
        Note: predict does not support distributed mode currently, consider it as a single-process operation.
        Make predictions on the given data.

        Args:
            data: Input data for prediction (file path, dict, DataFrame, or DataLoader).
            batch_size: Batch size for prediction (per process when distributed).
            save_path: Optional path to save predictions; if None, predictions are not saved to disk.
            save_format: Format to save predictions ('csv' or 'parquet').
            include_ids: Whether to include ID columns in the output; if None, includes if id_columns are set.
            id_columns: Column name(s) to use as IDs; if None, uses model's id_columns.
            return_dataframe: Whether to return predictions as a pandas DataFrame; if False, returns a NumPy array.
            streaming_chunk_size: Number of rows per chunk when using streaming mode for large datasets.
            num_workers: DataLoader worker count.
        """
        self.eval()
        # Use prediction-time id_columns if provided, otherwise fall back to model's id_columns
        predict_id_columns = id_columns if id_columns is not None else self.id_columns
        if isinstance(predict_id_columns, str):
            predict_id_columns = [predict_id_columns]

        if include_ids is None:
            include_ids = bool(predict_id_columns)
        include_ids = include_ids and bool(predict_id_columns)

        # Use streaming mode for large file saves without loading all data into memory
        if save_path is not None and not return_dataframe:
            return self.predict_streaming(
                data=data,
                batch_size=batch_size,
                save_path=save_path,
                save_format=save_format,
                include_ids=include_ids,
                streaming_chunk_size=streaming_chunk_size,
                return_dataframe=return_dataframe,
                id_columns=predict_id_columns,
            )

        # Create DataLoader based on data type
        if isinstance(data, DataLoader):
            data_loader = data
        elif isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=self.target_columns,
                id_columns=predict_id_columns,
            )
            data_loader = rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                load_full=False,
                chunk_size=streaming_chunk_size,
            )
        else:
            data_loader = self.prepare_data_loader(
                data, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        y_pred_list = []
        id_buffers = (
            {name: [] for name in (predict_id_columns or [])} if include_ids else {}
        )
        id_arrays = None

        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Predicting"):
                batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
                X_input, _ = self.get_input(batch_dict, require_labels=False)
                y_pred = self(X_input)
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.detach().cpu().numpy())
                if include_ids and predict_id_columns and batch_dict.get("ids"):
                    for id_name in predict_id_columns:
                        if id_name not in batch_dict["ids"]:
                            continue
                        id_tensor = batch_dict["ids"][id_name]
                        id_np = (
                            id_tensor.detach().cpu().numpy()
                            if isinstance(id_tensor, torch.Tensor)
                            else np.asarray(id_tensor)
                        )
                        id_buffers[id_name].append(
                            id_np.reshape(id_np.shape[0], -1)
                            if id_np.ndim == 1
                            else id_np
                        )
        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
        else:
            y_pred_all = np.array([])

        if y_pred_all.ndim == 1:
            y_pred_all = y_pred_all.reshape(-1, 1)
        if y_pred_all.size == 0:
            num_outputs = len(self.target_columns) if self.target_columns else 1
            y_pred_all = y_pred_all.reshape(0, num_outputs)
        num_outputs = y_pred_all.shape[1]
        pred_columns: list[str] = []
        if self.target_columns:
            for name in self.target_columns[:num_outputs]:
                pred_columns.append(f"{name}")
        while len(pred_columns) < num_outputs:
            pred_columns.append(f"pred_{len(pred_columns)}")
        if include_ids and predict_id_columns:
            id_arrays = {}
            for id_name, pieces in id_buffers.items():
                if pieces:
                    concatenated = np.concatenate(
                        [p.reshape(p.shape[0], -1) for p in pieces], axis=0
                    )
                    id_arrays[id_name] = concatenated.reshape(concatenated.shape[0])
                else:
                    id_arrays[id_name] = np.array([], dtype=np.int64)
            if return_dataframe:
                id_df = pd.DataFrame(id_arrays)
                pred_df = pd.DataFrame(y_pred_all, columns=pred_columns)
                if len(id_df) and len(pred_df) and len(id_df) != len(pred_df):
                    raise ValueError(
                        f"[BaseModel-predict Error] Mismatch between id rows ({len(id_df)}) and prediction rows ({len(pred_df)})."
                    )
                output = pd.concat([id_df, pred_df], axis=1)
            else:
                output = y_pred_all
        else:
            output = (
                pd.DataFrame(y_pred_all, columns=pred_columns)
                if return_dataframe
                else y_pred_all
            )
        if save_path is not None:
            if save_format not in ("csv", "parquet"):
                raise ValueError(
                    f"[BaseModel-predict Error] Unsupported save_format '{save_format}'. Choose from 'csv' or 'parquet'."
                )
            suffix = ".csv" if save_format == "csv" else ".parquet"
            target_path = resolve_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=suffix,
                add_timestamp=True if save_path is None else False,
            )
            if isinstance(output, pd.DataFrame):
                df_to_save = output
            else:
                df_to_save = pd.DataFrame(y_pred_all, columns=pred_columns)
                if include_ids and predict_id_columns and id_arrays is not None:
                    id_df = pd.DataFrame(id_arrays)
                    if len(id_df) and len(df_to_save) and len(id_df) != len(df_to_save):
                        raise ValueError(
                            f"[BaseModel-predict Error] Mismatch between id rows ({len(id_df)}) and prediction rows ({len(df_to_save)})."
                        )
                    df_to_save = pd.concat([id_df, df_to_save], axis=1)
            if save_format == "csv":
                df_to_save.to_csv(target_path, index=False)
            else:
                df_to_save.to_parquet(target_path, index=False)
            logging.info(
                colorize(f"Predictions saved to: {target_path}", color="green")
            )
        return output

    def predict_streaming(
        self,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: Literal["csv", "parquet"],
        include_ids: bool,
        streaming_chunk_size: int,
        return_dataframe: bool,
        id_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=self.target_columns,
                id_columns=id_columns,
            )
            data_loader = rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                load_full=False,
                chunk_size=streaming_chunk_size,
            )
        elif not isinstance(data, DataLoader):
            data_loader = self.prepare_data_loader(
                data,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            data_loader = data

        suffix = ".csv" if save_format == "csv" else ".parquet"
        target_path = resolve_save_path(
            path=save_path,
            default_dir=self.session.predictions_dir,
            default_name="predictions",
            suffix=suffix,
            add_timestamp=True if save_path is None else False,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        header_written = target_path.exists() and target_path.stat().st_size > 0
        parquet_writer = None

        pred_columns = None
        collected_frames = []  # only used when return_dataframe is True

        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Predicting"):
                batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
                X_input, _ = self.get_input(batch_dict, require_labels=False)
                y_pred = self.forward(X_input)
                if y_pred is None or not isinstance(y_pred, torch.Tensor):
                    continue
                y_pred_np = y_pred.detach().cpu().numpy()
                if y_pred_np.ndim == 1:
                    y_pred_np = y_pred_np.reshape(-1, 1)
                if pred_columns is None:
                    num_outputs = y_pred_np.shape[1]
                    pred_columns = []
                    if self.target_columns:
                        for name in self.target_columns[:num_outputs]:
                            pred_columns.append(f"{name}")
                    while len(pred_columns) < num_outputs:
                        pred_columns.append(f"pred_{len(pred_columns)}")

                id_arrays_batch = {}
                if include_ids and id_columns and batch_dict.get("ids"):
                    for id_name in id_columns:
                        if id_name not in batch_dict["ids"]:
                            continue
                        id_tensor = batch_dict["ids"][id_name]
                        id_np = (
                            id_tensor.detach().cpu().numpy()
                            if isinstance(id_tensor, torch.Tensor)
                            else np.asarray(id_tensor)
                        )
                        id_arrays_batch[id_name] = id_np.reshape(id_np.shape[0])

                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if id_arrays_batch:
                    id_df = pd.DataFrame(id_arrays_batch)
                    if len(id_df) and len(df_batch) and len(id_df) != len(df_batch):
                        raise ValueError(
                            f"Mismatch between id rows ({len(id_df)}) and prediction rows ({len(df_batch)})."
                        )
                    df_batch = pd.concat([id_df, df_batch], axis=1)

                if save_format == "csv":
                    df_batch.to_csv(
                        target_path, mode="a", header=not header_written, index=False
                    )
                    header_written = True
                else:
                    try:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                    except ImportError as exc:  # pragma: no cover
                        raise ImportError(
                            "[BaseModel-predict-streaming Error] Parquet streaming save requires pyarrow to be installed."
                        ) from exc
                    table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(target_path, table.schema)
                    parquet_writer.write_table(table)
                if return_dataframe:
                    collected_frames.append(df_batch)
        if parquet_writer is not None:
            parquet_writer.close()
        logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        if return_dataframe:
            return (
                pd.concat(collected_frames, ignore_index=True)
                if collected_frames
                else pd.DataFrame(columns=pred_columns or [])
            )
        return pd.DataFrame(columns=pred_columns or [])

    def save_model(
        self,
        save_path: str | Path | None = None,
        add_timestamp: bool | None = None,
        verbose: bool = True,
    ):
        add_timestamp = False if add_timestamp is None else add_timestamp
        target_path = resolve_save_path(
            path=save_path,
            default_dir=self.session_path,
            default_name=self.model_name,
            suffix=".pt",
            add_timestamp=add_timestamp,
        )
        model_path = Path(target_path)

        model_to_save = (
            self.ddp_model.module
            if getattr(self, "ddp_model", None) is not None
            else self
        )
        torch.save(model_to_save.state_dict(), model_path)
        # torch.save(self.state_dict(), model_path)

        config_path = self.features_config_path
        features_config = {
            "all_features": self.all_features,
            "target": self.target_columns,
            "id_columns": self.id_columns,
            "version": __version__,
        }
        with open(config_path, "wb") as f:
            pickle.dump(features_config, f)
        self.features_config_path = str(config_path)
        if verbose:
            logging.info(
                colorize(
                    f"Model saved to: {model_path}, features config saved to: {config_path}, NextRec version: {__version__}",
                    color="green",
                )
            )

    def load_model(
        self,
        save_path: str | Path,
        map_location: str | torch.device | None = "cpu",
        verbose: bool = True,
    ):
        self.to(self.device)
        base_path = Path(save_path)
        if base_path.is_dir():
            model_files = sorted(base_path.glob("*.pt"))
            if not model_files:
                raise FileNotFoundError(
                    f"[BaseModel-load-model Error] No *.pt file found in directory: {base_path}"
                )
            model_path = model_files[-1]
            config_dir = base_path
        else:
            model_path = (
                base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
            )
            config_dir = model_path.parent
        if not model_path.exists():
            raise FileNotFoundError(
                f"[BaseModel-load-model Error] Model file does not exist: {model_path}"
            )

        state_dict = torch.load(model_path, map_location=map_location)
        self.load_state_dict(state_dict)

        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(
                f"[BaseModel-load-model Error] features_config.pkl not found in: {config_dir}"
            )
        with open(features_config_path, "rb") as f:
            features_config = pickle.load(f)

        all_features = features_config.get("all_features", [])
        target = features_config.get("target", [])
        id_columns = features_config.get("id_columns", [])
        dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
        sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
        sequence_features = [f for f in all_features if isinstance(f, SequenceFeature)]
        self.set_all_features(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            id_columns=id_columns,
        )

        cfg_version = features_config.get("version")
        if verbose:
            logging.info(
                colorize(
                    f"Model weights loaded from: {model_path}, features config loaded from: {features_config_path}, NextRec version: {cfg_version}",
                    color="green",
                )
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device | None = "cpu",
        device: str | torch.device = "cpu",
        session_id: str | None = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """
        Load a model from a checkpoint path. The checkpoint path should contain:
        a .pt file and a features_config.pkl file.
        """
        base_path = Path(checkpoint_path)
        verbose = kwargs.pop("verbose", True)
        if base_path.is_dir():
            model_candidates = sorted(base_path.glob("*.pt"))
            if not model_candidates:
                raise FileNotFoundError(
                    f"[BaseModel-from-checkpoint Error] No *.pt file found under: {base_path}"
                )
            model_file = model_candidates[-1]
            config_dir = base_path
        else:
            model_file = (
                base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
            )
            config_dir = model_file.parent
        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(
                f"[BaseModel-from-checkpoint Error] features_config.pkl not found next to checkpoint: {features_config_path}"
            )
        with open(features_config_path, "rb") as f:
            features_config = pickle.load(f)
        all_features = features_config.get("all_features", [])
        target = features_config.get("target", [])
        id_columns = features_config.get("id_columns", [])

        dense_features = [f for f in all_features if isinstance(f, DenseFeature)]
        sparse_features = [f for f in all_features if isinstance(f, SparseFeature)]
        sequence_features = [f for f in all_features if isinstance(f, SequenceFeature)]

        model = cls(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            id_columns=id_columns,
            device=str(device),
            session_id=session_id,
            **kwargs,
        )
        model.load_model(model_file, map_location=map_location, verbose=verbose)
        return model

    def summary(self):
        logger = logging.getLogger()

        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        logger.info(
            colorize(
                f"Model Summary: {self.model_name}", color="bright_blue", bold=True
            )
        )
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))

        logger.info("")
        logger.info(colorize("[1] Feature Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))

        if self.dense_features:
            logger.info(f"Dense Features ({len(self.dense_features)}):")
            for i, feat in enumerate(self.dense_features, 1):
                embed_dim = feat.embedding_dim if hasattr(feat, "embedding_dim") else 1
                logger.info(f"  {i}. {feat.name:20s}")

        if self.sparse_features:
            logger.info(f"\nSparse Features ({len(self.sparse_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sparse_features)
            max_embed_name_len = max(
                len(feat.embedding_name) for feat in self.sparse_features
            )
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2

            logger.info(
                f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10}"
            )
            logger.info(
                f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10}"
            )
            for i, feat in enumerate(self.sparse_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, "vocab_size") else "N/A"
                embed_dim = (
                    feat.embedding_dim if hasattr(feat, "embedding_dim") else "N/A"
                )
                logger.info(
                    f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10}"
                )

        if self.sequence_features:
            logger.info(f"\nSequence Features ({len(self.sequence_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sequence_features)
            max_embed_name_len = max(
                len(feat.embedding_name) for feat in self.sequence_features
            )
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2

            logger.info(
                f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10} {'Max Len':>10}"
            )
            logger.info(
                f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10} {'-'*10}"
            )
            for i, feat in enumerate(self.sequence_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, "vocab_size") else "N/A"
                embed_dim = (
                    feat.embedding_dim if hasattr(feat, "embedding_dim") else "N/A"
                )
                max_len = feat.max_len if hasattr(feat, "max_len") else "N/A"
                logger.info(
                    f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10} {str(max_len):>10}"
                )

        logger.info("")
        logger.info(colorize("[2] Model Parameters", color="cyan", bold=True))
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

        logger.info("")
        logger.info(colorize("[3] Training Configuration", color="cyan", bold=True))
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
        if hasattr(self, "loss_weights"):
            logger.info(f"Loss Weights:            {self.loss_weights}")

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


class BaseMatchModel(BaseModel):
    """
    Base class for match (retrieval/recall) models

    - Pointwise: predicts a user-item match score/probability using labels (default target: 'label')
    - Pairwise/Listwise: trains with in-batch negatives; labels can be omitted by setting target=None
    """

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def default_task(self) -> str:
        return "binary"

    @property
    def support_training_modes(self) -> list[str]:
        """
        Returns list of supported training modes for this model.
        Override in subclasses to restrict training modes.

        Returns:
            List of supported modes: ['pointwise', 'pairwise', 'listwise']
        """
        return ["pointwise", "pairwise", "listwise"]

    def __init__(
        self,
        user_dense_features: list[DenseFeature] | None = None,
        user_sparse_features: list[SparseFeature] | None = None,
        user_sequence_features: list[SequenceFeature] | None = None,
        item_dense_features: list[DenseFeature] | None = None,
        item_sparse_features: list[SparseFeature] | None = None,
        item_sequence_features: list[SequenceFeature] | None = None,
        training_mode: Literal["pointwise", "pairwise", "listwise"] = "pointwise",
        num_negative_samples: int = 4,
        temperature: float = 1.0,
        similarity_metric: Literal["dot", "cosine", "euclidean"] = "dot",
        device: str = "cpu",
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        early_stop_patience: int = 20,
        target: list[str] | str | None = "label",
        id_columns: list[str] | str | None = None,
        task: str | list[str] | None = None,
        session_id: str | None = None,
        callbacks: list[Callback] | None = None,
        distributed: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        ddp_find_unused_parameters: bool = False,
        **kwargs,
    ):

        all_dense_features = []
        all_sparse_features = []
        all_sequence_features = []

        if user_dense_features:
            all_dense_features.extend(user_dense_features)
        if item_dense_features:
            all_dense_features.extend(item_dense_features)
        if user_sparse_features:
            all_sparse_features.extend(user_sparse_features)
        if item_sparse_features:
            all_sparse_features.extend(item_sparse_features)
        if user_sequence_features:
            all_sequence_features.extend(user_sequence_features)
        if item_sequence_features:
            all_sequence_features.extend(item_sequence_features)

        super(BaseMatchModel, self).__init__(
            dense_features=all_dense_features,
            sparse_features=all_sparse_features,
            sequence_features=all_sequence_features,
            target=target,
            id_columns=id_columns,
            task=task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            session_id=session_id,
            callbacks=callbacks,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            **kwargs,
        )

        self.user_dense_features = (
            list(user_dense_features) if user_dense_features else []
        )
        self.user_sparse_features = (
            list(user_sparse_features) if user_sparse_features else []
        )
        self.user_sequence_features = (
            list(user_sequence_features) if user_sequence_features else []
        )

        self.item_dense_features = (
            list(item_dense_features) if item_dense_features else []
        )
        self.item_sparse_features = (
            list(item_sparse_features) if item_sparse_features else []
        )
        self.item_sequence_features = (
            list(item_sequence_features) if item_sequence_features else []
        )

        self.training_mode = training_mode
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric

        self.user_feature_names = [
            f.name
            for f in (
                self.user_dense_features
                + self.user_sparse_features
                + self.user_sequence_features
            )
        ]
        self.item_feature_names = [
            f.name
            for f in (
                self.item_dense_features
                + self.item_sparse_features
                + self.item_sequence_features
            )
        ]

    def get_user_features(self, X_input: dict) -> dict:
        return {
            name: X_input[name] for name in self.user_feature_names if name in X_input
        }

    def get_item_features(self, X_input: dict) -> dict:
        return {
            name: X_input[name] for name in self.item_feature_names if name in X_input
        }

    def compile(
        self,
        optimizer: str | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: (
            str
            | torch.optim.lr_scheduler._LRScheduler
            | torch.optim.lr_scheduler.LRScheduler
            | type[torch.optim.lr_scheduler._LRScheduler]
            | type[torch.optim.lr_scheduler.LRScheduler]
            | None
        ) = None,
        scheduler_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Configure the match model for training.

        This mirrors `BaseModel.compile()` and additionally validates `training_mode`.
        """
        if self.training_mode not in self.support_training_modes:
            raise ValueError(
                f"{self.model_name} does not support training_mode='{self.training_mode}'. Supported modes: {self.support_training_modes}"
            )

        default_loss_by_mode: dict[str, str] = {
            "pointwise": "bce",
            "pairwise": "bpr",
            "listwise": "sampled_softmax",
        }

        effective_loss: str | nn.Module | list[str | nn.Module] | None = loss
        if effective_loss is None:
            effective_loss = default_loss_by_mode[self.training_mode]
        elif isinstance(effective_loss, (str,)):
            if (
                self.training_mode in {"pairwise", "listwise"}
                and effective_loss in {"bce", "binary_crossentropy"}
            ):
                effective_loss = default_loss_by_mode[self.training_mode]
        elif isinstance(effective_loss, list):
            if not effective_loss:
                effective_loss = [default_loss_by_mode[self.training_mode]]
            else:
                first = effective_loss[0]
                if (
                    self.training_mode in {"pairwise", "listwise"}
                    and isinstance(first, str)
                    and first in {"bce", "binary_crossentropy"}
                ):
                    effective_loss = [
                        default_loss_by_mode[self.training_mode],
                        *effective_loss[1:],
                    ]
        return super().compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            loss=effective_loss,
            loss_params=loss_params,
            loss_weights=loss_weights,
            callbacks=callbacks,
        )

    def inbatch_logits(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == "dot":
            logits = torch.matmul(user_emb, item_emb.t())
        elif self.similarity_metric == "cosine":
            user_norm = F.normalize(user_emb, p=2, dim=-1)
            item_norm = F.normalize(item_emb, p=2, dim=-1)
            logits = torch.matmul(user_norm, item_norm.t())
        elif self.similarity_metric == "euclidean":
            user_sq = (user_emb**2).sum(dim=1, keepdim=True)  # [B, 1]
            item_sq = (item_emb**2).sum(dim=1, keepdim=True).t()  # [1, B]
            logits = -(user_sq + item_sq - 2.0 * torch.matmul(user_emb, item_emb.t()))
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        return logits / self.temperature

    def compute_similarity(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        if self.similarity_metric == "dot":
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                # [batch_size, num_items, emb_dim] @ [batch_size, num_items, emb_dim]
                similarity = torch.sum(
                    user_emb * item_emb, dim=-1
                )  # [batch_size, num_items]
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                # [batch_size, emb_dim] @ [batch_size, num_items, emb_dim]
                user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
                similarity = torch.sum(
                    user_emb_expanded * item_emb, dim=-1
                )  # [batch_size, num_items]
            else:
                similarity = torch.sum(user_emb * item_emb, dim=-1)  # [batch_size]

        elif self.similarity_metric == "cosine":
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb_expanded = user_emb.unsqueeze(1)
                similarity = F.cosine_similarity(user_emb_expanded, item_emb, dim=-1)
            else:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)

        elif self.similarity_metric == "euclidean":
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                distance = torch.sum((user_emb - item_emb) ** 2, dim=-1)
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb_expanded = user_emb.unsqueeze(1)
                distance = torch.sum((user_emb_expanded - item_emb) ** 2, dim=-1)
            else:
                distance = torch.sum((user_emb - item_emb) ** 2, dim=-1)
            similarity = -distance

        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        similarity = similarity / self.temperature
        return similarity

    def user_tower(self, user_input: dict) -> torch.Tensor:
        raise NotImplementedError

    def item_tower(self, item_input: dict) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, X_input: dict
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        user_input = self.get_user_features(X_input)
        item_input = self.get_item_features(X_input)

        user_emb = self.user_tower(user_input)  # [B, D]
        item_emb = self.item_tower(item_input)  # [B, D]

        if self.training and self.training_mode in ["pairwise", "listwise"]:
            return user_emb, item_emb

        similarity = self.compute_similarity(user_emb, item_emb)  # [B]

        if self.training_mode == "pointwise":
            return torch.sigmoid(similarity)
        else:
            return similarity

    def compute_loss(self, y_pred, y_true):
        if self.training_mode == "pointwise":
            return super().compute_loss(y_pred, y_true)

        # pairwise / listwise using inbatch neg
        elif self.training_mode in ["pairwise", "listwise"]:
            if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
                raise ValueError(
                    "For pairwise/listwise training, forward should return (user_emb, item_emb). Please check BaseMatchModel.forward implementation."
                )
            user_emb, item_emb = y_pred  # [B, D], [B, D]
            batch_size = user_emb.size(0)
            if batch_size < 2:
                return torch.tensor(0.0, device=user_emb.device)

            logits = self.inbatch_logits(user_emb, item_emb)  # [B, B]

            eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
            pos_logits = logits.diag()  # [B]
            neg_logits = logits.masked_select(~eye).view(batch_size, batch_size - 1)  # [B, B-1]

            loss_fn = self.loss_fn[0] if getattr(self, "loss_fn", None) else None
            if isinstance(loss_fn, SampledSoftmaxLoss):
                loss = loss_fn(pos_logits, neg_logits)
            elif isinstance(loss_fn, (BPRLoss, HingeLoss)):
                loss = loss_fn(pos_logits, neg_logits)
            elif isinstance(loss_fn, TripletLoss):
                neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(
                    batch_size, batch_size - 1, item_emb.size(-1)
                )
                loss = loss_fn(user_emb, item_emb, neg_emb)
            elif isinstance(loss_fn, InfoNCELoss) and self.similarity_metric == "dot":
                neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(
                    batch_size, batch_size - 1, item_emb.size(-1)
                )
                loss = loss_fn(user_emb, item_emb, neg_emb)
            else:
                targets = torch.arange(batch_size, device=logits.device)
                loss = F.cross_entropy(logits, targets)

            if self.loss_weights is not None:
                loss *= float(self.loss_weights[0])
            return loss
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def prepare_feature_data(
        self, data: dict | pd.DataFrame | DataLoader, features: list, batch_size: int
    ) -> DataLoader:
        """Prepare data loader for specific features."""
        if isinstance(data, DataLoader):
            return data
        tensors = build_tensors_from_data(
            data=data,
            raw_data=data,
            features=features,
            target_columns=[],
            id_columns=[],
        )
        if tensors is None:
            raise ValueError(
                "[BaseMatchModel-prepare_feature_data Error] No data available to create DataLoader."
            )
        dataset = TensorDictDataset(tensors)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def encode_user(
        self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512
    ) -> np.ndarray:
        self.eval()
        all_user_features = (
            self.user_dense_features
            + self.user_sparse_features
            + self.user_sequence_features
        )
        data_loader = self.prepare_feature_data(data, all_user_features, batch_size)

        embeddings_list = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding users"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                user_input = self.get_user_features(batch_dict["features"])
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)

    def encode_item(
        self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512
    ) -> np.ndarray:
        self.eval()
        all_item_features = (
            self.item_dense_features
            + self.item_sparse_features
            + self.item_sequence_features
        )
        data_loader = self.prepare_feature_data(data, all_item_features, batch_size)

        embeddings_list = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding items"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                item_input = self.get_item_features(batch_dict["features"])
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
