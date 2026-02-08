"""
Base Model & Base Match Model Class

Date: create on 27/10/2025
Checkpoint: edit on 08/02/2026
Author: Yang Zhou,zyaztec@gmail.com
"""

import getpass
import inspect
import logging
import os
import sys
import pickle
import socket
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Literal, cast, overload
import io
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

try:
    import swanlab  # type: ignore
except Exception:  # pragma: no cover
    swanlab = None

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nextrec import __version__
from nextrec.basic.asserts import assert_task
from nextrec.basic.callback import (
    CallbackList,
    CheckpointSaver,
    EarlyStopper,
    LearningRateScheduler,
)
from nextrec.basic.features import (
    DenseFeature,
    FeatureSet,
    SequenceFeature,
    SparseFeature,
)
from nextrec.basic.heads import RetrievalHead
from nextrec.basic.loggers import TrainingLogger, colorize, format_kv, setup_logger
from nextrec.basic.metrics import (
    check_user_id,
    compute_confusion_matrix,
    configure_metrics,
    evaluate_metrics,
    resolve_thresholds,
)
from nextrec.basic.summary import SummarySet
from nextrec.basic.session import create_session, get_save_path
from nextrec.data.batch_utils import batch_to_dict, collate_fn
from nextrec.data.data_processing import (
    get_column_data,
    get_user_ids,
)
from nextrec.data.dataloader import (
    RecDataLoader,
    TensorDictDataset,
    build_tensors_from_data,
)
from nextrec.loss.grad_norm import GradNormLossWeighting
from nextrec.loss.listwise import InfoNCELoss, SampledSoftmaxLoss
from nextrec.loss.pairwise import BPRLoss, HingeLoss, TripletLoss
from nextrec.utils.loss import get_loss_fn
from nextrec.loss.grad_norm import get_grad_norm_shared_params
from nextrec.utils.console import display_metrics_table, progress, render_confusion_block
from nextrec.utils.timing import StageTimer
from nextrec.utils.torch_utils import (
    add_distributed_sampler,
    get_device,
    gather_numpy,
    get_optimizer,
    get_scheduler,
    init_process_group,
    to_tensor,
)
from nextrec.utils.config import safe_value
from nextrec.utils.model import (
    compute_ranking_loss,
    get_loss_list,
)
from nextrec.utils.onnx_utils import (
    OnnxModelWrapper,
    build_onnx_input_feed,
    create_dummy_inputs,
    load_onnx_session,
    merge_onnx_outputs,
    pad_onnx_inputs,
    pad_id_batch,
)

from nextrec.utils.types import (
    LossName,
    OptimizerName,
    SchedulerName,
    TrainingModeName,
    TaskTypeName,
    TaskTypeInput,
    MetricsName,
)


class BaseModel(SummarySet, FeatureSet, nn.Module):
    @property
    def model_name(self) -> str:  # type: ignore[override]
        raise NotImplementedError

    @property
    def default_task(self) -> TaskTypeName | list[TaskTypeName]:
        raise NotImplementedError

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        target: list[str] | str | None = None,
        id_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        training_mode: TrainingModeName | list[TrainingModeName] | None = None,
        embedding_l1_reg: float = 0.0,
        dense_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        dense_l2_reg: float = 0.0,
        device: str = "cpu",
        session_id: str | None = None,
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
            target: Target column name. e.g., 'label_ctr' or ['label_ctr', 'label_cvr'].
            id_columns: Identifier column name, only need to specify if GAUC is required. e.g., 'user_id'.
            task: Task types, e.g., 'binary', 'regression', or ['binary', 'regression']. If None, falls back to self.default_task.
            training_mode: Training mode for different tasks. e.g., 'pointwise', ['pointwise', 'pairwise'].

            embedding_l1_reg: L1 regularization strength for embedding params. e.g., 1e-6.
            dense_l1_reg: L1 regularization strength for dense params. e.g., 1e-5.
            embedding_l2_reg: L2 regularization strength for embedding params. e.g., 1e-5.
            dense_l2_reg: L2 regularization strength for dense params. e.g., 1e-4.

            device: Torch device string or torch.device. e.g., 'cpu', 'cuda:0'.
            session_id: Session id for logging. If None, a default id with timestamps will be created. e.g., 'session_tutorial'.

            distributed: Enable DistributedDataParallel flow, set True to enable distributed training.
            rank: Global rank (defaults to env RANK). e.g., 0 for the main process.
            world_size: Number of processes (defaults to env WORLD_SIZE). e.g., 4 for a 4-process training.
            local_rank: Local rank for selecting CUDA device (defaults to env LOCAL_RANK). e.g., 0 for the first GPU.
            ddp_find_unused_parameters: Default False, set it True only when exist unused parameters in ddp model, in most cases should be False.

        Note:
            Optimizer, scheduler, and loss are configured via compile().
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
        self.device = get_device(self.distributed, self.local_rank, device)

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = self.session.root  # pwd/session_id, path for this session
        self.checkpoint_path = os.path.join(
            self.session_path, self.model_name.upper() + "_checkpoint.pt"
        )  # e.g., pwd/session_id/DeepFM_checkpoint.pt
        self.best_path = os.path.join(self.session_path, self.model_name.upper() + "_best.pt")
        self.features_config_path = os.path.join(self.session_path, "features_config.pkl")
        self.set_all_features(dense_features, sparse_features, sequence_features, target, id_columns)

        self.task = cast(TaskTypeName | list[TaskTypeName], task or self.default_task)
        self.nums_task = len(self.task) if isinstance(self.task, list) else 1

        training_mode = training_mode or "pointwise"
        if isinstance(training_mode, list):
            self.training_modes = list(training_mode)
        else:
            self.training_modes = [training_mode] * self.nums_task

        self.embedding_l1_reg = embedding_l1_reg
        self.dense_l1_reg = dense_l1_reg
        self.embedding_l2_reg = embedding_l2_reg
        self.dense_l2_reg = dense_l2_reg
        self.regularization_weights = []
        self.embedding_params = []

        self.ignore_label = None
        self.compiled = False

        self.max_gradient_norm = 1.0
        self.logger_initialized = False
        self.training_logger = None
        self.steps_per_epoch = None
        self.callbacks = CallbackList()

        self.train_data_summary = None
        self.valid_data_summary = None
        self.note = None

    def register_regularization_weights(
        self,
        embedding_attr: str = "embedding",
        exclude_modules: list[str] | None = None,
        include_modules: list[str] | None = None,
    ):
        """
        Register parameters for regularization.
        By default, all nn.Linear weights (excluding those in BatchNorm/Dropout layers) and embedding weights under `embedding_attr` are registered.

        Args:
            embedding_attr: Attribute name of the embedding layer/module.
            exclude_modules: List of module name substrings to exclude from regularization.
            include_modules: List of module name substrings to include for regularization. If provided, only modules containing these substrings are included.
        """
        exclude_modules = exclude_modules or []
        include_modules = include_modules or []
        embedding_layer = getattr(self, embedding_attr, None)
        embed_dict = getattr(embedding_layer, "embed_dict", None)
        if embed_dict is not None:
            embedding_params = [embed.weight for embed in embed_dict.values() if hasattr(embed, "weight")]
        else:
            weight = getattr(embedding_layer, "weight", None)
            embedding_params = [weight] if isinstance(weight, torch.Tensor) else []

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
            if module is self or embedding_attr in name or isinstance(module, skip_types):
                continue
            if include_modules and not any(inc in name for inc in include_modules):
                continue
            if exclude_modules and any(exc in name for exc in exclude_modules):
                continue
            if isinstance(module, nn.Linear):
                if id(module.weight) not in existing_reg_ids:
                    self.regularization_weights.append(module.weight)
                    existing_reg_ids.add(id(module.weight))

    def add_reg_loss(self) -> torch.Tensor:
        """
        Compute the regularization loss based on registered parameters and their respective regularization strengths.
        """
        reg_loss = torch.tensor(0.0, device=self.device)

        if self.embedding_l1_reg > 0:
            reg_loss += self.embedding_l1_reg * sum(param.abs().sum() for param in self.embedding_params)
        if self.embedding_l2_reg > 0:
            reg_loss += self.embedding_l2_reg * sum((param**2).sum() for param in self.embedding_params)

        if self.dense_l1_reg > 0:
            reg_loss += self.dense_l1_reg * sum(param.abs().sum() for param in self.regularization_weights)
        if self.dense_l2_reg > 0:
            reg_loss += self.dense_l2_reg * sum((param**2).sum() for param in self.regularization_weights)
        return reg_loss

    # todo: support build pairwise/listwise label in input
    def get_input(self, input_data: dict, require_labels: bool = True):
        """
        Prepare unified input features and labels from the given input data.

        Args:
            input_data: Input data dictionary containing 'features' and optionally 'labels', e.g., {'features': {'feat1': [...], 'feat2': [...]}, 'labels': {'label': [...]}}.
            require_labels: Whether labels are required in the input data. Default is True: for training and evaluation with labels.

        Note:
            target tensor shape will always be (batch_size, num_targets)
        """
        feature_source = input_data.get("features", {})
        # todo: pairwise/listwise label support
        # "labels": {...} should contain pointwise/pair index/list index/ relevance scores
        # now only have pointwise label support
        label_source = input_data.get("labels")

        X_input = {}
        for feature in self.all_features:
            if feature.name not in feature_source:
                raise KeyError(f"[BaseModel-input Error] Feature '{feature.name}' not found in input data.")
            feature_data = get_column_data(feature_source, feature.name)
            X_input[feature.name] = to_tensor(
                feature_data,
                dtype=(torch.float32 if isinstance(feature, DenseFeature) else torch.long),
                device=self.device,
            )
        y = None
        # if need labels: training or eval with labels
        if len(self.target_columns) > 0 and (
            require_labels or (label_source and any(name in label_source for name in self.target_columns))
        ):
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
                        raise ValueError(f"[BaseModel-input Error] Target column '{target_name}' contains no data.")
                    continue
                target_tensor = to_tensor(target_data, dtype=torch.float32, device=self.device)
                target_tensor = target_tensor.reshape(
                    target_tensor.size(0), -1
                )  # always reshape to (batch_size, num_targets)
                target_tensors.append(target_tensor)
            if target_tensors:
                y = torch.cat(target_tensors, dim=1)
                if y.shape[1] == 1:  # no need to do that again
                    y = y.reshape(-1)
            elif require_labels:
                raise ValueError("[BaseModel-input Error] Labels are required but none were found in the input batch.")
        return X_input, y

    def handle_valid_split(
        self,
        train_data: dict | pd.DataFrame,
        valid_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ):
        """
        This function will split training data into training and validation sets when:
        1. valid_data is None;
        2. valid_split is provided.

        Returns:
            train_loader: DataLoader for training data.
            valid_split_data: Validation data dict/dataframe split from training data.
        """
        if not (0 < valid_split < 1):
            raise ValueError(f"[BaseModel-validation Error] valid_split must be between 0 and 1, got {valid_split}")
        if isinstance(train_data, pd.DataFrame):
            total_length = len(train_data)
        elif isinstance(train_data, dict):
            sample_key = next(iter(train_data))
            total_length = len(train_data[sample_key])
            for k, v in train_data.items():
                if len(v) != total_length:
                    raise ValueError(
                        f"[BaseModel-validation Error] Length of field '{k}' ({len(v)}) != length of field '{sample_key}' ({total_length})"
                    )
        else:
            raise TypeError(
                f"[BaseModel-validation Error] If you want to use valid_split, train_data must be DataFrame or a dict, now got {type(train_data)}"
            )
        rng = np.random.default_rng(42)
        indices = rng.permutation(total_length)
        split_idx = int(total_length * (1 - valid_split))
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]
        if isinstance(train_data, pd.DataFrame):
            train_split_data = train_data.iloc[train_indices].reset_index(drop=True)
            valid_split_data = train_data.iloc[valid_indices].reset_index(drop=True)
        else:
            train_split_data = {k: np.asarray(v)[train_indices] for k, v in train_data.items()}
            valid_split_data = {k: np.asarray(v)[valid_indices] for k, v in train_data.items()}
        train_loader = self.prepare_data_loader(
            train_split_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        logging.info(f"Split data: {len(train_indices)} training samples, {len(valid_indices)} validation samples")
        return train_loader, valid_split_data

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
        loss: LossName | nn.Module | list[LossName | nn.Module] | None = "bce",
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
            loss: Loss function name, instance, or list for multi-task. e.g., 'bce', 'mse', or torch.nn.BCELoss(), you can also use custom loss functions.
            loss_params: Loss function parameters, or list for multi-task. e.g., {'weight': tensor([0.25, 0.75])}.
            loss_weights: Weights for each task loss, int/float for single-task or list for multi-task. e.g., 1.0, or [1.0, 0.5].
                Use {"method": "grad_norm", ...} to enable GradNorm for multi-task loss balancing.
            ignore_label: Label value to ignore when computing loss. Use this to skip gradients for unknown labels.
        """
        self.ignore_label = ignore_label

        # get loss list
        loss_list = get_loss_list(loss, self.training_modes, self.nums_task)

        self.loss_params = loss_params or {}
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}

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

        self.loss_config = loss_list if self.nums_task > 1 else loss_list[0]
        if isinstance(self.loss_params, dict):
            loss_params_list = [self.loss_params] * self.nums_task
        else:
            loss_params_list = [self.loss_params[i] if i < len(self.loss_params) else {} for i in range(self.nums_task)]
        self.loss_fn = [get_loss_fn(loss=loss_list[i], **loss_params_list[i]) for i in range(self.nums_task)]

        # loss weighting (grad norm or fixed weights)
        self.grad_norm = None
        self.grad_norm_shared_params = None
        is_grad_norm = isinstance(loss_weights, dict) and loss_weights.get("method") == "grad_norm"
        if is_grad_norm:
            if self.nums_task == 1:
                raise ValueError("[BaseModel-compile Error] GradNorm requires multi-task setup.")
            grad_norm_params = dict(loss_weights) if isinstance(loss_weights, dict) else {}
            grad_norm_params.pop("method", None)
            self.grad_norm = GradNormLossWeighting(nums_task=self.nums_task, device=self.device, **grad_norm_params)
            self.loss_weights = None
        elif loss_weights is None:
            self.loss_weights = None
        elif self.nums_task == 1:
            if isinstance(loss_weights, (list, tuple)):
                if len(loss_weights) != 1:
                    raise ValueError(
                        "[BaseModel-compile Error] loss_weights list must have exactly one element for single-task setup."
                    )
                loss_weights = loss_weights[0]
            self.loss_weights = [float(loss_weights)]
        elif isinstance(loss_weights, (int, float)):
            self.loss_weights = [float(loss_weights)] * self.nums_task
        elif isinstance(loss_weights, (list, tuple)):
            weights = [float(w) for w in loss_weights]
            if len(weights) != self.nums_task:
                raise ValueError(
                    f"[BaseModel-compile Error] Number of loss_weights ({len(weights)}) must match number of tasks ({self.nums_task})."
                )
            self.loss_weights = weights
        else:
            raise TypeError(
                f"[BaseModel-compile Error] loss_weights must be int, float, list or tuple, got {type(loss_weights)}"
            )
        self.compiled = True

    def compute_loss(self, y_pred, y_true):
        """
        Compute the loss between predictions and ground truth labels, with loss weighting and ignore_label handling
        """
        if y_true is None:
            raise ValueError("[BaseModel-compute_loss Error] Ground truth labels (y_true) are required.")

        # single-task
        if self.nums_task == 1:
            if y_pred.dim() == 1:
                y_pred = y_pred.view(-1, 1)
            if y_true.dim() == 1:
                y_true = y_true.view(-1, 1)

            loss_fn = self.loss_fn[0]

            # mask ignored labels
            # we don't suggest using ignore_label for single task training
            if self.ignore_label is not None:
                valid_mask = y_true != self.ignore_label
                if valid_mask.dim() > 1:
                    valid_mask = valid_mask.all(dim=1)
                if not torch.any(valid_mask):  # if no valid labels, return zero loss
                    return y_pred.sum() * 0.0

                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]

            mode = self.training_modes[0]

            task_dim = self.task_dims[0] if hasattr(self, "task_dims") else y_pred.shape[1]  # type: ignore
            if mode in {"pairwise", "listwise"}:
                loss = compute_ranking_loss(
                    training_mode=mode,
                    loss_fn=loss_fn,
                    y_pred=y_pred,
                    y_true=y_true,
                )
            elif task_dim == 1:
                loss = loss_fn(y_pred.view(-1), y_true.view(-1))
            else:
                loss = loss_fn(y_pred, y_true)
            if self.loss_weights is not None:
                loss *= self.loss_weights[0]
            return loss

        # multi-task: slice predictions and labels per task
        slices = (
            self.prediction_layer.task_slices  # type: ignore
            if hasattr(self, "prediction_layer")
            else [(i, i + 1) for i in range(self.nums_task)]
        )
        task_losses = []
        for i, (start, end) in enumerate(slices):  # type: ignore
            y_pred_i = y_pred[:, start:end]
            y_true_i = y_true[:, start:end]
            # mask ignored labels
            if self.ignore_label is not None:
                valid_mask = y_true_i != self.ignore_label
                if valid_mask.dim() > 1:
                    valid_mask = valid_mask.all(dim=1)
                if not torch.any(valid_mask):
                    task_losses.append(y_pred_i.sum() * 0.0)
                    continue
                y_pred_i = y_pred_i[valid_mask]
                y_true_i = y_true_i[valid_mask]

            mode = self.training_modes[i]

            if mode in {"pairwise", "listwise"}:
                task_loss = compute_ranking_loss(
                    training_mode=mode,
                    loss_fn=self.loss_fn[i],
                    y_pred=y_pred_i,
                    y_true=y_true_i,
                )
            else:
                task_loss = self.loss_fn[i](y_pred_i, y_true_i)
                # task_loss = normalize_task_loss(
                #     task_loss, valid_count, total_count
                # )  # normalize by valid samples to avoid loss scale issues
            task_losses.append(task_loss)

        if self.grad_norm is not None:
            if self.grad_norm_shared_params is None:
                self.grad_norm_shared_params = get_grad_norm_shared_params(
                    self, getattr(self, "grad_norm_shared_modules", None)
                )
            return self.grad_norm.compute_weighted_loss(task_losses, self.grad_norm_shared_params)
        if isinstance(self.loss_weights, (list, tuple)):
            task_losses = [task_loss * self.loss_weights[i] for i, task_loss in enumerate(task_losses)]
        return torch.stack(task_losses).sum()

    @overload
    def prepare_data_loader(
        self,
        data: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        sampler: Any = None,
        return_dataset: Literal[False] = False,
    ) -> DataLoader: ...

    @overload
    def prepare_data_loader(
        self,
        data: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        sampler: Any = None,
        return_dataset: Literal[True] = True,
    ) -> tuple[DataLoader, TensorDictDataset | None]: ...

    def prepare_data_loader(
        self,
        data: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        sampler: Any = None,
        return_dataset: bool = False,
    ) -> DataLoader | tuple[DataLoader, TensorDictDataset | None]:
        """
        Prepare a DataLoader from input data. Only used when input data is not a DataLoader.

        Args:
            data: Input data (dict/df/DataLoader).
            batch_size: Batch size.
            shuffle: Whether to shuffle the data (ignored when a sampler is provided).
            num_workers: Number of DataLoader workers.
            prefetch_factor: Number of batches loaded in advance by each worker.
            sampler: Optional sampler for DataLoader.
            return_dataset: Whether to return the tensor dataset along with the DataLoader, used for valid data
        Returns:
            DataLoader (and tensor dataset if return_dataset is True).
        """
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
            raise ValueError("[BaseModel-prepare_data_loader Error] No data available to create DataLoader.")
        dataset = TensorDictDataset(tensors)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        return (loader, dataset) if return_dataset else loader

    def fit(
        self,
        train_data: dict | pd.DataFrame | DataLoader | None = None,
        valid_data: dict | pd.DataFrame | DataLoader | None = None,
        metrics: (
            list[MetricsName] | dict[str, list[MetricsName]] | None
        ) = None,  # ['auc', 'logloss'] or {'target1': ['auc', 'logloss'], 'target2': ['mse']}
        epochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 32,
        user_id_column: str | None = None,
        valid_split: float | None = None,
        early_stop_patience: int = 20,
        early_stop_monitor_task: str | None = None,
        num_workers: int = 0,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        use_swanlab: bool = False,
        wandb_api: str | None = None,
        swanlab_api: str | None = None,
        wandb_kwargs: dict | None = None,
        swanlab_kwargs: dict | None = None,
        auto_ddp_sampler: bool = True,
        log_interval: int = 1,
        note: str | None = None,
        summary_sections: list[Literal["feature", "model", "train", "data"]] | None = None,
    ):
        """
        Train the model.

        Args:
            train_data: Training data (dict/df/DataLoader). If distributed, each rank uses its own sampler/batches.
            valid_data: Optional validation data; if None and valid_split is set, a split is created.
            metrics: Metrics names or per-target dict. e.g. {'target1': ['auc', 'logloss'], 'target2': ['mse']} or ['auc', 'logloss'].
            epochs: Training epochs.
            shuffle: Whether to shuffle training data (ignored when a sampler enforces order).
            batch_size: Batch size (per process when distributed).
            user_id_column: Column name for GAUC-style metrics;.
            valid_split: Ratio to split training data when valid_data is None. e.g., 0.1 for 10% validation.

            early_stop_patience: Epochs for early stopping. 0 to disable. e.g., 20.
            early_stop_monitor_task: Task name to monitor for early stopping in multi-task scenario. If None, uses first target. e.g., 'click'.
            num_workers: DataLoader worker count.

            use_tensorboard: Enable tensorboard logging.
            use_wandb: Enable Weights & Biases logging.
            use_swanlab: Enable SwanLab logging.
            wandb_api: W&B API key for non-tty login.
            swanlab_api: SwanLab API key for non-tty login.
            wandb_kwargs: Optional kwargs for wandb.init(...).
            swanlab_kwargs: Optional kwargs for swanlab.init(...).
            auto_ddp_sampler: Attach DistributedSampler automatically when distributed, set False to when data is already sharded per rank.
            log_interval: Log validation metrics every N epochs (still computes metrics each epoch).
            note: Optional note for the training run.
            summary_sections: Optional summary sections to print. Choose from
                ["feature", "model", "train", "data"]. Defaults to all.

        Notes:
            - Distributed training uses DDP; init occurs via env vars (RANK/WORLD_SIZE/LOCAL_RANK).
            - All ranks must call evaluate() together because it performs collective ops.
        """
        device_id = self.local_rank if self.device.type == "cuda" else None
        init_process_group(self.distributed, self.rank, self.world_size, device_id=device_id)
        self.to(self.device)

        assert_task(self.task, len(self.target_columns), model_name=self.model_name)

        if not self.compiled:
            self.compile(
                optimizer="adam",
                optimizer_params={},
                scheduler=None,
                scheduler_params={},
                loss=None,
                loss_params={},
            )

        if self.distributed and dist.is_available() and dist.is_initialized() and self.ddp_model is None:
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

        if not self.logger_initialized and self.is_main_process:  # only main process initializes logger
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True
        self.metrics, self.task_specific_metrics, self.best_metrics_mode = configure_metrics(
            task=self.task, metrics=metrics, target_names=self.target_columns
        )  # ['auc', 'logloss'], {'target1': ['auc', 'logloss'], 'target2': ['mse']}, 'max'

        self.early_stop_patience = early_stop_patience
        self.early_stop_monitor_task = early_stop_monitor_task
        self.note = note

        training_config = {}
        if self.is_main_process:
            training_config = {
                "model_name": getattr(self, "model_name", self.__class__.__name__),
                "note": self.note,
                "task": self.task,
                "target_columns": self.target_columns,
                "batch_size": batch_size,
                "epochs": epochs,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "valid_split": valid_split,
                "optimizer": getattr(self, "optimizer_name", None),
                "optimizer_params": getattr(self, "optimizer_params", None),
                "scheduler": getattr(self, "scheduler_name", None),
                "scheduler_params": getattr(self, "scheduler_params", None),
                "loss": getattr(self, "loss_config", None),
                "loss_weights": getattr(self, "loss_weights", None),
                "early_stop_patience": self.early_stop_patience,
                "max_gradient_norm": self.max_gradient_norm,
                "embedding_l1_reg": self.embedding_l1_reg,
                "embedding_l2_reg": self.embedding_l2_reg,
                "dense_l1_reg": self.dense_l1_reg,
                "dense_l2_reg": self.dense_l2_reg,
                "session_id": self.session_id,
                "distributed": self.distributed,
                "device": str(self.device),
                "dense_feature_count": len(self.dense_features),
                "sparse_feature_count": len(self.sparse_features),
                "sequence_feature_count": len(self.sequence_features),
            }
            training_config = safe_value(training_config)

        if self.is_main_process:
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

        self.training_logger = (
            TrainingLogger(
                session=self.session,
                use_tensorboard=use_tensorboard,
                use_wandb=use_wandb,
                use_swanlab=use_swanlab,
                config=training_config,
                wandb_kwargs=wandb_kwargs,
                swanlab_kwargs=swanlab_kwargs,
            )
            if self.is_main_process
            else None
        )

        # Setup default callbacks if missing
        if self.nums_task == 1:
            monitor_metric = f"val_{self.metrics[0]}"
        else:
            # Determine which task to monitor for early stopping
            monitor_task = self.early_stop_monitor_task
            if monitor_task is None:
                monitor_task = self.target_columns[0]
            elif monitor_task not in self.target_columns:
                raise ValueError(
                    f"[BaseModel-fit Error] early_stop_monitor_task '{monitor_task}' not found in target_columns {self.target_columns}."
                )
            monitor_metric = f"val_{self.metrics[0]}_{monitor_task}"

        existing_callbacks = self.callbacks.callbacks

        has_validation = valid_data is not None or valid_split is not None
        checkpoint_monitor = monitor_metric
        checkpoint_mode = self.best_metrics_mode
        if not has_validation:
            checkpoint_monitor = "loss"
            checkpoint_mode = "min"

        if self.early_stop_patience > 0 and not any(isinstance(cb, EarlyStopper) for cb in existing_callbacks):
            self.callbacks.append(
                EarlyStopper(
                    monitor=monitor_metric,
                    patience=self.early_stop_patience,
                    mode=self.best_metrics_mode,
                    restore_best_weights=not self.distributed,
                    verbose=1 if self.is_main_process else 0,
                )
            )

        if self.is_main_process and not any(isinstance(cb, CheckpointSaver) for cb in existing_callbacks):
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
                    verbose=1 if self.is_main_process else 0,
                )
            )

        self.callbacks.set_model(self)
        self.callbacks.set_params({"epochs": epochs, "batch_size": batch_size, "metrics": self.metrics})
        self.best_metric = float("-inf") if self.best_metrics_mode == "max" else float("inf")

        self.needs_user_ids = check_user_id(
            self.metrics, self.task_specific_metrics
        )  # check user_id needed for GAUC metrics
        self.epoch_index = 0
        self.stop_training = False
        self.best_checkpoint_path = self.best_path
        use_ddp_sampler = auto_ddp_sampler and self.distributed and dist.is_available() and dist.is_initialized()

        if not auto_ddp_sampler and self.distributed and self.is_main_process:
            logging.info(
                colorize(
                    "[Distributed Info] auto_ddp_sampler=False; assuming data is already sharded per rank.",
                    color="yellow",
                )
            )

        train_sampler = None
        if valid_split is not None and valid_data is None:
            train_loader, valid_data = self.handle_valid_split(train_data=train_data, valid_split=valid_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # type: ignore
            if use_ddp_sampler:
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
                if use_ddp_sampler:
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
                else:
                    train_loader = train_data
            else:
                loader, dataset = self.prepare_data_loader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    return_dataset=True,
                )
                if use_ddp_sampler and dataset is not None:
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
        if self.distributed and auto_ddp_sampler and isinstance(train_loader, DataLoader) and train_sampler is None:
            raise NotImplementedError(
                "[BaseModel-fit Error] auto_ddp_sampler with pre-defined DataLoader is not supported yet."
            )
        valid_loader, valid_user_ids = self.prepare_validation_data(
            valid_data=valid_data,
            batch_size=batch_size,
            needs_user_ids=self.needs_user_ids,
            user_id_column=user_id_column,
            num_workers=num_workers,
            auto_ddp_sampler=auto_ddp_sampler,
        )
        try:
            self.steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:  # streaming data loader does not supported len()
            self.steps_per_epoch = None
            is_streaming = True
        self.collect_train_metrics = not is_streaming
        if is_streaming and self.is_main_process:
            logging.info(
                colorize(
                    "[Training Info] Streaming mode detected; training metrics collection is disabled to avoid memory growth.",
                    color="yellow",
                )
            )

        if self.is_main_process:
            self.train_data_summary = None if is_streaming else self.build_train_data_summary(train_data, train_loader)
            self.valid_data_summary = (
                None if valid_loader is None else self.build_valid_data_summary(valid_data, valid_loader)
            )
            self.summary(summary_sections)
            logging.info("")
            tb_dir = (
                self.training_logger.tensorboard_logdir
                if self.training_logger and self.training_logger.use_tensorboard
                else None
            )
            if tb_dir:
                user = getpass.getuser()
                host = socket.gethostname()
                tb_cmd = f"tensorboard --logdir {tb_dir} --port 6006"
                ssh_hint = f"ssh -L 6006:localhost:6006 {user}@{host}"
                logging.info(colorize(f"TensorBoard logs saved to: {tb_dir}", color="cyan"))
                logging.info(colorize("To view logs, run:", color="cyan"))
                logging.info(colorize(f"    {tb_cmd}", color="cyan"))
                logging.info(colorize("Then SSH port forward:", color="cyan"))
                logging.info(colorize(f"    {ssh_hint}", color="cyan"))

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

            if not isinstance(train_loader, DataLoader):
                raise TypeError(f"Expected DataLoader for training, got {type(train_loader)}")
            train_result = self.train_epoch(train_loader, is_streaming=is_streaming)
            if isinstance(train_result, tuple):  # [avg_loss, metrics_dict], e.g., (0.5, {'auc': 0.75, 'logloss': 0.45})
                train_loss, train_metrics = train_result
            else:
                train_loss = train_result
                train_metrics = None

            logging.info("")
            train_log_payload = {
                "loss": (float(np.sum(train_loss)) if isinstance(train_loss, np.ndarray) else float(train_loss))
            }
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
                is_main_process=self.is_main_process,
                colorize=lambda s: colorize(s),
            )
            if self.training_logger:
                self.training_logger.log_metrics(train_log_payload, step=epoch + 1, split="train")
            if valid_loader is not None:
                should_eval_valid = (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs
                val_metrics = None
                if should_eval_valid:
                    self.callbacks.on_validation_begin()
                    val_metrics = self.evaluate(
                        valid_loader,
                        user_ids=valid_user_ids if self.needs_user_ids else None,
                        num_workers=num_workers,
                    )
                    display_metrics_table(
                        epoch=epoch + 1,
                        epochs=epochs,
                        split="Valid",
                        loss=None,
                        metrics=val_metrics,
                        target_names=self.target_columns,
                        base_metrics=(self.metrics if isinstance(self.metrics, list) else None),
                        is_main_process=self.is_main_process,
                        colorize=lambda s: colorize("  " + s, color="cyan"),
                    )
                    self.callbacks.on_validation_end()
                    if val_metrics and self.training_logger:
                        self.training_logger.log_metrics(val_metrics, step=epoch + 1, split="valid")
                if not val_metrics:
                    if should_eval_valid and self.is_main_process:
                        logging.info(
                            colorize(
                                "Warning: No validation metrics computed. Skipping validation for this epoch.",
                                color="yellow",
                            )
                        )
                    epoch_logs = {**train_log_payload}
                else:
                    epoch_logs = {**train_log_payload}
                    for k, v in val_metrics.items():
                        epoch_logs[f"val_{k}"] = v
            else:
                epoch_logs = {**train_log_payload}

            # Call on_epoch_end for all callbacks (handles early stopping, checkpointing, lr scheduling)
            self.callbacks.on_epoch_end(epoch, epoch_logs)

            # Broadcast stop_training flag to all processes
            if self.distributed and dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor([int(self.stop_training)], device=self.device)
                dist.broadcast(stop_tensor, src=0)
                self.stop_training = bool(stop_tensor.item())

            if self.stop_training:
                break
        # Call on_train_end for all callbacks
        self.callbacks.on_train_end()

        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()  # dist.barrier() will wait for all processes, like async all_reduce()
        if self.is_main_process:
            logging.info("")
            logging.info(colorize("Training finished.", color="bright_blue", bold=True))
            logging.info("")
        if valid_loader is not None:
            if self.is_main_process:
                logging.info(format_kv("Load best model from", self.best_checkpoint_path))
            if os.path.exists(self.best_checkpoint_path):
                self.load_model(self.best_checkpoint_path, map_location=self.device, verbose=False)
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

    def train_epoch(self, train_loader: DataLoader, is_streaming: bool = False):
        # use ddp model for distributed training
        model = self.ddp_model if hasattr(self, "ddp_model") and self.ddp_model is not None else self
        accumulated_loss = 0.0
        model.train()  # type: ignore
        num_batches = 0
        y_true_list = []
        y_pred_list = []
        collect_metrics = getattr(self, "collect_train_metrics", True)

        user_ids_list = [] if self.needs_user_ids else None
        tqdm_disable = not self.is_main_process
        if self.steps_per_epoch is not None:
            batch_iter = enumerate(
                progress(
                    train_loader,
                    description=f"Epoch {self.epoch_index + 1}",
                    total=self.steps_per_epoch,
                    disable=tqdm_disable,
                )
            )
        else:
            desc = "Batches" if is_streaming else f"Epoch {self.epoch_index + 1}"
            batch_iter = enumerate(
                progress(
                    train_loader,
                    description=desc,
                    disable=tqdm_disable,
                )
            )
        for batch_index, batch_data in batch_iter:
            batch_dict = batch_to_dict(batch_data)
            X_input, y_true = self.get_input(batch_dict, require_labels=True)
            # call via __call__ so DDP hooks run
            y_pred = model(X_input)  # type: ignore

            loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()
            total_loss = loss + reg_loss
            self.optimizer_fn.zero_grad()
            total_loss.backward()

            params = model.parameters() if self.ddp_model is not None else self.parameters()  # type: ignore # ddp model parameters or self parameters
            if self.max_gradient_norm is not None:
                nn.utils.clip_grad_norm_(params, self.max_gradient_norm)
            self.optimizer_fn.step()
            if self.grad_norm is not None:
                # Synchronize GradNorm buffers across DDP ranks before stepping
                if self.distributed and dist.is_available() and dist.is_initialized():
                    self.grad_norm.sync()
                self.grad_norm.step()
            accumulated_loss += loss.item()

            if collect_metrics and y_true is not None and isinstance(y_pred, torch.Tensor):
                y_true_list.append(y_true.detach().cpu().numpy())
                y_pred_list.append(y_pred.detach().cpu().numpy())
                if self.needs_user_ids and user_ids_list is not None:
                    batch_user_id = get_user_ids(data=batch_dict, id_columns=self.id_columns)
                    if batch_user_id is not None:
                        user_ids_list.append(batch_user_id)
            num_batches += 1
        if self.distributed and dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor([accumulated_loss, num_batches], device=self.device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            accumulated_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        avg_loss = accumulated_loss / max(num_batches, 1)

        y_true_all_local = np.concatenate(y_true_list, axis=0) if y_true_list else None
        y_pred_all_local = np.concatenate(y_pred_list, axis=0) if y_pred_list else None
        combined_user_ids_local = (
            np.concatenate(user_ids_list, axis=0) if self.needs_user_ids and user_ids_list else None
        )

        # gather across ranks even when local is empty to avoid DDP hang
        y_true_all = gather_numpy(self, y_true_all_local)
        y_pred_all = gather_numpy(self, y_pred_all_local)
        combined_user_ids = gather_numpy(self, combined_user_ids_local) if self.needs_user_ids else None

        if y_true_all is not None and y_pred_all is not None and len(y_true_all) > 0 and len(y_pred_all) > 0:
            metrics_dict = evaluate_metrics(
                y_true=y_true_all,
                y_pred=y_pred_all,
                metrics=self.metrics,
                task=self.task,
                target_names=self.target_columns,
                task_specific_metrics=self.task_specific_metrics,
                user_ids=combined_user_ids,
                ignore_label=self.ignore_label,
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
        auto_ddp_sampler: bool = True,
    ):
        if valid_data is None:
            return None, None
        if isinstance(valid_data, DataLoader):
            if auto_ddp_sampler and self.distributed:
                raise NotImplementedError(
                    "[BaseModel-prepare_validation_data Error] auto_ddp_sampler with pre-defined DataLoader is not supported yet."
                )
                # valid_loader, _ = add_distributed_sampler(valid_data, distributed=self.distributed, world_size=self.world_size, rank=self.rank, shuffle=False, drop_last=False, default_batch_size=batch_size, is_main_process=self.is_main_process)
            else:
                valid_loader = valid_data
            return valid_loader, None
        valid_sampler = None
        valid_loader, valid_dataset = self.prepare_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, return_dataset=True)  # type: ignore
        if (
            auto_ddp_sampler
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
                pin_memory=self.device.type == "cuda",
                persistent_workers=num_workers > 0,
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
                valid_user_ids = get_user_ids(data=valid_data, id_columns=user_id_column)
        return valid_loader, valid_user_ids

    def evaluate(
        self,
        data: dict | pd.DataFrame | DataLoader,
        metrics: list[MetricsName] | dict[str, list[MetricsName]] | None = None,
        batch_size: int = 32,
        user_ids: np.ndarray | None = None,
        user_id_column: str = "user_id",
        num_workers: int = 0,
        thresholds: float | dict[str, float] | list[float] | None = None,
        show_data_summary: bool = False,
        show_confusion_matrix: bool = False,
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
            thresholds: Threshold(s) for binary metrics/confusion matrix. Supports a single
                float for all targets, a list aligned to target order, or a dict keyed by
                target name. Defaults to 0.5. e.g. 0.5, [0.5, 0.7], {'target1': 0.5, 'target2': 0.7}
            show_data_summary: If True, log data summary statistics.
            show_confusion_matrix: If True, render confusion matrix blocks to logs.
        """
        model = self.ddp_model if self.ddp_model is not None else self
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
            data_loader = self.prepare_data_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
                    batch_user_id = get_user_ids(data=batch_dict, id_columns=self.id_columns)
                    if batch_user_id is not None:
                        collected_user_ids.append(batch_user_id)

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
        final_user_ids = gather_numpy(self, final_user_ids_local) if needs_user_ids else None
        if y_true_all is None or y_pred_all is None or len(y_true_all) == 0 or len(y_pred_all) == 0:
            if self.is_main_process:
                logging.info(
                    colorize(
                        "  Warning: Not enough evaluation data to compute metrics after gathering",
                        color="yellow",
                    )
                )
            return {}

        logging.info("")
        metrics_dict = evaluate_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            metrics=metrics_to_use,
            task=self.task,
            target_names=self.target_columns,
            task_specific_metrics=self.task_specific_metrics,
            user_ids=final_user_ids,
            ignore_label=self.ignore_label,
            thresholds=thresholds,
        )

        if show_data_summary and self.is_main_process:
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

        if (show_data_summary or show_confusion_matrix) and self.is_main_process:
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
                    is_main_process=True,
                    colorize=lambda s: colorize(s),
                )
            else:
                logging.info(
                    colorize(
                        "[BaseModel-evaluate Warning] No metrics computed.",
                        color="yellow",
                    )
                )

        if show_confusion_matrix:
            task = self.task
            target_names = self.target_columns
            if isinstance(task, list):
                task_types = task
            else:
                task_types = [task] * max(1, len(target_names))
            thresholds_by_target = resolve_thresholds(thresholds, target_names)

            if show_confusion_matrix and self.is_main_process:
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
                        if self.is_main_process:
                            logging.info(
                                colorize(
                                    f"{target_names[0]} (threshold={thresholds_by_target.get(target_names[0], 0.5)})",
                                    color="cyan",
                                    bold=True,
                                )
                            )
                            rendered = render_confusion_block(
                                tn=cm["tn"],
                                fp=cm["fp"],
                                fn=cm["fn"],
                                tp=cm["tp"],
                            )
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
                        if self.is_main_process:
                            logging.info("")
                            logging.info(
                                colorize(
                                    f"{target_name} (threshold={thresholds_by_target.get(target_name, 0.5)})",
                                    color="cyan",
                                    bold=True,
                                )
                            )
                            rendered = render_confusion_block(
                                tn=cm["tn"],
                                fp=cm["fp"],
                                fn=cm["fn"],
                                tp=cm["tp"],
                            )
                            for line in rendered.splitlines():
                                logging.info(line)
        return metrics_dict

    @overload
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        return_dataframe: Literal[True] = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        profiler: StageTimer | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: None = None,
        save_format: str = "csv",
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        profiler: StageTimer | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        *,
        save_path: str | os.PathLike,
        save_format: str = "csv",
        return_dataframe: Literal[False] = False,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        profiler: StageTimer | None = None,
    ) -> Path: ...

    def predict(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        return_dataframe: bool = True,
        stream_chunk_size: int = 10000,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        profiler: StageTimer | None = None,
    ) -> pd.DataFrame | np.ndarray | Path:
        """
        Make predictions on the given data.

        Args:
            data: Input data for prediction (file path, dict, DataFrame, or DataLoader).
            batch_size: Batch size for prediction (per process when distributed).
            save_path: Optional path to save predictions; if None, predictions are not saved to disk.
            save_format: Format to save predictions ('csv' or 'parquet').
            return_dataframe: Whether to return predictions as a pandas DataFrame; if False, returns a NumPy array.
            stream_chunk_size: Number of rows per chunk when using streaming mode for large datasets.
            num_workers: DataLoader worker count.
            prefetch_factor: Number of batches prefetched per worker (only when num_workers > 0).
            num_processes: Number of inference processes for streaming file inference.
            processor: Optional DataProcessor for transforming input data.
            profiler: Optional StageTimer for profiling pipeline stages.

        Note:
            predict does not support distributed mode currently; streaming file inference can use
            multiple processes via num_processes > 1, which may change output order.
        """
        self.eval()

        # streaming mode prediction
        if save_path is not None and not return_dataframe:
            if num_processes > 1 and not isinstance(data, (str, os.PathLike)):
                raise ValueError("[BaseModel-predict Error] Multi-process streaming requires data to be a file path.")
            if num_processes > 1 and num_workers != 0:
                logging.info("[BaseModel-predict-streaming Info] Multi-process streaming enforces num_workers=0.")
                logging.info("")
                num_workers = 0
            return self.predict_streaming(
                data=data,
                batch_size=batch_size,
                save_path=save_path,
                save_format=save_format,
                stream_chunk_size=stream_chunk_size,
                return_dataframe=return_dataframe,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                num_processes=num_processes,
                processor=processor,
                profiler=profiler,
            )

        predict_id_columns = list(self.id_columns)
        include_ids = bool(predict_id_columns)
        if isinstance(data, DataLoader):
            data_loader = data
        elif isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=self.target_columns,
                id_columns=predict_id_columns,
                processor=processor,
            )
            data_loader = rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
                num_workers=0,
                prefetch_factor=prefetch_factor,
                profiler=profiler,
            )
        else:
            data_loader = self.prepare_data_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        y_pred_list = []
        id_buffers = {name: [] for name in predict_id_columns} if include_ids else {}

        with torch.no_grad():
            for batch_data in progress(data_loader, description="Predicting"):
                batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
                X_input, _ = self.get_input(batch_dict, require_labels=False)
                start = time.perf_counter()
                y_pred = self(X_input)
                if profiler is not None:
                    profiler.add("inference", time.perf_counter() - start)
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
                        id_buffers[id_name].append(id_np.reshape(id_np.shape[0], -1) if id_np.ndim == 1 else id_np)
        y_pred_all = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.array([])
        id_arrays = None

        if y_pred_all.ndim == 1:
            y_pred_all = y_pred_all.reshape(-1, 1)
        if y_pred_all.size == 0:
            num_outputs = len(self.target_columns) if self.target_columns else 1
            y_pred_all = y_pred_all.reshape(0, num_outputs)
        num_outputs = y_pred_all.shape[1]
        pred_columns: list[str] = list(self.target_columns[:num_outputs]) if self.target_columns else []
        while len(pred_columns) < num_outputs:
            pred_columns.append(f"pred_{len(pred_columns)}")
        if include_ids and predict_id_columns:
            id_arrays = {
                id_name: (
                    np.concatenate([p.reshape(p.shape[0], -1) for p in pieces], axis=0).reshape(-1)
                    if pieces
                    else np.array([], dtype=np.int64)
                )
                for id_name, pieces in id_buffers.items()
            }
            if return_dataframe:
                id_df = pd.DataFrame(id_arrays)
                pred_df = pd.DataFrame(y_pred_all, columns=pred_columns)
                output = pd.concat([id_df, pred_df], axis=1)
            else:
                output = y_pred_all
        else:
            output = pd.DataFrame(y_pred_all, columns=pred_columns) if return_dataframe else y_pred_all
        if save_path is not None:
            if save_format not in {"csv", "parquet"}:
                raise ValueError(f"Unsupported save format: {save_format}. " "Supported: csv, parquet")
            target_path = get_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=f".{save_format}",
                add_timestamp=False,
            )
            if isinstance(output, pd.DataFrame):
                df_to_save = output
            else:
                df_to_save = pd.DataFrame(y_pred_all, columns=pred_columns)
                if include_ids and predict_id_columns and id_arrays is not None:
                    id_df = pd.DataFrame(id_arrays)
                    df_to_save = pd.concat([id_df, df_to_save], axis=1)

            # Save based on format
            if save_format == "csv":
                start = time.perf_counter()
                df_to_save.to_csv(target_path, index=False)
                if profiler is not None:
                    profiler.add("write_output", time.perf_counter() - start)
            elif save_format == "parquet":
                start = time.perf_counter()
                df_to_save.to_parquet(target_path, index=False)
                if profiler is not None:
                    profiler.add("write_output", time.perf_counter() - start)

            logging.info("")
            logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        return output

    def predict_streaming(
        self,
        data: str | os.PathLike | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: str,
        stream_chunk_size: int,
        return_dataframe: bool,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
    ):
        """
        Make predictions on the given data using streaming mode for large datasets.

        Args:
            data: Input data for prediction (file path, dict, DataFrame, or DataLoader).
            batch_size: Batch size for prediction.
            save_path: Path to save predictions.
            save_format: Format to save predictions ('csv' or 'parquet').
            stream_chunk_size: Number of rows per chunk when using streaming mode.
            return_dataframe: Whether to return predictions as a pandas DataFrame.
            num_workers: DataLoader worker count.
            prefetch_factor: Number of batches prefetched per worker (only when num_workers > 0).
            num_processes: Number of inference processes for streaming file inference.
            processor: Optional DataProcessor for transforming input data.
            shard_rank: Process shard rank for multi-process inference.
            shard_count: Total number of shards for multi-process inference.
        """
        predict_id_columns = self.id_columns
        include_ids = bool(predict_id_columns)
        if save_format not in {"csv", "parquet"}:
            raise ValueError(f"Unsupported save format: {save_format}. Supported: csv, parquet")

        # Multi-process streaming
        if num_processes > 1:
            if num_workers > 0:
                logging.info("[BaseModel-predict-streaming Info] Multi-process streaming enforces num_workers=0.")
            num_workers = 0
            prefetch_factor = None
            target_path = Path(
                get_save_path(
                    path=save_path,
                    default_dir=self.session.predictions_dir,
                    default_name="predictions",
                    suffix=f".{save_format}",
                    add_timestamp=False,
                )
            )
            parts_dir = target_path.parent / f".{target_path.stem}_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_paths = [
                parts_dir / f"{target_path.stem}.part{rank}{target_path.suffix}" for rank in range(num_processes)
            ]
            profile_paths = (
                [parts_dir / f"{target_path.stem}.profile{rank}.json" for rank in range(num_processes)]
                if profiler is not None
                else None
            )
            error_paths = [parts_dir / f"{target_path.stem}.error{rank}.txt" for rank in range(num_processes)]

            ctx = mp.get_context("spawn")
            processes = []
            for rank in range(num_processes):
                process = ctx.Process(
                    target=predict_streaming_worker,
                    args=(
                        self,
                        data,
                        batch_size,
                        part_paths[rank],
                        save_format,
                        stream_chunk_size,
                        num_workers,
                        prefetch_factor,
                        processor,
                        rank,
                        num_processes,
                        profile_paths[rank] if profile_paths else None,
                        error_paths[rank],
                    ),
                )
                process.start()
                processes.append(process)

            for process in progress(iter(processes), description="Predicting...", total=None):
                process.join()

            failed_processes = [process for process in processes if process.exitcode not in (0, None)]
            if failed_processes:
                errors: list[str] = []
                for error_path in error_paths:
                    if not error_path.exists():
                        continue
                    text = error_path.read_text(encoding="utf-8").strip()
                    if text and text not in errors:
                        errors.append(text)
                if not errors:
                    errors = [
                        f"[PredictWorker Error] pid={process.pid}, exitcode={process.exitcode}, name={process.name}"
                        for process in failed_processes
                    ]
                error_text = "\n\n".join(errors)
                raise RuntimeError(
                    "[BaseModel-predict-streaming Error] One or more inference processes failed.\n" + error_text
                )

            existing_parts = [p for p in part_paths if p.exists()]
            if existing_parts:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if save_format == "csv":
                    start = time.perf_counter()
                    lazy_frames = [pl.scan_csv(p) for p in existing_parts]
                    pl.concat(lazy_frames).sink_csv(target_path)
                    if profiler is not None:
                        profiler.add("merge_parts", time.perf_counter() - start)
                elif save_format == "parquet":
                    start = time.perf_counter()
                    lazy_frames = [pl.scan_parquet(p) for p in existing_parts]
                    pl.concat(lazy_frames).sink_parquet(target_path)
                    if profiler is not None:
                        profiler.add("merge_parts", time.perf_counter() - start)

            for part_path in part_paths:
                if part_path.exists():
                    part_path.unlink()
            if profile_paths and profiler is not None:
                for profile_path in profile_paths:
                    if profile_path.exists():
                        profiler.merge(StageTimer.load(profile_path))
                        profile_path.unlink()
            for error_path in error_paths:
                if error_path.exists():
                    error_path.unlink()
            if parts_dir.exists() and not any(parts_dir.iterdir()):
                parts_dir.rmdir()

            logging.info("")
            logging.info(
                colorize(
                    f"Predictions saved to: {target_path} (merged from {num_processes} parts)",
                    color="green",
                )
            )
            return target_path
        # Single-process streaming
        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=self.target_columns,
                id_columns=predict_id_columns,
                processor=processor,
            )
            data_loader = rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
                num_workers=num_workers,
                prefetch_factor=None if num_workers == 0 else prefetch_factor,
                shard_rank=shard_rank,
                shard_count=shard_count,
                profiler=profiler,
            )
        elif isinstance(data, DataLoader):
            data_loader = data
        else:
            data_loader = self.prepare_data_loader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.predictions_dir,
            default_name="predictions",
            suffix=f".{save_format}",
            add_timestamp=False,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        header_written = target_path.exists()
        parquet_writer = None

        pred_columns = None
        cached_frames = []  # used when return_dataframe=True

        disable_progress = shard_count > 1
        with torch.no_grad():
            for batch_data in progress(data_loader, description="Predicting", disable=disable_progress):
                batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
                X_input, _ = self.get_input(batch_dict, require_labels=False)
                start = time.perf_counter()
                y_pred = self.forward(X_input)
                if profiler is not None:
                    profiler.add("inference", time.perf_counter() - start)
                if y_pred is None or not isinstance(y_pred, torch.Tensor):
                    continue
                y_pred_np = y_pred.detach().cpu().numpy()
                if y_pred_np.ndim == 1:
                    y_pred_np = y_pred_np.reshape(-1, 1)
                if pred_columns is None:
                    num_outputs = y_pred_np.shape[1]
                    pred_columns = list(self.target_columns[:num_outputs]) if self.target_columns else []
                    while len(pred_columns) < num_outputs:
                        pred_columns.append(f"pred_{len(pred_columns)}")

                ids = batch_dict.get("ids") if include_ids and predict_id_columns else None
                id_arrays_batch = {
                    id_name: (
                        ids[id_name].detach().cpu().numpy()
                        if isinstance(ids[id_name], torch.Tensor)
                        else np.asarray(ids[id_name])
                    ).reshape(-1)
                    for id_name in (predict_id_columns or [])
                    if ids and id_name in ids
                }

                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if id_arrays_batch:
                    id_df = pd.DataFrame(id_arrays_batch)
                    df_batch = pd.concat([id_df, df_batch], axis=1)

                # Streaming save based on format
                if save_format == "csv":
                    start = time.perf_counter()
                    df_batch.to_csv(target_path, mode="a", header=not header_written, index=False)
                    if profiler is not None:
                        profiler.add("write_output", time.perf_counter() - start)
                    header_written = True
                    if return_dataframe:
                        cached_frames.append(df_batch)
                elif save_format == "parquet":
                    start = time.perf_counter()
                    table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(target_path, table.schema)
                    parquet_writer.write_table(table)
                    if profiler is not None:
                        profiler.add("write_output", time.perf_counter() - start)
                    if return_dataframe:
                        cached_frames.append(df_batch)

        # Close writers
        if parquet_writer is not None:
            parquet_writer.close()

        logging.info("")
        logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        if return_dataframe:
            return (
                pd.concat(cached_frames, ignore_index=True)
                if cached_frames
                else pd.DataFrame(columns=pred_columns or [])
            )
        # Return the actual save path when not returning dataframe
        return target_path

    def prepare_onnx_dataloader(
        self,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        num_workers: int,
        id_columns: list[str] | None = None,
        processor: Any | None = None,
        shard_rank: int = 0,
        shard_count: int = 1,
    ) -> DataLoader:
        """
        Prepare a DataLoader for ONNX prediction.

        Args:
            data: Input data (file path, dict, DataFrame, or DataLoader).
            batch_size: Effective batch size. When `data` is a file path, this is
                treated as the streaming chunk size.
            num_workers: Number of DataLoader workers.

        """
        if isinstance(data, DataLoader):
            return data
        # if data is a file path, use streaming DataLoader
        # will set batch_size=1 cause each batch is a file chunk
        if isinstance(data, (str, os.PathLike)):
            effective_processor = processor
            rec_loader = RecDataLoader(
                dense_features=self.dense_features,
                sparse_features=self.sparse_features,
                sequence_features=self.sequence_features,
                target=self.target_columns,
                id_columns=id_columns if id_columns is not None else self.id_columns,
                processor=effective_processor,
            )
            return rec_loader.create_dataloader(
                data=data,
                batch_size=1,
                shuffle=False,
                streaming=True,
                chunk_size=batch_size,
                num_workers=num_workers,
                shard_rank=shard_rank,
                shard_count=shard_count,
            )
        return self.prepare_data_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def export_onnx(
        self,
        save_path: str | Path | None = None,
        batch_size: int = 1,
    ) -> Path:
        """
        Export the model to ONNX.

        Usage:
            onnx_path = model.export_onnx(
                save_path="model.onnx",
                batch_size=1,
            )

        Args:
            save_path: Path to save the ONNX model; if None, uses session root.
            batch_size: Dummy batch size for tracing.
        """
        model_to_export = self.ddp_model.module if self.ddp_model is not None else self
        model_to_export = model_to_export.to(self.device)
        model_to_export.eval()

        input_names = [feat.name for feat in self.all_features]
        dummy_inputs = create_dummy_inputs(
            self.all_features,
            batch_size=batch_size,
            device=self.device,
        )
        wrapper = OnnxModelWrapper(model_to_export, input_names)
        with torch.no_grad():
            output_sample = wrapper(*dummy_inputs)
        if isinstance(output_sample, (tuple, list)):  # multiple outputs
            output_names = [f"output_{idx}" for idx in range(len(output_sample))]
        else:
            output_names = ["output"]
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.root,
            default_name=f"{self.model_name}_onnx",
            suffix="onnx",
        )
        export_kwargs: dict[str, Any] = {}
        export_sig = inspect.signature(torch.onnx.export)
        if "dynamo" in export_sig.parameters:
            export_kwargs["dynamo"] = True

        with redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapper,
                tuple(dummy_inputs),
                target_path,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=18,
                do_constant_folding=True,
                **export_kwargs,
            )

        logging.info(colorize(f"ONNX model exported to: {target_path}", color="green"))
        return target_path

    @overload
    def predict_onnx(
        self,
        onnx_path: str | os.PathLike | Path,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[True] = True,
        num_workers: int = 0,
        onnx_session: Any | None = None,
        profiler: StageTimer | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def predict_onnx(
        self,
        onnx_path: str | os.PathLike | Path,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[False] = False,
        num_workers: int = 0,
        onnx_session: Any | None = None,
        profiler: StageTimer | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict_onnx(
        self,
        onnx_path: str | os.PathLike | Path,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        *,
        save_path: str | os.PathLike,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: Literal[False] = False,
        num_workers: int = 0,
        onnx_session: Any | None = None,
        profiler: StageTimer | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
    ) -> Path: ...

    def predict_onnx(
        self,
        onnx_path: str | os.PathLike | Path,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: str = "csv",
        include_ids: bool | None = None,
        id_columns: str | list[str] | None = None,
        return_dataframe: bool = True,
        num_workers: int = 0,
        onnx_session: Any | None = None,
        profiler: StageTimer | None = None,
        num_processes: int = 1,
        processor: Any | None = None,
    ) -> pd.DataFrame | np.ndarray | Path | None:
        """
        Run ONNX inference on the given data.

        Args:
            onnx_path: Path to the ONNX model file.
            data: Input data for prediction (file path, dict, DataFrame, or DataLoader).
            batch_size: Batch size for prediction.
            save_path: Optional path to save predictions; if None, predictions are not saved to disk.
            save_format: Format to save predictions ('csv' or 'parquet').
            include_ids: Whether to include ID columns in the output; if None, includes if id_columns are set.
            id_columns: Column name(s) to use as IDs; if None, uses model's id_columns.
            return_dataframe: Whether to return predictions as a pandas DataFrame; if False, returns a NumPy array.
            num_workers: DataLoader worker count.
            num_processes: Number of inference processes for ONNX streaming file inference.
            onnx_session: Optional pre-created ONNX Runtime session.
        """
        predict_id_columns = id_columns if id_columns is not None else self.id_columns
        if isinstance(predict_id_columns, str):
            predict_id_columns = [predict_id_columns]

        include_ids = bool(predict_id_columns) if include_ids is None else include_ids and bool(predict_id_columns)
        if save_path is not None and save_format not in {"csv", "parquet"}:
            raise ValueError(f"Unsupported save format: {save_format}. Supported: csv, parquet")

        if save_path is not None and not return_dataframe:
            if num_processes > 1 and not isinstance(data, (str, os.PathLike)):
                raise ValueError(
                    "[BaseModel-predict-onnx Error] Multi-process streaming requires data to be a file path."
                )
            if num_processes > 1 and num_workers != 0:
                logging.info("[BaseModel-predict-onnx-streaming Info] Multi-process streaming enforces num_workers=0.")
                num_workers = 0
            if num_processes > 1 and onnx_session is not None:
                raise ValueError(
                    "[BaseModel-predict-onnx Error] onnx_session is not supported when num_processes > 1. "
                    "Please pass onnx_path and let each worker create its own session."
                )
            return self.predict_onnx_streaming(
                onnx_path=onnx_path,
                data=data,
                batch_size=batch_size,
                save_path=save_path,
                save_format=save_format,
                include_ids=include_ids,
                return_dataframe=return_dataframe,
                id_columns=predict_id_columns,
                onnx_session=onnx_session,
                num_workers=num_workers,
                num_processes=num_processes,
                profiler=profiler,
                processor=processor,
            )

        session = onnx_session or load_onnx_session(Path(onnx_path))
        session_inputs = session.get_inputs()
        session_input_names = [inp.name for inp in session_inputs]
        batch_dim = session_inputs[0].shape[0] if session_inputs else None
        fixed_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
        data_loader = self.prepare_onnx_dataloader(
            data=data,
            batch_size=batch_size,
            num_workers=num_workers,
            id_columns=predict_id_columns,
            processor=processor,
        )

        y_pred_list = []
        id_buffers = {name: [] for name in (predict_id_columns or [])} if include_ids else {}

        for batch_data in progress(data_loader, description="Predicting with ONNX"):
            batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
            X_input, _ = self.get_input(batch_dict, require_labels=False)
            first_value = next(iter(X_input.values()), None)
            if first_value is None:
                continue
            current_batch = (
                int(first_value.shape[0])
                if isinstance(first_value, torch.Tensor)
                else int(np.asarray(first_value).shape[0])
            )
            if current_batch <= 0:
                continue
            if fixed_batch is not None and current_batch > fixed_batch:
                slice_ranges = [
                    (start, min(start + fixed_batch, current_batch)) for start in range(0, current_batch, fixed_batch)
                ]
            else:
                slice_ranges = [(0, current_batch)]

            for start_idx, end_idx in slice_ranges:
                X_sub = (
                    {name: value[start_idx:end_idx] for name, value in X_input.items()}
                    if len(slice_ranges) > 1
                    else X_input
                )
                orig_batch = None
                if fixed_batch is not None:
                    X_sub, orig_batch = pad_onnx_inputs(self.all_features, X_sub, target_batch=fixed_batch)
                feed = build_onnx_input_feed(self.all_features, X_sub, input_names=session_input_names)
                start = time.perf_counter()
                outputs = session.run(None, feed)
                if profiler is not None:
                    profiler.add("inference", time.perf_counter() - start)
                y_pred_np = merge_onnx_outputs(outputs)
                if y_pred_np.ndim == 1:
                    y_pred_np = y_pred_np.reshape(-1, 1)
                if orig_batch is not None and orig_batch > 0:
                    y_pred_np = y_pred_np[:orig_batch]
                y_pred_list.append(y_pred_np)

                if include_ids and predict_id_columns and batch_dict.get("ids"):
                    ids_dict = batch_dict["ids"]
                    ids_sub = (
                        {name: value[start_idx:end_idx] for name, value in ids_dict.items()}
                        if len(slice_ranges) > 1
                        else ids_dict
                    )
                    orig_id_batch = None
                    if fixed_batch is not None:
                        ids_sub, orig_id_batch = pad_id_batch(ids_sub, fixed_batch)
                    for id_name in predict_id_columns:
                        if id_name not in ids_sub:
                            continue
                        id_tensor = ids_sub[id_name]
                        id_np = (
                            id_tensor.detach().cpu().numpy()
                            if isinstance(id_tensor, torch.Tensor)
                            else np.asarray(id_tensor)
                        )
                        if orig_batch is not None and orig_batch > 0:
                            id_np = id_np[:orig_batch]
                        elif orig_id_batch is not None and orig_id_batch > 0:
                            id_np = id_np[:orig_id_batch]
                        id_buffers[id_name].append(id_np.reshape(id_np.shape[0], -1) if id_np.ndim == 1 else id_np)

        y_pred_all = np.concatenate(y_pred_list, axis=0) if y_pred_list else None
        if y_pred_all is None:
            return pd.DataFrame() if return_dataframe else np.array([])

        num_outputs = y_pred_all.shape[1] if y_pred_all.ndim > 1 else 1
        pred_columns = list(self.target_columns[:num_outputs]) if self.target_columns else []
        while len(pred_columns) < num_outputs:
            pred_columns.append(f"pred_{len(pred_columns)}")

        id_df = None
        if include_ids and predict_id_columns:
            id_arrays = {
                id_name: np.concatenate(id_buffers[id_name], axis=0)
                for id_name in predict_id_columns
                if id_buffers.get(id_name)
            }
            if id_arrays:
                id_df = pd.DataFrame(id_arrays)

        output = y_pred_all
        if return_dataframe:
            df_to_return = pd.DataFrame(y_pred_all, columns=pred_columns)
            if id_df is not None:
                df_to_return = pd.concat([id_df, df_to_return], axis=1)
            output = df_to_return

        if save_path is not None:
            target_path = get_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=f".{save_format}",
            )
            if return_dataframe and isinstance(output, pd.DataFrame):
                df_to_save = output
            else:
                df_to_save = pd.DataFrame(y_pred_all, columns=pred_columns)
                if id_df is not None:
                    df_to_save = pd.concat([id_df, df_to_save], axis=1)

            if save_format == "csv":
                start = time.perf_counter()
                df_to_save.to_csv(target_path, index=False)
                if profiler is not None:
                    profiler.add("write_output", time.perf_counter() - start)
            elif save_format == "parquet":
                start = time.perf_counter()
                df_to_save.to_parquet(target_path, index=False)
                if profiler is not None:
                    profiler.add("write_output", time.perf_counter() - start)
            logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        return output

    def predict_onnx_streaming(
        self,
        onnx_path: str | os.PathLike | Path,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: str,
        include_ids: bool,
        return_dataframe: bool,
        id_columns: list[str] | None = None,
        onnx_session: Any | None = None,
        num_workers: int = 0,
        num_processes: int = 1,
        shard_rank: int = 0,
        shard_count: int = 1,
        profiler: StageTimer | None = None,
        processor: Any | None = None,
    ):
        """
        Run ONNX inference using streaming mode for large datasets.
        """
        if save_format not in {"csv", "parquet"}:
            raise ValueError(f"Unsupported save format: {save_format}. Supported: csv, parquet")
        if num_processes > 1:
            if onnx_session is not None:
                raise ValueError(
                    "[BaseModel-predict-onnx Error] onnx_session is not supported when num_processes > 1. "
                    "Please pass onnx_path and let each worker create its own session."
                )
            if not isinstance(data, (str, os.PathLike)):
                raise ValueError(
                    "[BaseModel-predict-onnx Error] Multi-process streaming requires data to be a file path."
                )
            if num_workers > 0:
                logging.info("[BaseModel-predict-onnx-streaming Info] Multi-process streaming enforces num_workers=0.")
            if return_dataframe:
                raise ValueError(
                    "[BaseModel-predict-onnx Error] Multi-process streaming does not support return_dataframe=True."
                )
            num_workers = 0

            target_path = Path(
                get_save_path(
                    path=save_path,
                    default_dir=self.session.predictions_dir,
                    default_name="predictions",
                    suffix=f".{save_format}",
                    add_timestamp=False,
                )
            )
            parts_dir = target_path.parent / f".{target_path.stem}_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_paths = [
                parts_dir / f"{target_path.stem}.part{rank}{target_path.suffix}" for rank in range(num_processes)
            ]
            profile_paths = (
                [parts_dir / f"{target_path.stem}.profile{rank}.json" for rank in range(num_processes)]
                if profiler is not None
                else None
            )
            error_paths = [parts_dir / f"{target_path.stem}.error{rank}.txt" for rank in range(num_processes)]

            ctx = mp.get_context("spawn")
            processes = []
            for rank in range(num_processes):
                process = ctx.Process(
                    target=predict_onnx_streaming_worker,
                    args=(
                        self,
                        onnx_path,
                        data,
                        batch_size,
                        part_paths[rank],
                        save_format,
                        include_ids,
                        id_columns,
                        rank,
                        num_processes,
                        num_workers,
                        processor,
                        profile_paths[rank] if profile_paths else None,
                        error_paths[rank],
                    ),
                )
                process.start()
                processes.append(process)

            for process in progress(iter(processes), description="Predicting...", total=None):
                process.join()

            failed_processes = [process for process in processes if process.exitcode not in (0, None)]
            if failed_processes:
                errors: list[str] = []
                for error_path in error_paths:
                    if not error_path.exists():
                        continue
                    text = error_path.read_text(encoding="utf-8").strip()
                    if text and text not in errors:
                        errors.append(text)
                if not errors:
                    errors = [
                        f"[PredictOnnxWorker Error] pid={process.pid}, exitcode={process.exitcode}, name={process.name}"
                        for process in failed_processes
                    ]
                error_text = "\n\n".join(errors)
                raise RuntimeError(
                    "[BaseModel-predict-onnx-streaming Error] One or more inference processes failed.\n" + error_text
                )

            existing_parts = [p for p in part_paths if p.exists()]
            if existing_parts:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if save_format == "csv":
                    start = time.perf_counter()
                    lazy_frames = [pl.scan_csv(p) for p in existing_parts]
                    pl.concat(lazy_frames).sink_csv(target_path)
                    if profiler is not None:
                        profiler.add("merge_parts", time.perf_counter() - start)
                elif save_format == "parquet":
                    start = time.perf_counter()
                    lazy_frames = [pl.scan_parquet(p) for p in existing_parts]
                    pl.concat(lazy_frames).sink_parquet(target_path)
                    if profiler is not None:
                        profiler.add("merge_parts", time.perf_counter() - start)

            for part_path in part_paths:
                if part_path.exists():
                    part_path.unlink()
            if profile_paths and profiler is not None:
                for profile_path in profile_paths:
                    if profile_path.exists():
                        profiler.merge(StageTimer.load(profile_path))
                        profile_path.unlink()
            for error_path in error_paths:
                if error_path.exists():
                    error_path.unlink()
            if parts_dir.exists() and not any(parts_dir.iterdir()):
                parts_dir.rmdir()

            logging.info("")
            logging.info(
                colorize(
                    f"Predictions saved to: {target_path} (merged from {num_processes} parts)",
                    color="green",
                )
            )
            return target_path

        session = onnx_session or load_onnx_session(Path(onnx_path))
        session_inputs = session.get_inputs()
        session_input_names = [inp.name for inp in session_inputs]
        batch_dim = session_inputs[0].shape[0] if session_inputs else None
        fixed_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
        data_loader = self.prepare_onnx_dataloader(
            data=data,
            batch_size=batch_size,
            num_workers=num_workers,
            id_columns=id_columns,
            processor=processor,
            shard_rank=shard_rank,
            shard_count=shard_count,
        )

        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.predictions_dir,
            default_name="predictions",
            suffix=f".{save_format}",
            add_timestamp=False,
        )
        header_written = target_path.exists() and target_path.stat().st_size > 0
        parquet_writer = None
        pred_columns = None
        cached_frames = []

        disable_progress = shard_count > 1
        for batch_data in progress(data_loader, description="Predicting with ONNX", disable=disable_progress):
            batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
            X_input, _ = self.get_input(batch_dict, require_labels=False)
            # ONNX Runtime may require fixed batch sizes, need to slice and pad batches accordingly
            first_value = next(iter(X_input.values()), None)
            if first_value is None:
                continue
            current_batch = (
                int(first_value.shape[0])
                if isinstance(first_value, torch.Tensor)
                else int(np.asarray(first_value).shape[0])
            )
            if current_batch <= 0:
                continue
            if fixed_batch is not None and current_batch > fixed_batch:
                slice_ranges = [
                    (start, min(start + fixed_batch, current_batch)) for start in range(0, current_batch, fixed_batch)
                ]
            else:
                slice_ranges = [(0, current_batch)]

            for start_idx, end_idx in slice_ranges:
                X_sub = (
                    {name: value[start_idx:end_idx] for name, value in X_input.items()}
                    if len(slice_ranges) > 1
                    else X_input
                )
                orig_batch = None
                if fixed_batch is not None:
                    X_sub, orig_batch = pad_onnx_inputs(self.all_features, X_sub, target_batch=fixed_batch)
                feed = build_onnx_input_feed(self.all_features, X_sub, input_names=session_input_names)
                start = time.perf_counter()
                outputs = session.run(None, feed)
                if profiler is not None:
                    profiler.add("inference", time.perf_counter() - start)
                y_pred_np = merge_onnx_outputs(outputs)
                if y_pred_np.ndim == 1:
                    y_pred_np = y_pred_np.reshape(-1, 1)
                if orig_batch is not None and orig_batch > 0:
                    y_pred_np = y_pred_np[:orig_batch]
                if pred_columns is None:
                    num_outputs = y_pred_np.shape[1]
                    pred_columns = list(self.target_columns[:num_outputs]) if self.target_columns else []
                    while len(pred_columns) < num_outputs:
                        pred_columns.append(f"pred_{len(pred_columns)}")

                ids = batch_dict.get("ids") if include_ids and id_columns else None
                if ids is not None and len(slice_ranges) > 1:
                    ids = {name: value[start_idx:end_idx] for name, value in ids.items()}
                if ids and fixed_batch is not None:
                    ids, orig_id_batch = pad_id_batch(ids, fixed_batch)
                else:
                    orig_id_batch = None
                id_arrays_batch = {
                    id_name: (
                        ids[id_name].detach().cpu().numpy()
                        if isinstance(ids[id_name], torch.Tensor)
                        else np.asarray(ids[id_name])
                    ).reshape(-1)
                    for id_name in (id_columns or [])
                    if ids and id_name in ids
                }
                if orig_batch is not None and orig_batch > 0:
                    id_arrays_batch = {k: v[:orig_batch] for k, v in id_arrays_batch.items()}
                elif orig_id_batch is not None and orig_id_batch > 0:
                    id_arrays_batch = {k: v[:orig_id_batch] for k, v in id_arrays_batch.items()}

                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if id_arrays_batch:
                    id_df = pd.DataFrame(id_arrays_batch)
                    df_batch = pd.concat([id_df, df_batch], axis=1)

                should_collect = return_dataframe
                if should_collect:
                    cached_frames.append(df_batch)

                if save_format == "csv":
                    start = time.perf_counter()
                    df_batch.to_csv(target_path, mode="a", header=not header_written, index=False)
                    if profiler is not None:
                        profiler.add("write_output", time.perf_counter() - start)
                    header_written = True
                elif save_format == "parquet":
                    start = time.perf_counter()
                    table = pa.Table.from_pandas(df_batch, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(target_path, table.schema)
                    parquet_writer.write_table(table)
                    if profiler is not None:
                        profiler.add("write_output", time.perf_counter() - start)
            # Non-streaming formats are saved after collecting all batches.

        if parquet_writer is not None:
            parquet_writer.close()

        logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        if return_dataframe:
            return (
                pd.concat(cached_frames, ignore_index=True)
                if cached_frames
                else pd.DataFrame(columns=pred_columns or [])
            )
        return target_path

    def save_model(
        self,
        save_path: str | Path | None = None,
        add_timestamp: bool | None = None,
        verbose: bool = True,
    ):
        """
        Save the model state and features configuration to disk.

        Args:
            save_path: Path to save the model; if None, saves to the session's model directory.
            add_timestamp: Whether to add a timestamp to the filename; if None, defaults to True.
            verbose: Whether to log the save location.
        """
        add_timestamp = False if add_timestamp is None else add_timestamp
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session_path,
            default_name=self.model_name.upper(),
            suffix=".pt",
            add_timestamp=add_timestamp,
        )
        model_path = Path(target_path)

        ddp_model = self.ddp_model if hasattr(self, "ddp_model") else None
        if ddp_model is not None:
            model_to_save = ddp_model.module
        else:
            model_to_save = self
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
        """
        Load the model state and features configuration from disk.

        Args:
            save_path: Path to load the model from; can be a directory or a specific .pt file.
            map_location: Device mapping for loading the model (e.g., 'cpu', 'cuda:0').
            verbose: Whether to log the load location.
        """
        self.to(self.device)
        base_path = Path(save_path)
        if base_path.is_dir():
            model_files = sorted(base_path.glob("*.pt"))
            if not model_files:
                raise FileNotFoundError(f"[BaseModel-load-model Error] No *.pt file found in directory: {base_path}")
            model_path = model_files[-1]
            config_dir = base_path
        else:
            model_path = base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
            config_dir = model_path.parent
        if not model_path.exists():
            raise FileNotFoundError(f"[BaseModel-load-model Error] Model file does not exist: {model_path}")

        state_dict = torch.load(model_path, map_location=map_location)
        self.load_state_dict(state_dict)

        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(f"[BaseModel-load-model Error] features_config.pkl not found in: {config_dir}")
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

        Args:
            checkpoint_path: Path to the checkpoint directory or specific .pt file.
            map_location: Device mapping for loading the model (e.g., 'cpu', 'cuda:0').
            device: Device to place the model on after loading.
            session_id: Optional session ID for the model.
            **kwargs: Additional keyword arguments to pass to the model constructor.
        """
        base_path = Path(checkpoint_path)
        verbose = kwargs.pop("verbose", True)
        if base_path.is_dir():
            model_candidates = sorted(base_path.glob("*.pt"))
            if not model_candidates:
                raise FileNotFoundError(f"[BaseModel-from-checkpoint Error] No *.pt file found under: {base_path}")
            model_file = model_candidates[-1]
            config_dir = base_path
        else:
            model_file = base_path.with_suffix(".pt") if base_path.suffix == "" else base_path
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


def predict_streaming_worker(
    model: "BaseModel",
    data_path: str | os.PathLike,
    batch_size: int,
    save_path: str | os.PathLike,
    save_format: str,
    stream_chunk_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    processor: Any | None,
    shard_rank: int,
    shard_count: int,
    profile_path: str | os.PathLike | None = None,
    error_path: str | os.PathLike | None = None,
) -> None:
    try:
        model.eval()
        profiler = StageTimer(enabled=profile_path is not None) if profile_path else None
        model.predict_streaming(
            data=data_path,
            batch_size=batch_size,
            save_path=save_path,
            save_format=save_format,
            stream_chunk_size=stream_chunk_size,
            return_dataframe=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            processor=processor,
            num_processes=1,
            shard_rank=shard_rank,
            shard_count=shard_count,
            profiler=profiler,
        )
        if profiler is not None and profile_path is not None:
            profiler.dump(Path(profile_path))
    except BaseException:
        import traceback

        error_text = f"[PredictWorker Error] rank={shard_rank}\n{traceback.format_exc()}"
        if error_path is not None:
            path = Path(error_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(error_text, encoding="utf-8")
        raise


def predict_onnx_streaming_worker(
    model: "BaseModel",
    onnx_path: str | os.PathLike | Path,
    data_path: str | os.PathLike,
    batch_size: int,
    save_path: str | os.PathLike,
    save_format: str,
    include_ids: bool,
    id_columns: list[str] | None,
    shard_rank: int,
    shard_count: int,
    num_workers: int,
    processor: Any | None = None,
    profile_path: str | os.PathLike | None = None,
    error_path: str | os.PathLike | None = None,
) -> None:
    try:
        model.eval()
        profiler = StageTimer(enabled=profile_path is not None) if profile_path else None
        model.predict_onnx_streaming(
            onnx_path=onnx_path,
            data=data_path,
            batch_size=batch_size,
            save_path=save_path,
            save_format=save_format,
            include_ids=include_ids,
            return_dataframe=False,
            id_columns=id_columns,
            onnx_session=None,
            num_workers=num_workers,
            num_processes=1,
            shard_rank=shard_rank,
            shard_count=shard_count,
            profiler=profiler,
            processor=processor,
        )
        if profiler is not None and profile_path is not None:
            profiler.dump(Path(profile_path))
    except BaseException:
        import traceback

        error_text = f"[PredictOnnxWorker Error] rank={shard_rank}\n{traceback.format_exc()}"
        if error_path is not None:
            path = Path(error_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(error_text, encoding="utf-8")
        raise


class BaseMatchModel(BaseModel):
    """
    Base class for match (retrieval/recall) models

    - Pointwise: predicts a user-item match score/probability using labels (default target: 'label')
    - Pairwise/Listwise: trains with in-batch negatives; labels can be omitted by setting target=None
    """

    @property
    def model_name(self) -> str:  # type: ignore[override]
        raise NotImplementedError

    @property
    def default_task(self) -> TaskTypeName:
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
        target: list[str] | str | None = "label",
        id_columns: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        session_id: str | None = None,
        distributed: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        ddp_find_unused_parameters: bool = False,
        **kwargs,
    ):

        user_dense_features = list(user_dense_features or [])
        user_sparse_features = list(user_sparse_features or [])
        user_sequence_features = list(user_sequence_features or [])
        item_dense_features = list(item_dense_features or [])
        item_sparse_features = list(item_sparse_features or [])
        item_sequence_features = list(item_sequence_features or [])

        all_dense_features = user_dense_features + item_dense_features
        all_sparse_features = user_sparse_features + item_sparse_features
        all_sequence_features = user_sequence_features + item_sequence_features

        super(BaseMatchModel, self).__init__(
            dense_features=all_dense_features,
            sparse_features=all_sparse_features,
            sequence_features=all_sequence_features,
            target=target,
            id_columns=id_columns,
            task=task,
            training_mode=training_mode,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            session_id=session_id,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            **kwargs,
        )

        self.user_dense_features = user_dense_features
        self.user_sparse_features = user_sparse_features
        self.user_sequence_features = user_sequence_features

        self.item_dense_features = item_dense_features
        self.item_sparse_features = item_sparse_features
        self.item_sequence_features = item_sequence_features

        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        self.primary_mode = self.training_modes[0] if self.training_modes else "pointwise"
        if self.primary_mode not in self.support_training_modes:
            raise ValueError(
                f"{self.model_name.upper()} does not support training_mode='{self.primary_mode}'. Supported modes: {self.support_training_modes}"
            )
        self.user_features_all = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        self.item_features_all = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        self.user_feature_names = {feature.name for feature in self.user_features_all}
        self.item_feature_names = {feature.name for feature in self.item_features_all}
        self.head = RetrievalHead(
            similarity_metric=self.similarity_metric,
            temperature=self.temperature,
            training_mode=self.primary_mode,
            apply_sigmoid=True,
        )

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
        loss: LossName | nn.Module | list[LossName | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | dict | str | None = None,
        ignore_label: int | float | None = -1,
    ):
        """
        Configure the match model for training.

        Args:
            optimizer: Optimizer to use (name or instance). e.g., 'adam', 'sgd'.
            optimizer_params: Parameters for the optimizer. e.g., {'lr': 0.001}.
            scheduler: Learning rate scheduler (name, instance, or class). e.g., 'step_lr'.
            scheduler_params: Parameters for the scheduler. e.g., {'step_size': 10, 'gamma': 0.1}.
            loss: Loss function(s) to use (name, instance, or list). e.g., 'bce'.
            loss_params: Parameters for the loss function(s). e.g., {'reduction': 'mean'}.
            loss_weights: Weights for the loss function(s). e.g., 1.0 or [0.7, 0.3].
        """
        default_loss_by_mode: dict[str, LossName] = {
            "pointwise": "bce",
            "pairwise": "bpr",
            "listwise": "sampled_softmax",
        }

        effective_loss: LossName | nn.Module | list[LossName | nn.Module] | None = loss
        primary_mode = self.primary_mode
        if effective_loss is None:
            effective_loss = default_loss_by_mode[primary_mode]
        elif isinstance(effective_loss, str):
            if primary_mode in {"pairwise", "listwise"} and effective_loss in {
                "bce",
                "binary_crossentropy",
            }:
                effective_loss = default_loss_by_mode[primary_mode]
        elif isinstance(effective_loss, list):
            if not effective_loss:
                effective_loss = [default_loss_by_mode[primary_mode]]
            else:
                first = effective_loss[0]
                if (
                    primary_mode in {"pairwise", "listwise"}
                    and isinstance(first, str)
                    and first in {"bce", "binary_crossentropy"}
                ):
                    effective_loss = [
                        default_loss_by_mode[primary_mode],
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
            ignore_label=ignore_label,
        )

    def inbatch_logits(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute in-batch logits matrix between user and item embeddings."""
        if self.similarity_metric == "dot":
            logits = torch.matmul(user_emb, item_emb.t())
        elif self.similarity_metric == "cosine":
            user_norm = F.normalize(user_emb, p=2, dim=-1)
            item_norm = F.normalize(item_emb, p=2, dim=-1)
            logits = torch.matmul(user_norm, item_norm.t())
        elif self.similarity_metric == "euclidean":
            user_sq = torch.sum(user_emb**2, dim=1, keepdim=True)  # [B, 1]
            item_sq = torch.sum(item_emb**2, dim=1, keepdim=True).t()  # [1, B]
            logits = -(user_sq + item_sq - 2.0 * torch.matmul(user_emb, item_emb.t()))
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        return logits / self.temperature

    def compute_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute similarity score between user and item embeddings."""
        if user_emb.dim() == 2 and item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(1)

        if self.similarity_metric == "dot":
            similarity = torch.sum(user_emb * item_emb, dim=-1)
        elif self.similarity_metric == "cosine":
            similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        elif self.similarity_metric == "euclidean":
            similarity = -torch.sum((user_emb - item_emb) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        similarity = similarity / self.temperature
        return similarity

    def user_tower(self, user_input: dict) -> torch.Tensor:
        """User tower to encode user features into embeddings."""
        raise NotImplementedError

    def item_tower(self, item_input: dict) -> torch.Tensor:
        """Item tower to encode item features into embeddings."""
        raise NotImplementedError

    def forward(self, X_input: dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rewrite forward to handle user and item features separately."""
        user_input = {name: tensor for name, tensor in X_input.items() if name in self.user_feature_names}
        item_input = {name: tensor for name, tensor in X_input.items() if name in self.item_feature_names}

        user_emb = self.user_tower(user_input)  # [B, D]
        item_emb = self.item_tower(item_input)  # [B, D]

        return self.head(user_emb, item_emb, similarity_fn=self.compute_similarity)

    def compute_loss(self, y_pred, y_true):
        primary_mode = self.primary_mode
        if primary_mode == "pointwise":
            return super().compute_loss(y_pred, y_true)

        # pairwise / listwise using inbatch neg
        elif primary_mode in ["pairwise", "listwise"]:
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

            loss_fn = self.loss_fn[0] if hasattr(self, "loss_fn") and self.loss_fn else None
            if isinstance(loss_fn, SampledSoftmaxLoss):
                loss = loss_fn(pos_logits, neg_logits)
            elif isinstance(loss_fn, (BPRLoss, HingeLoss)):
                loss = loss_fn(pos_logits, neg_logits)
            elif isinstance(loss_fn, TripletLoss):
                neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(batch_size, batch_size - 1, item_emb.size(-1))
                loss = loss_fn(user_emb, item_emb, neg_emb)
            elif isinstance(loss_fn, InfoNCELoss) and self.similarity_metric == "dot":
                neg_emb = item_emb.masked_select(~eye.unsqueeze(-1)).view(batch_size, batch_size - 1, item_emb.size(-1))
                loss = loss_fn(user_emb, item_emb, neg_emb)
            else:
                targets = torch.arange(batch_size, device=logits.device)
                loss = F.cross_entropy(logits, targets)

            if self.loss_weights is not None:
                loss *= float(self.loss_weights[0])
            return loss
        else:
            raise ValueError(f"Unknown training mode: {primary_mode}")

    def prepare_feature_data(
        self,
        data,
        features: list,
        batch_size: int,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> DataLoader:
        """Prepare data loader for specific features."""
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, (str, os.PathLike)):
            dense_features = [f for f in features if isinstance(f, DenseFeature)]
            sparse_features = [f for f in features if isinstance(f, SparseFeature)]
            sequence_features = [f for f in features if isinstance(f, SequenceFeature)]
            rec_loader = RecDataLoader(
                dense_features=dense_features,
                sparse_features=sparse_features,
                sequence_features=sequence_features,
                target=[],
                id_columns=[],
            )
            return rec_loader.create_dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=False,
                streaming=True,
                chunk_size=stream_chunk_size,
                num_workers=num_workers,
            )
        tensors = build_tensors_from_data(
            data=data,
            raw_data=data,
            features=features,
            target_columns=[],
            id_columns=[],
        )
        if tensors is None:
            raise ValueError("[BaseMatchModel-prepare_feature_data Error] No data available to create DataLoader.")
        dataset = TensorDictDataset(tensors)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def build_feature_tensors(self, feature_source: dict, features: list) -> dict:
        """Convert feature values to tensors on the model device."""
        tensors = {}
        for feature in features:
            if feature.name not in feature_source:
                raise KeyError(f"[BaseMatchModel-feature Error] Feature '{feature.name}' not found in input data.")
            feature_data = get_column_data(feature_source, feature.name)
            tensors[feature.name] = to_tensor(
                feature_data,
                dtype=(torch.float32 if isinstance(feature, DenseFeature) else torch.long),
                device=self.device,
            )
        return tensors

    def encode_user(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        self.eval()
        data_loader = self.prepare_feature_data(
            data,
            self.user_features_all,
            batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )

        embeddings_list = []
        with torch.no_grad():
            for batch_data in progress(data_loader, description="Encoding users"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                user_input = self.build_feature_tensors(batch_dict["features"], self.user_features_all)
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)

    def encode_item(
        self,
        data: dict | pd.DataFrame | DataLoader | str | os.PathLike | list[str | os.PathLike],
        batch_size: int = 512,
        num_workers: int = 0,
        stream_chunk_size: int = 10000,
    ) -> np.ndarray:
        self.eval()
        data_loader = self.prepare_feature_data(
            data,
            self.item_features_all,
            batch_size,
            num_workers=num_workers,
            stream_chunk_size=stream_chunk_size,
        )

        embeddings_list = []
        with torch.no_grad():
            for batch_data in progress(data_loader, description="Encoding items"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                item_input = self.build_feature_tensors(batch_dict["features"], self.item_features_all)
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
