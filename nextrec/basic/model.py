"""
Base Model & Base Match Model Class

Date: create on 27/10/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from typing import Union, Literal
from torch.utils.data import DataLoader, TensorDataset

from nextrec.basic.callback import EarlyStopper
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature, FeatureConfig
from nextrec.basic.metrics import configure_metrics, evaluate_metrics

from nextrec.loss import get_loss_fn
from nextrec.data import get_column_data
from nextrec.data.dataloader import build_tensors_from_data
from nextrec.basic.loggers import setup_logger, colorize
from nextrec.utils import get_optimizer_fn, get_scheduler_fn
from nextrec.basic.session import resolve_save_path, create_session


class BaseModel(FeatureConfig, nn.Module):
    @property
    def model_name(self) -> str:
        raise NotImplementedError
    
    @property
    def task_type(self) -> str:
        raise NotImplementedError

    def __init__(self, 
                 dense_features: list[DenseFeature] | None = None, 
                 sparse_features: list[SparseFeature] | None = None, 
                 sequence_features: list[SequenceFeature] | None = None,
                 target: list[str] | str | None = None,
                 id_columns: list[str] | str | None = None,
                 task: str|list[str] = 'binary',
                 device: str = 'cpu',
                 embedding_l1_reg: float = 0.0,
                 dense_l1_reg: float = 0.0,
                 embedding_l2_reg: float = 0.0, 
                 dense_l2_reg: float = 0.0,
                 early_stop_patience: int = 20, 
                 session_id: str | None = None,): 
        
        super(BaseModel, self).__init__()

        try:
            self.device = torch.device(device)
        except Exception as e:
            logging.warning("Invalid device , defaulting to CPU.")
            self.device = torch.device('cpu')

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = Path(self.session.logs_dir)
        checkpoint_dir = self.session.checkpoints_dir / self.model_name

        self.checkpoint = resolve_save_path(
            path=None,
            default_dir=checkpoint_dir,
            default_name=self.model_name,
            suffix=".model",
            add_timestamp=True,
        )

        self.best = resolve_save_path(
            path="best.model",
            default_dir=checkpoint_dir,
            default_name="best",
            suffix=".model",
        )

        self._set_feature_config(dense_features, sparse_features, sequence_features)
        self._set_target_config(target, id_columns)

        self.target = self.target_columns
        self.target_index = {target_name: idx for idx, target_name in enumerate(self.target)}

        self.task = task
        self.nums_task = len(task) if isinstance(task, list) else 1

        self._embedding_l1_reg = embedding_l1_reg
        self._dense_l1_reg = dense_l1_reg
        self._embedding_l2_reg = embedding_l2_reg
        self._dense_l2_reg = dense_l2_reg

        self._regularization_weights = [] # list of dense weights for regularization, used to compute reg loss
        self._embedding_params = [] # list of embedding weights for regularization, used to compute reg loss

        self.early_stop_patience = early_stop_patience
        self._max_gradient_norm = 1.0   # Maximum gradient norm for gradient clipping

        self._logger_initialized = False
        self._verbose = 1

    def _register_regularization_weights(self, 
                                        embedding_attr: str = 'embedding',
                                        exclude_modules: list[str] | None = [], # modules wont add regularization, example: ['fm', 'lr'] / ['fm.fc'] / etc.
                                        include_modules: list[str] | None = []):

        exclude_modules = exclude_modules or []
        
        if hasattr(self, embedding_attr):
            embedding_layer = getattr(self, embedding_attr)
            if hasattr(embedding_layer, 'embed_dict'):
                for embed in embedding_layer.embed_dict.values():
                    self._embedding_params.append(embed.weight)

        for name, module in self.named_modules():
            # Skip self module
            if module is self:
                continue
            
            # Skip embedding layers
            if embedding_attr in name:
                continue
            
            # Skip BatchNorm and Dropout by checking module type
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                   nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                continue
            
            # White-list: only include modules whose names contain specific keywords
            if include_modules is not None:
                should_include = any(inc_name in name for inc_name in include_modules)
                if not should_include:
                    continue
            
            # Black-list: exclude modules whose names contain specific keywords
            if any(exc_name in name for exc_name in exclude_modules):
                continue
            
            # Only add regularization for Linear layers
            if isinstance(module, nn.Linear):
                self._regularization_weights.append(module.weight)

    def add_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        
        if self._embedding_l1_reg > 0 and len(self._embedding_params) > 0:
            for param in self._embedding_params:
                reg_loss += self._embedding_l1_reg * torch.sum(torch.abs(param))

        if self._embedding_l2_reg > 0 and len(self._embedding_params) > 0:
            for param in self._embedding_params:
                reg_loss += self._embedding_l2_reg * torch.sum(param ** 2)

        if self._dense_l1_reg > 0 and len(self._regularization_weights) > 0:
            for param in self._regularization_weights:
                reg_loss += self._dense_l1_reg * torch.sum(torch.abs(param))

        if self._dense_l2_reg > 0 and len(self._regularization_weights) > 0:
            for param in self._regularization_weights:
                reg_loss += self._dense_l2_reg * torch.sum(param ** 2)
        
        return reg_loss

    def _to_tensor(self, value, dtype: torch.dtype | None = None, device: str | torch.device | None = None) -> torch.Tensor:        
        if value is None:
            raise ValueError("Cannot convert None to tensor.")
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        target_device = device if device is not None else self.device
        return tensor.to(target_device)

    def get_input(self, input_data: dict|pd.DataFrame):
        X_input = {}
        
        all_features = self.dense_features + self.sparse_features + self.sequence_features
        
        for feature in all_features:
            if feature.name not in input_data:
                continue
            feature_data = get_column_data(input_data, feature.name)
            if feature_data is None:
                continue
            if isinstance(feature, DenseFeature):
                dtype = torch.float32
            else:
                dtype = torch.long
            feature_tensor = self._to_tensor(feature_data, dtype=dtype)
            X_input[feature.name] = feature_tensor

        y = None
        if len(self.target) > 0:
            target_tensors = []
            for target_name in self.target:
                if target_name not in input_data:
                    continue
                target_data = get_column_data(input_data, target_name)
                if target_data is None:
                    continue
                target_tensor = self._to_tensor(target_data, dtype=torch.float32)
                
                if target_tensor.dim() > 1:
                    target_tensor = target_tensor.view(target_tensor.size(0), -1)
                    target_tensors.extend(torch.chunk(target_tensor, chunks=target_tensor.shape[1], dim=1))
                else:
                    target_tensors.append(target_tensor.view(-1, 1))

            if target_tensors:
                stacked = torch.cat(target_tensors, dim=1)
                if stacked.shape[1] == 1:
                    y = stacked.view(-1)
                else:
                    y = stacked

        return X_input, y

    def _set_metrics(self, metrics: list[str] | dict[str, list[str]] | None = None):
        """Configure metrics for model evaluation using the metrics module."""
        self.metrics, self.task_specific_metrics, self.best_metrics_mode = configure_metrics(
            task=self.task,
            metrics=metrics,
            target_names=self.target
        ) # ['auc', 'logloss'], {'target1': ['auc', 'logloss'], 'target2': ['mse']}, 'max'
        
        if not hasattr(self, 'early_stopper') or self.early_stopper is None:
            self.early_stopper = EarlyStopper(patience=self.early_stop_patience, mode=self.best_metrics_mode)

    def _validate_task_configuration(self):
        """Validate that task type, number of tasks, targets, and loss functions are consistent."""
        # Check task and target consistency
        if isinstance(self.task, list):
            num_tasks_from_task = len(self.task)
        else:
            num_tasks_from_task = 1
        
        num_targets = len(self.target)
        
        if self.nums_task != num_tasks_from_task:
            raise ValueError(
                f"Number of tasks mismatch: nums_task={self.nums_task}, "
                f"but task list has {num_tasks_from_task} tasks."
            )
        
        if self.nums_task != num_targets:
            raise ValueError(
                f"Number of tasks ({self.nums_task}) does not match number of target columns ({num_targets}). "
                f"Tasks: {self.task}, Targets: {self.target}"
            )
        
        # Check loss function consistency
        if hasattr(self, 'loss_fn'):
            num_loss_fns = len(self.loss_fn)
            if num_loss_fns != self.nums_task:
                raise ValueError(
                    f"Number of loss functions ({num_loss_fns}) does not match number of tasks ({self.nums_task})."
                )
        
        # Validate task types with metrics and loss functions
        from nextrec.loss import VALID_TASK_TYPES
        from nextrec.basic.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
        
        tasks_to_check = self.task if isinstance(self.task, list) else [self.task]
        
        for i, task_type in enumerate(tasks_to_check):
            # Validate task type
            if task_type not in VALID_TASK_TYPES:
                raise ValueError(
                    f"Invalid task type '{task_type}' for task {i}. "
                    f"Valid types: {VALID_TASK_TYPES}"
                )
            
            # Check metrics compatibility
            if hasattr(self, 'task_specific_metrics') and self.task_specific_metrics:
                target_name = self.target[i] if i < len(self.target) else f"task_{i}"
                task_metrics = self.task_specific_metrics.get(target_name, self.metrics)
                
                for metric in task_metrics:
                    metric_lower = metric.lower()
                    # Skip gauc as it's valid for both classification and regression in some contexts
                    if metric_lower == 'gauc':
                        continue
                    
                    if task_type in ['binary', 'multiclass']:
                        # Classification task
                        if metric_lower in REGRESSION_METRICS:
                            raise ValueError(
                                f"Metric '{metric}' is not compatible with classification task type '{task_type}' "
                                f"for target '{target_name}'. Classification metrics: {CLASSIFICATION_METRICS}"
                            )
                    elif task_type in ['regression', 'multivariate_regression']:
                        # Regression task
                        if metric_lower in CLASSIFICATION_METRICS:
                            raise ValueError(
                                f"Metric '{metric}' is not compatible with regression task type '{task_type}' "
                                f"for target '{target_name}'. Regression metrics: {REGRESSION_METRICS}"
                            )

    def _handle_validation_split(self, 
                                 train_data: dict | pd.DataFrame | DataLoader,
                                 validation_split: float,
                                 batch_size: int,
                                 shuffle: bool) -> tuple[DataLoader, dict | pd.DataFrame]:
        """Handle validation split logic for training data.
        
        Args:
            train_data: Training data (dict, DataFrame, or DataLoader)
            validation_split: Fraction of data to use for validation (0 < validation_split < 1)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle training data
            
        Returns:
            tuple: (train_loader, valid_data)
        """
        if not (0 < validation_split < 1):
            raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")
        
        if isinstance(train_data, DataLoader):
            raise ValueError(
                "validation_split cannot be used when train_data is a DataLoader. "
                "Please provide dict or pd.DataFrame for train_data."
            )
        
        if isinstance(train_data, pd.DataFrame):
            # Shuffle and split DataFrame
            shuffled_df = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
            split_idx = int(len(shuffled_df) * (1 - validation_split))
            train_split = shuffled_df.iloc[:split_idx]
            valid_split = shuffled_df.iloc[split_idx:]
            
            train_loader = self._prepare_data_loader(train_split, batch_size=batch_size, shuffle=shuffle)
            
            if self._verbose:
                logging.info(colorize(
                    f"Split data: {len(train_split)} training samples, {len(valid_split)} validation samples",
                    color="cyan"
                ))
            
            return train_loader, valid_split
        
        elif isinstance(train_data, dict):
            # Get total length from any feature
            sample_key = list(train_data.keys())[0]
            total_length = len(train_data[sample_key])
            
            # Create indices and shuffle
            indices = np.arange(total_length)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            split_idx = int(total_length * (1 - validation_split))
            train_indices = indices[:split_idx]
            valid_indices = indices[split_idx:]
            
            # Split dict
            train_split = {}
            valid_split = {}
            for key, value in train_data.items():
                if isinstance(value, np.ndarray):
                    train_split[key] = value[train_indices]
                    valid_split[key] = value[valid_indices]
                elif isinstance(value, (list, tuple)):
                    value_array = np.array(value)
                    train_split[key] = value_array[train_indices].tolist()
                    valid_split[key] = value_array[valid_indices].tolist()
                elif isinstance(value, pd.Series):
                    train_split[key] = value.iloc[train_indices].values
                    valid_split[key] = value.iloc[valid_indices].values
                else:
                    train_split[key] = [value[i] for i in train_indices]
                    valid_split[key] = [value[i] for i in valid_indices]
            
            train_loader = self._prepare_data_loader(train_split, batch_size=batch_size, shuffle=shuffle)
            
            if self._verbose:
                logging.info(colorize(
                    f"Split data: {len(train_indices)} training samples, {len(valid_indices)} validation samples",
                    color="cyan"
                ))
            
            return train_loader, valid_split
        
        else:
            raise TypeError(f"Unsupported train_data type: {type(train_data)}")


    def compile(self, 
                optimizer = "adam",
                optimizer_params: dict | None = None,
                scheduler: str | torch.optim.lr_scheduler._LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | None = None,
                scheduler_params: dict | None = None,
                loss: str | nn.Module | list[str | nn.Module] | None= "bce"):
        if optimizer_params is None:
            optimizer_params = {}
        
        self._optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self._optimizer_params = optimizer_params
        if isinstance(scheduler, str):
            self._scheduler_name = scheduler
        elif scheduler is not None:
            # Try to get __name__ first (for class types), then __class__.__name__ (for instances)
            self._scheduler_name = getattr(scheduler, '__name__', getattr(scheduler.__class__, '__name__', str(scheduler)))
        else:
            self._scheduler_name = None
        self._scheduler_params = scheduler_params or {}
        self._loss_config = loss
        
        # set optimizer
        self.optimizer_fn = get_optimizer_fn(
            optimizer=optimizer, 
            params=self.parameters(), 
            **optimizer_params
        )
        
        # set loss functions
        if self.nums_task == 1:
            task_type = self.task if isinstance(self.task, str) else self.task[0]
            loss_value = loss[0] if isinstance(loss, list) else loss
            # For ranking and multitask, use pointwise training
            training_mode = 'pointwise' if self.task_type in ['ranking', 'multitask'] else None
            # Use task_type directly, not self.task_type for single task
            self.loss_fn = [get_loss_fn(task_type=task_type, training_mode=training_mode, loss=loss_value)]
        else:
            self.loss_fn = []
            for i in range(self.nums_task):
                task_type = self.task[i] if isinstance(self.task, list) else self.task
                
                if isinstance(loss, list):
                    loss_value = loss[i] if i < len(loss) else None
                else:
                    loss_value = loss
                
                # Multitask always uses pointwise training
                training_mode = 'pointwise'
                self.loss_fn.append(get_loss_fn(task_type=task_type, training_mode=training_mode, loss=loss_value))
        
        # set scheduler
        self.scheduler_fn = get_scheduler_fn(scheduler, self.optimizer_fn, **(scheduler_params or {})) if scheduler else None

    def compute_loss(self, y_pred, y_true):
        if y_true is None:
            return torch.tensor(0.0, device=self.device)
        
        if self.nums_task == 1:
            loss = self.loss_fn[0](y_pred, y_true)
            return loss
        
        else:
            task_losses = []
            for i in range(self.nums_task):
                task_loss = self.loss_fn[i](y_pred[:, i], y_true[:, i])
                task_losses.append(task_loss)
            return torch.stack(task_losses)


    def _prepare_data_loader(self, data: dict|pd.DataFrame|DataLoader, batch_size: int = 32, shuffle: bool = True):
        if isinstance(data, DataLoader):
            return data
        tensors = build_tensors_from_data(
            data=data,
            raw_data=data,
            features=self.all_features,
            target_columns=self.target,
            id_columns=getattr(self, "id_columns", []),
            on_missing_feature="raise",
        )
        assert tensors is not None, "No tensors were created from provided data."
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def _batch_to_dict(self, batch_data: tuple) -> dict:
        result = {}
        all_features = self.dense_features + self.sparse_features + self.sequence_features
        
        for i, feature in enumerate(all_features):
            if i < len(batch_data):
                result[feature.name] = batch_data[i]
        
        if len(batch_data) > len(all_features):
            labels = batch_data[-1]
            
            if self.nums_task == 1:
                result[self.target[0]] = labels
            else:
                if labels.dim() == 2 and labels.shape[1] == self.nums_task:
                    if len(self.target) == 1:
                        result[self.target[0]] = labels
                    else:
                        for i, target_name in enumerate(self.target):
                            if i < labels.shape[1]:
                                result[target_name] = labels[:, i]
                elif labels.dim() == 1:
                    result[self.target[0]] = labels
                else:
                    for i, target_name in enumerate(self.target):
                        if i < labels.shape[-1]:
                            result[target_name] = labels[..., i]
        
        return result


    def fit(self, 
            train_data: dict|pd.DataFrame|DataLoader, 
            valid_data: dict|pd.DataFrame|DataLoader|None=None, 
            metrics: list[str]|dict[str, list[str]]|None = None, # ['auc', 'logloss'] or {'target1': ['auc', 'logloss'], 'target2': ['mse']}
            epochs:int=1, verbose:int=1, shuffle:bool=True, batch_size:int=32,
            user_id_column: str = 'user_id',
            validation_split: float | None = None):

        self.to(self.device)
        if not self._logger_initialized:
            setup_logger(session_id=self.session_id)
            self._logger_initialized = True
        self._verbose = verbose
        self._set_metrics(metrics) # add self.metrics, self.task_specific_metrics, self.best_metrics_mode, self.early_stopper
        
        # Assert before training
        self._validate_task_configuration()
        
        if self._verbose:
            self.summary()

        # Handle validation_split parameter
        valid_loader = None
        if validation_split is not None and valid_data is None:
            train_loader, valid_data = self._handle_validation_split(
                train_data=train_data,
                validation_split=validation_split,
                batch_size=batch_size,
                shuffle=shuffle
            )
        else:
            if not isinstance(train_data, DataLoader):
                train_loader = self._prepare_data_loader(train_data, batch_size=batch_size, shuffle=shuffle)
            else:
                train_loader = train_data
        

        valid_user_ids: np.ndarray | None = None
        needs_user_ids = self._needs_user_ids_for_metrics()

        if valid_loader is None:
            if valid_data is not None and not isinstance(valid_data, DataLoader):
                valid_loader = self._prepare_data_loader(valid_data, batch_size=batch_size, shuffle=False)
                # Extract user_ids only if needed for GAUC
                if needs_user_ids:
                    if isinstance(valid_data, pd.DataFrame) and user_id_column in valid_data.columns:
                        valid_user_ids = np.asarray(valid_data[user_id_column].values)
                    elif isinstance(valid_data, dict) and user_id_column in valid_data:
                        valid_user_ids = np.asarray(valid_data[user_id_column])
            elif valid_data is not None:
                valid_loader = valid_data
        
        try:
            self._steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:
            self._steps_per_epoch = None
            is_streaming = True
        
        self._epoch_index = 0
        self._stop_training = False
        self._best_metric = float('-inf') if self.best_metrics_mode == 'max' else float('inf')
        
        if self._verbose:
            logging.info("")
            logging.info(colorize("=" * 80, color="bright_green", bold=True))
            if is_streaming:
                logging.info(colorize(f"Start training (Streaming Mode)", color="bright_green", bold=True))
            else:
                logging.info(colorize(f"Start training", color="bright_green", bold=True))
            logging.info(colorize("=" * 80, color="bright_green", bold=True))
            logging.info("")
            logging.info(colorize(f"Model device: {self.device}", color="bright_green"))
        
        for epoch in range(epochs):
            self._epoch_index = epoch
            
            # In streaming mode, print epoch header before progress bar
            if self._verbose and is_streaming:
                logging.info("")
                logging.info(colorize(f"Epoch {epoch + 1}/{epochs}", color="bright_green", bold=True))

            # Train with metrics computation
            train_result = self.train_epoch(train_loader, is_streaming=is_streaming, compute_metrics=True)
            
            # Unpack results
            if isinstance(train_result, tuple):
                train_loss, train_metrics = train_result
            else:
                train_loss = train_result
                train_metrics = None

            if self._verbose:
                if self.nums_task == 1:
                    log_str = f"Epoch {epoch + 1}/{epochs} - Train: loss={train_loss:.4f}"
                    if train_metrics:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
                        log_str += f", {metrics_str}"
                    logging.info(colorize(log_str, color="white"))
                else:
                    task_labels = []
                    for i in range(self.nums_task):
                        if i < len(self.target):
                            task_labels.append(self.target[i])
                        else:
                            task_labels.append(f"task_{i}")
                    
                    total_loss_val = np.sum(train_loss) if isinstance(train_loss, np.ndarray) else train_loss  # type: ignore
                    log_str = f"Epoch {epoch + 1}/{epochs} - Train: loss={total_loss_val:.4f}"
                    
                    if train_metrics:
                        # Group metrics by task
                        task_metrics = {}
                        for metric_key, metric_value in train_metrics.items():
                            for target_name in self.target:
                                if metric_key.endswith(f"_{target_name}"):
                                    if target_name not in task_metrics:
                                        task_metrics[target_name] = {}
                                    metric_name = metric_key.rsplit(f"_{target_name}", 1)[0]
                                    task_metrics[target_name][metric_name] = metric_value
                                    break
                        
                        if task_metrics:
                            task_metric_strs = []
                            for target_name in self.target:
                                if target_name in task_metrics:
                                    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in task_metrics[target_name].items()])
                                    task_metric_strs.append(f"{target_name}[{metrics_str}]")
                            log_str += ", " + ", ".join(task_metric_strs)
                    
                    logging.info(colorize(log_str, color="white"))
            
            if valid_loader is not None:
                # Pass user_ids only if needed for GAUC metric
                val_metrics = self.evaluate(valid_loader, user_ids=valid_user_ids if needs_user_ids else None) # {'auc': 0.75, 'logloss': 0.45} or {'auc_target1': 0.75, 'logloss_target1': 0.45, 'mse_target2': 3.2}
            
                if self._verbose:
                    if self.nums_task == 1:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                        logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - Valid: {metrics_str}", color="cyan"))
                    else:
                        # multi task metrics
                        task_metrics = {}
                        for metric_key, metric_value in val_metrics.items():
                            for target_name in self.target:
                                if metric_key.endswith(f"_{target_name}"):
                                    if target_name not in task_metrics:
                                        task_metrics[target_name] = {}
                                    metric_name = metric_key.rsplit(f"_{target_name}", 1)[0]
                                    task_metrics[target_name][metric_name] = metric_value
                                    break

                        task_metric_strs = []
                        for target_name in self.target:
                            if target_name in task_metrics:
                                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in task_metrics[target_name].items()])
                                task_metric_strs.append(f"{target_name}[{metrics_str}]")
                        
                        logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - Valid: " + ", ".join(task_metric_strs), color="cyan"))
                
                # Handle empty validation metrics
                if not val_metrics:
                    if self._verbose:
                        logging.info(colorize(f"Warning: No validation metrics computed. Skipping validation for this epoch.", color="yellow"))
                    continue
                
                if self.nums_task == 1:
                    primary_metric_key = self.metrics[0]
                else:
                    primary_metric_key = f"{self.metrics[0]}_{self.target[0]}"
                
                primary_metric = val_metrics.get(primary_metric_key, val_metrics[list(val_metrics.keys())[0]])
                improved = False
                
                if self.best_metrics_mode == 'max':
                    if primary_metric > self._best_metric:
                        self._best_metric = primary_metric
                        self.save_weights(self.best)
                        improved = True
                else:
                    if primary_metric < self._best_metric:
                        self._best_metric = primary_metric
                        improved = True
                
                if improved:
                    if self._verbose:
                        logging.info(colorize(f"Validation {primary_metric_key} improved to {self._best_metric:.4f}", color="yellow"))
                    self.save_weights(self.checkpoint)
                    self.early_stopper.trial_counter = 0
                else:
                    self.early_stopper.trial_counter += 1
                    if self._verbose:
                        logging.info(colorize(f"No improvement for {self.early_stopper.trial_counter} epoch(s)", color="yellow"))
                
                if self.early_stopper.trial_counter >= self.early_stopper.patience:
                    self._stop_training = True
                    if self._verbose:
                        logging.info(colorize(f"Early stopping triggered after {epoch + 1} epochs", color="bright_red", bold=True))
                    break
            else:
                self.save_weights(self.checkpoint)
            
            if self._stop_training:
                break
            
            if self.scheduler_fn is not None:
                if isinstance(self.scheduler_fn, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loader is not None:
                        self.scheduler_fn.step(primary_metric)
                else:
                    self.scheduler_fn.step()
                    
        if self._verbose:
            logging.info("\n")
            logging.info(colorize("Training finished.", color="bright_green", bold=True))
            logging.info("\n")
        
        if valid_loader is not None:
            if self._verbose:
                logging.info(colorize(f"Load best model from: {self.checkpoint}", color="bright_blue"))
            self.load_weights(self.checkpoint)
        
        return self

    def train_epoch(self, train_loader: DataLoader, is_streaming: bool = False, compute_metrics: bool = False) -> Union[float, np.ndarray, tuple[Union[float, np.ndarray], dict]]:
        if self.nums_task == 1:
            accumulated_loss = 0.0
        else:
            accumulated_loss = np.zeros(self.nums_task, dtype=np.float64)
        
        self.train()
        num_batches = 0
        
        # Lists to store predictions and labels for metric computation
        y_true_list = []
        y_pred_list = []

        if self._verbose:
            # For streaming datasets without known length, set total=None to show progress without percentage
            if self._steps_per_epoch is not None:
                batch_iter = enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {self._epoch_index + 1}", total=self._steps_per_epoch))
            else:
                # Streaming mode: show batch/file progress without epoch in desc
                if is_streaming:
                    batch_iter = enumerate(tqdm.tqdm(
                        train_loader, 
                        desc="Batches", 
                        # position=1,
                        # leave=False,
                        # unit="batch"
                    ))
                else:
                    batch_iter = enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {self._epoch_index + 1}"))
        else:
            batch_iter = enumerate(train_loader)

        for batch_index, batch_data in batch_iter:
            batch_dict = self._batch_to_dict(batch_data)
            X_input, y_true = self.get_input(batch_dict)

            y_pred = self.forward(X_input)
            loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()

            if self.nums_task == 1:
                total_loss = loss + reg_loss
            else:
                total_loss = loss.sum() + reg_loss
            
            self.optimizer_fn.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer_fn.step()
            
            if self.nums_task == 1:
                accumulated_loss += loss.item()
            else:
                accumulated_loss += loss.detach().cpu().numpy()
            
            # Collect predictions and labels for metrics if requested
            if compute_metrics:
                if y_true is not None:
                    y_true_list.append(y_true.detach().cpu().numpy())
                # For pairwise/listwise mode, y_pred is a tuple of embeddings, skip metric collection during training
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.detach().cpu().numpy())
            
            num_batches += 1

        if self.nums_task == 1:
            avg_loss = accumulated_loss / num_batches
        else:
            avg_loss = accumulated_loss / num_batches
        
        # Compute metrics if requested
        if compute_metrics and len(y_true_list) > 0 and len(y_pred_list) > 0:
            y_true_all = np.concatenate(y_true_list, axis=0)
            y_pred_all = np.concatenate(y_pred_list, axis=0)
            metrics_dict = self.evaluate_metrics(y_true_all, y_pred_all, self.metrics, user_ids=None)
            return avg_loss, metrics_dict
        
        return avg_loss


    def _needs_user_ids_for_metrics(self) -> bool:
        """Check if any configured metric requires user_ids (e.g., gauc)."""
        all_metrics = set()
        
        # Collect all metrics from different sources
        if hasattr(self, 'metrics') and self.metrics:
            all_metrics.update(m.lower() for m in self.metrics)
        
        if hasattr(self, 'task_specific_metrics') and self.task_specific_metrics:
            for task_metrics in self.task_specific_metrics.values():
                if isinstance(task_metrics, list):
                    all_metrics.update(m.lower() for m in task_metrics)
        
        # Check if gauc is in any of the metrics
        return 'gauc' in all_metrics

    def evaluate(self, 
                 data: dict | pd.DataFrame | DataLoader, 
                 metrics: list[str] | dict[str, list[str]] | None = None,
                 batch_size: int = 32,
                 user_ids: np.ndarray | None = None,
                 user_id_column: str = 'user_id') -> dict:
        """
        Evaluate the model on validation data.
        
        Args:
            data: Evaluation data (dict, DataFrame, or DataLoader)
            metrics: Optional metrics to use for evaluation. If None, uses metrics from fit()
            batch_size: Batch size for evaluation (only used if data is dict or DataFrame)
            user_ids: Optional user IDs for computing GAUC metric. If None and gauc is needed,
                     will try to extract from data using user_id_column
            user_id_column: Column name for user IDs (default: 'user_id')
            
        Returns:
            Dictionary of metric values
        """
        self.eval()
        
        # Use provided metrics or fall back to configured metrics
        eval_metrics = metrics if metrics is not None else self.metrics
        if eval_metrics is None:
            raise ValueError("No metrics specified for evaluation. Please provide metrics parameter or call fit() first.")
        
        # Prepare DataLoader if needed
        if isinstance(data, DataLoader):
            data_loader = data
            # Try to extract user_ids from original data if needed
            if user_ids is None and self._needs_user_ids_for_metrics():
                # Cannot extract user_ids from DataLoader, user must provide them
                if self._verbose:
                    logging.warning(colorize(
                        "GAUC metric requires user_ids, but data is a DataLoader. "
                        "Please provide user_ids parameter or use dict/DataFrame format.",
                        color="yellow"
                    ))
        else:
            # Extract user_ids if needed and not provided
            if user_ids is None and self._needs_user_ids_for_metrics():
                if isinstance(data, pd.DataFrame) and user_id_column in data.columns:
                    user_ids = np.asarray(data[user_id_column].values)
                elif isinstance(data, dict) and user_id_column in data:
                    user_ids = np.asarray(data[user_id_column])
            
            data_loader = self._prepare_data_loader(data, batch_size=batch_size, shuffle=False)
        
        y_true_list = []
        y_pred_list = []
        
        batch_count = 0
        with torch.no_grad():
            for batch_data in data_loader:
                batch_count += 1
                batch_dict = self._batch_to_dict(batch_data)
                X_input, y_true = self.get_input(batch_dict)
                y_pred = self.forward(X_input)
                
                if y_true is not None:
                    y_true_list.append(y_true.cpu().numpy())
                # Skip if y_pred is not a tensor (e.g., tuple in pairwise mode, though this shouldn't happen in eval mode)
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.cpu().numpy())

        if self._verbose:
            logging.info(colorize(f"  Evaluation batches processed: {batch_count}", color="cyan"))
        
        if len(y_true_list) > 0:
            y_true_all = np.concatenate(y_true_list, axis=0)
            if self._verbose:
                logging.info(colorize(f"  Evaluation samples: {y_true_all.shape[0]}", color="cyan"))
        else:
            y_true_all = None
            if self._verbose:
                logging.info(colorize(f"  Warning: No y_true collected from evaluation data", color="yellow"))
            
        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
        else:
            y_pred_all = None
            if self._verbose:
                logging.info(colorize(f"  Warning: No y_pred collected from evaluation data", color="yellow"))
        
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
        
        metrics_dict = self.evaluate_metrics(y_true_all, y_pred_all, metrics_to_use, user_ids)
        
        return metrics_dict


    def evaluate_metrics(self, y_true: np.ndarray|None, y_pred: np.ndarray|None, metrics: list[str], user_ids: np.ndarray|None = None) -> dict:
        """Evaluate metrics using the metrics module."""
        task_specific_metrics = getattr(self, 'task_specific_metrics', None)
        
        return evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            task=self.task,
            target_names=self.target,
            task_specific_metrics=task_specific_metrics,
            user_ids=user_ids
        )


    def predict(self, 
                data: str|dict|pd.DataFrame|DataLoader, 
                batch_size: int = 32,
                save_path: str | os.PathLike | None = None,
                save_format: Literal["npy", "csv"] = "npy") -> np.ndarray:
        self.eval()
        # todo: handle file path input later
        if isinstance(data, (str, os.PathLike)):
            pass
        if not isinstance(data, DataLoader):
            data_loader = self._prepare_data_loader(data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        y_pred_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Predicting", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                X_input, _ = self.get_input(batch_dict)
                y_pred = self.forward(X_input)

                if y_pred is not None:
                    y_pred_list.append(y_pred.cpu().numpy())

        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
        else:
            y_pred_all = np.array([])

        if save_path is not None:
            suffix = ".npy" if save_format == "npy" else ".csv"
            target_path = resolve_save_path(
                path=save_path,
                default_dir=self.session.predictions_dir,
                default_name="predictions",
                suffix=suffix,
                add_timestamp=True if save_path is None else False,
            )

            if save_format == "npy":
                np.save(target_path, y_pred_all)
            else:
                pd.DataFrame(y_pred_all).to_csv(target_path, index=False)

            if self._verbose:
                logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))

        return y_pred_all
    
    def save_weights(self, model_path: str | os.PathLike | None):
        target_path = resolve_save_path(
            path=model_path,
            default_dir=self.session.checkpoints_dir / self.model_name,
            default_name=self.model_name,
            suffix=".model",
            add_timestamp=model_path is None,
        )
        torch.save(self.state_dict(), target_path)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def summary(self):
        logger = logging.getLogger()
        
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        logger.info(colorize(f"Model Summary: {self.model_name}", color="bright_blue", bold=True))
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        
        logger.info("")
        logger.info(colorize("[1] Feature Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        if self.dense_features:
            logger.info(f"Dense Features ({len(self.dense_features)}):")
            for i, feat in enumerate(self.dense_features, 1):
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 1
                logger.info(f"  {i}. {feat.name:20s}")
        
        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sparse_features)
            max_embed_name_len = max(len(feat.embedding_name) for feat in self.sparse_features)
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10}")
            for i, feat in enumerate(self.sparse_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, 'vocab_size') else 'N/A'
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 'N/A'
                logger.info(f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10}")
        
        if self.sequence_features:
            logger.info(f"Sequence Features ({len(self.sequence_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sequence_features)
            max_embed_name_len = max(len(feat.embedding_name) for feat in self.sequence_features)
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10} {'Max Len':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10} {'-'*10}")
            for i, feat in enumerate(self.sequence_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, 'vocab_size') else 'N/A'
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 'N/A'
                max_len = feat.max_len if hasattr(feat, 'max_len') else 'N/A'
                logger.info(f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10} {str(max_len):>10}")
        
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
        logger.info(f"Target Columns:          {self.target}")
        logger.info(f"Device:                  {self.device}")
        
        if hasattr(self, '_optimizer_name'):
            logger.info(f"Optimizer:               {self._optimizer_name}")
            if self._optimizer_params:
                for key, value in self._optimizer_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, '_scheduler_name') and self._scheduler_name:
            logger.info(f"Scheduler:               {self._scheduler_name}")
            if self._scheduler_params:
                for key, value in self._scheduler_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, '_loss_config'):
            logger.info(f"Loss Function:           {self._loss_config}")
        
        logger.info("Regularization:")
        logger.info(f"  Embedding L1:          {self._embedding_l1_reg}")
        logger.info(f"  Embedding L2:          {self._embedding_l2_reg}")
        logger.info(f"  Dense L1:              {self._dense_l1_reg}")
        logger.info(f"  Dense L2:              {self._dense_l2_reg}")
        
        logger.info("Other Settings:")
        logger.info(f"  Early Stop Patience:   {self.early_stop_patience}")
        logger.info(f"  Max Gradient Norm:     {self._max_gradient_norm}")
        logger.info(f"  Session ID:            {self.session_id}")
        logger.info(f"  Checkpoint Path:       {self.checkpoint}")
        
        logger.info("")
        logger.info("")


class BaseMatchModel(BaseModel):
    """
    Base class for match (retrieval/recall) models
    Supports pointwise, pairwise, and listwise training modes
    """
    
    @property
    def task_type(self) -> str:
        return 'match'
    
    @property
    def support_training_modes(self) -> list[str]:
        """
        Returns list of supported training modes for this model.
        Override in subclasses to restrict training modes.
        
        Returns:
            List of supported modes: ['pointwise', 'pairwise', 'listwise']
        """
        return ['pointwise', 'pairwise', 'listwise']
    
    def __init__(self,
                 user_dense_features: list[DenseFeature] | None = None,
                 user_sparse_features: list[SparseFeature] | None = None,
                 user_sequence_features: list[SequenceFeature] | None = None,
                 item_dense_features: list[DenseFeature] | None = None,
                 item_sparse_features: list[SparseFeature] | None = None,
                 item_sequence_features: list[SequenceFeature] | None = None,
                 training_mode: Literal['pointwise', 'pairwise', 'listwise'] = 'pointwise',
                 num_negative_samples: int = 4,
                 temperature: float = 1.0,
                 similarity_metric: Literal['dot', 'cosine', 'euclidean'] = 'dot',
                 device: str = 'cpu',
                 embedding_l1_reg: float = 0.0,
                 dense_l1_reg: float = 0.0,
                 embedding_l2_reg: float = 0.0,
                 dense_l2_reg: float = 0.0,
                 early_stop_patience: int = 20,
                 **kwargs):
        
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
            target=['label'],  
            task='binary',  
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            **kwargs
        )
        
        self.user_dense_features = list(user_dense_features) if user_dense_features else []
        self.user_sparse_features = list(user_sparse_features) if user_sparse_features else []
        self.user_sequence_features = list(user_sequence_features) if user_sequence_features else []
        
        self.item_dense_features = list(item_dense_features) if item_dense_features else []
        self.item_sparse_features = list(item_sparse_features) if item_sparse_features else []
        self.item_sequence_features = list(item_sequence_features) if item_sequence_features else []
        
        self.training_mode = training_mode
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric
    
    def get_user_features(self, X_input: dict) -> dict:
        user_input = {}
        all_user_features = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        for feature in all_user_features:
            if feature.name in X_input:
                user_input[feature.name] = X_input[feature.name]
        return user_input
    
    def get_item_features(self, X_input: dict) -> dict:
        item_input = {}
        all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        for feature in all_item_features:
            if feature.name in X_input:
                item_input[feature.name] = X_input[feature.name]
        return item_input
    
    def compile(self, 
                optimizer = "adam",
                optimizer_params: dict | None = None,
                scheduler: str | torch.optim.lr_scheduler._LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | None = None,
                scheduler_params: dict | None = None,
                loss: str | nn.Module | list[str | nn.Module] | None= None):
        """
        Compile match model with optimizer, scheduler, and loss function.
        Validates that training_mode is supported by the model.
        """
        from nextrec.loss import validate_training_mode
        
        # Validate training mode is supported
        validate_training_mode(
            training_mode=self.training_mode,
            support_training_modes=self.support_training_modes,
            model_name=self.model_name
        )
        
        # Call parent compile with match-specific logic
        if optimizer_params is None:
            optimizer_params = {}
        
        self._optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self._optimizer_params = optimizer_params
        if isinstance(scheduler, str):
            self._scheduler_name = scheduler
        elif scheduler is not None:
            # Try to get __name__ first (for class types), then __class__.__name__ (for instances)
            self._scheduler_name = getattr(scheduler, '__name__', getattr(scheduler.__class__, '__name__', str(scheduler)))
        else:
            self._scheduler_name = None
        self._scheduler_params = scheduler_params or {}
        self._loss_config = loss
        
        # set optimizer
        self.optimizer_fn = get_optimizer_fn(
            optimizer=optimizer, 
            params=self.parameters(), 
            **optimizer_params
        )
        
        # Set loss function based on training mode
        loss_value = loss[0] if isinstance(loss, list) else loss
        self.loss_fn = [get_loss_fn(
            task_type='match',
            training_mode=self.training_mode,
            loss=loss_value
        )]
        
        # set scheduler
        self.scheduler_fn = get_scheduler_fn(scheduler, self.optimizer_fn, **(scheduler_params or {})) if scheduler else None

    def compute_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == 'dot':
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                # [batch_size, num_items, emb_dim] @ [batch_size, num_items, emb_dim]
                similarity = torch.sum(user_emb * item_emb, dim=-1)  # [batch_size, num_items]
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                # [batch_size, emb_dim] @ [batch_size, num_items, emb_dim]
                user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
                similarity = torch.sum(user_emb_expanded * item_emb, dim=-1)  # [batch_size, num_items]
            else:
                similarity = torch.sum(user_emb * item_emb, dim=-1)  # [batch_size]
        
        elif self.similarity_metric == 'cosine':
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb_expanded = user_emb.unsqueeze(1)
                similarity = F.cosine_similarity(user_emb_expanded, item_emb, dim=-1)
            else:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        
        elif self.similarity_metric == 'euclidean':
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
    
    def forward(self, X_input: dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        user_input = self.get_user_features(X_input)
        item_input = self.get_item_features(X_input)
        
        user_emb = self.user_tower(user_input)   # [B, D]
        item_emb = self.item_tower(item_input)   # [B, D]
        
        if self.training and self.training_mode in ['pairwise', 'listwise']:
            return user_emb, item_emb

        similarity = self.compute_similarity(user_emb, item_emb)  # [B]
        
        if self.training_mode == 'pointwise':
            return torch.sigmoid(similarity)
        else:
            return similarity
    
    def compute_loss(self, y_pred, y_true):
        if self.training_mode == 'pointwise':
            if y_true is None:
                return torch.tensor(0.0, device=self.device)
            return self.loss_fn[0](y_pred, y_true)
        
        # pairwise / listwise using inbatch neg
        elif self.training_mode in ['pairwise', 'listwise']:
            if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
                raise ValueError(
                    "For pairwise/listwise training, forward should return (user_emb, item_emb). "
                    "Please check BaseMatchModel.forward implementation."
                )
            
            user_emb, item_emb = y_pred  # [B, D], [B, D]
            
            logits = torch.matmul(user_emb, item_emb.t())  # [B, B]
            logits = logits / self.temperature            
            
            batch_size = logits.size(0)
            targets = torch.arange(batch_size, device=logits.device)  # [0, 1, 2, ..., B-1]
            
            # Cross-Entropy = InfoNCE
            loss = F.cross_entropy(logits, targets)
            return loss
        
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _set_metrics(self, metrics: list[str] | None = None):
        if metrics is not None and len(metrics) > 0:
            self.metrics = [m.lower() for m in metrics]
        else:
            self.metrics = ['auc', 'logloss']
        
        self.best_metrics_mode = 'max'  
        
        if not hasattr(self, 'early_stopper') or self.early_stopper is None:
            self.early_stopper = EarlyStopper(patience=self.early_stop_patience, mode=self.best_metrics_mode)
    
    def encode_user(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        
        if not isinstance(data, DataLoader):
            user_data = {}
            all_user_features = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
            for feature in all_user_features:
                if isinstance(data, dict):
                    if feature.name in data:
                        user_data[feature.name] = data[feature.name]
                elif isinstance(data, pd.DataFrame):
                    if feature.name in data.columns:
                        user_data[feature.name] = data[feature.name].values
            
            data_loader = self._prepare_data_loader(user_data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding users", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                user_input = self.get_user_features(batch_dict)
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings
    
    def encode_item(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        
        if not isinstance(data, DataLoader):
            item_data = {}
            all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
            for feature in all_item_features:
                if isinstance(data, dict):
                    if feature.name in data:
                        item_data[feature.name] = data[feature.name]
                elif isinstance(data, pd.DataFrame):
                    if feature.name in data.columns:
                        item_data[feature.name] = data[feature.name].values

            data_loader = self._prepare_data_loader(item_data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding items", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                item_input = self.get_item_features(batch_dict)
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings
