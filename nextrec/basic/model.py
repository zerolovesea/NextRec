"""
Base Model & Base Match Model Class

Date: create on 27/10/2025
Checkpoint: edit on 02/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import os
import tqdm
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Union, Literal, Any
from torch.utils.data import DataLoader

from nextrec.basic.callback import EarlyStopper
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature, FeatureSet
from nextrec.data.dataloader import TensorDictDataset, RecDataLoader

from nextrec.basic.loggers import setup_logger, colorize
from nextrec.basic.session import resolve_save_path, create_session
from nextrec.basic.metrics import configure_metrics, evaluate_metrics, check_user_id

from nextrec.data.dataloader import build_tensors_from_data
from nextrec.data.data_utils import get_column_data, collate_fn, batch_to_dict, get_user_ids

from nextrec.loss import get_loss_fn, get_loss_kwargs
from nextrec.utils import get_optimizer, get_scheduler, to_tensor

from nextrec import __version__

class BaseModel(FeatureSet, nn.Module):
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
            logging.warning("[BaseModel Warning] Invalid device , defaulting to CPU.")
            self.device = torch.device('cpu')

        self.session_id = session_id
        self.session = create_session(session_id)
        self.session_path = self.session.root # pwd/session_id, path for this session
        self.checkpoint_path = os.path.join(self.session_path, self.model_name+"_checkpoint.model") # example: pwd/session_id/DeepFM_checkpoint.model
        self.best_path = os.path.join(self.session_path, self.model_name+"_best.model")
        self.features_config_path = os.path.join(self.session_path, "features_config.pkl")
        self.set_all_features(dense_features, sparse_features, sequence_features, target, id_columns)

        self.task = task
        self.nums_task = len(task) if isinstance(task, list) else 1

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

    def register_regularization_weights(self, embedding_attr: str = "embedding", exclude_modules: list[str] | None = None, include_modules: list[str] | None = None) -> None:
        exclude_modules = exclude_modules or []
        include_modules = include_modules or []
        embedding_layer = getattr(self, embedding_attr, None)
        embed_dict = getattr(embedding_layer, "embed_dict", None)
        if embed_dict is not None:
            self.embedding_params.extend(embed.weight for embed in embed_dict.values())
        skip_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,nn.Dropout, nn.Dropout2d, nn.Dropout3d,)
        for name, module in self.named_modules():
            if (module is self or embedding_attr in name or isinstance(module, skip_types) or (include_modules and not any(inc in name for inc in include_modules)) or any(exc in name for exc in exclude_modules)):
                continue
            if isinstance(module, nn.Linear):
                self.regularization_weights.append(module.weight)

    def add_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.embedding_params:
            if self.embedding_l1_reg > 0:
                reg_loss += self.embedding_l1_reg * sum(param.abs().sum() for param in self.embedding_params)
            if self.embedding_l2_reg > 0:
                reg_loss += self.embedding_l2_reg * sum((param ** 2).sum() for param in self.embedding_params)
        if self.regularization_weights:
            if self.dense_l1_reg > 0:
                reg_loss += self.dense_l1_reg * sum(param.abs().sum() for param in self.regularization_weights)
            if self.dense_l2_reg > 0:
                reg_loss += self.dense_l2_reg * sum((param ** 2).sum() for param in self.regularization_weights)
        return reg_loss

    def get_input(self, input_data: dict, require_labels: bool = True):
        feature_source = input_data.get("features", {})
        label_source = input_data.get("labels")
        X_input = {}
        for feature in self.all_features:
            if feature.name not in feature_source:
                raise KeyError(f"[BaseModel-input Error] Feature '{feature.name}' not found in input data.")
            feature_data = get_column_data(feature_source, feature.name)
            X_input[feature.name] = to_tensor(feature_data, dtype=torch.float32 if isinstance(feature, DenseFeature) else torch.long, device=self.device)
        y = None
        if (len(self.target_columns) > 0 and (require_labels or (label_source and any(name in label_source for name in self.target_columns)))): # need labels: training or eval with labels
            target_tensors = []
            for target_name in self.target_columns:
                if label_source is None or target_name not in label_source:
                    if require_labels:
                        raise KeyError(f"[BaseModel-input Error] Target column '{target_name}' not found in input data.")
                    continue
                target_data = get_column_data(label_source, target_name)
                if target_data is None:
                    if require_labels:
                        raise ValueError(f"[BaseModel-input Error] Target column '{target_name}' contains no data.")
                    continue
                target_tensor = to_tensor(target_data, dtype=torch.float32, device=self.device)
                target_tensor = target_tensor.view(target_tensor.size(0), -1)
                target_tensors.append(target_tensor)
            if target_tensors:
                y = torch.cat(target_tensors, dim=1)
                if y.shape[1] == 1:
                    y = y.view(-1)
            elif require_labels:
                raise ValueError("[BaseModel-input Error] Labels are required but none were found in the input batch.")
        return X_input, y

    def handle_validation_split(self, train_data: dict | pd.DataFrame, validation_split: float, batch_size: int, shuffle: bool,) -> tuple[DataLoader, dict | pd.DataFrame]:
        """This function will split training data into training and validation sets when: 1. valid_data is None; 2. validation_split is provided."""
        if not (0 < validation_split < 1):
            raise ValueError(f"[BaseModel-validation Error] validation_split must be between 0 and 1, got {validation_split}")
        if not isinstance(train_data, (pd.DataFrame, dict)):
            raise TypeError(f"[BaseModel-validation Error] train_data must be a pandas DataFrame or a dict, got {type(train_data)}")
        if isinstance(train_data, pd.DataFrame):
            total_length = len(train_data)
        else:
            sample_key = next(iter(train_data)) # pick the first key to check length, for example: 'user_id': [1,2,3,4,5]
            total_length = len(train_data[sample_key]) # len(train_data['user_id'])
            for k, v in train_data.items():
                if len(v) != total_length:
                    raise ValueError(f"[BaseModel-validation Error] Length of field '{k}' ({len(v)}) != length of field '{sample_key}' ({total_length})")
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
        train_loader = self.prepare_data_loader(train_split, batch_size=batch_size, shuffle=shuffle)
        logging.info(f"Split data: {len(train_indices)} training samples, {len(valid_indices)} validation samples")
        return train_loader, valid_split

    def compile(
        self,
        optimizer: str | torch.optim.Optimizer = "adam",
        optimizer_params: dict | None = None,
        scheduler: str | torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        loss_weights: int | float | list[int | float] | None = None,
    ):
        optimizer_params = optimizer_params or {}
        self.optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self.optimizer_params = optimizer_params
        self.optimizer_fn = get_optimizer(optimizer=optimizer, params=self.parameters(), **optimizer_params,)

        scheduler_params = scheduler_params or {}
        if isinstance(scheduler, str):
            self.scheduler_name = scheduler
        elif scheduler is None:
            self.scheduler_name = None
        else: # for custom scheduler instance, need to provide class name for logging
            self.scheduler_name = getattr(scheduler, "__name__", scheduler.__class__.__name__) # type: ignore
        self.scheduler_params = scheduler_params
        self.scheduler_fn = (get_scheduler(scheduler, self.optimizer_fn, **scheduler_params) if scheduler else None)

        self.loss_config = loss
        self.loss_params = loss_params or {}
        self.loss_fn = []
        if isinstance(loss, list): # for example: ['bce', 'mse'] -> ['bce', 'mse']
            loss_list = [loss[i] if i < len(loss) else None for i in range(self.nums_task)]
        else: # for example: 'bce' -> ['bce', 'bce']
            loss_list = [loss] * self.nums_task

        if isinstance(self.loss_params, dict):
            params_list = [self.loss_params] * self.nums_task
        else:  # list[dict]
            params_list = [self.loss_params[i] if i < len(self.loss_params) else {} for i in range(self.nums_task)]
        self.loss_fn = [get_loss_fn(loss=loss_list[i], **params_list[i]) for i in range(self.nums_task)]

        if loss_weights is None:
            self.loss_weights = None
        elif self.nums_task == 1:
            if isinstance(loss_weights, (list, tuple)):
                if len(loss_weights) != 1 and isinstance(loss_weights, (list, tuple)):
                    raise ValueError("[BaseModel-compile Error] loss_weights list must have exactly one element for single-task setup.")
                weight_value = loss_weights[0]
            else:
                weight_value = loss_weights
            self.loss_weights = float(weight_value)
        else:
            if isinstance(loss_weights, (int, float)):
                weights = [float(loss_weights)] * self.nums_task
            elif isinstance(loss_weights, (list, tuple)):
                weights = [float(w) for w in loss_weights]
                if len(weights) != self.nums_task:
                    raise ValueError(f"[BaseModel-compile Error] Number of loss_weights ({len(weights)}) must match number of tasks ({self.nums_task}).")
            else:
                raise TypeError(f"[BaseModel-compile Error] loss_weights must be int, float, list or tuple, got {type(loss_weights)}")
            self.loss_weights = weights

    def compute_loss(self, y_pred, y_true):
        if y_true is None:
            raise ValueError("[BaseModel-compute_loss Error] Ground truth labels (y_true) are required to compute loss.")
        if self.nums_task == 1:
            loss = self.loss_fn[0](y_pred, y_true)
            if self.loss_weights is not None:
                loss = loss * self.loss_weights
            return loss
        else:
            task_losses = []
            for i in range(self.nums_task):
                task_loss = self.loss_fn[i](y_pred[:, i], y_true[:, i])
                if isinstance(self.loss_weights, (list, tuple)):
                    task_loss = task_loss * self.loss_weights[i]
                task_losses.append(task_loss)
            return torch.stack(task_losses).sum()

    def prepare_data_loader(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 32, shuffle: bool = True,):
        if isinstance(data, DataLoader):
            return data
        tensors = build_tensors_from_data(data=data, raw_data=data, features=self.all_features, target_columns=self.target_columns, id_columns=self.id_columns,)
        if tensors is None:
            raise ValueError("[BaseModel-prepare_data_loader Error] No data available to create DataLoader.")
        dataset = TensorDictDataset(tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def fit(self, 
            train_data: dict | pd.DataFrame | DataLoader, 
            valid_data: dict | pd.DataFrame | DataLoader | None = None, 
            metrics: list[str] | dict[str, list[str]] | None = None, # ['auc', 'logloss'] or {'target1': ['auc', 'logloss'], 'target2': ['mse']}
            epochs:int=1, shuffle:bool=True, batch_size:int=32,
            user_id_column: str | None = None,
            validation_split: float | None = None):
        self.to(self.device)
        if not self.logger_initialized:
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True

        self.metrics, self.task_specific_metrics, self.best_metrics_mode = configure_metrics(task=self.task, metrics=metrics, target_names=self.target_columns) # ['auc', 'logloss'], {'target1': ['auc', 'logloss'], 'target2': ['mse']}, 'max'
        self.early_stopper = EarlyStopper(patience=self.early_stop_patience, mode=self.best_metrics_mode)
        self.needs_user_ids = check_user_id(self.metrics, self.task_specific_metrics) # check user_id needed for GAUC metrics    
        self.epoch_index = 0
        self.stop_training = False
        self.best_checkpoint_path = self.best_path
        self.best_metric = float('-inf') if self.best_metrics_mode == 'max' else float('inf')

        if validation_split is not None and valid_data is None:
            train_loader, valid_data = self.handle_validation_split(train_data=train_data, validation_split=validation_split, batch_size=batch_size, shuffle=shuffle,) # type: ignore
        else:
            train_loader = (train_data if isinstance(train_data, DataLoader) else self.prepare_data_loader(train_data, batch_size=batch_size, shuffle=shuffle))
        
        valid_loader, valid_user_ids = self.prepare_validation_data(valid_data=valid_data, batch_size=batch_size, needs_user_ids=self.needs_user_ids, user_id_column=user_id_column)
        try:
            self.steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError: # streaming data loader does not supported len()
            self.steps_per_epoch = None
            is_streaming = True

        self.summary()
        logging.info("")
        logging.info(colorize("=" * 80, bold=True))
        if is_streaming:
            logging.info(colorize(f"Start streaming training", bold=True))
        else:
            logging.info(colorize(f"Start training", bold=True))
        logging.info(colorize("=" * 80, bold=True))
        logging.info("")
        logging.info(colorize(f"Model device: {self.device}", bold=True))
    
        for epoch in range(epochs):
            self.epoch_index = epoch
            if is_streaming:
                logging.info("")
                logging.info(colorize(f"Epoch {epoch + 1}/{epochs}", bold=True)) # streaming mode, print epoch header before progress bar

            # handle train result
            train_result = self.train_epoch(train_loader, is_streaming=is_streaming) 
            if isinstance(train_result, tuple): # [avg_loss, metrics_dict]
                train_loss, train_metrics = train_result
            else:
                train_loss = train_result
                train_metrics = None
            
            # handle logging for single-task and multi-task
            if self.nums_task == 1:
                log_str = f"Epoch {epoch + 1}/{epochs} - Train: loss={train_loss:.4f}"
                if train_metrics:
                    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
                    log_str += f", {metrics_str}"
                logging.info(colorize(log_str))
            else:
                total_loss_val = np.sum(train_loss) if isinstance(train_loss, np.ndarray) else train_loss  # type: ignore
                log_str = f"Epoch {epoch + 1}/{epochs} - Train: loss={total_loss_val:.4f}"
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
                                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in task_metrics[target_name].items()])
                                task_metric_strs.append(f"{target_name}[{metrics_str}]")
                        log_str += ", " + ", ".join(task_metric_strs)
                logging.info(colorize(log_str))
            if valid_loader is not None:
                # pass user_ids only if needed for GAUC metric
                val_metrics = self.evaluate(valid_loader, user_ids=valid_user_ids if self.needs_user_ids else None) # {'auc': 0.75, 'logloss': 0.45} or {'auc_target1': 0.75, 'logloss_target1': 0.45, 'mse_target2': 3.2}
                if self.nums_task == 1:
                    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                    logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - Valid: {metrics_str}", color="cyan"))
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
                            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in task_metrics[target_name].items()])
                            task_metric_strs.append(f"{target_name}[{metrics_str}]")
                    logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - Valid: " + ", ".join(task_metric_strs), color="cyan"))
                # Handle empty validation metrics
                if not val_metrics:
                    self.save_model(self.checkpoint_path, add_timestamp=False, verbose=False)
                    self.best_checkpoint_path = self.checkpoint_path
                    logging.info(colorize(f"Warning: No validation metrics computed. Skipping validation for this epoch.", color="yellow"))
                    continue
                if self.nums_task == 1:
                    primary_metric_key = self.metrics[0]
                else:
                    primary_metric_key = f"{self.metrics[0]}_{self.target_columns[0]}"
                primary_metric = val_metrics.get(primary_metric_key, val_metrics[list(val_metrics.keys())[0]]) # get primary metric value, default to first metric if not found
                improved = False
                # early stopping check
                if self.best_metrics_mode == 'max':
                    if primary_metric > self.best_metric:
                        self.best_metric = primary_metric
                        improved = True
                else:
                    if primary_metric < self.best_metric:
                        self.best_metric = primary_metric
                        improved = True
                self.save_model(self.checkpoint_path, add_timestamp=False, verbose=False)
                if improved:
                    logging.info(colorize(f"Validation {primary_metric_key} improved to {self.best_metric:.4f}"))
                    self.save_model(self.best_path, add_timestamp=False, verbose=False)
                    self.best_checkpoint_path = self.best_path
                    self.early_stopper.trial_counter = 0
                else:
                    self.early_stopper.trial_counter += 1
                    logging.info(colorize(f"No improvement for {self.early_stopper.trial_counter} epoch(s)"))
                if self.early_stopper.trial_counter >= self.early_stopper.patience:
                    self.stop_training = True
                    logging.info(colorize(f"Early stopping triggered after {epoch + 1} epochs", color="bright_red", bold=True))
                    break
            else:
                self.save_model(self.checkpoint_path, add_timestamp=False, verbose=False)
                self.save_model(self.best_path, add_timestamp=False, verbose=False)
                self.best_checkpoint_path = self.best_path
            if self.stop_training:
                break
            if self.scheduler_fn is not None:
                if isinstance(self.scheduler_fn, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loader is not None:
                        self.scheduler_fn.step(primary_metric)
                else:
                    self.scheduler_fn.step()                   
        logging.info(" ")
        logging.info(colorize("Training finished.", bold=True))
        logging.info(" ")
        if valid_loader is not None:
            logging.info(colorize(f"Load best model from: {self.best_checkpoint_path}"))
            self.load_model(self.best_checkpoint_path, map_location=self.device, verbose=False)
        return self

    def train_epoch(self, train_loader: DataLoader, is_streaming: bool = False) -> Union[float, np.ndarray, tuple[Union[float, np.ndarray], dict]]:
        accumulated_loss = 0.0
        self.train()
        num_batches = 0
        y_true_list = []
        y_pred_list = []

        user_ids_list = [] if self.needs_user_ids else None
        if self.steps_per_epoch is not None:
            batch_iter = enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {self.epoch_index + 1}", total=self.steps_per_epoch))
        else:
            desc = "Batches" if is_streaming else f"Epoch {self.epoch_index + 1}"
            batch_iter = enumerate(tqdm.tqdm(train_loader, desc=desc))
        for batch_index, batch_data in batch_iter:
            batch_dict = batch_to_dict(batch_data)
            X_input, y_true = self.get_input(batch_dict, require_labels=True)
            y_pred = self.forward(X_input)
            loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()
            total_loss = loss + reg_loss
            self.optimizer_fn.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
            self.optimizer_fn.step()
            accumulated_loss += loss.item()
            if y_true is not None:
                y_true_list.append(y_true.detach().cpu().numpy())
            if self.needs_user_ids and user_ids_list is not None:
                batch_user_id = get_user_ids(data=batch_dict, id_columns=self.id_columns)
                if batch_user_id is not None:
                    user_ids_list.append(batch_user_id)
            if y_pred is not None and isinstance(y_pred, torch.Tensor):
                y_pred_list.append(y_pred.detach().cpu().numpy())
            num_batches += 1
        avg_loss = accumulated_loss / max(num_batches, 1)
        if len(y_true_list) > 0 and len(y_pred_list) > 0: # Compute metrics if requested
            y_true_all = np.concatenate(y_true_list, axis=0)
            y_pred_all = np.concatenate(y_pred_list, axis=0)
            combined_user_ids = None
            if self.needs_user_ids and user_ids_list:
                combined_user_ids = np.concatenate(user_ids_list, axis=0)
            metrics_dict = evaluate_metrics(y_true=y_true_all, y_pred=y_pred_all, metrics=self.metrics, task=self.task, target_names=self.target_columns, task_specific_metrics=self.task_specific_metrics, user_ids=combined_user_ids)
            return avg_loss, metrics_dict
        return avg_loss

    def prepare_validation_data(self, valid_data: dict | pd.DataFrame | DataLoader | None, batch_size: int, needs_user_ids: bool, user_id_column: str | None = 'user_id') -> tuple[DataLoader | None, np.ndarray | None]:
        if valid_data is None:
            return None, None
        if isinstance(valid_data, DataLoader):
            return valid_data, None
        valid_loader = self.prepare_data_loader(valid_data, batch_size=batch_size, shuffle=False)
        valid_user_ids = None
        if needs_user_ids:
            if user_id_column is None:
                raise ValueError("[BaseModel-validation Error] user_id_column must be specified when user IDs are needed for validation metrics.")
            valid_user_ids = get_user_ids(data=valid_data, id_columns=user_id_column)
        return valid_loader, valid_user_ids

    def evaluate(self, 
                 data: dict | pd.DataFrame | DataLoader, 
                 metrics: list[str] | dict[str, list[str]] | None = None,
                 batch_size: int = 32,
                 user_ids: np.ndarray | None = None,
                 user_id_column: str = 'user_id') -> dict:
        self.eval()
        eval_metrics = metrics if metrics is not None else self.metrics
        if eval_metrics is None:
            raise ValueError("[BaseModel-evaluate Error] No metrics specified for evaluation. Please provide metrics parameter or call fit() first.")
        needs_user_ids = check_user_id(eval_metrics, self.task_specific_metrics)
        
        if isinstance(data, DataLoader):
            data_loader = data
        else:
            if user_ids is None and needs_user_ids:
                user_ids = get_user_ids(data=data, id_columns=user_id_column)
            data_loader = self.prepare_data_loader(data, batch_size=batch_size, shuffle=False)
        y_true_list = []
        y_pred_list = []
        collected_user_ids = []
        batch_count = 0
        with torch.no_grad():
            for batch_data in data_loader:
                batch_count += 1
                batch_dict = batch_to_dict(batch_data)
                X_input, y_true = self.get_input(batch_dict, require_labels=True)
                y_pred = self.forward(X_input)
                if y_true is not None:
                    y_true_list.append(y_true.cpu().numpy())
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.cpu().numpy())
                if needs_user_ids and user_ids is None:
                    batch_user_id = get_user_ids(data=batch_dict, id_columns=self.id_columns)
                    if batch_user_id is not None:
                        collected_user_ids.append(batch_user_id)
        logging.info(colorize(f"  Evaluation batches processed: {batch_count}", color="cyan"))
        if len(y_true_list) > 0:
            y_true_all = np.concatenate(y_true_list, axis=0)
            logging.info(colorize(f"  Evaluation samples: {y_true_all.shape[0]}", color="cyan"))
        else:
            y_true_all = None
            logging.info(colorize(f"  Warning: No y_true collected from evaluation data", color="yellow"))
            
        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
        else:
            y_pred_all = None
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
        final_user_ids = user_ids
        if final_user_ids is None and collected_user_ids:
            final_user_ids = np.concatenate(collected_user_ids, axis=0)
        metrics_dict = evaluate_metrics(y_true=y_true_all, y_pred=y_pred_all, metrics=metrics_to_use, task=self.task, target_names=self.target_columns, task_specific_metrics=self.task_specific_metrics, user_ids=final_user_ids,)
        return metrics_dict

    def predict(
        self,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int = 32,
        save_path: str | os.PathLike | None = None,
        save_format: Literal["csv", "parquet"] = "csv",
        include_ids: bool | None = None,
        return_dataframe: bool = True,
        streaming_chunk_size: int = 10000,
    ) -> pd.DataFrame | np.ndarray:
        self.eval()
        if include_ids is None:
            include_ids = bool(self.id_columns)
        include_ids = include_ids and bool(self.id_columns)

        if save_path is not None and not return_dataframe:
            return self._predict_streaming(data=data, batch_size=batch_size, save_path=save_path, save_format=save_format, include_ids=include_ids, streaming_chunk_size=streaming_chunk_size, return_dataframe=return_dataframe)
        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(dense_features=self.dense_features, sparse_features=self.sparse_features, sequence_features=self.sequence_features, target=self.target_columns, id_columns=self.id_columns,)
            data_loader = rec_loader.create_dataloader(data=data, batch_size=batch_size, shuffle=False, load_full=False, chunk_size=streaming_chunk_size,)
        elif not isinstance(data, DataLoader):
            data_loader = self.prepare_data_loader(data, batch_size=batch_size, shuffle=False,)
        else:
            data_loader = data
        
        y_pred_list: list[np.ndarray] = []
        id_buffers: dict[str, list[np.ndarray]] = {name: [] for name in (self.id_columns or [])} if include_ids else {}
        id_arrays: dict[str, np.ndarray] | None = None
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Predicting"):
                batch_dict = batch_to_dict(batch_data, include_ids=include_ids)
                X_input, _ = self.get_input(batch_dict, require_labels=False)
                y_pred = self.forward(X_input)
                if y_pred is not None and isinstance(y_pred, torch.Tensor):
                    y_pred_list.append(y_pred.detach().cpu().numpy())
                if include_ids and self.id_columns and batch_dict.get("ids"):
                    for id_name in self.id_columns:
                        if id_name not in batch_dict["ids"]:
                            continue
                        id_tensor = batch_dict["ids"][id_name]
                        id_np = id_tensor.detach().cpu().numpy() if isinstance(id_tensor, torch.Tensor) else np.asarray(id_tensor)
                        id_buffers[id_name].append(id_np.reshape(id_np.shape[0], -1) if id_np.ndim == 1 else id_np)
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
                pred_columns.append(f"{name}_pred")
        while len(pred_columns) < num_outputs:
            pred_columns.append(f"pred_{len(pred_columns)}")
        if include_ids and self.id_columns:
            id_arrays = {}
            for id_name, pieces in id_buffers.items():
                if pieces:
                    concatenated = np.concatenate([p.reshape(p.shape[0], -1) for p in pieces], axis=0)
                    id_arrays[id_name] = concatenated.reshape(concatenated.shape[0])
                else:
                    id_arrays[id_name] = np.array([], dtype=np.int64)
            if return_dataframe:
                id_df = pd.DataFrame(id_arrays)
                pred_df = pd.DataFrame(y_pred_all, columns=pred_columns)
                if len(id_df) and len(pred_df) and len(id_df) != len(pred_df):
                    raise ValueError(f"[BaseModel-predict Error] Mismatch between id rows ({len(id_df)}) and prediction rows ({len(pred_df)}).")
                output = pd.concat([id_df, pred_df], axis=1)
            else:
                output = y_pred_all
        else:
            output = pd.DataFrame(y_pred_all, columns=pred_columns) if return_dataframe else y_pred_all
        if save_path is not None:
            if save_format not in ("csv", "parquet"):
                raise ValueError(f"[BaseModel-predict Error] Unsupported save_format '{save_format}'. Choose from 'csv' or 'parquet'.")
            suffix = ".csv" if save_format == "csv" else ".parquet"
            target_path = resolve_save_path(path=save_path, default_dir=self.session.predictions_dir, default_name="predictions", suffix=suffix, add_timestamp=True if save_path is None else False)
            if isinstance(output, pd.DataFrame):
                df_to_save = output
            else:
                df_to_save = pd.DataFrame(y_pred_all, columns=pred_columns)
                if include_ids and self.id_columns and id_arrays is not None:
                    id_df = pd.DataFrame(id_arrays)
                    if len(id_df) and len(df_to_save) and len(id_df) != len(df_to_save):
                        raise ValueError(f"[BaseModel-predict Error] Mismatch between id rows ({len(id_df)}) and prediction rows ({len(df_to_save)}).")
                    df_to_save = pd.concat([id_df, df_to_save], axis=1)
            if save_format == "csv":
                df_to_save.to_csv(target_path, index=False)
            else:
                df_to_save.to_parquet(target_path, index=False)
            logging.info(colorize(f"Predictions saved to: {target_path}", color="green"))
        return output

    def _predict_streaming(
        self,
        data: str | dict | pd.DataFrame | DataLoader,
        batch_size: int,
        save_path: str | os.PathLike,
        save_format: Literal["csv", "parquet"],
        include_ids: bool,
        streaming_chunk_size: int,
        return_dataframe: bool,
    ) -> pd.DataFrame:
        if isinstance(data, (str, os.PathLike)):
            rec_loader = RecDataLoader(dense_features=self.dense_features, sparse_features=self.sparse_features, sequence_features=self.sequence_features, target=self.target_columns, id_columns=self.id_columns)
            data_loader = rec_loader.create_dataloader(data=data, batch_size=batch_size, shuffle=False, load_full=False, chunk_size=streaming_chunk_size,)
        elif not isinstance(data, DataLoader):
            data_loader = self.prepare_data_loader(data, batch_size=batch_size, shuffle=False,)
        else:
            data_loader = data

        suffix = ".csv" if save_format == "csv" else ".parquet"
        target_path = resolve_save_path(path=save_path, default_dir=self.session.predictions_dir, default_name="predictions", suffix=suffix, add_timestamp=True if save_path is None else False,)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        header_written = target_path.exists() and target_path.stat().st_size > 0
        parquet_writer = None

        pred_columns: list[str] | None = None
        collected_frames: list[pd.DataFrame] = []

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
                            pred_columns.append(f"{name}_pred")
                    while len(pred_columns) < num_outputs:
                        pred_columns.append(f"pred_{len(pred_columns)}")
                        
                id_arrays_batch: dict[str, np.ndarray] = {}
                if include_ids and self.id_columns and batch_dict.get("ids"):
                    for id_name in self.id_columns:
                        if id_name not in batch_dict["ids"]:
                            continue
                        id_tensor = batch_dict["ids"][id_name]
                        id_np = id_tensor.detach().cpu().numpy() if isinstance(id_tensor, torch.Tensor) else np.asarray(id_tensor)
                        id_arrays_batch[id_name] = id_np.reshape(id_np.shape[0])

                df_batch = pd.DataFrame(y_pred_np, columns=pred_columns)
                if id_arrays_batch:
                    id_df = pd.DataFrame(id_arrays_batch)
                    if len(id_df) and len(df_batch) and len(id_df) != len(df_batch):
                        raise ValueError(f"Mismatch between id rows ({len(id_df)}) and prediction rows ({len(df_batch)}).")
                    df_batch = pd.concat([id_df, df_batch], axis=1)

                if save_format == "csv":
                    df_batch.to_csv(target_path, mode="a", header=not header_written, index=False)
                    header_written = True
                else:
                    try:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                    except ImportError as exc:  # pragma: no cover
                        raise ImportError("[BaseModel-predict-streaming Error] Parquet streaming save requires pyarrow to be installed.") from exc
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
            return pd.concat(collected_frames, ignore_index=True) if collected_frames else pd.DataFrame(columns=pred_columns or [])
        return pd.DataFrame(columns=pred_columns or [])

    def save_model(self, save_path: str | Path | None = None, add_timestamp: bool | None = None, verbose: bool = True):
        add_timestamp = False if add_timestamp is None else add_timestamp
        target_path = resolve_save_path(path=save_path, default_dir=self.session_path, default_name=self.model_name, suffix=".model", add_timestamp=add_timestamp)
        model_path = Path(target_path)
        torch.save(self.state_dict(), model_path)

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
            logging.info(colorize(f"Model saved to: {model_path}, features config saved to: {config_path}, NextRec version: {__version__}",color="green",))
    
    def load_model(self, save_path: str | Path, map_location: str | torch.device | None = "cpu", verbose: bool = True):
        self.to(self.device)
        base_path = Path(save_path)
        if base_path.is_dir():
            model_files = sorted(base_path.glob("*.model"))
            if not model_files:
                raise FileNotFoundError(f"[BaseModel-load-model Error] No *.model file found in directory: {base_path}")
            model_path = model_files[-1]
            config_dir = base_path
        else:
            model_path = base_path.with_suffix(".model") if base_path.suffix == "" else base_path
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
        self.set_all_features(dense_features=dense_features, sparse_features=sparse_features, sequence_features=sequence_features, target=target, id_columns=id_columns)

        cfg_version = features_config.get("version")
        if verbose:
            logging.info(colorize(f"Model weights loaded from: {model_path}, features config loaded from: {features_config_path}, NextRec version: {cfg_version}",color="green",))

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
        Factory that reconstructs a model instance (including feature specs)
        from a saved checkpoint directory or *.model file.
        """
        base_path = Path(checkpoint_path)
        verbose = kwargs.pop("verbose", True)
        if base_path.is_dir():
            model_candidates = sorted(base_path.glob("*.model"))
            if not model_candidates:
                raise FileNotFoundError(f"[BaseModel-from-checkpoint Error] No *.model file found under: {base_path}")
            model_file = model_candidates[-1]
            config_dir = base_path
        else:
            model_file = base_path.with_suffix(".model") if base_path.suffix == "" else base_path
            config_dir = model_file.parent
        features_config_path = config_dir / "features_config.pkl"
        if not features_config_path.exists():
            raise FileNotFoundError(f"[BaseModel-from-checkpoint Error] features_config.pkl not found next to checkpoint: {features_config_path}")
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
            logger.info(f"\nSparse Features ({len(self.sparse_features)}):")

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
            logger.info(f"\nSequence Features ({len(self.sequence_features)}):")

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
        logger.info(f"Target Columns:          {self.target_columns}")
        logger.info(f"Device:                  {self.device}")
        
        if hasattr(self, 'optimizer_name'):
            logger.info(f"Optimizer:               {self.optimizer_name}")
            if self.optimizer_params:
                for key, value in self.optimizer_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, 'scheduler_name') and self.scheduler_name:
            logger.info(f"Scheduler:               {self.scheduler_name}")
            if self.scheduler_params:
                for key, value in self.scheduler_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, 'loss_config'):
            logger.info(f"Loss Function:           {self.loss_config}")
        if hasattr(self, 'loss_weights'):
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
        
        logger.info("")
        logger.info("")


class BaseMatchModel(BaseModel):
    """
    Base class for match (retrieval/recall) models
    Supports pointwise, pairwise, and listwise training modes
    """
    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def task_type(self) -> str:
        raise NotImplementedError
    
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

        self.user_feature_names = [f.name for f in (self.user_dense_features + self.user_sparse_features + self.user_sequence_features)]
        self.item_feature_names = [f.name for f in (self.item_dense_features + self.item_sparse_features + self.item_sequence_features)]

    def get_user_features(self, X_input: dict) -> dict:
        return {
            name: X_input[name]
            for name in self.user_feature_names
            if name in X_input
        }

    def get_item_features(self, X_input: dict) -> dict:
        return {
            name: X_input[name]
            for name in self.item_feature_names
            if name in X_input
        }
        
    def compile(self, 
                optimizer: str | torch.optim.Optimizer = "adam",
                optimizer_params: dict | None = None,
                scheduler: str | torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | type[torch.optim.lr_scheduler.LRScheduler] | None = None,
                scheduler_params: dict | None = None,
                loss: str | nn.Module | list[str | nn.Module] | None = "bce",
                loss_params: dict | list[dict] | None = None):
        """
        Compile match model with optimizer, scheduler, and loss function.
        Mirrors BaseModel.compile while adding training_mode validation for match tasks.
        """
        if self.training_mode not in self.support_training_modes:
            raise ValueError(f"{self.model_name} does not support training_mode='{self.training_mode}'. Supported modes: {self.support_training_modes}")
        # Call parent compile with match-specific logic
        optimizer_params = optimizer_params or {}
        
        self.optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self.optimizer_params = optimizer_params
        if isinstance(scheduler, str):
            self.scheduler_name = scheduler
        elif scheduler is not None:
            # Try to get __name__ first (for class types), then __class__.__name__ (for instances)
            self.scheduler_name = getattr(scheduler, '__name__', getattr(scheduler.__class__, '__name__', str(scheduler)))
        else:
            self.scheduler_name = None
        self.scheduler_params = scheduler_params or {}
        self.loss_config = loss
        self.loss_params = loss_params or {}

        self.optimizer_fn = get_optimizer(optimizer=optimizer, params=self.parameters(), **optimizer_params)
        # Set loss function based on training mode
        default_losses = {
            'pointwise': 'bce',
            'pairwise': 'bpr',
            'listwise': 'sampled_softmax',
        }

        if loss is None:
            loss_value = default_losses.get(self.training_mode, "bce")
        elif isinstance(loss, list):
            loss_value = loss[0] if loss and loss[0] is not None else default_losses.get(self.training_mode, "bce")
        else:
            loss_value = loss

        # Pairwise/listwise modes do not support BCE, fall back to sensible defaults
        if self.training_mode in {"pairwise", "listwise"} and loss_value in {"bce", "binary_crossentropy"}:
            loss_value = default_losses.get(self.training_mode, loss_value)
        loss_kwargs = get_loss_kwargs(self.loss_params, 0)
        self.loss_fn = [get_loss_fn(loss=loss_value, **loss_kwargs)]
        # set scheduler
        self.scheduler_fn = get_scheduler(scheduler, self.optimizer_fn, **(scheduler_params or {})) if scheduler else None

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
                raise ValueError("For pairwise/listwise training, forward should return (user_emb, item_emb). Please check BaseMatchModel.forward implementation.")
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

    
    def prepare_feature_data(self, data: dict | pd.DataFrame | DataLoader, features: list, batch_size: int) -> DataLoader:
        """Prepare data loader for specific features."""
        if isinstance(data, DataLoader):
            return data
        
        feature_data = {}
        for feature in features:
            if isinstance(data, dict):
                if feature.name in data:
                    feature_data[feature.name] = data[feature.name]
            elif isinstance(data, pd.DataFrame):
                if feature.name in data.columns:
                    feature_data[feature.name] = data[feature.name].values
        return self.prepare_data_loader(feature_data, batch_size=batch_size, shuffle=False)

    def encode_user(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        all_user_features = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        data_loader = self.prepare_feature_data(data, all_user_features, batch_size)
        
        embeddings_list = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding users"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                user_input = self.get_user_features(batch_dict["features"])
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
    
    def encode_item(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        data_loader = self.prepare_feature_data(data, all_item_features, batch_size)
        
        embeddings_list = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding items"):
                batch_dict = batch_to_dict(batch_data, include_ids=False)
                item_input = self.get_item_features(batch_dict["features"])
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)
