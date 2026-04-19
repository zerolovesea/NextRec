"""
PyTorch-related utilities for NextRec.

This module groups device setup, optimizers/schedulers, initialization,
and tensor helpers.

Date: create on 27/10/2025
Checkpoint: edit on 13/03/2026
Author: Yang Zhou, zyaztec@gmail.com
"""

from __future__ import annotations

import functools
import logging
import numbers
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

from nextrec.utils.types import (
    EmbeddingInitType,
    InitializerActivationType,
    OptimizerName,
    SchedulerName,
)


def smart_inference_mode():
    """Return an inference decorator that prefers torch.inference_mode over torch.no_grad.

    The decorated function executes under torch.inference_mode when available. If the
    current call site is already inside inference mode, the wrapper becomes a pass-through.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
                return fn(*args, **kwargs)

            context_decorator = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            return context_decorator()(fn)(*args, **kwargs)

        return wrapper

    return decorate


def to_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def to_float(value: Any) -> float | None:
    if isinstance(value, numbers.Number):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def to_numpy(values: Any, as_2d: bool = False) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if as_2d and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def to_tensor(value: Any, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    if value is None:
        raise ValueError("[Tensor Utils Error] Cannot convert None to tensor.")
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        if isinstance(value, np.ndarray):
            value = value.copy()
        tensor = torch.as_tensor(value)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)

    if device is not None:
        target_device = device if isinstance(device, torch.device) else torch.device(device)
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
    return tensor


def get_nonlinearity(activation: str) -> str:
    if activation in [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
        "gelu",
    ]:
        return activation
    return "linear"


def get_gain(activation: str, param: Dict[str, Any]) -> float:
    if "gain" in param:
        return param["gain"]
    nonlinearity = get_nonlinearity(activation)
    try:
        return nn.init.calculate_gain(nonlinearity, param.get("param"))  # type: ignore
    except ValueError:
        return 1.0


def get_initializer(
    init_type: EmbeddingInitType = "normal",
    activation: InitializerActivationType = "linear",
    param: Dict[str, Any] | None = None,
):
    param = param or {}
    nonlinearity = get_nonlinearity(activation)
    gain = get_gain(activation, param)

    def initializer_fn(tensor):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(tensor, gain=gain)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(tensor, gain=gain)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor, a=param.get("a", 0), nonlinearity=nonlinearity)  # type: ignore
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(tensor, a=param.get("a", 0), nonlinearity=nonlinearity)  # type: ignore
        elif init_type == "orthogonal":
            nn.init.orthogonal_(tensor, gain=gain)
        elif init_type == "normal":
            nn.init.normal_(tensor, mean=param.get("mean", 0.0), std=param.get("std", 0.0001))
        elif init_type == "uniform":
            nn.init.uniform_(tensor, a=param.get("a", -0.05), b=param.get("b", 0.05))
        else:
            raise ValueError(f"[Initializer Error] Unknown init_type: {init_type}")
        return tensor

    return initializer_fn


def get_device(base_device: torch.device | str | list[str] | tuple[str, ...] | None = "cpu") -> torch.device:
    if isinstance(base_device, (list, tuple)):
        device_tokens = [str(item).strip() for item in base_device if str(item).strip()]
    elif base_device is None:
        device_tokens = []
    else:
        device_tokens = [part.strip() for part in str(base_device).split(",") if part.strip()]

    if len(device_tokens) > 1:
        raise ValueError("[Device Error] Multi-GPU training is no longer supported. Please specify a single device.")

    raw_device = device_tokens[0].lower() if device_tokens else "cpu"
    if raw_device.isdigit():
        raw_device = f"cuda:{raw_device}"

    try:
        device = torch.device(raw_device)
    except Exception:
        logging.warning("[get_device Warning] Invalid base_device, falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            logging.warning("[get_device Warning] CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if device.index is not None and not (0 <= device.index < torch.cuda.device_count()):
            logging.warning(f"[get_device Warning] CUDA device index {device.index} is invalid. Falling back to CPU.")
            return torch.device("cpu")
        if device.index is not None:
            try:
                torch.cuda.set_device(device.index)
            except Exception as exc:
                logging.warning(
                    f"[get_device Warning] Failed to set CUDA device {device.index}: {exc}. Falling back to CPU."
                )
                return torch.device("cpu")

    return device


def get_optimizer(
    optimizer: OptimizerName | torch.optim.Optimizer = "adam",
    params: Iterable[torch.nn.Parameter] | None = None,
    **optimizer_params,
):
    if params is None:
        raise ValueError("params cannot be None. Please provide model parameters.")

    if "lr" not in optimizer_params:
        optimizer_params["lr"] = 1e-3
    if isinstance(optimizer, str):
        opt_name = optimizer.lower()
        if opt_name == "adam":
            opt_class = torch.optim.Adam
        elif opt_name == "sgd":
            opt_class = torch.optim.SGD
        elif opt_name == "adamw":
            opt_class = torch.optim.AdamW
        elif opt_name == "adagrad":
            opt_class = torch.optim.Adagrad
        elif opt_name == "rmsprop":
            opt_class = torch.optim.RMSprop
        else:
            raise NotImplementedError(f"[Optimizer Error] Unsupported optimizer: {optimizer}")
        optimizer_fn = opt_class(params=params, **optimizer_params)
    elif isinstance(optimizer, torch.optim.Optimizer):
        optimizer_fn = optimizer
    else:
        raise TypeError(f"[Optimizer Error] Invalid optimizer type: {type(optimizer)}")
    return optimizer_fn


def get_scheduler(
    scheduler: (
        SchedulerName
        | torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.LRScheduler
        | type[torch.optim.lr_scheduler._LRScheduler]
        | type[torch.optim.lr_scheduler.LRScheduler]
        | None
    ),
    optimizer,
    **scheduler_params,
):
    if isinstance(scheduler, str):
        if scheduler == "step":
            scheduler_fn = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler == "cosine":
            scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            raise NotImplementedError(f"[Scheduler Error] Unsupported scheduler: {scheduler}")
    elif isinstance(scheduler, type) and issubclass(
        scheduler,
        (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.LRScheduler),
    ):
        scheduler_fn = scheduler(optimizer, **scheduler_params)
    elif isinstance(
        scheduler,
        (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.LRScheduler),
    ):
        scheduler_fn = scheduler
    else:
        raise TypeError(f"[Scheduler Error] Invalid scheduler type: {type(scheduler)}")

    return scheduler_fn


def get_warmup(
    warmup: bool | dict | None,
) -> dict | None:
    if warmup in (None, False):
        return None

    config = {"enabled": True, "epochs": 1, "start_factor": 0.1, "end_factor": 1.0}
    if warmup is True:
        pass
    elif isinstance(warmup, dict):
        config.update(warmup)
    else:
        raise TypeError(f"[BaseModel-compile Error] warmup must be bool, dict, or None, got {type(warmup)}")

    if not bool(config.get("enabled", True)):
        return None

    warmup_epochs = int(config.get("epochs", 0))
    if warmup_epochs <= 0:
        return None

    start_factor = float(config.get("start_factor", 0.1))
    end_factor = float(config.get("end_factor", 1.0))
    if start_factor <= 0 or start_factor > 1:
        raise ValueError(f"[BaseModel-compile Error] warmup.start_factor must be in (0, 1], got {start_factor}.")
    if end_factor <= 0 or end_factor > 1:
        raise ValueError(f"[BaseModel-compile Error] warmup.end_factor must be in (0, 1], got {end_factor}.")
    if start_factor > end_factor:
        raise ValueError("[BaseModel-compile Error] warmup.start_factor must be <= end_factor.")

    return {
        "enabled": True,
        "epochs": warmup_epochs,
        "start_factor": start_factor,
        "end_factor": end_factor,
    }
