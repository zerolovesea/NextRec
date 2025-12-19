"""
PyTorch-related utilities for NextRec.

This module groups device setup, distributed helpers, optimizers/schedulers,
initialization, and tensor helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from nextrec.basic.loggers import colorize

KNOWN_NONLINEARITIES: Set[str] = {
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
}


def resolve_nonlinearity(activation: str) -> str:
    if activation in KNOWN_NONLINEARITIES:
        return activation
    return "linear"


def resolve_gain(activation: str, param: Dict[str, Any]) -> float:
    if "gain" in param:
        return param["gain"]
    nonlinearity = resolve_nonlinearity(activation)
    try:
        return nn.init.calculate_gain(nonlinearity, param.get("param"))  # type: ignore
    except ValueError:
        return 1.0


def get_initializer(
    init_type: str = "normal",
    activation: str = "linear",
    param: Dict[str, Any] | None = None,
):
    param = param or {}
    nonlinearity = resolve_nonlinearity(activation)
    gain = resolve_gain(activation, param)

    def initializer_fn(tensor):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(tensor, gain=gain)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(tensor, gain=gain)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                tensor, a=param.get("a", 0), nonlinearity=nonlinearity  # type: ignore
            )
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(
                tensor, a=param.get("a", 0), nonlinearity=nonlinearity  # type: ignore
            )
        elif init_type == "orthogonal":
            nn.init.orthogonal_(tensor, gain=gain)
        elif init_type == "normal":
            nn.init.normal_(
                tensor, mean=param.get("mean", 0.0), std=param.get("std", 0.0001)
            )
        elif init_type == "uniform":
            nn.init.uniform_(tensor, a=param.get("a", -0.05), b=param.get("b", 0.05))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        return tensor

    return initializer_fn


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        import platform

        mac_ver = platform.mac_ver()[0]
        try:
            major, _ = (int(x) for x in mac_ver.split(".")[:2])
        except Exception:
            major, _ = 0, 0
        if major >= 14:
            return "mps"
    return "cpu"


def get_device_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "mps_available": torch.backends.mps.is_available(),
        "current_device": resolve_device(),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)

    return info


def configure_device(
    distributed: bool, local_rank: int, base_device: torch.device | str = "cpu"
) -> torch.device:
    try:
        device = torch.device(base_device)
    except Exception:
        logging.warning(
            "[configure_device Warning] Invalid base_device, falling back to CPU."
        )
        return torch.device("cpu")

    if distributed:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                logging.warning(
                    "[Distributed Warning] CUDA requested but unavailable. Falling back to CPU."
                )
                return torch.device("cpu")
            if not (0 <= local_rank < torch.cuda.device_count()):
                logging.warning(
                    f"[Distributed Warning] local_rank {local_rank} is invalid for available CUDA devices. Falling back to CPU."
                )
                return torch.device("cpu")
            try:
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
            except Exception as exc:
                logging.warning(
                    f"[Distributed Warning] Failed to set CUDA device for local_rank {local_rank}: {exc}. Falling back to CPU."
                )
                return torch.device("cpu")
        return torch.device("cpu")
    return device


def get_optimizer(
    optimizer: str | torch.optim.Optimizer = "adam",
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
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")
        optimizer_fn = opt_class(params=params, **optimizer_params)
    elif isinstance(optimizer, torch.optim.Optimizer):
        optimizer_fn = optimizer
    else:
        raise TypeError(f"Invalid optimizer type: {type(optimizer)}")
    return optimizer_fn


def get_scheduler(
    scheduler: (
        str
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
            scheduler_fn = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_params
            )
        elif scheduler == "cosine":
            scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
        else:
            raise NotImplementedError(f"Unsupported scheduler: {scheduler}")
    elif isinstance(
        scheduler,
        (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.LRScheduler),
    ):
        scheduler_fn = scheduler
    else:
        raise TypeError(f"Invalid scheduler type: {type(scheduler)}")

    return scheduler_fn


def to_tensor(
    value: Any, dtype: torch.dtype, device: torch.device | str | None = None
) -> torch.Tensor:
    if value is None:
        raise ValueError("[Tensor Utils Error] Cannot convert None to tensor.")
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)

    if device is not None:
        target_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
    return tensor


def stack_tensors(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("[Tensor Utils Error] Cannot stack empty list of tensors.")
    return torch.stack(tensors, dim=dim)


def concat_tensors(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError(
            "[Tensor Utils Error] Cannot concatenate empty list of tensors."
        )
    return torch.cat(tensors, dim=dim)


def pad_sequence_tensors(
    tensors: list[torch.Tensor],
    max_len: int | None = None,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> torch.Tensor:
    if not tensors:
        raise ValueError("[Tensor Utils Error] Cannot pad empty list of tensors.")
    if max_len is None:
        max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)
    padded = torch.full(
        (batch_size, max_len),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, tensor in enumerate(tensors):
        length = min(tensor.size(0), max_len)
        if padding_side == "right":
            padded[i, :length] = tensor[:length]
        elif padding_side == "left":
            padded[i, -length:] = tensor[:length]
        else:
            raise ValueError(
                f"[Tensor Utils Error] padding_side must be 'right' or 'left', got {padding_side}"
            )
    return padded


def init_process_group(
    distributed: bool, rank: int, world_size: int, device_id: int | None = None
) -> None:
    """
    initialize distributed process group for multi-GPU training.

    Args:
        distributed: whether to enable distributed training
        rank: global rank of the current process
        world_size: total number of processes
    """
    if (not distributed) or (not dist.is_available()) or dist.is_initialized():
        return
    backend = "nccl" if device_id is not None else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size
    )


def gather_numpy(self, array: np.ndarray | None) -> np.ndarray | None:
    """
    Gather numpy arrays (or None) across ranks. Uses all_gather_object to avoid
    shape mismatches and ensures every rank participates even when local data is empty.
    """
    if not (self.distributed and dist.is_available() and dist.is_initialized()):
        return array

    world_size = dist.get_world_size()
    gathered: list[np.ndarray | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, array)
    pieces: list[np.ndarray] = []
    for item in gathered:
        if item is None:
            continue
        item_np = np.asarray(item)
        if item_np.size > 0:
            pieces.append(item_np)
    if not pieces:
        return None
    return np.concatenate(pieces, axis=0)


def add_distributed_sampler(
    loader: DataLoader,
    distributed: bool,
    world_size: int,
    rank: int,
    shuffle: bool,
    drop_last: bool,
    default_batch_size: int,
    is_main_process: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    add distributedsampler to a dataloader, this for distributed training
    when each device has its own dataloader
    """
    # early return if not distributed
    if not (distributed and dist.is_available() and dist.is_initialized()):
        return loader, None
    # return if already has DistributedSampler
    if isinstance(loader.sampler, DistributedSampler):
        return loader, loader.sampler
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader, None
    if isinstance(dataset, IterableDataset):
        if is_main_process:
            logging.info(
                colorize(
                    "[Distributed Info] Iterable/streaming DataLoader provided; DistributedSampler is skipped. Ensure dataset handles sharding per rank.",
                    color="yellow",
                )
            )
        return loader, None
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    loader_kwargs = {
        "batch_size": (
            loader.batch_size if loader.batch_size is not None else default_batch_size
        ),
        "shuffle": False,
        "sampler": sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "drop_last": drop_last,
    }
    if getattr(loader, "pin_memory", False):
        loader_kwargs["pin_memory"] = True
    pin_memory_device = getattr(loader, "pin_memory_device", None)
    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = pin_memory_device
    timeout = getattr(loader, "timeout", None)
    if timeout:
        loader_kwargs["timeout"] = timeout
    worker_init_fn = getattr(loader, "worker_init_fn", None)
    if worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn
    generator = getattr(loader, "generator", None)
    if generator is not None:
        loader_kwargs["generator"] = generator
    if loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = getattr(
            loader, "persistent_workers", False
        )
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    distributed_loader = DataLoader(dataset, **loader_kwargs)
    if is_main_process:
        logging.info(
            colorize(
                "[Distributed Info] Attached DistributedSampler to provided DataLoader",
                color="cyan",
            )
        )
    return distributed_loader, sampler
