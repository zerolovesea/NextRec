"""
Distributed utilities for NextRec.

Date: create on 04/12/2025
Checkpoint: edit on 05/12/2025
Author: Yang Zhou,zyaztec@gmail.com
"""

import logging
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from nextrec.basic.loggers import colorize


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
