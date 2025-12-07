"""
Optimizer and Scheduler utilities for NextRec

Date: create on 13/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
from typing import Iterable


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
