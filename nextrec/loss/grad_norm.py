"""
GradNorm loss weighting for multi-task learning.

Date: create on 27/10/2025
Checkpoint: edit on 24/12/2025
Author: Yang Zhou,zyaztec@gmail.com

Reference:
Chen, Zhao, et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks." ICML 2018.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grad_norm_shared_params(
    model: torch.nn.Module,
    shared_modules: Iterable[str] | None = None,
) -> list[torch.nn.Parameter]:
    if not shared_modules:
        return [p for p in model.parameters() if p.requires_grad]
    shared_params = []
    seen = set()
    for name in shared_modules:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                shared_params.append(param)
                seen.add(id(param))
    if not shared_params:
        return [p for p in model.parameters() if p.requires_grad]
    return shared_params


class GradNormLossWeighting:
    """
    Adaptive multi-task loss weighting with GradNorm.

    Args:
        nums_task: Number of tasks.
        alpha: GradNorm balancing strength.
        lr: Learning rate for the weight optimizer.
        init_weights: Optional initial weights per task.
        device: Torch device for weights.
        ema_decay: Optional EMA decay for smoothing loss ratios.
        init_ema_steps: Number of steps to build EMA for initial losses.
        init_ema_decay: EMA decay for initial losses when init_ema_steps > 0.
        eps: Small value for numerical stability.
    """

    def __init__(
        self,
        nums_task: int,
        alpha: float = 1.5,
        lr: float = 0.025,
        init_weights: Iterable[float] | None = None,
        device: torch.device | str | None = None,
        ema_decay: float | None = None,
        init_ema_steps: int = 0,
        init_ema_decay: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        if nums_task <= 1:
            raise ValueError("GradNorm requires nums_task > 1.")
        self.nums_task = nums_task
        self.alpha = alpha
        self.eps = eps
        if ema_decay is not None:
            ema_decay = ema_decay
            if ema_decay < 0.0 or ema_decay >= 1.0:
                raise ValueError("ema_decay must be in [0.0, 1.0).")
        self.ema_decay = ema_decay
        self.init_ema_steps = init_ema_steps
        if self.init_ema_steps < 0:
            raise ValueError("init_ema_steps must be >= 0.")
        self.init_ema_decay = init_ema_decay
        if self.init_ema_decay < 0.0 or self.init_ema_decay >= 1.0:
            raise ValueError("init_ema_decay must be in [0.0, 1.0).")
        self.init_ema_count = 0

        if init_weights is None:
            weights = torch.ones(self.nums_task, dtype=torch.float32)
        else:
            weights = torch.tensor(list(init_weights), dtype=torch.float32)
            if weights.numel() != self.nums_task:
                raise ValueError(
                    "init_weights length must match nums_task for GradNorm."
                )
        if device is not None:
            weights = weights.to(device)
        self.weights = nn.Parameter(weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=float(lr))

        self.initial_losses = None
        self.initial_losses_ema = None
        self.loss_ema = None
        self.pending_grad = None

    def to(self, device):
        device = torch.device(device)
        self.weights.data = self.weights.data.to(device)
        if self.initial_losses is not None:
            self.initial_losses = self.initial_losses.to(device)
        if self.initial_losses_ema is not None:
            self.initial_losses_ema = self.initial_losses_ema.to(device)
        if self.loss_ema is not None:
            self.loss_ema = self.loss_ema.to(device)
        return self

    def compute_weighted_loss(
        self,
        task_losses: list[torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter],
    ) -> torch.Tensor:
        """
        Return weighted total loss and update task weights with GradNorm.
        """
        if len(task_losses) != self.nums_task:
            raise ValueError(
                f"Expected {self.nums_task} task losses, got {len(task_losses)}."
            )
        shared_params = [p for p in shared_params if p.requires_grad]
        if not shared_params:
            return torch.stack(task_losses).sum()

        with torch.no_grad():
            loss_values = torch.tensor(
                [loss.item() for loss in task_losses], device=self.weights.device
            )
            if self.initial_losses is None:
                if self.init_ema_steps > 0:
                    if self.initial_losses_ema is None:
                        self.initial_losses_ema = loss_values
                    else:
                        self.initial_losses_ema = (
                            self.init_ema_decay * self.initial_losses_ema
                            + (1.0 - self.init_ema_decay) * loss_values
                        )
                    self.init_ema_count += 1
                    if self.init_ema_count >= self.init_ema_steps:
                        self.initial_losses = self.initial_losses_ema.clone()
                else:
                    self.initial_losses = loss_values

        weights_detached = self.weights.detach()
        weighted_losses = [
            weights_detached[i] * task_losses[i] for i in range(self.nums_task)
        ]
        total_loss = torch.stack(weighted_losses).sum()

        grad_norms = self.compute_grad_norms(task_losses, shared_params)
        with torch.no_grad():
            if self.ema_decay is not None:
                if self.loss_ema is None:
                    self.loss_ema = loss_values
                else:
                    self.loss_ema = (
                        self.ema_decay * self.loss_ema
                        + (1.0 - self.ema_decay) * loss_values
                    )
                ratio_source = self.loss_ema
            else:
                ratio_source = loss_values
            if self.initial_losses is not None:
                base_initial = self.initial_losses
            elif self.initial_losses_ema is not None:
                base_initial = self.initial_losses_ema
            else:
                base_initial = loss_values
            loss_ratios = ratio_source / (base_initial + self.eps)
            inv_rate = loss_ratios / (loss_ratios.mean() + self.eps)
            target = grad_norms.mean() * (inv_rate**self.alpha)

        grad_norm_loss = F.l1_loss(grad_norms, target.detach(), reduction="sum")
        grad_w = torch.autograd.grad(grad_norm_loss, self.weights, retain_graph=True)[0]
        self.pending_grad = grad_w.detach()

        return total_loss

    def compute_grad_norms(self, task_losses, shared_params):
        grad_norms = []
        for i, task_loss in enumerate(task_losses):
            grads = torch.autograd.grad(
                self.weights[i] * task_loss,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )

            sq_sum = torch.zeros((), device=self.weights.device)
            any_used = False
            for g in grads:
                if g is not None:
                    any_used = True
                    sq_sum = sq_sum + g.pow(2).sum()

            if not any_used:
                total_norm = torch.tensor(self.eps, device=self.weights.device)
            else:
                total_norm = torch.sqrt(sq_sum + self.eps)

            grad_norms.append(total_norm)

        return torch.stack(grad_norms)

    def step(self) -> None:
        if self.pending_grad is None:
            return

        self.optimizer.zero_grad(set_to_none=True)

        if self.weights.grad is None:
            self.weights.grad = torch.zeros_like(self.weights)
        self.weights.grad.copy_(self.pending_grad)

        self.optimizer.step()

        with torch.no_grad():
            w = self.weights.clamp(min=self.eps)
            w = w * self.nums_task / (w.sum() + self.eps)
            self.weights.copy_(w)

        self.pending_grad = None
