"""
GradNorm loss weighting for multi-task learning.

Date: create on 27/10/2025
Checkpoint: edit on 07/02/2026
Author: Yang Zhou,zyaztec@gmail.com

Reference:
Chen, Zhao, et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks." ICML 2018.

pseudocode:
---
Initialize w_i = 1
Record L_i(0)

for each step:
    1. Forward: compute each task loss L_i
    2. Compute G_i = ||∇_W (w_i * L_i)||
    3. Compute r_i = (L_i / L_i(0)) / mean(...)
    4. Compute target: Ĝ_i = mean(G) * r_i^α
    5. L_grad = sum |G_i - Ĝ_i|
    6. Update w_i using ∇ L_grad
    7. Backprop with sum_i (w_i * L_i) to update model

伪代码：
---
初始化 w_i = 1
记录 L_i(0)

for each step:
    1. 前向算各 task loss: L_i
    2. 计算 G_i = ||∇_W (w_i * L_i)||
    3. 计算 r_i = (L_i / L_i(0)) / mean(...)
    4. 计算 target: Ĝ_i = mean(G) * r_i^α
    5. L_grad = sum |G_i - Ĝ_i|
    6. 对 w_i 用 ∇L_grad 更新
    7. 用 ∑ w_i * L_i 反传更新模型
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grad_norm_shared_params(
    model,
    shared_modules=None,
):
    """
    Get shared parameters for GradNorm.

    Args:
        model: A pytorch model instance containing grad_norm_shared_modules attribute.
        shared_modules: Optional list of module names to consider as shared.
    """
    if not shared_modules:
        # If no specific shared modules are provided, consider all parameters as shared
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
    return shared_params


class GradNormLossWeighting:

    def __init__(
        self,
        nums_task: int,
        alpha: float = 1.5,
        lr: float = 0.025,
        init_weights: Iterable[float] | None = None,
        device: torch.device | str | None = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Adaptive multi-task loss weighting with GradNorm.

        Args:
            nums_task: Number of tasks.
            alpha: GradNorm balancing strength.
            lr: Learning rate for the weight optimizer.
            init_weights: Optional initial weights per task.
            device: Torch device for weights.
            eps: Small value for numerical stability.



        """

        if nums_task <= 1:
            raise ValueError("GradNorm requires nums_task > 1.")

        self.nums_task = nums_task
        self.alpha = alpha
        self.eps = eps

        if init_weights is None:
            weights = torch.ones(self.nums_task, dtype=torch.float32)
        else:
            weights = torch.tensor(list(init_weights), dtype=torch.float32)

        if device is not None:
            weights = weights.to(device)
        self.weights = nn.Parameter(weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=float(lr))

        self.initial_losses = None
        self.pending_grad = None

    def compute_weighted_loss(
        self,
        task_losses: list[torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter],
    ) -> torch.Tensor:
        """
        Return weighted total loss and update task weights with GradNorm.

        BaseModel will use this method to compute the weighted loss when self.grad_norm is enabled.
        """
        if len(task_losses) != self.nums_task:
            raise ValueError(f"Expected {self.nums_task} task losses, got {len(task_losses)}.")
        shared_params = [p for p in shared_params if p.requires_grad]
        if not shared_params:
            return torch.stack(task_losses).sum()

        with torch.no_grad():
            loss_values = torch.tensor([loss.item() for loss in task_losses], device=self.weights.device)
            if self.initial_losses is None:
                self.initial_losses = loss_values.clone()

        weights_detached = self.weights.detach()
        weighted_losses = [weights_detached[i] * task_losses[i] for i in range(self.nums_task)]
        total_loss = torch.stack(weighted_losses).sum()

        grad_norms = self.compute_grad_norms(task_losses, shared_params)

        # compute inverse training rate, inv rate = loss_ratio / mean(loss_ratio)
        with torch.no_grad():
            if self.initial_losses is not None:
                base_initial = self.initial_losses
            else:
                base_initial = loss_values
            loss_ratios = loss_values / base_initial.clamp(min=self.eps)
            inv_rate = loss_ratios / loss_ratios.mean().clamp(min=self.eps)
            target = grad_norms.mean() * (inv_rate**self.alpha)

        grad_norm_loss = F.l1_loss(grad_norms, target.detach(), reduction="sum")
        grad_w = torch.autograd.grad(grad_norm_loss, self.weights, retain_graph=True)[0]
        self.pending_grad = grad_w.detach()

        return total_loss

    def compute_grad_norms(self, task_losses, shared_params):
        grad_norms = []
        # compute gradient norms for each task, gradient norms = sqrt(sum(grad^2))
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
                total_norm = torch.sqrt(sq_sum.clamp(min=self.eps))

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
            safe_weight_sum = w.sum().clamp(min=self.eps)
            w = w * self.nums_task / safe_weight_sum
            self.weights.copy_(w)

        self.pending_grad = None
