"""
Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Wang H, Chang T-W, Liu T, Huang J, Chen Z, Yu C, Li R, Chu W. ESCM²: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation. Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22), 2022:363–372.
URL: https://arxiv.org/abs/2204.05125
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

Entire Space Counterfactual Model (ESCM) extends ESMM with counterfactual
training objectives (e.g., IPS/DR) to debias CVR estimation. The architecture
keeps separate CTR/CVR towers and derives CTCVR as the product of probabilities.
Optional exposure propensity (IMP) prediction is included for DR-style variants.
"""

from __future__ import annotations

import torch

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.heads import TaskHead
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.basic.model import BaseModel
from nextrec.loss.grad_norm import get_grad_norm_shared_params
from nextrec.utils.model import compute_ranking_loss
from nextrec.utils.types import TaskTypeName


class ESCM(BaseModel):
    """
    Entire Space Counterfactual Model.
    """

    @property
    def model_name(self) -> str:
        return "ESCM"

    @property
    def default_task(self) -> TaskTypeName | list[TaskTypeName]:
        nums_task = getattr(self, "nums_task", None)
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        ctr_mlp_params: dict | None = None,
        cvr_mlp_params: dict | None = None,
        imp_mlp_params: dict | None = None,
        use_dr: bool = False,
        target: list[str] | str | None = None,
        task: TaskTypeName | list[TaskTypeName] | None = None,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        ctr_mlp_params = ctr_mlp_params or {}
        cvr_mlp_params = cvr_mlp_params or {}
        imp_mlp_params = imp_mlp_params or {}

        if target is None:
            target = ["ctr", "cvr", "ctcvr"]
            if use_dr:
                target.append("imp")
        elif isinstance(target, str):
            target = [target]

        self.nums_task = len(target) if target else 1

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        if not target:
            raise ValueError("ESCM requires at least one target.")

        valid_targets = {"ctr", "cvr", "ctcvr", "imp"}
        default_roles = ["ctr", "cvr", "ctcvr", "imp"]
        if all(name in valid_targets for name in target):
            target_roles = list(target)
        else:
            if len(target) > len(default_roles):
                raise ValueError(
                    f"ESCM supports up to {len(default_roles)} targets, got {len(target)}."
                )
            target_roles = default_roles[: len(target)]

        self.target_roles = target_roles
        self.use_dr = use_dr or ("imp" in self.target_roles)
        base_targets = ["ctr", "cvr"]
        if self.use_dr:
            base_targets.append("imp")

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        self.ctr_tower = MLP(input_dim=input_dim, output_dim=1, **ctr_mlp_params)
        self.cvr_tower = MLP(input_dim=input_dim, output_dim=1, **cvr_mlp_params)
        if self.use_dr:
            self.imp_tower = MLP(input_dim=input_dim, output_dim=1, **imp_mlp_params)

        self.base_task_types = ["binary"] * len(base_targets)
        self.prediction_layer = TaskHead(
            task_type=self.base_task_types, task_dims=[1] * len(base_targets)
        )

        self.grad_norm_shared_modules = ["embedding"]
        reg_modules = ["ctr_tower", "cvr_tower"]
        if self.use_dr:
            reg_modules.append("imp_tower")
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=reg_modules
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        ctr_logit = self.ctr_tower(input_flat)
        cvr_logit = self.cvr_tower(input_flat)
        base_logits = [ctr_logit, cvr_logit]
        if self.use_dr:
            imp_logit = self.imp_tower(input_flat)
            base_logits.append(imp_logit)

        base_logits_cat = torch.cat(base_logits, dim=1)
        base_preds = self.prediction_layer(base_logits_cat)
        base_preds = base_preds.split(1, dim=1)

        pred_map = {"ctr": base_preds[0], "cvr": base_preds[1]}
        if self.use_dr:
            pred_map["imp"] = base_preds[2]

        ctcvr_pred = pred_map["ctr"] * pred_map["cvr"]

        outputs = []
        for name in self.target_roles:
            if name == "ctcvr":
                outputs.append(ctcvr_pred)
            else:
                outputs.append(pred_map[name])
        return torch.cat(outputs, dim=1)

    def _loss_no_reduce(
        self,
        loss_fn: torch.nn.Module,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(loss_fn, "reduction"):
            reduction = loss_fn.reduction
            if reduction != "none":
                loss_fn.reduction = "none"
                loss = loss_fn(y_pred, y_true)
                loss_fn.reduction = reduction
            else:
                loss = loss_fn(y_pred, y_true)
        else:
            loss = loss_fn(y_pred, y_true)

        if loss.dim() == 0:
            return loss
        if loss.dim() > 1:
            loss = loss.view(loss.size(0), -1).mean(dim=1)
        return loss.view(-1)

    def _compute_cvr_loss(
        self,
        loss_fn: torch.nn.Module,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        click_label: torch.Tensor | None,
        prop_pred: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        if click_label is None:
            return loss_fn(y_pred.view(-1), y_true.view(-1))

        click = click_label
        if valid_mask is not None:
            click = click[valid_mask]
        click = click.detach()

        if prop_pred is not None:
            prop = prop_pred
            if valid_mask is not None:
                prop = prop[valid_mask]
            prop = prop.detach()
            prop = torch.clamp(prop, min=eps, max=1.0 - eps)
            weight = (click / prop).view(-1)
        else:
            weight = click.view(-1)

        per_sample = self._loss_no_reduce(loss_fn, y_pred, y_true).view(-1)
        if self.use_dr and prop_pred is not None:
            impute_target = y_pred.detach()
            impute_loss = self._loss_no_reduce(loss_fn, y_pred, impute_target).view(-1)
            return (impute_loss + weight * (per_sample - impute_loss)).mean()
        return (per_sample * weight).mean()

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true is None:
            raise ValueError(
                "[ESCM-compute_loss Error] Ground truth labels (y_true) are required."
            )

        if y_pred.dim() == 1:
            y_pred = y_pred.view(-1, 1)
        if y_true.dim() == 1:
            y_true = y_true.view(-1, 1)

        role_to_index = {role: idx for idx, role in enumerate(self.target_roles)}
        ctr_index = role_to_index.get("ctr")
        imp_index = role_to_index.get("imp")

        ctr_pred = (
            y_pred[:, ctr_index : ctr_index + 1] if ctr_index is not None else None
        )
        ctr_true = (
            y_true[:, ctr_index : ctr_index + 1] if ctr_index is not None else None
        )
        imp_pred = (
            y_pred[:, imp_index : imp_index + 1] if imp_index is not None else None
        )

        task_losses = []
        for i, role in enumerate(self.target_roles):
            y_pred_i = y_pred[:, i : i + 1]
            y_true_i = y_true[:, i : i + 1]
            valid_mask = None
            if self.ignore_label is not None:
                valid_mask = y_true_i != self.ignore_label
                if valid_mask.dim() > 1:
                    valid_mask = valid_mask.all(dim=1)
                if not torch.any(valid_mask):
                    task_losses.append(y_pred_i.sum() * 0.0)
                    continue
                y_pred_i = y_pred_i[valid_mask]
                y_true_i = y_true_i[valid_mask]

            if role == "cvr":
                prop_pred = imp_pred if self.use_dr else ctr_pred
                if prop_pred is None:
                    prop_pred = ctr_pred
                task_loss = self._compute_cvr_loss(
                    loss_fn=self.loss_fn[i],
                    y_pred=y_pred_i,
                    y_true=y_true_i,
                    click_label=ctr_true,
                    prop_pred=prop_pred,
                    valid_mask=valid_mask,
                )
            else:
                mode = self.training_modes[i]
                if mode in {"pairwise", "listwise"}:
                    task_loss = compute_ranking_loss(
                        training_mode=mode,
                        loss_fn=self.loss_fn[i],
                        y_pred=y_pred_i,
                        y_true=y_true_i,
                    )
                elif y_pred_i.shape[1] == 1:
                    task_loss = self.loss_fn[i](y_pred_i.view(-1), y_true_i.view(-1))
                else:
                    task_loss = self.loss_fn[i](y_pred_i, y_true_i)
            task_losses.append(task_loss)

        if self.grad_norm is not None:
            if self.grad_norm_shared_params is None:
                self.grad_norm_shared_params = get_grad_norm_shared_params(
                    self, getattr(self, "grad_norm_shared_modules", None)
                )
            return self.grad_norm.compute_weighted_loss(
                task_losses, self.grad_norm_shared_params
            )
        if isinstance(self.loss_weights, (list, tuple)):
            task_losses = [
                task_loss * self.loss_weights[i]
                for i, task_loss in enumerate(task_losses)
            ]
        return torch.stack(task_losses).sum()
