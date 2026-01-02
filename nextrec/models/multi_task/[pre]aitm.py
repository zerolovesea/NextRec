"""
Date: create on 01/01/2026 - prerelease version: need to overwrite compute_loss later
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Xi D, Chen Z, Yan P, Zhang Y, Zhu Y, Zhuang F, Chen Y. Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising. Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD ’21), 2021, pp. 3745–3755.
URL: https://arxiv.org/abs/2105.08489
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import get_mlp_output_dim
from nextrec.utils.types import TaskTypeName


class AITMTransfer(nn.Module):
    """Attentive information transfer from previous task to current task."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.prev_proj = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)

    def forward(self, prev_feat: torch.Tensor, curr_feat: torch.Tensor) -> torch.Tensor:
        prev = self.prev_proj(prev_feat).unsqueeze(1)
        curr = curr_feat.unsqueeze(1)
        stacked = torch.cat([prev, curr], dim=1)
        value = self.value(stacked)
        key = self.key(stacked)
        query = self.query(stacked)
        attn_scores = torch.sum(key * query, dim=2, keepdim=True) / math.sqrt(
            self.input_dim
        )
        attn = torch.softmax(attn_scores, dim=1)
        return torch.sum(attn * value, dim=1)


class AITM(BaseModel):
    """
    Attentive Information Transfer Multi-Task model.

    AITM learns task-specific representations and transfers information from
    task i-1 to task i via attention, enabling sequential task dependency modeling.
    """

    @property
    def model_name(self):
        return "AITM"

    @property
    def default_task(self):
        nums_task = getattr(self, "nums_task", None)
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        bottom_mlp_params: dict | list[dict] | None = None,
        tower_mlp_params_list: list[dict] | None = None,
        calibrator_alpha: float = 0.1,
        target: list[str] | str | None = None,
        task: list[TaskTypeName] | None = None,
        **kwargs,
    ):
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        bottom_mlp_params = bottom_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []
        self.calibrator_alpha = calibrator_alpha

        if target is None:
            raise ValueError("AITM requires target names for all tasks.")
        if isinstance(target, str):
            target = [target]

        self.nums_task = len(target)
        if self.nums_task < 2:
            raise ValueError("AITM requires at least 2 tasks.")

        super(AITM, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                "Number of tower mlp params "
                f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})."
            )

        bottom_mlp_params_list: list[dict]
        if isinstance(bottom_mlp_params, list):
            if len(bottom_mlp_params) != self.nums_task:
                raise ValueError(
                    "Number of bottom mlp params "
                    f"({len(bottom_mlp_params)}) must match number of tasks ({self.nums_task})."
                )
            bottom_mlp_params_list = [params.copy() for params in bottom_mlp_params]
        else:
            bottom_mlp_params_list = [
                bottom_mlp_params.copy() for _ in range(self.nums_task)
            ]

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        self.bottoms = nn.ModuleList(
            [
                MLP(input_dim=input_dim, output_dim=None, **params)
                for params in bottom_mlp_params_list
            ]
        )
        bottom_dims = [
            get_mlp_output_dim(params, input_dim) for params in bottom_mlp_params_list
        ]
        if len(set(bottom_dims)) != 1:
            raise ValueError(f"All bottom output dims must match, got {bottom_dims}.")
        bottom_output_dim = bottom_dims[0]

        self.transfers = nn.ModuleList(
            [AITMTransfer(bottom_output_dim) for _ in range(self.nums_task - 1)]
        )
        self.grad_norm_shared_modules = ["embedding", "transfers"]

        self.towers = nn.ModuleList(
            [
                MLP(input_dim=bottom_output_dim, output_dim=1, **params)
                for params in tower_mlp_params_list
            ]
        )
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["bottoms", "transfers", "towers"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        task_feats = [bottom(input_flat) for bottom in self.bottoms]

        for idx in range(1, self.nums_task):
            task_feats[idx] = self.transfers[idx - 1](
                task_feats[idx - 1], task_feats[idx]
            )

        task_outputs = [tower(task_feats[idx]) for idx, tower in enumerate(self.towers)]
        logits = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(logits)
