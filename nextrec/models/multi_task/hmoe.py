"""
Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
[1] Zhao Z, Liu Y, Jin R, Zhu X, He X. HMOE: Improving Multi-Scenario Learning to Rank in E-commerce by Exploiting Task Relationships in the Label Space. Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM ’20), 2020, pp. 2069–2078.
URL: https://dl.acm.org/doi/10.1145/3340531.3412713
[2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation:
https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

Hierarchical Mixture-of-Experts (HMOE) extends MMOE with task-to-task
feature aggregation. Each task builds a tower representation from expert
mixtures, then a task-weight network mixes all tower features with
stop-gradient on non-target tasks to reduce negative transfer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import get_mlp_output_dim
from nextrec.utils.types import TaskTypeName


class HMOE(BaseModel):
    """
    Hierarchical Mixture-of-Experts.
    """

    @property
    def model_name(self) -> str:
        return "HMOE"

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
        expert_mlp_params: dict | None = None,
        num_experts: int = 4,
        gate_mlp_params: dict | None = None,
        tower_mlp_params_list: list[dict] | None = None,
        task_weight_mlp_params: list[dict] | None = None,
        target: list[str] | str | None = None,
        task: TaskTypeName | list[TaskTypeName] | None = None,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        expert_mlp_params = expert_mlp_params or {}
        gate_mlp_params = gate_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []

        if target is None:
            target = []
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

        self.nums_task = len(target) if target else 1
        self.num_experts = num_experts

        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                "Number of tower mlp params "
                f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})."
            )

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        self.experts = nn.ModuleList(
            [
                MLP(input_dim=input_dim, output_dim=None, **expert_mlp_params)
                for _ in range(num_experts)
            ]
        )
        expert_output_dim = get_mlp_output_dim(expert_mlp_params, input_dim)

        self.gates = nn.ModuleList(
            [
                MLP(input_dim=input_dim, output_dim=num_experts, **gate_mlp_params)
                for _ in range(self.nums_task)
            ]
        )
        self.grad_norm_shared_modules = [
            "embedding",
            "experts",
            "gates",
            "task_weights",
        ]

        tower_params = [params.copy() for params in tower_mlp_params_list]
        tower_output_dims = [
            get_mlp_output_dim(params, expert_output_dim) for params in tower_params
        ]
        if len(set(tower_output_dims)) != 1:
            raise ValueError(
                f"All tower output dims must match, got {tower_output_dims}."
            )
        tower_output_dim = tower_output_dims[0]

        self.towers = nn.ModuleList(
            [
                MLP(input_dim=expert_output_dim, output_dim=None, **params)
                for params in tower_params
            ]
        )
        self.tower_logits = nn.ModuleList(
            [nn.Linear(tower_output_dim, 1, bias=False) for _ in range(self.nums_task)]
        )

        if task_weight_mlp_params is None:
            raise ValueError("task_weight_mlp_params must be a list of dicts.")
        if len(task_weight_mlp_params) != self.nums_task:
            raise ValueError(
                "Length of task_weight_mlp_params "
                f"({len(task_weight_mlp_params)}) must match number of tasks ({self.nums_task})."
            )
        task_weight_mlp_params_list = [
            params.copy() for params in task_weight_mlp_params
        ]
        self.task_weights = nn.ModuleList(
            [
                MLP(input_dim=input_dim, output_dim=self.nums_task, **params)
                for params in task_weight_mlp_params_list
            ]
        )

        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )

        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "experts",
                "gates",
                "task_weights",
                "towers",
                "tower_logits",
            ],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        expert_outputs = [expert(input_flat) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [E, B, D]
        expert_outputs_t = expert_outputs.permute(1, 0, 2)  # [B, E, D]

        tower_features = []
        for task_idx in range(self.nums_task):
            gate_logits = self.gates[task_idx](input_flat)
            gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(2)
            gated_output = torch.sum(gate_weights * expert_outputs_t, dim=1)
            tower_features.append(self.towers[task_idx](gated_output))

        task_weight_probs = [
            torch.softmax(task_weight(input_flat), dim=1)
            for task_weight in self.task_weights
        ]

        task_logits = []
        for task_idx in range(self.nums_task):
            task_feat = (
                task_weight_probs[task_idx][:, task_idx].view(-1, 1)
                * tower_features[task_idx]
            )
            for other_idx in range(self.nums_task):
                if other_idx == task_idx:
                    continue
                task_feat = (
                    task_feat
                    + task_weight_probs[task_idx][:, other_idx].view(-1, 1)
                    * tower_features[other_idx].detach()
                )
            task_logits.append(self.tower_logits[task_idx](task_feat))

        logits = torch.cat(task_logits, dim=1)
        return self.prediction_layer(logits)
