"""
Date: create on 01/01/2026
Checkpoint: edit on 17/02/2026
Author: Yang Zhou, zyaztec@gmail.com
[1] Zhao Z, Liu Y, Jin R, Zhu X, He X. HMOE: Improving Multi-Scenario Learning to Rank in E-commerce by Exploiting Task Relationships in the Label Space. Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM ’20), 2020, pp. 2069–2078.
URL: https://dl.acm.org/doi/10.1145/3340531.3412713
[2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation:
https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

Hierarchical Mixture-of-Experts (HMOE) was proposed by the Meituan team in ICDE 2023 for multi-task and
multi-scenario learning in e-commerce.
HMOE extends MMOE for multi-task and
multi-scenario ranking by adding a task-level relation modeling stage in
label/representation space.
Compared with vanilla MMOE, HMOE keeps shared experts + task gates, then
adds a task-weight network for each task to aggregate representations from
all task towers. Non-target tower features are detached during this
aggregation to reduce harmful gradient interference (negative transfer).

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- Experts: each expert(input_flat) -> [Batch, Dim_expert]
- Stack experts: [Num_experts, Batch, Dim_expert]
- Task gate: gate_t(input_flat) -> [Batch, Num_experts]
- Expert aggregation:
  [Batch, Num_experts] x [Batch, Num_experts, Dim_expert]
  -> mmoe_fea_t: [Batch, Dim_expert]
- Task tower: tower_t(mmoe_fea_t) -> tower_fea_t: [Batch, Dim_tower]
- Task-weight net: tw_t(input_flat) -> [Batch, Task_num]
- Cross-task feature fusion for task t:
  sum_k alpha_tk * tower_fea_k, where k != t uses stop-gradient(detach)
  -> fused_fea_t: [Batch, Dim_tower]
- Task logit: linear_t(fused_fea_t) -> [Batch, 1]
- Concatenate logits: cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- Output head: [Batch, Task_num] -> task activations -> [Batch, Task_num]

HMOE由美团团队发表于ICDE 2023，是针对电商多任务多场景学习提出的模型。
在 MMOE 的基础上进一步建模任务关系：先通过专家池与任务门控学习任务表征，
再使用任务权重网络将所有任务塔特征做二次融合。为避免跨任务反向传播带来的负迁移，
在融合非目标任务特征时使用 stop-gradient（detach）。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- 专家网络：expert(input_flat) -> [Batch, Dim_expert]
- 专家堆叠：stack 后 -> [Num_experts, Batch, Dim_expert]
- 任务门控：gate_t(input_flat) -> [Batch, Num_experts]
- 专家加权汇聚：
  [Batch, Num_experts] 与 [Batch, Num_experts, Dim_expert] 加权求和
  -> mmoe_fea_t: [Batch, Dim_expert]
- 任务塔：tower_t(mmoe_fea_t) -> tower_fea_t: [Batch, Dim_tower]
- 任务权重网络：tw_t(input_flat) -> [Batch, Task_num]
- 任务融合：
  sum_k alpha_tk * tower_fea_k，其中 k != t 使用 detach
  -> fused_fea_t: [Batch, Dim_tower]
- 任务输出：linear_t(fused_fea_t) -> [Batch, 1]
- 任务拼接：cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- 输出头：[Batch, Task_num] -> 各任务激活 -> [Batch, Task_num]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.utils.model import get_mlp_output_dim
from nextrec.utils.types import TaskTypeInput, TaskTypeName


class HMOE(BaseMultitaskModel):
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
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize HMOE model.
        初始化 HMOE 模型。

        Args:
            expert_mlp_params: Shared expert MLP parameters, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1}.
                共享专家 MLP 参数，例如 {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1}。
            num_experts: Number of shared experts.
                共享专家数量。
            gate_mlp_params: Per-task gate MLP parameters that output expert logits.
                任务门控 MLP 参数（输出专家权重 logits）。
            tower_mlp_params_list: Per-task tower MLP parameter list.
                Length must equal number of tasks.
                每个任务的塔网络 MLP 参数列表，长度必须等于任务数。
            task_weight_mlp_params: Per-task task-relation weight MLP parameter list.
                Each task outputs weights over all tasks for cross-task feature fusion.
                每个任务的任务关系权重网络参数列表；每个任务会输出对全部任务的融合权重。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        expert_mlp_params = (expert_mlp_params or {}).copy()
        gate_mlp_params = (gate_mlp_params or {}).copy()
        tower_mlp_params_list = tower_mlp_params_list or []

        # Experts/gates/towers/task-weight nets have fixed output_dim in HMOE;
        expert_mlp_params.pop("output_dim", None)
        gate_mlp_params.pop("output_dim", None)

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
            [MLP(input_dim=input_dim, output_dim=None, **expert_mlp_params) for _ in range(num_experts)]
        )
        expert_output_dim = get_mlp_output_dim(expert_mlp_params, input_dim)

        self.gates = nn.ModuleList(
            [MLP(input_dim=input_dim, output_dim=num_experts, **gate_mlp_params) for _ in range(self.nums_task)]
        )
        self.grad_norm_shared_modules = [
            "embedding",
            "experts",
            "gates",
            "task_weights",
        ]

        tower_params = [params.copy() for params in tower_mlp_params_list]
        for params in tower_params:
            params.pop("output_dim", None)
        tower_output_dims = [get_mlp_output_dim(params, expert_output_dim) for params in tower_params]
        if len(set(tower_output_dims)) != 1:
            raise ValueError(f"All tower output dims must match, got {tower_output_dims}.")
        tower_output_dim = tower_output_dims[0]

        self.towers = nn.ModuleList(
            [MLP(input_dim=expert_output_dim, output_dim=None, **params) for params in tower_params]
        )
        self.tower_logits = nn.ModuleList([nn.Linear(tower_output_dim, 1, bias=False) for _ in range(self.nums_task)])

        if task_weight_mlp_params is None:
            raise ValueError("task_weight_mlp_params must be a list of dicts.")
        if len(task_weight_mlp_params) != self.nums_task:
            raise ValueError(
                "Length of task_weight_mlp_params "
                f"({len(task_weight_mlp_params)}) must match number of tasks ({self.nums_task})."
            )
        task_weight_mlp_params_list = [params.copy() for params in task_weight_mlp_params]
        for params in task_weight_mlp_params_list:
            params.pop("output_dim", None)
        self.task_weights = nn.ModuleList(
            [MLP(input_dim=input_dim, output_dim=self.nums_task, **params) for params in task_weight_mlp_params_list]
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
        # Embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Experts:
        # each expert(input_flat): [Batch, Dim_expert]
        # stacked experts: [Num_experts, Batch, Dim_expert]
        expert_outputs = [expert(input_flat) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        expert_outputs_t = expert_outputs.permute(1, 0, 2)  # [Batch, Num_experts, Dim_expert]

        # MMOE routing + task towers:
        # gate_t: [Batch, Num_experts], gated_output: [Batch, Dim_expert]
        # tower_fea_t: [Batch, Dim_tower]
        tower_features = []
        for task_idx in range(self.nums_task):
            gate_logits = self.gates[task_idx](input_flat)
            gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(2)  # [Batch, Num_experts, 1]
            gated_output = torch.sum(gate_weights * expert_outputs_t, dim=1)
            tower_features.append(self.towers[task_idx](gated_output))

        # Task-relation weights:
        # task_weight_probs[t]: [Batch, Task_num]
        task_weight_probs = [torch.softmax(task_weight(input_flat), dim=1) for task_weight in self.task_weights]

        # Cross-task tower fusion for each task:
        # fused_fea_t: [Batch, Dim_tower], task_logit_t: [Batch, 1]
        task_logits = []
        for task_idx in range(self.nums_task):
            task_feat = task_weight_probs[task_idx][:, task_idx].view(-1, 1) * tower_features[task_idx]
            for other_idx in range(self.nums_task):
                if other_idx == task_idx:
                    continue
                task_feat = (
                    task_feat
                    + task_weight_probs[task_idx][:, other_idx].view(-1, 1) * tower_features[other_idx].detach()
                )
            task_logits.append(self.tower_logits[task_idx](task_feat))

        # logits: [Batch, Task_num] -> task head -> [Batch, Task_num]
        logits = torch.cat(task_logits, dim=1)
        return logits
