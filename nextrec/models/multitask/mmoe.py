"""
Date: create on 09/11/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou,zyaztec@gmail.com
Reference:
- [1] Ma J, Zhao Z, Yi X, Chen J, Hong L, Chi E H. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18), 2018, pp. 1930–1939.
URL: https://dl.acm.org/doi/10.1145/3219819.3220007

Multi-gate Mixture-of-Experts (MMoE) was proposed by Google in 2018 for multi-task learning scenarios
where tasks are related but not identical. Compared to shared-bottom architecture, MMoE achieves soft
routing through an "expert pool + task gating" mechanism: different tasks learn different weights over
the same set of experts, allowing for a balance between shared information and task-specific representations,
mitigating negative transfer and gradient conflicts.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- Experts: each expert(input_flat) -> [Batch, Dim_expert]
- Stack experts: [Num_experts, Batch, Dim_expert]
- Task gate: gate_t(input_flat) -> [Batch, Num_experts]
- Weighted expert sum: [Batch, Num_experts] x [Batch, Num_experts, Dim_expert]
  -> task_fea_t: [Batch, Dim_expert]
- Task tower: tower_t(task_fea_t) -> [Batch, 1]
- Concatenate logits: cat(tower_1 ... tower_T, dim=1) -> [Batch, Task_num]
- Output head: [Batch, Task_num] -> task activations -> [Batch, Task_num]

MMoE由Google在2018年提出，用于“任务相关但不完全一致”的多任务学习场景。
相较于 Shared-Bottom，MMoE 通过“专家池 + 任务门控”实现软路由：不同任务对同一组专家
学习不同权重，从而在共享信息与任务特化之间取得平衡，缓解负迁移与梯度冲突。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- 专家层：每个 expert(input_flat) -> [Batch, Dim_expert]
- 专家堆叠：stack 后 -> [Num_experts, Batch, Dim_expert]
- 任务门控：gate_t(input_flat) -> [Batch, Num_experts]
- 专家加权汇聚：[Batch, Num_experts] 与 [Batch, Num_experts, Dim_expert] 加权求和
  -> task_fea_t: [Batch, Dim_expert]
- 任务塔：tower_t(task_fea_t) -> [Batch, 1]
- 任务拼接：cat(tower_1 ... tower_T, dim=1) -> [Batch, Task_num]
- 输出头：[Batch, Task_num] -> 各任务激活 -> [Batch, Task_num]

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.utils.types import TaskTypeInput


class MMOE(BaseMultitaskModel):
    """
    Multi-gate Mixture-of-Experts

    MMOE improves upon shared-bottom architecture by using multiple expert networks
    and task-specific gating networks. Each task has its own gate that learns to
    weight the contributions of different experts, allowing for both task-specific
    and shared representations.
    """

    @property
    def model_name(self):
        return "MMOE"

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
        expert_mlp_params: dict | None = None,
        num_experts: int = 3,
        tower_mlp_params_list: list[dict] | None = None,
        target: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ):
        """
        Initialize MMOE model.
        初始化 MMOE 模型。

        Args:
            expert_mlp_params: Shared expert MLP parameters, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1}.
                专家网络 MLP 参数字典（所有任务共享同一组专家结构）。
            num_experts: Number of experts in expert pool.
                专家网络数量。
            tower_mlp_params_list: Per-task tower MLP parameter list. Its length must
                equal the number of tasks; each tower outputs one logit.
                每个任务对应一个 tower 的参数列表，长度必须等于任务数；每个 tower 输出一个 logit。
        """

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        expert_mlp_params = expert_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []

        if target is None:
            target = []
        elif isinstance(target, str):
            target = [target]

        self.nums_task = len(target) if target else 1

        super(MMOE, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Number of tasks and experts
        self.nums_task = len(target)
        self.num_experts = num_experts

        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                "Number of tower mlp params "
                f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})"
            )

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # Expert networks (shared by all tasks)
        expert_mlp_params = expert_mlp_params.copy()
        expert_mlp_params.pop("output_dim", None)

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = MLP(input_dim=input_dim, output_dim=None, **expert_mlp_params)
            self.experts.append(expert)

        # Get expert output dimension
        if "hidden_dims" in expert_mlp_params and len(expert_mlp_params["hidden_dims"]) > 0:
            expert_output_dim = expert_mlp_params["hidden_dims"][-1]
        else:
            expert_output_dim = input_dim

        # Task-specific gates
        self.gates = nn.ModuleList()
        for _ in range(self.nums_task):
            gate = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
            self.gates.append(gate)
        self.grad_norm_shared_modules = ["embedding", "experts", "gates"]

        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_mlp_params in tower_mlp_params_list:
            tower_params = tower_mlp_params.copy()
            # Ignore optional output_dim in params to avoid duplicate kwargs conflict.
            tower_params.pop("output_dim", None)
            tower = MLP(input_dim=expert_output_dim, output_dim=1, **tower_params)
            self.towers.append(tower)
        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["experts", "gates", "towers"])

    def forward(self, x):
        # Embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Expert outputs:
        # each expert(input_flat): [Batch, Dim_expert]
        # stacked experts: [Num_experts, Batch, Dim_expert]
        expert_outputs = [expert(input_flat) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        expert_outputs_t = expert_outputs.permute(1, 0, 2)  # [Batch, Num_experts, Dim_expert]

        # Task-specific routing and towers
        task_outputs = []
        for task_idx in range(self.nums_task):
            gate_weights = self.gates[task_idx](input_flat)  # [Batch, Num_experts]
            gate_weights = gate_weights.unsqueeze(2)  # [Batch, Num_experts, 1]
            # Weighted expert sum -> task_fea_t: [Batch, Dim_expert]
            gated_output = torch.sum(gate_weights * expert_outputs_t, dim=1)

            # tower_t: [Batch, Dim_expert] -> [Batch, 1]
            tower_output = self.towers[task_idx](gated_output)
            task_outputs.append(tower_output)

        logits = torch.cat(task_outputs, dim=1)  # [Batch, Task_num]
        return logits
