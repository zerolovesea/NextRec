"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
[1] Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with
multi-gate mixture-of-experts[C]//KDD. 2018: 1930-1939.
(https://dl.acm.org/doi/10.1145/3219819.3220007)

Multi-gate Mixture-of-Experts (MMoE) extends shared-bottom multi-task learning by
introducing multiple experts and task-specific softmax gates. Each task learns its
own routing weights over the expert pool, enabling both shared and task-specialized
representations while alleviating gradient conflicts across tasks.

In each forward pass:
  (1) Shared embeddings encode all dense/sparse/sequence features
  (2) Each expert processes the same input to produce candidate shared representations
  (3) Every task gate outputs a simplex over experts to softly route information
  (4) The task-specific weighted sum is passed into its tower and prediction head

Key Advantages:
- Soft parameter sharing reduces negative transfer between heterogeneous tasks
- Gates let tasks adaptively allocate expert capacity based on their difficulty
- Supports many tasks without duplicating full networks
- Works with mixed feature types via unified embeddings
- Simple to scale the number of experts or gates for capacity control

MMoE（Multi-gate Mixture-of-Experts）是多任务学习框架，通过多个专家网络与
任务特定门控进行软路由，兼顾共享表示与任务特化，减轻梯度冲突问题。

一次前向流程：
  (1) 共享 embedding 统一编码稠密、稀疏与序列特征
  (2) 每个专家对相同输入进行特征变换，得到候选共享表示
  (3) 每个任务的门控产生对专家的概率分布，完成软选择与加权
  (4) 加权结果输入到对应任务的塔网络与预测头

主要优点：
- 软参数共享，缓解任务间负迁移
- 按任务难度自适应分配专家容量
- 便于扩展多任务，而无需复制完整网络
- 支持多种特征类型的统一建模
- 专家与门控数量可灵活调节以控制模型容量
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class MMOE(BaseModel):
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
        expert_params: dict | None = None,
        num_experts: int = 3,
        tower_params_list: list[dict] | None = None,
        target: list[str] | str | None = None,
        task: str | list[str] = "binary",
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        expert_params = expert_params or {}
        tower_params_list = tower_params_list or []
        optimizer_params = optimizer_params or {}
        if loss is None:
            loss = "bce"
        if target is None:
            target = []
        elif isinstance(target, str):
            target = [target]

        self.nums_task = len(target) if target else 1

        resolved_task = task
        if resolved_task is None:
            resolved_task = self.default_task
        elif isinstance(resolved_task, str):
            resolved_task = [resolved_task] * self.nums_task
        elif len(resolved_task) == 1 and self.nums_task > 1:
            resolved_task = resolved_task * self.nums_task
        elif len(resolved_task) != self.nums_task:
            raise ValueError(
                f"Length of task ({len(resolved_task)}) must match number of targets ({self.nums_task})."
            )

        super(MMOE, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=resolved_task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            **kwargs,
        )

        self.loss = loss

        # Number of tasks and experts
        self.nums_task = len(target)
        self.num_experts = num_experts

        if len(tower_params_list) != self.nums_task:
            raise ValueError(
                f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.nums_task})"
            )

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        # Expert networks (shared by all tasks)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = MLP(input_dim=input_dim, output_layer=False, **expert_params)
            self.experts.append(expert)

        # Get expert output dimension
        if "dims" in expert_params and len(expert_params["dims"]) > 0:
            expert_output_dim = expert_params["dims"][-1]
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
        for tower_params in tower_params_list:
            tower = MLP(input_dim=expert_output_dim, output_layer=True, **tower_params)
            self.towers.append(tower)
        self.prediction_layer = TaskHead(
            task_type=self.default_task, task_dims=[1] * self.nums_task
        )
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["experts", "gates", "towers"]
        )
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=self.loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Expert outputs: [num_experts, B, expert_dim]
        expert_outputs = [expert(input_flat) for expert in self.experts]
        expert_outputs = torch.stack(
            expert_outputs, dim=0
        )  # [num_experts, B, expert_dim]

        # Task-specific processing
        task_outputs = []
        for task_idx in range(self.nums_task):
            # Gate weights for this task: [B, num_experts]
            gate_weights = self.gates[task_idx](input_flat)  # [B, num_experts]

            # Weighted sum of expert outputs
            # gate_weights: [B, num_experts, 1]
            # expert_outputs: [num_experts, B, expert_dim]
            gate_weights = gate_weights.unsqueeze(2)  # [B, num_experts, 1]
            expert_outputs_t = expert_outputs.permute(
                1, 0, 2
            )  # [B, num_experts, expert_dim]
            gated_output = torch.sum(
                gate_weights * expert_outputs_t, dim=1
            )  # [B, expert_dim]

            # Tower output
            tower_output = self.towers[task_idx](gated_output)  # [B, 1]
            task_outputs.append(tower_output)

        # Stack outputs: [B, nums_task]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
