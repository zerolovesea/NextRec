"""
Date: create on 01/01/2026 - prerelease version: still need to align with the source paper
Checkpoint: edit on 07/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Ma J, Zhao Z, Chen J, Li A, Hong L, Chi EH. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning in E-Commerce by Exploiting Task Relationships in the Label Space. Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI 2019), 2019, pp. 216-223.
URL: https://ojs.aaai.org/index.php/AAAI/article/view/3788
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

SNR (Sub-Network Routing) was proposed by Alibaba in AAAI 2019 for
multi-task recommendation where tasks are correlated but not identical.
Compared with shared-bottom and vanilla MoE-style sharing, SNR introduces
learnable sparse routing masks between sub-networks so each task can select
different transfer paths layer by layer.

This SNR-Trans variant stacks multiple expert layers and uses a sparse gate
with per-output transform matrices. Intermediate gates route expert outputs
to the next expert stage, while the final gate routes to each task tower.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- Expert layer k:
  each expert_k(input): [Batch, Dim_k], total experts: Num_experts
- Gate layer k:
  inputs [Num_experts, Batch, Dim_k] -> routed outputs:
  if k is not last layer: [Num_experts, Batch, Dim_k]
  if k is last layer: [Task_num, Batch, Dim_k]
- Task towers:
  tower_t(gate_out_t): [Batch, 1]
- Concatenate logits: cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- Output head: [Batch, Task_num] -> task activations -> [Batch, Task_num]

SNR（Sub-Network Routing）由阿里巴巴在 AAAI 2019 提出，面向“任务相关但不完全一致”的多任务推荐场景。
相较于 shared-bottom 和普通 MoE 共享方式，SNR 在子网络之间引入可学习的稀疏路由，
使不同任务能够逐层选择不同的信息传递路径，从而缓解负迁移。

本实现采用 SNR-Trans 形式：堆叠多层专家网络，并在门控中加入按输出节点划分的变换矩阵。
中间层 gate 将专家输出路由到下一层专家，最后一层 gate 将路由结果分发给各任务塔。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- 第 k 层专家：
  每个 expert_k(input) -> [Batch, Dim_k]，共 Num_experts 个专家
- 第 k 层门控：
  输入 [Num_experts, Batch, Dim_k] -> 路由输出：
  若不是最后一层 -> [Num_experts, Batch, Dim_k]
  若是最后一层 -> [Task_num, Batch, Dim_k]
- 任务塔：
  tower_t(gate_out_t) -> [Batch, 1]
- 任务拼接：cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- 输出头：[Batch, Task_num] -> 各任务激活 -> [Batch, Task_num]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.utils.types import TaskTypeInput, TaskTypeName


class SNRTransGate(nn.Module):
    """
    Sparse routing gate with per-output transform matrices.
    """

    def __init__(self, num_inputs: int, num_outputs: int, units: int) -> None:
        super().__init__()
        if num_inputs < 1 or num_outputs < 1:
            raise ValueError("num_inputs and num_outputs must be >= 1")

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.units = units

        self.alpha = nn.Parameter(torch.rand(1), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.epsilon = 1.1
        self.eps = 1e-8

        u_init = torch.empty(num_outputs, num_inputs)
        u_init = nn.init.uniform_(u_init, self.eps, 1 - self.eps)
        self.u = nn.Parameter(u_init, requires_grad=True)

        trans = torch.empty(num_outputs, num_inputs, units, units)
        nn.init.xavier_normal_(trans)
        self.trans_matrix = nn.Parameter(trans, requires_grad=True)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs) != self.num_inputs:
            raise ValueError(f"SNRTransGate expects {self.num_inputs} inputs, got {len(inputs)}")

        s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        s_ = s * (self.epsilon - self.gamma) + self.gamma
        z = torch.clamp(s_, min=0.0, max=1.0)

        x_stack = torch.stack(inputs, dim=1)  # [B, num_inputs, units]
        transformed = torch.einsum("bnu,onuv->bonv", x_stack, self.trans_matrix)  # [B, num_outputs, num_inputs, units]
        weighted = transformed * z.unsqueeze(0).unsqueeze(-1)
        outputs = weighted.sum(dim=2)  # [B, num_outputs, units]
        return [outputs[:, i, :] for i in range(self.num_outputs)]


class SNRTrans(BaseMultitaskModel):
    """
    SNR-Trans with sparse expert routing.
    """

    @property
    def model_name(self) -> str:
        return "SNRTrans"

    @property
    def default_task(self) -> TaskTypeName | list[TaskTypeName]:
        nums_task = self.nums_task if hasattr(self, "nums_task") else None
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
        tower_mlp_params_list: list[dict] | None = None,
        target: list[str] | str | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize SNR-Trans model.
        初始化 SNR-Trans 模型。

        Args:
            expert_mlp_params: Shared expert-layer MLP parameters, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1, "norm_type": "none"}.
                共享专家层 MLP 参数，例如 {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1, "norm_type": "none"}。
            num_experts: Number of experts in each expert layer.
                每一层专家网络的专家数量。
            tower_mlp_params_list: Per-task tower MLP parameter list.
                Length must equal number of tasks; each tower outputs one logit.
                每个任务对应一个塔网络参数字典，长度必须等于任务数；每个塔输出 1 个 logit。
        """
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        expert_mlp_params = expert_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []

        expert_mlp_params.setdefault("hidden_dims", [256, 128])
        expert_mlp_params.setdefault("activation", "relu")
        expert_mlp_params.setdefault("dropout", 0.0)
        expert_mlp_params.setdefault("norm_type", "none")
        expert_hidden_dims = expert_mlp_params["hidden_dims"]

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

        if self.nums_task <= 1:
            raise ValueError("SNRTrans requires at least 2 tasks.")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1.")
        if not expert_hidden_dims:
            raise ValueError("expert_mlp_params['hidden_dims'] must not be empty.")

        if tower_mlp_params_list:
            if len(tower_mlp_params_list) != self.nums_task:
                raise ValueError(
                    "Number of tower mlp params "
                    f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})."
                )
            tower_params = [params.copy() for params in tower_mlp_params_list]
        else:
            tower_params = [{} for _ in range(self.nums_task)]
        for params in tower_params:
            # SNRTrans fixes each task tower output to 1 logit.
            # Ignore optional output_dim in tower params to avoid kwargs conflict.
            params.pop("output_dim", None)

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        self.expert_layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(expert_hidden_dims):
            layer_experts = nn.ModuleList(
                [
                    MLP(
                        input_dim=prev_dim,
                        hidden_dims=[hidden_dim],
                        output_dim=None,
                        dropout=expert_mlp_params["dropout"],
                        activation=expert_mlp_params["activation"],
                        norm_type=expert_mlp_params["norm_type"],
                    )
                    for _ in range(self.num_experts)
                ]
            )
            self.expert_layers.append(layer_experts)
            output_dim = self.nums_task if idx == len(expert_hidden_dims) - 1 else self.num_experts
            self.gates.append(
                SNRTransGate(
                    num_inputs=self.num_experts,
                    num_outputs=output_dim,
                    units=hidden_dim,
                )
            )
            prev_dim = hidden_dim

        self.towers = nn.ModuleList(
            [MLP(input_dim=expert_hidden_dims[-1], output_dim=1, **params) for params in tower_params]
        )
        self.grad_norm_shared_modules = ["embedding", "expert_layers", "gates"]
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["expert_layers", "gates", "towers"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # Embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Progressive expert routing:
        # layer 0 input for each expert: [Batch, Dim_embedding]
        # layer k>0 input for each expert: [Batch, Dim_k-1]
        # each expert output at layer k: [Batch, Dim_k]
        # gate outputs:
        #   non-last layer: Num_experts tensors, each [Batch, Dim_k]
        #   last layer: Task_num tensors, each [Batch, Dim_last]
        gate_outputs: list[torch.Tensor] | None = None
        for layer_idx, (layer_experts, gate) in enumerate(zip(self.expert_layers, self.gates)):
            expert_outputs = []
            if layer_idx == 0:
                expert_inputs = [input_flat] * self.num_experts
            else:
                if gate_outputs is None:
                    raise RuntimeError("SNRTrans gate outputs are not initialized.")
                expert_inputs = gate_outputs
            for expert, expert_input in zip(layer_experts, expert_inputs):
                expert_outputs.append(expert(expert_input))
            gate_outputs = gate(expert_outputs)

        if gate_outputs is None or len(gate_outputs) != self.nums_task:
            raise RuntimeError("SNRTrans gate outputs do not match task count.")

        # Task towers:
        # gate_outputs[task_idx]: [Batch, Dim_last]
        # tower output/logit per task: [Batch, 1]
        task_outputs = []
        for task_idx in range(self.nums_task):
            tower_output = self.towers[task_idx](gate_outputs[task_idx])
            task_outputs.append(tower_output)

        # Concatenate logits: [Batch, Task_num] -> prediction head -> [Batch, Task_num]
        logits = torch.cat(task_outputs, dim=1)
        return logits
