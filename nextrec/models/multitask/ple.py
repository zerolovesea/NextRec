"""
Date: create on 09/11/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou,zyaztec@gmail.com
Reference:
- [1] Tang H, Liu J, Zhao M, Gong X. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. In: Proceedings of the 14th ACM Conference on Recommender Systems (RecSys ’20), 2020, pp. 269–278.
URL: https://dl.acm.org/doi/10.1145/3383313.3412236

Progressive Layered Extraction (PLE) was proposed by Tencent's PCG team and won the Best Long Paper Award at RecSys 2020.
It is designed for multi-task scenarios where tasks are related but not fully consistent.
To address the issues of negative transfer and task interference,
PLE introduces a progressive layered extraction architecture with customized
gate control (CGC) that explicitly distinguishes between shared experts and
task-specific experts.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> input_flat: [Batch, Dim_embedding]
- CGC level 0 input:
  task_fea_i: [Batch, Dim_embedding], shared_fea: [Batch, Dim_embedding]
- CGC level k output:
  task_fea_i: [Batch, Dim_expert], shared_fea: [Batch, Dim_expert]
- Progressive stacking:
  (task_fea_i, shared_fea)_k -> CGC_k -> (task_fea_i, shared_fea)_{k+1}
- Task towers: task_fea_i: [Batch, Dim_expert] -> tower_i -> [Batch, 1]
- Concatenate logits: cat(tower_1 ... tower_T, dim=1) -> [Batch, Task_num]
- Output head: [Batch, Task_num] -> task activations -> [Batch, Task_num]

PLE 由腾讯PCG团队提出并获得RecSys 2020最佳长论文奖，主要用于“任务相关但不一致”的多任务场景。
为了解决跷跷板和负迁移的现象，论文提出了一种共享结构设计的渐进式分层提取（PLE）模型。
其包含两部分， 一部分是一种显式区分共享专家塔和特定任务专家塔的门控 (CGC) 模型，
另一部分是由单层CGC结构扩展到多层的PLE模型。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> input_flat: [Batch, Dim_embedding]
- 第 0 层 CGC 输入：
  task_fea_i: [Batch, Dim_embedding], shared_fea: [Batch, Dim_embedding]
- 第 k 层 CGC 输出：
  task_fea_i: [Batch, Dim_expert], shared_fea: [Batch, Dim_expert]
- 逐层堆叠：
  (task_fea_i, shared_fea)_k -> CGC_k -> (task_fea_i, shared_fea)_{k+1}
- 任务塔：task_fea_i: [Batch, Dim_expert] -> tower_i -> [Batch, 1]
- 任务拼接：cat(tower_1 ... tower_T, dim=1) -> [Batch, Task_num]
- 输出头：[Batch, Task_num] -> 各任务激活 -> [Batch, Task_num]

"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.utils.model import get_mlp_output_dim
from nextrec.utils.types import TaskTypeInput


class CGCLayer(nn.Module):
    """
    CGC (Customized Gate Control) block used by PLE.
    It routes shared and task-specific experts with task gates and a shared gate.
    """

    def __init__(
        self,
        input_dim: int,
        nums_task: int,
        num_shared_experts: int,
        num_specific_experts: int,
        shared_expert_mlp_params: dict,
        specific_expert_mlp_params: list[dict],
    ):
        super().__init__()
        if nums_task < 1:
            raise ValueError("nums_task must be >= 1")

        if len(specific_expert_mlp_params) != nums_task:
            raise ValueError(
                "Length of specific_expert_mlp_params "
                f"({len(specific_expert_mlp_params)}) must match number of tasks ({nums_task})."
            )
        specific_params_list = [params.copy() for params in specific_expert_mlp_params]

        self.output_dim = get_mlp_output_dim(shared_expert_mlp_params, input_dim)
        specific_dims = [get_mlp_output_dim(params, input_dim) for params in specific_params_list]
        dims_set = set(specific_dims + [self.output_dim])
        if len(dims_set) != 1:
            raise ValueError(f"Shared/specific expert output dims must match, got {dims_set}")

        # experts
        self.shared_experts = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim,
                    output_dim=None,
                    **shared_expert_mlp_params,
                )
                for _ in range(num_shared_experts)
            ]
        )
        self.specific_experts = nn.ModuleList()
        for params in specific_params_list:
            task_experts = nn.ModuleList(
                [
                    MLP(
                        input_dim=input_dim,
                        output_dim=None,
                        **params,
                    )
                    for _ in range(num_specific_experts)
                ]
            )
            self.specific_experts.append(task_experts)

        # gates
        task_gate_expert_num = num_shared_experts + num_specific_experts
        self.task_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, task_gate_expert_num),
                    nn.Softmax(dim=1),
                )
                for _ in range(nums_task)
            ]
        )
        shared_gate_expert_num = num_shared_experts + num_specific_experts * nums_task
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, shared_gate_expert_num),
            nn.Softmax(dim=1),
        )

        self.nums_task = nums_task

    def forward(
        self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if len(task_inputs) != self.nums_task:
            raise ValueError(f"Expected {self.nums_task} task inputs, got {len(task_inputs)}")

        shared_outputs = [expert(shared_input) for expert in self.shared_experts]
        shared_stack = torch.stack(shared_outputs, dim=0)  # [num_shared, B, D]

        new_task_fea: list[torch.Tensor] = []
        all_specific_for_shared: list[torch.Tensor] = []

        for task_idx in range(self.nums_task):
            task_input = task_inputs[task_idx]
            task_specific_outputs = [expert(task_input) for expert in self.specific_experts[task_idx]]  # type: ignore
            all_specific_for_shared.extend(task_specific_outputs)
            specific_stack = torch.stack(task_specific_outputs, dim=0)

            all_experts = torch.cat([shared_stack, specific_stack], dim=0)
            all_experts_t = all_experts.permute(1, 0, 2)  # [B, num_expert, D]

            gate_weights = self.task_gates[task_idx](task_input).unsqueeze(2)
            gated_output = torch.sum(gate_weights * all_experts_t, dim=1)
            new_task_fea.append(gated_output)

        all_for_shared = all_specific_for_shared + shared_outputs
        all_for_shared_tensor = torch.stack(all_for_shared, dim=1)  # [B, num_all, D]
        shared_gate_weights = self.shared_gate(shared_input).unsqueeze(1)
        new_shared = torch.bmm(shared_gate_weights, all_for_shared_tensor).squeeze(1)

        return new_task_fea, new_shared


class PLE(BaseMultitaskModel):
    """
    Progressive Layered Extraction

    PLE is an advanced multi-task learning model that extends MMOE by introducing
    both task-specific experts and shared experts at each level. It uses a progressive
    routing mechanism where experts from level k feed into gates at level k+1.
    This design better captures task-specific and shared information progressively.
    """

    @property
    def model_name(self):
        return "PLE"

    @property
    def default_task(self):
        nums_task = self.nums_task if hasattr(self, "nums_task") else None
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature] | None = None,
        sparse_features: list[SparseFeature] | None = None,
        sequence_features: list[SequenceFeature] | None = None,
        shared_expert_mlp_params: dict | None = None,
        specific_expert_mlp_params: list[dict] | None = None,
        num_shared_experts: int = 2,
        num_specific_experts: int = 2,
        num_levels: int = 2,
        tower_mlp_params_list: list[dict] | None = None,
        target: list[str] | None = None,
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        **kwargs,
    ):
        """
        Initialize PLE model.
        初始化 PLE 模型。

        Args:
            shared_expert_mlp_params: Shared expert MLP parameters, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1}.
                共享专家 MLP 参数，例如 {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.1}。
            specific_expert_mlp_params: Per-task specific expert MLP parameter list.
                Length must equal number of tasks.
                每个任务的专属专家 MLP 参数列表，长度必须等于任务数。
            num_shared_experts: Number of shared experts in each CGC layer.
                每层 CGC 的共享专家数量。
            num_specific_experts: Number of task-specific experts per task in each CGC layer.
                每层 CGC 中每个任务的专属专家数量。
            num_levels: Number of stacked CGC layers (progressive depth).
                CGC 堆叠层数（渐进深度）。
            tower_mlp_params_list: Task tower MLP parameter list. Length must equal number of tasks.
                任务塔 MLP 参数列表，长度必须等于任务数。
        """

        self.nums_task = len(target) if target is not None else 1

        shared_expert_mlp_params = shared_expert_mlp_params or {}
        if specific_expert_mlp_params is None:
            raise ValueError("specific_expert_mlp_params must be a list of dicts, one per task.")
        tower_mlp_params_list = tower_mlp_params_list or []

        super(PLE, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Number of tasks, experts, and levels
        self.nums_task = len(target)
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_levels = num_levels

        if len(tower_mlp_params_list) != self.nums_task:
            raise ValueError(
                "Number of tower mlp params "
                f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})"
            )
        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        input_dim = self.embedding.input_dim
        # emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        # dense_input_dim = sum([(f.embedding_dim or 1) for f in dense_features])
        # input_dim = emb_dim_total + dense_input_dim

        # Get expert output dimension
        if "hidden_dims" in shared_expert_mlp_params and len(shared_expert_mlp_params["hidden_dims"]) > 0:
            expert_output_dim = shared_expert_mlp_params["hidden_dims"][-1]
        else:
            expert_output_dim = input_dim

        # Build CGC layers
        self.cgc_layers = nn.ModuleList()
        for level in range(num_levels):
            level_input_dim = input_dim if level == 0 else expert_output_dim
            cgc_layer = CGCLayer(
                input_dim=level_input_dim,
                nums_task=self.nums_task,
                num_shared_experts=num_shared_experts,
                num_specific_experts=num_specific_experts,
                shared_expert_mlp_params=shared_expert_mlp_params,
                specific_expert_mlp_params=specific_expert_mlp_params,
            )
            self.cgc_layers.append(cgc_layer)
            expert_output_dim = cgc_layer.output_dim
        self.grad_norm_shared_modules = ["embedding", "cgc_layers"]

        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_mlp_params in tower_mlp_params_list:
            tower = MLP(input_dim=expert_output_dim, output_dim=1, **tower_mlp_params)
            self.towers.append(tower)
        # Register regularization weights
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["cgc_layers", "towers"])

    def forward(self, x):
        # Embedding flatten: [Batch, Dim_embedding]
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Initialize branch inputs:
        # task_fea: list length = Task_num, each tensor [Batch, Dim_embedding]
        # shared_fea: [Batch, Dim_embedding]
        task_fea = [input_flat for _ in range(self.nums_task)]
        shared_fea = input_flat

        # Progressive CGC extraction:
        # (task_fea_i, shared_fea)_k -> CGC_k -> (task_fea_i, shared_fea)_{k+1}
        # After each layer, each branch tensor is [Batch, Dim_expert]
        for layer in self.cgc_layers:
            task_fea, shared_fea = layer(task_fea, shared_fea)

        # Task towers: task_fea_i [Batch, Dim_expert] -> tower_i -> [Batch, 1]
        task_outputs = []
        for task_idx in range(self.nums_task):
            tower_output = self.towers[task_idx](task_fea[task_idx])  # [B, 1]
            task_outputs.append(tower_output)

        logits = torch.cat(task_outputs, dim=1)  # [Batch, Task_num]
        return logits
