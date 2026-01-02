"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
- [1] Tang H, Liu J, Zhao M, Gong X. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. In: Proceedings of the 14th ACM Conference on Recommender Systems (RecSys ’20), 2020, pp. 269–278.
URL: https://dl.acm.org/doi/10.1145/3383313.3412236

Progressive Layered Extraction (PLE) advances multi-task learning by stacking CGC
(Customized Gate Control) blocks that mix shared and task-specific experts. Each
layer routes information via task gates and a shared gate, then feeds the outputs
forward to deeper layers, progressively disentangling shared vs. task-specific
signals and mitigating gradient interference.

Layer workflow:
  (1) Shared and per-task experts transform the same inputs
  (2) Task gates select among shared + task-specific experts
  (3) A shared gate aggregates all experts for the shared branch
  (4) Outputs become inputs to the next CGC layer (progressive refinement)
  (5) Final task towers operate on the last-layer task representations

Key Advantages:
- Progressive routing reduces negative transfer across layers
- Explicit shared/specific experts improve feature disentanglement
- Flexible depth and expert counts to match task complexity
- Works with heterogeneous features via unified embeddings
- Stable training by separating gates for shared and task branches

PLE（Progressive Layered Extraction）通过堆叠 CGC 模块，联合共享与任务特定专家，
利用任务门与共享门逐层软路由，逐步分离共享与任务差异信息，缓解多任务间的梯度冲突。

层内流程：
  (1) 共享与任务专家对同一输入做特征变换
  (2) 任务门在共享+任务专家上进行软选择
  (3) 共享门汇总全部专家，更新共享分支
  (4) 输出作为下一层输入，完成逐层细化
  (5) 最后由任务塔完成各任务预测

主要优点：
- 逐层路由降低负迁移
- 显式区分共享/特定专家，增强特征解耦
- 专家数量与层数可按任务复杂度灵活设置
- 统一 embedding 支持多种特征类型
- 共享与任务门分离，训练更稳定
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import get_mlp_output_dim


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
        specific_dims = [
            get_mlp_output_dim(params, input_dim) for params in specific_params_list
        ]
        dims_set = set(specific_dims + [self.output_dim])
        if len(dims_set) != 1:
            raise ValueError(
                f"Shared/specific expert output dims must match, got {dims_set}"
            )

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
            raise ValueError(
                f"Expected {self.nums_task} task inputs, got {len(task_inputs)}"
            )

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


class PLE(BaseModel):
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
        task: str | list[str] | None = None,
        **kwargs,
    ):

        self.nums_task = len(target) if target is not None else 1

        shared_expert_mlp_params = shared_expert_mlp_params or {}
        if specific_expert_mlp_params is None:
            raise ValueError(
                "specific_expert_mlp_params must be a list of dicts, one per task."
            )
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
        if (
            "hidden_dims" in shared_expert_mlp_params
            and len(shared_expert_mlp_params["hidden_dims"]) > 0
        ):
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
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["cgc_layers", "towers"]
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Initial features for each task and shared
        task_fea = [input_flat for _ in range(self.nums_task)]
        shared_fea = input_flat

        # Progressive Layered Extraction: CGC
        for layer in self.cgc_layers:
            task_fea, shared_fea = layer(task_fea, shared_fea)

        # task tower
        task_outputs = []
        for task_idx in range(self.nums_task):
            tower_output = self.towers[task_idx](task_fea[task_idx])  # [B, 1]
            task_outputs.append(tower_output)

        # [B, nums_task]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
