"""
Date: create on 01/01/2026 - prerelease version: still need to align with the source paper
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Ma J, Zhao Z, Chen J, Li A, Hong L, Chi EH. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning in E-Commerce by Exploiting Task Relationships in the Label Space. Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI 2019), 2019, pp. 216-223.
URL: https://ojs.aaai.org/index.php/AAAI/article/view/3788
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

SNR-Trans stacks multiple expert layers and applies sparse routing with
learnable per-output transform matrices. Intermediate gates route expert
outputs to the next expert stage, while the final gate routes to each task
tower for prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeName


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
            raise ValueError(
                f"SNRTransGate expects {self.num_inputs} inputs, got {len(inputs)}"
            )

        s = torch.sigmoid(
            torch.log(self.u)
            - torch.log(1 - self.u)
            + torch.log(self.alpha) / self.beta
        )
        s_ = s * (self.epsilon - self.gamma) + self.gamma
        z = torch.clamp(s_, min=0.0, max=1.0)

        x_stack = torch.stack(inputs, dim=1)  # [B, num_inputs, units]
        transformed = torch.einsum(
            "bnu,onuv->bonv", x_stack, self.trans_matrix
        )  # [B, num_outputs, num_inputs, units]
        weighted = transformed * z.unsqueeze(0).unsqueeze(-1)
        outputs = weighted.sum(dim=2)  # [B, num_outputs, units]
        return [outputs[:, i, :] for i in range(self.num_outputs)]


class SNRTrans(BaseModel):
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
        task: TaskTypeName | list[TaskTypeName] | None = None,
        **kwargs,
    ) -> None:
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
            output_dim = (
                self.nums_task
                if idx == len(expert_hidden_dims) - 1
                else self.num_experts
            )
            self.gates.append(
                SNRTransGate(
                    num_inputs=self.num_experts,
                    num_outputs=output_dim,
                    units=hidden_dim,
                )
            )
            prev_dim = hidden_dim

        self.towers = nn.ModuleList(
            [
                MLP(input_dim=expert_hidden_dims[-1], output_dim=1, **params)
                for params in tower_params
            ]
        )
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )
        self.grad_norm_shared_modules = ["embedding", "expert_layers", "gates"]
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["expert_layers", "gates", "towers"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        gate_outputs: list[torch.Tensor] | None = None
        for layer_idx, (layer_experts, gate) in enumerate(
            zip(self.expert_layers, self.gates)
        ):
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

        task_outputs = []
        for task_idx in range(self.nums_task):
            tower_output = self.towers[task_idx](gate_outputs[task_idx])
            task_outputs.append(tower_output)

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
