"""
Date: create on 28/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
[1] Wang et al. "POSO: Personalized Cold Start Modules for Large-scale Recommender Systems", 2021.

POSO (Personalized cOld-start mOdules) augments backbone recommenders by injecting a
personalized cold-start vector `pc` that gates hidden units layer by layer. Each fully
connected layer or expert output is multiplied by gate(pc), letting the backbone adapt
its hidden representations to user profiles even when behavioral signals are scarce.

Core idea:
  (1) A lightweight two-layer MLP maps `pc` to gate(pc) = C * sigmoid(W2 * phi(W1 * pc + b1) + b2)
  (2) gate(pc) scales each hidden unit element-wise, masking or amplifying features
  (3) Existing task gates/towers remain intact; POSO only overlays personalization

Key advantages:
- Plug-and-play personalization for cold-start users without redesigning the backbone
- Per-layer/expert gating with minimal additional parameters
- Compatible with plain MLP towers and MMoE structures, keeping training stable
- Works with split features: main features feed the backbone, PC features drive gates

POSO 通过个性化冷启动向量 `pc` 为推荐模型叠加逐层的门控系数，
在每个全连接层或专家输出上乘以 gate(pc) 做元素级缩放，
即使行为信号稀缺也能按用户画像调整隐藏表示。

实现思路：
  (1) 用轻量两层 MLP 生成 gate(pc) = C * sigmoid(W2 * phi(W1 * pc + b1) + b2)
  (2) gate(pc) 对神经元逐元素放大或抑制
  (3) 原有任务门/塔不变，POSO 仅叠加个性化门控

主要优点：
- 冷启动场景的可插拔个性化，无需重做骨干结构
- 每层/每专家独立门控，新增参数量小
- 兼容 MLP、MMoE 等多任务骨干，训练过程平稳
- 主特征做建模，PC 特征驱动门控，解耦表征与个性化信号
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import select_features


class POSOGate(nn.Module):
    """
    Two-layer MLP that maps the personalized cold-start vector to a gate:
        gate(pc) = C * sigmoid( W2 * phi(W1 * pc + b1) + b2 )
    The output shares the same dimension as the hidden vector to be masked and
    is applied element-wise.
    """

    def __init__(
        self,
        pc_dim: int,
        out_dim: int,
        hidden_dim: int = 32,
        scale_factor: float = 2.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(pc_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.scale_factor = scale_factor
        self.act = activation_layer(activation)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        pc: (B, pc_dim)
        return: (B, out_dim) in (0, C)
        """
        h = self.act(self.fc1(pc))
        g = torch.sigmoid(self.fc2(h))  # (B, out_dim) in (0,1)
        return self.scale_factor * g


class POSOFC(nn.Module):
    """
    Single POSO fully connected layer mirroring Eq. (11):
        h = phi(Wx + b)
        h_hat = gate(pc) ⊙ h
    where gate(pc) = C * sigmoid(MLP(pc)).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pc_dim: int,
        gate_hidden_dim: int = 32,
        scale_factor: float = 2.0,
        activation: str = "relu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.act = activation_layer(activation)
        self.gate = POSOGate(
            pc_dim=pc_dim,
            out_dim=out_dim,
            hidden_dim=gate_hidden_dim,
            scale_factor=scale_factor,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """
        x:  (B, in_dim)
        pc: (B, pc_dim)
        return: (B, out_dim)
        """
        h = self.act(self.linear(x))  # Standard FC with activation
        g = self.gate(pc)  # (B, out_dim)
        return g * h  # Element-wise gating


class POSOMLP(nn.Module):
    """
    POSO-enhanced MLP that stacks multiple POSOFC layers.

    dims: e.g., [256, 128, 64] means
        in_dim -> 256 -> 128 -> 64
    Each layer has its own gate g_l(pc) following Eq. (11).
    """

    def __init__(
        self,
        input_dim: int,
        pc_dim: int,
        dims: list[int],
        gate_hidden_dim: int = 32,
        scale_factor: float = 2.0,
        activation: str = "relu",
        use_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim
        for out_dim in dims:
            layers.append(
                POSOFC(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    pc_dim=pc_dim,
                    gate_hidden_dim=gate_hidden_dim,
                    scale_factor=scale_factor,
                    activation=activation,
                    use_bias=use_bias,
                )
            )
            in_dim = out_dim

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """
        x:  (B, input_dim)
        pc: (B, pc_dim)
        """
        h = x
        for layer in self.layers:
            h = layer(h, pc)
            if self.dropout is not None:
                h = self.dropout(h)
        return h


class POSOMMoE(nn.Module):
    """
    POSO(MMoE) mirrors Section 4.4 and Eq. (15)-(18) of the paper:
        - Keep the original experts and task gates gate_t(x)
        - Add a PC gate g_e(pc) for every expert_e
        - Task gates aggregate the PC-masked expert outputs

    Concretely:
        h_e = expert_e(x)                 # (B, D)
        g_e = POSOGate(pc) in (0, C)^{D}  # (B, D)
        h_e_tilde = g_e ⊙ h_e            # (B, D)
        z_t = Σ_e gate_t,e(x) * h_e_tilde
    """

    def __init__(
        self,
        input_dim: int,
        pc_dim: int,  # for poso feature dimension
        num_experts: int,
        expert_hidden_dims: list[int],
        nums_task: int,
        activation: str = "relu",
        expert_dropout: float = 0.0,
        gate_hidden_dim: int = 32,  # for poso gate hidden dimension
        scale_factor: float = 2.0,  # for poso gate scale factor
        gate_use_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.nums_task = nums_task

        # Experts built with framework MLP, same as standard MMoE
        self.experts = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim,
                    output_layer=False,
                    dims=expert_hidden_dims,
                    activation=activation,
                    dropout=expert_dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.expert_output_dim = (
            expert_hidden_dims[-1] if expert_hidden_dims else input_dim
        )

        # Task-specific gates: gate_t(x) over experts
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, num_experts) for _ in range(nums_task)]
        )
        self.gate_use_softmax = gate_use_softmax

        # PC gate per expert: g_e(pc) ∈ R^D
        self.expert_pc_gates = nn.ModuleList(
            [
                POSOGate(
                    pc_dim=pc_dim,
                    out_dim=self.expert_output_dim,
                    hidden_dim=gate_hidden_dim,
                    scale_factor=scale_factor,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor, pc: torch.Tensor) -> list[torch.Tensor]:
        """
        x:  (B, input_dim)
        pc: (B, pc_dim)
        return: list of task outputs z_t with length nums_task, each (B, D)
        """
        # 1) Expert outputs with POSO PC gate
        masked_expert_outputs = []
        for e, expert in enumerate(self.experts):
            h_e = expert(x)  # (B, D)
            g_e = self.expert_pc_gates[e](pc)  # (B, D)
            h_e_tilde = g_e * h_e  # (B, D)
            masked_expert_outputs.append(h_e_tilde)

        masked_expert_outputs = torch.stack(masked_expert_outputs, dim=1)  # (B, E, D)

        # 2) Task gates depend on x as in standard MMoE
        task_outputs: list[torch.Tensor] = []
        for t in range(self.nums_task):
            logits = self.gates[t](x)  # (B, E)
            if self.gate_use_softmax:
                gate = F.softmax(logits, dim=1)
            else:
                gate = logits

            gate = gate.unsqueeze(-1)  # (B, E, 1)
            z_t = torch.sum(gate * masked_expert_outputs, dim=1)  # (B, D)
            task_outputs.append(z_t)

        return task_outputs


class POSO(BaseModel):
    """
    POSO model implemented with the NextRec framework. It supports two backbones:
    - "mlp": per-task POSO-MLP towers with PC gating on every hidden layer
    - "mmoe": POSO-gated MMoE experts plus task-specific towers
    """

    @property
    def model_name(self) -> str:
        return "POSO"

    @property
    def default_task(self) -> list[str]:
        nums_task = getattr(self, "nums_task", None)
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature] | None,
        sparse_features: list[SparseFeature] | None,
        sequence_features: list[SequenceFeature] | None,
        main_dense_features: list[str] | None,
        main_sparse_features: list[str] | None,
        main_sequence_features: list[str] | None,
        pc_dense_features: list[str] | None,
        pc_sparse_features: list[str] | None,
        pc_sequence_features: list[str] | None,
        tower_params_list: list[dict],
        target: list[str],
        task: str | list[str] = "binary",
        architecture: str = "mlp",
        # POSO gating defaults
        gate_hidden_dim: int = 32,
        gate_scale_factor: float = 2.0,
        gate_activation: str = "relu",
        gate_use_bias: bool = True,
        # MMoE-specific params
        num_experts: int = 4,
        expert_hidden_dims: list[int] | None = None,
        expert_activation: str = "relu",
        expert_dropout: float = 0.0,
        expert_gate_hidden_dim: int = 32,
        expert_gate_scale_factor: float = 2.0,
        gate_use_softmax: bool = True,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg: float = 1e-6,
        dense_l1_reg: float = 1e-5,
        embedding_l2_reg: float = 1e-5,
        dense_l2_reg: float = 1e-4,
        **kwargs,
    ):
        self.nums_task = len(target)

        # Normalize task to match nums_task
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

        if len(tower_params_list) != self.nums_task:
            raise ValueError(
                f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.nums_task})"
            )

        super().__init__(
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

        self.main_dense_feature_names = list(main_dense_features or [])
        self.main_sparse_feature_names = list(main_sparse_features or [])
        self.main_sequence_feature_names = list(main_sequence_features or [])
        self.pc_dense_feature_names = list(pc_dense_features or [])
        self.pc_sparse_feature_names = list(pc_sparse_features or [])
        self.pc_sequence_feature_names = list(pc_sequence_features or [])

        if loss is None:
            self.loss = "bce"
        self.loss = loss

        optimizer_params = optimizer_params or {}

        self.main_dense_features = select_features(
            self.dense_features, self.main_dense_feature_names, "main_dense_features"
        )
        self.main_sparse_features = select_features(
            self.sparse_features, self.main_sparse_feature_names, "main_sparse_features"
        )
        self.main_sequence_features = select_features(
            self.sequence_features,
            self.main_sequence_feature_names,
            "main_sequence_features",
        )

        self.pc_dense_features = select_features(
            self.dense_features, self.pc_dense_feature_names, "pc_dense_features"
        )
        self.pc_sparse_features = select_features(
            self.sparse_features, self.pc_sparse_feature_names, "pc_sparse_features"
        )
        self.pc_sequence_features = select_features(
            self.sequence_features,
            self.pc_sequence_feature_names,
            "pc_sequence_features",
        )

        self.main_features = (
            self.main_dense_features
            + self.main_sparse_features
            + self.main_sequence_features
        )
        self.pc_features = (
            self.pc_dense_features + self.pc_sparse_features + self.pc_sequence_features
        )

        if not self.main_features:
            raise ValueError("POSO requires at least one main feature.")
        if not self.pc_features:
            raise ValueError(
                "POSO requires at least one PC feature for personalization."
            )

        self.embedding = EmbeddingLayer(features=self.all_features)
        self.main_input_dim = self.embedding.get_input_dim(self.main_features)
        self.pc_input_dim = self.embedding.get_input_dim(self.pc_features)

        self.architecture = architecture.lower()
        if self.architecture not in {"mlp", "mmoe"}:
            raise ValueError(
                f"Unsupported architecture '{architecture}', choose from ['mlp', 'mmoe']."
            )

        # Build backbones
        if self.architecture == "mlp":
            self.towers = nn.ModuleList()
            self.tower_heads = nn.ModuleList()
            for tower_params in tower_params_list:
                dims = tower_params.get("dims")
                if not dims:
                    raise ValueError(
                        "tower_params must include a non-empty 'dims' list for POSO-MLP towers."
                    )
                dropout = tower_params.get("dropout", 0.0)
                tower = POSOMLP(
                    input_dim=self.main_input_dim,
                    pc_dim=self.pc_input_dim,
                    dims=dims,
                    gate_hidden_dim=tower_params.get(
                        "gate_hidden_dim", gate_hidden_dim
                    ),
                    scale_factor=tower_params.get("scale_factor", gate_scale_factor),
                    activation=tower_params.get("activation", gate_activation),
                    use_bias=tower_params.get("use_bias", gate_use_bias),
                    dropout=dropout,
                )
                self.towers.append(tower)
                tower_output_dim = dims[-1] if dims else self.main_input_dim
                self.tower_heads.append(nn.Linear(tower_output_dim, 1))
        else:
            if expert_hidden_dims is None or not expert_hidden_dims:
                raise ValueError(
                    "expert_hidden_dims must be provided for MMoE architecture."
                )
            self.mmoe = POSOMMoE(
                input_dim=self.main_input_dim,
                pc_dim=self.pc_input_dim,
                num_experts=num_experts,
                expert_hidden_dims=expert_hidden_dims,
                nums_task=self.nums_task,
                activation=expert_activation,
                expert_dropout=expert_dropout,
                gate_hidden_dim=expert_gate_hidden_dim,
                scale_factor=expert_gate_scale_factor,
                gate_use_softmax=gate_use_softmax,
            )
            self.towers = nn.ModuleList(
                [
                    MLP(
                        input_dim=self.mmoe.expert_output_dim,
                        output_layer=True,
                        **tower_params,
                    )
                    for tower_params in tower_params_list
                ]
            )
            self.tower_heads = None
        if self.architecture == "mlp":
            self.grad_norm_shared_modules = ["embedding"]
        else:
            self.grad_norm_shared_modules = ["embedding", "mmoe"]
        self.prediction_layer = TaskHead(
            task_type=self.default_task,
            task_dims=[1] * self.nums_task,
        )
        include_modules = (
            ["towers", "tower_heads"]
            if self.architecture == "mlp"
            else ["mmoe", "towers"]
        )
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=include_modules
        )
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
        )

    def forward(self, x):
        # Embed main and PC features separately so PC can gate hidden units
        main_input = self.embedding(x=x, features=self.main_features, squeeze_dim=True)
        pc_input = self.embedding(x=x, features=self.pc_features, squeeze_dim=True)

        task_outputs = []
        if self.architecture == "mlp":
            for tower, head in zip(self.towers, self.tower_heads):
                hidden = tower(main_input, pc_input)
                logit = head(hidden)
                task_outputs.append(logit)
        else:
            expert_outputs = self.mmoe(main_input, pc_input)
            for idx, tower in enumerate(self.towers):
                logit = tower(expert_outputs[idx])
                task_outputs.append(logit)

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
