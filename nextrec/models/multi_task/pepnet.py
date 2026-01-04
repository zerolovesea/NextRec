"""
Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Chang J, Zhang C, Hui Y, Leng D, Niu Y, Song Y, Gai K. PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information. In: Proceedings of the 29th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’23), 2023.
URL: https://arxiv.org/abs/2302.01115
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

PEPNet (Parameter and Embedding Personalized Network) is a multi-task learning
model that personalizes both input features and layer transformations with
context (scene/domain, user, item). It applies a shared feature gate to the
backbone embedding and then uses per-task gated MLP blocks (PPNet blocks) whose
gates are conditioned on task-specific context. This enables task-aware routing
at both feature and layer levels, improving adaptation across scenarios/tasks.

Workflow:
  (1) Embed all features and build the backbone input
  (2) Build task context embedding from domain/user/item features
  (3) Feature gate masks backbone input using domain context
  (4) Each task tower applies layer-wise gates conditioned on context + backbone embedding output
  (5) Task heads produce per-task predictions

Key Advantages:
- Two-level personalization: feature gate + layer gates
- Context-driven routing for multi-scenario/multi-task recommendation
- Task towers share embeddings while adapting via gates
- Gate input uses stop-grad on backbone embedding output for stable training
- Compatible with heterogeneous features via unified embeddings

PEPNet（Parameter and Embedding Personalized Network）通过场景/用户/物品等上下文
对输入特征与网络层进行双层门控个性化。先用共享特征门控调整主干输入，再在每个
任务塔中使用条件门控的 MLP 层（PPNet block），实现任务与场景感知的逐层路由。

流程：
  (1) 对全部特征做 embedding，得到主干输入
  (2) 由场景/用户/物品特征构建任务上下文向量
  (3) 共享特征门控按场景调制主干输入
  (4) 任务塔逐层门控，结合上下文与主干 embedding 输出进行路由
  (5) 任务头输出各任务预测结果

主要优点：
- 特征级与层级双重个性化
- 上下文驱动的多场景/多任务适配
- 共享 embedding 的同时通过门控实现任务定制
- 对主干 embedding 输出 stop-grad，提高训练稳定性
- 统一 embedding 支持多类特征
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, GateMLP
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import select_features
from nextrec.utils.types import TaskTypeName


class PPNet(nn.Module):
    """
    PPNet: per-task tower with layer-wise gates conditioned on task context.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gate_input_dim: int,
        mlp_params: dict | None = None,
        gate_mlp_params: dict | None = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        mlp_params = mlp_params or {}
        gate_mlp_params = gate_mlp_params or {}

        mlp_params.setdefault("hidden_dims", [])
        mlp_params.setdefault("activation", "relu")
        mlp_params.setdefault("dropout", 0.0)
        mlp_params.setdefault("norm_type", "none")

        gate_mlp_params.setdefault("hidden_dim", None)
        gate_mlp_params.setdefault("activation", "relu")
        gate_mlp_params.setdefault("dropout", 0.0)
        gate_mlp_params.setdefault("use_bn", False)

        hidden_units = mlp_params["hidden_dims"]
        norm_type = mlp_params["norm_type"]

        if isinstance(mlp_params["dropout"], list):
            if len(mlp_params["dropout"]) != len(hidden_units):
                raise ValueError("dropout_rates length must match hidden_units length.")
            dropout_list = mlp_params["dropout"]
        else:
            dropout_list = [mlp_params["dropout"]] * len(hidden_units)

        if isinstance(mlp_params["activation"], list):
            if len(mlp_params["activation"]) != len(hidden_units):
                raise ValueError(
                    "hidden_activations length must match hidden_units length."
                )
            activation_list = mlp_params["activation"]
        else:
            activation_list = [mlp_params["activation"]] * len(hidden_units)

        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        layer_units = [input_dim] + hidden_units
        for idx in range(len(layer_units) - 1):
            dense_layers: list[nn.Module] = [
                nn.Linear(layer_units[idx], layer_units[idx + 1], bias=use_bias)
            ]
            if norm_type == "batch_norm":
                dense_layers.append(nn.BatchNorm1d(layer_units[idx + 1]))
            dense_layers.append(activation_layer(activation_list[idx]))
            if dropout_list[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_list[idx]))

            self.gate_layers.append(
                GateMLP(
                    input_dim=gate_input_dim,
                    hidden_dim=gate_mlp_params["hidden_dim"],
                    output_dim=layer_units[idx],
                    activation=gate_mlp_params["activation"],
                    dropout=gate_mlp_params["dropout"],
                    use_bn=gate_mlp_params["use_bn"],
                    scale_factor=2.0,
                )
            )
            self.mlp_layers.append(nn.Sequential(*dense_layers))

        self.gate_layers.append(
            GateMLP(
                input_dim=gate_input_dim,
                hidden_dim=gate_mlp_params["hidden_dim"],
                output_dim=layer_units[-1],
                activation=gate_mlp_params["activation"],
                dropout=gate_mlp_params["dropout"],
                use_bn=gate_mlp_params["use_bn"],
                scale_factor=1.0,
            )
        )
        self.mlp_layers.append(nn.Linear(layer_units[-1], output_dim, bias=use_bias))

    def forward(self, o_ep: torch.Tensor, o_prior: torch.Tensor) -> torch.Tensor:
        """
        o_ep: EPNet output embedding (will be stop-grad in gate input)
        o_prior: prior/task context embedding
        """
        gate_input = torch.cat([o_prior, o_ep.detach()], dim=-1)

        hidden = o_ep
        for gate, mlp in zip(self.gate_layers, self.mlp_layers):
            gw = gate(gate_input)
            hidden = mlp(hidden * gw)
        return hidden


class PEPNet(BaseModel):
    """
    PEPNet: feature-gated multi-task tower with task-conditioned gates.
    """

    @property
    def model_name(self) -> str:
        return "PepNet"

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
        target: list[str] | str | None = None,
        task: TaskTypeName | list[TaskTypeName] | None = None,
        mlp_params: dict | None = None,
        feature_gate_mlp_params: dict | None = None,
        gate_mlp_params: dict | None = None,
        domain_features: list[str] | str | None = None,
        user_features: list[str] | str | None = None,
        item_features: list[str] | str | None = None,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        feature_gate_mlp_params = feature_gate_mlp_params or {}
        gate_mlp_params = gate_mlp_params or {}

        mlp_params.setdefault("hidden_dims", [256, 128])
        mlp_params.setdefault("activation", "relu")
        mlp_params.setdefault("dropout", 0.0)
        mlp_params.setdefault("norm_type", "none")

        feature_gate_mlp_params.setdefault("hidden_dim", 128)
        feature_gate_mlp_params.setdefault("activation", "relu")
        feature_gate_mlp_params.setdefault("dropout", 0.0)
        feature_gate_mlp_params.setdefault("use_bn", False)

        gate_mlp_params.setdefault("hidden_dim", None)
        gate_mlp_params.setdefault("activation", "relu")
        gate_mlp_params.setdefault("dropout", 0.0)
        gate_mlp_params.setdefault("use_bn", False)

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

        if isinstance(domain_features, str):
            domain_features = [domain_features]
        if isinstance(user_features, str):
            user_features = [user_features]
        if isinstance(item_features, str):
            item_features = [item_features]

        self.scene_feature_names = list(domain_features or [])
        self.user_feature_names = list(user_features or [])
        self.item_feature_names = list(item_features or [])

        if not self.scene_feature_names:
            raise ValueError("PepNet requires at least one scene feature name.")

        self.domain_features = select_features(
            self.all_features, self.scene_feature_names, "domain_features"
        )
        self.user_features = select_features(
            self.all_features, self.user_feature_names, "user_features"
        )
        self.item_features = select_features(
            self.all_features, self.item_feature_names, "item_features"
        )

        if not self.all_features:
            raise ValueError("PepNet requires at least one input feature.")

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.get_input_dim(self.all_features)
        domain_dim = self.embedding.get_input_dim(self.domain_features)
        user_dim = (
            self.embedding.get_input_dim(self.user_features)
            if self.user_features
            else 0
        )
        item_dim = (
            self.embedding.get_input_dim(self.item_features)
            if self.item_features
            else 0
        )
        task_dim = domain_dim + user_dim + item_dim

        # EPNet: shared feature-level gate (paper's EPNet).
        self.epnet = GateMLP(
            input_dim=input_dim + domain_dim,
            hidden_dim=feature_gate_mlp_params["hidden_dim"],
            output_dim=input_dim,
            activation=feature_gate_mlp_params["activation"],
            dropout=feature_gate_mlp_params["dropout"],
            use_bn=feature_gate_mlp_params["use_bn"],
            scale_factor=2.0,
        )

        # PPNet: per-task gated towers (paper's PPNet).
        self.ppnet_blocks = nn.ModuleList(
            [
                PPNet(
                    input_dim=input_dim,
                    output_dim=1,
                    gate_input_dim=input_dim + task_dim,
                    mlp_params=mlp_params,
                    gate_mlp_params=gate_mlp_params,
                    use_bias=use_bias,
                )
                for _ in range(self.nums_task)
            ]
        )

        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )
        self.grad_norm_shared_modules = ["embedding", "epnet"]
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["epnet", "ppnet_blocks"]
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dnn_input = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        domain_emb = self.embedding(
            x=x, features=self.domain_features, squeeze_dim=True
        ).detach()

        task_parts = [domain_emb]
        if self.user_features:
            task_parts.append(
                self.embedding(
                    x=x, features=self.user_features, squeeze_dim=True
                ).detach()
            )
        if self.item_features:
            task_parts.append(
                self.embedding(
                    x=x, features=self.item_features, squeeze_dim=True
                ).detach()
            )
        task_sf_emb = torch.cat(task_parts, dim=-1)

        gate_input = torch.cat([dnn_input.detach(), domain_emb], dim=-1)
        dnn_input = self.epnet(gate_input) * dnn_input

        task_logits = []
        for block in self.ppnet_blocks:
             task_logits.append(block(o_ep=dnn_input, o_prior=task_sf_emb))

        y = torch.cat(task_logits, dim=1)
        return self.prediction_layer(y)
