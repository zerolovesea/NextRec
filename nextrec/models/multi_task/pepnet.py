"""
Date: create on 01/01/2026
Checkpoint: edit on 15/02/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Chang J, Zhang C, Hui Y, Leng D, Niu Y, Song Y, Gai K. PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information. In: Proceedings of the 29th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’23), 2023.
URL: https://arxiv.org/abs/2302.01115
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

PEPNet was proposed by Kuaishou team in 2023 for multi-task recommendation with significant
distribution shifts across scenarios/domains and user/item priors.
It consists of two levels of personalization:
1) EPNet: feature-level gate to modulate the shared backbone embedding.
2) PPNet: layer-wise parameter gate for each task tower, conditioned on priors.
This helps reduce cross-scenario mismatch while keeping multi-task sharing.

Dimension Flow:
- Input: dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding: all features -> dnn_input: [Batch, Dim_input]
- Domain context: domain_features -> domain_emb: [Batch, Dim_domain]
- Task prior context:
  cat(domain_emb, user_emb?, item_emb?) -> task_sf_emb: [Batch, Dim_prior]
- EPNet gate input:
  cat(dnn_input_detach, domain_emb) -> [Batch, Dim_input + Dim_domain]
- EPNet output:
  ep_gate: [Batch, Dim_input], o_ep = ep_gate * dnn_input -> [Batch, Dim_input]
- PPNet (for each task t):
  gate_input_t = cat(task_sf_emb, o_ep_detach) -> [Batch, Dim_prior + Dim_input]
  tower_t(o_ep, gate_input_t) -> logit_t: [Batch, 1]
- Logit concat:
  cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- Output head:
  [Batch, Task_num] -> task activations -> [Batch, Task_num]

PEPNet由快手团队发布于2023，用于多任务推荐场景，用于处理
样本在不同场景/域、用户与物品先验上差异明显的情况。它包含两级个性化机制：
1）EPNet：在特征层面对共享输入进行门控；
2）PPNet：在任务塔每一层进行条件参数门控。
两者结合可在保留任务共享的同时缓解跨场景分布偏移。

维度变化：
- 输入：dense[Batch] + sparse[Batch] + sequence[Batch, Length]
- Embedding：所有特征拼接展平 -> dnn_input: [Batch, Dim_input]
- 场景上下文：domain_features -> domain_emb: [Batch, Dim_domain]
- 任务先验上下文：
  cat(domain_emb, user_emb?, item_emb?) -> task_sf_emb: [Batch, Dim_prior]
- EPNet 门控输入：
  cat(dnn_input_detach, domain_emb) -> [Batch, Dim_input + Dim_domain]
- EPNet 输出：
  ep_gate: [Batch, Dim_input], o_ep = ep_gate * dnn_input -> [Batch, Dim_input]
- 每个任务的 PPNet：
  gate_input_t = cat(task_sf_emb, o_ep_detach) -> [Batch, Dim_prior + Dim_input]
  tower_t(o_ep, gate_input_t) -> logit_t: [Batch, 1]
- 任务拼接：
  cat(logit_1 ... logit_T, dim=1) -> [Batch, Task_num]
- 输出头：
  [Batch, Task_num] -> 各任务激活 -> [Batch, Task_num]

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
from nextrec.utils.types import TaskTypeInput, TaskTypeName


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
                raise ValueError("hidden_activations length must match hidden_units length.")
            activation_list = mlp_params["activation"]
        else:
            activation_list = [mlp_params["activation"]] * len(hidden_units)

        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        layer_units = [input_dim] + hidden_units
        for idx in range(len(layer_units) - 1):
            dense_layers: list[nn.Module] = [nn.Linear(layer_units[idx], layer_units[idx + 1], bias=use_bias)]
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
        task: TaskTypeInput | list[TaskTypeInput] | None = None,
        mlp_params: dict | None = None,
        feature_gate_mlp_params: dict | None = None,
        gate_mlp_params: dict | None = None,
        domain_features: list[str] | str | None = None,
        user_features: list[str] | str | None = None,
        item_features: list[str] | str | None = None,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize PEPNet model.
        初始化 PEPNet 模型。

        Args:
            mlp_params: Shared PPNet tower MLP settings, e.g.
                {"hidden_dims": [256, 128], "activation": "relu", "dropout": 0.0, "norm_type": "none"}.
                PPNet 任务塔的 MLP 配置。
            feature_gate_mlp_params: EPNet gate MLP settings, e.g.
                {"hidden_dim": 128, "activation": "relu", "dropout": 0.0, "use_bn": False}.
                EPNet 特征门控网络配置。
            gate_mlp_params: PPNet per-layer gate MLP settings, e.g.
                {"hidden_dim": None, "activation": "relu", "dropout": 0.0, "use_bn": False}.
                PPNet 各层门控网络配置。
            domain_features: Domain/scene feature name(s) used by EPNet and prior context.
                场景/域特征名，至少需要提供一个。
            user_features: Optional user prior feature name(s).
                用户先验特征名（可选）。
            item_features: Optional item prior feature name(s).
                物品先验特征名（可选）。
            use_bias: Whether to enable bias terms in linear layers.
                是否在线性层中启用 bias。
        """
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

        self.domain_features = select_features(self.all_features, self.scene_feature_names, "domain_features")
        self.user_features = select_features(self.all_features, self.user_feature_names, "user_features")
        self.item_features = select_features(self.all_features, self.item_feature_names, "item_features")

        if not self.all_features:
            raise ValueError("PepNet requires at least one input feature.")

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.get_input_dim(self.all_features)
        domain_dim = self.embedding.get_input_dim(self.domain_features)
        user_dim = self.embedding.get_input_dim(self.user_features) if self.user_features else 0
        item_dim = self.embedding.get_input_dim(self.item_features) if self.item_features else 0
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

        self.prediction_layer = TaskHead(task_type=self.task, task_dims=[1] * self.nums_task)
        self.grad_norm_shared_modules = ["embedding", "epnet"]
        self.register_regularization_weights(embedding_attr="embedding", include_modules=["epnet", "ppnet_blocks"])

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dnn_input = self.embedding(x=x, features=self.all_features, squeeze_dim=True)  # [Batch, Dim_input]
        domain_emb = self.embedding(
            x=x, features=self.domain_features, squeeze_dim=True
        ).detach()  # [Batch, Dim_domain]

        task_parts = [domain_emb]  # first prior block: [Batch, Dim_domain]
        if self.user_features:
            task_parts.append(
                self.embedding(x=x, features=self.user_features, squeeze_dim=True).detach()
            )  # [Batch, Dim_user]
        if self.item_features:
            task_parts.append(
                self.embedding(x=x, features=self.item_features, squeeze_dim=True).detach()
            )  # [Batch, Dim_item]
        task_sf_emb = torch.cat(task_parts, dim=-1)  # [Batch, Dim_prior]

        gate_input = torch.cat([dnn_input.detach(), domain_emb], dim=-1)  # [Batch, Dim_input + Dim_domain]
        dnn_input = self.epnet(gate_input) * dnn_input  # [Batch, Dim_input] (o_ep)

        task_logits = []
        for block in self.ppnet_blocks:
            # gate_input_t is built inside PPNet as cat(task_sf_emb, o_ep_detach): [Batch, Dim_prior + Dim_input]
            task_logits.append(block(o_ep=dnn_input, o_prior=task_sf_emb))  # each logit_t: [Batch, 1]

        y = torch.cat(task_logits, dim=1)  # [Batch, Task_num]
        return self.prediction_layer(y)  # [Batch, Task_num]
