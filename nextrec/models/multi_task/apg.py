"""
Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Yan B, Wang P, Zhang K, Li F, Deng H, Xu J, Zheng B. APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction. Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022.
URL: https://arxiv.org/abs/2203.16218
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.model import select_features
from nextrec.utils.types import ActivationName, TaskTypeName


class APGLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scene_emb_dim: int,
        activation: ActivationName = "relu",
        generate_activation: ActivationName | None = None,
        inner_activation: ActivationName | None = None,
        use_uv_shared: bool = True,
        use_mf_p: bool = False,
        mf_k: int = 16,
        mf_p: int = 4,
    ) -> None:
        super().__init__()
        self.use_uv_shared = use_uv_shared
        self.use_mf_p = use_mf_p
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activation = (
            activation_layer(activation) if activation is not None else nn.Identity()
        )
        self.inner_activation = (
            activation_layer(inner_activation) if inner_activation is not None else None
        )

        min_dim = min(int(input_dim), int(output_dim))
        self.p_dim = math.ceil(float(min_dim) / float(mf_p))
        self.k_dim = math.ceil(float(min_dim) / float(mf_k))

        if use_uv_shared:
            if use_mf_p:
                self.shared_weight_np = nn.Parameter(
                    torch.empty(self.input_dim, self.p_dim)
                )
                self.shared_bias_np = nn.Parameter(torch.zeros(self.p_dim))
                self.shared_weight_pk = nn.Parameter(
                    torch.empty(self.p_dim, self.k_dim)
                )
                self.shared_bias_pk = nn.Parameter(torch.zeros(self.k_dim))

                self.shared_weight_kp = nn.Parameter(
                    torch.empty(self.k_dim, self.p_dim)
                )
                self.shared_bias_kp = nn.Parameter(torch.zeros(self.p_dim))
                self.shared_weight_pm = nn.Parameter(
                    torch.empty(self.p_dim, self.output_dim)
                )
                self.shared_bias_pm = nn.Parameter(torch.zeros(self.output_dim))
            else:
                self.shared_weight_nk = nn.Parameter(
                    torch.empty(self.input_dim, self.k_dim)
                )
                self.shared_bias_nk = nn.Parameter(torch.zeros(self.k_dim))
                self.shared_weight_km = nn.Parameter(
                    torch.empty(self.k_dim, self.output_dim)
                )
                self.shared_bias_km = nn.Parameter(torch.zeros(self.output_dim))
        self.specific_weight_kk = MLP(
            input_dim=scene_emb_dim,
            hidden_dims=None,
            output_dim=self.k_dim * self.k_dim,
            activation="relu",
            output_activation=generate_activation or "none",
        )
        self.specific_bias_kk = MLP(
            input_dim=scene_emb_dim,
            hidden_dims=None,
            output_dim=self.k_dim,
            activation="relu",
            output_activation=generate_activation or "none",
        )
        if not use_uv_shared:
            self.specific_weight_nk = MLP(
                input_dim=scene_emb_dim,
                hidden_dims=None,
                output_dim=self.input_dim * self.k_dim,
                activation="relu",
                output_activation=generate_activation or "none",
            )
            self.specific_bias_nk = MLP(
                input_dim=scene_emb_dim,
                hidden_dims=None,
                output_dim=self.k_dim,
                activation="relu",
                output_activation=generate_activation or "none",
            )
            self.specific_weight_km = MLP(
                input_dim=scene_emb_dim,
                hidden_dims=None,
                output_dim=self.k_dim * self.output_dim,
                activation="relu",
                output_activation=generate_activation or "none",
            )
            self.specific_bias_km = MLP(
                input_dim=scene_emb_dim,
                hidden_dims=None,
                output_dim=self.output_dim,
                activation="relu",
                output_activation=generate_activation or "none",
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_uv_shared:
            if self.use_mf_p:
                nn.init.xavier_uniform_(self.shared_weight_np)
                nn.init.zeros_(self.shared_bias_np)
                nn.init.xavier_uniform_(self.shared_weight_pk)
                nn.init.zeros_(self.shared_bias_pk)
                nn.init.xavier_uniform_(self.shared_weight_kp)
                nn.init.zeros_(self.shared_bias_kp)
                nn.init.xavier_uniform_(self.shared_weight_pm)
                nn.init.zeros_(self.shared_bias_pm)
            else:
                nn.init.xavier_uniform_(self.shared_weight_nk)
                nn.init.zeros_(self.shared_bias_nk)
                nn.init.xavier_uniform_(self.shared_weight_km)
                nn.init.zeros_(self.shared_bias_km)

    def forward(self, inputs: torch.Tensor, scene_emb: torch.Tensor) -> torch.Tensor:
        specific_weight_kk = self.specific_weight_kk(scene_emb)
        specific_weight_kk = specific_weight_kk.view(-1, self.k_dim, self.k_dim)
        specific_bias_kk = self.specific_bias_kk(scene_emb)

        if self.use_uv_shared:
            if self.use_mf_p:
                output_np = inputs @ self.shared_weight_np + self.shared_bias_np
                if self.inner_activation is not None:
                    output_np = self.inner_activation(output_np)
                output_pk = output_np @ self.shared_weight_pk + self.shared_bias_pk
                if self.inner_activation is not None:
                    output_pk = self.inner_activation(output_pk)
                output_kk = (
                    torch.bmm(output_pk.unsqueeze(1), specific_weight_kk).squeeze(1)
                    + specific_bias_kk
                )
                if self.inner_activation is not None:
                    output_kk = self.inner_activation(output_kk)
                output_kp = output_kk @ self.shared_weight_kp + self.shared_bias_kp
                if self.inner_activation is not None:
                    output_kp = self.inner_activation(output_kp)
                output = output_kp @ self.shared_weight_pm + self.shared_bias_pm
            else:
                output_nk = inputs @ self.shared_weight_nk + self.shared_bias_nk
                if self.inner_activation is not None:
                    output_nk = self.inner_activation(output_nk)
                output_kk = (
                    torch.bmm(output_nk.unsqueeze(1), specific_weight_kk).squeeze(1)
                    + specific_bias_kk
                )
                if self.inner_activation is not None:
                    output_kk = self.inner_activation(output_kk)
                output = output_kk @ self.shared_weight_km + self.shared_bias_km
        else:
            specific_weight_nk = self.specific_weight_nk(scene_emb)
            specific_weight_nk = specific_weight_nk.view(-1, self.input_dim, self.k_dim)
            specific_bias_nk = self.specific_bias_nk(scene_emb)
            specific_weight_km = self.specific_weight_km(scene_emb)
            specific_weight_km = specific_weight_km.view(
                -1, self.k_dim, self.output_dim
            )
            specific_bias_km = self.specific_bias_km(scene_emb)

            output_nk = (
                torch.bmm(inputs.unsqueeze(1), specific_weight_nk).squeeze(1)
                + specific_bias_nk
            )
            if self.inner_activation is not None:
                output_nk = self.inner_activation(output_nk)
            output_kk = (
                torch.bmm(output_nk.unsqueeze(1), specific_weight_kk).squeeze(1)
                + specific_bias_kk
            )
            if self.inner_activation is not None:
                output_kk = self.inner_activation(output_kk)
            output = (
                torch.bmm(output_kk.unsqueeze(1), specific_weight_km).squeeze(1)
                + specific_bias_km
            )

        return self.activation(output)


class APG(BaseModel):
    """
    Adaptive Parameter Generation (APG) model.

    APG stacks APG layers whose middle transformation matrix is generated from
    a scene embedding, enabling scenario-conditioned multi-task learning.
    """

    @property
    def model_name(self) -> str:
        return "APG"

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
        inner_activation: ActivationName | None = None,
        generate_activation: ActivationName | None = None,
        scene_features: list[str] | str | None = None,
        detach_scene: bool = True,
        use_uv_shared: bool = True,
        use_mf_p: bool = False,
        mf_k: int = 16,
        mf_p: int = 4,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        mlp_params.setdefault("hidden_dims", [256, 128])
        mlp_params.setdefault("activation", "relu")

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

        if not scene_features:
            raise ValueError("APG requires scene_features to generate parameters.")
        if isinstance(scene_features, str):
            scene_features = [scene_features]
        self.scene_features = select_features(
            self.all_features, scene_features, "scene_features"
        )
        self.detach_scene = detach_scene

        if len(mlp_params["hidden_dims"]) == 0:
            raise ValueError("mlp_params['hidden_dims'] cannot be empty for APG.")

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim
        scene_emb_dim = self.embedding.compute_output_dim(self.scene_features)

        layer_units = [input_dim] + list(mlp_params["hidden_dims"])
        self.apg_layers = nn.ModuleList(
            [
                APGLayer(
                    input_dim=layer_units[idx],
                    output_dim=layer_units[idx + 1],
                    scene_emb_dim=scene_emb_dim,
                    activation=mlp_params["activation"],
                    generate_activation=generate_activation,
                    inner_activation=inner_activation,
                    use_uv_shared=use_uv_shared,
                    use_mf_p=use_mf_p,
                    mf_k=mf_k,
                    mf_p=mf_p,
                )
                for idx in range(len(mlp_params["hidden_dims"]))
            ]
        )

        self.towers = nn.ModuleList(
            [nn.Linear(mlp_params["hidden_dims"][-1], 1) for _ in range(self.nums_task)]
        )
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )

        self.grad_norm_shared_modules = ["embedding", "apg_layers"]
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["apg_layers", "towers"]
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        scene_emb = self.embedding(x=x, features=self.scene_features, squeeze_dim=True)
        if self.detach_scene:
            scene_emb = scene_emb.detach()

        apg_output = input_flat
        for layer in self.apg_layers:
            apg_output = layer(apg_output, scene_emb)

        task_outputs = [tower(apg_output) for tower in self.towers]
        logits = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(logits)
