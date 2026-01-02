"""
Date: create on 01/01/2026 - prerelease version: still need to align with the source paper
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Sheng XR, Zhao L, Zhou G, Ding X, Dai B, Luo Q, Yang S, Lv J, Zhang C, Deng H, Zhu X. One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. arXiv preprint arXiv:2101.11427, 2021.
URL: https://arxiv.org/abs/2101.11427
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

STAR uses shared-specific linear layers to adapt representations per task while
optionally reusing shared parameters. It can also apply domain-specific batch
normalization on the first hidden layer when a domain mask is provided.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.activation import activation_layer
from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.heads import TaskHead
from nextrec.basic.layers import DomainBatchNorm, EmbeddingLayer
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeName


class SharedSpecificLinear(nn.Module):
    """
    Shared-specific linear layer: task-specific projection plus optional shared one.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nums_task: int,
        use_shared: bool = True,
    ) -> None:
        super().__init__()
        self.use_shared = use_shared
        self.shared = nn.Linear(input_dim, output_dim) if use_shared else None
        self.specific = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(nums_task)]
        )

    def forward(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        output = self.specific[task_idx](x)
        if self.use_shared and self.shared is not None:
            output = output + self.shared(x)
        return output


class STAR(BaseModel):
    """
    STAR: shared-specific multi-task tower with optional domain-specific batch norm.
    """

    @property
    def model_name(self) -> str:
        return "STAR"

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
        use_shared: bool = True,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        mlp_params = mlp_params or {}
        mlp_params.setdefault("hidden_dims", [256, 128])
        mlp_params.setdefault("activation", "relu")
        mlp_params.setdefault("dropout", 0.0)
        mlp_params.setdefault("norm_type", "none")

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

        if not mlp_params["hidden_dims"]:
            raise ValueError("mlp_params['hidden_dims'] must not be empty.")

        norm_type = mlp_params["norm_type"]
        self.dnn_use_bn = norm_type == "batch_norm"
        self.dnn_dropout = mlp_params["dropout"]

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        layer_units = [input_dim] + list(mlp_params["hidden_dims"])
        self.star_layers = nn.ModuleList(
            [
                SharedSpecificLinear(
                    input_dim=layer_units[idx],
                    output_dim=layer_units[idx + 1],
                    nums_task=self.nums_task,
                    use_shared=use_shared,
                )
                for idx in range(len(mlp_params["hidden_dims"]))
            ]
        )
        self.activation_layers = nn.ModuleList(
            [
                activation_layer(mlp_params["activation"])
                for _ in range(len(mlp_params["hidden_dims"]))
            ]
        )
        if mlp_params["dropout"] > 0:
            self.dropout_layers = nn.ModuleList(
                [
                    nn.Dropout(mlp_params["dropout"])
                    for _ in range(len(mlp_params["hidden_dims"]))
                ]
            )
        else:
            self.dropout_layers = nn.ModuleList(
                [nn.Identity() for _ in range(len(mlp_params["hidden_dims"]))]
            )

        self.domain_bn = (
            DomainBatchNorm(
                num_features=mlp_params["hidden_dims"][0], num_domains=self.nums_task
            )
            if self.dnn_use_bn
            else None
        )

        self.final_layer = SharedSpecificLinear(
            input_dim=mlp_params["hidden_dims"][-1],
            output_dim=1,
            nums_task=self.nums_task,
            use_shared=use_shared,
        )
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )

        self.grad_norm_shared_modules = ["embedding", "star_layers", "final_layer"]
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["star_layers", "final_layer"],
        )

    def forward(
        self, x: dict[str, torch.Tensor], domain_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        task_outputs = []
        for task_idx in range(self.nums_task):
            output = input_flat
            for layer_idx, layer in enumerate(self.star_layers):
                output = layer(output, task_idx)
                output = self.activation_layers[layer_idx](output)
                output = self.dropout_layers[layer_idx](output)
                if (
                    layer_idx == 0
                    and self.dnn_use_bn
                    and self.domain_bn is not None
                    and domain_mask is not None
                ):
                    output = self.domain_bn(output, domain_mask)
            task_outputs.append(self.final_layer(output, task_idx))

        logits = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(logits)
