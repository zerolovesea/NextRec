"""
Date: create on 01/01/2026
Checkpoint: edit on 01/01/2026
Author: Yang Zhou, zyaztec@gmail.com
Reference:
- [1] Misra I, Shrivastava A, Gupta A, Hebert M. Cross-Stitch Networks for Multi-Task Learning. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), 2016, pp. 3994–4003.
URL: https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html
- [2] MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation: https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation/

Cross-Stitch networks mix task-specific representations with a learnable
linear combination at each layer, enabling soft sharing while preserving
task-specific subspaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import EmbeddingLayer, MLP
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel
from nextrec.utils.types import TaskTypeName


class CrossStitchLayer(nn.Module):
    """
    Cross-stitch layer to linearly mix task-specific representations.
    """

    def __init__(self, input_dims: list[int]) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("CrossStitchLayer requires at least 2 inputs.")
        self.input_dims = list(input_dims)
        if len(set(self.input_dims)) != 1:
            raise ValueError(
                "CrossStitchLayer expects all input dims to be equal to align channels."
            )
        self.num_tasks = len(self.input_dims)
        self.unit_dim = self.input_dims[0]
        identity = torch.eye(self.num_tasks).unsqueeze(-1)
        weight = identity.repeat(1, 1, self.unit_dim)
        self.cross_stitch_weight = nn.Parameter(weight)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs) != len(self.input_dims):
            raise ValueError(
                f"CrossStitchLayer expects {len(self.input_dims)} inputs, got {len(inputs)}"
            )
        stacked = torch.stack(inputs, dim=1)
        mixed = torch.einsum("b s d, t s d -> b t d", stacked, self.cross_stitch_weight)
        return [mixed[:, task_idx, :] for task_idx in range(self.num_tasks)]


class CrossStitch(BaseModel):
    """
    Cross-Stitch Networks for multi-task learning.
    """

    @property
    def model_name(self) -> str:
        return "CrossStitch"

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
        shared_mlp_params: dict | None = None,
        task_mlp_params: dict | None = None,
        tower_mlp_params: dict | None = None,
        tower_mlp_params_list: list[dict] | None = None,
        **kwargs,
    ) -> None:
        dense_features = dense_features or []
        sparse_features = sparse_features or []
        sequence_features = sequence_features or []
        shared_mlp_params = shared_mlp_params or {}
        task_mlp_params = task_mlp_params or {}
        tower_mlp_params = tower_mlp_params or {}
        tower_mlp_params_list = tower_mlp_params_list or []

        shared_mlp_params.setdefault("hidden_dims", [])
        task_mlp_params.setdefault("hidden_dims", [256, 128])
        tower_mlp_params.setdefault("hidden_dims", [64])

        default_activation = task_mlp_params.get("activation", "relu")
        default_dropout = task_mlp_params.get("dropout", 0.0)
        default_norm_type = task_mlp_params.get("norm_type", "none")

        shared_mlp_params.setdefault("activation", default_activation)
        shared_mlp_params.setdefault("dropout", default_dropout)
        shared_mlp_params.setdefault("norm_type", default_norm_type)
        task_mlp_params.setdefault("activation", default_activation)
        task_mlp_params.setdefault("dropout", default_dropout)
        task_mlp_params.setdefault("norm_type", default_norm_type)
        tower_mlp_params.setdefault("activation", default_activation)
        tower_mlp_params.setdefault("dropout", default_dropout)
        tower_mlp_params.setdefault("norm_type", default_norm_type)

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
        if self.nums_task <= 1:
            raise ValueError("CrossStitch requires at least 2 tasks.")
        if not task_mlp_params["hidden_dims"]:
            raise ValueError("task_mlp_params['hidden_dims'] must not be empty.")
        shared_hidden_dims = shared_mlp_params["hidden_dims"]

        if tower_mlp_params_list:
            if len(tower_mlp_params_list) != self.nums_task:
                raise ValueError(
                    "Number of tower mlp params "
                    f"({len(tower_mlp_params_list)}) must match number of tasks ({self.nums_task})."
                )
            tower_params = [params.copy() for params in tower_mlp_params_list]
        else:
            tower_params = [tower_mlp_params.copy() for _ in range(self.nums_task)]

        self.embedding = EmbeddingLayer(features=self.all_features)
        input_dim = self.embedding.input_dim

        if shared_hidden_dims:
            self.shared_layer = MLP(
                input_dim=input_dim,
                hidden_dims=shared_hidden_dims,
                output_dim=None,
                dropout=shared_mlp_params["dropout"],
                activation=shared_mlp_params["activation"],
                norm_type=shared_mlp_params["norm_type"],
            )
            prev_dim = shared_hidden_dims[-1]
        else:
            self.shared_layer = nn.Identity()
            prev_dim = input_dim
        self.grad_norm_shared_modules = [
            "embedding",
            "shared_layer",
            "task_layers",
            "cross_stitch_layers",
        ]

        self.task_layers = nn.ModuleList()
        self.cross_stitch_layers = nn.ModuleList()
        for hidden_dim in task_mlp_params["hidden_dims"]:
            layer_tasks = nn.ModuleList(
                [
                    MLP(
                        input_dim=prev_dim,
                        hidden_dims=[hidden_dim],
                        output_dim=None,
                        dropout=task_mlp_params["dropout"],
                        activation=task_mlp_params["activation"],
                        norm_type=task_mlp_params["norm_type"],
                    )
                    for _ in range(self.nums_task)
                ]
            )
            self.task_layers.append(layer_tasks)
            self.cross_stitch_layers.append(
                CrossStitchLayer(input_dims=[hidden_dim] * self.nums_task)
            )
            prev_dim = hidden_dim

        self.towers = nn.ModuleList()
        for params in tower_params:
            if tower_mlp_params_list:
                tower = MLP(input_dim=prev_dim, output_dim=1, **params)
            else:
                tower = MLP(
                    input_dim=prev_dim,
                    hidden_dims=params.get("hidden_dims"),
                    output_dim=1,
                    dropout=params.get("dropout", tower_mlp_params["dropout"]),
                    activation=params.get("activation", tower_mlp_params["activation"]),
                    norm_type=params.get("norm_type", tower_mlp_params["norm_type"]),
                )
            self.towers.append(tower)

        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )
        self.register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["shared_layer", "task_layers", "towers"],
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        task_reps = [self.shared_layer(input_flat) for _ in range(self.nums_task)]

        for layer_idx in range(len(self.task_layers)):
            for task_idx in range(self.nums_task):
                task_reps[task_idx] = self.task_layers[layer_idx][task_idx](
                    task_reps[task_idx]
                )
            task_reps = self.cross_stitch_layers[layer_idx](task_reps)

        task_outputs = []
        for task_idx, tower in enumerate(self.towers):
            task_outputs.append(tower(task_reps[task_idx]))

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
