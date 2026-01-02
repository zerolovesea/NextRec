"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com

Shared-Bottom is the classic hard-parameter-sharing baseline for multi-task learning.
All tasks share a common bottom network to learn general representations, and each
task has its own tower head for task-specific refinement and prediction. This
architecture is simple, parameter-efficient, and helps regularize related tasks.

Workflow:
  (1) Unified embeddings convert dense/sparse/sequence features
  (2) A shared bottom MLP learns common representations
  (3) Task-specific towers further transform the shared features
  (4) Separate prediction heads output each task’s logits/probabilities

Key Advantages:
- Strong inductive bias via hard parameter sharing, reducing overfitting
- Parameter-efficient compared to duplicating full models per task
- Easy to extend to many tasks with small incremental cost
- Serves as a stable baseline for evaluating advanced MTL architectures

Share-Bottom（硬共享底层）是多任务学习的经典基线：所有任务共享一个底层网络，
各任务拥有独立塔头进行细化与预测，简单高效且能通过共享正则化相关任务。

流程：
  (1) 统一 embedding 处理稠密、稀疏与序列特征
  (2) 共享底层 MLP 学习通用表示
  (3) 任务塔在共享表示上做任务特定变换
  (4) 各任务预测头输出对应结果

主要优点：
- 硬参数共享提供强正则，减少过拟合
- 相比单独模型更节省参数与计算
- 易于扩展到多任务，增量开销小
- 是评估更复杂 MTL 结构的稳健基线
"""

import torch
import torch.nn as nn

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.basic.layers import MLP, EmbeddingLayer
from nextrec.basic.heads import TaskHead
from nextrec.basic.model import BaseModel


class ShareBottom(BaseModel):
    @property
    def model_name(self):
        return "ShareBottom"

    @property
    def default_task(self):
        nums_task = self.nums_task if hasattr(self, "nums_task") else None
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        bottom_mlp_params: dict,
        tower_mlp_params_list: list[dict],
        target: list[str],
        task: str | list[str] | None = None,
        **kwargs,
    ):

        self.nums_task = len(target)

        super(ShareBottom, self).__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            **kwargs,
        )

        # Number of tasks
        self.nums_task = len(target)
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

        # Shared bottom network
        self.bottom = MLP(input_dim=input_dim, output_dim=None, **bottom_mlp_params)
        self.grad_norm_shared_modules = ["embedding", "bottom"]

        # Get bottom output dimension
        if (
            "hidden_dims" in bottom_mlp_params
            and len(bottom_mlp_params["hidden_dims"]) > 0
        ):
            bottom_output_dim = bottom_mlp_params["hidden_dims"][-1]
        else:
            bottom_output_dim = input_dim

        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_mlp_params in tower_mlp_params_list:
            tower = MLP(input_dim=bottom_output_dim, output_dim=1, **tower_mlp_params)
            self.towers.append(tower)
        self.prediction_layer = TaskHead(
            task_type=self.task, task_dims=[1] * self.nums_task
        )
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["bottom", "towers"]
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        # Shared bottom
        bottom_output = self.bottom(input_flat)  # [B, bottom_dim]

        # Task-specific towers
        task_outputs = []
        for tower in self.towers:
            tower_output = tower(bottom_output)  # [B, 1]
            task_outputs.append(tower_output)

        # Stack outputs: [B, nums_task]
        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
