"""
Date: create on 09/11/2025
Checkpoint: edit on 23/12/2025
Author: Yang Zhou,zyaztec@gmail.com
Reference:
[1] Caruana R. Multitask learning[J]. Machine Learning, 1997, 28: 41-75.
(https://link.springer.com/article/10.1023/A:1007379606734)

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
        nums_task = getattr(self, "nums_task", None)
        if nums_task is not None and nums_task > 0:
            return ["binary"] * nums_task
        return ["binary"]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        bottom_params: dict,
        tower_params_list: list[dict],
        target: list[str],
        task: str | list[str] | None = None,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        loss_params: dict | list[dict] | None = None,
        device: str = "cpu",
        embedding_l1_reg=1e-6,
        dense_l1_reg=1e-5,
        embedding_l2_reg=1e-5,
        dense_l2_reg=1e-4,
        **kwargs,
    ):

        optimizer_params = optimizer_params or {}

        self.nums_task = len(target)

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

        super(ShareBottom, self).__init__(
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

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"
        # Number of tasks
        self.nums_task = len(target)
        if len(tower_params_list) != self.nums_task:
            raise ValueError(
                f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.nums_task})"
            )
        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)
        # Calculate input dimension
        input_dim = self.embedding.input_dim
        # emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        # dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        # input_dim = emb_dim_total + dense_input_dim

        # Shared bottom network
        self.bottom = MLP(input_dim=input_dim, output_layer=False, **bottom_params)
        self.grad_norm_shared_modules = ["embedding", "bottom"]

        # Get bottom output dimension
        if "dims" in bottom_params and len(bottom_params["dims"]) > 0:
            bottom_output_dim = bottom_params["dims"][-1]
        else:
            bottom_output_dim = input_dim

        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_params in tower_params_list:
            tower = MLP(input_dim=bottom_output_dim, output_layer=True, **tower_params)
            self.towers.append(tower)
        self.prediction_layer = TaskHead(
            task_type=self.default_task, task_dims=[1] * self.nums_task
        )
        # Register regularization weights
        self.register_regularization_weights(
            embedding_attr="embedding", include_modules=["bottom", "towers"]
        )
        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
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
